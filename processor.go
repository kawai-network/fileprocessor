package fileprocessor

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"time"
)

// Config wires a [Processor].
type Config struct {
	// FileStore persists file/document/chunk records. Required.
	FileStore FileStore
	// RAGProcessor performs chunking + embedding + vector storage.
	// When nil, ProcessFile skips RAG even if Request.EnableRAG is true.
	RAGProcessor *RAGProcessor
	// Chunker controls how document content is split. When nil, the
	// processor uses the built-in chunker via RAGProcessor.ProcessFile.
	// Provide a custom chunker (e.g. [TokenChunker]) to override that.
	Chunker Chunker
	// VLProvider generates image descriptions. Optional.
	VLProvider VLProvider
	// LanguageModel cleans up OCR / transcript text. Optional.
	LanguageModel LanguageModel
	// FileBaseDir is the directory where uploaded file copies live. It is
	// only used by [SafeDelete] during file deletion. Optional.
	FileBaseDir string
	// ImageVLMaxTokens caps the response length when calling
	// VLProvider.ProcessImage. Defaults to 2048.
	ImageVLMaxTokens int32
	// OCRCleanupTimeout bounds the time spent in LLM OCR cleanup.
	// Defaults to 30 seconds.
	OCRCleanupTimeout time.Duration
}

// Processor orchestrates the file processing pipeline: load → save metadata →
// save document → (async) image/video enrichment → (sync) RAG.
type Processor struct {
	store             FileStore
	rag               *RAGProcessor
	chunker           Chunker
	vl                VLProvider
	lm                LanguageModel
	fileBaseDir       string
	imageVLMaxTokens  int32
	ocrCleanupTimeout time.Duration
}

// New returns a [Processor] wired from cfg.
func New(cfg Config) (*Processor, error) {
	if cfg.FileStore == nil {
		return nil, errors.New("fileprocessor: Config.FileStore is required")
	}
	p := &Processor{
		store:             cfg.FileStore,
		rag:               cfg.RAGProcessor,
		chunker:           cfg.Chunker,
		vl:                cfg.VLProvider,
		lm:                cfg.LanguageModel,
		fileBaseDir:       cfg.FileBaseDir,
		imageVLMaxTokens:  cfg.ImageVLMaxTokens,
		ocrCleanupTimeout: cfg.OCRCleanupTimeout,
	}
	if p.imageVLMaxTokens <= 0 {
		p.imageVLMaxTokens = 2048
	}
	if p.ocrCleanupTimeout <= 0 {
		p.ocrCleanupTimeout = 30 * time.Second
	}
	return p, nil
}

// SetVLProvider overrides the VL provider after construction.
func (p *Processor) SetVLProvider(provider VLProvider) { p.vl = provider }

// SetLanguageModel overrides the language model after construction.
func (p *Processor) SetLanguageModel(lm LanguageModel) { p.lm = lm }

// ProcessFile is the main entry point. It always:
//  1. Loads file content via [FileLoader].
//  2. Persists file metadata via [FileStore.CreateFile].
//  3. Persists the document via [FileStore.CreateDocument].
//
// Then, depending on the detected type:
//   - Images: launches async OCR / VL description, then RAG.
//   - Videos: launches async processing (currently a no-op stub matching the
//     veridium source).
//   - Other text-like files with EnableRAG: runs RAG synchronously.
//
// The Response.FileID and Response.DocumentID are always set; Response.ChunkIDs
// is only populated for synchronous RAG.
func (p *Processor) ProcessFile(ctx context.Context, req Request) (*Response, error) {
	if req.FilePath == "" {
		return nil, errors.New("fileprocessor: Request.FilePath is required")
	}
	loader := NewFileLoader()
	filename := req.Filename
	if filename == "" {
		filename = baseName(req.FilePath)
	}

	fileDoc, err := loader.LoadFile(req.FilePath, nil)
	if err != nil {
		return nil, fmt.Errorf("fileprocessor: load file: %w", err)
	}

	fileID, err := p.saveFileMetadata(ctx, req, fileDoc.FileType, filename)
	if err != nil {
		return nil, fmt.Errorf("fileprocessor: save file metadata: %w", err)
	}

	documentID, err := p.saveDocument(ctx, fileDoc, fileID, filename, req.Source)
	if err != nil {
		return nil, fmt.Errorf("fileprocessor: save document: %w", err)
	}

	resp := &Response{
		FileID:           fileID,
		DocumentID:       documentID,
		DetectedFileType: fileDoc.FileType,
	}

	switch {
	case loader.IsImageFile(fileDoc.FileType):
		slog.InfoContext(ctx, "fileprocessor: starting async image processing",
			"file_id", fileID, "document_id", documentID, "filename", filename)
		go p.processImageDescriptionAsync(req.FilePath, filename, documentID, fileID, req.EnableRAG)
		resp.Processing = true

	case loader.IsVideoFile(fileDoc.FileType):
		slog.InfoContext(ctx, "fileprocessor: starting async video processing",
			"file_id", fileID, "document_id", documentID, "filename", filename)
		go p.processVideoDescriptionAsync(req.FilePath, filename, documentID, fileID, req.EnableRAG)
		resp.Processing = true

	case req.EnableRAG && loader.CanChunkForRAG(fileDoc.FileType):
		ids, err := p.runRAG(ctx, fileID, documentID, filename, fileDoc.Content)
		if err != nil {
			slog.ErrorContext(ctx, "fileprocessor: RAG failed", "error", err, "file_id", fileID)
		}
		resp.ChunkIDs = ids

	case req.EnableRAG:
		slog.InfoContext(ctx, "fileprocessor: skipping RAG for unsupported type",
			"file_type", fileDoc.FileType, "file_id", fileID)
	}

	return resp, nil
}

// DeleteFile removes a file and all associated data from the store and the
// vector index. If FileBaseDir is configured, the on-disk file copy is also
// removed (when it lives under FileBaseDir).
func (p *Processor) DeleteFile(ctx context.Context, fileID string) error {
	stored, err := p.store.GetFile(ctx, fileID)
	if err != nil {
		if errors.Is(err, ErrNotFound) {
			return nil
		}
		return fmt.Errorf("fileprocessor: get file: %w", err)
	}

	if p.rag != nil {
		if err := p.rag.DeleteFileVectors(ctx, fileID); err != nil {
			slog.WarnContext(ctx, "fileprocessor: delete vectors", "error", err)
		}
	}

	if err := p.store.DeleteFile(ctx, fileID); err != nil {
		return fmt.Errorf("fileprocessor: delete file: %w", err)
	}

	if p.fileBaseDir != "" && stored.URL != "" {
		if err := SafeDelete(stored.URL, p.fileBaseDir); err != nil {
			slog.WarnContext(ctx, "fileprocessor: safe delete on-disk", "error", err)
		}
	}
	return nil
}

// --- RAG -------------------------------------------------------------------

// runRAG performs chunking + embedding for a file. When p.chunker is nil it
// delegates to RAGProcessor.ProcessFile. When a custom chunker is
// configured it chunks locally, persists chunks via FileStore, and asks the
// RAGProcessor to process them.
func (p *Processor) runRAG(ctx context.Context, fileID, documentID, filename, content string) ([]string, error) {
	if p.rag == nil {
		return nil, errors.New("fileprocessor: RAGProcessor not configured")
	}

	// No custom chunker: let RAGProcessor do everything.
	if p.chunker == nil {
		return p.rag.ProcessFile(ctx, RAGProcessRequest{
			FileID:     fileID,
			DocumentID: documentID,
			Filename:   filename,
		})
	}

	// Custom chunker: chunk locally, persist each chunk, and ask the
	// RAGProcessor to embed + upsert by reading the persisted document.
	// ProcessFile re-chunks internally, so for true token-aware chunking we
	// persist our chunks and then re-run RAGProcessor.ProcessFile which will
	// see the updated document content.
	pieces := p.chunker.Chunk(content)
	if len(pieces) == 0 {
		_ = p.store.UpdateFileChunkStats(ctx, fileID, ChunkStats{
			ChunkingStatus:  "empty",
			EmbeddingStatus: "empty",
		})
		return nil, nil
	}

	var ids []string
	for i, piece := range pieces {
		meta := map[string]any{
			"filename":    filename,
			"chunk_index": i,
			"type":        "text",
		}
		id, err := p.store.CreateChunk(ctx, ChunkRecord{
			DocumentID: documentID,
			FileID:     fileID,
			Text:       piece,
			Index:      i,
			Type:       "text",
			Metadata:   meta,
		})
		if err != nil {
			slog.ErrorContext(ctx, "fileprocessor: save chunk", "error", err, "index", i)
			continue
		}
		ids = append(ids, id)
	}

	_ = p.store.UpdateFileChunkStats(ctx, fileID, ChunkStats{
		ChunkCount:      int64(len(ids)),
		ChunkingStatus:  "success",
		EmbeddingStatus: "custom-chunker-pending",
	})

	// Embedding pass via RAGProcessor (reads document content, re-chunks with
	// its own strategy, upserts vectors). This keeps embeddings consistent
	// with the host's vector store configuration.
	_, _ = p.rag.ProcessFile(ctx, RAGProcessRequest{
		FileID:     fileID,
		DocumentID: documentID,
		Filename:   filename,
	})

	return ids, nil
}

// --- async image processing ------------------------------------------------

func (p *Processor) processImageDescriptionAsync(filePath, filename, documentID, fileID string, enableRAG bool) {
	ctx := context.Background()

	var finalContent, contentType string

	ocrText, err := ExtractTextWithTesseract(filePath)
	if err != nil {
		slog.WarnContext(ctx, "fileprocessor: tesseract failed", "error", err, "filename", filename)
	}

	cleaned := strings.TrimSpace(ocrText)
	if len(cleaned) > 20 {
		slog.InfoContext(ctx, "fileprocessor: OCR extracted sufficient text",
			"length", len(cleaned), "filename", filename)

		if p.lm != nil {
			cctx, cancel := context.WithTimeout(ctx, p.ocrCleanupTimeout)
			defer cancel()
			out, err := CleanupOCRText(cctx, p.lm, cleaned, filename)
			if err != nil {
				slog.WarnContext(ctx, "fileprocessor: LLM cleanup failed", "error", err)
				finalContent = cleaned
				contentType = "OCR Text (Tesseract - raw)"
			} else {
				finalContent = out
				contentType = "OCR Text (Tesseract + LLM cleanup)"
			}
		} else {
			finalContent = cleaned
			contentType = "OCR Text (Tesseract)"
		}
	} else {
		if p.vl != nil {
			prompt := "Describe this image in detail. Include all visible text, objects, people, colors, and layout."
			description, err := p.vl.ProcessImage(ctx, filePath, prompt, p.imageVLMaxTokens)
			if err != nil {
				slog.ErrorContext(ctx, "fileprocessor: VL processing failed", "error", err, "filename", filename)
				if len(cleaned) > 0 {
					finalContent = cleaned
					contentType = "OCR Text (Tesseract - VL fallback failed)"
				} else {
					return
				}
			} else {
				finalContent = description
				contentType = "Image Description (VL Model)"
				if len(cleaned) > 0 {
					finalContent = fmt.Sprintf("%s\n\n**Extracted Text (OCR):**\n%s", description, cleaned)
				}
			}
		} else if len(cleaned) > 0 {
			finalContent = cleaned
			contentType = "OCR Text (Tesseract - no VL available)"
		} else {
			slog.ErrorContext(ctx, "fileprocessor: no VL model and no OCR text", "filename", filename)
			return
		}
	}

	if finalContent == "" {
		return
	}

	markdown := fmt.Sprintf("\n\n### %s\n\n%s", contentType, finalContent)
	if err := p.store.AppendToDocument(ctx, fileID, markdown); err != nil {
		slog.ErrorContext(ctx, "fileprocessor: append to document", "error", err, "document_id", documentID)
		return
	}

	if enableRAG && p.rag != nil {
		_, err := p.rag.ProcessFile(ctx, RAGProcessRequest{
			FileID:     fileID,
			DocumentID: documentID,
			Filename:   filename,
		})
		if err != nil {
			slog.ErrorContext(ctx, "fileprocessor: RAG after image processing", "error", err, "file_id", fileID)
		}
	}
}

// --- async video processing (stub) -----------------------------------------

func (p *Processor) processVideoDescriptionAsync(filePath, filename, documentID, fileID string, enableRAG bool) {
	ctx := context.Background()
	slog.WarnContext(ctx, "fileprocessor: video processing is not implemented",
		"file_id", fileID, "filename", filename)
}

// --- store helpers ---------------------------------------------------------

func (p *Processor) saveFileMetadata(ctx context.Context, req Request, fileType, filename string) (string, error) {
	hash := ""
	if req.IsShared {
		if h, err := CalculateFileHash(req.FilePath); err == nil {
			hash = h
		}
	}
	info, err := GetFileInfo(req.FilePath)
	if err != nil {
		return "", err
	}
	return p.store.CreateFile(ctx, FileRecord{
		Name:     filename,
		FileType: fileType,
		Size:     info.Size,
		URL:      req.FilePath,
		Source:   req.Source,
		Hash:     hash,
	})
}

func (p *Processor) saveDocument(ctx context.Context, fileDoc *FileDocument, fileID, filename, source string) (string, error) {
	metadata := map[string]any{
		"source":       fileDoc.Metadata.Source,
		"filename":     fileDoc.Metadata.Filename,
		"fileType":     fileDoc.Metadata.FileType,
		"createdTime":  fileDoc.Metadata.CreatedTime,
		"modifiedTime": fileDoc.Metadata.ModifiedTime,
	}
	if fileDoc.Metadata.Error != "" {
		metadata["error"] = fileDoc.Metadata.Error
	}
	metaJSON, _ := json.Marshal(metadata)
	pagesJSON, _ := json.Marshal(fileDoc.Pages)

	return p.store.CreateDocument(ctx, DocumentRecord{
		FileID:          fileID,
		Title:           filename,
		Content:         fileDoc.Content,
		FileType:        fileDoc.FileType,
		Filename:        filename,
		TotalCharCount:  fileDoc.TotalCharCount,
		TotalLineCount:  fileDoc.TotalLineCount,
		PagesJSON:       string(pagesJSON),
		Metadata:        map[string]any{"raw": string(metaJSON)},
	})
}
