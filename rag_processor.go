package fileprocessor

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"time"

	"github.com/google/uuid"
)

// RAGProcessor orchestrates the ingestion path: chunk a parsed document, embed
// each chunk, and persist chunk text + vector. Chunk text goes to the
// ChunkStore; vectors go to the VectorStore.
type RAGProcessor struct {
	chunks      ChunkStore
	store       VectorStore
	chunker     *RAGChunker
	embedder    Embedder
	chunkSize   int
	overlapSize int
}

// NewRAGProcessor wires a RAGProcessor. The embedder, store, and chunks are
// required. A nil chunker is replaced with a default NewRAGChunker().
func NewRAGProcessor(chunks ChunkStore, store VectorStore, embedder Embedder, chunker *RAGChunker) *RAGProcessor {
	if chunker == nil {
		chunker = NewRAGChunker()
	}
	return &RAGProcessor{
		chunks:      chunks,
		store:       store,
		chunker:     chunker,
		embedder:    embedder,
		chunkSize:   1000,
		overlapSize: 200,
	}
}

// SetChunkSize overrides the default 1000/200 chunk/overlap sizes.
func (r *RAGProcessor) SetChunkSize(chunkSize, overlapSize int) {
	r.chunkSize = chunkSize
	r.overlapSize = overlapSize
}

// RAGProcessRequest is the input to ProcessFile.
type RAGProcessRequest struct {
	FileID     string
	DocumentID string
	Filename   string
}

// ProcessFile chunks the document referenced by req.DocumentID, embeds each
// chunk, persists chunk text to the ChunkStore and vectors to the VectorStore,
// and updates per-file stats. The function returns the list of new chunk IDs.
func (r *RAGProcessor) ProcessFile(ctx context.Context, req RAGProcessRequest) ([]string, error) {
	if req.DocumentID == "" {
		return nil, fmt.Errorf("ragcore: documentID is required")
	}

	slog.InfoContext(ctx, "fileprocessor: RAG processing",
		"file_id", req.FileID,
		"document_id", req.DocumentID,
	)

	doc, err := r.chunks.GetDocument(ctx, req.DocumentID)
	if err != nil {
		return nil, fmt.Errorf("ragcore: get document: %w", err)
	}
	if doc.Content == "" {
		slog.WarnContext(ctx, "fileprocessor: empty document", "document_id", req.DocumentID)
		_ = r.chunks.UpdateFileChunkStats(ctx, UpdateFileStatsParams{
			FileID:          req.FileID,
			ChunkingStatus:  "empty",
			EmbeddingStatus: "empty",
		})
		return []string{}, nil
	}

	pages := r.parsePages(doc.Pages)

	fileDoc := &FileDocument{
		Content:  doc.Content,
		FileType: doc.FileType,
		Filename: req.Filename,
		Pages:    pages,
	}

	chunks := r.chunker.ChunkDocument(fileDoc, ChunkingConfig{
		Enabled:     true,
		ChunkSize:   r.chunkSize,
		OverlapSize: r.overlapSize,
	})
	if len(chunks) == 0 {
		slog.WarnContext(ctx, "fileprocessor: no chunks produced", "document_id", req.DocumentID)
		_ = r.chunks.UpdateFileChunkStats(ctx, UpdateFileStatsParams{
			FileID:          req.FileID,
			ChunkingStatus:  "empty",
			EmbeddingStatus: "empty",
		})
		return []string{}, nil
	}

	chunkIDs := make([]string, 0, len(chunks))
	for i, ch := range chunks {
		chunkID := req.DocumentID + "-" + uuid.New().String()

		// Materialize metadata as JSON, matching the prior schema.
		metaJSON := encodeChunkMetadata(req.Filename, i, ch.Metadata)

		chunkID, err := r.chunks.CreateChunk(ctx, CreateChunkParams{
			ID:         chunkID,
			DocumentID: req.DocumentID,
			Text:       ch.Content,
			Index:      i,
			Type:       "text",
			Metadata:   metaJSON,
		})
		if err != nil {
			slog.ErrorContext(ctx, "fileprocessor: save chunk", "error", err, "chunk_index", i)
			continue
		}

		embeddings, err := r.embedder.Embed(ctx, []string{ch.Content})
		if err != nil {
			slog.ErrorContext(ctx, "fileprocessor: embed chunk", "error", err, "chunk_index", i)
			continue
		}
		if len(embeddings) == 0 || len(embeddings[0]) == 0 {
			slog.ErrorContext(ctx, "fileprocessor: empty embedding", "chunk_index", i)
			continue
		}

		if r.store != nil {
			if err := r.store.Upsert(ctx, chunkID, req.FileID, embeddings[0]); err != nil {
				slog.ErrorContext(ctx, "fileprocessor: upsert vector", "error", err, "chunk_id", chunkID)
				// ChunkStore is the source of truth; continue.
			}
		}

		chunkIDs = append(chunkIDs, chunkID)
	}

	chunkCount := int64(len(chunkIDs))
	chunkingStatus := "success"
	embeddingStatus := "success"
	if chunkCount == 0 {
		chunkingStatus = "empty"
		embeddingStatus = "empty"
	}

	if err := r.chunks.UpdateFileChunkStats(ctx, UpdateFileStatsParams{
		FileID:          req.FileID,
		ChunkCount:      chunkCount,
		ChunkingStatus:  chunkingStatus,
		EmbeddingStatus: embeddingStatus,
	}); err != nil {
		slog.ErrorContext(ctx, "fileprocessor: update file stats", "error", err, "file_id", req.FileID)
	}

	return chunkIDs, nil
}

// DeleteFileVectors removes all vectors for a file from the VectorStore.
func (r *RAGProcessor) DeleteFileVectors(ctx context.Context, fileID string) error {
	if r.store == nil {
		return nil
	}
	if err := r.store.DeleteByFileID(ctx, fileID); err != nil {
		return fmt.Errorf("ragcore: delete file vectors: %w", err)
	}
	return nil
}

// parsePages decodes a JSON-encoded []DocumentPage payload. An empty or
// invalid payload yields nil and the chunker falls back to text-based
// chunking.
func (r *RAGProcessor) parsePages(raw string) []DocumentPage {
	if raw == "" {
		return nil
	}
	var pages []DocumentPage
	if err := json.Unmarshal([]byte(raw), &pages); err != nil {
		slog.Warn("fileprocessor: failed to unmarshal pages", "error", err)
		return nil
	}
	return pages
}

// encodeChunkMetadata builds a JSON string mirroring the original schema:
// {"filename": "...", "chunk_index": N, "type": "..."}.
func encodeChunkMetadata(filename string, index int, chunkMeta map[string]interface{}) string {
	chunkType, _ := chunkMeta["type"].(string)
	payload := map[string]interface{}{
		"filename":    filename,
		"chunk_index": index,
		"type":        chunkType,
	}
	for k, v := range chunkMeta {
		if _, taken := payload[k]; taken {
			continue
		}
		payload[k] = v
	}
	b, err := json.Marshal(payload)
	if err != nil {
		return fmt.Sprintf(`{"filename":"%s","chunk_index":%d,"type":"%s"}`, filename, index, chunkType)
	}
	return string(b)
}

// unused time import shim — kept to make goimports happy if all code above
// ever changes shape. Safe to remove.
var _ = time.Time{}
