package fileprocessor

import (
	"context"
	"time"
)

// FileRecord is the immutable metadata about an uploaded file. The library
// passes it to [FileStore.CreateFile]; the store is responsible for assigning
// an ID (or accepting FileRecord.ID if non-empty) and persisting it.
type FileRecord struct {
	// ID is an optional pre-generated file ID. If empty the store assigns one.
	ID string
	// Name is the original filename, e.g. "report.pdf".
	Name string
	// FileType is the lowercase extension without the dot, e.g. "pdf".
	FileType string
	// Size is the file size in bytes.
	Size int64
	// URL is the location of the file on disk, relative to the app's file
	// base directory or absolute. The library does not interpret this value.
	URL string
	// Source is an optional source identifier (e.g. absolute upload path).
	Source string
	// Hash is an optional content hash (e.g. SHA-256) used for deduplication.
	// Empty means "not computed".
	Hash string
	// Metadata is arbitrary client-defined metadata, serialized as JSON by
	// the store implementation if needed. May be nil.
	Metadata map[string]any
}

// DocumentRecord is the parsed textual representation of a file, persisted
// immediately after extraction so the UI can show content before async
// processing (OCR/VL/RAG) finishes.
type DocumentRecord struct {
	// ID is an optional pre-generated document ID. If empty the store assigns one.
	ID string
	// FileID links the document to its parent [FileRecord].
	FileID string
	// Title is typically the filename.
	Title string
	// Content is the parsed markdown content of the file.
	Content string
	// FileType is the lowercase extension (mirrors FileRecord.FileType).
	FileType string
	// Filename mirrors FileRecord.Name.
	Filename string
	// TotalCharCount is the total number of characters in Content.
	TotalCharCount int
	// TotalLineCount is the total number of lines in Content.
	TotalLineCount int
	// PagesJSON is the JSON-encoded list of extracted pages. May be empty.
	PagesJSON string
	// Source is the source identifier (e.g. the absolute file path or a URL).
	Source string
	// SourceType is a free-form label (e.g. "file", "url").
	SourceType string
	// Metadata is arbitrary client-defined metadata. May be nil.
	Metadata map[string]any
}

// ChunkRecord is a single chunk produced by a [Chunker] and persisted via
// [FileStore.CreateChunk].
type ChunkRecord struct {
	// ID is the chunk ID. If empty the store assigns one.
	ID string
	// FileID is the parent file ID (for fast deletion / filtering).
	FileID string
	// DocumentID is the parent document ID.
	DocumentID string
	// Text is the chunk content.
	Text string
	// Index is the 0-based position of the chunk within the document.
	Index int
	// Type is a free-form label (e.g. "text", "image_caption").
	Type string
	// Metadata is arbitrary metadata, serialized as JSON by the store if
	// needed. May be nil.
	Metadata map[string]any
}

// ChunkStats is the per-file ingestion outcome reported to the store.
type ChunkStats struct {
	ChunkCount      int64
	ChunkingStatus  string // e.g. "success", "empty", "skipped"
	EmbeddingStatus string // e.g. "success", "empty", "skipped"
}

// StoredDocument is the minimal view of a document fetched back from the
// store, used when appending async-generated content (e.g. image descriptions).
type StoredDocument struct {
	ID      string
	FileID  string
	Title   string
	Content string
}

// StoredFile is the minimal view of a file fetched back from the store.
type StoredFile struct {
	ID       string
	Name     string
	URL      string
	FileType string
	Hash     string
}

// FileStore abstracts all durable storage operations. The library never
// touches SQL or any specific database directly; the client implements this
// interface against its own schema.
//
// All methods must be safe for concurrent use. Implementations are encouraged
// to use transactions where appropriate (e.g. [FileStore.DeleteFile] should
// cascade to documents and chunks atomically).
type FileStore interface {
	// CreateFile persists file metadata and returns the assigned file ID.
	// If rec.ID is non-empty, the implementation should use it.
	CreateFile(ctx context.Context, rec FileRecord) (string, error)

	// CreateDocument persists the parsed document content and returns the
	// assigned document ID. If rec.ID is non-empty, the implementation
	// should use it.
	CreateDocument(ctx context.Context, rec DocumentRecord) (string, error)

	// GetDocumentByFileID fetches the document linked to a file. Must return
	// an error wrapping [ErrNotFound] when no document exists.
	GetDocumentByFileID(ctx context.Context, fileID string) (StoredDocument, error)

	// AppendToDocument appends content to the document identified by fileID.
	// Used for async operations such as image-description generation.
	AppendToDocument(ctx context.Context, fileID, additionalContent string) error

	// UpdateFileChunkStats records ingestion outcome for a file.
	UpdateFileChunkStats(ctx context.Context, fileID string, stats ChunkStats) error

	// CreateChunk persists a single chunk and returns its ID. The store
	// must accept ChunkRecord.ID if non-empty.
	CreateChunk(ctx context.Context, rec ChunkRecord) (string, error)

	// GetFile fetches a file by ID. Must return an error wrapping
	// [ErrNotFound] when absent.
	GetFile(ctx context.Context, id string) (StoredFile, error)

	// DeleteFile removes the file and all associated documents and chunks.
	// Implementations should cascade the deletion atomically.
	DeleteFile(ctx context.Context, fileID string) error
}

// VLProvider generates a natural-language description for an image. It is
// optional; when nil, image files are processed only via OCR (if tesseract is
// available) and produce no VL caption.
type VLProvider interface {
	// ProcessImage generates a description for the image at imagePath using
	// the given prompt, capped at maxTokens.
	ProcessImage(ctx context.Context, imagePath, prompt string, maxTokens int32) (string, error)
}

// LanguageModel is a minimal text-generation interface used for OCR /
// transcript cleanup. The client adapts its own LLM client to this interface.
type LanguageModel interface {
	// Generate produces text from a system + user prompt pair.
	Generate(ctx context.Context, systemPrompt, userPrompt string) (string, error)
}

// Request describes a file to process.
type Request struct {
	// FilePath is the absolute path to the file on disk.
	FilePath string
	// Filename is the original name (e.g. from the upload). If empty, the
	// library uses filepath.Base(FilePath).
	Filename string
	// Source is an optional source label stored on the file record.
	Source string
	// EnableRAG turns on chunking + embedding. When false, the library only
	// extracts content and persists metadata + document.
	EnableRAG bool
	// IsShared hints the store that the file should also be registered in a
	// shared/deduplicated table. The library itself does not implement
	// dedup; it only forwards Hash + IsShared to [FileStore.CreateFile],
	// which may use them as it sees fit.
	IsShared bool
}

// Response is the result of [Processor.ProcessFile].
type Response struct {
	FileID       string   `json:"fileId"`
	DocumentID   string   `json:"documentId"`
	ChunkIDs     []string `json:"chunkIds,omitempty"`
	Processing   bool     `json:"processing,omitempty"` // true when async work is in-flight
	// DetectedFileType is the lowercase extension detected from the filename.
	DetectedFileType string `json:"detectedFileType"`
}

// ChunkOptions controls chunking behaviour. Both [CharChunker] and
// [TokenChunker] accept it.
type ChunkOptions struct {
	ChunkSize   int // maximum chunk size in characters (char-based) or tokens (token-based)
	OverlapSize int // overlap between consecutive chunks in the same unit as ChunkSize
}

// FileDocument is the in-memory representation of a parsed file, before it is
// persisted or chunked. It mirrors ragcore.FileDocument for compatibility.
type FileDocument struct {
	Content        string
	CreatedTime    time.Time
	FileType       string
	Filename       string
	Pages          []DocumentPage
	Source         string
	TotalCharCount int
	TotalLineCount int
	Metadata       FileMetadata
}

// FileMetadata is optional metadata about a loaded file.
type FileMetadata struct {
	Source       string    `json:"source,omitempty"`
	Filename     string    `json:"filename,omitempty"`
	FileType     string    `json:"fileType,omitempty"`
	CreatedTime  time.Time `json:"createdTime,omitempty"`
	ModifiedTime time.Time `json:"modifiedTime,omitempty"`
	Error        string    `json:"error,omitempty"`
}

// DocumentPage is a single page or section extracted from a file.
type DocumentPage struct {
	CharCount   int            `json:"charCount"`
	LineCount   int            `json:"lineCount"`
	Metadata    map[string]any `json:"metadata"`
	PageContent string         `json:"pageContent"`
}
