// Package fileprocessor provides a portable file processing pipeline. It
// bundles the original fileprocessor responsibilities (type detection, content
// extraction, OCR / VL enrichment, file-store persistence) with the RAG core
// (DuckDB-backed vector store, embedding cache, semantic search, and the
// RAGProcessor ingestion path).
//
// Typical wiring:
//
//	duck, _ := fileprocessor.NewDuckDBStore("/var/lib/myapp/vectors.db", 384)
//	embedder := fileprocessor.NewEmbeddingCache(myEmbedder, nil)
//	searcher := fileprocessor.NewSearcher(duck, myChunkStore, embedder)
//	ragProc  := fileprocessor.NewRAGProcessor(myChunkStore, duck, embedder, nil)
//	proc, _  := fileprocessor.New(fileprocessor.Config{
//	    FileStore:    myChunkStore,
//	    RAGProcessor: ragProc,
//	    Chunker:      fileprocessor.NewCharChunker(1000, 200),
//	    FileBaseDir:  "/var/myapp/files",
//	})
package fileprocessor

import (
	"context"

	"github.com/cloudwego/eino/schema"
)

// DistanceMetric is the distance function used by a VectorStore for similarity
// scoring. The exact mapping is implementation-defined.
type DistanceMetric string

const (
	// DistanceCosine uses cosine distance (1 - cosine_similarity).
	DistanceCosine DistanceMetric = "cosine"
	// DistanceEuclidean uses squared L2 (Euclidean) distance.
	DistanceEuclidean DistanceMetric = "euclidean"
	// DistanceInnerProduct uses negative inner product.
	DistanceInnerProduct DistanceMetric = "inner_product"
)

// VectorMatch is a single hit from a VectorStore search.
type VectorMatch struct {
	ID         string
	FileID     string
	Similarity float64
}

// SearchParams controls a VectorStore search call.
type SearchParams struct {
	// Limit is the maximum number of matches to return.
	Limit int
	// Metric overrides the store's default metric for this call. Empty keeps default.
	Metric DistanceMetric
	// FileIDs restricts the search to chunks belonging to these files when the
	// backing store supports scoped search.
	FileIDs []string
	// TenantID, when non-empty on stores that enforce it (e.g.
	// PublicEmbeddingsStore), scopes the search via a SET LOCAL app.tenant_id
	// transaction so Postgres RLS policies filter at the source rather than via
	// in-memory allowlists. Empty keeps the legacy pool-query path (architecture
	// review R3).
	TenantID string
}

// KeywordSearcher is an optional text-search capability implemented by stores
// that can search their chunk corpus with lexical ranking such as SQLite FTS5
// BM25. VectorStore implementations that do not implement it remain vector-only.
type KeywordSearcher interface {
	KeywordSearch(ctx context.Context, query string, params SearchParams) ([]VectorMatch, error)
}

// VectorStore persists embeddings and supports similarity search.
//
// Implementations are responsible for choosing their own index (HNSW, flat, etc.)
// and distance function. The store must be safe for concurrent use.
type VectorStore interface {
	// Upsert inserts or updates a single vector.
	Upsert(ctx context.Context, id, fileID string, embedding []float32) error
	// UpsertBatch inserts or updates many vectors in one call. Implementations
	// may fall back to per-row Upsert if no native batch path exists.
	UpsertBatch(ctx context.Context, items []VectorItem) error
	// Search returns the top-K most similar vectors to the query embedding.
	Search(ctx context.Context, embedding []float32, params SearchParams) ([]VectorMatch, error)
	// DeleteByID removes a single vector.
	DeleteByID(ctx context.Context, id string) error
	// DeleteByFileID removes all vectors belonging to a file.
	DeleteByFileID(ctx context.Context, fileID string) error
	// Close releases all resources held by the store.
	Close() error
}

// VectorItem is a single upsert payload for VectorStore.UpsertBatch.
type VectorItem struct {
	ID        string
	FileID    string
	Embedding []float32
}

// Embedder turns text into a fixed-dimension float32 vector.
type Embedder interface {
	// Embed returns one embedding per input text, in the same order.
	Embed(ctx context.Context, texts []string) ([][]float32, error)
	// Dimension returns the embedding vector size. Used to validate VectorStore
	// configuration. Implementations may return 0 if unknown.
	Dimension() int
}

// ChunkStore is the durable, queryable side of the RAG pipeline. The VectorStore
// only holds embeddings; ChunkStore holds the chunk text, file linkage, and
// metadata. Concrete adapters (e.g. SQLite, Postgres) live in the host app.
type ChunkStore interface {
	// GetDocument fetches a parsed document by ID. Returns ErrNotFound if absent.
	// The returned *schema.Document uses MetaData keys DocumentMetaFileType and
	// DocumentPages for file type and page data respectively.
	GetDocument(ctx context.Context, id string) (*schema.Document, error)
	// CreateChunk inserts a new chunk row and returns its ID.
	CreateChunk(ctx context.Context, p CreateChunkParams) (string, error)
	// GetChunksByIDs returns chunks in the same order as the IDs. The chunk
	// records must include their parent FileID so callers can hydrate filenames
	// and apply per-file filters.
	GetChunksByIDs(ctx context.Context, ids []string) ([]Chunk, error)
	// GetFile fetches a file record by ID. Returns ErrNotFound if absent.
	GetFile(ctx context.Context, id string) (RAGFile, error)
	// UpdateFileChunkStats records the outcome of an ingestion run.
	UpdateFileChunkStats(ctx context.Context, p UpdateFileStatsParams) error
}

// CreateChunkParams describes a chunk to persist.
type CreateChunkParams struct {
	// ID is an optional pre-generated chunk ID. If empty, the store assigns one.
	ID string
	// DocumentID is the parent document.
	DocumentID string
	// Text is the chunk content.
	Text string
	// Index is the 0-based position of the chunk within the document.
	Index int
	// Type is a free-form chunk type (e.g. "text", "image_caption").
	Type string
	// Metadata is a JSON-encoded string of arbitrary metadata.
	Metadata string
}

// UpdateFileStatsParams reports ingestion status for a file.
type UpdateFileStatsParams struct {
	FileID          string
	ChunkCount      int64
	ChunkingStatus  string
	EmbeddingStatus string
}

// RAGFile is the minimal file view used by ChunkStore.GetFile. It carries only
// the identity + name needed by the search hydration path. Stores that need to
// expose additional fields should return a richer type from a dedicated API.
type RAGFile struct {
	ID   string
	Name string
}

// DocumentMetaFileType is the metadata key for the document's file type extension.
const DocumentMetaFileType = "fileprocessor.file_type"

// DocumentMetaPages is the metadata key for the JSON-encoded page list.
// Empty string means pages are not available; consumers must fall back to
// text chunking.
const DocumentMetaPages = "fileprocessor.pages"

// DocumentFileType returns the file type extension from a schema.Document's
// MetaData, or an empty string when absent.
func DocumentFileType(doc *schema.Document) string {
	return DocumentStringMetadata(doc, DocumentMetaFileType)
}

// DocumentPages returns the JSON-encoded page list from a schema.Document's
// MetaData, or an empty string when absent.
func DocumentPages(doc *schema.Document) string {
	return DocumentStringMetadata(doc, DocumentMetaPages)
}

// Chunk is a single stored chunk.
type Chunk struct {
	ID       string
	Text     string
	FileID   string
	Type     string
	Index    int64
	ParentID string // UUID of the parent chunk (ParentChunk type), empty if none
	// Metadata is the raw JSON string as stored.
	Metadata string
}

// Document metadata keys populated by Searcher.SemanticSearch.
//
// The document content, identity, and score use schema.Document's canonical
// fields. These keys preserve retrieval-specific provenance without requiring
// a second result type.
const (
	DocumentMetaFileID   = "fileprocessor.file_id"
	DocumentMetaFileName = "fileprocessor.file_name"
	DocumentMetaType     = "fileprocessor.type"
	DocumentMetaIndex    = "fileprocessor.index"
	DocumentMetaParentID = "fileprocessor.parent_id"
	DocumentMetaRaw      = "fileprocessor.raw_metadata"
)

// DocumentStringMetadata returns a string metadata value, or an empty string
// when the key is absent or contains a different type.
func DocumentStringMetadata(doc *schema.Document, key string) string {
	if doc == nil || doc.MetaData == nil {
		return ""
	}
	v, ok := doc.MetaData[key].(string)
	if !ok {
		return ""
	}
	return v
}

// DocumentIntMetadata returns an int metadata value, or zero when the key is
// absent or contains a different type.
func DocumentIntMetadata(doc *schema.Document, key string) int {
	if doc == nil || doc.MetaData == nil {
		return 0
	}
	v, ok := doc.MetaData[key].(int)
	if !ok {
		return 0
	}
	return v
}

// BatchSearchRequest is a single query inside a [VectorStore.BatchSearch] call.
type BatchSearchRequest struct {
	QueryID   string
	Embedding []float32
}

// BatchSearchResult groups the matches for one query.
type BatchSearchResult struct {
	QueryID string
	Matches []VectorMatch
}
