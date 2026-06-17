# fileprocessor

A reusable Go library for file upload processing pipelines: content extraction, chunking, OCR/VL enrichment, and RAG integration. The library bundles a **pgvector**-backed vector store, embedding cache, semantic search, and the ingestion pipeline — all in a single module.

The library is **database-agnostic** for the durable file/document side: clients implement the [`FileStore`](types.go) interface (SQLite, Postgres, in-memory, anything). The vector store uses [PostgreSQL + pgvector](https://github.com/pgvector/pgvector) with an HNSW index.

## Features

- **Type detection & extraction**: PDF, DOCX, XLSX, PPTX, TXT, Markdown, images, videos.
- **Pluggable chunker**: `CharChunker` (character-based, matches the built-in `RAGChunker`) or `TokenChunker` (token-aware, client provides a tokenizer). Custom chunkers are supported via the `Chunker` interface.
- **Async image enrichment**: Tesseract OCR → LLM cleanup → optional VL model description → RAG.
- **Vector store**: `PgVectorStore` with HNSW index, batch search via LATERAL joins, per-query metric override. Pure Go — **no CGO required**.
- **Embedding cache**: `EmbeddingCache` (SHA256-keyed, TTL+LRU) wraps any `Embedder`.
- **RAG ingestion**: `RAGProcessor` chunks documents, embeds each chunk, and persists chunks + vectors atomically.
- **Semantic search**: `Searcher` (embed → ANN search → chunk hydration → file-ID filter).
- **No DB deps for the document side**: The library never touches SQL directly for files/documents. All persistence flows through `FileStore`.

## Installation

```bash
go get github.com/kawai-network/fileprocessor
```

Requires Go 1.26+. **No CGO required** — the library is pure Go. The only external dependency is a running PostgreSQL (≥ 12 recommended) with the `pgvector` extension available.

### Database setup

The store auto-creates the `vector` extension (or probes for it on hosted providers like Supabase/Neon where `CREATE EXTENSION` is restricted). It also creates a dedicated schema (`fileprocessor` by default) and an HNSW index.

## Quick start

```go
package main

import (
    "context"
    "log"

    "github.com/kawai-network/fileprocessor"
)

func main() {
    // Client-provided (see adapters/ directory for a SQLite example).
    store := myapp.NewFileStore(db)
    embedder := myEmbedder{}  // implements fileprocessor.Embedder

    // pgvector vector store.
    pg, err := fileprocessor.NewPgVectorStore(ctx,
        "postgresql://user:pass@host:5432/db?sslmode=require",
        embedder.Dimension())
    if err != nil { log.Fatal(err) }
    defer pg.Close()

    // RAG processor with embedded cache.
    rag := fileprocessor.NewRAGProcessor(store, pg,
        fileprocessor.NewEmbeddingCache(embedder, nil), nil)

    // Pick a chunker: char-based (default) or token-aware.
    chunker := fileprocessor.NewCharChunker(1000, 200)
    // or: fileprocessor.NewTokenChunker(256, 50, myTokenizer)

    proc, err := fileprocessor.New(fileprocessor.Config{
        FileStore:    store,
        RAGProcessor: rag,
        Chunker:      chunker,
        FileBaseDir:  "/var/myapp/files",
    })
    if err != nil { log.Fatal(err) }

    resp, err := proc.ProcessFile(context.Background(), fileprocessor.Request{
        FilePath:  "/tmp/report.pdf",
        Filename:  "report.pdf",
        EnableRAG: true,
    })
    if err != nil { log.Fatal(err) }
    log.Printf("file_id=%s document_id=%s chunks=%d",
        resp.FileID, resp.DocumentID, len(resp.ChunkIDs))

    // Semantic search.
    searcher := fileprocessor.NewSearcher(pg, store, embedder)
    results, _ := searcher.SemanticSearch(ctx, fileprocessor.SearchParamsSearcher{
        Query: "project timeline",
        Limit: 10,
    })
}
```

## Implementing `FileStore`

The `FileStore` interface has 8 methods. Here's the contract:

| Method | Purpose |
| --- | --- |
| `CreateFile` | Persist file metadata (name, type, size, URL, hash). Return assigned ID. |
| `CreateDocument` | Persist parsed markdown content + pages. Return assigned ID. |
| `GetDocumentByFileID` | Fetch document by file ID. Return `ErrNotFound` if absent. |
| `AppendToDocument` | Append async-generated content (e.g. image description). |
| `UpdateFileChunkStats` | Record ingestion outcome (chunk count, status). |
| `CreateChunk` | Persist a single chunk. |
| `GetFile` | Fetch a file by ID. Return `ErrNotFound` if absent. |
| `DeleteFile` | Remove file + associated documents/chunks (cascade). |

Wrap your DB errors with `ErrNotFound` so the library can distinguish missing records from real errors:

```go
func (s *myFileStore) GetFile(ctx context.Context, id string) (fileprocessor.StoredFile, error) {
    row, err := s.queries.GetFile(ctx, id)
    if err != nil {
        if errors.Is(err, sql.ErrNoRows) {
            return fileprocessor.StoredFile{}, fileprocessor.ErrNotFound
        }
        return fileprocessor.StoredFile{}, err
    }
    return fileprocessor.StoredFile{ID: row.ID, Name: row.Name, /* ... */}, nil
}
```

### Ready-made adapter: `PostgresFileStore` (lobehub schema)

This module ships a `PostgresFileStore` that targets the existing lobehub public schema (files, documents, chunks, file_chunks, document_chunks). It implements both `FileStore` and `ChunkStore` so it works with `Processor` and `RAGProcessor` directly. Tenancy is stamped at construction; chunk stats live in `files.metadata->'chunk_stats'` (no migration needed for the stats columns).

```go
store, _ := fileprocessor.NewPostgresFileStore(ctx,
    "postgresql://user:pass@host:5432/db?sslmode=require",
    fileprocessor.PostgresFileStoreOwner{UserID: "u_abc", WorkspaceID: "ws_1"},
)
defer store.Close()

// ChunkStore adapter for RAGProcessor.
rag := fileprocessor.NewRAGProcessor(store.ChunkStore(), vectorStore, embedder, nil)

proc, _ := fileprocessor.New(fileprocessor.Config{
    FileStore:    store,
    RAGProcessor: rag,
    Chunker:      fileprocessor.NewCharChunker(1000, 200),
    FileBaseDir:  "/var/myapp/files",
})
```

Schema requirements:
- The `users` row referenced by `Owner.UserID` must exist (FK enforced by `files_user_id_users_id_fk`).
- For workspace-scoped rows, `Owner.WorkspaceID` must exist in `workspaces`.
- Migration `0112_fileprocessor_file_size_bigint.sql` widens `files.size` and `global_files.size` from `integer` to `bigint`, and widens `document_chunks.document_id` and `document_histories.document_id` from `varchar(30)` to `varchar(255)`.
- `files.file_hash` is **not** written by the library (FK to `global_files.hash_id`); the content hash is surfaced under `files.metadata->'file_hash'` instead.

## Chunker choice

- **`CharChunker`** — character-based recursive splitter. Equivalent to the built-in `RAGChunker`. Use when token counting is unavailable or unnecessary.
- **`TokenChunker`** — token-aware splitter. Pass any `Tokenizer func(string) int` (e.g. `tiktoken-go`, `unillm`, or a simple whitespace counter). Recommended when you need precise control over embedding context windows.
- **Custom** — implement the `Chunker` interface directly.

When `Config.Chunker` is nil, the processor delegates to `RAGProcessor.ProcessFile` which uses the built-in `RAGChunker`.

## Vector store & search

Two stores ship in this module. Pick by your embedder's output dim and where you want vectors to live.

### `PgVectorStore` (default, schema-isolated)

Writes into a dedicated `fileprocessor` schema. Any dim accepted. Use when you want full control over vector storage and isolation from the host app's tables.

```go
store, _ := fileprocessor.NewPgVectorStore(ctx,
    "postgresql://user:pass@host:5432/db?sslmode=require", 384)
store.SetEfSearch(200)                     // per-query recall/latency trade-off

// Batch search (60× faster than looping Search for large fan-outs).
results, _ := store.BatchSearch(ctx, []fileprocessor.BatchSearchRequest{
    {QueryID: "q1", Embedding: vec1},
    {QueryID: "q2", Embedding: vec2},
}, 10)

// Inspect index health.
stats, _ := store.GetIndexStats(ctx)
log.Printf("vectors=%d index_size=%s", stats.TotalVectors, stats.IndexSize)
```

### `PublicEmbeddingsStore` (lobehub, dim 1024)

Targets the existing `public.embeddings` table directly. Dim is hard-pinned to **1024** (the schema's `vector(1024)` constraint). Use it when your embedder produces 1024-dim vectors and you want to share storage with the host app's existing RAG pipeline — deletes cascade from `chunks` and `file_id` is hydrated via a join to `public.file_chunks`.

```go
es, _ := fileprocessor.NewPublicEmbeddingsStore(ctx,
    "postgresql://user:pass@host:5432/db?sslmode=require", 1024, nil)
defer es.Close()

// Upsert/Search/Delete are identical to PgVectorStore from the caller's
// perspective — they all implement VectorStore.
es.Upsert(ctx, chunkID, "", vec)             // chunkID is the natural key
matches, _ := es.Search(ctx, vec, fileprocessor.SearchParams{Limit: 10})
es.DeleteByFileID(ctx, fileID)              // cascades via file_chunks join
```

Rows written by `PublicEmbeddingsStore` are stamped with `model = 'fileprocessor'` so they coexist with the host app's existing RAG rows. The store creates an HNSW index named `public_embeddings_hnsw_idx` on first use; pre-existing HNSW indexes are left alone.

### Custom HNSW config

```go
store, _ := fileprocessor.NewPgVectorStoreWithConfig(ctx, dsn, 384, &fileprocessor.PgHNSWConfig{
    Metric:         fileprocessor.DistanceCosine, // or DistanceEuclidean / DistanceInnerProduct
    M:              16,
    EfConstruction: 64,
    EfSearch:       40,
})
```

The same `PgHNSWConfig` is accepted by `NewPublicEmbeddingsStoreWithConfig`. Note: `Metric`, `M`, and `EfConstruction` are baked into the HNSW index at build time. Changing them requires dropping and recreating the index. `EfSearch` is a runtime knob.

### Schema isolation

By default `PgVectorStore` creates a `fileprocessor` schema (separate from `public`) so it never collides with your application's tables. If you already manage schemas explicitly, use `NewPgVectorStoreWithPool` and pass your own pool + schema name.

## RAG processor

```go
rag := fileprocessor.NewRAGProcessor(chunkStore, vectorStore, embedder, nil)
rag.SetChunkSize(512, 50)                  // tune chunk/overlap
ids, _ := rag.ProcessFile(ctx, fileprocessor.RAGProcessRequest{
    FileID: "f1", DocumentID: "d1", Filename: "notes.md",
})
rag.DeleteFileVectors(ctx, "f1")           // cleanup on delete
```

## Testing

Unit tests run with the standard `go test`. Integration tests for the Postgres-backed stores are gated by the `integration` build tag and require a live Postgres with pgvector plus the lobehub tables (files/documents/chunks/file_chunks/document_chunks).

```bash
# Unit tests only
go test ./...

# Integration tests (requires Postgres + pgvector + lobehub tables)
# Two env vars; the same DSN can power both stores.
FILEPROCESSOR_TEST_PG_DSN="postgresql://..." go test -tags=integration ./...
```

Integration tests:
- `PgVectorStore` tests create an isolated, randomly-named schema (`fileprocessor_test_*`) and drop it on cleanup. They never touch the lobehub `public` schema.
- `PostgresFileStore` tests write to the lobehub `public` schema using a per-run user_id (`fp_test_user_seed`, seeded if absent). Cleanup wipes any rows stamped with that user. They never read or write app data.

## License

Apache-2.0
