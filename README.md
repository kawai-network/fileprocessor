# fileprocessor

A reusable Go library for file upload processing pipelines: content extraction, chunking, OCR/VL enrichment, and RAG integration. The library bundles a DuckDB-backed vector store, embedding cache, semantic search, and the ingestion pipeline — all in a single module.

The library is **database-agnostic** for the durable file/document side: clients implement the [`FileStore`](types.go) interface (SQLite, Postgres, in-memory, anything). The vector store uses [DuckDB](https://duckdb.org/) with the `vss` extension and an HNSW index.

## Features

- **Type detection & extraction**: PDF, DOCX, XLSX, PPTX, TXT, Markdown, images, videos.
- **Pluggable chunker**: `CharChunker` (character-based, matches the built-in `RAGChunker`) or `TokenChunker` (token-aware, client provides a tokenizer). Custom chunkers are supported via the `Chunker` interface.
- **Async image enrichment**: Tesseract OCR → LLM cleanup → optional VL model description → RAG.
- **Vector store**: `DuckDBStore` with HNSW index, batch search via LATERAL joins, per-query metric override.
- **Embedding cache**: `EmbeddingCache` (SHA256-keyed, TTL+LRU) wraps any `Embedder`.
- **RAG ingestion**: `RAGProcessor` chunks documents, embeds each chunk, and persists chunks + vectors atomically.
- **Semantic search**: `Searcher` (embed → ANN search → chunk hydration → file-ID filter).
- **No DB deps**: The library never touches SQL directly for the file/document side. All persistence flows through `FileStore`.

## Installation

```bash
go get github.com/kawai-network/fileprocessor
```

Requires Go 1.26+ and CGO (DuckDB engine).

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

    // DuckDB vector store.
    duck, err := fileprocessor.NewDuckDBStore("/var/myapp/vectors.db", embedder.Dimension())
    if err != nil { log.Fatal(err) }
    defer duck.Close()

    // RAG processor with embedded cache.
    rag := fileprocessor.NewRAGProcessor(store, duck,
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
    searcher := fileprocessor.NewSearcher(duck, store, embedder)
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

## Chunker choice

- **`CharChunker`** — character-based recursive splitter. Equivalent to the built-in `RAGChunker`. Use when token counting is unavailable or unnecessary.
- **`TokenChunker`** — token-aware splitter. Pass any `Tokenizer func(string) int` (e.g. `tiktoken-go`, `unillm`, or a simple whitespace counter). Recommended when you need precise control over embedding context windows.
- **Custom** — implement the `Chunker` interface directly.

When `Config.Chunker` is nil, the processor delegates to `RAGProcessor.ProcessFile` which uses the built-in `RAGChunker`.

## Vector store & search

```go
// DuckDB-backed HNSW vector store.
store, _ := fileprocessor.NewDuckDBStore("/path/vectors.db", 384)
store.SetEfSearch(200)                     // per-query recall/latency trade-off
store.CompactIndex("vec_idx")              // reclaim tombstones

// Batch search (66× faster than loop for 1000 queries).
results, _ := store.BatchSearch(ctx, []fileprocessor.BatchSearchRequest{
    {QueryID: "q1", Embedding: vec1},
    {QueryID: "q2", Embedding: vec2},
}, 10)
```

## RAG processor

```go
rag := fileprocessor.NewRAGProcessor(chunkStore, vectorStore, embedder, nil)
rag.SetChunkSize(512, 50)                  // tune chunk/overlap
ids, _ := rag.ProcessFile(ctx, fileprocessor.RAGProcessRequest{
    FileID: "f1", DocumentID: "d1", Filename: "notes.md",
})
rag.DeleteFileVectors(ctx, "f1")           // cleanup on delete
```

## License

Apache-2.0
