# fileprocessor

A reusable Go library for file upload processing pipelines: content extraction, chunking, OCR/VL enrichment, and RAG integration.

The library is **database-agnostic**. Clients implement the [`FileStore`](types.go) interface (SQLite, Postgres, in-memory, anything) and pass a `*ragcore.RAGProcessor` for chunking + embedding + vector storage.

## Features

- **Type detection & extraction**: PDF, DOCX, XLSX, PPTX, TXT, Markdown, images, videos.
- **Pluggable chunker**: `CharChunker` (default, matches ragcore) or `TokenChunker` (token-aware, client provides a tokenizer). Custom chunkers are supported via the `Chunker` interface.
- **Async image enrichment**: Tesseract OCR → LLM cleanup → optional VL model description → RAG.
- **RAG integration**: Accepts `*ragcore.RAGProcessor` directly for chunking + embedding + vector storage.
- **No DB deps**: The library never touches SQL directly. All persistence flows through `FileStore`.

## Installation

```bash
go get github.com/kawai-network/fileprocessor
```

## Quick start

```go
package main

import (
    "context"
    "log"

    "github.com/kawai-network/fileprocessor"
    "github.com/kawai-network/ragcore"
)

func main() {
    // Client-provided (see adapters/ directory for a SQLite example).
    store := myapp.NewFileStore(db)

    // Client wires ragcore.
    ragProc := ragcore.NewRAGProcessor(store, duckdbStore, embedder, nil)

    // Pick a chunker: char-based (default) or token-aware.
    chunker := fileprocessor.NewCharChunker(1000, 200)
    // or: fileprocessor.NewTokenChunker(256, 50, myTokenizer)

    proc, err := fileprocessor.New(fileprocessor.Config{
        FileStore:    store,
        RAGProcessor: ragProc,
        Chunker:      chunker,
        FileBaseDir:  "/var/myapp/files",
        // Optional providers:
        // VLProvider:    myVLModel,
        // LanguageModel: myLLM,
    })
    if err != nil {
        log.Fatal(err)
    }

    resp, err := proc.ProcessFile(context.Background(), fileprocessor.Request{
        FilePath:  "/tmp/report.pdf",
        Filename:  "report.pdf",
        EnableRAG: true,
    })
    if err != nil {
        log.Fatal(err)
    }
    log.Printf("file_id=%s document_id=%s chunks=%d",
        resp.FileID, resp.DocumentID, len(resp.ChunkIDs))
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

- **`CharChunker`** — character-based recursive splitter. Equivalent to ragcore's default. Use when token counting is unavailable or unnecessary.
- **`TokenChunker`** — token-aware splitter. Pass any `Tokenizer func(string) int` (e.g. `tiktoken-go`, `unillm`, or a simple whitespace counter). Recommended when you need precise control over embedding context windows.
- **Custom** — implement the `Chunker` interface directly.

When `Config.Chunker` is nil, the processor delegates to `ragcore.RAGProcessor.ProcessFile` which uses ragcore's built-in char-based chunker.

## License

Apache-2.0
