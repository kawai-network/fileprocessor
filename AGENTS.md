# AGENTS.md

Guidance for AI agents working in the `fileprocessor` repository.

## What this is

A reusable Go library that bundles two responsibilities in one module:

1. **File processing pipeline** — type detection, content extraction (PDF/DOCX/XLSX/PPTX/TXT/Markdown/Image/Video), chunking, OCR/VL enrichment, and durable file/document/chunk persistence.
2. **RAG core** — pgvector-backed vector store (HNSW), embedding cache (SHA256 + LRU + TTL), semantic search, and the ingestion path.

The library is **database-agnostic on the file/document side**; clients implement the `FileStore` interface. The vector store is opinionated — it ships as a `PgVectorStore` backed by PostgreSQL + the `pgvector` extension. The library is **pure Go** (no CGO).

Module: `github.com/kawai-network/fileprocessor`. Go 1.26+.

## Commands

```bash
# Build (pure Go; no CGO required)
CGO_ENABLED=0 go build ./...

# Unit tests
go test ./...

# Integration tests against a real Postgres with pgvector
FILEPROCESSOR_TEST_PG_DSN="postgresql://user:pass@host:5432/db?sslmode=require" \
  go test -tags=integration ./...

# Vet
go vet ./...

# Tidy
go mod tidy
```

Notes:
- The repo currently has **no `Makefile`**, no `golangci-lint` config, and no CI workflow under `.github/`. Just `go test`, `go vet`, `go build`.
- Integration tests create an isolated, randomly-named Postgres schema per test and drop it on cleanup. They **never touch `public`**.
- The integration env var is `FILEPROCESSOR_TEST_PG_DSN`. `t.Skip` is used when it is unset, so `go test ./...` stays green without a DB.

## File layout

| File | Purpose |
| --- | --- |
| `doc.go` / `types.go` | Package doc and the public types (records, `FileStore`, `Embedder`, `VectorStore`, `ChunkStore`, `Chunker`, etc.). |
| `errors.go` | `ErrNotFound`, `ErrUnsupportedFileType` (sentinel errors; use `errors.Is`). |
| `loader.go` | `FileLoader` — extension detection + dispatch. Lists `textExtensions`, `imageExtensions`, `videoExtensions`. |
| `loader_text.go` | Text/markdown/image/video content extractors (synchronous). |
| `loader_office.go` | PDF/DOCX/XLSX/PPTX extractors via `github.com/getkawai/tools/gooxml` and `github.com/kawai-network/x/pdf`. |
| `processor.go` | `Processor` — the high-level orchestrator. `ProcessFile`, `DeleteFile`, async image/video, custom-chunker path. |
| `chunker.go` | Pluggable `Chunker` interface plus `CharChunker` (recursive char splitter) and `TokenChunker` (caller-provided `Tokenizer`). |
| `ocr.go` | Tesseract wrapper (`ExtractTextWithTesseract`) + LLM-based OCR cleanup (`CleanupOCRText`). |
| `fs.go` | Hookable helpers (`timeNow`, `statFile`, `baseName`, `mkdirAll`, `copyFile`) for testability. |
| `storage.go` | SHA-256 hashing, `CopyToStorage`, `SafeDelete` (only deletes if path is under `baseDir`). |
| `mdsplitter.go` | Markdown header splitter used by the built-in `RAGChunker`. |
| `ragcore.go` | Public RAG types: `VectorMatch`, `VectorItem`, `SearchParams`, `Embedder`, `ChunkStore`, `DistanceMetric`, `SearchResult`, `BatchSearchRequest`, `BatchSearchResult`. |
| `rag_processor.go` | `RAGProcessor` — chunk → embed → persist chunk text + vector → update file stats. |
| `rag_chunker.go` | `RAGChunker` — built-in chunker that routes by file type (PDF page-merge, markdown header split, recursive fallback). |
| `rag_cache.go` | `EmbeddingCache` — in-memory SHA256-keyed LRU+TTL wrapping any `Embedder`. |
| `rag_searcher.go` | `Searcher` — embed query → vector search → chunk hydration → fileID filter. |
| `pgvector_store.go` | `PgVectorStore` — pgvector + HNSW vector store. Schema isolation (lives in `fileprocessor` schema), batch search via LATERAL, `SetEfSearch`/`ResetEfSearch`, `GetIndexStats`. |
| `public_embeddings_store.go` | `PublicEmbeddingsStore` — `PgVectorStore` lookalike that targets `public.embeddings` (the lobehub table). Pinned to dim=1024 (the schema's hard constraint). Hydrates `file_id` via a join to `public.file_chunks`. Use it when your embedder produces 1024-dim vectors and you want to share storage with the host app's existing RAG. |
| `pg_file_store.go` | `PostgresFileStore` (+ `PostgresChunkStore` adapter) — durable storage against the lobehub `public` schema (files/documents/chunks/file_chunks/document_chunks). One struct, two interfaces. |
| `*_test.go` | `fileprocessor_test.go` (chunkers, loader, hash), `ragcore_test.go` (cache), `pgvector_store_test.go` (integration), `pg_file_store_test.go` (integration), `public_embeddings_store_test.go` (integration). |

## Architecture and data flow

```
Processor.ProcessFile(req)
  ├─ FileLoader.LoadFile                    (loader.go / loader_text.go / loader_office.go)
  │    └─ dispatch by extension → loadContent
  ├─ store.CreateFile(FileRecord)           ← FileStore interface
  ├─ store.CreateDocument(DocumentRecord)
  └─ switch on file type:
        image  → go processImageDescriptionAsync    (tesseract → optional LLM cleanup → optional VL → AppendToDocument → RAG)
        video  → go processVideoDescriptionAsync    (currently a no-op stub — logs warn)
        other  → if EnableRAG && CanChunkForRAG → runRAG (sync)
                   ├─ chunker.Chunk(content)        ← user-provided Chunker
                   ├─ store.CreateChunk for each
                   └─ ragProcessor.ProcessFile      (re-chunks via RAGChunker, embeds, upserts vectors)
```

```
Searcher.SemanticSearch(query)
  ├─ embedder.Embed(query)
  ├─ store.Search(2×limit)                  (PgVectorStore, returns VectorMatch[])
  ├─ chunks.GetChunksByIDs(ids)             (ChunkStore interface)
  └─ filter by FileIDs (if provided), then truncate to limit
```

`DeleteFile` always: `GetFile` → `RAGProcessor.DeleteFileVectors` → `FileStore.DeleteFile` → `SafeDelete` (if `FileBaseDir` configured).

## Key interfaces the host app must implement

```go
type FileStore interface {
    CreateFile(ctx, FileRecord) (string, error)
    CreateDocument(ctx, DocumentRecord) (string, error)
    GetDocumentByFileID(ctx, fileID) (StoredDocument, error)
    AppendToDocument(ctx, fileID, additionalContent string) error
    UpdateFileChunkStats(ctx, fileID, ChunkStats) error
    CreateChunk(ctx, ChunkRecord) (string, error)
    GetFile(ctx, id) (StoredFile, error)
    DeleteFile(ctx, fileID) error
}

type Embedder interface {
    Embed(ctx, texts []string) ([][]float32, error)
    Dimension() int
}

type VectorStore interface { /* Upsert, UpsertBatch, Search, DeleteByID, DeleteByFileID, Close */ }

type ChunkStore interface { /* GetDocument, CreateChunk, GetChunksByIDs, GetFile, UpdateFileChunkStats */ }
```

Optional interfaces: `VLProvider` (image description), `LanguageModel` (OCR cleanup), `Chunker` (override default chunker).

## Conventions and gotchas

- **Error sentinels**: `ErrNotFound`, `ErrUnsupportedFileType`. Detect with `errors.Is`. The library wraps store errors with these so the higher layers can distinguish missing records from real errors.
- **ID assignment**: `FileStore`/`ChunkStore` should honor pre-populated `ID` fields when non-empty. `RAGProcessor` does pre-generate chunk IDs as `documentID-uuid`.
- **Dimensions must match**: `NewPgVectorStore(ctx, dsn, dim)` validates the existing `<schema>.vectors.embedding` column dimension via `pg_attribute.atttypmod`. Mismatch = error. Choose a dimension per database/schema and stick with it.
- **Schema isolation**: `PgVectorStore` writes inside the `fileprocessor` schema (see `DefaultSchema`) by default — never `public`. `PostgresFileStore` writes inside `public` (the lobehub schema) directly because that's where `files`/`documents`/`chunks` already live.
- **Vector types**: `pgvector.NewVector(embedding)` wraps the `[]float32` for binding. The store also validates the slice length on every `Upsert`/`Search` so a wrong-dim embedder fails fast.
- **Embedding cache** is **in-memory only** (no persistence). It is safe for concurrent use and is the right place to wrap any embedder.
- **Distance → similarity** is metric-aware in `distanceToSimilarityPg`:
  - cosine: `1 - distance`
  - inner_product: `-distance`
  - l2 / others: `1 / (1 + d)` (matches the historical DuckDB transform)
  Cross-metric similarity scores are not comparable.
- **HNSW config knobs**: `Metric`, `M`, `EfConstruction` are baked into the index at build time. Changing them requires `DROP INDEX` + `CREATE INDEX` (or just call `NewPgVectorStoreWithConfig` after dropping the old index). `EfSearch` is a per-session GUC: use `SetEfSearch` / `ResetEfSearch`.
- **`BatchSearch`** issues a single SQL with a LATERAL join over a `VALUES` table. Reuse it for any fan-out scenario. Expect ~60× speedup vs. looping `Search` in Go.
- **`UpsertBatch`**: tries `COPY FROM` first for raw bulk inserts. If any row in the batch collides with an existing PK, the COPY fails (pgvector cannot express `ON CONFLICT` in `COPY`) and the store transparently retries the whole batch via per-row UPSERT inside a single transaction. The batch is all-or-nothing.
- **Test hooks**: `timeNow`, `statFile`, `baseName`, `mkdirAll`, `copyFile` are package-level vars to allow mocking in tests.
- **Async image processing** runs in a goroutine spawned by `Processor.ProcessFile`. The response returns immediately with `Processing: true`; the RAG pass happens after OCR/VL finishes. Always check `Response.Processing` rather than waiting on `ChunkIDs`.
- **Video processing is a stub** (`processVideoDescriptionAsync` only logs a warning). Don't expect chunk IDs for video uploads.
- **Tesseract** must be installed on `PATH` and is invoked via `exec.LookPath("tesseract")`. The library tries `eng+ind+jpn+chi_sim` first and falls back to default. If neither produces >20 chars, it falls back to the configured `VLProvider` (and finally to skipping).
- **OCR cleanup** runs with `OCRCleanupTimeout` (default 30s) and is skipped entirely if `LanguageModel` is nil — in which case raw Tesseract output is used.
- **Custom chunker path** (`Processor.Config.Chunker != nil`) persists chunks via `FileStore.CreateChunk` first, then re-runs `RAGProcessor.ProcessFile` which **re-chunks internally** with `RAGChunker`. Embeddings come from the re-chunked content, not the caller's chunks. Status is set to `"custom-chunker-pending"` between the two passes.
- **Office docs** use `github.com/getkawai/tools/gooxml` and pass `/files` as the image URL prefix in `ToMarkdownWithImageURLs`. This is a string the host app is expected to serve images from.
- **PDF extraction** uses `github.com/kawai-network/x/pdf`. Empty pages are replaced with `[Unable to extract text from this page]` so the chunker still has something to work with.
- **Markdown header splitter** (`mdsplitter.go`) honors code fences (``` and ~~~) and tracks the most-recently-active headers as `h2`/`h3` metadata on each chunk.
- **Module path** is `github.com/kawai-network/fileprocessor`. Tests use the standard `testing` package (no testify in the actual `*_test.go` files in this repo, even though it's a transitive dep in `go.sum`).
- **Module name + license** are already wired in `go.mod` and `LICENSE` (Apache-2.0).

### PublicEmbeddingsStore gotchas (lobehub `public.embeddings`)

- **Dim is pinned to 1024**. The lobehub schema hard-codes `vector(1024)`. The constructor rejects any other dim with a clear error. If your embedder produces 384/768/1536, use [`PgVectorStore`](#conventions-and-gotchas) in the `fileprocessor` schema instead.
- **One embedding per chunk**: there's a UNIQUE constraint on `embeddings.chunk_id` (and the table has no `file_id` column). The `Upsert` uses `ON CONFLICT (chunk_id) DO UPDATE` to replace the vector in place.
- **`file_id` is hydrated via JOIN to `public.file_chunks`** in `Search` results. Chunks not linked to any file return an empty `file_id`.
- **Deletes cascade from `chunks`**: `embeddings.chunk_id` FKs to `chunks.id` with `ON DELETE CASCADE`. When `PostgresFileStore.DeleteFile` removes the chunks, the embeddings go with them. The store's own `DeleteByID`/`DeleteByFileID` are explicit deletes (faster, don't depend on FK cascade timing).
- **HNSW index name is `public_embeddings_hnsw_idx`**: created on `init` if missing, with `vector_cosine_ops` by default. Reuses the index across process restarts.
- **Rows written by this store have `model = 'fileprocessor'`**. The host app's own embeddings pipeline should use a different `model` value so the two can coexist.
- **Empty `client_id`/`workspace_id` from the test user FKs** are wrapped in `NULLIF($N, '')` in the file-store SQL; the existing `files_client_id_user_id_unique` constraint treats two NULL client_ids as distinct, so multiple files per user work fine.

### PostgresFileStore gotchas (lobehub schema)

- **File vs Chunk store split**: `PostgresFileStore` implements `FileStore`. `PostgresChunkStore` (returned by `fileStore.ChunkStore()`) implements `ChunkStore`. The two interfaces have name collisions (`CreateChunk`, `GetFile`, `UpdateFileChunkStats`) with incompatible signatures — a single struct can't implement both. Always use the adapter.
- **Tenancy**: every write stamps the `Owner.UserID` (required), plus optional `WorkspaceID` and `ClientID`. The existing `users.id` FK is enforced — non-existent user IDs are rejected. Integration tests use a seeded test user `fp_test_user_seed`.
- **`files.file_hash` is NOT written** by the library. That column FKs to `global_files.hash_id` (the lobehub dedup table), so writing a content hash here would require a matching global_files row first. The library surfaces the hash under `files.metadata->'file_hash'` instead. `StoredFile.Hash` is therefore empty for files created via this adapter.
- **Chunk stats live in `files.metadata->'chunk_stats'`** (jsonb), not in dedicated columns. This avoids a migration; `jsonb_set` handles the merge.
- **`workspace_id` empty-string handling**: the adapter wraps every `workspace_id` bind in `NULLIF($N, '')` so empty strings become SQL NULL and bypass the `workspace_id → workspaces(id)` FK. Pass an empty `WorkspaceID` to opt out of workspace scoping.
- **`DeleteFile` manual cascade**: the existing FK from `documents.file_id → files.id` is `ON DELETE SET NULL`, not CASCADE. The adapter deletes in this order inside one transaction: `document_chunks` → `file_chunks` (cascades to `chunks`/`embeddings`) → `documents` → `files`.
- **`document_chunks.document_id` widened**: the original schema had `varchar(30)`. Migration `0112_fileprocessor_file_size_bigint.sql` widens it to `varchar(255)` so library-generated UUIDs fit.
- **`files.size` widened**: same migration widens `files.size` and `global_files.size` from `integer` to `bigint` to support files > 2 GiB.
- **`ChunkStore.CreateChunk` fills both junctions** (`file_chunks` AND `document_chunks`) by looking up the document's `file_id`. This lets `Searcher.SemanticSearch` hydrate file names via the `file_chunks` join.
- **`ChunkStore.GetChunksByIDs` joins `file_chunks`** to populate `Chunk.FileID`. A chunk with no file link returns an empty `FileID`.

## Testing approach

- Standard `testing` package only. No testify, no gomock.
- **Unit tests** run always: `go test ./...`
  - `fileprocessor_test.go`: chunkers, file loader detection, hash, processor constructor validation.
  - `ragcore_test.go`: embedding cache with `fakeEmbedder`.
  - `pgvector_store_test.go`: HNSW config defaults, normalize, dimension parser. These are pure unit tests.
- **Integration tests** are gated by the `//go:build integration` tag and require `FILEPROCESSOR_TEST_PG_DSN`:
  - `TestPgVectorStoreRoundTrip`, `TestPgVectorStoreBatchSearch`, `TestPgVectorStoreDeleteByID`, `TestPgVectorStoreDeleteByFileID`, `TestPgVectorStoreDimensionMismatch`, `TestPgVectorStoreSetEfSearch`, `TestPgVectorStoreUpsertBatchFallback`, `TestPgVectorStoreGetIndexStats`.
  - Each test creates a randomly-named schema (`fp_test_<TestName>_<randomhex>`) and drops it CASCADE on cleanup. The library's default `public` schema is never touched.
  - The dimension of the test store is fixed per test (`4`, `3`, etc.) — keep test fixtures small to keep the suite fast.
- `fakeEmbedder` in `ragcore_test.go` is the reference for faking an `Embedder` in new unit tests.
- If you change HNSW config semantics, run the integration suite to verify.

## Things NOT in this repo

- No `FileStore` reference implementation for arbitrary schemas (the `PostgresFileStore` is lobehub-specific). Other hosts need their own adapter.
- No video processing implementation (stub only).
- No retry/backoff for `VLProvider` or `LanguageModel` calls.
- No persistent embedding cache (in-memory only).
- No chunk-deduplication on upsert; the unique key is the chunk ID assigned by `RAGProcessor` (`documentID-uuid`).
- No automatic HNSW index rebuild on config drift (the previous DuckDB impl had this). Callers changing `Metric` / `M` / `EfConstruction` must drop and recreate the index manually.

## End-to-end wiring (Postgres + lobehub)

```go
store, err := fileprocessor.NewPostgresFileStore(ctx,
    "postgresql://user:pass@host:5432/db?sslmode=require",
    fileprocessor.PostgresFileStoreOwner{UserID: "u_abc", WorkspaceID: "ws_1"},
)
if err != nil { /* fatal */ }

vecStore, err := fileprocessor.NewPgVectorStore(ctx, dsn, 1024)
if err != nil { /* fatal */ }

embedder := fileprocessor.NewEmbeddingCache(myEmbedder, nil)

rag := fileprocessor.NewRAGProcessor(store.ChunkStore(), vecStore, embedder, nil)

proc, err := fileprocessor.New(fileprocessor.Config{
    FileStore:    store,
    RAGProcessor: rag,
    Chunker:      fileprocessor.NewCharChunker(1000, 200),
    FileBaseDir:  "/var/myapp/files",
})

resp, err := proc.ProcessFile(ctx, fileprocessor.Request{
    FilePath: "/tmp/upload.pdf", Filename: "upload.pdf", EnableRAG: true,
})

searcher := fileprocessor.NewSearcher(vecStore, store.ChunkStore(), embedder)
results, _ := searcher.SemanticSearch(ctx, fileprocessor.SearchParamsSearcher{
    Query: "what is the timeline?", Limit: 10,
})
```

## When you change something

- After any change: `CGO_ENABLED=0 go build ./... && go test ./... && go vet ./...`.
- If you also changed pgvector or pg_file_store behavior, run the integration suite: `FILEPROCESSOR_TEST_PG_DSN=... go test -tags=integration ./...`.
- If you add a new file type, update the `textExtensions`/`imageExtensions`/`videoExtensions` maps in `loader.go`, the `loadContent` switch in `loader_text.go`, and the `ChunkDocument` switch in `rag_chunker.go`.
- If you change `FileStore`/`ChunkStore`/`Embedder` interfaces, this is a breaking change for all downstream consumers.
- If you add a new operator class or distance metric, update `metricOpsClass`, `metricDistanceOp`, and `distanceToSimilarityPg` together.
- If you add a new HNSW knob to `PgHNSWConfig`, also add it to `DefaultPgHNSWConfig` and `(*PgHNSWConfig).normalize()`.
- If you change a lobehub table column, the matching `PostgresFileStore` SQL in `pg_file_store.go` likely needs an update too. Mirror any new column constraints with a `NULLIF($N, '')` wrap or equivalent when binding from Go.
