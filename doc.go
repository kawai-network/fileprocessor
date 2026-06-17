// Package fileprocessor is a reusable file processing pipeline. It handles
// loading, content extraction, chunking, embedding, and optional
// Vision-Language / OCR description generation for a wide range of document
// types (PDF, DOCX, XLSX, PPTX, TXT, images, videos), and it bundles the RAG
// core (DuckDB-backed vector store, embedding cache, semantic search, and the
// RAGProcessor ingestion path) into the same module.
//
// The library is database-agnostic. Clients provide a [FileStore]
// implementation (e.g. backed by SQLite, Postgres, or anything else). The
// chunker is pluggable: callers can choose between a character-based splitter
// ([CharChunker]) or a token-aware splitter ([TokenChunker]). For semantic
// search the package ships a DuckDB-backed [VectorStore] ([NewDuckDBStore])
// and a [Searcher] that combines an [Embedder] (optionally wrapped in
// [EmbeddingCache]) with any [ChunkStore].
//
// Typical wiring:
//
//	store := myapp.NewFileStore(db)               // implements FileStore
//	embedder := myEmbedder{}                      // implements Embedder
//	duck, _ := fileprocessor.NewDuckDBStore("/var/lib/myapp/vectors.db", embedder.Dimension())
//	rag   := fileprocessor.NewRAGProcessor(store, duck, fileprocessor.NewEmbeddingCache(embedder, nil), nil)
//	proc  := fileprocessor.New(fileprocessor.Config{
//	    FileStore:    store,
//	    RAGProcessor: rag,
//	    Chunker:      fileprocessor.NewTokenChunker(1024, 200, tok),
//	    FileBaseDir:  "/var/myapp/files",
//	})
//	resp, err := proc.ProcessFile(ctx, fileprocessor.Request{
//	    FilePath:  "/tmp/upload.pdf",
//	    Filename:  "upload.pdf",
//	    EnableRAG: true,
//	})
package fileprocessor
