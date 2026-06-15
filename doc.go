// Package fileprocessor is a reusable file processing pipeline that handles
// loading, content extraction, chunking, embedding, and optional
// Vision-Language / OCR description generation for a wide range of document
// types (PDF, DOCX, XLSX, PPTX, TXT, images, videos).
//
// The library is database-agnostic. Clients provide a [FileStore]
// implementation (e.g. backed by SQLite, Postgres, or anything else) and a
// [*ragcore.RAGProcessor] instance for chunking + embedding + vector storage.
// The chunker is pluggable: callers can choose between a character-based
// splitter ([CharChunker]) or a token-aware splitter ([TokenChunker]).
//
// Typical wiring:
//
//	store := myapp.NewFileStore(db)              // implements FileStore
//	rag   := ragcore.NewRAGProcessor(store, vec, embedder, nil)
//	proc  := fileprocessor.New(fileprocessor.Config{
//	    FileStore:    store,
//	    RAGProcessor: rag,
//	    Chunker:      fileprocessor.NewTokenChunker(1024, 200, tok),
//	    FileBaseDir:  "/var/myapp/files",
//	})
//	resp, err := proc.ProcessFile(ctx, fileprocessor.Request{
//	    FilePath: "/tmp/upload.pdf",
//	    Filename: "upload.pdf",
//	    EnableRAG: true,
//	})
package fileprocessor
