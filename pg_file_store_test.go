//go:build integration

package fileprocessor

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

// pgFileStoreDSN is the env var that must hold a Postgres DSN for
// PostgresFileStore integration tests. The DSN must point at the lobehub
// database (which has the files/documents/chunks tables and a UUID
// gen_random_uuid() default).
//
// The tests use a per-run test user_id and clean up by deleting all rows
// stamped with that user_id. They never read or write app data.
const pgFileStoreDSN = "FILEPROCESSOR_TEST_PG_DSN"

// testUserSeed is the user_id the test helper writes as. It must exist in
// the `users` table (FK enforced). One row is created at test setup and
// shared across all tests in the run.
const testUserSeed = "fp_test_user_seed"

func requirePgFileStore(t *testing.T) string {
	t.Helper()
	dsn := os.Getenv(pgFileStoreDSN)
	if dsn == "" {
		t.Skipf("%s not set; skipping pg_file_store integration test", pgFileStoreDSN)
	}
	return dsn
}

// ensureTestUser makes sure the test seed user exists in the users table.
// Safe to call from multiple tests.
func ensureTestUser(t *testing.T, dsn string) {
	t.Helper()
	pool, err := pgxpool.New(context.Background(), dsn)
	if err != nil {
		t.Fatalf("ensureTestUser: pgxpool.New: %v", err)
	}
	defer pool.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_, err = pool.Exec(ctx, `INSERT INTO users (id, username) VALUES ($1, $1) ON CONFLICT (id) DO NOTHING`, testUserSeed)
	if err != nil {
		t.Fatalf("ensureTestUser: INSERT users: %v", err)
	}
}

// cleanupByUser removes every row owned by the given user_id. Used by
// t.Cleanup. Cascade-safe (deletes junctions first, then chunks, then
// documents, then files).
func cleanupByUser(t *testing.T, dsn, userID string) {
	t.Helper()
	pool, err := pgxpool.New(context.Background(), dsn)
	if err != nil {
		t.Logf("cleanup: pgxpool.New: %v", err)
		return
	}
	defer pool.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	for _, q := range []string{
		`DELETE FROM document_chunks WHERE user_id = $1`,
		`DELETE FROM file_chunks WHERE user_id = $1`,
		`DELETE FROM chunks WHERE user_id = $1`,
		`DELETE FROM documents WHERE user_id = $1`,
		`DELETE FROM files WHERE user_id = $1`,
	} {
		if _, err := pool.Exec(ctx, q, userID); err != nil {
			t.Logf("cleanup %q: %v", q, err)
		}
	}
}

// newPgFileStore builds a PostgresFileStore against the test seed user.
// Registers cleanup that wipes any rows stamped with that user_id.
func newPgFileStore(t *testing.T, dsn string) (*PostgresFileStore, func()) {
	t.Helper()
	ensureTestUser(t, dsn)
	store, err := NewPostgresFileStore(context.Background(), dsn, PostgresFileStoreOwner{UserID: testUserSeed})
	if err != nil {
		t.Fatalf("NewPostgresFileStore: %v", err)
	}
	t.Cleanup(func() {
		cleanupByUser(t, dsn, testUserSeed)
		_ = store.Close()
	})
	return store, func() { _ = store.Close() }
}

func TestPgFileStoreCreateAndGetFile(t *testing.T) {
	dsn := requirePgFileStore(t)
	store, _ := newPgFileStore(t, dsn)
	ctx := context.Background()

	id, err := store.CreateFile(ctx, FileRecord{
		Name:     "report.pdf",
		FileType: "pdf",
		Size:     5_000_000_000, // 5 GB — tests bigint migration
		URL:      "/files/report.pdf",
		Source:   "upload",
		Hash:     "abc123",
	})
	if err != nil {
		t.Fatalf("CreateFile: %v", err)
	}
	if id == "" {
		t.Fatal("CreateFile returned empty ID")
	}

	got, err := store.GetFile(ctx, id)
	if err != nil {
		t.Fatalf("GetFile: %v", err)
	}
	if got.Name != "report.pdf" || got.FileType != "pdf" {
		t.Errorf("GetFile = %+v, want Name=report.pdf FileType=pdf", got)
	}
	// Hash is surfaced under files.metadata->'file_hash' (the column FKs to
	// global_files.hash_id and is not written by the library). StoredFile.Hash
	// is therefore empty.
	if got.Hash != "" {
		t.Errorf("GetFile.Hash = %q, want empty (written to metadata instead)", got.Hash)
	}

	// GetFile on missing ID wraps ErrNotFound.
	_, err = store.GetFile(ctx, "does-not-exist")
	if err == nil || !errorsIs(err, ErrNotFound) {
		t.Errorf("GetFile missing: want ErrNotFound, got %v", err)
	}
}

func TestPgFileStoreCreateAndReadDocument(t *testing.T) {
	dsn := requirePgFileStore(t)
	store, _ := newPgFileStore(t, dsn)
	ctx := context.Background()

	fileID, err := store.CreateFile(ctx, FileRecord{Name: "x.txt", FileType: "txt", Size: 10, URL: "/x.txt"})
	if err != nil {
		t.Fatalf("CreateFile: %v", err)
	}
	docID, err := store.CreateDocument(ctx, DocumentRecord{
		FileID:          fileID,
		Title:           "x.txt",
		Filename:        "x.txt",
		Content:         "hello world",
		FileType:        "txt",
		TotalCharCount:  11,
		TotalLineCount:  1,
		Source:          "test",
		SourceType:      "file",
		PagesJSON:       "[{\"pageContent\":\"hello world\"}]",
		Metadata:        map[string]any{"k": "v"},
	})
	if err != nil {
		t.Fatalf("CreateDocument: %v", err)
	}
	if docID == "" {
		t.Fatal("empty docID")
	}

	doc, err := store.GetDocumentByFileID(ctx, fileID)
	if err != nil {
		t.Fatalf("GetDocumentByFileID: %v", err)
	}
	if doc.Content != "hello world" || doc.Title != "x.txt" {
		t.Errorf("doc = %+v", doc)
	}
}

func TestPgFileStoreAppendToDocument(t *testing.T) {
	dsn := requirePgFileStore(t)
	store, _ := newPgFileStore(t, dsn)
	ctx := context.Background()

	fileID, _ := store.CreateFile(ctx, FileRecord{Name: "a.md", FileType: "md", Size: 1, URL: "/a.md"})
	_, _ = store.CreateDocument(ctx, DocumentRecord{
		FileID: fileID, Title: "a.md", Filename: "a.md", Content: "first",
		FileType: "md", TotalCharCount: 5, TotalLineCount: 1, Source: "t", SourceType: "file",
	})
	if err := store.AppendToDocument(ctx, fileID, "\n\nsecond"); err != nil {
		t.Fatalf("AppendToDocument: %v", err)
	}
	doc, _ := store.GetDocumentByFileID(ctx, fileID)
	if doc.Content != "first\n\nsecond" {
		t.Errorf("after append: %q", doc.Content)
	}
}

func TestPgFileStoreUpdateFileChunkStats(t *testing.T) {
	dsn := requirePgFileStore(t)
	store, _ := newPgFileStore(t, dsn)
	ctx := context.Background()

	fileID, _ := store.CreateFile(ctx, FileRecord{Name: "b.txt", FileType: "txt", Size: 1, URL: "/b.txt"})
	if err := store.UpdateFileChunkStats(ctx, fileID, ChunkStats{
		ChunkCount:      42,
		ChunkingStatus:  "success",
		EmbeddingStatus: "success",
	}); err != nil {
		t.Fatalf("UpdateFileChunkStats: %v", err)
	}

	// Read back via raw SQL.
	pool, _ := pgxpool.New(ctx, dsn)
	defer pool.Close()
	var meta []byte
	if err := pool.QueryRow(ctx, `SELECT metadata FROM files WHERE id = $1`, fileID).Scan(&meta); err != nil {
		t.Fatalf("read metadata: %v", err)
	}
	if len(meta) == 0 || !contains(string(meta), `"chunk_stats"`) {
		t.Errorf("metadata missing chunk_stats: %s", string(meta))
	}
	if !contains(string(meta), `"ChunkCount": 42`) {
		t.Errorf("metadata missing ChunkCount=42: %s", string(meta))
	}
}

func TestPgFileStoreCreateChunkFillsJunctions(t *testing.T) {
	dsn := requirePgFileStore(t)
	store, _ := newPgFileStore(t, dsn)
	ctx := context.Background()

	fileID, _ := store.CreateFile(ctx, FileRecord{Name: "c.md", FileType: "md", Size: 1, URL: "/c.md"})
	docID, _ := store.CreateDocument(ctx, DocumentRecord{
		FileID: fileID, Title: "c.md", Filename: "c.md", Content: "c",
		FileType: "md", TotalCharCount: 1, TotalLineCount: 1, Source: "t", SourceType: "file",
	})

	chunkID, err := store.CreateChunk(ctx, ChunkRecord{
		FileID:     fileID,
		DocumentID: docID,
		Text:       "chunk text",
		Index:      0,
		Type:       "text",
		Metadata:   map[string]any{"filename": "c.md"},
	})
	if err != nil {
		t.Fatalf("CreateChunk: %v", err)
	}
	if chunkID == "" {
		t.Fatal("empty chunkID")
	}

	// Verify the junctions were filled.
	pool, _ := pgxpool.New(ctx, dsn)
	defer pool.Close()
	var fcCount, dcCount int
	_ = pool.QueryRow(ctx, `SELECT COUNT(*) FROM file_chunks WHERE chunk_id = $1 AND file_id = $2`, chunkID, fileID).Scan(&fcCount)
	_ = pool.QueryRow(ctx, `SELECT COUNT(*) FROM document_chunks WHERE chunk_id = $1 AND document_id = $2`, chunkID, docID).Scan(&dcCount)
	if fcCount != 1 {
		t.Errorf("file_chunks missing: count=%d", fcCount)
	}
	if dcCount != 1 {
		t.Errorf("document_chunks missing: count=%d", dcCount)
	}
}

func TestPgFileStoreDeleteFileCascades(t *testing.T) {
	dsn := requirePgFileStore(t)
	store, _ := newPgFileStore(t, dsn)
	ctx := context.Background()

	fileID, _ := store.CreateFile(ctx, FileRecord{Name: "d.md", FileType: "md", Size: 1, URL: "/d.md"})
	docID, _ := store.CreateDocument(ctx, DocumentRecord{
		FileID: fileID, Title: "d.md", Filename: "d.md", Content: "d",
		FileType: "md", TotalCharCount: 1, TotalLineCount: 1, Source: "t", SourceType: "file",
	})
	_, _ = store.CreateChunk(ctx, ChunkRecord{
		FileID: fileID, DocumentID: docID, Text: "x", Index: 0, Type: "text",
	})

	if err := store.DeleteFile(ctx, fileID); err != nil {
		t.Fatalf("DeleteFile: %v", err)
	}

	pool, _ := pgxpool.New(ctx, dsn)
	defer pool.Close()
	var fCount, dCount, cCount int
	_ = pool.QueryRow(ctx, `SELECT COUNT(*) FROM files WHERE id = $1`, fileID).Scan(&fCount)
	_ = pool.QueryRow(ctx, `SELECT COUNT(*) FROM documents WHERE id = $1`, docID).Scan(&dCount)
	_ = pool.QueryRow(ctx, `SELECT COUNT(*) FROM chunks WHERE id IN (SELECT chunk_id FROM document_chunks WHERE document_id = $1)`, docID).Scan(&cCount)
	if fCount != 0 {
		t.Errorf("file not deleted: count=%d", fCount)
	}
	if dCount != 0 {
		t.Errorf("document not deleted: count=%d", dCount)
	}
	if cCount != 0 {
		t.Errorf("chunk not deleted via cascade: count=%d", cCount)
	}
}

func TestPgFileStoreLargeFileSize(t *testing.T) {
	// Verifies the bigint migration: 5 GB > max int32 (2,147,483,647).
	dsn := requirePgFileStore(t)
	store, _ := newPgFileStore(t, dsn)
	ctx := context.Background()

	const bigSize = int64(5_000_000_000)
	fileID, err := store.CreateFile(ctx, FileRecord{
		Name:     "big.bin",
		FileType: "bin",
		Size:     bigSize,
		URL:      "/big.bin",
	})
	if err != nil {
		t.Fatalf("CreateFile: %v", err)
	}

	pool, _ := pgxpool.New(ctx, dsn)
	defer pool.Close()
	var got int64
	if err := pool.QueryRow(ctx, `SELECT size FROM files WHERE id = $1`, fileID).Scan(&got); err != nil {
		t.Fatalf("read size: %v", err)
	}
	if got != bigSize {
		t.Errorf("size = %d, want %d (bigint migration regression?)", got, bigSize)
	}
}

func TestPgChunkStoreGetDocumentAndChunk(t *testing.T) {
	dsn := requirePgFileStore(t)
	store, _ := newPgFileStore(t, dsn)
	cs := store.ChunkStore()
	ctx := context.Background()

	fileID, _ := store.CreateFile(ctx, FileRecord{Name: "e.md", FileType: "md", Size: 1, URL: "/e.md"})
	docID, _ := store.CreateDocument(ctx, DocumentRecord{
		FileID: fileID, Title: "e.md", Filename: "e.md", Content: "e body",
		FileType: "md", TotalCharCount: 6, TotalLineCount: 1, Source: "t", SourceType: "file",
	})

	doc, err := cs.GetDocument(ctx, docID)
	if err != nil {
		t.Fatalf("GetDocument: %v", err)
	}
	if doc.Content != "e body" {
		t.Errorf("doc.Content = %q, want %q", doc.Content, "e body")
	}

	// CreateChunk through ChunkStore.
	chunkID, err := cs.CreateChunk(ctx, CreateChunkParams{
		ID:         "lib-gen-id-123",
		DocumentID: docID,
		Text:       "chunk body",
		Index:      0,
		Type:       "text",
		Metadata:   `{"filename":"e.md"}`,
	})
	if err != nil {
		t.Fatalf("CreateChunk: %v", err)
	}

	// GetChunksByIDs should return the chunk with file_id populated via join.
	chunks, err := cs.GetChunksByIDs(ctx, []string{chunkID})
	if err != nil {
		t.Fatalf("GetChunksByIDs: %v", err)
	}
	if len(chunks) != 1 {
		t.Fatalf("got %d chunks, want 1", len(chunks))
	}
	if chunks[0].Text != "chunk body" {
		t.Errorf("chunk text = %q", chunks[0].Text)
	}
	if chunks[0].FileID != fileID {
		t.Errorf("chunk file_id = %q, want %q (joined from file_chunks)", chunks[0].FileID, fileID)
	}

	// GetFile on ChunkStore returns RAGFile.
	rf, err := cs.GetFile(ctx, fileID)
	if err != nil {
		t.Fatalf("GetFile: %v", err)
	}
	if rf.Name != "e.md" {
		t.Errorf("RAGFile name = %q", rf.Name)
	}

	// UpdateFileChunkStats through ChunkStore.
	if err := cs.UpdateFileChunkStats(ctx, UpdateFileStatsParams{
		FileID:          fileID,
		ChunkCount:      5,
		ChunkingStatus:  "success",
		EmbeddingStatus: "success",
	}); err != nil {
		t.Fatalf("UpdateFileChunkStats: %v", err)
	}
}

// errorsIs is a tiny local helper to avoid importing errors in the test file.
func errorsIs(err, target error) bool {
	for e := err; e != nil; {
		if e == target {
			return true
		}
		type unwrapper interface{ Unwrap() error }
		if u, ok := e.(unwrapper); ok {
			e = u.Unwrap()
		} else {
			return false
		}
	}
	return false
}

func contains(s, substr string) bool {
	for i := 0; i+len(substr) <= len(s); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
