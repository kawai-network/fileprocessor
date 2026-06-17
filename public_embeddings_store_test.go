//go:build integration

package fileprocessor

import (
	"context"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

// The PublicEmbeddingsStore tests reuse the lobehub seeded test user
// and the public schema. Each test creates a file/document/chunk via
// PostgresFileStore, writes embeddings, exercises Search / BatchSearch
// / Delete, and the per-user cleanup registered by newPgFileStore
// wipes everything at the end.

func newPublicEmbeddingsStore(t *testing.T, dsn string) *PublicEmbeddingsStore {
	t.Helper()
	store, err := NewPublicEmbeddingsStore(context.Background(), dsn, 1024, nil)
	if err != nil {
		t.Fatalf("NewPublicEmbeddingsStore: %v", err)
	}
	// HNSW index is created on init; no per-test cleanup needed because
	// the index is idempotent and the rows belong to specific chunk_ids
	// that get removed via the file-store cleanup.
	t.Cleanup(func() { _ = store.Close() })
	return store
}

// makeChunkWithFile creates a file + document + chunk and returns the
// chunk id (uuid string). Uses PostgresFileStore so the FK chain is
// correctly populated. Each file gets a unique client_id (derived from
// the test name + nanosecond timestamp) to avoid the
// files_client_id_user_id_unique constraint when multiple files are
// created in the same test for the same user.
func makeChunkWithFile(t *testing.T, fs *PostgresFileStore, name string) string {
	t.Helper()
	ctx := context.Background()
	fileID, err := fs.CreateFile(ctx, FileRecord{
		Name: name, FileType: "md", Size: 1, URL: "/" + name,
		Source: "test:" + name + ":" + t.Name(),
	})
	if err != nil {
		t.Fatalf("CreateFile: %v", err)
	}
	docID, err := fs.CreateDocument(ctx, DocumentRecord{
		FileID: fileID, Title: name, Filename: name, Content: "x",
		FileType: "md", TotalCharCount: 1, TotalLineCount: 1, Source: "t", SourceType: "file",
	})
	if err != nil {
		t.Fatalf("CreateDocument: %v", err)
	}
	chunkID, err := fs.CreateChunk(ctx, ChunkRecord{
		FileID: fileID, DocumentID: docID, Text: "chunk " + name, Index: 0, Type: "text",
	})
	if err != nil {
		t.Fatalf("CreateChunk: %v", err)
	}
	return chunkID
}

func TestPublicEmbeddingsStoreRoundTrip(t *testing.T) {
	dsn := requirePgFileStore(t)
	fs, _ := newPgFileStore(t, dsn)
	es := newPublicEmbeddingsStore(t, dsn)
	ctx := context.Background()

	chunkA := makeChunkWithFile(t, fs, "a.md")
	chunkB := makeChunkWithFile(t, fs, "b.md")

	if err := es.Upsert(ctx, chunkA, "", makeVec1024FromSeed(1)); err != nil {
		t.Fatalf("Upsert a: %v", err)
	}
	if err := es.Upsert(ctx, chunkB, "", makeVec1024FromSeed(2)); err != nil {
		t.Fatalf("Upsert b: %v", err)
	}

	matches, err := es.Search(ctx, makeVec1024FromSeed(1), SearchParams{Limit: 5})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(matches) == 0 {
		t.Fatal("no matches")
	}
	if matches[0].ID != chunkA {
		t.Errorf("top match = %s, want %s", matches[0].ID, chunkA)
	}

	// Update via re-upsert: chunk_id UNIQUE → ON CONFLICT updates the row.
	// Verify by reading the row back from the DB and checking the new
	// model is still "fileprocessor" and the row was actually replaced
	// (not inserted twice). The cosine-distance behavior of the
	// upserted vector is already covered by TestPublicEmbeddingsStoreRoundTrip's
	// first search assertion.
	if err := es.Upsert(ctx, chunkA, "", makeVec1024FromSeed(3)); err != nil {
		t.Fatalf("re-Upsert a: %v", err)
	}
	pool, _ := pgxpool.New(context.Background(), dsn)
	defer pool.Close()
	verifyCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	var n int
	if err := pool.QueryRow(verifyCtx,
		`SELECT COUNT(*) FROM public.embeddings WHERE chunk_id = $1 AND model = 'fileprocessor'`,
		chunkA,
	).Scan(&n); err != nil {
		t.Fatalf("verify: %v", err)
	}
	if n != 1 {
		t.Errorf("expected 1 row for chunkA after re-upsert, got %d", n)
	}
}

func TestPublicEmbeddingsStoreBatchSearch(t *testing.T) {
	dsn := requirePgFileStore(t)
	fs, _ := newPgFileStore(t, dsn)
	es := newPublicEmbeddingsStore(t, dsn)
	ctx := context.Background()

	c1 := makeChunkWithFile(t, fs, "bs1.md")
	c2 := makeChunkWithFile(t, fs, "bs2.md")
	c3 := makeChunkWithFile(t, fs, "bs3.md")

	items := []VectorItem{
		{ID: c1, FileID: "", Embedding: makeVec1024FromSeed(10)},
		{ID: c2, FileID: "", Embedding: makeVec1024FromSeed(20)},
		{ID: c3, FileID: "", Embedding: makeVec1024FromSeed(30)},
	}
	if err := es.UpsertBatch(ctx, items); err != nil {
		t.Fatalf("UpsertBatch: %v", err)
	}

	queries := []BatchSearchRequest{
		{QueryID: "q1", Embedding: makeVec1024FromSeed(10)},
		{QueryID: "q2", Embedding: makeVec1024FromSeed(30)},
	}
	results, err := es.BatchSearch(ctx, queries, 1)
	if err != nil {
		t.Fatalf("BatchSearch: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("got %d results, want 2", len(results))
	}
	if len(results[0].Matches) != 1 || results[0].Matches[0].ID != c1 {
		t.Errorf("q1: expected %s first, got %+v", c1, results[0].Matches)
	}
	if len(results[1].Matches) != 1 || results[1].Matches[0].ID != c3 {
		t.Errorf("q2: expected %s first, got %+v", c3, results[1].Matches)
	}
}

func TestPublicEmbeddingsStoreDeleteByID(t *testing.T) {
	dsn := requirePgFileStore(t)
	fs, _ := newPgFileStore(t, dsn)
	es := newPublicEmbeddingsStore(t, dsn)
	ctx := context.Background()

	chunk := makeChunkWithFile(t, fs, "del1.md")
	if err := es.Upsert(ctx, chunk, "", makeVec1024FromSeed(50)); err != nil {
		t.Fatalf("Upsert: %v", err)
	}
	if err := es.DeleteByID(ctx, chunk); err != nil {
		t.Fatalf("DeleteByID: %v", err)
	}
	matches, _ := es.Search(ctx, makeVec1024FromSeed(50), SearchParams{Limit: 5})
	for _, m := range matches {
		if m.ID == chunk {
			t.Errorf("embedding for %s still present after DeleteByID", chunk)
		}
	}
}

func TestPublicEmbeddingsStoreDeleteByFileID(t *testing.T) {
	dsn := requirePgFileStore(t)
	fs, _ := newPgFileStore(t, dsn)
	es := newPublicEmbeddingsStore(t, dsn)
	ctx := context.Background()

	// Create one file with one chunk + embedding, and another file with
	// another chunk + embedding. DeleteByFileID on the first should only
	// remove the first chunk's embedding.
	chunkA := makeChunkWithFile(t, fs, "delfile_a.md")
	chunkB := makeChunkWithFile(t, fs, "delfile_b.md")
	if err := es.Upsert(ctx, chunkA, "", makeVec1024FromSeed(60)); err != nil {
		t.Fatalf("Upsert a: %v", err)
	}
	if err := es.Upsert(ctx, chunkB, "", makeVec1024FromSeed(60)); err != nil {
		t.Fatalf("Upsert b: %v", err)
	}

	// Look up file_id for chunkA.
	pool, _ := pgxpool.New(ctx, dsn)
	defer pool.Close()
	var fileA string
	_ = pool.QueryRow(ctx, `SELECT file_id FROM file_chunks WHERE chunk_id = $1`, chunkA).Scan(&fileA)
	if fileA == "" {
		t.Fatal("no file_chunks row for chunkA")
	}

	if err := es.DeleteByFileID(ctx, fileA); err != nil {
		t.Fatalf("DeleteByFileID: %v", err)
	}

	matches, _ := es.Search(ctx, makeVec1024FromSeed(60), SearchParams{Limit: 10})
	sawA, sawB := false, false
	for _, m := range matches {
		if m.ID == chunkA {
			sawA = true
		}
		if m.ID == chunkB {
			sawB = true
		}
	}
	if sawA {
		t.Errorf("chunkA embedding not deleted by file_id")
	}
	if !sawB {
		t.Errorf("chunkB embedding was deleted (should still exist)")
	}
}

func TestPublicEmbeddingsStoreRejectsWrongDim(t *testing.T) {
	dsn := requirePgFileStore(t)
	es := newPublicEmbeddingsStore(t, dsn)
	ctx := context.Background()

	// Search with wrong-dim vector.
	if _, err := es.Search(ctx, []float32{0.1, 0.2, 0.3}, SearchParams{Limit: 1}); err == nil {
		t.Error("expected dim mismatch error on Search, got nil")
	}
	// Upsert with wrong-dim vector.
	if err := es.Upsert(ctx, "x", "", []float32{0.1, 0.2, 0.3}); err == nil {
		t.Error("expected dim mismatch error on Upsert, got nil")
	}
}

func TestNewPublicEmbeddingsStoreRejectsNon1024(t *testing.T) {
	dsn := requirePgFileStore(t)
	if _, err := NewPublicEmbeddingsStore(context.Background(), dsn, 384, nil); err == nil {
		t.Error("expected dim=384 to be rejected")
	}
}

// makeVec1024 returns a 1024-dim vector with every component equal to v.
// Two vectors built from different v values are scalar multiples of each
// other — distance 0 in cosine space — so for the tests below we mix
// in an offset pattern to produce vectors that point in different
// directions. The exact magnitude does not matter; what matters is
// that the test vectors are not collinear.
//
// makeVec1024FromSeed returns a 1024-dim vector generated from a
// deterministic pseudo-random sequence. Vectors from different seeds
// point in sufficiently different directions that cosine search can
// rank them correctly. The seed only needs to be a different integer
// per test; values do not need to be cryptographic.
func makeVec1024FromSeed(seed uint32) []float32 {
	out := make([]float32, 1024)
	s := seed
	for i := range out {
		// Simple LCG (Numerical Recipes). Good enough for test vectors.
		s = s*1664525 + 1013904223
		// Map to [-0.5, 0.5].
		out[i] = float32(int32(s)) / float32(1<<32) * 0.5
	}
	return out
}

// makeVec1024 retained for tests that don't need distinguishability
// (e.g. dim-mismatch tests).
func makeVec1024(v float32) []float32 {
	out := make([]float32, 1024)
	for i := range out {
		out[i] = v
	}
	return out
}
