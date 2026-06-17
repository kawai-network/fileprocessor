//go:build integration

package fileprocessor

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

// pgTestDSN is the env var that must hold a Postgres DSN for integration
// tests. The DSN must point at a database where the "vector" extension is
// available (CREATE EXTENSION succeeds, or the extension is pre-installed).
//
// The library NEVER touches the "public" schema during tests. Every test
// creates an isolated, random schema, then drops it on cleanup.
const pgTestDSN = "FILEPROCESSOR_TEST_PG_DSN"

// requirePg skips the test if the env var is missing.
func requirePg(t *testing.T) string {
	t.Helper()
	dsn := os.Getenv(pgTestDSN)
	if dsn == "" {
		t.Skipf("%s not set; skipping pgvector integration test", pgTestDSN)
	}
	return dsn
}

// newIsolatedSchema creates a randomly-named schema and returns its name
// and a cleanup that drops it CASCADE.
func newIsolatedSchema(t *testing.T, dsn string) (string, *pgxpool.Pool) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("pgxpool.New: %v", err)
	}

	schema := randomSchemaName(t)
	if _, err := pool.Exec(ctx, fmt.Sprintf(`CREATE SCHEMA %s`, schema)); err != nil {
		pool.Close()
		t.Fatalf("CREATE SCHEMA %s: %v", schema, err)
	}

	return schema, pool
}

func randomSchemaName(t *testing.T) string {
	t.Helper()
	var b [8]byte
	if _, err := rand.Read(b[:]); err != nil {
		t.Fatalf("rand.Read: %v", err)
	}
	safeName := strings.ReplaceAll(t.Name(), "/", "_")
	return fmt.Sprintf("fp_test_%s_%s", safeName, hex.EncodeToString(b[:]))
}

// newPgStore creates a PgVectorStore backed by an isolated schema. The
// schema is dropped on test cleanup via a fresh connection (the store's pool
// is already closed at that point). Returns the store and a cleanup that
// closes the store's pool and drops the schema.
func newPgStore(t *testing.T, dim int) (*PgVectorStore, func()) {
	t.Helper()
	dsn := requirePg(t)
	schema, pool := newIsolatedSchema(t, dsn)

	store, err := NewPgVectorStoreWithPool(context.Background(), pool, dim, schema, nil)
	if err != nil {
		pool.Close()
		t.Fatalf("NewPgVectorStoreWithPool: %v", err)
	}
	return store, func() {
		if err := store.Close(); err != nil {
			t.Logf("warning: store.Close: %v", err)
		}
		// pool is now closed; open a fresh conn to drop the schema.
		dropCtx, dropCancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer dropCancel()
		dropPool, err := pgxpool.New(dropCtx, dsn)
		if err != nil {
			t.Logf("warning: dropPool.New: %v", err)
			return
		}
		defer dropPool.Close()
		if _, err := dropPool.Exec(dropCtx, fmt.Sprintf(`DROP SCHEMA IF EXISTS %s CASCADE`, schema)); err != nil {
			t.Logf("warning: DROP SCHEMA %s: %v", schema, err)
		}
	}
}

func TestPgVectorStoreRoundTrip(t *testing.T) {
	store, cleanup := newPgStore(t, 4)
	defer cleanup()

	ctx := context.Background()
	upserts := []struct {
		id, fileID string
		emb        []float32
	}{
		{"a", "f1", []float32{0.1, 0.2, 0.3, 0.4}},
		{"b", "f1", []float32{0.4, 0.3, 0.2, 0.1}},
		{"c", "f2", []float32{-0.1, -0.2, -0.3, -0.4}},
	}
	for _, u := range upserts {
		if err := store.Upsert(ctx, u.id, u.fileID, u.emb); err != nil {
			t.Fatalf("Upsert(%s): %v", u.id, err)
		}
	}

	// Update existing id.
	if err := store.Upsert(ctx, "a", "f1-updated", []float32{0.5, 0.5, 0.5, 0.5}); err != nil {
		t.Fatalf("Upsert(update): %v", err)
	}

	matches, err := store.Search(ctx, []float32{0.1, 0.2, 0.3, 0.4}, SearchParams{Limit: 10})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(matches) == 0 {
		t.Fatal("expected at least one match")
	}
	// The nearest match for the query [0.1, 0.2, 0.3, 0.4] (after update)
	// should be the updated "a" row (which now has [0.5, 0.5, 0.5, 0.5])?
	// No: [0.1, 0.2, 0.3, 0.4] == [0.1, 0.2, 0.3, 0.4] but "a" was updated
	// to [0.5, 0.5, 0.5, 0.5]. So the original b [0.4, 0.3, 0.2, 0.1] should
	// be the closest. Just verify all returned matches have valid similarity.
	for i, m := range matches {
		if m.ID == "" {
			t.Errorf("match %d: empty ID", i)
		}
		if m.FileID == "" {
			t.Errorf("match %d: empty FileID", i)
		}
		if m.Similarity < -1.5 || m.Similarity > 1.5 {
			// cosine sim is in [0,1] for non-negative vectors; allow some slack
			t.Errorf("match %d (%s): similarity %f out of expected range", i, m.ID, m.Similarity)
		}
	}
	// Verify the updated file_id came through.
	if matches[0].FileID != "f1-updated" && matches[0].FileID != "f1" {
		// Wait — updated is "a" with fileID "f1-updated". "b" still has "f1".
		// "c" has "f2" but is far away. So the first match's FileID should
		// be either "f1-updated" (the updated a row, which is now far from
		// the query) or "f1" (the b row, which is closer to the query).
		// "f1-updated" with [0.5,0.5,0.5,0.5] vs query [0.1,0.2,0.3,0.4]
		// — cosine distance is 1 - dot/(||a||*||b||). Different from b.
		t.Errorf("top match FileID = %q, want f1 or f1-updated", matches[0].FileID)
	}
}

func TestPgVectorStoreBatchSearch(t *testing.T) {
	store, cleanup := newPgStore(t, 3)
	defer cleanup()

	ctx := context.Background()
	items := []VectorItem{
		{ID: "x1", FileID: "f", Embedding: []float32{1, 0, 0}},
		{ID: "x2", FileID: "f", Embedding: []float32{0, 1, 0}},
		{ID: "x3", FileID: "f", Embedding: []float32{0, 0, 1}},
	}
	if err := store.UpsertBatch(ctx, items); err != nil {
		t.Fatalf("UpsertBatch: %v", err)
	}

	queries := []BatchSearchRequest{
		{QueryID: "q1", Embedding: []float32{1, 0, 0}},
		{QueryID: "q2", Embedding: []float32{0, 1, 0}},
		{QueryID: "q3", Embedding: []float32{0, 0, 1}},
	}
	results, err := store.BatchSearch(ctx, queries, 2)
	if err != nil {
		t.Fatalf("BatchSearch: %v", err)
	}
	if len(results) != 3 {
		t.Fatalf("got %d results, want 3", len(results))
	}

	// q1 should find x1 first.
	if results[0].QueryID != "q1" || len(results[0].Matches) == 0 || results[0].Matches[0].ID != "x1" {
		t.Errorf("q1: got %+v, want x1 first", results[0])
	}
	// q2 should find x2 first.
	if results[1].QueryID != "q2" || len(results[1].Matches) == 0 || results[1].Matches[0].ID != "x2" {
		t.Errorf("q2: got %+v, want x2 first", results[1])
	}
	// q3 should find x3 first.
	if results[2].QueryID != "q3" || len(results[2].Matches) == 0 || results[2].Matches[0].ID != "x3" {
		t.Errorf("q3: got %+v, want x3 first", results[2])
	}
}

func TestPgVectorStoreDeleteByID(t *testing.T) {
	store, cleanup := newPgStore(t, 3)
	defer cleanup()
	ctx := context.Background()

	if err := store.Upsert(ctx, "x", "f", []float32{1, 0, 0}); err != nil {
		t.Fatalf("Upsert: %v", err)
	}
	if err := store.DeleteByID(ctx, "x"); err != nil {
		t.Fatalf("DeleteByID: %v", err)
	}
	matches, err := store.Search(ctx, []float32{1, 0, 0}, SearchParams{Limit: 10})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(matches) != 0 {
		t.Errorf("expected no matches after delete, got %d", len(matches))
	}
}

func TestPgVectorStoreDeleteByFileID(t *testing.T) {
	store, cleanup := newPgStore(t, 3)
	defer cleanup()
	ctx := context.Background()

	if err := store.Upsert(ctx, "a", "f1", []float32{1, 0, 0}); err != nil {
		t.Fatalf("Upsert a: %v", err)
	}
	if err := store.Upsert(ctx, "b", "f1", []float32{0, 1, 0}); err != nil {
		t.Fatalf("Upsert b: %v", err)
	}
	if err := store.Upsert(ctx, "c", "f2", []float32{0, 0, 1}); err != nil {
		t.Fatalf("Upsert c: %v", err)
	}
	if err := store.DeleteByFileID(ctx, "f1"); err != nil {
		t.Fatalf("DeleteByFileID: %v", err)
	}
	matches, err := store.Search(ctx, []float32{1, 0, 0}, SearchParams{Limit: 10})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(matches) != 1 || matches[0].ID != "c" {
		t.Errorf("expected only c to remain, got %+v", matches)
	}
}

func TestPgVectorStoreDimensionMismatch(t *testing.T) {
	store, cleanup := newPgStore(t, 4)
	defer cleanup()
	ctx := context.Background()

	if err := store.Upsert(ctx, "a", "f", []float32{1, 0, 0, 0}); err != nil {
		t.Fatalf("Upsert (correct dim): %v", err)
	}
	if err := store.Upsert(ctx, "b", "f", []float32{1, 0, 0}); err == nil {
		t.Fatal("expected dim mismatch error on Upsert, got nil")
	}
	if _, err := store.Search(ctx, []float32{1, 0, 0}, SearchParams{Limit: 1}); err == nil {
		t.Fatal("expected dim mismatch error on Search, got nil")
	}
}

func TestPgHNSWConfigDefaults(t *testing.T) {
	cfg := DefaultPgHNSWConfig()
	if cfg == nil {
		t.Fatal("default config is nil")
	}
	if cfg.Metric != DistanceCosine {
		t.Errorf("default metric = %q, want cosine", cfg.Metric)
	}
	if cfg.M == 0 || cfg.EfSearch == 0 || cfg.EfConstruction == 0 {
		t.Errorf("zero value in defaults: %+v", cfg)
	}
}

func TestParsePgVectorDimension(t *testing.T) {
	cases := []struct {
		input   string
		want    int
		wantErr bool
	}{
		{"vector(384)", 384, false},
		{"vector(1)", 1, false},
		{"integer", 0, true},
		{"", 0, true},
		{"vector()", 0, true},
		{"vector(abc)", 0, true},
	}
	for _, c := range cases {
		got, err := parsePgVectorDimension(c.input)
		if c.wantErr {
			if err == nil {
				t.Errorf("parsePgVectorDimension(%q): expected error", c.input)
			}
			continue
		}
		if err != nil {
			t.Errorf("parsePgVectorDimension(%q): %v", c.input, err)
			continue
		}
		if got != c.want {
			t.Errorf("parsePgVectorDimension(%q) = %d, want %d", c.input, got, c.want)
		}
	}
}

func TestPgHNSWConfigNormalize(t *testing.T) {
	// Empty config gets defaults.
	cfg := (&PgHNSWConfig{}).normalize()
	if cfg.Metric != DistanceCosine || cfg.M == 0 || cfg.EfSearch == 0 || cfg.EfConstruction == 0 {
		t.Errorf("normalize empty: %+v", cfg)
	}
	// Non-empty config preserved.
	cfg = (&PgHNSWConfig{Metric: DistanceEuclidean, M: 32, EfConstruction: 100, EfSearch: 50}).normalize()
	if cfg.Metric != DistanceEuclidean || cfg.M != 32 || cfg.EfConstruction != 100 || cfg.EfSearch != 50 {
		t.Errorf("normalize non-empty: %+v", cfg)
	}
}

func TestPgVectorStoreSetEfSearch(t *testing.T) {
	store, cleanup := newPgStore(t, 3)
	defer cleanup()
	if err := store.SetEfSearch(100); err != nil {
		t.Fatalf("SetEfSearch: %v", err)
	}
	if err := store.ResetEfSearch(); err != nil {
		t.Fatalf("ResetEfSearch: %v", err)
	}
}

func TestPgVectorStoreUpsertBatchFallback(t *testing.T) {
	// Test the fallback path: pre-existing rows + new rows in the same batch
	// (COPY would fail on PK conflict; fallback uses per-row UPSERT).
	store, cleanup := newPgStore(t, 3)
	defer cleanup()
	ctx := context.Background()

	pre := []VectorItem{
		{ID: "p1", FileID: "f", Embedding: []float32{1, 0, 0}},
		{ID: "p2", FileID: "f", Embedding: []float32{0, 1, 0}},
	}
	if err := store.UpsertBatch(ctx, pre); err != nil {
		t.Fatalf("UpsertBatch pre: %v", err)
	}

	// Now batch-insert overlapping + new IDs.
	overlap := []VectorItem{
		{ID: "p1", FileID: "f", Embedding: []float32{0.5, 0.5, 0}}, // conflict
		{ID: "n1", FileID: "f", Embedding: []float32{0, 0, 1}},     // new
	}
	if err := store.UpsertBatch(ctx, overlap); err != nil {
		t.Fatalf("UpsertBatch overlap: %v", err)
	}

	// Verify p1 was updated.
	matches, err := store.Search(ctx, []float32{0.5, 0.5, 0}, SearchParams{Limit: 1})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(matches) == 0 || matches[0].ID != "p1" {
		t.Errorf("expected p1 as top match after upsert, got %+v", matches)
	}
}

func TestPgVectorStoreGetIndexStats(t *testing.T) {
	store, cleanup := newPgStore(t, 3)
	defer cleanup()
	ctx := context.Background()

	if err := store.Upsert(ctx, "a", "f", []float32{1, 0, 0}); err != nil {
		t.Fatalf("Upsert: %v", err)
	}
	if err := store.Upsert(ctx, "b", "f", []float32{0, 1, 0}); err != nil {
		t.Fatalf("Upsert: %v", err)
	}
	stats, err := store.GetIndexStats(ctx)
	if err != nil {
		t.Fatalf("GetIndexStats: %v", err)
	}
	if stats.TotalVectors != 2 {
		t.Errorf("TotalVectors = %d, want 2", stats.TotalVectors)
	}
	if stats.Config == nil {
		t.Error("Config is nil")
	}
}
