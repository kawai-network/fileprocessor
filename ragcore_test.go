package fileprocessor

import (
	"context"
	"path/filepath"
	"testing"
)

func TestHNSWConfigDefaults(t *testing.T) {
	cfg := DefaultHNSWConfig()
	if cfg == nil {
		t.Fatal("default config is nil")
	}
	if cfg.Metric != "l2sq" {
		t.Errorf("default metric = %q, want l2sq", cfg.Metric)
	}
	if cfg.M == 0 || cfg.EfSearch == 0 || cfg.EfConstruction == 0 {
		t.Errorf("zero value in defaults: %+v", cfg)
	}
}

func TestNormalizeHNSWConfigFillsDefaults(t *testing.T) {
	cfg := &HNSWConfig{Metric: ""}
	out := normalizeHNSWConfig(cfg)
	if out.Metric == "" {
		t.Error("metric not filled with default")
	}
	if out.M == 0 {
		t.Error("M not filled with default")
	}
}

func TestNormalizeHNSWConfigNilReturnsDefaults(t *testing.T) {
	out := normalizeHNSWConfig(nil)
	if out == nil {
		t.Fatal("nil config returned nil")
	}
	if out.Metric == "" {
		t.Error("nil config did not get default metric")
	}
}

func TestParseEmbeddingDimension(t *testing.T) {
	cases := []struct {
		input   string
		want    int
		wantErr bool
	}{
		{"FLOAT[384]", 384, false},
		{"FLOAT[1]", 1, false},
		{"float[768]", 768, false},
		{"VARCHAR", 0, true},
		{"", 0, true},
	}
	for _, c := range cases {
		got, err := parseEmbeddingDimension(c.input)
		if c.wantErr {
			if err == nil {
				t.Errorf("parseEmbeddingDimension(%q): expected error", c.input)
			}
			continue
		}
		if err != nil {
			t.Errorf("parseEmbeddingDimension(%q): %v", c.input, err)
		}
		if got != c.want {
			t.Errorf("parseEmbeddingDimension(%q) = %d, want %d", c.input, got, c.want)
		}
	}
}

func TestParseHNSWConfigFromCreateIndexSQL(t *testing.T) {
	sql := `CREATE INDEX vec_idx ON vectors USING HNSW (embedding) WITH (metric = 'cosine', ef_construction = 200, ef_search = 100, M = 24)`
	cfg, err := parseHNSWConfigFromCreateIndexSQL(sql)
	if err != nil {
		t.Fatalf("parseHNSWConfigFromCreateIndexSQL: %v", err)
	}
	if cfg.Metric != "cosine" {
		t.Errorf("metric = %q, want cosine", cfg.Metric)
	}
	if cfg.EfConstruction != 200 {
		t.Errorf("ef_construction = %d, want 200", cfg.EfConstruction)
	}
	if cfg.EfSearch != 100 {
		t.Errorf("ef_search = %d, want 100", cfg.EfSearch)
	}
	if cfg.M != 24 {
		t.Errorf("M = %d, want 24", cfg.M)
	}
}

func TestDuckDBStoreInMemoryRoundTrip(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "vectors.db")

	store, err := NewDuckDBStore(dbPath, 4)
	if err != nil {
		t.Fatalf("NewDuckDBStore: %v", err)
	}
	defer func() { _ = store.Close() }()

	ctx := context.Background()
	if err := store.Upsert(ctx, "a", "f1", []float32{0.1, 0.2, 0.3, 0.4}); err != nil {
		t.Fatalf("Upsert: %v", err)
	}
	if err := store.Upsert(ctx, "b", "f1", []float32{0.4, 0.3, 0.2, 0.1}); err != nil {
		t.Fatalf("Upsert: %v", err)
	}

	matches, err := store.Search(ctx, []float32{0.1, 0.2, 0.3, 0.4}, SearchParams{Limit: 2})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(matches) == 0 {
		t.Fatal("expected at least one match")
	}
	if matches[0].ID != "a" {
		t.Errorf("nearest match = %q, want a", matches[0].ID)
	}
	if matches[0].Similarity <= 0 || matches[0].Similarity > 1 {
		t.Errorf("similarity = %f, want (0,1]", matches[0].Similarity)
	}
}

func TestDuckDBStoreBatchSearch(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "vectors.db")

	store, err := NewDuckDBStore(dbPath, 3)
	if err != nil {
		t.Fatalf("NewDuckDBStore: %v", err)
	}
	defer func() { _ = store.Close() }()

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
	}
	results, err := store.BatchSearch(ctx, queries, 2)
	if err != nil {
		t.Fatalf("BatchSearch: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("got %d results, want 2", len(results))
	}
	for _, r := range results {
		if len(r.Matches) == 0 {
			t.Errorf("query %s: no matches", r.QueryID)
		}
	}
}

func TestEmbeddingCacheRoundTrip(t *testing.T) {
	inner := &fakeEmbedder{
		vecs: map[string][]float32{
			"hello":   {1, 0, 0},
			"world":   {0, 1, 0},
			"unique!": {0, 0, 1},
		},
		dim: 3,
	}
	cache := NewEmbeddingCache(inner, &EmbeddingCacheConfig{MaxSize: 8, TTL: 0, AvgEmbedMs: 1})
	defer cache.Clear()

	ctx := context.Background()

	out1, err := cache.Embed(ctx, []string{"hello", "world"})
	if err != nil {
		t.Fatalf("first embed: %v", err)
	}
	if inner.calls != 2 {
		t.Errorf("expected 2 calls on first miss, got %d", inner.calls)
	}

	out2, err := cache.Embed(ctx, []string{"hello", "world", "unique!"})
	if err != nil {
		t.Fatalf("second embed: %v", err)
	}
	if inner.calls != 3 {
		t.Errorf("expected 3 calls after one new miss, got %d", inner.calls)
	}

	if len(out1) != 2 || len(out2) != 3 {
		t.Errorf("unexpected output lengths: %d %d", len(out1), len(out2))
	}

	stats := cache.Stats()
	if stats.Hits != 2 || stats.Misses != 3 {
		t.Errorf("stats: hits=%d misses=%d, want hits=2 misses=3", stats.Hits, stats.Misses)
	}
}

type fakeEmbedder struct {
	vecs  map[string][]float32
	calls int
	dim   int
}

func (f *fakeEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	f.calls += len(texts)
	out := make([][]float32, len(texts))
	for i, t := range texts {
		out[i] = f.vecs[t]
	}
	return out, nil
}

func (f *fakeEmbedder) Dimension() int { return f.dim }
