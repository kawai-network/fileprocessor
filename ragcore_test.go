package fileprocessor

import (
	"context"
	"testing"
)

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

func TestEmbeddingCacheEmptyInput(t *testing.T) {
	cache := NewEmbeddingCache(&fakeEmbedder{dim: 2}, nil)
	defer cache.Clear()
	out, err := cache.Embed(context.Background(), nil)
	if err != nil {
		t.Fatalf("empty embed: %v", err)
	}
	if out != nil {
		t.Errorf("expected nil output for empty input, got %v", out)
	}
}

func TestEmbeddingCacheClearResetsStats(t *testing.T) {
	inner := &fakeEmbedder{vecs: map[string][]float32{"a": {1}}, dim: 1}
	cache := NewEmbeddingCache(inner, &EmbeddingCacheConfig{MaxSize: 8})
	defer cache.Clear()

	_, _ = cache.Embed(context.Background(), []string{"a"})
	_, _ = cache.Embed(context.Background(), []string{"a"}) // hit
	if cache.Stats().Hits == 0 {
		t.Fatal("expected at least one hit before clear")
	}

	cache.Clear()
	if cache.Stats().Hits != 0 || cache.Stats().Misses != 0 {
		t.Errorf("after Clear: hits=%d misses=%d, want 0/0", cache.Stats().Hits, cache.Stats().Misses)
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
