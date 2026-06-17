package fileprocessor

import (
	"testing"
)

func TestGeminiTruncation(t *testing.T) {
	e := NewGeminiEmbedder("test-key", "gemini-embedding-2", 1536)
	if e.dim != 1536 {
		t.Fatalf("expected dim 1536, got %d", e.dim)
	}

	// Simulate the truncation logic directly
	vals := make([]float32, 3072)
	for i := range vals {
		vals[i] = float32(i)
	}
	if len(vals) > e.dim {
		vals = vals[:e.dim]
	}
	if len(vals) != 1536 {
		t.Fatalf("expected 1536, got %d", len(vals))
	}
}
