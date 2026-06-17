package fileprocessor

import (
	"context"
	"os"
	"testing"
	"time"
)

func TestOpenAIEmbedder_Integration(t *testing.T) {
	url := os.Getenv("FILEPROC_EMBEDDING_URL")
	if url == "" {
		url = "https://openrouter.ai/api/v1/embeddings"
	}
	model := os.Getenv("FILEPROC_EMBEDDING_MODEL")
	if model == "" {
		model = "text-embedding-3-small"
	}
	key := os.Getenv("FILEPROC_EMBEDDING_KEY")

	e := NewOpenAIEmbedder(url, key, model, 0)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	vecs, err := e.Embed(ctx, []string{"hello world", "test embedding"})
	if err != nil {
		t.Fatalf("Embed() failed: %v", err)
	}

	if len(vecs) != 2 {
		t.Fatalf("expected 2 embeddings, got %d", len(vecs))
	}
	if len(vecs[0]) == 0 {
		t.Fatal("first embedding is empty")
	}
	if len(vecs[0]) != len(vecs[1]) {
		t.Fatalf("dimension mismatch: %d vs %d", len(vecs[0]), len(vecs[1]))
	}

	t.Logf("dimension: %d, non-zero: %v", len(vecs[0]), vecs[0][:3])
}
