package fileprocessor

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/kawai-network/x/constant"
)

// DefaultEmbeddingDim is the universal embedding dimension. Both OpenRouter
// (text-embedding-3-small, max 1536) and Gemini fallback (gemini-embedding-2,
// truncated to 1536) are pinned to this value so vectors are interchangeable.
const DefaultEmbeddingDim = 1536

// DefaultEmbedHTTPTimeout is the HTTP client timeout for embedding API calls.
const DefaultEmbedHTTPTimeout = 30 * time.Second

// embedHTTPClient is the shared HTTP client used by OpenAIEmbedder.
// It has a 30-second timeout to prevent goroutines from hanging indefinitely.
var embedHTTPClient = &http.Client{Timeout: DefaultEmbedHTTPTimeout}

// OpenAIEmbedder is an [Embedder] that calls any OpenAI-compatible embeddings
// API (OpenAI, OpenRouter, Azure OpenAI, etc.).
//
// It uses the standard POST /v1/embeddings request shape:
//
//	{"input": [...], "model": "..."}
//
// Authorization is passed as a Bearer token. An empty apiKey means no
// Authorization header is sent (for locally-hosted models without auth).
type OpenAIEmbedder struct {
	url      string
	apiKey   string
	model    string
	dim      int
}

// NewOpenAIEmbedder creates an embedder that calls the given OpenAI-compatible
// endpoint. The model parameter is the embedding model name (e.g.
// "text-embedding-3-small"). Pass 0 for dim to let Dimension() return 0
// (the caller should then configure the vector store dimension externally).
//
// When apiKey is empty, the embedder falls back to
// [constant.GetRandomOpenRouterApiKey] as a first default. If the primary
// key returns HTTP 402 (insufficient credits), it automatically retries
// with a Gemini embedder using [constant.GetRandomGeminiApiKey].
func NewOpenAIEmbedder(url, apiKey, model string, dim int) Embedder {
	if apiKey == "" {
		if k := constant.GetRandomOpenRouterApiKey(); k != "" {
			apiKey = k
		}
	}
	if dim <= 0 {
		dim = DefaultEmbeddingDim
	}
	primary := &OpenAIEmbedder{
		url:    url,
		apiKey: apiKey,
		model:  model,
		dim:    dim,
	}
	fallback := NewGeminiEmbedder("", "gemini-embedding-2", dim)
	return NewFallbackEmbedder(primary, fallback)
}

// Embed sends texts to the embedding API and returns float32 vectors.
// It returns an error if the API is unreachable, returns a non-200 status,
// or returns malformed JSON.
func (e *OpenAIEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	body := map[string]any{"input": texts, "model": e.model}
	if e.dim > 0 {
		body["dimensions"] = e.dim
	}
	b, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("openai_embedder: marshal: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", e.url, bytes.NewReader(b))
	if err != nil {
		return nil, fmt.Errorf("openai_embedder: create req: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if e.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+e.apiKey)
	}

	resp, err := embedHTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("openai_embedder: http: %w", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("openai_embedder: read body: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("openai_embedder: %s HTTP %d: %s", e.url, resp.StatusCode, string(raw))
	}

	var api struct {
		Data []struct {
			Embedding []float64 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.Unmarshal(raw, &api); err != nil {
		return nil, fmt.Errorf("openai_embedder: decode: %w", err)
	}

	out := make([][]float32, len(api.Data))
	for i, d := range api.Data {
		v := make([]float32, len(d.Embedding))
		for j, f := range d.Embedding {
			v[j] = float32(f)
		}
		out[i] = v
	}
	return out, nil
}

// Dimension returns the configured embedding dimension. Returns 0
// if the caller didn't provide one at construction (e.g. when using
// a variable-dimension model like v3-small with dimension 512/1024).
func (e *OpenAIEmbedder) Dimension() int {
	return e.dim
}
