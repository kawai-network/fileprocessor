package fileprocessor

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"

	"github.com/kawai-network/x/constant"
)

// GeminiEmbedder is an [Embedder] for Google Gemini embedding models
// (gemini-embedding-2, gemini-embedding-001, etc.).
//
// Auth: x-goog-api-key header (Gemini doesn't use Bearer tokens).
// Dimension: gemini-embedding-2 defaults to 768.
type GeminiEmbedder struct {
	apiKey string
	model  string
	dim    int
	client *http.Client
	mu     sync.Mutex
}

// NewGeminiEmbedder creates a Gemini embedder. The model should be the
// short name (e.g. "gemini-embedding-2"). An empty apiKey falls back to
// [constant.GetRandomGeminiApiKey]. Pass 0 for dim to default to 768.
//
// When dim is smaller than the model's native output (e.g. 1536 vs
// gemini-embedding-2's 3072), the first dim values are used.
func NewGeminiEmbedder(apiKey, model string, dim int) *GeminiEmbedder {
	if apiKey == "" {
		if k := constant.GetRandomGeminiApiKey(); k != "" {
			apiKey = k
		}
	}
	if model == "" {
		model = "gemini-embedding-2"
	}
	if dim <= 0 {
		dim = 768
	}
	return &GeminiEmbedder{
		apiKey: apiKey,
		model:  model,
		dim:    dim,
		client: &http.Client{Timeout: DefaultEmbedHTTPTimeout},
	}
}

// embedURL builds the embedContent URL for Gemini.
func (e *GeminiEmbedder) embedURL() string {
	return fmt.Sprintf("https://generativelanguage.googleapis.com/v1beta/models/%s:embedContent", e.model)
}

// Embed sends texts to the Gemini embedding API via per-text embedContent.
// Gemini's embedding API only supports single-text requests, so we issue
// concurrent requests for multiple texts.
func (e *GeminiEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	results := make([][]float32, len(texts))
	errs := make([]error, len(texts))
	var wg sync.WaitGroup

	for i, t := range texts {
		wg.Add(1)
		go func(idx int, text string) {
			defer wg.Done()
			vec, err := e.embedOne(ctx, text)
			if err != nil {
				errs[idx] = err
				return
			}
			results[idx] = vec
		}(i, t)
	}
	wg.Wait()

	for _, err := range errs {
		if err != nil {
			return nil, err
		}
	}
	return results, nil
}

func (e *GeminiEmbedder) embedOne(ctx context.Context, text string) ([]float32, error) {
	payload := map[string]any{
		"model": "models/" + e.model,
		"content": map[string]any{
			"parts": []map[string]any{{"text": text}},
		},
	}
	b, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("gemini_embedder: marshal: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", e.embedURL(), bytes.NewReader(b))
	if err != nil {
		return nil, fmt.Errorf("gemini_embedder: create req: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if e.apiKey != "" {
		req.Header.Set("x-goog-api-key", e.apiKey)
	}

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("gemini_embedder: http: %w", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("gemini_embedder: read body: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("gemini_embedder: HTTP %d: %s", resp.StatusCode, string(raw))
	}

	var api struct {
		Embedding struct {
			Values []float32 `json:"values"`
		} `json:"embedding"`
	}
	if err := json.Unmarshal(raw, &api); err != nil {
		return nil, fmt.Errorf("gemini_embedder: decode: %w", err)
	}
	vals := api.Embedding.Values
	if e.dim > 0 && len(vals) > e.dim {
		vals = vals[:e.dim]
	}
	return vals, nil
}

// Dimension returns the configured dimension (default 768 for gemini-embedding-2).
func (e *GeminiEmbedder) Dimension() int {
	return e.dim
}
