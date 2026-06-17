package fileprocessor

import (
	"context"
	"log/slog"
	"strings"
)

// NewFallbackEmbedder wraps primary with a fallback that is used when the
// primary returns a credit/forbidden error.
func NewFallbackEmbedder(primary, fallback Embedder) *FallbackEmbedder {
	return &FallbackEmbedder{primary: primary, fallback: fallback}
}

// FallbackEmbedder tries the primary [Embedder] first. If the primary returns
// an HTTP 402 (Insufficient Credits) or 403 (Forbidden) error, it retries the
// request using the fallback embedder.
//
// If the fallback also fails, the fallback error is returned (the primary
// error is logged at debug level).
type FallbackEmbedder struct {
	primary  Embedder
	fallback Embedder
}

// Embed delegates to primary; on credit/permission errors, retries with fallback.
func (f *FallbackEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	vecs, err := f.primary.Embed(ctx, texts)
	if err == nil {
		return vecs, nil
	}

	if isCreditError(err) && f.fallback != nil {
		slog.Warn("openai_embedder: primary failed with credit error, falling back to Gemini",
			"error", err)
		return f.fallback.Embed(ctx, texts)
	}

	return nil, err
}

// Dimension returns the primary embedder's dimension.
func (f *FallbackEmbedder) Dimension() int {
	return f.primary.Dimension()
}

// isCreditError reports whether the error is an HTTP 402 or 403.
func isCreditError(err error) bool {
	msg := err.Error()
	return strings.Contains(msg, "HTTP 402") || strings.Contains(msg, "HTTP 403")
}
