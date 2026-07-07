package fileprocessor

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"mime"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// DefaultVLHTTPTimeout is the HTTP client timeout for vision/chat API calls.
// Vision requests carry a base64 image and can take longer than embeddings, so
// this is larger than [DefaultEmbedHTTPTimeout].
const DefaultVLHTTPTimeout = 120 * time.Second

// visionHTTPClient is the shared HTTP client used by OpenAIChatClient.
var visionHTTPClient = &http.Client{Timeout: DefaultVLHTTPTimeout}

// OpenAIChatClient calls any OpenAI-compatible POST /chat/completions endpoint
// (OpenAI, Plano's kawai-* gateway, OpenRouter, a local vLLM, etc.). It
// implements BOTH [VLProvider] (image → natural-language description) and
// [LanguageModel] (OCR / transcript cleanup), so one instance can be wired to
// both Config.VLProvider and Config.LanguageModel.
//
// Authorization is sent as a Bearer token when apiKey is non-empty. Extra
// headers (added via [OpenAIChatClient.SetHeader]) let a caller satisfy Plano's
// internal-ingress gate — `x-arch-internal-key` plus a per-user
// `x-arch-actor-id` for billing — when routing through :12010.
type OpenAIChatClient struct {
	url     string
	apiKey  string
	model   string
	headers map[string]string
	client  *http.Client
}

// NewOpenAIChatClient builds a client for the given endpoint. baseURL may be a
// gateway base (e.g. "http://localhost:12010/v1") or a full chat-completions
// URL; "/chat/completions" is appended when absent. model is the (aliased)
// model id to request, e.g. "kawai-vision".
func NewOpenAIChatClient(baseURL, apiKey, model string) *OpenAIChatClient {
	u := strings.TrimRight(strings.TrimSpace(baseURL), "/")
	if !strings.HasSuffix(u, "/chat/completions") {
		u += "/chat/completions"
	}
	return &OpenAIChatClient{url: u, apiKey: strings.TrimSpace(apiKey), model: model, client: visionHTTPClient}
}

// SetHeader registers an extra HTTP header sent on every request. Use it for
// Plano's internal-ingress gate (`x-arch-internal-key`) and the billing actor
// (`x-arch-actor-id`).
func (c *OpenAIChatClient) SetHeader(key, value string) {
	if c.headers == nil {
		c.headers = make(map[string]string)
	}
	c.headers[key] = value
}

// chatMessage is one OpenAI chat message. Content is `any` so text messages use
// a plain string while vision messages use the content-part array.
type chatMessage struct {
	Role    string `json:"role"`
	Content any    `json:"content"`
}

// ProcessImage implements [VLProvider]. It base64-encodes the image at
// imagePath and asks the model to describe it, capped at maxTokens.
func (c *OpenAIChatClient) ProcessImage(ctx context.Context, imagePath, prompt string, maxTokens int32) (string, error) {
	data, err := os.ReadFile(imagePath)
	if err != nil {
		return "", fmt.Errorf("openai_vision: read image: %w", err)
	}
	dataURL := "data:" + imageMIME(imagePath) + ";base64," + base64.StdEncoding.EncodeToString(data)
	content := []any{
		map[string]any{"type": "text", "text": prompt},
		map[string]any{"type": "image_url", "image_url": map[string]any{"url": dataURL}},
	}
	return c.complete(ctx, []chatMessage{{Role: "user", Content: content}}, maxTokens)
}

// Generate implements [LanguageModel]. systemPrompt is optional.
func (c *OpenAIChatClient) Generate(ctx context.Context, systemPrompt, userPrompt string) (string, error) {
	msgs := make([]chatMessage, 0, 2)
	if strings.TrimSpace(systemPrompt) != "" {
		msgs = append(msgs, chatMessage{Role: "system", Content: systemPrompt})
	}
	msgs = append(msgs, chatMessage{Role: "user", Content: userPrompt})
	return c.complete(ctx, msgs, 0)
}

// complete POSTs a chat-completions request and returns the first choice's
// message content.
func (c *OpenAIChatClient) complete(ctx context.Context, messages []chatMessage, maxTokens int32) (string, error) {
	payload := map[string]any{"model": c.model, "messages": messages}
	if maxTokens > 0 {
		payload["max_tokens"] = maxTokens
	}
	b, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("openai_vision: marshal: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.url, bytes.NewReader(b))
	if err != nil {
		return "", fmt.Errorf("openai_vision: create req: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}
	for k, v := range c.headers {
		req.Header.Set(k, v)
	}

	resp, err := c.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("openai_vision: http: %w", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("openai_vision: read body: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("openai_vision: %s HTTP %d: %s", c.url, resp.StatusCode, string(raw))
	}

	var api struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(raw, &api); err != nil {
		return "", fmt.Errorf("openai_vision: decode: %w", err)
	}
	if len(api.Choices) == 0 {
		return "", fmt.Errorf("openai_vision: empty choices from %s", c.url)
	}
	return strings.TrimSpace(api.Choices[0].Message.Content), nil
}

// imageMIME resolves an image MIME type from a file path. It prefers a small
// explicit table (the types VL models accept) and falls back to the stdlib
// extension table, then to a safe default.
func imageMIME(path string) string {
	switch normalizeExt(filepath.Ext(path)) {
	case "jpg", "jpeg":
		return "image/jpeg"
	case "png":
		return "image/png"
	case "gif":
		return "image/gif"
	case "webp":
		return "image/webp"
	case "bmp":
		return "image/bmp"
	case "tiff":
		return "image/tiff"
	}
	if t := mime.TypeByExtension(filepath.Ext(path)); t != "" {
		if i := strings.IndexByte(t, ';'); i >= 0 {
			t = t[:i]
		}
		return t
	}
	return "application/octet-stream"
}

var (
	_ VLProvider    = (*OpenAIChatClient)(nil)
	_ LanguageModel = (*OpenAIChatClient)(nil)
)
