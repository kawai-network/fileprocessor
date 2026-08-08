package fileprocessor

import (
	"context"
	"fmt"
	"log/slog"
	"strings"

	"github.com/cloudwego/eino/callbacks"
	"github.com/cloudwego/eino/components"
	"github.com/cloudwego/eino/components/retriever"
	"github.com/cloudwego/eino/schema"
)

const hybridTyp = "KawaiHybrid"

// RetrieverConfig configures a hybrid vector+keyword retriever that satisfies
// the eino [retriever.Retriever] interface.
type RetrieverConfig struct {
	// Store is the vector similarity backend (pgvector, etc.).
	Store VectorStore
	// Chunks holds chunk text and file linkage. Required for hydration
	// and parent context expansion.
	Chunks ChunkStore
	// Embedder turns queries into vectors. Used as the default embedder;
	// callers may override per-call via [retriever.WithEmbedding].
	Embedder Embedder
	// Keyword provides BM25 lexical ranking. When nil, the retriever
	// falls back to vector-only search.
	Keyword KeywordSearcher

	// VectorThreshold is the minimum similarity score for vector candidates
	// before RRF fusion. Default 0.15.
	VectorThreshold float64
	// KeywordThreshold is the minimum BM25 score for keyword candidates
	// before RRF fusion. Default 0.3.
	KeywordThreshold float64
	// DefaultMetric is the distance metric used for vector search.
	DefaultMetric DistanceMetric
	// ExpandParent prepends parent chunk context to child results when set.
	ExpandParent bool
}

// Retriever wraps eino sub-retrievers (vector + keyword) behind a
// [retriever.Retriever] interface using weighted RRF fusion, callbacks,
// and the standard option pattern.
type Retriever struct {
	router       retriever.Retriever
	expandParent bool
	chunks       ChunkStore
}

// NewRetriever constructs a [Retriever] from config. When Keyword is
// provided, vector and keyword sub-retrievers run in parallel and results
// are fused with weighted RRF (vector 0.7, keyword 0.3, K=60).
func NewRetriever(_ context.Context, cfg *RetrieverConfig) (*Retriever, error) {
	if cfg == nil {
		return nil, fmt.Errorf("kawai retriever: config is required")
	}
	if cfg.Store == nil {
		return nil, fmt.Errorf("kawai retriever: store is required")
	}
	if cfg.Chunks == nil {
		return nil, fmt.Errorf("kawai retriever: chunk store is required")
	}
	if cfg.Embedder == nil {
		return nil, fmt.Errorf("kawai retriever: embedder is required")
	}

	vt := cfg.VectorThreshold
	if vt <= 0 {
		vt = defaultVectorThreshold
	}
	kt := cfg.KeywordThreshold
	if kt <= 0 {
		kt = defaultKeywordThreshold
	}

	var kw KeywordSearcher
	if cfg.Keyword != nil {
		kw = cfg.Keyword
	} else if candidate, ok := cfg.Store.(KeywordSearcher); ok {
		kw = candidate
	}

	subRetrievers := map[string]retriever.Retriever{
		"vector": &vectorRetriever{
			store:     cfg.Store,
			chunks:    cfg.Chunks,
			embedder:  cfg.Embedder,
			threshold: vt,
			metric:    cfg.DefaultMetric,
		},
	}

	fusionWeights := map[string]float64{
		"vector": defaultVectorWeight,
	}

	if kw != nil {
		subRetrievers["keyword"] = &keywordRetriever{
			keyword:   kw,
			chunks:    cfg.Chunks,
			threshold: kt,
		}
		fusionWeights["keyword"] = defaultKeywordWeight
	}

	routerRetriever, err := newRouterRetriever(subRetrievers, fusionWeights)
	if err != nil {
		return nil, fmt.Errorf("kawai retriever: create router: %w", err)
	}

	return &Retriever{
		router:       routerRetriever,
		expandParent: cfg.ExpandParent,
		chunks:       cfg.Chunks,
	}, nil
}

// newRouterRetriever creates a simple router retriever that fans out to all
// sub-retrievers and fuses with weighted RRF.
func newRouterRetriever(subs map[string]retriever.Retriever, weights map[string]float64) (retriever.Retriever, error) {
	if len(subs) == 0 {
		return nil, fmt.Errorf("no sub-retrievers")
	}
	fusionFn := WeightedRRFFusion(weights, defaultRRFK)
	return &simpleRouterRetriever{
		retrievers: subs,
		fusionFunc: fusionFn,
	}, nil
}

// simpleRouterRetriever fans out Retrieve to all sub-retrievers concurrently
// and fuses results with a configurable fusion function.
type simpleRouterRetriever struct {
	retrievers map[string]retriever.Retriever
	fusionFunc func(ctx context.Context, result map[string][]*schema.Document) ([]*schema.Document, error)
}

func (r *simpleRouterRetriever) Retrieve(ctx context.Context, query string, opts ...retriever.Option) ([]*schema.Document, error) {
	type result struct {
		name string
		docs []*schema.Document
		err  error
	}

	ch := make(chan result, len(r.retrievers))
	for name, sub := range r.retrievers {
		go func(n string, s retriever.Retriever) {
			docs, err := s.Retrieve(ctx, query, opts...)
			ch <- result{name: n, docs: docs, err: err}
		}(name, sub)
	}

	merged := make(map[string][]*schema.Document, len(r.retrievers))
	for range r.retrievers {
		r := <-ch
		if r.err != nil {
			return nil, fmt.Errorf("retriever %s: %w", r.name, r.err)
		}
		merged[r.name] = r.docs
	}

	return r.fusionFunc(ctx, merged)
}

func (r *simpleRouterRetriever) GetType() string        { return "SimpleRouter" }
func (r *simpleRouterRetriever) IsCallbacksEnabled() bool { return true }

// Retrieve performs hybrid vector+BM25 retrieval, satisfying the eino
// [retriever.Retriever] interface.
//
// Common options ([retriever.TopK], [retriever.ScoreThreshold],
// [retriever.Embedding]) are merged via [retriever.GetCommonOptions].
// Implementation-specific options ([WithFileIDs], [WithVectorThreshold],
// [WithKeywordThreshold], [WithMetric], [WithExpandParent]) are extracted
// via [retriever.GetImplSpecificOptions].
func (ret *Retriever) Retrieve(ctx context.Context, query string, opts ...retriever.Option) (docs []*schema.Document, err error) {
	if err = validateQuery(query); err != nil {
		return nil, err
	}

	co := retriever.GetCommonOptions(&retriever.Options{}, opts...)

	limit := 30
	if co.TopK != nil && *co.TopK > 0 {
		limit = *co.TopK
	}

	io := retriever.GetImplSpecificOptions(&HybridOptions{}, opts...)

	ctx = callbacks.EnsureRunInfo(ctx, ret.GetType(), components.ComponentOfRetriever)
	ctx = callbacks.OnStart(ctx, &retriever.CallbackInput{
		Query:          query,
		TopK:           limit,
		ScoreThreshold: co.ScoreThreshold,
	})
	defer func() {
		if err != nil {
			callbacks.OnError(ctx, err)
		}
	}()

	docs, err = ret.router.Retrieve(ctx, query, opts...)
	if err != nil {
		return nil, fmt.Errorf("kawai retriever: %w", err)
	}

	if io.ExpandParent || ret.expandParent {
		docs = ret.expandParentContext(ctx, docs)
	}

	callbacks.OnEnd(ctx, &retriever.CallbackOutput{Docs: docs})

	return docs, nil
}

// GetType returns the component type name for callback routing.
func (ret *Retriever) GetType() string { return hybridTyp }

// IsCallbacksEnabled returns true so callbacks fire.
func (ret *Retriever) IsCallbacksEnabled() bool { return true }

// --- impl-specific options ---

// HybridOptions carries kawai-specific parameters through the eino option
// chain. Callers set these via [WithFileIDs], [WithVectorThreshold], etc.
type HybridOptions struct {
	FileIDs          []string
	VectorThreshold  float64
	KeywordThreshold float64
	Metric           DistanceMetric
	ExpandParent     bool
	TenantID         string
}

// WithFileIDs restricts results to chunks belonging to the given files.
func WithFileIDs(ids ...string) retriever.Option {
	return retriever.WrapImplSpecificOptFn(func(o *HybridOptions) {
		o.FileIDs = ids
	})
}

// WithVectorThreshold overrides the default vector similarity threshold (0.15).
func WithVectorThreshold(t float64) retriever.Option {
	return retriever.WrapImplSpecificOptFn(func(o *HybridOptions) {
		o.VectorThreshold = t
	})
}

// WithKeywordThreshold overrides the default BM25 keyword threshold (0.3).
func WithKeywordThreshold(t float64) retriever.Option {
	return retriever.WrapImplSpecificOptFn(func(o *HybridOptions) {
		o.KeywordThreshold = t
	})
}

// WithMetric overrides the distance metric for vector search.
func WithMetric(m DistanceMetric) retriever.Option {
	return retriever.WrapImplSpecificOptFn(func(o *HybridOptions) {
		o.Metric = m
	})
}

// WithExpandParent overrides the ExpandParent toggle per call.
func WithExpandParent(v bool) retriever.Option {
	return retriever.WrapImplSpecificOptFn(func(o *HybridOptions) {
		o.ExpandParent = v
	})
}

// WithTenantID scopes retrieval to a tenant via SET LOCAL app.tenant_id on
// stores that enforce it (PublicEmbeddingsStore + RLS). Empty keeps the legacy
// in-memory allowlist path (architecture review R3).
func WithTenantID(tenantID string) retriever.Option {
	return retriever.WrapImplSpecificOptFn(func(o *HybridOptions) {
		o.TenantID = tenantID
	})
}

// --- parent context expansion ---

// ExpandParentContext fetches parent chunks for results that have a
// DocumentMetaParentID and prepends the parent text as context. It is the
// exported form of the expansion step so callers can run it as a
// post-retrieval transformation (e.g. after deduplication) instead of inside
// Retrieve. When chunks is nil or no result carries a parent id, results are
// returned unchanged.
func ExpandParentContext(ctx context.Context, chunks ChunkStore, results []*schema.Document) []*schema.Document {
	if chunks == nil || len(results) == 0 {
		return results
	}

	parentIDs := make(map[string]bool)
	for _, doc := range results {
		if pid := DocumentStringMetadata(doc, DocumentMetaParentID); pid != "" {
			parentIDs[pid] = true
		}
	}
	if len(parentIDs) == 0 {
		return results
	}

	ids := make([]string, 0, len(parentIDs))
	for id := range parentIDs {
		ids = append(ids, id)
	}

	parents, err := chunks.GetChunksByIDs(ctx, ids)
	if err != nil {
		slog.Warn("kawai retriever: expand parent context failed", "error", err)
		return results
	}

	parentText := make(map[string]string, len(parents))
	for _, p := range parents {
		parentText[p.ID] = p.Text
	}

	expanded := make([]*schema.Document, 0, len(results))
	for _, doc := range results {
		if pid := DocumentStringMetadata(doc, DocumentMetaParentID); pid != "" {
			if pt, ok := parentText[pid]; ok && pt != "" {
				doc.Content = "[Context from section]\n" + pt + "\n\n[Excerpt]\n" + doc.Content
			}
		}
		expanded = append(expanded, doc)
	}
	return expanded
}

// expandParentContext is the in-Retrieve path; it delegates to the exported
// [ExpandParentContext] using the Retriever's own chunk store.
func (ret *Retriever) expandParentContext(ctx context.Context, results []*schema.Document) []*schema.Document {
	if ret.chunks == nil {
		return results
	}
	return ExpandParentContext(ctx, ret.chunks, results)
}

var _ retriever.Retriever = (*Retriever)(nil)
var _ components.Typer = (*Retriever)(nil)

func validateQuery(query string) error {
	if strings.TrimSpace(query) == "" {
		return fmt.Errorf("kawai retriever: query cannot be empty")
	}
	return nil
}
