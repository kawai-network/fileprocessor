package fileprocessor

import (
	"context"
	"fmt"
	"sort"

	"github.com/cloudwego/eino/components/retriever"
	"github.com/cloudwego/eino/schema"
)

// --- vector sub-retriever ---

// vectorRetriever implements [retriever.Retriever] over a [VectorStore] +
// [ChunkStore]. It embeds the query, searches, filters by threshold,
// hydrates chunks, and returns [schema.Document] values.
type vectorRetriever struct {
	store     VectorStore
	chunks    ChunkStore
	embedder  Embedder
	threshold float64
	metric    DistanceMetric
}

func (v *vectorRetriever) Retrieve(ctx context.Context, query string, opts ...retriever.Option) ([]*schema.Document, error) {
	co := retriever.GetCommonOptions(&retriever.Options{}, opts...)
	io := retriever.GetImplSpecificOptions(&HybridOptions{
		VectorThreshold: v.threshold,
		Metric:          v.metric,
	}, opts...)

	limit := 30
	if co.TopK != nil && *co.TopK > 0 {
		limit = *co.TopK
	}

	vecs, err := v.embedder.Embed(ctx, []string{query})
	if err != nil {
		return nil, fmt.Errorf("vector retriever: embed: %w", err)
	}
	if len(vecs) == 0 || len(vecs[0]) == 0 {
		return nil, fmt.Errorf("vector retriever: empty embedding")
	}

	candidateLimit := limit * candidateOversample
	if candidateLimit < minCandidateLimit {
		candidateLimit = minCandidateLimit
	}

	matches, err := v.store.Search(ctx, vecs[0], SearchParams{
		Limit:   candidateLimit,
		Metric:  io.Metric,
		FileIDs: io.FileIDs,
		TenantID: io.TenantID,
	})
	if err != nil {
		return nil, fmt.Errorf("vector retriever: search: %w", err)
	}

	threshold := io.VectorThreshold
	if threshold <= 0 {
		threshold = defaultVectorThreshold
	}
	matches = filterMatchesByScore(matches, threshold)

	return v.hydrate(ctx, matches, limit)
}

func (v *vectorRetriever) GetType() string        { return "Vector" }
func (v *vectorRetriever) IsCallbacksEnabled() bool { return true }

func (v *vectorRetriever) hydrate(ctx context.Context, matches []VectorMatch, limit int) ([]*schema.Document, error) {
	return hydrateChunks(ctx, v.chunks, matches, limit, "vector retriever")
}

// --- keyword sub-retriever ---

// keywordRetriever implements [retriever.Retriever] over a [KeywordSearcher]
// (SQLite FTS5 BM25) + [ChunkStore]. It searches, filters by threshold,
// hydrates chunks, and returns [schema.Document] values.
type keywordRetriever struct {
	keyword   KeywordSearcher
	chunks    ChunkStore
	threshold float64
}

func (k *keywordRetriever) Retrieve(ctx context.Context, query string, opts ...retriever.Option) ([]*schema.Document, error) {
	co := retriever.GetCommonOptions(&retriever.Options{}, opts...)
	io := retriever.GetImplSpecificOptions(&HybridOptions{
		KeywordThreshold: k.threshold,
	}, opts...)

	limit := 30
	if co.TopK != nil && *co.TopK > 0 {
		limit = *co.TopK
	}

	candidateLimit := limit * candidateOversample
	if candidateLimit < minCandidateLimit {
		candidateLimit = minCandidateLimit
	}

	matches, err := k.keyword.KeywordSearch(ctx, query, SearchParams{
		Limit:   candidateLimit,
		FileIDs: io.FileIDs,
		TenantID: io.TenantID,
	})
	if err != nil {
		return nil, fmt.Errorf("keyword retriever: search: %w", err)
	}

	threshold := io.KeywordThreshold
	if threshold <= 0 {
		threshold = defaultKeywordThreshold
	}
	matches = filterMatchesByScore(matches, threshold)

	return k.hydrate(ctx, matches, limit)
}

func (k *keywordRetriever) GetType() string        { return "Keyword" }
func (k *keywordRetriever) IsCallbacksEnabled() bool { return true }

func (k *keywordRetriever) hydrate(ctx context.Context, matches []VectorMatch, limit int) ([]*schema.Document, error) {
	return hydrateChunks(ctx, k.chunks, matches, limit, "keyword retriever")
}

// hydrateChunks is the shared hydration logic for both vector and keyword
// retrievers. It fetches chunk text from the ChunkStore and builds
// schema.Document values with standard metadata keys.
func hydrateChunks(ctx context.Context, chunks ChunkStore, matches []VectorMatch, limit int, errPrefix string) ([]*schema.Document, error) {
	if len(matches) == 0 {
		return []*schema.Document{}, nil
	}
	ids := make([]string, len(matches))
	sim := make(map[string]float64, len(matches))
	for i, m := range matches {
		ids[i] = m.ID
		sim[m.ID] = m.Similarity
	}

	storeChunks, err := chunks.GetChunksByIDs(ctx, ids)
	if err != nil {
		return nil, fmt.Errorf("%s: hydrate: %w", errPrefix, err)
	}

	results := make([]*schema.Document, 0, len(storeChunks))
	for _, ch := range storeChunks {
		meta := map[string]any{}
		if ch.FileID != "" {
			meta[DocumentMetaFileID] = ch.FileID
		}
		if ch.Type != "" {
			meta[DocumentMetaType] = ch.Type
		}
		meta[DocumentMetaIndex] = int(ch.Index)
		if ch.ParentID != "" {
			meta[DocumentMetaParentID] = ch.ParentID
		}
		if ch.Metadata != "" {
			meta[DocumentMetaRaw] = ch.Metadata
		}
		results = append(results, (&schema.Document{
			ID:       ch.ID,
			Content:  ch.Text,
			MetaData: meta,
		}).WithScore(sim[ch.ID]))
		if len(results) >= limit {
			break
		}
	}
	return results, nil
}

// --- weighted RRF fusion ---

// WeightedRRFFusion returns a fusion function compatible with
// [github.com/cloudwego/eino/flow/retriever/router].Config.FusionFunc.
// Weights are applied per retriever name: docs from a retriever with
// weight=0.7 contribute 0.7/(K+rank) to the fused score.
func WeightedRRFFusion(weights map[string]float64, k float64) func(ctx context.Context, result map[string][]*schema.Document) ([]*schema.Document, error) {
	if k <= 0 {
		k = defaultRRFK
	}
	return func(_ context.Context, result map[string][]*schema.Document) ([]*schema.Document, error) {
		if len(result) == 0 {
			return nil, fmt.Errorf("weighted rrf: no documents")
		}
		if len(result) == 1 {
			for _, docs := range result {
				return docs, nil
			}
		}

		docScore := make(map[string]float64)
		docMap := make(map[string]*schema.Document)

		for name, docs := range result {
			w := weights[name]
			if w == 0 {
				w = 1.0
			}
			for i, doc := range docs {
				docMap[doc.ID] = doc
				if _, ok := docScore[doc.ID]; !ok {
					docScore[doc.ID] = w / (k + float64(i+1))
				} else {
					docScore[doc.ID] += w / (k + float64(i+1))
				}
			}
		}

		type scored struct {
			doc   *schema.Document
			score float64
		}
		list := make([]scored, 0, len(docMap))
		for id, s := range docScore {
			list = append(list, scored{doc: docMap[id], score: s})
		}
		sort.SliceStable(list, func(i, j int) bool {
			return list[i].score > list[j].score
		})

		out := make([]*schema.Document, len(list))
		for i, s := range list {
			out[i] = s.doc.WithScore(s.score)
		}
		return out, nil
	}
}

var _ retriever.Retriever = (*vectorRetriever)(nil)
var _ retriever.Retriever = (*keywordRetriever)(nil)
