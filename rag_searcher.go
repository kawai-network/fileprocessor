package fileprocessor

import (
	"context"
	"fmt"
	"sort"
	"strings"

	"github.com/cloudwego/eino/schema"
)

const (
	defaultVectorWeight  = 0.7
	defaultKeywordWeight = 0.3
	defaultRRFK          = 60.0
)

// Searcher performs semantic search: embed a query, look up similar vectors in
// the VectorStore, hydrate chunk text from the ChunkStore, and apply optional
// fileID filters.
type Searcher struct {
	store    VectorStore
	chunks   ChunkStore
	embedder Embedder
	keyword  KeywordSearcher
}

const (
	defaultVectorThreshold  = 0.15
	defaultKeywordThreshold = 0.3
	minCandidateLimit       = 50
	candidateOversample     = 5
)

// NewSearcher wires a Searcher.
func NewSearcher(store VectorStore, chunks ChunkStore, embedder Embedder) *Searcher {
	var keyword KeywordSearcher
	if candidate, ok := store.(KeywordSearcher); ok {
		keyword = candidate
	}
	return &Searcher{store: store, chunks: chunks, embedder: embedder, keyword: keyword}
}

// NewSearcherWithKeywordSearcher wires a separate lexical index, such as a
// SQLite FTS5 BM25 projection, alongside the vector store.
func NewSearcherWithKeywordSearcher(store VectorStore, chunks ChunkStore, embedder Embedder, keyword KeywordSearcher) *Searcher {
	return &Searcher{store: store, chunks: chunks, embedder: embedder, keyword: keyword}
}

// SearchParamsSearcher controls a single Search call.
type SearchParamsSearcher struct {
	Query   string
	FileIDs []string
	Limit   int
	// Metric overrides the store's default metric when supported.
	Metric DistanceMetric
	// VectorThreshold filters vector candidates before hybrid fusion. A value
	// <= 0 uses the default of 0.15.
	VectorThreshold float64
	// KeywordThreshold filters keyword candidates before hybrid fusion. A value
	// <= 0 uses the default of 0.3.
	KeywordThreshold float64
}

// SemanticSearch embeds the query, fetches top-K+overshoot from the VectorStore,
// hydrates chunk text from the ChunkStore, and applies fileID filters.
//
// The store-side limit is max(5*Limit, 50) to absorb threshold and
// hydration-time filtering before the final limit is applied.
func (s *Searcher) SemanticSearch(ctx context.Context, p SearchParamsSearcher) ([]*schema.Document, error) {
	if strings.TrimSpace(p.Query) == "" {
		return nil, fmt.Errorf("ragcore: search query cannot be empty")
	}
	limit := p.Limit
	if limit <= 0 {
		limit = 30
	}

	embeddings, err := s.embedder.Embed(ctx, []string{p.Query})
	if err != nil {
		return nil, fmt.Errorf("ragcore: embed query: %w", err)
	}
	if len(embeddings) == 0 || len(embeddings[0]) == 0 {
		return nil, fmt.Errorf("ragcore: empty embedding")
	}

	if s.store == nil {
		return nil, fmt.Errorf("ragcore: vector store not configured")
	}

	candidateLimit := limit * candidateOversample
	if candidateLimit < minCandidateLimit {
		candidateLimit = minCandidateLimit
	}
	searchParams := SearchParams{Limit: candidateLimit, Metric: p.Metric, FileIDs: p.FileIDs}
	matches, err := s.store.Search(ctx, embeddings[0], searchParams)
	if err != nil {
		return nil, fmt.Errorf("ragcore: vector search: %w", err)
	}
	vectorThreshold := p.VectorThreshold
	if vectorThreshold <= 0 {
		vectorThreshold = defaultVectorThreshold
	}
	matches = filterMatchesByScore(matches, vectorThreshold)

	if s.keyword != nil {
		keywordMatches, keywordErr := s.keyword.KeywordSearch(ctx, p.Query, searchParams)
		if keywordErr != nil {
			return nil, fmt.Errorf("ragcore: keyword search: %w", keywordErr)
		}
		keywordThreshold := p.KeywordThreshold
		if keywordThreshold <= 0 {
			keywordThreshold = defaultKeywordThreshold
		}
		keywordMatches = filterMatchesByScore(keywordMatches, keywordThreshold)
		matches = fuseRRF(matches, keywordMatches, limit*2)
	}
	if len(matches) == 0 {
		return []*schema.Document{}, nil
	}

	ids := make([]string, len(matches))
	sim := make(map[string]float64, len(matches))
	for i, m := range matches {
		ids[i] = m.ID
		sim[m.ID] = m.Similarity
	}

	chunks, err := s.chunks.GetChunksByIDs(ctx, ids)
	if err != nil {
		return nil, fmt.Errorf("ragcore: hydrate chunks: %w", err)
	}

	want := func(fileID string) bool {
		if len(p.FileIDs) == 0 {
			return true
		}
		for _, f := range p.FileIDs {
			if f == fileID {
				return true
			}
		}
		return false
	}

	results := make([]*schema.Document, 0, len(chunks))
	for _, ch := range chunks {
		if !want(ch.FileID) {
			continue
		}
		fileName := ""
		if ch.FileID != "" {
			if f, err := s.chunks.GetFile(ctx, ch.FileID); err == nil {
				fileName = f.Name
			}
		}
		meta := map[string]any{}
		if ch.Metadata != "" {
			meta[DocumentMetaRaw] = ch.Metadata
		}
		if ch.FileID != "" {
			meta[DocumentMetaFileID] = ch.FileID
		}
		if fileName != "" {
			meta[DocumentMetaFileName] = fileName
		}
		if ch.Type != "" {
			meta[DocumentMetaType] = ch.Type
		}
		meta[DocumentMetaIndex] = int(ch.Index)
		if ch.ParentID != "" {
			meta[DocumentMetaParentID] = ch.ParentID
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

func filterMatchesByScore(matches []VectorMatch, threshold float64) []VectorMatch {
	filtered := make([]VectorMatch, 0, len(matches))
	for _, match := range matches {
		if match.Similarity >= threshold {
			filtered = append(filtered, match)
		}
	}
	return filtered
}

// fuseRRF merges vector and keyword rankings. RRF deliberately uses rank
// rather than raw scores because cosine similarity and ts_rank are not on the
// same scale.
func fuseRRF(vectorMatches, keywordMatches []VectorMatch, limit int) []VectorMatch {
	type fused struct {
		match VectorMatch
		score float64
	}
	byID := make(map[string]*fused, len(vectorMatches)+len(keywordMatches))
	for rank, match := range vectorMatches {
		entry := byID[match.ID]
		if entry == nil {
			entry = &fused{match: match}
			byID[match.ID] = entry
		}
		entry.score += defaultVectorWeight / (defaultRRFK + float64(rank+1))
	}
	for rank, match := range keywordMatches {
		entry := byID[match.ID]
		if entry == nil {
			entry = &fused{match: match}
			byID[match.ID] = entry
		}
		if entry.match.FileID == "" {
			entry.match.FileID = match.FileID
		}
		entry.score += defaultKeywordWeight / (defaultRRFK + float64(rank+1))
	}
	results := make([]fused, 0, len(byID))
	for _, entry := range byID {
		entry.match.Similarity = entry.score
		results = append(results, *entry)
	}
	sort.SliceStable(results, func(i, j int) bool {
		return results[i].score > results[j].score
	})
	if limit > len(results) {
		limit = len(results)
	}
	out := make([]VectorMatch, limit)
	for i := range out {
		out[i] = results[i].match
	}
	return out
}

// SemanticSearchMultipleFiles is a convenience wrapper for callers that prefer
// separate arguments over a search-params struct.
func (s *Searcher) SemanticSearchMultipleFiles(ctx context.Context, query string, fileIDs []string, limit int) ([]*schema.Document, error) {
	return s.SemanticSearch(ctx, SearchParamsSearcher{Query: query, FileIDs: fileIDs, Limit: limit})
}
