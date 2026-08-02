package fileprocessor

import (
	"context"
	"fmt"
	"sort"
	"strings"
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
}

// SemanticSearch embeds the query, fetches top-K+overshoot from the VectorStore,
// hydrates chunk text from the ChunkStore, and applies fileID filters.
//
// The store-side limit is 2*Limit to absorb hydration-time filtering.
func (s *Searcher) SemanticSearch(ctx context.Context, p SearchParamsSearcher) ([]SearchResult, error) {
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

	searchParams := SearchParams{Limit: limit * 2, Metric: p.Metric, FileIDs: p.FileIDs}
	matches, err := s.store.Search(ctx, embeddings[0], searchParams)
	if err != nil {
		return nil, fmt.Errorf("ragcore: vector search: %w", err)
	}
	if s.keyword != nil {
		keywordMatches, keywordErr := s.keyword.KeywordSearch(ctx, p.Query, searchParams)
		if keywordErr != nil {
			return nil, fmt.Errorf("ragcore: keyword search: %w", keywordErr)
		}
		matches = fuseRRF(matches, keywordMatches, limit*2)
	}
	if len(matches) == 0 {
		return []SearchResult{}, nil
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

	results := make([]SearchResult, 0, len(chunks))
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
		meta := map[string]string{}
		if ch.Metadata != "" {
			meta["raw"] = ch.Metadata
		}
		if ch.FileID != "" {
			meta["fileId"] = ch.FileID
		}
		results = append(results, SearchResult{
			ID:         ch.ID,
			Similarity: sim[ch.ID],
			Text:       ch.Text,
			FileID:     ch.FileID,
			FileName:   fileName,
			Type:       ch.Type,
			Index:      int(ch.Index),
			ParentID:   ch.ParentID,
			Metadata:   meta,
		})
		if len(results) >= limit {
			break
		}
	}

	return results, nil
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
func (s *Searcher) SemanticSearchMultipleFiles(ctx context.Context, query string, fileIDs []string, limit int) ([]SearchResult, error) {
	return s.SemanticSearch(ctx, SearchParamsSearcher{Query: query, FileIDs: fileIDs, Limit: limit})
}
