package fileprocessor

import (
	"context"
	"fmt"
	"strings"
)

// Searcher performs semantic search: embed a query, look up similar vectors in
// the VectorStore, hydrate chunk text from the ChunkStore, and apply optional
// fileID filters.
type Searcher struct {
	store    VectorStore
	chunks   ChunkStore
	embedder Embedder
}

// NewSearcher wires a Searcher.
func NewSearcher(store VectorStore, chunks ChunkStore, embedder Embedder) *Searcher {
	return &Searcher{store: store, chunks: chunks, embedder: embedder}
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

	matches, err := s.store.Search(ctx, embeddings[0], SearchParams{
		Limit:  limit * 2,
		Metric: p.Metric,
	})
	if err != nil {
		return nil, fmt.Errorf("ragcore: vector search: %w", err)
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
			Metadata:   meta,
		})
		if len(results) >= limit {
			break
		}
	}

	return results, nil
}

// SemanticSearchMultipleFiles is a convenience wrapper for callers that prefer
// separate arguments over a search-params struct.
func (s *Searcher) SemanticSearchMultipleFiles(ctx context.Context, query string, fileIDs []string, limit int) ([]SearchResult, error) {
	return s.SemanticSearch(ctx, SearchParamsSearcher{Query: query, FileIDs: fileIDs, Limit: limit})
}
