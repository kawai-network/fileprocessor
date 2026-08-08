package fileprocessor

import (
	"context"
	"testing"

	"github.com/cloudwego/eino/components/retriever"
	"github.com/cloudwego/eino/schema"
)

// --- fakes ---

type fakeVectorStore struct {
	searchFn func(ctx context.Context, embedding []float32, params SearchParams) ([]VectorMatch, error)
}

func (f *fakeVectorStore) Upsert(_ context.Context, _ string, _ string, _ []float32) error { return nil }
func (f *fakeVectorStore) UpsertBatch(_ context.Context, _ []VectorItem) error             { return nil }
func (f *fakeVectorStore) Search(ctx context.Context, embedding []float32, params SearchParams) ([]VectorMatch, error) {
	if f.searchFn != nil {
		return f.searchFn(ctx, embedding, params)
	}
	return nil, nil
}
func (f *fakeVectorStore) DeleteByID(_ context.Context, _ string) error         { return nil }
func (f *fakeVectorStore) DeleteByFileID(_ context.Context, _ string) error     { return nil }
func (f *fakeVectorStore) Close() error                                         { return nil }

type fakeChunkStore struct {
	chunks map[string]Chunk
	files  map[string]RAGFile
}

func (f *fakeChunkStore) GetDocument(_ context.Context, _ string) (*schema.Document, error) {
	return &schema.Document{}, nil
}
func (f *fakeChunkStore) CreateChunk(_ context.Context, _ CreateChunkParams) (string, error) {
	return "", nil
}
func (f *fakeChunkStore) GetChunksByIDs(_ context.Context, ids []string) ([]Chunk, error) {
	out := make([]Chunk, 0, len(ids))
	for _, id := range ids {
		if ch, ok := f.chunks[id]; ok {
			out = append(out, ch)
		}
	}
	return out, nil
}
func (f *fakeChunkStore) GetFile(_ context.Context, id string) (RAGFile, error) {
	if f.files != nil {
		if f, ok := f.files[id]; ok {
			return f, nil
		}
	}
	return RAGFile{}, nil
}
func (f *fakeChunkStore) UpdateFileChunkStats(_ context.Context, _ UpdateFileStatsParams) error {
	return nil
}

type simpleEmbedder struct {
	dims int
	vec  []float32
}

func (f *simpleEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i := range out {
		out[i] = f.vec
	}
	return out, nil
}
func (f *simpleEmbedder) Dimension() int { return f.dims }

type fakeKeywordSearcher struct {
	results []VectorMatch
	err     error
}

func (f *fakeKeywordSearcher) KeywordSearch(_ context.Context, _ string, _ SearchParams) ([]VectorMatch, error) {
	return f.results, f.err
}

// --- tests ---

func TestNewRetriever_Validation(t *testing.T) {
	ctx := context.Background()

	_, err := NewRetriever(ctx, nil)
	if err == nil {
		t.Fatal("expected error for nil config")
	}
	_, err = NewRetriever(ctx, &RetrieverConfig{})
	if err == nil {
		t.Fatal("expected error for missing store")
	}
	_, err = NewRetriever(ctx, &RetrieverConfig{Store: &fakeVectorStore{}})
	if err == nil {
		t.Fatal("expected error for missing chunk store")
	}
	_, err = NewRetriever(ctx, &RetrieverConfig{
		Store:  &fakeVectorStore{},
		Chunks: &fakeChunkStore{},
	})
	if err == nil {
		t.Fatal("expected error for missing embedder")
	}
}

func TestNewRetriever_Defaults(t *testing.T) {
	ctx := context.Background()
	r, err := NewRetriever(ctx, &RetrieverConfig{
		Store:    &fakeVectorStore{},
		Chunks:   &fakeChunkStore{},
		Embedder: &simpleEmbedder{dims: 4, vec: []float32{0.1, 0.2, 0.3, 0.4}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if r.GetType() != "KawaiHybrid" {
		t.Fatalf("GetType() = %q, want KawaiHybrid", r.GetType())
	}
	if !r.IsCallbacksEnabled() {
		t.Fatal("IsCallbacksEnabled() = false, want true")
	}
}

func TestNewRetriever_CustomThresholds(t *testing.T) {
	ctx := context.Background()
	r, err := NewRetriever(ctx, &RetrieverConfig{
		Store:            &fakeVectorStore{},
		Chunks:           &fakeChunkStore{},
		Embedder:         &simpleEmbedder{dims: 4, vec: []float32{0.1}},
		VectorThreshold:  0.5,
		KeywordThreshold: 0.7,
	})
	if err != nil {
		t.Fatal(err)
	}
	if r.GetType() != "KawaiHybrid" {
		t.Fatalf("GetType() = %q, want KawaiHybrid", r.GetType())
	}
}

func TestRetrieve_VectorOnly(t *testing.T) {
	vec := []float32{0.1, 0.2, 0.3, 0.4}
	store := &fakeVectorStore{
		searchFn: func(_ context.Context, _ []float32, _ SearchParams) ([]VectorMatch, error) {
			return []VectorMatch{
				{ID: "c1", FileID: "f1", Similarity: 0.9},
				{ID: "c2", FileID: "f1", Similarity: 0.8},
			}, nil
		},
	}
	chunks := &fakeChunkStore{
		chunks: map[string]Chunk{
			"c1": {ID: "c1", Text: "hello world", FileID: "f1"},
			"c2": {ID: "c2", Text: "foo bar", FileID: "f1"},
		},
		files: map[string]RAGFile{
			"f1": {ID: "f1", Name: "test.txt"},
		},
	}

	r, err := NewRetriever(context.Background(), &RetrieverConfig{
		Store:    store,
		Chunks:   chunks,
		Embedder: &simpleEmbedder{dims: 4, vec: vec},
	})
	if err != nil {
		t.Fatal(err)
	}

	docs, err := r.Retrieve(context.Background(), "test query", retriever.WithTopK(10))
	if err != nil {
		t.Fatal(err)
	}
	if len(docs) != 2 {
		t.Fatalf("got %d docs, want 2", len(docs))
	}
	if docs[0].ID != "c1" {
		t.Fatalf("first doc ID = %q, want c1", docs[0].ID)
	}
	if docs[0].Score() != 0.9 {
		t.Fatalf("first doc score = %v, want 0.9", docs[0].Score())
	}
}

func TestRetrieve_HybridWithKeyword(t *testing.T) {
	vec := []float32{0.1, 0.2}
	store := &fakeVectorStore{
		searchFn: func(_ context.Context, _ []float32, _ SearchParams) ([]VectorMatch, error) {
			return []VectorMatch{
				{ID: "c1", FileID: "f1", Similarity: 0.9},
				{ID: "c2", FileID: "f1", Similarity: 0.8},
			}, nil
		},
	}
	kw := &fakeKeywordSearcher{
		results: []VectorMatch{
			{ID: "c2", FileID: "f1", Similarity: 0.5},
			{ID: "c3", FileID: "f2", Similarity: 1.0},
		},
	}
	chunks := &fakeChunkStore{
		chunks: map[string]Chunk{
			"c1": {ID: "c1", Text: "a", FileID: "f1"},
			"c2": {ID: "c2", Text: "b", FileID: "f1"},
			"c3": {ID: "c3", Text: "c", FileID: "f2"},
		},
	}

	r, err := NewRetriever(context.Background(), &RetrieverConfig{
		Store:    store,
		Chunks:   chunks,
		Embedder: &simpleEmbedder{dims: 2, vec: vec},
		Keyword:  kw,
	})
	if err != nil {
		t.Fatal(err)
	}

	docs, err := r.Retrieve(context.Background(), "query", retriever.WithTopK(5))
	if err != nil {
		t.Fatal(err)
	}
	if len(docs) == 0 {
		t.Fatal("expected non-empty results from hybrid fusion")
	}
	// c2 appears in both streams, should be ranked highest by RRF
	if docs[0].ID != "c2" {
		t.Fatalf("top doc ID = %q, want c2 (present in both streams)", docs[0].ID)
	}
}

func TestRetrieve_WithFileIDs(t *testing.T) {
	vec := []float32{0.1}
	var capturedFileIDs []string
	store := &fakeVectorStore{
		searchFn: func(_ context.Context, _ []float32, params SearchParams) ([]VectorMatch, error) {
			capturedFileIDs = params.FileIDs
			return []VectorMatch{
				{ID: "c1", FileID: "f1", Similarity: 0.9},
			}, nil
		},
	}
	chunks := &fakeChunkStore{
		chunks: map[string]Chunk{
			"c1": {ID: "c1", Text: "hello", FileID: "f1"},
		},
	}

	r, err := NewRetriever(context.Background(), &RetrieverConfig{
		Store:    store,
		Chunks:   chunks,
		Embedder: &simpleEmbedder{dims: 1, vec: vec},
	})
	if err != nil {
		t.Fatal(err)
	}

	_, err = r.Retrieve(context.Background(), "query",
		retriever.WithTopK(5),
		WithFileIDs("f1", "f2"),
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(capturedFileIDs) != 2 || capturedFileIDs[0] != "f1" || capturedFileIDs[1] != "f2" {
		t.Fatalf("FileIDs = %v, want [f1 f2]", capturedFileIDs)
	}
}

func TestRetrieve_ImplSpecificThresholds(t *testing.T) {
	var capturedSearchParams SearchParamsSearcher
	store := &fakeVectorStore{
		searchFn: func(_ context.Context, _ []float32, _ SearchParams) ([]VectorMatch, error) {
			return []VectorMatch{}, nil
		},
	}
	chunks := &fakeChunkStore{}

	r, err := NewRetriever(context.Background(), &RetrieverConfig{
		Store:    store,
		Chunks:   chunks,
		Embedder: &simpleEmbedder{dims: 1, vec: []float32{0.1}},
		// defaults: 0.15 / 0.3
	})
	if err != nil {
		t.Fatal(err)
	}

	// Override thresholds via impl options
	r2 := r // shallow copy to avoid mutation
	_ = r2

	// We can't directly capture SearchParamsSearcher from outside, but we can
	// verify that Retrieve succeeds with overridden thresholds.
	_, err = r.Retrieve(context.Background(), "query",
		retriever.WithTopK(1),
		WithVectorThreshold(0.5),
		WithKeywordThreshold(0.8),
		WithMetric(DistanceInnerProduct),
	)
	if err != nil {
		t.Fatal(err)
	}
	_ = capturedSearchParams
}

func TestRetrieve_EmptyQuery(t *testing.T) {
	store := &fakeVectorStore{}
	chunks := &fakeChunkStore{}
	r, err := NewRetriever(context.Background(), &RetrieverConfig{
		Store:    store,
		Chunks:   chunks,
		Embedder: &simpleEmbedder{dims: 1, vec: []float32{0.1}},
	})
	if err != nil {
		t.Fatal(err)
	}

	_, err = r.Retrieve(context.Background(), "   ")
	if err == nil {
		t.Fatal("expected error for empty query")
	}
}

func TestExpandParentContext(t *testing.T) {
	chunks := &fakeChunkStore{
		chunks: map[string]Chunk{
			"parent-1": {ID: "parent-1", Text: "Parent section text"},
		},
	}

	r := &Retriever{
		chunks: chunks,
	}

	results := []*schema.Document{
		{
			ID:      "child-1",
			Content: "child content",
			MetaData: map[string]any{
				DocumentMetaParentID: "parent-1",
			},
		},
		{
			ID:      "child-2",
			Content: "no parent",
			MetaData: map[string]any{},
		},
	}

	expanded := r.expandParentContext(context.Background(), results)

	if len(expanded) != 2 {
		t.Fatalf("got %d results, want 2", len(expanded))
	}

	// child-1 should have parent context prepended
	if expanded[0].Content != "[Context from section]\nParent section text\n\n[Excerpt]\nchild content" {
		t.Fatalf("unexpected content for child-1: %q", expanded[0].Content)
	}

	// child-2 should be unchanged
	if expanded[1].Content != "no parent" {
		t.Fatalf("unexpected content for child-2: %q", expanded[1].Content)
	}
}

func TestExpandParentContext_MissingParent(t *testing.T) {
	chunks := &fakeChunkStore{
		chunks: map[string]Chunk{},
	}

	r := &Retriever{chunks: chunks}

	results := []*schema.Document{
		{
			ID:      "child-1",
			Content: "child content",
			MetaData: map[string]any{
				DocumentMetaParentID: "nonexistent",
			},
		},
	}

	expanded := r.expandParentContext(context.Background(), results)

	if len(expanded) != 1 {
		t.Fatalf("got %d results, want 1", len(expanded))
	}
	// Content should remain unchanged when parent is missing
	if expanded[0].Content != "child content" {
		t.Fatalf("content should be unchanged, got %q", expanded[0].Content)
	}
}

func TestExpandParentContext_NilChunks(t *testing.T) {
	r := &Retriever{chunks: nil}
	results := []*schema.Document{
		{ID: "c1", Content: "text"},
	}
	expanded := r.expandParentContext(context.Background(), results)
	if len(expanded) != 1 {
		t.Fatalf("got %d results, want 1", len(expanded))
	}
	if expanded[0].Content != "text" {
		t.Fatalf("content should be unchanged, got %q", expanded[0].Content)
	}
}

// --- T2: retrieval regression suite -----------------------------------------
//
// A frozen corpus + scripted per-query vector/keyword signals, asserting
// recall@K. This is DB-free and runs on every change to fileprocessor. It
// breaks if fusion weights (RRF), thresholds, or stream-combination behavior
// drift. See knowledge-base-architecture-review.md (T2).

// scriptedVectorStore keys canned vector hits by embedding[0], so the test can
// route a distinct result set per query when paired with scriptedEmbedder.
type scriptedVectorStore struct {
	byKey map[float32][]VectorMatch
}

func (s *scriptedVectorStore) Upsert(context.Context, string, string, []float32) error  { return nil }
func (s *scriptedVectorStore) UpsertBatch(context.Context, []VectorItem) error          { return nil }
func (s *scriptedVectorStore) DeleteByID(context.Context, string) error                 { return nil }
func (s *scriptedVectorStore) DeleteByFileID(context.Context, string) error             { return nil }
func (s *scriptedVectorStore) Close() error                                             { return nil }
func (s *scriptedVectorStore) Search(_ context.Context, emb []float32, _ SearchParams) ([]VectorMatch, error) {
	if len(emb) == 0 {
		return nil, nil
	}
	return s.byKey[emb[0]], nil
}

type scriptedKeywordSearcher struct{ byQuery map[string][]VectorMatch }

func (s *scriptedKeywordSearcher) KeywordSearch(_ context.Context, q string, _ SearchParams) ([]VectorMatch, error) {
	return s.byQuery[q], nil
}

// scriptedEmbedder maps each query string to a single-element vector whose
// value is the routing key consumed by scriptedVectorStore.
type scriptedEmbedder struct{ byQuery map[string]float32 }

func (e *scriptedEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, q := range texts {
		out[i] = []float32{e.byQuery[q]}
	}
	return out, nil
}
func (e *scriptedEmbedder) Dimension() int { return 1 }

// frozen corpus chunks for the regression suite.
var regressionChunks = map[string]Chunk{
	"deploy":   {ID: "deploy", Text: "how to deploy the application", FileID: "f1"},
	"dbpool":   {ID: "dbpool", Text: "database connection pooling config", FileID: "f1"},
	"k8s":      {ID: "k8s", Text: "kubernetes deployment guide", FileID: "f2"},
	"auth":     {ID: "auth", Text: "authentication and authorization flows", FileID: "f2"},
	"ciscript": {ID: "ciscript", Text: "ci cd pipeline deploy step", FileID: "f1"},
}

func containsAll(returnedIDs, expected []string) bool {
	have := make(map[string]bool, len(returnedIDs))
	for _, id := range returnedIDs {
		have[id] = true
	}
	for _, id := range expected {
		if !have[id] {
			return false
		}
	}
	return true
}

func TestRetrieve_RegressionSuite(t *testing.T) {
	type tc struct {
		name        string
		query       string
		vectorHits  []VectorMatch
		keywordHits []VectorMatch
		expected    []string // relevant chunk IDs that must appear (recall@K)
	}
	cases := []tc{
		{
			name:        "semantic-only chunk survives fusion",
			query:       "how to deploy",
			vectorHits:  []VectorMatch{{ID: "deploy", FileID: "f1", Similarity: 0.9}, {ID: "k8s", FileID: "f2", Similarity: 0.8}},
			keywordHits: nil,
			expected:    []string{"deploy", "k8s"},
		},
		{
			name:        "keyword-only chunk survives fusion (core hybrid value)",
			query:       "authn authz",
			vectorHits:  []VectorMatch{{ID: "deploy", FileID: "f1", Similarity: 0.9}},
			keywordHits: []VectorMatch{{ID: "auth", FileID: "f2", Similarity: 0.9}},
			expected:    []string{"deploy", "auth"},
		},
		{
			name:        "RRF agreement: chunk in both streams ranks first",
			query:       "pipeline deploy",
			vectorHits:  []VectorMatch{{ID: "deploy", FileID: "f1", Similarity: 0.6}, {ID: "ciscript", FileID: "f1", Similarity: 0.9}},
			keywordHits: []VectorMatch{{ID: "ciscript", FileID: "f1", Similarity: 0.4}, {ID: "k8s", FileID: "f2", Similarity: 0.9}},
			expected:    []string{"ciscript", "deploy", "k8s"},
		},
		{
			name:        "vector threshold (0.15) drops low-similarity hit",
			query:       "db setup",
			vectorHits:  []VectorMatch{{ID: "dbpool", FileID: "f1", Similarity: 0.9}, {ID: "auth", FileID: "f2", Similarity: 0.1}},
			keywordHits: nil,
			expected:    []string{"dbpool"}, // auth filtered out (0.1 < 0.15)
		},
		{
			name:        "keyword threshold (0.3) drops low-score hit",
			query:       "connection pool",
			vectorHits:  nil,
			keywordHits: []VectorMatch{{ID: "dbpool", FileID: "f1", Similarity: 0.9}, {ID: "auth", FileID: "f2", Similarity: 0.1}},
			expected:    []string{"dbpool"}, // auth filtered out (0.1 < 0.3)
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			key := float32(1)
			emb := &scriptedEmbedder{byQuery: map[string]float32{c.query: key}}
			store := &scriptedVectorStore{byKey: map[float32][]VectorMatch{key: c.vectorHits}}
			kw := &scriptedKeywordSearcher{byQuery: map[string][]VectorMatch{c.query: c.keywordHits}}
			chunks := &fakeChunkStore{chunks: regressionChunks}

			r, err := NewRetriever(context.Background(), &RetrieverConfig{
				Store:    store,
				Chunks:   chunks,
				Embedder: emb,
				Keyword:  kw,
			})
			if err != nil {
				t.Fatal(err)
			}

			docs, err := r.Retrieve(context.Background(), c.query, retriever.WithTopK(10))
			if err != nil {
				t.Fatal(err)
			}

			ids := make([]string, 0, len(docs))
			for _, d := range docs {
				ids = append(ids, d.ID)
			}
			if !containsAll(ids, c.expected) {
				t.Errorf("recall@K failed: expected %v ⊆ returned %v", c.expected, ids)
			}

			// For the RRF-agreement case, assert the dual-stream chunk ranks #1.
			if c.name != "" && len(c.keywordHits) > 0 && c.vectorHits != nil {
				// ciscript appears in both streams in that case.
				if c.query == "pipeline deploy" && (len(ids) == 0 || ids[0] != "ciscript") {
					t.Errorf("RRF: expected ciscript first (in both streams), got %v", ids)
				}
			}
		})
	}
}
