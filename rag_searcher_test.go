package fileprocessor

import "testing"

func TestFuseRRFCombinesRankings(t *testing.T) {
	vector := []VectorMatch{
		{ID: "vector-only", Similarity: 0.99},
		{ID: "both", Similarity: 0.80},
	}
	keyword := []VectorMatch{
		{ID: "both", Similarity: 0.50},
		{ID: "keyword-only", Similarity: 1.00},
	}

	got := fuseRRF(vector, keyword, 3)
	if len(got) != 3 {
		t.Fatalf("got %d results, want 3", len(got))
	}
	if got[0].ID != "both" {
		t.Fatalf("top result = %q, want result present in both rankings", got[0].ID)
	}
	if got[0].Similarity <= got[1].Similarity {
		t.Fatalf("top fused score = %v, next = %v; want descending scores", got[0].Similarity, got[1].Similarity)
	}
}

func TestFuseRRFRespectsLimit(t *testing.T) {
	got := fuseRRF([]VectorMatch{{ID: "a"}, {ID: "b"}}, []VectorMatch{{ID: "c"}}, 2)
	if len(got) != 2 {
		t.Fatalf("got %d results, want 2", len(got))
	}
}

func TestFilterMatchesByScore(t *testing.T) {
	got := filterMatchesByScore([]VectorMatch{
		{ID: "low", Similarity: 0.14},
		{ID: "equal", Similarity: 0.15},
		{ID: "high", Similarity: 0.9},
	}, 0.15)
	if len(got) != 2 {
		t.Fatalf("got %d results, want 2", len(got))
	}
	if got[0].ID != "equal" || got[1].ID != "high" {
		t.Fatalf("got IDs %q, %q; want equal, high", got[0].ID, got[1].ID)
	}
}
