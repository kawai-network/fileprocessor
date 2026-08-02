package fileprocessor

import (
	"context"
	"testing"
)

func TestSQLiteKeywordIndexBM25SearchAndFileFilter(t *testing.T) {
	idx, err := NewSQLiteKeywordIndex(":memory:")
	if err != nil {
		t.Fatalf("NewSQLiteKeywordIndex: %v", err)
	}
	defer idx.Close()

	_, err = idx.db.Exec(`
		INSERT INTO chunks_fts (chunk_id, file_id, tenant_id, content) VALUES
			('chunk-a', 'file-a', 'tenant-a', 'database migration deployment'),
			('chunk-b', 'file-b', 'tenant-a', 'database migration migration migration'),
			('chunk-c', 'file-c', 'tenant-b', 'weather forecast')
	`)
	if err != nil {
		t.Fatalf("insert FTS rows: %v", err)
	}

	results, err := idx.KeywordSearch(context.Background(), "migration", SearchParams{Limit: 5})
	if err != nil {
		t.Fatalf("KeywordSearch: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("got %d results, want 2", len(results))
	}
	if results[0].ID != "chunk-b" {
		t.Fatalf("top result = %q, want chunk-b", results[0].ID)
	}

	filtered, err := idx.KeywordSearch(context.Background(), "migration", SearchParams{
		Limit:   5,
		FileIDs: []string{"file-a"},
	})
	if err != nil {
		t.Fatalf("filtered KeywordSearch: %v", err)
	}
	if len(filtered) != 1 || filtered[0].ID != "chunk-a" {
		t.Fatalf("filtered results = %#v, want chunk-a", filtered)
	}
}

func TestSQLiteFTSQueryQuotesTerms(t *testing.T) {
	got := sqliteFTSQuery(`ERR-AUTH-401 "login failed"`)
	if got == "" || got == `ERR-AUTH-401 "login failed"` {
		t.Fatalf("sqliteFTSQuery did not quote terms: %q", got)
	}
}
