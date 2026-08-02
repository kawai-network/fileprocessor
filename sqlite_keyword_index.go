package fileprocessor

import (
	"context"
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"github.com/jackc/pgx/v5/pgxpool"
	_ "modernc.org/sqlite"
)

// SQLiteKeywordIndex provides BM25 keyword retrieval through SQLite FTS5.
// PostgreSQL remains the source of truth; this database is a rebuildable
// search projection containing only chunk identity, scope, and text.
type SQLiteKeywordIndex struct {
	db   *sql.DB
	path string
	mu   sync.RWMutex
}

var _ KeywordSearcher = (*SQLiteKeywordIndex)(nil)

// NewSQLiteKeywordIndex opens or creates an FTS5 index at path. Use ":memory:"
// in tests. Parent directories are created for file-backed indexes.
func NewSQLiteKeywordIndex(path string) (*SQLiteKeywordIndex, error) {
	if strings.TrimSpace(path) == "" {
		return nil, fmt.Errorf("sqlite keyword index: path is required")
	}
	if path != ":memory:" {
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			return nil, fmt.Errorf("sqlite keyword index: create parent directory: %w", err)
		}
	}
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, fmt.Errorf("sqlite keyword index: open: %w", err)
	}
	// A single connection keeps :memory: databases consistent and avoids
	// SQLITE_BUSY errors while the periodic rebuild swaps the projection.
	db.SetMaxOpenConns(1)
	idx := &SQLiteKeywordIndex{db: db, path: path}
	if err := idx.init(); err != nil {
		_ = db.Close()
		return nil, err
	}
	return idx, nil
}

func (i *SQLiteKeywordIndex) init() error {
	for _, q := range []string{
		`PRAGMA journal_mode=WAL`,
		`PRAGMA synchronous=NORMAL`,
		`PRAGMA busy_timeout=5000`,
		`CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
			chunk_id UNINDEXED,
			file_id UNINDEXED,
			tenant_id UNINDEXED,
			content,
			tokenize='unicode61'
		)`,
	} {
		if _, err := i.db.Exec(q); err != nil {
			return fmt.Errorf("sqlite keyword index: initialize: %w", err)
		}
	}
	return nil
}

// RefreshFromPostgres rebuilds the projection atomically from PostgreSQL.
// Parent chunks are naturally excluded because only file-linked chunks are
// copied into the keyword index.
func (i *SQLiteKeywordIndex) RefreshFromPostgres(ctx context.Context, pool *pgxpool.Pool) error {
	if pool == nil {
		return fmt.Errorf("sqlite keyword index: postgres pool is nil")
	}
	rows, err := pool.Query(ctx, `
		SELECT c.id::text, COALESCE(fc.file_id::text, ''),
		       COALESCE(c.tenant_id::text, ''), c.text
		FROM public.chunks c
		JOIN public.file_chunks fc ON fc.chunk_id = c.id
		ORDER BY c.id
	`)
	if err != nil {
		return fmt.Errorf("sqlite keyword index: read postgres chunks: %w", err)
	}
	defer rows.Close()

	i.mu.Lock()
	defer i.mu.Unlock()
	tx, err := i.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("sqlite keyword index: begin refresh: %w", err)
	}
	rollback := func() { _ = tx.Rollback() }
	if _, err := tx.ExecContext(ctx, `DELETE FROM chunks_fts`); err != nil {
		rollback()
		return fmt.Errorf("sqlite keyword index: clear: %w", err)
	}
	stmt, err := tx.PrepareContext(ctx, `INSERT INTO chunks_fts (chunk_id, file_id, tenant_id, content) VALUES (?, ?, ?, ?)`)
	if err != nil {
		rollback()
		return fmt.Errorf("sqlite keyword index: prepare refresh: %w", err)
	}
	defer stmt.Close()
	for rows.Next() {
		var chunkID, fileID, tenantID, content string
		if err := rows.Scan(&chunkID, &fileID, &tenantID, &content); err != nil {
			rollback()
			return fmt.Errorf("sqlite keyword index: scan postgres chunk: %w", err)
		}
		if _, err := stmt.ExecContext(ctx, chunkID, fileID, tenantID, content); err != nil {
			rollback()
			return fmt.Errorf("sqlite keyword index: insert chunk %s: %w", chunkID, err)
		}
	}
	if err := rows.Err(); err != nil {
		rollback()
		return fmt.Errorf("sqlite keyword index: iterate postgres chunks: %w", err)
	}
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("sqlite keyword index: commit refresh: %w", err)
	}
	return nil
}

// KeywordSearch performs BM25-ranked search using SQLite FTS5. FTS5 returns
// lower BM25 values for better matches; the sign is inverted for the common
// higher-is-better VectorMatch score convention. RRF uses rank, not this raw
// score, when combining results.
func (i *SQLiteKeywordIndex) KeywordSearch(ctx context.Context, query string, params SearchParams) ([]VectorMatch, error) {
	ftsQuery := sqliteFTSQuery(query)
	if ftsQuery == "" {
		return []VectorMatch{}, nil
	}
	limit := params.Limit
	if limit <= 0 {
		limit = 10
	}

	args := []any{ftsQuery}
	where := []string{"chunks_fts MATCH ?"}
	if len(params.FileIDs) > 0 {
		placeholders := make([]string, len(params.FileIDs))
		for n, fileID := range params.FileIDs {
			placeholders[n] = "?"
			args = append(args, fileID)
		}
		where = append(where, "file_id IN ("+strings.Join(placeholders, ",")+")")
	}
	args = append(args, limit)

	i.mu.RLock()
	defer i.mu.RUnlock()
	q := `SELECT chunk_id, file_id, bm25(chunks_fts) FROM chunks_fts WHERE ` +
		strings.Join(where, " AND ") + ` ORDER BY bm25(chunks_fts) ASC, rowid LIMIT ?`
	rows, err := i.db.QueryContext(ctx, q, args...)
	if err != nil {
		return nil, fmt.Errorf("sqlite keyword index: search: %w", err)
	}
	defer rows.Close()
	out := make([]VectorMatch, 0, limit)
	for rows.Next() {
		var match VectorMatch
		var score float64
		if err := rows.Scan(&match.ID, &match.FileID, &score); err != nil {
			return nil, fmt.Errorf("sqlite keyword index: scan search row: %w", err)
		}
		match.Similarity = -score
		out = append(out, match)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("sqlite keyword index: iterate search rows: %w", err)
	}
	return out, nil
}

// Close closes the SQLite projection.
func (i *SQLiteKeywordIndex) Close() error {
	if i == nil || i.db == nil {
		return nil
	}
	return i.db.Close()
}

func sqliteFTSQuery(query string) string {
	words := strings.Fields(strings.TrimSpace(query))
	quoted := make([]string, 0, len(words))
	for _, word := range words {
		word = strings.TrimSpace(word)
		if word == "" {
			continue
		}
		quoted = append(quoted, `"`+strings.ReplaceAll(word, `"`, `""`)+`"`)
	}
	sort.Strings(quoted)
	return strings.Join(quoted, " OR ")
}
