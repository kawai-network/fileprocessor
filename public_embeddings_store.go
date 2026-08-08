package fileprocessor

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/lib/pq"
	"github.com/pgvector/pgvector-go"
)

// PublicEmbeddingsStore implements [VectorStore] against the lobehub
// public.embeddings table. Use it when the host app wants to share
// embeddings storage with its existing RAG pipeline.
//
// Constraints:
//   - The embeddings column is hard-pinned to vector(1024) by the lobehub
//     schema. The constructor rejects any other dim.
//   - The store uses public.embeddings.chunk_id as the natural key (one
//     embedding per chunk; UNIQUE on chunk_id). When the chunk row is
//     deleted, the FK CASCADE removes the embedding automatically.
//   - There is no file_id column on embeddings. file_id is hydrated via a
//     JOIN to public.file_chunks in Search results, and used as the
//     subquery in DeleteByFileID.
//
// The store creates an HNSW index on public.embeddings if one does not
// already exist. The index name is "public_embeddings_hnsw_idx" and
// uses vector_cosine_ops. Existing indexes (and their operator classes)
// are left alone.
type PublicEmbeddingsStore struct {
	pool   *pgxpool.Pool
	config *PgHNSWConfig
	model  string
}

// Compile-time interface check.
var _ VectorStore = (*PublicEmbeddingsStore)(nil)

// defaultModelTag is stamped into the embeddings.model column when a caller
// has not set a real model name via SetModel. It distinguishes rows written by
// the fileprocessor store from rows written by the host app's other pipelines.
const defaultModelTag = "fileprocessor"

// modelTag returns the value written to embeddings.model. Callers that use this
// store for writes should SetModel with the real embedding model name so the
// column carries useful information (architecture review R4). Note: the kawai
// production ingest path (egent-jobs) writes embeddings directly with the real
// model and bypasses this store's write methods.
func (s *PublicEmbeddingsStore) modelTag() string {
	if s.model != "" {
		return s.model
	}
	return defaultModelTag
}

// SetModel sets the value written to the embeddings.model column on subsequent
// Upsert/UpsertBatch calls, so callers using this store for writes stamp the
// real embedding model name rather than the generic "fileprocessor" default.
// Reads (Search/KeywordSearch) are unaffected.
func (s *PublicEmbeddingsStore) SetModel(model string) { s.model = model }

// hnswIndexName is the dedicated HNSW index this store creates on
// public.embeddings. Keep this name stable so the index is reused
// across process restarts.
const hnswIndexName = "public_embeddings_hnsw_idx"

// NewPublicEmbeddingsStore opens a connection pool to dsn and
// initializes the HNSW index on public.embeddings. dim must be 1024
// (the schema's fixed vector dimension). Pass nil for cfg to use
// default HNSW parameters.
func NewPublicEmbeddingsStore(ctx context.Context, dsn string, dim int, cfg *PgHNSWConfig) (*PublicEmbeddingsStore, error) {
	if dim != DefaultEmbeddingDim {
		return nil, fmt.Errorf("public_embeddings_store: public.embeddings is pinned to vector(%d), got dim=%d", DefaultEmbeddingDim, dim)
	}
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		return nil, fmt.Errorf("public_embeddings_store: pgxpool.New: %w", err)
	}
	s := &PublicEmbeddingsStore{pool: pool, config: cfg.normalize()}
	if err := s.init(ctx); err != nil {
		pool.Close()
		return nil, err
	}
	return s, nil
}

// NewPublicEmbeddingsStoreWithPool is for callers that already manage
// a pool. The caller owns the pool's lifecycle.
func NewPublicEmbeddingsStoreWithPool(ctx context.Context, pool *pgxpool.Pool, cfg *PgHNSWConfig) (*PublicEmbeddingsStore, error) {
	s := &PublicEmbeddingsStore{pool: pool, config: cfg.normalize()}
	if err := s.init(ctx); err != nil {
		return nil, err
	}
	return s, nil
}

// Close releases the underlying connection pool. A store created via
// [NewPublicEmbeddingsStore] owns its pool. Callers that created the
// pool externally and used [NewPublicEmbeddingsStoreWithPool] must
// manage the pool lifecycle themselves.
func (s *PublicEmbeddingsStore) Close() error {
	if s.pool != nil {
		s.pool.Close()
		s.pool = nil
	}
	return nil
}

// SetEfSearch overrides the runtime ef_search for subsequent queries
// on this store's connections. Higher values improve recall at the
// cost of latency.
func (s *PublicEmbeddingsStore) SetEfSearch(efSearch int) error {
	if s.pool == nil {
		return errors.New("public_embeddings_store: store is closed")
	}
	if _, err := s.pool.Exec(context.Background(),
		fmt.Sprintf("SET hnsw.ef_search = %d", efSearch)); err != nil {
		return fmt.Errorf("public_embeddings_store: set hnsw.ef_search: %w", err)
	}
	return nil
}

// ResetEfSearch restores the runtime ef_search to the value configured
// at construction time.
func (s *PublicEmbeddingsStore) ResetEfSearch() error {
	if s.pool == nil {
		return errors.New("public_embeddings_store: store is closed")
	}
	if _, err := s.pool.Exec(context.Background(), "RESET hnsw.ef_search"); err != nil {
		return fmt.Errorf("public_embeddings_store: reset hnsw.ef_search: %w", err)
	}
	return nil
}

func (s *PublicEmbeddingsStore) init(ctx context.Context) error {
	// Validate the table exists with vector(1024).
	if err := s.verifyColumn(ctx); err != nil {
		return err
	}

	// Create HNSW index if missing. Use IF NOT EXISTS so we don't fight
	// any pre-existing index the host app may have created.
	ops := metricOpsClass(s.config.Metric)
	q := fmt.Sprintf(`
		CREATE INDEX IF NOT EXISTS %s
		ON public.embeddings USING hnsw (embeddings %s)
	`, hnswIndexName, ops)
	if _, err := s.pool.Exec(ctx, q); err != nil {
		return fmt.Errorf("public_embeddings_store: create hnsw index: %w", err)
	}

	// Apply per-session ef_search.
	if s.config != nil && s.config.EfSearch > 0 {
		if _, err := s.pool.Exec(ctx, fmt.Sprintf("SET hnsw.ef_search = %d", s.config.EfSearch)); err != nil {
			slog.Warn("public_embeddings_store: failed to set hnsw.ef_search on init", "error", err)
		}
	}
	return nil
}

func (s *PublicEmbeddingsStore) verifyColumn(ctx context.Context) error {
	row := s.pool.QueryRow(ctx, `
		SELECT format_type(atttypid, atttypmod)
		FROM pg_attribute
		WHERE attrelid = 'public.embeddings'::regclass
		  AND attname = 'embeddings'
		LIMIT 1
	`)
	var dataType string
	if err := row.Scan(&dataType); err != nil {
		return fmt.Errorf("public_embeddings_store: read embeddings column: %w", err)
	}
	dim, err := parsePgVectorDimension(dataType)
	if err != nil {
		return fmt.Errorf("public_embeddings_store: parse embeddings type %q: %w", dataType, err)
	}
	if dim != DefaultEmbeddingDim {
		return fmt.Errorf("public_embeddings_store: expected dim %d, found dim %d", DefaultEmbeddingDim, dim)
	}
	return nil
}

// pgQueryer is the query capability shared by *pgxpool.Pool and pgx.Tx.
type pgQueryer interface {
	Query(ctx context.Context, sql string, args ...any) (pgx.Rows, error)
}

// withTenantGuard runs fn inside a short-lived transaction with
// SET LOCAL app.tenant_id = tenantID when tenantID is non-empty, so the RLS
// policies on files/file_chunks/chunks enforce tenant isolation at the DB
// boundary rather than via the in-memory file-ID allowlist. Rows must be fully
// consumed inside fn (before it returns), because the tx is rolled back
// afterwards. When tenantID is empty, fn runs against the pool directly
// (legacy path, unchanged behavior). See architecture review R3.
func (s *PublicEmbeddingsStore) withTenantGuard(ctx context.Context, tenantID string, fn func(pgQueryer) error) error {
	if tenantID == "" {
		return fn(s.pool)
	}
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("public_embeddings_store: begin tenant-guarded tx: %w", err)
	}
	defer func() { _ = tx.Rollback(ctx) }()
	if _, err := tx.Exec(ctx, fmt.Sprintf("SET LOCAL app.tenant_id = %s", pq.QuoteLiteral(tenantID))); err != nil {
		return fmt.Errorf("public_embeddings_store: set tenant: %w", err)
	}
	return fn(tx)
}

// --- VectorStore implementation -------------------------------------------

// Upsert inserts or updates a single embedding. The id parameter is
// interpreted as the chunk_id (the natural key for embeddings). The
// fileID is accepted for interface compatibility but is not stored —
// the relationship is recomputed via the file_chunks join on demand.
func (s *PublicEmbeddingsStore) Upsert(ctx context.Context, id, fileID string, embedding []float32) error {
	if len(embedding) != DefaultEmbeddingDim {
		return fmt.Errorf("public_embeddings_store: embedding dim %d != %d", len(embedding), DefaultEmbeddingDim)
	}
	q := `INSERT INTO public.embeddings (chunk_id, embeddings, model)
		  VALUES ($1, $2, $3)
		  ON CONFLICT (chunk_id) DO UPDATE
		  SET embeddings = EXCLUDED.embeddings,
		      model = EXCLUDED.model`
	if _, err := s.pool.Exec(ctx, q, id, pgvector.NewVector(embedding), s.modelTag()); err != nil {
		return fmt.Errorf("public_embeddings_store: upsert: %w", err)
	}
	return nil
}

// UpsertBatch inserts or updates many embeddings in a single
// transaction. Uses COPY FROM for bulk inserts; falls back to per-row
// upsert on failure.
func (s *PublicEmbeddingsStore) UpsertBatch(ctx context.Context, items []VectorItem) error {
	if len(items) == 0 {
		return nil
	}
	for _, it := range items {
		if len(it.Embedding) != DefaultEmbeddingDim {
			return fmt.Errorf("public_embeddings_store: embedding dim %d != %d", len(it.Embedding), DefaultEmbeddingDim)
		}
	}

	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("public_embeddings_store: begin batch: %w", err)
	}
	defer func() { _ = tx.Rollback(ctx) }()

	// Try COPY first.
	rows := make([][]any, len(items))
	for i, it := range items {
		rows[i] = []any{it.ID, pgvector.NewVector(it.Embedding), s.modelTag()}
	}
	_, copyErr := tx.CopyFrom(ctx,
		pgx.Identifier{"public", "embeddings"},
		[]string{"chunk_id", "embeddings", "model"},
		pgx.CopyFromRows(rows),
	)
	if copyErr == nil {
		if err := tx.Commit(ctx); err != nil {
			return fmt.Errorf("public_embeddings_store: commit batch: %w", err)
		}
		return nil
	}
	_ = tx.Rollback(ctx)

	// Fallback: per-row upsert.
	tx, err = s.pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("public_embeddings_store: begin batch fallback: %w", err)
	}
	defer func() { _ = tx.Rollback(ctx) }()

	q := `INSERT INTO public.embeddings (chunk_id, embeddings, model)
		  VALUES ($1, $2, $3)
		  ON CONFLICT (chunk_id) DO UPDATE
		  SET embeddings = EXCLUDED.embeddings,
		      model = EXCLUDED.model`
	for _, it := range items {
		if _, err := tx.Exec(ctx, q, it.ID, pgvector.NewVector(it.Embedding), s.modelTag()); err != nil {
			return fmt.Errorf("public_embeddings_store: batch upsert id=%s: %w", it.ID, err)
		}
	}
	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("public_embeddings_store: commit batch fallback: %w", err)
	}
	return nil
}

// Search returns the top-K most similar embeddings. file_id is
// hydrated via a JOIN to public.file_chunks.
func (s *PublicEmbeddingsStore) Search(ctx context.Context, embedding []float32, params SearchParams) ([]VectorMatch, error) {
	if len(embedding) != DefaultEmbeddingDim {
		return nil, fmt.Errorf("public_embeddings_store: embedding dim %d != %d", len(embedding), DefaultEmbeddingDim)
	}
	metric := params.Metric
	if metric == "" {
		metric = s.config.Metric
	}
	op := metricDistanceOp(metric)

	limit := params.Limit
	if limit <= 0 {
		limit = 10
	}

	where := ""
	args := []any{pgvector.NewVector(embedding), limit}
	if len(params.FileIDs) > 0 {
		where = "WHERE fc.file_id = ANY($3)"
		args = append(args, params.FileIDs)
	}
	q := fmt.Sprintf(`
		SELECT e.chunk_id, COALESCE(fc.file_id, ''), e.embeddings %s $1 AS distance
		FROM public.embeddings e
		LEFT JOIN public.file_chunks fc ON fc.chunk_id = e.chunk_id
		%s
		ORDER BY distance ASC
		LIMIT $2
	`, op, where)

	var out []VectorMatch
	if err := s.withTenantGuard(ctx, params.TenantID, func(qr pgQueryer) error {
		collected, err := s.searchImpl(ctx, qr, q, args, metric, limit)
		if err != nil {
			return err
		}
		out = collected
		return nil
	}); err != nil {
		return nil, err
	}
	return out, nil
}

// searchImpl builds and runs the vector-search query against queryer q.
func (s *PublicEmbeddingsStore) searchImpl(ctx context.Context, q pgQueryer, sql string, args []any, metric DistanceMetric, limit int) ([]VectorMatch, error) {
	rows, err := q.Query(ctx, sql, args...)
	if err != nil {
		return nil, fmt.Errorf("public_embeddings_store: search: %w", err)
	}
	defer rows.Close()
	out, err := collectSearchResults(rows, metric, limit)
	if err != nil {
		return nil, fmt.Errorf("public_embeddings_store: %w", err)
	}
	return out, nil
}

// KeywordSearch performs PostgreSQL full-text search over chunks. It is used
// by Searcher as the keyword half of hybrid RRF search.
func (s *PublicEmbeddingsStore) KeywordSearch(ctx context.Context, query string, params SearchParams) ([]VectorMatch, error) {
	limit := params.Limit
	if limit <= 0 {
		limit = 10
	}
	where := "c.fts_vector @@ plainto_tsquery('simple', $1)"
	args := []any{query, limit}
	if len(params.FileIDs) > 0 {
		where += " AND fc.file_id = ANY($3)"
		args = append(args, params.FileIDs)
	}
	q := fmt.Sprintf(`
		SELECT c.id, COALESCE(fc.file_id, ''),
		       ts_rank_cd(c.fts_vector, plainto_tsquery('simple', $1)) AS score
		FROM public.chunks c
		JOIN public.file_chunks fc ON fc.chunk_id = c.id
		WHERE %s
		ORDER BY score DESC, c.id
		LIMIT $2
	`, where)
	var out []VectorMatch
	if err := s.withTenantGuard(ctx, params.TenantID, func(qr pgQueryer) error {
		collected, err := s.keywordSearchImpl(ctx, qr, q, args, limit)
		if err != nil {
			return err
		}
		out = collected
		return nil
	}); err != nil {
		return nil, err
	}
	return out, nil
}

// keywordSearchImpl runs the FTS query against queryer q and collects matches.
func (s *PublicEmbeddingsStore) keywordSearchImpl(ctx context.Context, q pgQueryer, sql string, args []any, limit int) ([]VectorMatch, error) {
	rows, err := q.Query(ctx, sql, args...)
	if err != nil {
		return nil, fmt.Errorf("public_embeddings_store: keyword search: %w", err)
	}
	defer rows.Close()
	out := make([]VectorMatch, 0, limit)
	for rows.Next() {
		var match VectorMatch
		if err := rows.Scan(&match.ID, &match.FileID, &match.Similarity); err != nil {
			return nil, fmt.Errorf("public_embeddings_store: scan keyword row: %w", err)
		}
		out = append(out, match)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("public_embeddings_store: iterate keyword rows: %w", err)
	}
	return out, nil
}

// DeleteByID removes a single embedding. The id parameter is the
// chunk_id (matches Upsert / Search).
func (s *PublicEmbeddingsStore) DeleteByID(ctx context.Context, id string) error {
	if _, err := s.pool.Exec(ctx, `DELETE FROM public.embeddings WHERE chunk_id = $1`, id); err != nil {
		return fmt.Errorf("public_embeddings_store: delete by id: %w", err)
	}
	return nil
}

// DeleteByFileID removes all embeddings whose chunk is linked to the
// given file. The lookup is via public.file_chunks.
func (s *PublicEmbeddingsStore) DeleteByFileID(ctx context.Context, fileID string) error {
	q := `DELETE FROM public.embeddings
		  WHERE chunk_id IN (SELECT chunk_id FROM public.file_chunks WHERE file_id = $1)`
	if _, err := s.pool.Exec(ctx, q, fileID); err != nil {
		return fmt.Errorf("public_embeddings_store: delete by file_id: %w", err)
	}
	return nil
}

// --- Batch search ---------------------------------------------------------

// BatchSearch runs many queries in a single round-trip using a
// LATERAL join. Same shape as [PgVectorStore.BatchSearch]; dim is
// fixed at 1024.
func (s *PublicEmbeddingsStore) BatchSearch(ctx context.Context, queries []BatchSearchRequest, limit int) ([]BatchSearchResult, error) {
	if len(queries) == 0 {
		return nil, nil
	}
	for _, q := range queries {
		if len(q.Embedding) != DefaultEmbeddingDim {
			return nil, fmt.Errorf("public_embeddings_store: query dim %d != %d", len(q.Embedding), DefaultEmbeddingDim)
		}
	}
	if limit <= 0 {
		limit = 10
	}

	op := metricDistanceOp(s.config.Metric)

	args := make([]any, 0, len(queries)*2+1)
	valuesSQL := make([]string, len(queries))
	for i, q := range queries {
		valuesSQL[i] = fmt.Sprintf("($%d::text, $%d::vector(1024))", i*2+1, i*2+2)
		args = append(args, q.QueryID, pgvector.NewVector(q.Embedding))
	}
	limitSlot := len(args) + 1
	args = append(args, limit)

	q := fmt.Sprintf(`
		WITH queries(query_id, embedding) AS (VALUES %s)
		SELECT q.query_id, e.chunk_id, COALESCE(fc.file_id, ''), e.embeddings %s q.embedding AS distance
		FROM queries q
		CROSS JOIN LATERAL (
			SELECT chunk_id, embeddings
			FROM public.embeddings
			ORDER BY embeddings %s q.embedding ASC
			LIMIT $%d
		) e
		LEFT JOIN public.file_chunks fc ON fc.chunk_id = e.chunk_id
		ORDER BY q.query_id, distance ASC
	`, strings.Join(valuesSQL, ", "), op, op, limitSlot)

	rows, err := s.pool.Query(ctx, q, args...)
	if err != nil {
		return nil, fmt.Errorf("public_embeddings_store: batch search: %w", err)
	}
	defer rows.Close()

	byID, err := collectBatchResults(rows, s.config.Metric)
	if err != nil {
		return nil, fmt.Errorf("public_embeddings_store: %w", err)
	}

	out := make([]BatchSearchResult, 0, len(queries))
	for _, q := range queries {
		out = append(out, BatchSearchResult{QueryID: q.QueryID, Matches: byID[q.QueryID]})
	}
	return out, nil
}

// --- helpers ---------------------------------------------------------------

// (parsePgVectorDimension and metricOpsClass/metricDistanceOp/
// distanceToSimilarityPg are defined in pgvector_store.go and shared.)
