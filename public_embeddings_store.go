package fileprocessor

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
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
}

// Compile-time interface check.
var _ VectorStore = (*PublicEmbeddingsStore)(nil)

// modelTag is stamped into the embeddings.model column so callers can
// distinguish rows written by the fileprocessor from rows written by
// the host app's other pipelines.
const modelTag = "fileprocessor"

// hnswIndexName is the dedicated HNSW index this store creates on
// public.embeddings. Keep this name stable so the index is reused
// across process restarts.
const hnswIndexName = "public_embeddings_hnsw_idx"

// NewPublicEmbeddingsStore opens a connection pool to dsn and
// initializes the HNSW index on public.embeddings. dim must be 1024
// (the schema's fixed vector dimension). Pass nil for cfg to use
// default HNSW parameters.
func NewPublicEmbeddingsStore(ctx context.Context, dsn string, dim int, cfg *PgHNSWConfig) (*PublicEmbeddingsStore, error) {
	if dim != 1024 {
		return nil, fmt.Errorf("public_embeddings_store: public.embeddings is pinned to vector(1024), got dim=%d", dim)
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
	if dim != 1024 {
		return fmt.Errorf("public_embeddings_store: expected dim 1024, found dim %d", dim)
	}
	return nil
}

// --- VectorStore implementation -------------------------------------------

// Upsert inserts or updates a single embedding. The id parameter is
// interpreted as the chunk_id (the natural key for embeddings). The
// fileID is accepted for interface compatibility but is not stored —
// the relationship is recomputed via the file_chunks join on demand.
func (s *PublicEmbeddingsStore) Upsert(ctx context.Context, id, fileID string, embedding []float32) error {
	if len(embedding) != 1024 {
		return fmt.Errorf("public_embeddings_store: embedding dim %d != 1024", len(embedding))
	}
	q := `INSERT INTO public.embeddings (chunk_id, embeddings, model)
		  VALUES ($1, $2, $3)
		  ON CONFLICT (chunk_id) DO UPDATE
		  SET embeddings = EXCLUDED.embeddings,
		      model = EXCLUDED.model`
	if _, err := s.pool.Exec(ctx, q, id, pgvector.NewVector(embedding), modelTag); err != nil {
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
		if len(it.Embedding) != 1024 {
			return fmt.Errorf("public_embeddings_store: embedding dim %d != 1024", len(it.Embedding))
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
		rows[i] = []any{it.ID, pgvector.NewVector(it.Embedding), modelTag}
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
		if _, err := tx.Exec(ctx, q, it.ID, pgvector.NewVector(it.Embedding), modelTag); err != nil {
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
	if len(embedding) != 1024 {
		return nil, fmt.Errorf("public_embeddings_store: embedding dim %d != 1024", len(embedding))
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

	q := fmt.Sprintf(`
		SELECT e.chunk_id, COALESCE(fc.file_id, ''), e.embeddings %s $1 AS distance
		FROM public.embeddings e
		LEFT JOIN public.file_chunks fc ON fc.chunk_id = e.chunk_id
		ORDER BY distance ASC
		LIMIT $2
	`, op)

	rows, err := s.pool.Query(ctx, q, pgvector.NewVector(embedding), limit)
	if err != nil {
		return nil, fmt.Errorf("public_embeddings_store: search: %w", err)
	}
	defer rows.Close()

	out := make([]VectorMatch, 0, limit)
	for rows.Next() {
		var id, fileID string
		var distance float64
		if err := rows.Scan(&id, &fileID, &distance); err != nil {
			return nil, fmt.Errorf("public_embeddings_store: scan search row: %w", err)
		}
		out = append(out, VectorMatch{
			ID:         id,
			FileID:     fileID,
			Similarity: distanceToSimilarityPg(distance, metric),
		})
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("public_embeddings_store: iterate search rows: %w", err)
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
		if len(q.Embedding) != 1024 {
			return nil, fmt.Errorf("public_embeddings_store: query dim %d != 1024", len(q.Embedding))
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

	byID := make(map[string][]VectorMatch, len(queries))
	for rows.Next() {
		var qid, id, fileID string
		var distance float64
		if err := rows.Scan(&qid, &id, &fileID, &distance); err != nil {
			return nil, fmt.Errorf("public_embeddings_store: scan batch row: %w", err)
		}
		byID[qid] = append(byID[qid], VectorMatch{
			ID:         id,
			FileID:     fileID,
			Similarity: distanceToSimilarityPg(distance, s.config.Metric),
		})
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("public_embeddings_store: iterate batch rows: %w", err)
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
