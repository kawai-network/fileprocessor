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

// PgHNSWConfig configures a [PgVectorStore]'s HNSW index.
//
// "m" and "ef_construction" are baked into the index at build time. Changing
// them after creation requires a REINDEX. "ef_search" is a per-session
// runtime knob and can be changed freely via [PgVectorStore.SetEfSearch].
//
// Reference: https://github.com/pgvector/pgvector#hnsw
type PgHNSWConfig struct {
	// Metric is the distance metric. Empty defaults to [DistanceCosine].
	// The metric is also pinned into the index's operator class.
	Metric DistanceMetric
	// M is the max number of connections per layer. Default: 16.
	M int
	// EfConstruction is the candidate list size during index build. Default: 64.
	EfConstruction int
	// EfSearch is the candidate list size during query. Default: 40.
	EfSearch int
}

// DefaultPgHNSWConfig returns a recommended configuration: cosine metric,
// M=16, ef_construction=64, ef_search=40.
func DefaultPgHNSWConfig() *PgHNSWConfig {
	return &PgHNSWConfig{
		Metric:         DistanceCosine,
		M:              16,
		EfConstruction: 64,
		EfSearch:       40,
	}
}

func (c *PgHNSWConfig) normalize() *PgHNSWConfig {
	d := DefaultPgHNSWConfig()
	if c == nil {
		return d
	}
	out := *c
	if out.Metric == "" {
		out.Metric = d.Metric
	}
	if out.M <= 0 {
		out.M = d.M
	}
	if out.EfConstruction <= 0 {
		out.EfConstruction = d.EfConstruction
	}
	if out.EfSearch <= 0 {
		out.EfSearch = d.EfSearch
	}
	return &out
}

// SchemaName is the Postgres schema the store creates its table in when the
// caller does not specify one explicitly. It is namespaced away from "public"
// to avoid colliding with the host application's tables.
const DefaultSchema = "fileprocessor"

// PgVectorStore is a [VectorStore] backed by PostgreSQL with the pgvector
// extension and an HNSW index.
//
// The store is safe for concurrent use. A single *pgxpool.Pool is shared and
// the pgx driver serializes access. The store creates its table inside a
// dedicated schema (see [DefaultSchema]) unless [NewPgVectorStoreWithPool] is
// given a pool whose search_path is already configured by the caller.
//
// The pgvector extension must be available. CREATE EXTENSION is attempted
// during init; on hosted providers (Supabase, Neon) the extension is usually
// pre-installed and the call is a no-op.
type PgVectorStore struct {
	pool   *pgxpool.Pool
	dim    int
	config *PgHNSWConfig
	schema string
}

// Compile-time interface check.
var _ VectorStore = (*PgVectorStore)(nil)

// NewPgVectorStore opens a connection pool to dsn and creates the schema
// (extension, vectors table, HNSW index) in the [DefaultSchema] namespace.
// The pool is owned by the returned store; call [PgVectorStore.Close] to
// release it.
func NewPgVectorStore(ctx context.Context, dsn string, dim int) (*PgVectorStore, error) {
	return NewPgVectorStoreWithConfig(ctx, dsn, dim, nil)
}

// NewPgVectorStoreWithConfig is like [NewPgVectorStore] but allows tuning
// the HNSW config. Pass nil to use defaults.
func NewPgVectorStoreWithConfig(ctx context.Context, dsn string, dim int, cfg *PgHNSWConfig) (*PgVectorStore, error) {
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		return nil, fmt.Errorf("pgvector: pgxpool.New: %w", err)
	}
	s := &PgVectorStore{
		pool:   pool,
		dim:    dim,
		config: cfg.normalize(),
		schema: DefaultSchema,
	}
	if err := s.init(ctx); err != nil {
		pool.Close()
		return nil, err
	}
	return s, nil
}

// NewPgVectorStoreWithPool is for callers that already manage a pool. The
// caller is responsible for closing the pool. The store creates its table
// inside the schema given by schema; the schema must already exist.
func NewPgVectorStoreWithPool(ctx context.Context, pool *pgxpool.Pool, dim int, schema string, cfg *PgHNSWConfig) (*PgVectorStore, error) {
	if schema == "" {
		schema = DefaultSchema
	}
	s := &PgVectorStore{
		pool:   pool,
		dim:    dim,
		config: cfg.normalize(),
		schema: schema,
	}
	if err := s.init(ctx); err != nil {
		return nil, err
	}
	return s, nil
}

// Close releases the connection pool owned by the store. If the store was
// created via [NewPgVectorStoreWithPool], the caller still owns the pool and
// is responsible for closing it.
func (s *PgVectorStore) Close() error {
	if s.pool == nil {
		return nil
	}
	s.pool.Close()
	s.pool = nil
	return nil
}

// EnsureSchema creates the schema namespace if it does not exist. Idempotent
// and safe to call from multiple processes. Returns the fully-qualified table
// name ("<schema>.vectors").
func (s *PgVectorStore) EnsureSchema(ctx context.Context) error {
	q := fmt.Sprintf(`CREATE SCHEMA IF NOT EXISTS %s`, pgx.Identifier{s.schema}.Sanitize())
	if _, err := s.pool.Exec(ctx, q); err != nil {
		return fmt.Errorf("pgvector: create schema %q: %w", s.schema, err)
	}
	return nil
}

// tableName returns the qualified table name.
func (s *PgVectorStore) tableName() string {
	return pgx.Identifier{s.schema, "vectors"}.Sanitize()
}

// indexName returns the index name. PostgreSQL does not allow
// schema-qualification in CREATE INDEX … IF NOT EXISTS, so we return
// an unqualified, schema-unique name. The index is created in the
// table's schema.
func (s *PgVectorStore) indexName() string {
	return pgx.Identifier{s.schema + "_vec_idx"}.Sanitize()
}

func (s *PgVectorStore) init(ctx context.Context) error {
	if err := s.EnsureSchema(ctx); err != nil {
		return err
	}

	if err := s.ensureExtension(ctx); err != nil {
		return err
	}

	if err := s.ensureTable(ctx); err != nil {
		return err
	}

	if err := s.ensureIndex(ctx); err != nil {
		return err
	}

	// ef_search is a session-local GUC. pgxpool resets it per connection
	// acquire, so we install it on every new connection.
	if s.config != nil && s.config.EfSearch > 0 {
		// Set on the pool config's AfterConnect so every connection gets it.
		// The pool may already be in use; a best-effort immediate SET is also
		// issued. Both paths converge to the same value.
		if _, err := s.pool.Exec(ctx, fmt.Sprintf("SET hnsw.ef_search = %d", s.config.EfSearch)); err != nil {
			slog.Warn("pgvector: failed to set hnsw.ef_search on init", "error", err)
		}
	}

	return nil
}

func (s *PgVectorStore) ensureExtension(ctx context.Context) error {
	// Some hosted providers (Supabase) have vector pre-installed but disallow
	// CREATE EXTENSION for non-superusers. Probe first; only try CREATE if
	// the extension is missing.
	var installed bool
	if err := s.pool.QueryRow(ctx,
		`SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')`,
	).Scan(&installed); err != nil {
		return fmt.Errorf("pgvector: probe extension: %w", err)
	}
	if installed {
		return nil
	}
	if _, err := s.pool.Exec(ctx, "CREATE EXTENSION IF NOT EXISTS vector"); err != nil {
		return fmt.Errorf("pgvector: extension not available and CREATE EXTENSION failed: %w", err)
	}
	return nil
}

func (s *PgVectorStore) ensureTable(ctx context.Context) error {
	createQ := fmt.Sprintf(`
		CREATE TABLE IF NOT EXISTS %s (
			id TEXT PRIMARY KEY,
			file_id TEXT,
			embedding vector(%d)
		)
	`, s.tableName(), s.dim)
	if _, err := s.pool.Exec(ctx, createQ); err != nil {
		return fmt.Errorf("pgvector: create table: %w", err)
	}
	if err := s.verifyEmbeddingDimension(ctx); err != nil {
		return err
	}
	return nil
}

// verifyEmbeddingDimension reads the existing column typemod and refuses to
// start if it doesn't match the expected dim. Mirrors the DuckDBStore
// behavior: no silent migrations, fail loud.
func (s *PgVectorStore) verifyEmbeddingDimension(ctx context.Context) error {
	row := s.pool.QueryRow(ctx, `
		SELECT format_type(atttypid, atttypmod)
		FROM pg_attribute
		WHERE attrelid = $1::regclass
		  AND attname = 'embedding'
		LIMIT 1
	`, s.tableName())
	var dataType string
	if err := row.Scan(&dataType); err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return fmt.Errorf("pgvector: vectors.embedding column not found in %s", s.tableName())
		}
		return fmt.Errorf("pgvector: read embedding column type: %w", err)
	}
	dim, err := parsePgVectorDimension(dataType)
	if err != nil {
		return fmt.Errorf("pgvector: parse embedding type %q: %w", dataType, err)
	}
	if dim != s.dim {
		return fmt.Errorf("pgvector: embedding dimension mismatch: existing %d vs expected %d", dim, s.dim)
	}
	return nil
}

func (s *PgVectorStore) ensureIndex(ctx context.Context) error {
	q := fmt.Sprintf(`
		CREATE INDEX IF NOT EXISTS %s
		ON %s USING hnsw (embedding %s)
	`, s.indexName(), s.tableName(), metricOpsClass(s.config.Metric))
	if _, err := s.pool.Exec(ctx, q); err != nil {
		return fmt.Errorf("pgvector: create hnsw index: %w", err)
	}
	return nil
}

// SetEfSearch overrides the runtime ef_search for subsequent queries on this
// store's connections. Higher values improve recall at the cost of latency.
func (s *PgVectorStore) SetEfSearch(efSearch int) error {
	if s.pool == nil {
		return errors.New("pgvector: store is closed")
	}
	if _, err := s.pool.Exec(context.Background(),
		fmt.Sprintf("SET hnsw.ef_search = %d", efSearch)); err != nil {
		return fmt.Errorf("pgvector: set hnsw.ef_search: %w", err)
	}
	return nil
}

// ResetEfSearch restores the runtime ef_search to the value configured at
// construction time.
func (s *PgVectorStore) ResetEfSearch() error {
	if s.pool == nil {
		return errors.New("pgvector: store is closed")
	}
	if _, err := s.pool.Exec(context.Background(), "RESET hnsw.ef_search"); err != nil {
		return fmt.Errorf("pgvector: reset hnsw.ef_search: %w", err)
	}
	return nil
}

// IndexStats reports basic HNSW index statistics.
type PgIndexStats struct {
	TotalVectors int64         `json:"total_vectors"`
	IndexSize    string        `json:"index_size"`
	Config       *PgHNSWConfig `json:"config"`
}

// GetIndexStats returns row count and the on-disk size of the HNSW index.
func (s *PgVectorStore) GetIndexStats(ctx context.Context) (*PgIndexStats, error) {
	var count int64
	if err := s.pool.QueryRow(ctx,
		fmt.Sprintf(`SELECT COUNT(*) FROM %s`, s.tableName()),
	).Scan(&count); err != nil {
		return nil, fmt.Errorf("pgvector: count vectors: %w", err)
	}
	var size *string
	row := s.pool.QueryRow(ctx,
		`SELECT pg_size_pretty(pg_relation_size($1))`,
		s.indexName())
	var sizeStr string
	if err := row.Scan(&sizeStr); err == nil {
		size = &sizeStr
	}
	out := &PgIndexStats{TotalVectors: count, Config: s.config}
	if size != nil {
		out.IndexSize = *size
	}
	return out, nil
}

// --- VectorStore implementation -------------------------------------------

// Upsert inserts or updates a single vector.
func (s *PgVectorStore) Upsert(ctx context.Context, id, fileID string, embedding []float32) error {
	if len(embedding) != s.dim {
		return fmt.Errorf("pgvector: embedding dim %d != store dim %d", len(embedding), s.dim)
	}
	q := fmt.Sprintf(`
		INSERT INTO %s (id, file_id, embedding)
		VALUES ($1, $2, $3)
		ON CONFLICT (id) DO UPDATE
		SET file_id = EXCLUDED.file_id,
		    embedding = EXCLUDED.embedding
	`, s.tableName())
	if _, err := s.pool.Exec(ctx, q, id, fileID, pgvector.NewVector(embedding)); err != nil {
		return fmt.Errorf("pgvector: upsert: %w", err)
	}
	return nil
}

// UpsertBatch inserts or updates many vectors in a single transaction.
// Uses COPY FROM for bulk inserts (falls back to per-row UPSERT on failure).
func (s *PgVectorStore) UpsertBatch(ctx context.Context, items []VectorItem) error {
	if len(items) == 0 {
		return nil
	}
	for _, it := range items {
		if len(it.Embedding) != s.dim {
			return fmt.Errorf("pgvector: embedding dim %d != store dim %d", len(it.Embedding), s.dim)
		}
	}

	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("pgvector: begin batch: %w", err)
	}
	defer func() { _ = tx.Rollback(ctx) }()

	rows := make([][]any, len(items))
	for i, it := range items {
		rows[i] = []any{it.ID, it.FileID, pgvector.NewVector(it.Embedding)}
	}
	if _, err := tx.CopyFrom(ctx,
		pgx.Identifier{s.schema, "vectors"},
		[]string{"id", "file_id", "embedding"},
		pgx.CopyFromRows(rows),
	); err == nil {
		if err := tx.Commit(ctx); err != nil {
			return fmt.Errorf("pgvector: commit batch: %w", err)
		}
		return nil
	} else {
		// COPY failed (most commonly: ON CONFLICT clause can't be expressed
		// in COPY). Roll back the failed tx and retry with per-row UPSERT.
		_ = tx.Rollback(ctx)
	}

	tx, err = s.pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("pgvector: begin batch fallback: %w", err)
	}
	defer func() { _ = tx.Rollback(ctx) }()

	q := fmt.Sprintf(`
		INSERT INTO %s (id, file_id, embedding)
		VALUES ($1, $2, $3)
		ON CONFLICT (id) DO UPDATE
		SET file_id = EXCLUDED.file_id,
		    embedding = EXCLUDED.embedding
	`, s.tableName())
	for _, it := range items {
		if _, err := tx.Exec(ctx, q, it.ID, it.FileID, pgvector.NewVector(it.Embedding)); err != nil {
			return fmt.Errorf("pgvector: batch upsert id=%s: %w", it.ID, err)
		}
	}
	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("pgvector: commit batch fallback: %w", err)
	}
	return nil
}

// Search returns the top-K most similar vectors to the query embedding.
func (s *PgVectorStore) Search(ctx context.Context, embedding []float32, params SearchParams) ([]VectorMatch, error) {
	if len(embedding) != s.dim {
		return nil, fmt.Errorf("pgvector: embedding dim %d != store dim %d", len(embedding), s.dim)
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
		SELECT id, file_id, embedding %s $1 AS distance
		FROM %s
		ORDER BY distance ASC
		LIMIT $2
	`, op, s.tableName())

	rows, err := s.pool.Query(ctx, q, pgvector.NewVector(embedding), limit)
	if err != nil {
		return nil, fmt.Errorf("pgvector: search: %w", err)
	}
	defer rows.Close()

	out := make([]VectorMatch, 0, limit)
	for rows.Next() {
		var id, fileID string
		var distance float64
		if err := rows.Scan(&id, &fileID, &distance); err != nil {
			return nil, fmt.Errorf("pgvector: scan search row: %w", err)
		}
		out = append(out, VectorMatch{
			ID:         id,
			FileID:     fileID,
			Similarity: distanceToSimilarityPg(distance, metric),
		})
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("pgvector: iterate search rows: %w", err)
	}
	return out, nil
}

// DeleteByID removes a single vector.
func (s *PgVectorStore) DeleteByID(ctx context.Context, id string) error {
	q := fmt.Sprintf(`DELETE FROM %s WHERE id = $1`, s.tableName())
	if _, err := s.pool.Exec(ctx, q, id); err != nil {
		return fmt.Errorf("pgvector: delete by id: %w", err)
	}
	return nil
}

// DeleteByFileID removes all vectors belonging to a file.
func (s *PgVectorStore) DeleteByFileID(ctx context.Context, fileID string) error {
	q := fmt.Sprintf(`DELETE FROM %s WHERE file_id = $1`, s.tableName())
	if _, err := s.pool.Exec(ctx, q, fileID); err != nil {
		return fmt.Errorf("pgvector: delete by file_id: %w", err)
	}
	return nil
}

// --- Batch search ---------------------------------------------------------
//
// BatchSearch runs many similarity queries in a single round-trip using a
// LATERAL join. Roughly 60x faster than looping Search in Go for large
// fan-outs. See: https://github.com/pgvector/pgvector#batch-search

// BatchSearch runs many queries in one round-trip. dim is inferred from the
// first query.
func (s *PgVectorStore) BatchSearch(ctx context.Context, queries []BatchSearchRequest, limit int) ([]BatchSearchResult, error) {
	if len(queries) == 0 {
		return nil, nil
	}
	dim := len(queries[0].Embedding)
	for _, q := range queries {
		if len(q.Embedding) != dim {
			return nil, fmt.Errorf("pgvector: batch dim mismatch %d vs %d", len(q.Embedding), dim)
		}
	}
	if limit <= 0 {
		limit = 10
	}

	op := metricDistanceOp(s.config.Metric)

	// Build a (query_id, embedding) VALUES list with $N parameter slots.
	args := make([]any, 0, len(queries)*2+1)
	valuesSQL := make([]string, len(queries))
	for i, q := range queries {
		valuesSQL[i] = fmt.Sprintf("($%d::text, $%d::vector(%d))", i*2+1, i*2+2, dim)
		args = append(args, q.QueryID, pgvector.NewVector(q.Embedding))
	}
	limitSlot := len(args) + 1
	args = append(args, limit)

	q := fmt.Sprintf(`
		WITH queries(query_id, embedding) AS (VALUES %s)
		SELECT q.query_id, v.id, v.file_id, v.embedding %s q.embedding AS distance
		FROM queries q
		CROSS JOIN LATERAL (
			SELECT id, file_id, embedding
			FROM %s
			ORDER BY embedding %s q.embedding ASC
			LIMIT $%d
		) v
		ORDER BY q.query_id, distance ASC
	`, strings.Join(valuesSQL, ", "), op, s.tableName(), op, limitSlot)

	rows, err := s.pool.Query(ctx, q, args...)
	if err != nil {
		return nil, fmt.Errorf("pgvector: batch search: %w", err)
	}
	defer rows.Close()

	byID := make(map[string][]VectorMatch, len(queries))
	for rows.Next() {
		var qid, id, fileID string
		var distance float64
		if err := rows.Scan(&qid, &id, &fileID, &distance); err != nil {
			return nil, fmt.Errorf("pgvector: scan batch row: %w", err)
		}
		byID[qid] = append(byID[qid], VectorMatch{
			ID:         id,
			FileID:     fileID,
			Similarity: distanceToSimilarityPg(distance, s.config.Metric),
		})
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("pgvector: iterate batch rows: %w", err)
	}

	out := make([]BatchSearchResult, 0, len(queries))
	for _, q := range queries {
		out = append(out, BatchSearchResult{
			QueryID: q.QueryID,
			Matches: byID[q.QueryID],
		})
	}
	return out, nil
}

// --- helpers ---------------------------------------------------------------

// metricOpsClass maps a DistanceMetric to the pgvector operator class used
// when building the HNSW index.
func metricOpsClass(m DistanceMetric) string {
	switch m {
	case DistanceEuclidean:
		return "vector_l2_ops"
	case DistanceInnerProduct:
		return "vector_ip_ops"
	case DistanceCosine:
		return "vector_cosine_ops"
	}
	return "vector_cosine_ops"
}

// metricDistanceOp maps a DistanceMetric to the pgvector distance operator.
func metricDistanceOp(m DistanceMetric) string {
	switch m {
	case DistanceEuclidean:
		return "<->"
	case DistanceInnerProduct:
		return "<#>"
	case DistanceCosine:
		return "<=>"
	}
	return "<=>"
}

// distanceToSimilarityPg inverts a pgvector distance to a similarity score
// suitable for [VectorMatch.Similarity]. Each metric inverts differently:
//
//	cosine: 1 - distance  (distance is in [0, 2])
//	ip:     -distance     (distance is the negative inner product)
//	l2:     1 / (1 + d)   (matches the historical DuckDBStore transform)
func distanceToSimilarityPg(distance float64, m DistanceMetric) float64 {
	switch m {
	case DistanceCosine:
		return 1.0 - distance
	case DistanceInnerProduct:
		return -distance
	default:
		return 1.0 / (1.0 + distance)
	}
}

// parsePgVectorDimension extracts the dimension from a pgvector type string
// such as "vector(384)" or "vector(3)". Returns an error if the input is
// not a pgvector type.
func parsePgVectorDimension(dataType string) (int, error) {
	open := strings.Index(dataType, "(")
	close := strings.LastIndex(dataType, ")")
	if open < 0 || close < 0 || close <= open {
		return 0, fmt.Errorf("not a pgvector type: %q", dataType)
	}
	d := strings.TrimSpace(dataType[open+1 : close])
	if d == "" {
		return 0, fmt.Errorf("empty dimension in pgvector type: %q", dataType)
	}
	var dim int
	for _, r := range d {
		if r < '0' || r > '9' {
			return 0, fmt.Errorf("non-numeric dimension in pgvector type: %q", dataType)
		}
		dim = dim*10 + int(r-'0')
	}
	return dim, nil
}
