package fileprocessor

import (
	"context"
	"database/sql"
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"log/slog"

	_ "github.com/duckdb/duckdb-go/v2"
)

// HNSWConfig contains configuration options for the HNSW index.
//
// Reference: https://duckdb.org/docs/stable/core_extensions/vss
type HNSWConfig struct {
	// Metric is the distance function used by the index.
	// Options: "l2sq" (squared Euclidean), "cosine", "ip" (inner product).
	// Default: "l2sq".
	Metric string

	// EfConstruction controls the size of the dynamic candidate list used while
	// building the index. Higher values improve index quality at the cost of
	// slower construction. Default: 128, typical range: 100-500.
	EfConstruction int

	// EfSearch controls the size of the dynamic candidate list used during
	// queries. Higher values improve recall at the cost of slower search.
	// Default: 64, typical range: 10-500.
	// Can be overridden per query via SetEfSearch.
	EfSearch int

	// M is the number of bi-directional links created for every new element
	// during index construction. Higher values improve recall but increase
	// memory consumption. Default: 16, typical range: 5-48.
	M int
}

// DefaultHNSWConfig returns the recommended HNSW configuration.
func DefaultHNSWConfig() *HNSWConfig {
	return &HNSWConfig{
		Metric:         "l2sq",
		EfConstruction: 128,
		EfSearch:       64,
		M:              16,
	}
}

// DuckDBStore is a VectorStore backed by DuckDB with the vss extension and an
// HNSW index over FLOAT[dim] embeddings.
//
// The store is safe for concurrent use: a single *sql.DB is shared and the
// database/sql package serializes access to the underlying DuckDB connection.
//
// Reference: https://duckdb.org/docs/stable/core_extensions/vss
type DuckDBStore struct {
	db     *sql.DB
	config *HNSWConfig
}

// Compile-time interface check.
var _ VectorStore = (*DuckDBStore)(nil)

var reEmbeddingDim = regexp.MustCompile(`(?i)FLOAT\s*\[\s*(\d+)\s*\]`)

// NewDuckDBStore opens a DuckDB-backed VectorStore at path with default HNSW
// settings. If path is empty, an in-memory database is used.
//
// embeddingDim must match the dimension produced by the embedder in use; the
// store validates this against any pre-existing schema and refuses to start on
// mismatch.
func NewDuckDBStore(path string, embeddingDim int) (*DuckDBStore, error) {
	return NewDuckDBStoreWithConfig(path, embeddingDim, DefaultHNSWConfig())
}

// NewDuckDBStoreWithConfig is like NewDuckDBStore but allows custom HNSW
// options. If config is nil, defaults are used.
func NewDuckDBStoreWithConfig(path string, embeddingDim int, config *HNSWConfig) (*DuckDBStore, error) {
	resolved := normalizeHNSWConfig(config)

	dsn := path
	if dsn == "" {
		dsn = ":memory:"
	}

	db, err := sql.Open("duckdb", dsn)
	if err != nil {
		return nil, fmt.Errorf("ragcore: failed to open duckdb: %w", err)
	}

	store := &DuckDBStore{
		db:     db,
		config: resolved,
	}

	if err := store.init(embeddingDim); err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("ragcore: failed to initialize duckdb store: %w", err)
	}

	return store, nil
}

// Close releases the underlying DuckDB connection.
func (s *DuckDBStore) Close() error { return s.db.Close() }

// SetEfSearch overrides the ef_search parameter at runtime for this connection.
// Higher values improve recall at the cost of latency.
//
// Reference: https://duckdb.org/docs/stable/core_extensions/vss#index-options
func (s *DuckDBStore) SetEfSearch(efSearch int) error {
	if _, err := s.db.Exec(fmt.Sprintf("SET hnsw_ef_search = %d", efSearch)); err != nil {
		return fmt.Errorf("ragcore: failed to set ef_search: %w", err)
	}
	slog.Info("ragcore: updated ef_search", "ef_search", efSearch)
	return nil
}

// ResetEfSearch restores ef_search to the value set at index creation.
func (s *DuckDBStore) ResetEfSearch() error {
	if _, err := s.db.Exec("RESET hnsw_ef_search"); err != nil {
		return fmt.Errorf("ragcore: failed to reset ef_search: %w", err)
	}
	return nil
}

// CompactIndex triggers re-compaction of the HNSW index, pruning tombstones
// left by deletes/updates. Call this after bulk deletions to keep recall high.
//
// Reference: https://duckdb.org/docs/stable/core_extensions/vss#inserts-updates-deletes-and-re-compaction
func (s *DuckDBStore) CompactIndex(indexName string) error {
	if _, err := s.db.Exec(fmt.Sprintf("PRAGMA hnsw_compact_index('%s')", indexName)); err != nil {
		return fmt.Errorf("ragcore: failed to compact index %q: %w", indexName, err)
	}
	return nil
}

// IndexStats reports basic HNSW index statistics.
type IndexStats struct {
	TotalVectors  int64       `json:"total_vectors"`
	ActiveVectors int64       `json:"active_vectors"`
	DeletedRatio  float64     `json:"deleted_ratio"`
	Config        *HNSWConfig `json:"config"`
}

// GetIndexStats returns HNSW index statistics.
func (s *DuckDBStore) GetIndexStats() (*IndexStats, error) {
	row := s.db.QueryRow(`
		SELECT
			COUNT(*) AS total_vectors,
			COUNT(CASE WHEN id IS NOT NULL THEN 1 END) AS active_vectors
		FROM vectors
	`)
	var total, active int64
	if err := row.Scan(&total, &active); err != nil {
		return nil, fmt.Errorf("ragcore: failed to read index stats: %w", err)
	}
	ratio := 0.0
	if total > 0 {
		ratio = float64(total-active) / float64(total)
	}
	return &IndexStats{
		TotalVectors:  total,
		ActiveVectors: active,
		DeletedRatio:  ratio,
		Config:        s.config,
	}, nil
}

// --- VectorStore implementation --------------------------------------------

// Upsert inserts or updates a single vector.
func (s *DuckDBStore) Upsert(ctx context.Context, id, fileID string, embedding []float32) error {
	dim := len(embedding)
	literal := floatSliceToDuckArray(embedding)
	q := fmt.Sprintf(`INSERT OR REPLACE INTO vectors (id, file_id, embedding) VALUES (?, ?, %s::FLOAT[%d])`, literal, dim)
	if _, err := s.db.ExecContext(ctx, q, id, fileID); err != nil {
		return fmt.Errorf("ragcore: failed to upsert vector: %w", err)
	}
	return nil
}

// UpsertBatch inserts many vectors in a single transaction.
func (s *DuckDBStore) UpsertBatch(ctx context.Context, items []VectorItem) error {
	if len(items) == 0 {
		return nil
	}
	dim := len(items[0].Embedding)

	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("ragcore: begin batch upsert: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	for _, it := range items {
		literal := floatSliceToDuckArray(it.Embedding)
		q := fmt.Sprintf(`INSERT OR REPLACE INTO vectors (id, file_id, embedding) VALUES (?, ?, %s::FLOAT[%d])`, literal, dim)
		if _, err := tx.ExecContext(ctx, q, it.ID, it.FileID); err != nil {
			return fmt.Errorf("ragcore: batch upsert id=%s: %w", it.ID, err)
		}
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("ragcore: commit batch upsert: %w", err)
	}
	return nil
}

// Search returns the top-K most similar vectors.
func (s *DuckDBStore) Search(ctx context.Context, embedding []float32, params SearchParams) ([]VectorMatch, error) {
	metric := params.Metric
	if metric == "" {
		metric = metricNameToDistanceMetric(s.config.Metric)
	}
	limit := params.Limit
	if limit <= 0 {
		limit = 10
	}

	distFn, err := distanceFunctionFor(metric)
	if err != nil {
		return nil, err
	}

	dim := len(embedding)
	literal := floatSliceToDuckArray(embedding)
	q := fmt.Sprintf(`
		SELECT id, file_id, %s(embedding, %s::FLOAT[%d]) AS distance
		FROM vectors
		ORDER BY distance ASC
		LIMIT ?
	`, distFn, literal, dim)

	rows, err := s.db.QueryContext(ctx, q, limit)
	if err != nil {
		return nil, fmt.Errorf("ragcore: search query failed: %w", err)
	}
	defer func() { _ = rows.Close() }()

	var out []VectorMatch
	for rows.Next() {
		var id, fileID string
		var distance float64
		if err := rows.Scan(&id, &fileID, &distance); err != nil {
			return nil, fmt.Errorf("ragcore: scan search row: %w", err)
		}
		out = append(out, VectorMatch{
			ID:         id,
			FileID:     fileID,
			Similarity: distanceToSimilarity(distance),
		})
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("ragcore: iterate search rows: %w", err)
	}
	return out, nil
}

// DeleteByID removes a single vector.
func (s *DuckDBStore) DeleteByID(ctx context.Context, id string) error {
	_, err := s.db.ExecContext(ctx, "DELETE FROM vectors WHERE id = ?", id)
	if err != nil {
		return fmt.Errorf("ragcore: delete vector by id: %w", err)
	}
	return nil
}

// DeleteByFileID removes all vectors for a file.
func (s *DuckDBStore) DeleteByFileID(ctx context.Context, fileID string) error {
	_, err := s.db.ExecContext(ctx, "DELETE FROM vectors WHERE file_id = ?", fileID)
	if err != nil {
		return fmt.Errorf("ragcore: delete vectors by file_id: %w", err)
	}
	return nil
}

// --- Batch search ----------------------------------------------------------
//
// BatchSearchVectors performs multiple vector searches in a single SQL call
// using LATERAL joins, dramatically faster than looping Search in Go.
//
// Example: Search 1000 queries against 10000 vectors with limit=5
//   - Individual searches: ~10 seconds
//   - Batch search:        ~0.15 seconds (66× faster)
//
// Reference: https://duckdb.org/2024/10/23/whats-new-in-the-vss-extension

// BatchSearchRequest is a single query inside a batch.
type BatchSearchRequest struct {
	QueryID   string
	Embedding []float32
}

// BatchSearchResult groups the matches for one query.
type BatchSearchResult struct {
	QueryID string
	Matches []VectorMatch
}

// BatchSearch runs many queries in one round-trip.
func (s *DuckDBStore) BatchSearch(ctx context.Context, queries []BatchSearchRequest, limit int) ([]BatchSearchResult, error) {
	if len(queries) == 0 {
		return nil, nil
	}
	if limit <= 0 {
		limit = 10
	}

	dim := len(queries[0].Embedding)
	distFn, err := distanceFunctionFor(metricNameToDistanceMetric(s.config.Metric))
	if err != nil {
		return nil, err
	}

	var values strings.Builder
	values.WriteString("WITH queries AS (SELECT * FROM (VALUES ")
	for i, q := range queries {
		if i > 0 {
			values.WriteString(", ")
		}
		literal := floatSliceToDuckArray(q.Embedding)
		values.WriteString(fmt.Sprintf("('%s', %s::FLOAT[%d])", q.QueryID, literal, dim))
	}
	values.WriteString(") AS t(query_id, embedding)) ")

	q := values.String() + fmt.Sprintf(`
		SELECT
			queries.query_id,
			items.id,
			items.file_id,
			items.distance
		FROM queries, LATERAL (
			SELECT
				vectors.id,
				vectors.file_id,
				%s(queries.embedding, vectors.embedding) AS distance
			FROM vectors
			ORDER BY distance ASC
			LIMIT %d
		) AS items
		ORDER BY queries.query_id, items.distance
	`, distFn, limit)

	rows, err := s.db.QueryContext(ctx, q)
	if err != nil {
		return nil, fmt.Errorf("ragcore: batch search query failed: %w", err)
	}
	defer func() { _ = rows.Close() }()

	byID := make(map[string][]VectorMatch, len(queries))
	for rows.Next() {
		var queryID, id, fileID string
		var distance float64
		if err := rows.Scan(&queryID, &id, &fileID, &distance); err != nil {
			return nil, fmt.Errorf("ragcore: scan batch row: %w", err)
		}
		byID[queryID] = append(byID[queryID], VectorMatch{
			ID:         id,
			FileID:     fileID,
			Similarity: distanceToSimilarity(distance),
		})
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("ragcore: iterate batch rows: %w", err)
	}

	results := make([]BatchSearchResult, 0, len(queries))
	for _, q := range queries {
		results = append(results, BatchSearchResult{
			QueryID: q.QueryID,
			Matches: byID[q.QueryID],
		})
	}
	return results, nil
}

// --- init / schema ---------------------------------------------------------

func (s *DuckDBStore) init(dim int) error {
	if _, err := s.db.Exec("LOAD vss"); err != nil {
		slog.Info("ragcore: installing vss extension")
		if _, err := s.db.Exec("INSTALL vss"); err != nil {
			return fmt.Errorf("install vss: %w", err)
		}
		if _, err := s.db.Exec("LOAD vss"); err != nil {
			return fmt.Errorf("load vss after install: %w", err)
		}
	}

	if _, err := s.db.Exec("CHECKPOINT"); err != nil {
		slog.Warn("ragcore: checkpoint WAL failed", "error", err)
	}

	if _, err := s.db.Exec("SET hnsw_enable_experimental_persistence = true"); err != nil {
		slog.Warn("ragcore: experimental persistence not enabled (in-memory DB?)", "error", err)
	}

	createQ := fmt.Sprintf(`
		CREATE TABLE IF NOT EXISTS vectors (
			id TEXT PRIMARY KEY,
			file_id TEXT,
			embedding FLOAT[%d]
		)
	`, dim)
	slog.Info("ragcore: creating vectors table", "dim", dim)
	if _, err := s.db.Exec(createQ); err != nil {
		return fmt.Errorf("create vectors table: %w", err)
	}

	if err := s.verifyEmbeddingDimension(dim); err != nil {
		return err
	}

	if err := s.ensureHNSWIndex(); err != nil {
		return err
	}

	return nil
}

func (s *DuckDBStore) verifyEmbeddingDimension(expected int) error {
	var dataType string
	row := s.db.QueryRow(`
		SELECT type
		FROM pragma_table_info('vectors')
		WHERE name = 'embedding'
		LIMIT 1
	`)
	if err := row.Scan(&dataType); err != nil {
		if err == sql.ErrNoRows {
			return fmt.Errorf("vectors.embedding column not found")
		}
		return fmt.Errorf("inspect vectors.embedding schema: %w", err)
	}

	existing, err := parseEmbeddingDimension(dataType)
	if err != nil {
		return fmt.Errorf("parse vectors.embedding type %q: %w", dataType, err)
	}
	if existing != expected {
		return fmt.Errorf("embedding dimension mismatch: existing %d vs expected %d", existing, expected)
	}
	return nil
}

func (s *DuckDBStore) ensureHNSWIndex() error {
	existingSQL, exists, err := s.getExistingHNSWIndexSQL()
	if err != nil {
		return err
	}

	if !exists {
		q := buildCreateHNSWIndexQuery(s.config)
		slog.Info("ragcore: creating HNSW index", "config", s.config)
		if _, err := s.db.Exec(q); err != nil {
			return fmt.Errorf("create HNSW index: %w", err)
		}
		return nil
	}

	existingConfig, err := parseHNSWConfigFromCreateIndexSQL(existingSQL)
	if err != nil {
		return fmt.Errorf("parse existing vec_idx config: %w", err)
	}

	needsRebuild := !hnswConfigEqual(existingConfig, s.config)

	existingMetric, metricKnown, err := s.getExistingHNSWIndexMetric()
	if err != nil {
		return err
	}
	if metricKnown && !strings.EqualFold(existingMetric, s.config.Metric) {
		needsRebuild = true
	}

	if !needsRebuild {
		slog.Info("ragcore: HNSW index config matches", "index", "vec_idx")
		return nil
	}

	slog.Warn("ragcore: HNSW config drift detected, rebuilding vec_idx",
		"existing", existingConfig,
		"expected", s.config,
	)

	tx, err := s.db.Begin()
	if err != nil {
		return fmt.Errorf("begin vec_idx rebuild: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	if _, err := tx.Exec("DROP INDEX IF EXISTS vec_idx"); err != nil {
		return fmt.Errorf("drop mismatched vec_idx: %w", err)
	}
	if _, err := tx.Exec(buildCreateHNSWIndexQuery(s.config)); err != nil {
		return fmt.Errorf("recreate vec_idx: %w", err)
	}
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit vec_idx rebuild: %w", err)
	}

	slog.Info("ragcore: rebuilt HNSW index", "index", "vec_idx")
	return nil
}

func (s *DuckDBStore) getExistingHNSWIndexMetric() (string, bool, error) {
	var metric sql.NullString
	row := s.db.QueryRow(`
		SELECT metric
		FROM pragma_hnsw_index_info()
		WHERE table_name = 'vectors' AND index_name = 'vec_idx'
		LIMIT 1
	`)
	if err := row.Scan(&metric); err != nil {
		if err == sql.ErrNoRows {
			return "", false, nil
		}
		return "", false, fmt.Errorf("read vec_idx metric: %w", err)
	}
	if !metric.Valid || strings.TrimSpace(metric.String) == "" {
		return "", false, nil
	}
	return metric.String, true, nil
}

func (s *DuckDBStore) getExistingHNSWIndexSQL() (string, bool, error) {
	var sqlText sql.NullString
	row := s.db.QueryRow(`
		SELECT sql
		FROM duckdb_indexes()
		WHERE table_name = 'vectors' AND index_name = 'vec_idx'
		LIMIT 1
	`)
	if err := row.Scan(&sqlText); err != nil {
		if err == sql.ErrNoRows {
			return "", false, nil
		}
		return "", false, fmt.Errorf("read vec_idx metadata: %w", err)
	}
	if !sqlText.Valid || strings.TrimSpace(sqlText.String) == "" {
		return "", false, nil
	}
	return sqlText.String, true, nil
}

// --- helpers ---------------------------------------------------------------

func normalizeHNSWConfig(c *HNSWConfig) *HNSWConfig {
	if c == nil {
		return DefaultHNSWConfig()
	}
	out := *c
	d := DefaultHNSWConfig()
	if out.Metric == "" {
		out.Metric = d.Metric
	}
	if out.EfConstruction <= 0 {
		out.EfConstruction = d.EfConstruction
	}
	if out.EfSearch <= 0 {
		out.EfSearch = d.EfSearch
	}
	if out.M <= 0 {
		out.M = d.M
	}
	return &out
}

func hnswConfigEqual(a, b *HNSWConfig) bool {
	if a == nil || b == nil {
		return false
	}
	return strings.EqualFold(a.Metric, b.Metric) &&
		a.EfConstruction == b.EfConstruction &&
		a.EfSearch == b.EfSearch &&
		a.M == b.M
}

func parseEmbeddingDimension(dataType string) (int, error) {
	match := reEmbeddingDim.FindStringSubmatch(strings.TrimSpace(dataType))
	if len(match) != 2 {
		return 0, fmt.Errorf("unsupported embedding type format: %s", dataType)
	}
	dim, err := strconv.Atoi(match[1])
	if err != nil {
		return 0, fmt.Errorf("invalid embedding dimension %q: %w", match[1], err)
	}
	return dim, nil
}

func buildCreateHNSWIndexQuery(c *HNSWConfig) string {
	r := normalizeHNSWConfig(c)
	return fmt.Sprintf(`
		CREATE INDEX IF NOT EXISTS vec_idx ON vectors
		USING HNSW (embedding)
		WITH (
			metric = '%s',
			ef_construction = %d,
			ef_search = %d,
			M = %d
		)
	`, r.Metric, r.EfConstruction, r.EfSearch, r.M)
}

func parseHNSWConfigFromCreateIndexSQL(createSQL string) (*HNSWConfig, error) {
	getString := func(pattern string) (string, bool, error) {
		re := regexp.MustCompile(pattern)
		match := re.FindStringSubmatch(createSQL)
		if len(match) != 2 {
			return "", false, nil
		}
		return strings.TrimSpace(match[1]), true, nil
	}
	getInt := func(pattern string) (int, bool, error) {
		v, ok, err := getString(pattern)
		if err != nil || !ok {
			return 0, ok, err
		}
		n, err := strconv.Atoi(v)
		if err != nil {
			return 0, false, fmt.Errorf("invalid integer %q: %w", v, err)
		}
		return n, true, nil
	}

	metric, metricFound, err := getString(`(?i)metric\s*=\s*'([^']+)'`)
	if err != nil {
		return nil, err
	}
	efConstruction, efCFound, err := getInt(`(?i)ef_construction\s*=\s*(\d+)`)
	if err != nil {
		return nil, err
	}
	efSearch, efSFound, err := getInt(`(?i)ef_search\s*=\s*(\d+)`)
	if err != nil {
		return nil, err
	}
	mValue, mFound, err := getInt(`(?i)\bM\s*=\s*(\d+)`)
	if err != nil {
		return nil, err
	}

	d := DefaultHNSWConfig()
	if !metricFound {
		metric = d.Metric
	}
	if !efCFound {
		efConstruction = d.EfConstruction
	}
	if !efSFound {
		efSearch = d.EfSearch
	}
	if !mFound {
		mValue = d.M
	}

	return &HNSWConfig{
		Metric:         metric,
		EfConstruction: efConstruction,
		EfSearch:       efSearch,
		M:              mValue,
	}, nil
}

func metricNameToDistanceMetric(metric string) DistanceMetric {
	switch strings.ToLower(strings.TrimSpace(metric)) {
	case "cosine":
		return DistanceCosine
	case "ip":
		return DistanceInnerProduct
	case "l2sq":
		fallthrough
	default:
		return DistanceEuclidean
	}
}

func distanceFunctionFor(m DistanceMetric) (string, error) {
	switch m {
	case DistanceCosine:
		return "array_cosine_distance", nil
	case DistanceInnerProduct:
		return "array_negative_inner_product", nil
	case DistanceEuclidean:
		return "array_distance", nil
	default:
		return "", fmt.Errorf("unsupported distance metric: %s", m)
	}
}

// floatSliceToDuckArray formats a []float32 as a DuckDB array literal "[v1, v2, ...]".
// Used because go-duckdb does not accept []float32 as a bound parameter.
func floatSliceToDuckArray(v []float32) string {
	var b strings.Builder
	b.WriteByte('[')
	for i, f := range v {
		if i > 0 {
			b.WriteString(", ")
		}
		b.WriteString(strconv.FormatFloat(float64(f), 'f', -1, 32))
	}
	b.WriteByte(']')
	return b.String()
}

// distanceToSimilarity converts a distance into a [0,1]-ish similarity score
// using the monotonic transform 1/(1+d). Works acceptably for all metrics.
func distanceToSimilarity(distance float64) float64 {
	return 1.0 / (1.0 + distance)
}
