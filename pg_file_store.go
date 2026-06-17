package fileprocessor

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// PostgresFileStoreOwner identifies the tenant that owns rows created by a
// [PostgresFileStore]. The existing lobehub schema stamps every row with
// user_id (required) and workspace_id / client_id (optional).
type PostgresFileStoreOwner struct {
	UserID      string
	WorkspaceID string // may be empty
	ClientID    string // may be empty
}

// PostgresFileStore implements [FileStore] backed by the lobehub public
// schema (files, documents, chunks, file_chunks, document_chunks tables).
//
// It is designed to work alongside [PgVectorStore] (vectors live in the
// "fileprocessor" schema) and [Processor].
//
// The store uses the existing FK chain:
//
//	files ──→ file_chunks ──→ chunks
//	          document_chunks ──→ chunks
//	          embeddings ──→ chunks (NOT written here — use VectorStore)
//
// Chunk stats (count + status) are stored inside the files.metadata jsonb
// key "chunk_stats" to avoid a schema migration.
//
// Tenancy: every row is stamped with the Owner's user_id (and optionally
// workspace_id / client_id). The Owner is configured at construction and
// applies to all operations on this store instance.
//
// DeleteFile performs a manual cascade:
//  1. Deletes document_chunks rows for the file's documents.
//  2. Deletes file_chunks rows (cascades to chunks, embeddings).
//  3. Deletes documents.
//  4. Deletes the file row.
//
// [PostgresFileStore.ChunkStore] returns a [PostgresChunkStore] adapter that
// implements the [ChunkStore] interface for use with [RAGProcessor].
type PostgresFileStore struct {
	pool  *pgxpool.Pool
	owner PostgresFileStoreOwner
}

// Compile-time interface check.
var _ FileStore = (*PostgresFileStore)(nil)

// NewPostgresFileStore creates a store against dsn. The owner is stamped on
// every inserted row.
func NewPostgresFileStore(ctx context.Context, dsn string, owner PostgresFileStoreOwner) (*PostgresFileStore, error) {
	if owner.UserID == "" {
		return nil, errors.New("fileprocessor: PostgresFileStoreOwner.UserID is required")
	}
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		return nil, fmt.Errorf("pg_file_store: pgxpool.New: %w", err)
	}
	return &PostgresFileStore{pool: pool, owner: owner}, nil
}

// NewPostgresFileStoreWithPool is for callers that already manage a pool.
// The caller owns the pool's lifecycle.
func NewPostgresFileStoreWithPool(pool *pgxpool.Pool, owner PostgresFileStoreOwner) (*PostgresFileStore, error) {
	if owner.UserID == "" {
		return nil, errors.New("fileprocessor: PostgresFileStoreOwner.UserID is required")
	}
	return &PostgresFileStore{pool: pool, owner: owner}, nil
}

// Close releases the underlying connection pool. A store created via
// [NewPostgresFileStore] owns its pool. Callers that created the pool
// externally and used [NewPostgresFileStoreWithPool] must manage the pool
// lifecycle themselves.
func (s *PostgresFileStore) Close() error {
	if s.pool != nil {
		s.pool.Close()
		s.pool = nil
	}
	return nil
}

// ChunkStore returns a [ChunkStore] adapter backed by the same pool. Use it
// to wire [RAGProcessor]:
//
//	fileStore, _ := fileprocessor.NewPostgresFileStore(...)
//	rag := fileprocessor.NewRAGProcessor(fileStore.ChunkStore(), vectorStore, embedder, nil)
//	proc, _ := fileprocessor.New(fileprocessor.Config{
//	    FileStore:    fileStore,
//	    RAGProcessor: rag,
//	    ...
//	})
func (s *PostgresFileStore) ChunkStore() *PostgresChunkStore {
	return &PostgresChunkStore{store: s}
}

// PostgresChunkStore adapts a [PostgresFileStore] to the [ChunkStore]
// interface. The two interfaces have method-name collisions (CreateChunk,
// GetFile, UpdateFileChunkStats) with incompatible signatures, so a single
// struct cannot implement both. This adapter bridges them.
type PostgresChunkStore struct {
	store *PostgresFileStore
}

// Compile-time interface check.
var _ ChunkStore = (*PostgresChunkStore)(nil)

// ============================================================================
// FileStore implementation
// ============================================================================

func (s *PostgresFileStore) CreateFile(ctx context.Context, rec FileRecord) (string, error) {
	id := rec.ID
	if id == "" {
		id = uuid.New().String()
	}

	now := time.Now()
	// NOTE: We deliberately do NOT write rec.Hash to files.file_hash. That
	// column FKs to global_files.hash_id (the lobehub dedup table). Writing
	// a content hash here would require a matching global_files row first,
	// which is the host app's responsibility. We surface the hash in
	// files.metadata->'file_hash' instead so callers can still inspect it.
	metaArg, err := jsonbArg(rec.Metadata, "file_hash", rec.Hash)
	if err != nil {
		return "", fmt.Errorf("marshal metadata: %w", err)
	}

	q := `INSERT INTO files (
		id, user_id, file_type, name, size, url, source,
		metadata, created_at, updated_at, accessed_at, client_id, workspace_id
	) VALUES (
		$1, $2, $3, $4, $5, $6, $7,
		$8, $9, $9, $9, $10, NULLIF($11, '')
	) ON CONFLICT (id) DO UPDATE SET
		file_type = EXCLUDED.file_type,
		name      = EXCLUDED.name,
		size      = EXCLUDED.size,
		url       = EXCLUDED.url,
		source    = EXCLUDED.source,
		metadata  = EXCLUDED.metadata,
		updated_at = EXCLUDED.updated_at`

	if _, err := s.pool.Exec(ctx, q,
		id, s.owner.UserID, rec.FileType, rec.Name, rec.Size, rec.URL,
		rec.Source, metaArg, now, s.owner.ClientID, s.owner.WorkspaceID,
	); err != nil {
		return "", fmt.Errorf("pg_file_store: CreateFile: %w", err)
	}
	return id, nil
}

func (s *PostgresFileStore) CreateDocument(ctx context.Context, rec DocumentRecord) (string, error) {
	id := rec.ID
	if id == "" {
		id = uuid.New().String()
	}

	metaArg, err := jsonbArg(rec.Metadata)
	if err != nil {
		return "", fmt.Errorf("marshal metadata: %w", err)
	}

	// Pages: empty → SQL NULL; non-empty → bind as jsonb.
	var pagesArg any
	if rec.PagesJSON != "" {
		pagesArg = rec.PagesJSON
	}

	q := `INSERT INTO documents (
		id, user_id, file_id, title, content, file_type, filename,
		total_char_count, total_line_count, metadata, pages,
		source_type, source, client_id, workspace_id
	) VALUES (
		$1, $2, $3, $4, $5, $6, $7,
		$8, $9, $10, $11::jsonb,
		$12, $13, $14, NULLIF($15, '')
	) ON CONFLICT (id) DO UPDATE SET
		file_id          = EXCLUDED.file_id,
		title            = EXCLUDED.title,
		content          = EXCLUDED.content,
		file_type        = EXCLUDED.file_type,
		filename         = EXCLUDED.filename,
		total_char_count  = EXCLUDED.total_char_count,
		total_line_count  = EXCLUDED.total_line_count,
		metadata         = EXCLUDED.metadata,
		pages            = EXCLUDED.pages,
		source_type      = EXCLUDED.source_type,
		source           = EXCLUDED.source`

	_, err = s.pool.Exec(ctx, q,
		id, s.owner.UserID, rec.FileID, rec.Title, rec.Content, rec.FileType,
		rec.Filename, rec.TotalCharCount, rec.TotalLineCount, metaArg,
		pagesArg, rec.SourceType, rec.Source,
		s.owner.ClientID, s.owner.WorkspaceID,
	)
	if err != nil {
		return "", fmt.Errorf("pg_file_store: CreateDocument: %w", err)
	}
	return id, nil
}

func (s *PostgresFileStore) GetDocumentByFileID(ctx context.Context, fileID string) (StoredDocument, error) {
	q := `SELECT id, COALESCE(file_id, ''), COALESCE(title, ''), COALESCE(content, '')
		  FROM documents WHERE file_id = $1 LIMIT 1`
	var doc StoredDocument
	if err := s.pool.QueryRow(ctx, q, fileID).Scan(
		&doc.ID, &doc.FileID, &doc.Title, &doc.Content,
	); err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return StoredDocument{}, fmt.Errorf("%w: document for file_id=%s", ErrNotFound, fileID)
		}
		return StoredDocument{}, fmt.Errorf("pg_file_store: GetDocumentByFileID: %w", err)
	}
	return doc, nil
}

func (s *PostgresFileStore) AppendToDocument(ctx context.Context, fileID, additionalContent string) error {
	q := `UPDATE documents SET content = COALESCE(content, '') || $2 WHERE file_id = $1`
	res, err := s.pool.Exec(ctx, q, fileID, additionalContent)
	if err != nil {
		return fmt.Errorf("pg_file_store: AppendToDocument: %w", err)
	}
	if res.RowsAffected() == 0 {
		return fmt.Errorf("%w: document for file_id=%s", ErrNotFound, fileID)
	}
	return nil
}

func (s *PostgresFileStore) UpdateFileChunkStats(ctx context.Context, fileID string, stats ChunkStats) error {
	payload, _ := json.Marshal(stats)
	q := `UPDATE files
		  SET metadata = jsonb_set(COALESCE(metadata, '{}'::jsonb), '{chunk_stats}', $2::jsonb)
		  WHERE id = $1`
	if _, err := s.pool.Exec(ctx, q, fileID, string(payload)); err != nil {
		return fmt.Errorf("pg_file_store: UpdateFileChunkStats: %w", err)
	}
	return nil
}

func (s *PostgresFileStore) CreateChunk(ctx context.Context, rec ChunkRecord) (string, error) {
	chunkID := rec.ID
	if chunkID == "" {
		chunkID = uuid.New().String()
	}

	metaArg, err := jsonbArg(rec.Metadata)
	if err != nil {
		return "", fmt.Errorf("marshal metadata: %w", err)
	}

	now := time.Now()

	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return "", fmt.Errorf("pg_file_store: CreateChunk begin: %w", err)
	}
	defer func() { _ = tx.Rollback(ctx) }()

	if _, err = tx.Exec(ctx, `INSERT INTO chunks (
		id, text, index, type, metadata,
		user_id, client_id, workspace_id,
		created_at, updated_at, accessed_at
	) VALUES (
		$1, $2, $3, $4, $5,
		$6, $7, NULLIF($8, ''),
		$9, $9, $9
	)`, chunkID, rec.Text, rec.Index, rec.Type, metaArg,
		s.owner.UserID, s.owner.ClientID, s.owner.WorkspaceID,
		now,
	); err != nil {
		return "", fmt.Errorf("pg_file_store: CreateChunk insert chunk: %w", err)
	}

	if rec.FileID != "" {
		if _, err = tx.Exec(ctx, `INSERT INTO file_chunks (file_id, chunk_id, user_id, workspace_id, created_at)
			VALUES ($1, $2, $3, NULLIF($4, ''), $5)`,
			rec.FileID, chunkID, s.owner.UserID, s.owner.WorkspaceID, now); err != nil {
			return "", fmt.Errorf("pg_file_store: CreateChunk insert file_chunks: %w", err)
		}
	}

	if rec.DocumentID != "" {
		if _, err = tx.Exec(ctx, `INSERT INTO document_chunks (document_id, chunk_id, user_id, workspace_id, created_at)
			VALUES ($1, $2, $3, NULLIF($4, ''), $5)`,
			rec.DocumentID, chunkID, s.owner.UserID, s.owner.WorkspaceID, now); err != nil {
			return "", fmt.Errorf("pg_file_store: CreateChunk insert document_chunks: %w", err)
		}
	}

	if err := tx.Commit(ctx); err != nil {
		return "", fmt.Errorf("pg_file_store: CreateChunk commit: %w", err)
	}
	return chunkID, nil
}

func (s *PostgresFileStore) GetFile(ctx context.Context, id string) (StoredFile, error) {
	q := `SELECT id, COALESCE(name, ''), COALESCE(url, ''), COALESCE(file_type, ''), COALESCE(file_hash, '')
		  FROM files WHERE id = $1`
	var f StoredFile
	if err := s.pool.QueryRow(ctx, q, id).Scan(
		&f.ID, &f.Name, &f.URL, &f.FileType, &f.Hash,
	); err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return StoredFile{}, fmt.Errorf("%w: file id=%s", ErrNotFound, id)
		}
		return StoredFile{}, fmt.Errorf("pg_file_store: GetFile: %w", err)
	}
	return f, nil
}

func (s *PostgresFileStore) DeleteFile(ctx context.Context, fileID string) error {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("pg_file_store: DeleteFile begin: %w", err)
	}
	defer func() { _ = tx.Rollback(ctx) }()

	// 1. Remove document-chunk junctions for documents belonging to this file.
	if _, err := tx.Exec(ctx, `DELETE FROM document_chunks
		WHERE document_id IN (SELECT id FROM documents WHERE file_id = $1)`, fileID); err != nil {
		return fmt.Errorf("pg_file_store: delete document_chunks: %w", err)
	}

	// 2. Remove file-chunk junctions (cascades to chunks via FK).
	if _, err := tx.Exec(ctx, `DELETE FROM file_chunks WHERE file_id = $1`, fileID); err != nil {
		return fmt.Errorf("pg_file_store: delete file_chunks: %w", err)
	}

	// 3. Remove documents (FK to files is ON DELETE SET NULL — no auto-cascade).
	if _, err := tx.Exec(ctx, `DELETE FROM documents WHERE file_id = $1`, fileID); err != nil {
		return fmt.Errorf("pg_file_store: delete documents: %w", err)
	}

	// 4. Remove the file itself.
	if _, err := tx.Exec(ctx, `DELETE FROM files WHERE id = $1`, fileID); err != nil {
		return fmt.Errorf("pg_file_store: delete file: %w", err)
	}

	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("pg_file_store: DeleteFile commit: %w", err)
	}
	return nil
}

// ============================================================================
// ChunkStore adapter implementation
// ============================================================================

// GetDocument fetches a parsed document by ID.
func (c *PostgresChunkStore) GetDocument(ctx context.Context, id string) (Document, error) {
	q := `SELECT id, COALESCE(content, ''), COALESCE(file_type, ''), COALESCE(pages::text, '')
		  FROM documents WHERE id = $1`
	var doc Document
	if err := c.store.pool.QueryRow(ctx, q, id).Scan(
		&doc.ID, &doc.Content, &doc.FileType, &doc.Pages,
	); err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return Document{}, fmt.Errorf("%w: document id=%s", ErrNotFound, id)
		}
		return Document{}, fmt.Errorf("pg_file_store: GetDocument: %w", err)
	}
	return doc, nil
}

// CreateChunk inserts a new chunk row and returns its ID.
func (c *PostgresChunkStore) CreateChunk(ctx context.Context, p CreateChunkParams) (string, error) {
	chunkID := uuid.New().String()
	now := time.Now()
	owner := c.store.owner

	tx, err := c.store.pool.Begin(ctx)
	if err != nil {
		return "", fmt.Errorf("pg_file_store: CreateChunk begin: %w", err)
	}
	defer func() { _ = tx.Rollback(ctx) }()

	meta := map[string]any{}
	if p.Metadata != "" {
		if err := json.Unmarshal([]byte(p.Metadata), &meta); err != nil {
			meta = map[string]any{"raw": p.Metadata}
		}
	}
	if p.ID != "" {
		meta["fileprocessor_id"] = p.ID
	}
	metaJSON, _ := json.Marshal(meta)

	if _, err := tx.Exec(ctx, `INSERT INTO chunks (
		id, text, index, type, metadata,
		user_id, client_id, workspace_id, created_at, updated_at, accessed_at
	) VALUES (
		$1, $2, $3, $4, $5::jsonb,
		$6, $7, NULLIF($8, ''), $9, $9, $9
	)`, chunkID, p.Text, p.Index, p.Type, string(metaJSON),
		owner.UserID, owner.ClientID, owner.WorkspaceID, now,
	); err != nil {
		return "", fmt.Errorf("pg_file_store: CreateChunk insert chunk: %w", err)
	}

	if _, err := tx.Exec(ctx, `INSERT INTO document_chunks (document_id, chunk_id, user_id, workspace_id, created_at)
		VALUES ($1, $2, $3, NULLIF($4, ''), $5)`,
		p.DocumentID, chunkID, owner.UserID, owner.WorkspaceID, now); err != nil {
		return "", fmt.Errorf("pg_file_store: CreateChunk insert document_chunks: %w", err)
	}

	// Look up file_id from the document and insert into file_chunks so that
	// [Searcher] can hydrate filenames via the file_chunks join.
	var fileID string
	if err := tx.QueryRow(ctx, `SELECT file_id FROM documents WHERE id = $1`, p.DocumentID).Scan(&fileID); err != nil {
		if !errors.Is(err, pgx.ErrNoRows) {
			return "", fmt.Errorf("pg_file_store: lookup file_id: %w", err)
		}
	}
	if fileID != "" {
		if _, err := tx.Exec(ctx, `INSERT INTO file_chunks (file_id, chunk_id, user_id, workspace_id, created_at)
			VALUES ($1, $2, $3, NULLIF($4, ''), $5)`,
			fileID, chunkID, owner.UserID, owner.WorkspaceID, now); err != nil {
			return "", fmt.Errorf("pg_file_store: CreateChunk insert file_chunks: %w", err)
		}
	}

	if err := tx.Commit(ctx); err != nil {
		return "", fmt.Errorf("pg_file_store: CreateChunk commit: %w", err)
	}
	return chunkID, nil
}

// GetChunksByIDs returns chunks in the same order as the IDs.
func (c *PostgresChunkStore) GetChunksByIDs(ctx context.Context, ids []string) ([]Chunk, error) {
	if len(ids) == 0 {
		return nil, nil
	}

	q := `SELECT c.id, COALESCE(c.text, ''), COALESCE(fc.file_id, ''), COALESCE(c.type, ''), c.index, COALESCE(c.metadata::text, '')
		  FROM chunks c
		  LEFT JOIN file_chunks fc ON fc.chunk_id = c.id
		  WHERE c.id = ANY($1)`

	rows, err := c.store.pool.Query(ctx, q, ids)
	if err != nil {
		return nil, fmt.Errorf("pg_file_store: GetChunksByIDs: %w", err)
	}
	defer rows.Close()

	byID := make(map[string]Chunk, len(ids))
	for rows.Next() {
		var ch Chunk
		if err := rows.Scan(&ch.ID, &ch.Text, &ch.FileID, &ch.Type, &ch.Index, &ch.Metadata); err != nil {
			return nil, fmt.Errorf("pg_file_store: scan chunk: %w", err)
		}
		byID[ch.ID] = ch
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("pg_file_store: iterate chunks: %w", err)
	}

	out := make([]Chunk, 0, len(ids))
	for _, id := range ids {
		if ch, ok := byID[id]; ok {
			out = append(out, ch)
		}
	}
	return out, nil
}

// GetFile fetches a file record by ID.
func (c *PostgresChunkStore) GetFile(ctx context.Context, id string) (RAGFile, error) {
	q := `SELECT id, COALESCE(name, '') FROM files WHERE id = $1`
	var f RAGFile
	if err := c.store.pool.QueryRow(ctx, q, id).Scan(&f.ID, &f.Name); err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return RAGFile{}, fmt.Errorf("%w: file id=%s", ErrNotFound, id)
		}
		return RAGFile{}, fmt.Errorf("pg_file_store: GetFile: %w", err)
	}
	return f, nil
}

// UpdateFileChunkStats records the outcome of an ingestion run.
func (c *PostgresChunkStore) UpdateFileChunkStats(ctx context.Context, p UpdateFileStatsParams) error {
	return c.store.UpdateFileChunkStats(ctx, p.FileID, ChunkStats{
		ChunkCount:      p.ChunkCount,
		ChunkingStatus:  p.ChunkingStatus,
		EmbeddingStatus: p.EmbeddingStatus,
	})
}

// ============================================================================
// helpers
// ============================================================================

// marshalNullableJSON converts a map to a JSON string for binding.
// When v is nil it returns ("", nil) so pgx receives SQL NULL — not
// the JSONB null literal — avoiding jsonb_set's "cannot set path in
// scalar" error when the column is later treated as a JSONB object.
//
// Caller: pass the returned *string (not the string) to pgx via a nullable
// pointer so pgx knows to send SQL NULL when the pointer is nil.
func marshalNullableJSON(v map[string]any) (string, error) {
	if v == nil {
		return "", nil // empty string; caller should send NULL via a nullable binding
	}
	b, err := json.Marshal(v)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// jsonbArg returns a value suitable for binding to a jsonb column. nil maps
// become nil (SQL NULL); non-nil maps become a JSON string. The extraKV
// pair (key, value), if non-empty, is merged into the map before encoding
// so callers can add a single top-level field without copying the map.
func jsonbArg(v map[string]any, extraKV ...string) (any, error) {
	if v == nil && len(extraKV) == 0 {
		return nil, nil
	}
	out := make(map[string]any, len(v)+len(extraKV)/2)
	for k, val := range v {
		out[k] = val
	}
	for i := 0; i+1 < len(extraKV); i += 2 {
		if extraKV[i+1] == "" {
			continue
		}
		out[extraKV[i]] = extraKV[i+1]
	}
	if len(out) == 0 {
		return nil, nil
	}
	b, err := json.Marshal(out)
	if err != nil {
		return nil, err
	}
	return string(b), nil
}
