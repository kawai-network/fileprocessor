package fileprocessor

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"log/slog"
	"sort"
	"sync"
	"time"
)

// EmbeddingCacheConfig configures an EmbeddingCache.
type EmbeddingCacheConfig struct {
	// MaxSize is the maximum number of entries retained. Default: 10000.
	MaxSize int
	// TTL is the per-entry time-to-live. 0 disables expiry. Default: 24h.
	TTL time.Duration
	// AvgEmbedMs is an estimate used for "time saved" stats. Default: 100.
	AvgEmbedMs int64
}

// DefaultEmbeddingCacheConfig returns the recommended cache configuration.
func DefaultEmbeddingCacheConfig() *EmbeddingCacheConfig {
	return &EmbeddingCacheConfig{
		MaxSize:    10000,
		TTL:        24 * time.Hour,
		AvgEmbedMs: 100,
	}
}

// EmbeddingCacheStats reports cumulative cache behavior.
type EmbeddingCacheStats struct {
	Hits       int64
	Misses     int64
	Size       int
	TotalSaved time.Duration
}

type embeddingCacheEntry struct {
	embedding []float32
	createdAt time.Time
	hitCount  int64
}

// EmbeddingCache is an in-memory SHA256-keyed TTL+LRU cache for embeddings.
// It wraps an Embedder and is safe for concurrent use.
type EmbeddingCache struct {
	inner      Embedder
	cache      map[string]*embeddingCacheEntry
	mu         sync.RWMutex
	maxSize    int
	ttl        time.Duration
	avgEmbedMs int64

	statsMu sync.Mutex
	stats   EmbeddingCacheStats
}

// Compile-time interface check.
var _ Embedder = (*EmbeddingCache)(nil)

// NewEmbeddingCache wraps inner with an LRU+TTL cache.
func NewEmbeddingCache(inner Embedder, config *EmbeddingCacheConfig) *EmbeddingCache {
	if config == nil {
		config = DefaultEmbeddingCacheConfig()
	}
	c := &EmbeddingCache{
		inner:      inner,
		cache:      make(map[string]*embeddingCacheEntry),
		maxSize:    config.MaxSize,
		ttl:        config.TTL,
		avgEmbedMs: config.AvgEmbedMs,
	}
	slog.Info("ragcore: embedding cache initialized",
		"max_size", config.MaxSize, "ttl", config.TTL)
	return c
}

// Embed implements Embedder. Cache hits skip the inner Embedder; misses are
// batched into a single inner call and stored with createdAt for TTL/eviction.
func (c *EmbeddingCache) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	results := make([][]float32, len(texts))
	toEmbed := make([]string, 0)
	toEmbedIdx := make([]int, 0)
	hashes := make([]string, len(texts))

	c.mu.RLock()
	for i, text := range texts {
		hash := hashText(text)
		hashes[i] = hash
		if entry, ok := c.cache[hash]; ok {
			if c.ttl == 0 || time.Since(entry.createdAt) < c.ttl {
				results[i] = entry.embedding
				entry.hitCount++
				c.recordHit()
				continue
			}
		}
		toEmbed = append(toEmbed, text)
		toEmbedIdx = append(toEmbedIdx, i)
		c.recordMiss()
	}
	c.mu.RUnlock()

	if len(toEmbed) == 0 {
		return results, nil
	}

	embeddings, err := c.inner.Embed(ctx, toEmbed)
	if err != nil {
		return nil, err
	}

	c.mu.Lock()
	for j, emb := range embeddings {
		idx := toEmbedIdx[j]
		results[idx] = emb
		hash := hashes[idx]
		c.cache[hash] = &embeddingCacheEntry{
			embedding: emb,
			createdAt: time.Now(),
		}
	}
	if len(c.cache) > c.maxSize {
		c.evictOldest(len(c.cache) - c.maxSize)
	}
	c.mu.Unlock()

	slog.DebugContext(ctx, "ragcore: generated embeddings", "count", len(toEmbed))
	return results, nil
}

// Dimension reports the wrapped embedder's dimension.
func (c *EmbeddingCache) Dimension() int {
	if c.inner == nil {
		return 0
	}
	return c.inner.Dimension()
}

// Stats returns a snapshot of cache statistics.
func (c *EmbeddingCache) Stats() EmbeddingCacheStats {
	c.statsMu.Lock()
	defer c.statsMu.Unlock()

	c.mu.RLock()
	c.stats.Size = len(c.cache)
	c.mu.RUnlock()

	return c.stats
}

// HitRate returns hits / (hits + misses). Returns 0 if no traffic yet.
func (c *EmbeddingCache) HitRate() float64 {
	c.statsMu.Lock()
	defer c.statsMu.Unlock()
	total := c.stats.Hits + c.stats.Misses
	if total == 0 {
		return 0
	}
	return float64(c.stats.Hits) / float64(total)
}

// Clear empties the cache and resets statistics.
func (c *EmbeddingCache) Clear() {
	c.mu.Lock()
	c.cache = make(map[string]*embeddingCacheEntry)
	c.mu.Unlock()

	c.statsMu.Lock()
	c.stats = EmbeddingCacheStats{}
	c.statsMu.Unlock()

	slog.Info("ragcore: embedding cache cleared")
}

// Warmup preloads embeddings for the given texts.
func (c *EmbeddingCache) Warmup(ctx context.Context, texts []string) error {
	if len(texts) == 0 {
		return nil
	}
	slog.InfoContext(ctx, "ragcore: cache warmup", "count", len(texts))
	_, err := c.Embed(ctx, texts)
	return err
}

// evictOldest removes count entries with the oldest createdAt timestamps.
// Caller must hold the write lock.
func (c *EmbeddingCache) evictOldest(count int) {
	if count <= 0 || len(c.cache) == 0 {
		return
	}

	type kv struct {
		hash      string
		createdAt time.Time
	}
	entries := make([]kv, 0, len(c.cache))
	for hash, e := range c.cache {
		entries = append(entries, kv{hash: hash, createdAt: e.createdAt})
	}
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].createdAt.Before(entries[j].createdAt)
	})

	removed := 0
	for i := 0; i < count && i < len(entries); i++ {
		delete(c.cache, entries[i].hash)
		removed++
	}
	slog.Debug("ragcore: evicted cache entries", "count", removed)
}

func (c *EmbeddingCache) recordHit() {
	c.statsMu.Lock()
	c.stats.Hits++
	c.stats.TotalSaved += time.Duration(c.avgEmbedMs) * time.Millisecond
	c.statsMu.Unlock()
}

func (c *EmbeddingCache) recordMiss() {
	c.statsMu.Lock()
	c.stats.Misses++
	c.statsMu.Unlock()
}

func hashText(text string) string {
	h := sha256.New()
	h.Write([]byte(text))
	return hex.EncodeToString(h.Sum(nil))
}
