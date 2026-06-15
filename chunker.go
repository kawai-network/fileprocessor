package fileprocessor

import "strings"

// Chunker splits document content into pieces suitable for embedding. The
// library ships two implementations:
//   - [CharChunker]: character-based recursive text splitter.
//   - [TokenChunker]: token-aware splitter using a caller-provided counter.
//
// Implement the interface yourself for custom chunking strategies.
type Chunker interface {
	// Chunk splits content into a slice of chunks.
	Chunk(content string) []string
}

// --- CharChunker -----------------------------------------------------------

// CharChunker is a character-based recursive text splitter. It matches the
// behaviour of github.com/kawai-network/ragcore's default chunker and is the
// recommended starting point.
type CharChunker struct {
	opts ChunkOptions
}

// NewCharChunker returns a character-based chunker. Size defaults to 1000,
// overlap to 200.
func NewCharChunker(size, overlap int) *CharChunker {
	if size <= 0 {
		size = 1000
	}
	if overlap < 0 {
		overlap = 200
	}
	return &CharChunker{opts: ChunkOptions{ChunkSize: size, OverlapSize: overlap}}
}

// Chunk splits text recursively by decreasingly small separators.
func (c *CharChunker) Chunk(content string) []string {
	return splitChars(content, c.opts.ChunkSize, c.opts.OverlapSize)
}

// --- TokenChunker ----------------------------------------------------------

// Tokenizer is the primitive needed by [TokenChunker]. The caller decides
// which token-counting library to use (e.g. tiktoken-go).
type Tokenizer func(text string) int

// TokenChunker splits content so each chunk stays within a token budget. It
// uses a caller-provided [Tokenizer] to count tokens.
type TokenChunker struct {
	opts     ChunkOptions
	tokenize Tokenizer
}

// NewTokenChunker returns a token-aware chunker. Typical values: 256 tokens
// with 50 overlap, or 512 with 100.
func NewTokenChunker(size, overlap int, tokenize Tokenizer) *TokenChunker {
	if size <= 0 {
		size = 256
	}
	if overlap < 0 {
		overlap = 50
	}
	return &TokenChunker{
		opts:     ChunkOptions{ChunkSize: size, OverlapSize: overlap},
		tokenize: tokenize,
	}
}

// Chunk splits text into token-bounded chunks.
func (c *TokenChunker) Chunk(content string) []string {
	if c.tokenize == nil || content == "" {
		return []string{}
	}

	lines := strings.Split(content, "\n")
	var chunks []string
	var cur strings.Builder

	for i, line := range lines {
		lineWithSep := line
		if i < len(lines)-1 {
			lineWithSep += "\n"
		}

		candidate := cur.String() + lineWithSep
		if candidate != "" && c.tokenize(candidate) > c.opts.ChunkSize && cur.Len() > 0 {
			chunks = append(chunks, cur.String())
			cur.Reset()

			if c.opts.OverlapSize > 0 && len(chunks) > 0 {
				tail := overlapTail(chunks[len(chunks)-1], c.opts.OverlapSize, c.tokenize)
				cur.WriteString(tail)
			}
		}

		cur.WriteString(lineWithSep)
	}

	if cur.Len() > 0 {
		chunks = append(chunks, cur.String())
	}

	return chunks
}

// overlapTail returns the suffix of src whose token count is ≤ maxTokens.
func overlapTail(src string, maxTokens int, tokenize Tokenizer) string {
	if src == "" || maxTokens <= 0 {
		return ""
	}
	runes := []rune(src)
	for i := len(runes) - 1; i >= 0; i-- {
		if tokenize(string(runes[i:])) <= maxTokens {
			return string(runes[i:])
		}
	}
	return src
}

// --- internal char-based helpers -------------------------------------------

func splitChars(text string, chunkSize, overlap int) []string {
	if text == "" || chunkSize <= 0 {
		return nil
	}
	return recursiveSplit(text, []string{"\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "}, 0, chunkSize, overlap)
}

func recursiveSplit(text string, seps []string, depth, size, overlap int) []string {
	if len(text) <= size {
		return []string{text}
	}
	if depth >= len(seps) {
		return forceSplit(text, size, overlap)
	}

	if !strings.Contains(text, seps[depth]) {
		return recursiveSplit(text, seps, depth+1, size, overlap)
	}

	parts := strings.Split(text, seps[depth])
	var chunks, good []string

	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		if len(part) > size {
			if len(good) > 0 {
				chunks = append(chunks, mergeParts(good, seps[depth], size, overlap)...)
				good = nil
			}
			chunks = append(chunks, recursiveSplit(part, seps, depth+1, size, overlap)...)
		} else {
			good = append(good, part)
		}
	}
	if len(good) > 0 {
		chunks = append(chunks, mergeParts(good, seps[depth], size, overlap)...)
	}
	return chunks
}

func mergeParts(parts []string, sep string, size, overlap int) []string {
	var chunks []string
	var cur strings.Builder
	for _, p := range parts {
		if cur.Len() > 0 && cur.Len()+len(sep)+len(p) > size {
			chunks = append(chunks, cur.String())
			cur.Reset()
			if overlap > 0 && len(chunks) > 0 {
				prev := chunks[len(chunks)-1]
				start := len(prev) - overlap
				if start < 0 {
					start = 0
				}
				cur.WriteString(prev[start:])
				cur.WriteString(sep)
			}
		}
		cur.WriteString(p)
		if cur.Len() > 0 && cur.Len() != len(p) {
			cur.WriteString(sep)
		}
	}
	if cur.Len() > 0 {
		chunks = append(chunks, cur.String())
	}
	return chunks
}

func forceSplit(text string, size, overlap int) []string {
	var chunks []string
	for len(text) > 0 {
		if len(text) <= size {
			chunks = append(chunks, text)
			break
		}
		chunks = append(chunks, text[:size])
		step := size - overlap
		if step <= 0 {
			step = size
		}
		text = text[step:]
	}
	return chunks
}
