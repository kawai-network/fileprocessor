package fileprocessor

import (
	"fmt"
	"strings"
)

// ChunkingConfig controls how a document is split into chunks.
type ChunkingConfig struct {
	Enabled     bool
	ChunkSize   int
	OverlapSize int
}

// DocumentChunk is a single chunk emitted by the RAGChunker.
type DocumentChunk struct {
	ID       string
	Content  string
	Metadata map[string]interface{}
}

// RAGChunker splits a [FileDocument] into chunks. The default implementation
// routes by file type (PDF page-merge, markdown header split, recursive
// fallback). The fileprocessor package also ships a pluggable [Chunker]
// interface (CharChunker / TokenChunker) used by [Processor] for
// caller-controlled chunking.
type RAGChunker struct{}

// NewRAGChunker returns a RAGChunker with default settings.
func NewRAGChunker() *RAGChunker { return &RAGChunker{} }

// ChunkDocument chunks a FileDocument based on its type and configuration.
func (c *RAGChunker) ChunkDocument(doc *FileDocument, config ChunkingConfig) []DocumentChunk {
	if !config.Enabled || doc == nil {
		return nil
	}

	switch SupportedFileType(doc.FileType) {
	case FileTypePDF:
		if len(doc.Pages) > 0 {
			return c.chunkPDFByPages(doc.Pages, config)
		}
		return c.chunkMarkdownWithHeaders(doc.Content, config)
	case FileTypeDOCX, FileTypePPTX, FileTypeXLSX, FileTypeTXT, FileTypeMarkdown:
		return c.chunkMarkdownWithHeaders(doc.Content, config)
	default:
		return c.chunkByRecursiveSplit(doc.Content, config)
	}
}

// chunkMarkdownWithHeaders splits content using configured markdown headers,
// then recursively splits any oversized chunk.
func (c *RAGChunker) chunkMarkdownWithHeaders(content string, config ChunkingConfig) []DocumentChunk {
	splitter, err := newMDSplitter(&mdSplitterConfig{
		Headers: map[string]string{
			"##":  "h2",
			"###": "h3",
		},
		TrimHeaders: false,
	})
	if err != nil {
		return c.chunkByRecursiveSplit(content, config)
	}

	mdChunks := splitter.Split(content)

	var chunks []DocumentChunk
	chunkID := 1

	for _, md := range mdChunks {
		body := md.Content
		if len(body) > config.ChunkSize {
			for _, sub := range c.chunkText(body, config) {
				meta := map[string]interface{}{
					"type":   "markdown-header",
					"h2":     md.Metadata["h2"],
					"h3":     md.Metadata["h3"],
					"source": "mdsplitter-split",
				}
				chunks = append(chunks, DocumentChunk{
					ID:       fmt.Sprintf("chunk-%d", chunkID),
					Content:  sub,
					Metadata: meta,
				})
				chunkID++
			}
			continue
		}
		chunks = append(chunks, DocumentChunk{
			ID:      fmt.Sprintf("chunk-%d", chunkID),
			Content: body,
			Metadata: map[string]interface{}{
				"type":   "markdown-header",
				"h2":     md.Metadata["h2"],
				"h3":     md.Metadata["h3"],
				"source": "mdsplitter",
			},
		})
		chunkID++
	}

	return chunks
}

// chunkPDFByPages merges PDF pages up to ChunkSize, carrying page numbers.
func (c *RAGChunker) chunkPDFByPages(pages []DocumentPage, config ChunkingConfig) []DocumentChunk {
	var chunks []DocumentChunk
	var current strings.Builder
	var currentPages []int
	chunkID := 1

	for i, page := range pages {
		pageNum := i + 1

		if current.Len() > 0 && current.Len()+page.CharCount > config.ChunkSize {
			chunks = append(chunks, DocumentChunk{
				ID:      fmt.Sprintf("chunk-%d", chunkID),
				Content: current.String(),
				Metadata: map[string]interface{}{
					"type":  "pdf-pages",
					"pages": currentPages,
				},
			})
			chunkID++

			current.Reset()
			currentPages = nil

			if config.OverlapSize > 0 && i > 0 {
				prev := pages[i-1].PageContent
				start := len(prev) - config.OverlapSize
				if start < 0 {
					start = 0
				}
				current.WriteString(prev[start:])
				current.WriteString("\n\n")
			}
		}

		current.WriteString(page.PageContent)
		current.WriteString("\n\n")
		currentPages = append(currentPages, pageNum)
	}

	if current.Len() > 0 {
		chunks = append(chunks, DocumentChunk{
			ID:      fmt.Sprintf("chunk-%d", chunkID),
			Content: current.String(),
			Metadata: map[string]interface{}{
				"type":  "pdf-pages",
				"pages": currentPages,
			},
		})
	}

	return chunks
}

// chunkByRecursiveSplit splits content by progressively smaller separators.
func (c *RAGChunker) chunkByRecursiveSplit(content string, config ChunkingConfig) []DocumentChunk {
	parts := c.chunkText(content, config)
	chunks := make([]DocumentChunk, 0, len(parts))
	for i, t := range parts {
		chunks = append(chunks, DocumentChunk{
			ID:      fmt.Sprintf("chunk-%d", i+1),
			Content: t,
			Metadata: map[string]interface{}{
				"type":   "recursive-split",
				"source": "fallback",
			},
		})
	}
	return chunks
}

// chunkText splits text into pieces no larger than ChunkSize using a cascade of
// separators, with OverlapSize overlap between consecutive pieces.
func (c *RAGChunker) chunkText(text string, config ChunkingConfig) []string {
	if text == "" || !config.Enabled {
		return []string{}
	}

	chunkSize := config.ChunkSize
	overlap := config.OverlapSize
	if chunkSize <= 0 {
		chunkSize = 1000
	}
	if overlap < 0 {
		overlap = 200
	}

	separators := []string{
		"\n\n",
		"\n",
		". ",
		"? ",
		"! ",
		"; ",
		", ",
		" ",
	}

	return c.recursiveSplit(text, separators, 0, chunkSize, overlap)
}

func (c *RAGChunker) recursiveSplit(text string, separators []string, depth, chunkSize, overlap int) []string {
	if len(text) <= chunkSize {
		return []string{text}
	}
	if depth >= len(separators) {
		return c.forceSplitBySize(text, chunkSize, overlap)
	}

	sep := separators[depth]
	if !strings.Contains(text, sep) {
		return c.recursiveSplit(text, separators, depth+1, chunkSize, overlap)
	}

	parts := strings.Split(text, sep)

	var finalChunks []string
	var goodParts []string

	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}

		if len(part) > chunkSize {
			if len(goodParts) > 0 {
				finalChunks = append(finalChunks, c.mergeParts(goodParts, sep, chunkSize, overlap)...)
				goodParts = nil
			}
			finalChunks = append(finalChunks, c.recursiveSplit(part, separators, depth+1, chunkSize, overlap)...)
		} else {
			goodParts = append(goodParts, part)
		}
	}

	if len(goodParts) > 0 {
		finalChunks = append(finalChunks, c.mergeParts(goodParts, sep, chunkSize, overlap)...)
	}

	return finalChunks
}

func (c *RAGChunker) mergeParts(parts []string, sep string, chunkSize, overlap int) []string {
	var chunks []string
	var current strings.Builder

	for _, part := range parts {
		partLen := len(part)
		sepLen := 0
		if current.Len() > 0 {
			sepLen = len(sep)
		}

		if current.Len() > 0 && current.Len()+sepLen+partLen > chunkSize {
			chunks = append(chunks, current.String())
			current.Reset()
			if overlap > 0 && len(chunks) > 0 {
				prev := chunks[len(chunks)-1]
				start := len(prev) - overlap
				if start < 0 {
					start = 0
				}
				current.WriteString(prev[start:])
				current.WriteString(sep)
			}
		}

		if current.Len() > 0 {
			current.WriteString(sep)
		}
		current.WriteString(part)
	}

	if current.Len() > 0 {
		chunks = append(chunks, current.String())
	}

	return chunks
}

func (c *RAGChunker) forceSplitBySize(text string, chunkSize, overlap int) []string {
	var chunks []string
	for len(text) > 0 {
		if len(text) <= chunkSize {
			chunks = append(chunks, text)
			break
		}
		chunks = append(chunks, text[:chunkSize])
		step := chunkSize - overlap
		if step <= 0 {
			step = chunkSize
		}
		text = text[step:]
	}
	return chunks
}
