package fileprocessor

import (
	"fmt"
	"strings"
)

// mdSplitterConfig configures the markdown header splitter.
type mdSplitterConfig struct {
	// Headers maps header prefixes (e.g. "##", "###") to metadata keys.
	Headers map[string]string

	// TrimHeaders excludes the header line from chunk content when true.
	TrimHeaders bool
}

// mdSplitter splits markdown text by configured header levels.
type mdSplitter struct {
	headers     map[string]string
	trimHeaders bool
}

// mdSplitterChunk is a single output chunk.
type mdSplitterChunk struct {
	Content  string
	Metadata map[string]string
}

// newMDSplitter constructs a splitter. Returns an error if config is invalid.
func newMDSplitter(config *mdSplitterConfig) (*mdSplitter, error) {
	if config == nil {
		return nil, fmt.Errorf("config cannot be nil")
	}
	if len(config.Headers) == 0 {
		return nil, fmt.Errorf("no headers specified")
	}
	for header := range config.Headers {
		for _, c := range header {
			if c != '#' {
				return nil, fmt.Errorf("header can only consist of '#': %s", header)
			}
		}
	}
	return &mdSplitter{
		headers:     config.Headers,
		trimHeaders: config.TrimHeaders,
	}, nil
}

const (
	codeFenceBacktick = "```"
	codeFenceTilde    = "~~~"
)

type metaRecord struct {
	name  string
	level int
	data  string
}

// Split splits text by configured markdown headers, carrying active header
// metadata into each chunk.
func (s *mdSplitter) Split(text string) []mdSplitterChunk {
	var recordedMetaList []metaRecord
	recordedMetaMap := make(map[string]string)
	var currentLines []string
	var inCodeBlock bool
	var openingFence string
	var chunks []mdSplitterChunk

	lines := strings.Split(text, "\n")

	for _, line := range lines {
		if len(line) == 0 {
			currentLines = append(currentLines, line)
			continue
		}

		trimmedLine := strings.TrimSpace(line)

		if !inCodeBlock {
			if strings.HasPrefix(trimmedLine, codeFenceBacktick) && strings.Count(trimmedLine, codeFenceBacktick) == 1 {
				inCodeBlock = true
				openingFence = codeFenceBacktick
			} else if strings.HasPrefix(trimmedLine, codeFenceTilde) {
				inCodeBlock = true
				openingFence = codeFenceTilde
			}
		} else {
			if strings.HasPrefix(trimmedLine, openingFence) {
				inCodeBlock = false
				openingFence = ""
			}
		}

		if inCodeBlock {
			currentLines = append(currentLines, line)
			continue
		}

		isNewHeader := false
		for header, name := range s.headers {
			if strings.HasPrefix(trimmedLine, header) && (len(trimmedLine) == len(header) || trimmedLine[len(header)] == ' ') {
				if len(currentLines) > 0 {
					chunks = append(chunks, mdSplitterChunk{
						Content:  strings.Join(currentLines, "\n"),
						Metadata: deepCopyStringMap(recordedMetaMap),
					})
					currentLines = nil
				}

				if !s.trimHeaders {
					currentLines = append(currentLines, line)
				}

				newLevel := len(header)
				for i := len(recordedMetaList) - 1; i >= 0; i-- {
					if recordedMetaList[i].level >= newLevel {
						delete(recordedMetaMap, recordedMetaList[i].name)
						recordedMetaList = recordedMetaList[:i]
					} else {
						break
					}
				}

				headerData := strings.TrimSpace(trimmedLine[len(header):])
				recordedMetaList = append(recordedMetaList, metaRecord{
					name:  name,
					level: newLevel,
					data:  headerData,
				})
				recordedMetaMap[name] = headerData

				isNewHeader = true
				break
			}
		}

		if !isNewHeader {
			currentLines = append(currentLines, line)
		}
	}

	if len(currentLines) > 0 {
		chunks = append(chunks, mdSplitterChunk{
			Content:  strings.Join(currentLines, "\n"),
			Metadata: deepCopyStringMap(recordedMetaMap),
		})
	}

	return chunks
}

func deepCopyStringMap(m map[string]string) map[string]string {
	if m == nil {
		return nil
	}
	out := make(map[string]string, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}
