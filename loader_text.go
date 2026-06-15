package fileprocessor

import (
	"fmt"
	"os"
	"strings"
)

// loadContent dispatches to the per-type content extractors. It returns the
// per-page list, the aggregated markdown content, and any error.
func (l *FileLoader) loadContent(filePath string, fileType SupportedFileType) ([]DocumentPage, string, error) {
	switch fileType {
	case FileTypeTXT, FileTypeMarkdown:
		return l.loadTextFile(filePath)
	case FileTypePDF:
		return l.loadPDFFile(filePath)
	case FileTypeDOCX:
		return l.loadDOCXFile(filePath)
	case FileTypeXLSX:
		return l.loadExcelFile(filePath)
	case FileTypePPTX:
		return l.loadPPTXFile(filePath)
	case FileTypeImage:
		return l.loadImageFile(filePath)
	case FileTypeVideo:
		return l.loadVideoFile(filePath)
	default:
		return nil, "", fmt.Errorf("%w: %s", ErrUnsupportedFileType, fileType)
	}
}

func (l *FileLoader) loadTextFile(filePath string) ([]DocumentPage, string, error) {
	content, err := os.ReadFile(filePath)
	if err != nil {
		return nil, "", fmt.Errorf("read text: %w", err)
	}
	text := string(content)
	lines := strings.Split(text, "\n")
	page := DocumentPage{
		CharCount:   len(text),
		LineCount:   len(lines),
		Metadata:    map[string]any{"lineNumberEnd": len(lines), "lineNumberStart": 1},
		PageContent: text,
	}
	return []DocumentPage{page}, fmt.Sprintf("```\n%s\n```", text), nil
}

func (l *FileLoader) loadImageFile(filePath string) ([]DocumentPage, string, error) {
	filename := baseName(filePath)
	markdown := fmt.Sprintf("![%s](/files/%s)\n\n", filename, filename)
	return []DocumentPage{{
		CharCount:   len(markdown),
		LineCount:   1,
		Metadata:    map[string]any{"type": "image"},
		PageContent: markdown,
	}}, markdown, nil
}

func (l *FileLoader) loadVideoFile(filePath string) ([]DocumentPage, string, error) {
	filename := baseName(filePath)
	markdown := fmt.Sprintf("# Video: %s\n\n*Video processing in progress...*\n", filename)
	return []DocumentPage{{
		CharCount:   len(markdown),
		LineCount:   3,
		Metadata:    map[string]any{"type": "video"},
		PageContent: markdown,
	}}, markdown, nil
}
