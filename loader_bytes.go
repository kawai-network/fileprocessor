package fileprocessor

import (
	"bytes"
	"fmt"
	"path/filepath"
	"strings"

	"github.com/kawai-network/x/pdf/extractor"
	"github.com/kawai-network/x/pdf/model"
	"github.com/yudaprama/tools/gooxml/document"
	"github.com/yudaprama/tools/gooxml/presentation"
	"github.com/yudaprama/tools/gooxml/spreadsheet"
)

// LoadBytes extracts a file that is already available in memory. It mirrors
// LoadFile without requiring a temporary filesystem path, which is useful for
// workers that download files from object storage or HTTP.
func (l *FileLoader) LoadBytes(filename string, content []byte) (*FileDocument, error) {
	if len(content) == 0 {
		return nil, fmt.Errorf("fileprocessor: empty file content")
	}

	fileType, err := l.detectFileType(filename)
	if err != nil {
		return nil, err
	}

	var pages []DocumentPage
	var aggregated string
	switch fileType {
	case FileTypeTXT, FileTypeMarkdown:
		text := string(content)
		lines := strings.Split(text, "\n")
		pages = []DocumentPage{{
			CharCount:   len(text),
			LineCount:   len(lines),
			Metadata:    map[string]any{"lineNumberEnd": len(lines), "lineNumberStart": 1},
			PageContent: text,
		}}
		aggregated = fmt.Sprintf("```\n%s\n```", text)
	case FileTypePDF:
		pages, aggregated, err = loadPDFBytes(content)
	case FileTypeDOCX:
		doc, readErr := document.Read(bytes.NewReader(content), int64(len(content)))
		if readErr != nil {
			err = fmt.Errorf("open DOCX: %w", readErr)
			break
		}
		aggregated, err = doc.ToMarkdownWithImageURLs("/files")
		if err == nil && aggregated == "" {
			aggregated = "# DOCX Document\n\n*No content found in document*"
		}
		if err == nil {
			pages = []DocumentPage{{
				CharCount:   len(aggregated),
				LineCount:   len(strings.Split(aggregated, "\n")),
				PageContent: aggregated,
			}}
		}
	case FileTypeXLSX:
		wb, readErr := spreadsheet.Read(bytes.NewReader(content), int64(len(content)))
		if readErr != nil {
			err = fmt.Errorf("open XLSX: %w", readErr)
			break
		}
		aggregated, err = wb.ToMarkdownWithImageURLs("/files")
		_ = wb.Close()
		if err == nil && aggregated == "" {
			aggregated = "# Excel Workbook\n\n*No content found in workbook*"
		}
		if err == nil {
			pages = []DocumentPage{{
				CharCount:   len(aggregated),
				LineCount:   len(strings.Split(aggregated, "\n")),
				PageContent: aggregated,
			}}
		}
	case FileTypePPTX:
		pres, readErr := presentation.Read(bytes.NewReader(content), int64(len(content)))
		if readErr != nil {
			err = fmt.Errorf("open PPTX: %w", readErr)
			break
		}
		aggregated, err = pres.ToMarkdownWithImageURLs("/files")
		if err == nil && aggregated == "" {
			aggregated = "# PowerPoint Presentation\n\n*No content found in presentation*"
		}
		if err == nil {
			pages = []DocumentPage{{
				CharCount:   len(aggregated),
				LineCount:   len(strings.Split(aggregated, "\n")),
				PageContent: aggregated,
			}}
		}
	case FileTypeImage:
		name := baseName(filename)
		aggregated = fmt.Sprintf("![%s](/files/%s)\n\n", name, name)
		pages = []DocumentPage{{CharCount: len(aggregated), LineCount: 1, Metadata: map[string]any{"type": "image"}, PageContent: aggregated}}
	case FileTypeVideo:
		name := baseName(filename)
		aggregated = fmt.Sprintf("# Video: %s\n\n*Video processing in progress...*\n", name)
		pages = []DocumentPage{{CharCount: len(aggregated), LineCount: 3, Metadata: map[string]any{"type": "video"}, PageContent: aggregated}}
	default:
		err = fmt.Errorf("%w: %s", ErrUnsupportedFileType, fileType)
	}
	if err != nil {
		return nil, fmt.Errorf("fileprocessor: load content: %w", err)
	}

	return &FileDocument{
		Content:  aggregated,
		FileType: normalizeExt(filepath.Ext(filename)),
		Filename: baseName(filename),
		Pages:    pages,
	}, nil
}

func loadPDFBytes(content []byte) ([]DocumentPage, string, error) {
	reader, err := model.NewPdfReader(bytes.NewReader(content))
	if err != nil {
		return nil, "", fmt.Errorf("create PDF reader: %w", err)
	}
	numPages, err := reader.GetNumPages()
	if err != nil {
		return nil, "", fmt.Errorf("get number of pages: %w", err)
	}
	return extractPDFPages(reader, numPages)
}

// extractPDFPages iterates over every page in a PDF reader, extracts text,
// and returns the page metadata + aggregated markdown. Shared by both the
// in-memory (LoadBytes) and file-based (loadPDFFile) paths.
func extractPDFPages(reader *model.PdfReader, numPages int) ([]DocumentPage, string, error) {
	var pages []DocumentPage
	var sb strings.Builder
	sb.WriteString("# PDF Document\n\n")
	for i := 1; i <= numPages; i++ {
		page, err := reader.GetPage(i)
		if err != nil {
			return nil, "", fmt.Errorf("get page %d: %w", i, err)
		}
		ex, err := extractor.New(page)
		if err != nil {
			return nil, "", fmt.Errorf("create extractor for page %d: %w", i, err)
		}
		text, err := ex.ExtractText()
		if err != nil {
			return nil, "", fmt.Errorf("extract page %d: %w", i, err)
		}
		if strings.TrimSpace(text) == "" {
			text = "[Unable to extract text from this page]"
		}
		pages = append(pages, DocumentPage{
			CharCount: len(text), LineCount: len(strings.Split(text, "\n")),
			Metadata: map[string]any{"pageNumber": i}, PageContent: text,
		})
		fmt.Fprintf(&sb, "## Page %d\n\n%s\n\n", i, text)
	}
	return pages, sb.String(), nil
}
