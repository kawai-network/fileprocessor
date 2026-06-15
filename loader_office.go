package fileprocessor

import (
	"fmt"
	"os"
	"strings"

	"github.com/getkawai/tools/gooxml/document"
	"github.com/getkawai/tools/gooxml/presentation"
	"github.com/getkawai/tools/gooxml/spreadsheet"
	"github.com/kawai-network/x/pdf/extractor"
	"github.com/kawai-network/x/pdf/model"
)

// loadPDFFile extracts markdown text and per-page metadata from a PDF using
// github.com/kawai-network/x/pdf.
func (l *FileLoader) loadPDFFile(filePath string) ([]DocumentPage, string, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, "", fmt.Errorf("open PDF: %w", err)
	}
	defer file.Close()

	reader, err := model.NewPdfReader(file)
	if err != nil {
		return nil, "", fmt.Errorf("create PDF reader: %w", err)
	}

	numPages, err := reader.GetNumPages()
	if err != nil {
		return nil, "", fmt.Errorf("get number of pages: %w", err)
	}

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

		lines := strings.Split(text, "\n")
		pages = append(pages, DocumentPage{
			CharCount:   len(text),
			LineCount:   len(lines),
			Metadata:    map[string]any{"pageNumber": i},
			PageContent: text,
		})

		fmt.Fprintf(&sb, "## Page %d\n\n%s\n\n", i, text)
	}

	return pages, sb.String(), nil
}

// --- DOCX ------------------------------------------------------------------

func (l *FileLoader) loadDOCXFile(filePath string) ([]DocumentPage, string, error) {
	doc, err := document.Open(filePath)
	if err != nil {
		return nil, "", fmt.Errorf("open DOCX: %w", err)
	}
	markdown, err := doc.ToMarkdownWithImageURLs("/files")
	if err != nil {
		return nil, "", fmt.Errorf("convert DOCX to markdown: %w", err)
	}
	if markdown == "" {
		markdown = "# DOCX Document\n\n*No content found in document*"
	}
	lines := strings.Split(markdown, "\n")
	page := DocumentPage{
		CharCount:   len(markdown),
		LineCount:   len(lines),
		PageContent: markdown,
	}
	return []DocumentPage{page}, markdown, nil
}

// --- XLSX ------------------------------------------------------------------

func (l *FileLoader) loadExcelFile(filePath string) ([]DocumentPage, string, error) {
	wb, err := spreadsheet.Open(filePath)
	if err != nil {
		return nil, "", fmt.Errorf("open XLSX: %w", err)
	}
	defer func() { _ = wb.Close() }()

	markdown, err := wb.ToMarkdownWithImageURLs("/files")
	if err != nil {
		return nil, "", fmt.Errorf("convert XLSX to markdown: %w", err)
	}
	if markdown == "" {
		markdown = "# Excel Workbook\n\n*No content found in workbook*"
	}
	lines := strings.Split(markdown, "\n")
	page := DocumentPage{
		CharCount:   len(markdown),
		LineCount:   len(lines),
		PageContent: markdown,
	}
	return []DocumentPage{page}, markdown, nil
}

// --- PPTX ------------------------------------------------------------------

func (l *FileLoader) loadPPTXFile(filePath string) ([]DocumentPage, string, error) {
	pres, err := presentation.Open(filePath)
	if err != nil {
		return nil, "", fmt.Errorf("open PPTX: %w", err)
	}
	markdown, err := pres.ToMarkdownWithImageURLs("/files")
	if err != nil {
		return nil, "", fmt.Errorf("convert PPTX to markdown: %w", err)
	}
	if markdown == "" {
		markdown = "# PowerPoint Presentation\n\n*No content found in presentation*"
	}
	lines := strings.Split(markdown, "\n")
	page := DocumentPage{
		CharCount:   len(markdown),
		LineCount:   len(lines),
		PageContent: markdown,
	}
	return []DocumentPage{page}, markdown, nil
}
