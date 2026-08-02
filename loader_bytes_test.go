package fileprocessor

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestFileLoaderLoadBytesPDF(t *testing.T) {
	doc, err := NewFileLoader().LoadBytes("fixture.pdf", minimalPDF())
	if err != nil {
		t.Fatalf("LoadBytes PDF: %v", err)
	}
	if len(doc.Pages) == 0 || doc.Content == "" {
		t.Fatalf("expected extracted PDF content, pages=%d content=%d", len(doc.Pages), len(doc.Content))
	}
}

func minimalPDF() []byte {
	objects := []string{
		"<< /Type /Catalog /Pages 2 0 R >>",
		"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
		"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
		"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
		"<< /Length 34 >>\nstream\nBT /F1 24 Tf 72 720 Td (hello pdf) Tj ET\nendstream",
	}
	var b strings.Builder
	b.WriteString("%PDF-1.4\n")
	offsets := make([]int, 0, len(objects))
	for i, obj := range objects {
		offsets = append(offsets, b.Len())
		fmt.Fprintf(&b, "%d 0 obj\n%s\nendobj\n", i+1, obj)
	}
	xref := b.Len()
	fmt.Fprintf(&b, "xref\n0 %d\n0000000000 65535 f \n", len(objects)+1)
	for _, offset := range offsets {
		fmt.Fprintf(&b, "%010d 00000 n \n", offset)
	}
	fmt.Fprintf(&b, "trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n", len(objects)+1, xref)
	return []byte(b.String())
}

func TestFileLoaderLoadBytesDOCX(t *testing.T) {
	path := filepath.Join("..", "tools", "gooxml", "document", "testdata", "simple-1.docx")
	content, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("DOCX fixture unavailable: %v", err)
	}

	doc, err := NewFileLoader().LoadBytes("simple-1.docx", content)
	if err != nil {
		t.Fatalf("LoadBytes DOCX: %v", err)
	}
	if len(doc.Pages) == 0 || doc.Content == "" {
		t.Fatalf("expected extracted DOCX content, pages=%d content=%d", len(doc.Pages), len(doc.Content))
	}
}
