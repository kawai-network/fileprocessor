package fileprocessor

import (
	"os"
	"strings"
	"testing"
)

func mustWrite(path string, data []byte) error { return os.WriteFile(path, data, 0644) }

func TestCharChunker(t *testing.T) {
	c := NewCharChunker(100, 20)
	text := strings.Repeat("hello world. ", 30)
	chunks := c.Chunk(text)
	if len(chunks) < 2 {
		t.Fatalf("expected multiple chunks, got %d", len(chunks))
	}
	for i, ch := range chunks {
		if len(ch) > 120 { // allow slight overshoot from merge logic
			t.Errorf("chunk %d too large: %d bytes", i, len(ch))
		}
	}
}

func TestCharChunker_ShortText(t *testing.T) {
	c := NewCharChunker(1000, 200)
	got := c.Chunk("short")
	if len(got) != 1 || got[0] != "short" {
		t.Fatalf("got %v", got)
	}
}

func TestTokenChunker(t *testing.T) {
	// whitespace tokenizer: 1 token per word.
	tok := func(s string) int { return len(strings.Fields(s)) }
	c := NewTokenChunker(10, 2, tok)

	// Build multi-line text so the chunker has natural boundaries to work
	// with. Each line has 5 words; 30 lines yields 150 tokens total.
	var sb strings.Builder
	for i := 0; i < 30; i++ {
		sb.WriteString("one two three four five\n")
	}
	chunks := c.Chunk(sb.String())
	if len(chunks) < 3 {
		t.Fatalf("expected multiple chunks, got %d", len(chunks))
	}
}

func TestFileLoader_DetectImage(t *testing.T) {
	l := NewFileLoader()
	if !l.IsImageFile("jpg") {
		t.Error("expected jpg to be image")
	}
	if !l.IsImageFile(".PNG") {
		t.Error("expected .PNG to be image")
	}
	if l.IsImageFile("pdf") {
		t.Error("pdf should not be image")
	}
}

func TestFileLoader_DetectVideo(t *testing.T) {
	l := NewFileLoader()
	if !l.IsVideoFile("mp4") {
		t.Error("expected mp4 to be video")
	}
	if l.IsVideoFile("txt") {
		t.Error("txt should not be video")
	}
}

func TestCalculateFileHash(t *testing.T) {
	tmp := t.TempDir() + "/file.txt"
	if err := writeFile(tmp, []byte("hello")); err != nil {
		t.Fatal(err)
	}
	h, err := CalculateFileHash(tmp)
	if err != nil {
		t.Fatal(err)
	}
	if len(h) != 64 {
		t.Errorf("expected 64-char sha256, got %d", len(h))
	}
}

func TestNewProcessor_RequiresFileStore(t *testing.T) {
	_, err := New(Config{})
	if err == nil {
		t.Fatal("expected error when FileStore is nil")
	}
}

// writeFile is a tiny helper to avoid pulling in ioutil.
func writeFile(path string, data []byte) error {
	return mustWrite(path, data)
}
