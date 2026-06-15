package fileprocessor

import (
	"context"
	"fmt"
	"os/exec"
	"strings"
)

// ExtractTextWithTesseract runs the external `tesseract` binary on an image
// and returns the extracted text. If tesseract is not on PATH the function
// returns an error; callers should fall back to the VL provider.
//
// The language pack list tries "eng+ind+jpn+chi_sim" first and falls back to
// the default if those packs are not installed.
func ExtractTextWithTesseract(imagePath string) (string, error) {
	tesseractPath, err := exec.LookPath("tesseract")
	if err != nil {
		return "", fmt.Errorf("fileprocessor: tesseract not found: %w", err)
	}

	out, err := exec.Command(tesseractPath, imagePath, "stdout", "-l", "eng+ind+jpn+chi_sim").Output()
	if err != nil {
		out, err = exec.Command(tesseractPath, imagePath, "stdout").Output()
		if err != nil {
			return "", fmt.Errorf("fileprocessor: tesseract failed: %w", err)
		}
	}
	return string(out), nil
}

// CleanupOCRText uses an optional [LanguageModel] to clean up raw OCR output
// and reformat it as markdown. If lm is nil, the raw text is returned
// unchanged.
func CleanupOCRText(ctx context.Context, lm LanguageModel, rawText, filename string) (string, error) {
	if lm == nil {
		return rawText, nil
	}

	docHint := "This is from an image file."
	switch ext := strings.ToLower(extOf(filename)); ext {
	case "pdf":
		docHint = "This appears to be from a PDF document."
	case "png", "jpg", "jpeg":
		docHint = "This is from a screenshot or image."
	}

	system := `You are an OCR text editor. Clean up raw OCR output:
- Fix obvious typos and character recognition errors
- Preserve original structure and formatting
- Format as proper markdown when appropriate
- Output ONLY the cleaned text without explanations.`

	user := fmt.Sprintf(`Clean up the following OCR-extracted text and format it as clean markdown.

%s

Instructions:
1. Fix obvious OCR errors and typos
2. Remove artifacts like random characters, broken words
3. Preserve the original meaning and structure
4. Format as proper markdown:
   - Use headers (##, ###) for section titles
   - Use bullet points or numbered lists where appropriate
   - Use **bold** for emphasis or important terms
   - Use code blocks for any code or technical content
5. If it's a table, format it as a markdown table
6. Keep the content concise but complete
7. Output ONLY the cleaned markdown, no explanations

Raw OCR text:
---
%s
---

Cleaned markdown:`, docHint, rawText)

	result, err := lm.Generate(ctx, system, user)
	if err != nil {
		return "", fmt.Errorf("fileprocessor: LLM cleanup failed: %w", err)
	}
	result = strings.TrimSpace(result)
	if len(result) < 10 {
		return rawText, nil
	}
	return result, nil
}

func extOf(name string) string {
	for i := len(name) - 1; i >= 0; i-- {
		if name[i] == '.' {
			return name[i+1:]
		}
	}
	return ""
}
