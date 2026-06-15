package fileprocessor

import (
	"fmt"
	"path/filepath"
	"strings"
)

// SupportedFileType enumerates the file types the loader understands.
type SupportedFileType string

const (
	FileTypeTXT      SupportedFileType = "txt"
	FileTypePDF      SupportedFileType = "pdf"
	FileTypeDOC      SupportedFileType = "doc"
	FileTypeDOCX     SupportedFileType = "docx"
	FileTypeXLS      SupportedFileType = "xls"
	FileTypeXLSX     SupportedFileType = "xlsx"
	FileTypePPTX     SupportedFileType = "pptx"
	FileTypeImage    SupportedFileType = "image"
	FileTypeVideo    SupportedFileType = "video"
	FileTypeMarkdown SupportedFileType = "markdown"
)

// textExtensions are extensions that are treated as raw text and wrapped in a
// markdown code block by [FileLoader.LoadFile].
var textExtensions = map[string]struct{}{
	"txt": {}, "md": {}, "markdown": {}, "json": {}, "xml": {}, "html": {}, "htm": {},
	"css": {}, "js": {}, "ts": {}, "py": {}, "java": {}, "cpp": {}, "c": {}, "h": {},
	"hpp": {}, "cs": {}, "php": {}, "rb": {}, "go": {}, "rs": {}, "sh": {},
	"yml": {}, "yaml": {}, "toml": {}, "ini": {}, "cfg": {}, "conf": {}, "log": {},
	"csv": {}, "tsv": {},
}

// imageExtensions are recognized image types.
var imageExtensions = map[string]struct{}{
	"jpg": {}, "jpeg": {}, "png": {}, "gif": {}, "webp": {}, "svg": {},
	"bmp": {}, "tiff": {},
}

// videoExtensions are recognized video types.
var videoExtensions = map[string]struct{}{
	"mp4": {}, "mkv": {}, "avi": {}, "mov": {}, "wmv": {}, "flv": {},
	"webm": {}, "m4v": {}, "mpeg": {}, "mpg": {}, "3gp": {},
}

// FileLoader detects a file's type by extension and extracts its textual
// content (and, where possible, page structure) as markdown. It performs no
// I/O outside reading the input file and is safe for concurrent use.
type FileLoader struct{}

// NewFileLoader returns a FileLoader.
func NewFileLoader() *FileLoader { return &FileLoader{} }

// LoadFile loads the file at filePath and returns a [FileDocument]. The
// metadata argument is optional; when non-nil, [FileMetadata.Source],
// [FileMetadata.Filename], [FileMetadata.CreatedTime], and
// [FileMetadata.ModifiedTime] override the values inferred from the file
// itself. [FileMetadata.FileType] is always taken from the extension and
// cannot be overridden.
func (l *FileLoader) LoadFile(filePath string, fileMetadata *FileMetadata) (*FileDocument, error) {
	stats, err := statFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("fileprocessor: stat: %w", err)
	}

	fileType, err := l.detectFileType(filePath)
	if err != nil {
		return nil, err
	}

	ext := normalizeExt(filepath.Ext(filePath))
	baseFilename := filepath.Base(filePath)

	source := filePath
	filename := baseFilename
	createdTime := stats.ModTime()
	modifiedTime := stats.ModTime()
	if fileMetadata != nil {
		if fileMetadata.Source != "" {
			source = fileMetadata.Source
		}
		if fileMetadata.Filename != "" {
			filename = fileMetadata.Filename
		}
		if !fileMetadata.CreatedTime.IsZero() {
			createdTime = fileMetadata.CreatedTime
		}
		if !fileMetadata.ModifiedTime.IsZero() {
			modifiedTime = fileMetadata.ModifiedTime
		}
	}

	pages, aggregated, err := l.loadContent(filePath, fileType)
	if err != nil {
		return nil, fmt.Errorf("fileprocessor: load content: %w", err)
	}

	totalChars, totalLines := 0, 0
	for _, p := range pages {
		totalChars += p.CharCount
		totalLines += p.LineCount
	}

	return &FileDocument{
		Content:        aggregated,
		CreatedTime:    createdTime,
		FileType:       ext,
		Filename:       filename,
		Pages:          pages,
		Source:         source,
		TotalCharCount: totalChars,
		TotalLineCount: totalLines,
		Metadata: FileMetadata{
			Source:       source,
			Filename:     filename,
			FileType:     ext,
			CreatedTime:  createdTime,
			ModifiedTime: modifiedTime,
		},
	}, nil
}

// detectFileType determines a [SupportedFileType] from a file's extension.
// Unknown types return [ErrUnsupportedFileType].
func (l *FileLoader) detectFileType(filePath string) (SupportedFileType, error) {
	ext := normalizeExt(filepath.Ext(filePath))

	switch ext {
	case "pdf":
		return FileTypePDF, nil
	case "doc":
		return FileTypeDOC, nil
	case "docx":
		return FileTypeDOCX, nil
	case "xlsx", "xls":
		return FileTypeXLSX, nil
	case "pptx":
		return FileTypePPTX, nil
	case "txt", "":
		return FileTypeTXT, nil
	case "md", "markdown":
		return FileTypeMarkdown, nil
	}

	if _, ok := textExtensions[ext]; ok {
		return FileTypeTXT, nil
	}
	if l.IsImageFile(ext) {
		return FileTypeImage, nil
	}
	if l.IsVideoFile(ext) {
		return FileTypeVideo, nil
	}
	return "", fmt.Errorf("%w: %s", ErrUnsupportedFileType, ext)
}

// IsImageFile reports whether the extension (with or without leading dot,
// lowercased) is an image type.
func (l *FileLoader) IsImageFile(ext string) bool {
	_, ok := imageExtensions[normalizeExt(ext)]
	return ok
}

// IsVideoFile reports whether the extension is a video type.
func (l *FileLoader) IsVideoFile(ext string) bool {
	_, ok := videoExtensions[normalizeExt(ext)]
	return ok
}

// IsTextReadableFile reports whether the extension is treated as raw text.
func (l *FileLoader) IsTextReadableFile(ext string) bool {
	_, ok := textExtensions[normalizeExt(ext)]
	return ok
}

// CanChunkForRAG reports whether content of this type can be chunked for
// RAG. Images are chunkable (after VL/OCR enrichment), but videos and binary
// archives are not.
func (l *FileLoader) CanChunkForRAG(ext string) bool {
	ext = normalizeExt(ext)
	if strings.HasPrefix(ext, "video/") || strings.HasPrefix(ext, "audio/") {
		return false
	}
	if ext == "image/" {
		return true
	}
	switch ext {
	case "application/octet-stream",
		"application/zip",
		"application/x-rar",
		"application/x-7z-compressed",
		"application/x-tar",
		"application/gzip",
		"application/x-bzip2",
		"application/x-xz":
		return false
	}
	return true
}

func normalizeExt(ext string) string {
	ext = strings.ToLower(strings.TrimPrefix(ext, "."))
	return ext
}
