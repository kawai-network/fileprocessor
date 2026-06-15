package fileprocessor

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// CalculateFileHash returns the SHA-256 hex digest of the file at path.
func CalculateFileHash(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", fmt.Errorf("fileprocessor: open: %w", err)
	}
	defer f.Close()

	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", fmt.Errorf("fileprocessor: hash: %w", err)
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// FileInfo contains basic filesystem metadata for a file.
type FileInfo struct {
	Size int64
}

// GetFileInfo returns size information for the file at path.
func GetFileInfo(path string) (FileInfo, error) {
	st, err := os.Stat(path)
	if err != nil {
		return FileInfo{}, fmt.Errorf("fileprocessor: stat: %w", err)
	}
	return FileInfo{Size: st.Size()}, nil
}

// CopyToStorage copies the file at src into the storage base directory.
// The resulting path is filepath.Join(baseDir, "uploads", prefix+"-"+name)
// where prefix is a millisecond timestamp. It returns the destination
// absolute path and the relative "uploads/..." key.
func CopyToStorage(src, baseDir string) (absPath, relKey string, err error) {
	name := filepath.Base(src)
	relKey = filepath.Join("uploads", fmt.Sprintf("%d-%s", timeNow(), name))
	absPath = filepath.Join(baseDir, relKey)

	if err := mkdirAll(filepath.Dir(absPath)); err != nil {
		return "", "", fmt.Errorf("fileprocessor: mkdir: %w", err)
	}

	if _, err := copyFile(absPath, src); err != nil {
		return "", "", fmt.Errorf("fileprocessor: copy: %w", err)
	}
	return absPath, relKey, nil
}

// SafeDelete deletes a file at filePath only if it is inside baseDir. It is
// used by the library to clean up local copies after deletion.
func SafeDelete(filePath, baseDir string) error {
	cleaned := filepath.Clean(filePath)
	expectedPrefix := filepath.Clean(baseDir) + string(filepath.Separator)

	if strings.HasPrefix(cleaned, "http") || filepath.IsAbs(cleaned) && !strings.HasPrefix(cleaned, expectedPrefix) {
		return nil
	}
	if !strings.HasPrefix(cleaned, expectedPrefix) {
		return nil
	}

	info, err := os.Stat(cleaned)
	if err != nil {
		return nil
	}
	if info.IsDir() {
		return nil
	}
	return os.Remove(cleaned)
}
