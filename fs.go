package fileprocessor

import (
	"os"
	"path/filepath"
	"time"
)

// timeNow returns the current unix millisecond timestamp. Hookable for tests.
var timeNow = func() int64 { return time.Now().UnixMilli() }

// statFile returns os.FileInfo for the given path.
func statFile(path string) (os.FileInfo, error) { return os.Stat(path) }

// baseName returns filepath.Base(path).
func baseName(path string) string { return filepath.Base(path) }

// mkdirAll calls os.MkdirAll(path, 0755).
func mkdirAll(path string) error { return os.MkdirAll(path, 0755) }

// copyFile copies src to dst. The caller must ensure dst's directory exists.
func copyFile(dst, src string) (int64, error) {
	sf, err := os.Open(src)
	if err != nil {
		return 0, err
	}
	defer sf.Close()

	df, err := os.Create(dst)
	if err != nil {
		return 0, err
	}
	defer df.Close()

	return df.ReadFrom(sf)
}
