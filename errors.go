package fileprocessor

import "errors"

// ErrNotFound is returned by [FileStore] implementations when a record is
// missing. Callers should use [errors.Is] to detect it.
var ErrNotFound = errors.New("fileprocessor: not found")

// ErrUnsupportedFileType is returned when the loader cannot determine a
// supported type for the given file.
var ErrUnsupportedFileType = errors.New("fileprocessor: unsupported file type")
