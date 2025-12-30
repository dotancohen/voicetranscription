//! Error types for the transcription library.

use thiserror::Error;

/// Errors that can occur during transcription.
#[derive(Error, Debug)]
pub enum TranscriptionError {
    /// The audio file could not be found.
    #[error("Audio file not found: {0}")]
    FileNotFound(String),

    /// The audio file format is not supported.
    #[error("Unsupported audio format: {0}")]
    UnsupportedFormat(String),

    /// The model could not be loaded.
    #[error("Failed to load model: {0}")]
    ModelLoadError(String),

    /// The transcription operation failed.
    #[error("Transcription failed: {0}")]
    TranscriptionFailed(String),

    /// The backend is not available or not configured.
    #[error("Backend not available: {0}")]
    BackendNotAvailable(String),

    /// Invalid configuration provided.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// An I/O error occurred.
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// An unexpected error occurred.
    #[error("Unexpected error: {0}")]
    Other(String),
}

/// Result type alias for transcription operations.
pub type Result<T> = std::result::Result<T, TranscriptionError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display_file_not_found() {
        let err = TranscriptionError::FileNotFound("/path/to/file.wav".to_string());
        assert_eq!(err.to_string(), "Audio file not found: /path/to/file.wav");
    }

    #[test]
    fn test_error_display_unsupported_format() {
        let err = TranscriptionError::UnsupportedFormat("mp4".to_string());
        assert_eq!(err.to_string(), "Unsupported audio format: mp4");
    }

    #[test]
    fn test_error_display_model_load_error() {
        let err = TranscriptionError::ModelLoadError("model not found".to_string());
        assert_eq!(err.to_string(), "Failed to load model: model not found");
    }

    #[test]
    fn test_error_display_transcription_failed() {
        let err = TranscriptionError::TranscriptionFailed("timeout".to_string());
        assert_eq!(err.to_string(), "Transcription failed: timeout");
    }

    #[test]
    fn test_error_display_backend_not_available() {
        let err = TranscriptionError::BackendNotAvailable("whisper".to_string());
        assert_eq!(err.to_string(), "Backend not available: whisper");
    }

    #[test]
    fn test_error_display_invalid_config() {
        let err = TranscriptionError::InvalidConfig("missing model path".to_string());
        assert_eq!(err.to_string(), "Invalid configuration: missing model path");
    }

    #[test]
    fn test_error_display_other() {
        let err = TranscriptionError::Other("unknown error".to_string());
        assert_eq!(err.to_string(), "Unexpected error: unknown error");
    }

    #[test]
    fn test_error_from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err: TranscriptionError = io_err.into();
        assert!(matches!(err, TranscriptionError::IoError(_)));
    }

    #[test]
    fn test_result_type_ok() {
        let result: Result<i32> = Ok(42);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn test_result_type_err() {
        let result: Result<i32> = Err(TranscriptionError::Other("test".to_string()));
        assert!(result.is_err());
    }
}
