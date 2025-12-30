//! Async transcription client.

use std::path::Path;
use std::sync::Arc;
use tracing::{debug, info};

use crate::backend::TranscriptionBackend;
use crate::error::{Result, TranscriptionError};
use crate::types::{TranscriptionConfig, TranscriptionResult};

/// Async transcription client.
///
/// This is the main entry point for transcription operations. It wraps a backend
/// and provides a convenient async API for transcription.
pub struct TranscriptionClient {
    backend: Arc<dyn TranscriptionBackend>,
}

impl TranscriptionClient {
    /// Create a new transcription client with the given backend.
    pub fn new(backend: impl TranscriptionBackend + 'static) -> Self {
        Self {
            backend: Arc::new(backend),
        }
    }

    /// Create a new transcription client from an Arc'd backend.
    pub fn from_arc(backend: Arc<dyn TranscriptionBackend>) -> Self {
        Self { backend }
    }

    /// Get the name of the current backend.
    pub fn backend_name(&self) -> &str {
        self.backend.name()
    }

    /// Get the features supported by the current backend.
    pub fn supported_features(&self) -> Vec<&str> {
        self.backend.supported_features()
    }

    /// Check if the backend is ready to transcribe.
    pub fn is_ready(&self) -> bool {
        self.backend.is_ready()
    }

    /// Transcribe an audio file with default configuration.
    ///
    /// # Arguments
    /// * `audio_path` - Path to the audio file to transcribe.
    ///
    /// # Returns
    /// The transcription result, or an error if transcription failed.
    pub async fn transcribe(&self, audio_path: impl AsRef<Path>) -> Result<TranscriptionResult> {
        self.transcribe_with_config(audio_path, &TranscriptionConfig::default())
            .await
    }

    /// Transcribe an audio file with custom configuration.
    ///
    /// # Arguments
    /// * `audio_path` - Path to the audio file to transcribe.
    /// * `config` - Configuration for this transcription request.
    ///
    /// # Returns
    /// The transcription result, or an error if transcription failed.
    pub async fn transcribe_with_config(
        &self,
        audio_path: impl AsRef<Path>,
        config: &TranscriptionConfig,
    ) -> Result<TranscriptionResult> {
        let path = audio_path.as_ref();
        info!("Starting transcription of: {:?}", path);

        if !self.backend.is_ready() {
            return Err(TranscriptionError::BackendNotAvailable(
                self.backend.name().to_string(),
            ));
        }

        let result = self.backend.transcribe(path, config).await?;

        debug!(
            "Transcription complete: {} characters, {} segments",
            result.content.len(),
            result.segments.len()
        );

        Ok(result)
    }

    /// Transcribe an audio file with a specific language.
    ///
    /// # Arguments
    /// * `audio_path` - Path to the audio file to transcribe.
    /// * `language` - Language code (ISO 639-1, e.g., "en", "he").
    ///
    /// # Returns
    /// The transcription result, or an error if transcription failed.
    pub async fn transcribe_with_language(
        &self,
        audio_path: impl AsRef<Path>,
        language: &str,
    ) -> Result<TranscriptionResult> {
        let config = TranscriptionConfig::new().with_language(language);
        self.transcribe_with_config(audio_path, &config).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Segment;
    use async_trait::async_trait;

    /// Mock backend for testing.
    struct MockBackend {
        result: TranscriptionResult,
    }

    impl MockBackend {
        fn new(content: &str) -> Self {
            Self {
                result: TranscriptionResult::new(content.to_string()),
            }
        }
    }

    #[async_trait]
    impl TranscriptionBackend for MockBackend {
        fn name(&self) -> &str {
            "mock"
        }

        fn supported_features(&self) -> Vec<&str> {
            vec!["test_feature"]
        }

        fn is_ready(&self) -> bool {
            true
        }

        async fn transcribe(
            &self,
            _audio_path: &Path,
            _config: &TranscriptionConfig,
        ) -> Result<TranscriptionResult> {
            Ok(self.result.clone())
        }
    }

    #[tokio::test]
    async fn test_client_transcribe() {
        let backend = MockBackend::new("Hello, world!");
        let client = TranscriptionClient::new(backend);

        assert_eq!(client.backend_name(), "mock");
        assert!(client.is_ready());

        let result = client.transcribe("/fake/path.wav").await.unwrap();
        assert_eq!(result.content, "Hello, world!");
    }

    #[tokio::test]
    async fn test_client_with_language() {
        let backend = MockBackend::new("Test content");
        let client = TranscriptionClient::new(backend);

        let result = client
            .transcribe_with_language("/fake/path.wav", "en")
            .await
            .unwrap();
        assert_eq!(result.content, "Test content");
    }

    #[tokio::test]
    async fn test_client_with_config() {
        let backend = MockBackend::new("Configured result");
        let client = TranscriptionClient::new(backend);

        let config = TranscriptionConfig::new()
            .with_language("he")
            .with_speaker_count(2);

        let result = client
            .transcribe_with_config("/fake/path.wav", &config)
            .await
            .unwrap();
        assert_eq!(result.content, "Configured result");
    }

    #[tokio::test]
    async fn test_client_supported_features() {
        let backend = MockBackend::new("test");
        let client = TranscriptionClient::new(backend);

        let features = client.supported_features();
        assert_eq!(features, vec!["test_feature"]);
    }

    #[tokio::test]
    async fn test_client_from_arc() {
        let backend = Arc::new(MockBackend::new("arc test"));
        let client = TranscriptionClient::from_arc(backend);

        assert_eq!(client.backend_name(), "mock");
        let result = client.transcribe("/fake/path.wav").await.unwrap();
        assert_eq!(result.content, "arc test");
    }

    /// Mock backend that is not ready.
    struct NotReadyBackend;

    #[async_trait]
    impl TranscriptionBackend for NotReadyBackend {
        fn name(&self) -> &str {
            "not_ready"
        }

        fn supported_features(&self) -> Vec<&str> {
            vec![]
        }

        fn is_ready(&self) -> bool {
            false
        }

        async fn transcribe(
            &self,
            _audio_path: &Path,
            _config: &TranscriptionConfig,
        ) -> Result<TranscriptionResult> {
            unreachable!("Should not be called when not ready")
        }
    }

    #[tokio::test]
    async fn test_client_not_ready_error() {
        let backend = NotReadyBackend;
        let client = TranscriptionClient::new(backend);

        assert!(!client.is_ready());

        let result = client.transcribe("/fake/path.wav").await;
        assert!(result.is_err());
        match result.unwrap_err() {
            TranscriptionError::BackendNotAvailable(name) => {
                assert_eq!(name, "not_ready");
            }
            _ => panic!("Expected BackendNotAvailable error"),
        }
    }

    /// Mock backend that returns an error.
    struct ErrorBackend;

    #[async_trait]
    impl TranscriptionBackend for ErrorBackend {
        fn name(&self) -> &str {
            "error"
        }

        fn supported_features(&self) -> Vec<&str> {
            vec![]
        }

        fn is_ready(&self) -> bool {
            true
        }

        async fn transcribe(
            &self,
            _audio_path: &Path,
            _config: &TranscriptionConfig,
        ) -> Result<TranscriptionResult> {
            Err(TranscriptionError::TranscriptionFailed("simulated error".to_string()))
        }
    }

    #[tokio::test]
    async fn test_client_transcription_error() {
        let backend = ErrorBackend;
        let client = TranscriptionClient::new(backend);

        let result = client.transcribe("/fake/path.wav").await;
        assert!(result.is_err());
        match result.unwrap_err() {
            TranscriptionError::TranscriptionFailed(msg) => {
                assert_eq!(msg, "simulated error");
            }
            _ => panic!("Expected TranscriptionFailed error"),
        }
    }

    /// Mock backend with segments.
    struct SegmentedBackend;

    #[async_trait]
    impl TranscriptionBackend for SegmentedBackend {
        fn name(&self) -> &str {
            "segmented"
        }

        fn supported_features(&self) -> Vec<&str> {
            vec!["timestamps", "diarization"]
        }

        fn is_ready(&self) -> bool {
            true
        }

        async fn transcribe(
            &self,
            _audio_path: &Path,
            _config: &TranscriptionConfig,
        ) -> Result<TranscriptionResult> {
            let segments = vec![
                Segment::new("Hello".to_string(), 0.0, 0.5).with_speaker("A".to_string()),
                Segment::new("World".to_string(), 0.5, 1.0).with_speaker("B".to_string()),
            ];
            Ok(TranscriptionResult::with_segments("Hello World".to_string(), segments)
                .with_languages(vec!["en".to_string()])
                .with_duration(1.0)
                .with_speaker_count(2))
        }
    }

    #[tokio::test]
    async fn test_client_with_segments() {
        let backend = SegmentedBackend;
        let client = TranscriptionClient::new(backend);

        let features = client.supported_features();
        assert!(features.contains(&"timestamps"));
        assert!(features.contains(&"diarization"));

        let result = client.transcribe("/fake/path.wav").await.unwrap();
        assert_eq!(result.content, "Hello World");
        assert_eq!(result.segments.len(), 2);
        assert_eq!(result.segments[0].text, "Hello");
        assert_eq!(result.segments[0].speaker, Some("A".to_string()));
        assert_eq!(result.segments[1].text, "World");
        assert_eq!(result.segments[1].speaker, Some("B".to_string()));
        assert_eq!(result.languages, Some(vec!["en".to_string()]));
        assert_eq!(result.duration_seconds, Some(1.0));
        assert_eq!(result.speaker_count, Some(2));
    }
}
