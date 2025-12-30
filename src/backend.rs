//! Transcription backend trait and implementations.

use async_trait::async_trait;
use std::path::Path;

use crate::error::Result;
use crate::types::{TranscriptionConfig, TranscriptionResult};

/// Trait for transcription backends.
///
/// This trait defines the interface that all transcription backends must implement.
/// Backends can be local (e.g., Whisper) or cloud-based (e.g., Google Cloud Speech,
/// AssemblyAI).
#[async_trait]
pub trait TranscriptionBackend: Send + Sync {
    /// Get the name of this backend.
    fn name(&self) -> &str;

    /// Get the features supported by this backend.
    fn supported_features(&self) -> Vec<&str>;

    /// Check if this backend is ready to transcribe.
    fn is_ready(&self) -> bool;

    /// Transcribe an audio file.
    ///
    /// # Arguments
    /// * `audio_path` - Path to the audio file to transcribe.
    /// * `config` - Configuration for this transcription request.
    ///
    /// # Returns
    /// The transcription result, or an error if transcription failed.
    async fn transcribe(
        &self,
        audio_path: &Path,
        config: &TranscriptionConfig,
    ) -> Result<TranscriptionResult>;
}

/// Configuration for initializing a backend.
#[derive(Debug, Clone, Default)]
pub struct BackendConfig {
    /// Path to the model file (for local backends).
    pub model_path: Option<String>,
    /// Model name/size to use (e.g., "base", "small", "medium", "large").
    pub model_name: Option<String>,
    /// API key (for cloud backends).
    pub api_key: Option<String>,
    /// Additional backend-specific options.
    pub options: std::collections::HashMap<String, String>,
}

impl BackendConfig {
    /// Create a new default backend configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the model path.
    pub fn with_model_path(mut self, path: impl Into<String>) -> Self {
        self.model_path = Some(path.into());
        self
    }

    /// Set the model name.
    pub fn with_model_name(mut self, name: impl Into<String>) -> Self {
        self.model_name = Some(name.into());
        self
    }

    /// Set the API key.
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Add a custom option.
    pub fn with_option(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.options.insert(key.into(), value.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_config_default() {
        let config = BackendConfig::default();
        assert!(config.model_path.is_none());
        assert!(config.model_name.is_none());
        assert!(config.api_key.is_none());
        assert!(config.options.is_empty());
    }

    #[test]
    fn test_backend_config_new() {
        let config = BackendConfig::new();
        assert!(config.model_path.is_none());
        assert!(config.model_name.is_none());
        assert!(config.api_key.is_none());
        assert!(config.options.is_empty());
    }

    #[test]
    fn test_backend_config_with_model_path() {
        let config = BackendConfig::new().with_model_path("/path/to/model.bin");
        assert_eq!(config.model_path, Some("/path/to/model.bin".to_string()));
    }

    #[test]
    fn test_backend_config_with_model_name() {
        let config = BackendConfig::new().with_model_name("large-v3");
        assert_eq!(config.model_name, Some("large-v3".to_string()));
    }

    #[test]
    fn test_backend_config_with_api_key() {
        let config = BackendConfig::new().with_api_key("secret-key-123");
        assert_eq!(config.api_key, Some("secret-key-123".to_string()));
    }

    #[test]
    fn test_backend_config_with_option() {
        let config = BackendConfig::new()
            .with_option("threads", "4")
            .with_option("device", "cuda");
        assert_eq!(config.options.get("threads"), Some(&"4".to_string()));
        assert_eq!(config.options.get("device"), Some(&"cuda".to_string()));
    }

    #[test]
    fn test_backend_config_builder_chain() {
        let config = BackendConfig::new()
            .with_model_path("/models/whisper.bin")
            .with_model_name("base")
            .with_api_key("key123")
            .with_option("threads", "8");
        assert_eq!(config.model_path, Some("/models/whisper.bin".to_string()));
        assert_eq!(config.model_name, Some("base".to_string()));
        assert_eq!(config.api_key, Some("key123".to_string()));
        assert_eq!(config.options.get("threads"), Some(&"8".to_string()));
    }

    #[test]
    fn test_backend_config_debug() {
        let config = BackendConfig::new().with_model_name("test");
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("BackendConfig"));
        assert!(debug_str.contains("test"));
    }

    #[test]
    fn test_backend_config_clone() {
        let config = BackendConfig::new()
            .with_model_path("/path")
            .with_option("key", "value");
        let cloned = config.clone();
        assert_eq!(config.model_path, cloned.model_path);
        assert_eq!(config.options, cloned.options);
    }
}
