//! Android/Kotlin bindings for VoiceTranscription via UniFFI.

use std::sync::Arc;
use tokio::runtime::Runtime;

use voice_transcription::{
    backends::LocalWhisperBackend, BackendConfig,
    TranscriptionClient as CoreClient, TranscriptionConfig as CoreConfig,
};

uniffi::include_scaffolding!("voice_transcription");

/// Error type for UniFFI bindings.
#[derive(Debug, thiserror::Error)]
pub enum TranscriptionError {
    #[error("File not found: {0}")]
    FileNotFound(String),
    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),
    #[error("Model load error: {0}")]
    ModelLoadError(String),
    #[error("Transcription failed: {0}")]
    TranscriptionFailed(String),
    #[error("Backend not available: {0}")]
    BackendNotAvailable(String),
    #[error("Invalid config: {0}")]
    InvalidConfig(String),
    #[error("I/O error: {0}")]
    IoError(String),
    #[error("Other error: {0}")]
    Other(String),
}

impl From<voice_transcription::TranscriptionError> for TranscriptionError {
    fn from(e: voice_transcription::TranscriptionError) -> Self {
        match e {
            voice_transcription::TranscriptionError::FileNotFound(s) => Self::FileNotFound(s),
            voice_transcription::TranscriptionError::UnsupportedFormat(s) => Self::UnsupportedFormat(s),
            voice_transcription::TranscriptionError::ModelLoadError(s) => Self::ModelLoadError(s),
            voice_transcription::TranscriptionError::TranscriptionFailed(s) => Self::TranscriptionFailed(s),
            voice_transcription::TranscriptionError::BackendNotAvailable(s) => Self::BackendNotAvailable(s),
            voice_transcription::TranscriptionError::InvalidConfig(s) => Self::InvalidConfig(s),
            voice_transcription::TranscriptionError::IoError(e) => Self::IoError(e.to_string()),
            voice_transcription::TranscriptionError::Other(s) => Self::Other(s),
        }
    }
}

/// A segment of transcribed audio (UniFFI-compatible).
#[derive(Debug, Clone, uniffi::Record)]
pub struct Segment {
    pub text: String,
    pub start_seconds: f64,
    pub end_seconds: f64,
    pub speaker: Option<String>,
    pub confidence: Option<f64>,
}

impl From<voice_transcription::Segment> for Segment {
    fn from(s: voice_transcription::Segment) -> Self {
        Self {
            text: s.text,
            start_seconds: s.start_seconds,
            end_seconds: s.end_seconds,
            speaker: s.speaker,
            confidence: s.confidence,
        }
    }
}

/// The result of a transcription operation (UniFFI-compatible).
#[derive(Debug, Clone, uniffi::Record)]
pub struct TranscriptionResult {
    pub content: String,
    pub segments: Vec<Segment>,
    pub languages: Option<Vec<String>>,
    pub duration_seconds: Option<f64>,
    pub confidence: Option<f64>,
    pub speaker_count: Option<u32>,
}

impl From<voice_transcription::TranscriptionResult> for TranscriptionResult {
    fn from(r: voice_transcription::TranscriptionResult) -> Self {
        Self {
            content: r.content,
            segments: r.segments.into_iter().map(|s| s.into()).collect(),
            languages: r.languages,
            duration_seconds: r.duration_seconds,
            confidence: r.confidence,
            speaker_count: r.speaker_count,
        }
    }
}

/// Configuration for a transcription request (UniFFI-compatible).
#[derive(Debug, Clone, Default, uniffi::Record)]
pub struct TranscriptionConfig {
    pub language: Option<String>,
    pub speaker_count: Option<u32>,
    pub word_timestamps: bool,
    pub model: Option<String>,
}

impl From<&TranscriptionConfig> for CoreConfig {
    fn from(c: &TranscriptionConfig) -> Self {
        let mut config = CoreConfig::new();
        if let Some(ref lang) = c.language {
            config = config.with_language(lang);
        }
        if let Some(count) = c.speaker_count {
            config = config.with_speaker_count(count);
        }
        if c.word_timestamps {
            config = config.with_word_timestamps();
        }
        if let Some(ref model) = c.model {
            config = config.with_model(model);
        }
        config
    }
}

/// Transcription client exposed to Kotlin via UniFFI.
#[derive(uniffi::Object)]
pub struct TranscriptionClient {
    client: Arc<CoreClient>,
    runtime: Runtime,
}

/// Create a new TranscriptionClient with a local Whisper backend.
#[uniffi::export]
pub fn create_local_whisper_client(model_path: String) -> Result<Arc<TranscriptionClient>, TranscriptionError> {
    let backend_config = BackendConfig::new().with_model_path(&model_path);

    let backend = LocalWhisperBackend::new(backend_config)
        .map_err(|e| TranscriptionError::ModelLoadError(e.to_string()))?;

    let runtime = Runtime::new()
        .map_err(|e| TranscriptionError::Other(format!("Failed to create runtime: {}", e)))?;

    Ok(Arc::new(TranscriptionClient {
        client: Arc::new(CoreClient::new(backend)),
        runtime,
    }))
}

#[uniffi::export]
impl TranscriptionClient {
    /// Get the name of the current backend.
    pub fn backend_name(&self) -> String {
        self.client.backend_name().to_string()
    }

    /// Get the features supported by the current backend.
    pub fn supported_features(&self) -> Vec<String> {
        self.client
            .supported_features()
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// Check if the backend is ready to transcribe.
    pub fn is_ready(&self) -> bool {
        self.client.is_ready()
    }

    /// Transcribe an audio file with default configuration.
    pub fn transcribe(&self, audio_path: String) -> Result<TranscriptionResult, TranscriptionError> {
        let client = self.client.clone();
        let result = self
            .runtime
            .block_on(async move { client.transcribe(&audio_path).await })?;
        Ok(result.into())
    }

    /// Transcribe an audio file with custom configuration.
    pub fn transcribe_with_config(
        &self,
        audio_path: String,
        config: TranscriptionConfig,
    ) -> Result<TranscriptionResult, TranscriptionError> {
        let client = self.client.clone();
        let core_config: CoreConfig = (&config).into();
        let result = self
            .runtime
            .block_on(async move { client.transcribe_with_config(&audio_path, &core_config).await })?;
        Ok(result.into())
    }

    /// Transcribe an audio file with a specific language.
    pub fn transcribe_with_language(
        &self,
        audio_path: String,
        language: String,
    ) -> Result<TranscriptionResult, TranscriptionError> {
        let client = self.client.clone();
        let result = self
            .runtime
            .block_on(async move { client.transcribe_with_language(&audio_path, &language).await })?;
        Ok(result.into())
    }
}
