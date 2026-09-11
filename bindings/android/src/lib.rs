//! Android/Kotlin bindings for VoiceTranscription via UniFFI (proc-macro
//! interface; the Kotlin side is generated from the compiled library with
//! `uniffi-bindgen generate --library`).
//!
//! The Android app has no ffmpeg, so callers hand this crate audio that is
//! already a 16 kHz mono 16-bit WAV file; anything else is rejected by the
//! core library's converter.

use std::sync::Arc;
use tokio::runtime::Runtime;

use voice_transcription::{
    backends::LocalWhisperBackend, BackendConfig,
    TranscriptionClient as CoreClient, TranscriptionConfig as CoreConfig,
};

uniffi::setup_scaffolding!("voice_transcription");

/// Error type for UniFFI bindings.
#[derive(Debug, thiserror::Error, uniffi::Error)]
#[uniffi(flat_error)]
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
    #[error("Audio conversion failed: {0}")]
    ConversionFailed(String),
    #[error("I/O error: {0}")]
    IoError(String),
    #[error("Transcription stopped by the user")]
    Cancelled,
    #[error("Other error: {0}")]
    Other(String),
}

impl From<voice_transcription::TranscriptionError> for TranscriptionError {
    fn from(e: voice_transcription::TranscriptionError) -> Self {
        use voice_transcription::TranscriptionError as E;
        match e {
            E::FileNotFound(s) => Self::FileNotFound(s),
            E::UnsupportedFormat(s) => Self::UnsupportedFormat(s),
            E::ModelLoadError(s) => Self::ModelLoadError(s),
            E::TranscriptionFailed(s) => Self::TranscriptionFailed(s),
            E::BackendNotAvailable(s) => Self::BackendNotAvailable(s),
            E::InvalidConfig(s) => Self::InvalidConfig(s),
            E::ConversionFailed(s) => Self::ConversionFailed(s),
            E::ApiError(s) | E::ApiTimeout(s) => Self::Other(s),
            E::IoError(e) => Self::IoError(e.to_string()),
            E::Cancelled => Self::Cancelled,
            E::Other(s) => Self::Other(s),
        }
    }
}

/// A segment of transcribed audio.
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

/// The result of a transcription.
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

/// Configuration for one transcription request.
#[derive(Debug, Clone, Default, uniffi::Record)]
pub struct TranscriptionConfig {
    /// ISO 639-1 code ("he", "en", "ar", "ru"); None = detect automatically.
    pub language: Option<String>,
    pub speaker_count: Option<u32>,
    pub word_timestamps: bool,
    pub model: Option<String>,
    /// Beam search width; None or 1 = greedy, 5 = most accurate in practice.
    pub beam_size: Option<u32>,
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
        if let Some(n) = c.beam_size {
            config = config.with_beam_size(n);
        }
        config
    }
}

/// Transcription client exposed to Kotlin. One instance holds one loaded model.
#[derive(uniffi::Object)]
pub struct TranscriptionClient {
    client: Arc<CoreClient>,
    runtime: Runtime,
}

/// Ask the transcription that is running now to stop.
///
/// It stops between windows of audio, within a second or so, and the call
/// that was running returns `Cancelled`. Nothing partial is kept. This is
/// what the "Stop" button on the phone's notification calls.
///
/// The flag stays raised until [`clear_transcription_cancel`] lowers it, so
/// a queue of files stops as a whole rather than one file at a time.
#[uniffi::export]
pub fn request_transcription_cancel() {
    voice_transcription::request_cancel();
}

/// Lower the stop flag. Call this before starting a batch of work that
/// should actually run.
#[uniffi::export]
pub fn clear_transcription_cancel() {
    voice_transcription::clear_cancel();
}

/// Whether a stop has been asked for and not yet cleared.
#[uniffi::export]
pub fn transcription_cancel_requested() -> bool {
    voice_transcription::cancel_requested()
}

/// Load a ggml Whisper model from `model_path` (a file the app downloaded).
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
    /// Name of the backend ("local_whisper").
    pub fn backend_name(&self) -> String {
        self.client.backend_name().to_string()
    }

    /// Features the backend supports.
    pub fn supported_features(&self) -> Vec<String> {
        self.client
            .supported_features()
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// Whether the model is loaded and ready.
    pub fn is_ready(&self) -> bool {
        self.client.is_ready()
    }

    /// Transcribe a 16 kHz mono WAV file with automatic language detection.
    pub fn transcribe(&self, audio_path: String) -> Result<TranscriptionResult, TranscriptionError> {
        let client = self.client.clone();
        let result = self
            .runtime
            .block_on(async move { client.transcribe(&audio_path).await })?;
        Ok(result.into())
    }

    /// Transcribe with an explicit configuration (language, beam size, ...).
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

    /// Transcribe in a given language (greedy decoding).
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
