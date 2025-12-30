//! Python bindings for VoiceTranscription.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::sync::Arc;
use tokio::runtime::Runtime;

use voice_transcription::{
    backends::LocalWhisperBackend, BackendConfig, TranscriptionClient as CoreClient,
    TranscriptionConfig as CoreConfig,
};

/// A segment of transcribed audio.
#[pyclass]
#[derive(Clone)]
pub struct Segment {
    #[pyo3(get)]
    pub text: String,
    #[pyo3(get)]
    pub start_seconds: f64,
    #[pyo3(get)]
    pub end_seconds: f64,
    #[pyo3(get)]
    pub speaker: Option<String>,
    #[pyo3(get)]
    pub confidence: Option<f64>,
}

#[pymethods]
impl Segment {
    fn __repr__(&self) -> String {
        format!(
            "Segment(text={:?}, start={:.2}, end={:.2})",
            self.text, self.start_seconds, self.end_seconds
        )
    }
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

/// The result of a transcription operation.
#[pyclass]
#[derive(Clone)]
pub struct TranscriptionResult {
    #[pyo3(get)]
    pub content: String,
    #[pyo3(get)]
    pub segments: Vec<Segment>,
    #[pyo3(get)]
    pub languages: Option<Vec<String>>,
    #[pyo3(get)]
    pub duration_seconds: Option<f64>,
    #[pyo3(get)]
    pub confidence: Option<f64>,
    #[pyo3(get)]
    pub speaker_count: Option<u32>,
}

#[pymethods]
impl TranscriptionResult {
    fn __repr__(&self) -> String {
        format!(
            "TranscriptionResult(content={:?}, segments={}, languages={:?})",
            if self.content.len() > 50 {
                format!("{}...", &self.content[..50])
            } else {
                self.content.clone()
            },
            self.segments.len(),
            self.languages
        )
    }
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

/// Configuration for a transcription request.
#[pyclass]
#[derive(Clone, Default)]
pub struct TranscriptionConfig {
    language: Option<String>,
    speaker_count: Option<u32>,
    word_timestamps: bool,
    model: Option<String>,
}

#[pymethods]
impl TranscriptionConfig {
    #[new]
    #[pyo3(signature = (language=None, speaker_count=None, word_timestamps=false, model=None))]
    fn new(
        language: Option<String>,
        speaker_count: Option<u32>,
        word_timestamps: bool,
        model: Option<String>,
    ) -> Self {
        Self {
            language,
            speaker_count,
            word_timestamps,
            model,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "TranscriptionConfig(language={:?}, speaker_count={:?})",
            self.language, self.speaker_count
        )
    }
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

/// Transcription client for audio transcription.
#[pyclass]
pub struct TranscriptionClient {
    client: Arc<CoreClient>,
    runtime: Arc<Runtime>,
}

#[pymethods]
impl TranscriptionClient {
    /// Create a new TranscriptionClient with a local Whisper backend.
    ///
    /// Args:
    ///     model_path: Path to the Whisper model file (.bin).
    ///
    /// Returns:
    ///     A new TranscriptionClient instance.
    ///
    /// Raises:
    ///     RuntimeError: If the model cannot be loaded.
    #[staticmethod]
    #[pyo3(signature = (model_path))]
    fn with_local_whisper(model_path: &str) -> PyResult<Self> {
        let backend_config = BackendConfig::new().with_model_path(model_path);

        let backend = LocalWhisperBackend::new(backend_config)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create backend: {}", e)))?;

        let runtime = Runtime::new()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?;

        Ok(Self {
            client: Arc::new(CoreClient::new(backend)),
            runtime: Arc::new(runtime),
        })
    }

    /// Get the name of the current backend.
    fn backend_name(&self) -> String {
        self.client.backend_name().to_string()
    }

    /// Get the features supported by the current backend.
    fn supported_features(&self) -> Vec<String> {
        self.client
            .supported_features()
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// Check if the backend is ready to transcribe.
    fn is_ready(&self) -> bool {
        self.client.is_ready()
    }

    /// Transcribe an audio file.
    ///
    /// Args:
    ///     audio_path: Path to the audio file.
    ///     config: Optional transcription configuration.
    ///
    /// Returns:
    ///     TranscriptionResult with the transcription.
    ///
    /// Raises:
    ///     RuntimeError: If transcription fails.
    #[pyo3(signature = (audio_path, config=None))]
    fn transcribe(
        &self,
        audio_path: &str,
        config: Option<TranscriptionConfig>,
    ) -> PyResult<TranscriptionResult> {
        let core_config = config.as_ref().map(|c| c.into()).unwrap_or_default();
        let client = self.client.clone();
        let path = audio_path.to_string();

        let result = self
            .runtime
            .block_on(async move { client.transcribe_with_config(&path, &core_config).await })
            .map_err(|e| PyRuntimeError::new_err(format!("Transcription failed: {}", e)))?;

        Ok(result.into())
    }

    /// Transcribe an audio file with a specific language.
    ///
    /// Args:
    ///     audio_path: Path to the audio file.
    ///     language: Language code (ISO 639-1, e.g., "en", "he").
    ///
    /// Returns:
    ///     TranscriptionResult with the transcription.
    ///
    /// Raises:
    ///     RuntimeError: If transcription fails.
    fn transcribe_with_language(
        &self,
        audio_path: &str,
        language: &str,
    ) -> PyResult<TranscriptionResult> {
        let config = TranscriptionConfig::new(Some(language.to_string()), None, false, None);
        self.transcribe(audio_path, Some(config))
    }

    fn __repr__(&self) -> String {
        format!(
            "TranscriptionClient(backend={:?}, ready={})",
            self.client.backend_name(),
            self.client.is_ready()
        )
    }
}

/// VoiceTranscription Python module.
#[pymodule]
fn voice_transcription(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Segment>()?;
    m.add_class::<TranscriptionResult>()?;
    m.add_class::<TranscriptionConfig>()?;
    m.add_class::<TranscriptionClient>()?;
    Ok(())
}
