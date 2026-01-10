//! Python bindings for VoiceTranscription.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::sync::Arc;
use tokio::runtime::Runtime;

use ::voice_transcription::{
    audio::{get_cache_info as core_get_cache_info, clear_cache as core_clear_cache},
    backends::{LocalWhisperBackend, SpeechTextAIBackend, SpeechTextAIOptions as CoreSpeechTextAIOptions},
    enable_debug_logging as core_enable_debug_logging,
    BackendConfig, TranscriptionClient as CoreClient,
    TranscriptionConfig as CoreConfig, Segment as CoreSegment, TranscriptionResult as CoreResult,
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

impl From<CoreSegment> for Segment {
    fn from(s: CoreSegment) -> Self {
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

impl From<CoreResult> for TranscriptionResult {
    fn from(r: CoreResult) -> Self {
        Self {
            content: r.content,
            segments: r.segments.into_iter().map(|s: CoreSegment| s.into()).collect(),
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
    nocache: bool,
}

#[pymethods]
impl TranscriptionConfig {
    #[new]
    #[pyo3(signature = (language=None, speaker_count=None, word_timestamps=false, model=None, nocache=false))]
    fn new(
        language: Option<String>,
        speaker_count: Option<u32>,
        word_timestamps: bool,
        model: Option<String>,
        nocache: bool,
    ) -> Self {
        Self {
            language,
            speaker_count,
            word_timestamps,
            model,
            nocache,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "TranscriptionConfig(language={:?}, speaker_count={:?}, nocache={})",
            self.language, self.speaker_count, self.nocache
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
        if c.nocache {
            config = config.with_nocache();
        }
        config
    }
}

/// Options specific to SpeechText.AI transcription provider.
#[pyclass]
#[derive(Clone)]
pub struct SpeechTextAIOptions {
    #[pyo3(get)]
    pub punctuation: bool,
    #[pyo3(get)]
    pub summary: bool,
    #[pyo3(get)]
    pub highlights: bool,
    #[pyo3(get)]
    pub number_of_speakers: u32,
    #[pyo3(get)]
    pub caption_output: Option<String>,
    #[pyo3(get)]
    pub custom_vocabulary: Vec<String>,
}

#[pymethods]
impl SpeechTextAIOptions {
    #[new]
    #[pyo3(signature = (punctuation=true, summary=false, highlights=false, number_of_speakers=0, caption_output=None, custom_vocabulary=None))]
    fn new(
        punctuation: bool,
        summary: bool,
        highlights: bool,
        number_of_speakers: u32,
        caption_output: Option<String>,
        custom_vocabulary: Option<Vec<String>>,
    ) -> Self {
        Self {
            punctuation,
            summary,
            highlights,
            number_of_speakers,
            caption_output,
            custom_vocabulary: custom_vocabulary.unwrap_or_default(),
        }
    }

    /// Convert options to JSON string.
    fn to_json(&self) -> String {
        let core_opts = CoreSpeechTextAIOptions {
            punctuation: self.punctuation,
            summary: self.summary,
            highlights: self.highlights,
            number_of_speakers: self.number_of_speakers,
            caption_output: self.caption_output.clone(),
            custom_vocabulary: self.custom_vocabulary.clone(),
        };
        core_opts.to_json()
    }

    fn __repr__(&self) -> String {
        format!(
            "SpeechTextAIOptions(punctuation={}, summary={}, highlights={}, speakers={})",
            self.punctuation, self.summary, self.highlights, self.number_of_speakers
        )
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

    /// Create a new TranscriptionClient with SpeechText.AI cloud backend.
    ///
    /// Args:
    ///     api_key: SpeechText.AI API key.
    ///     punctuation: Enable punctuation (default: True).
    ///     summary: Generate summary (default: False).
    ///     highlights: Extract highlights (default: False).
    ///
    /// Returns:
    ///     A new TranscriptionClient instance.
    ///
    /// Raises:
    ///     RuntimeError: If the backend cannot be created.
    #[staticmethod]
    #[pyo3(signature = (api_key, punctuation=true, summary=false, highlights=false))]
    fn with_speechtext_ai(
        api_key: &str,
        punctuation: bool,
        summary: bool,
        highlights: bool,
    ) -> PyResult<Self> {
        let backend_config = BackendConfig::new()
            .with_api_key(api_key)
            .with_option("punctuation", if punctuation { "true" } else { "false" })
            .with_option("summary", if summary { "true" } else { "false" })
            .with_option("highlights", if highlights { "true" } else { "false" });

        let backend = SpeechTextAIBackend::new(backend_config)
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
        let config = TranscriptionConfig::new(Some(language.to_string()), None, false, None, false);
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

/// Information about the audio cache.
#[pyclass]
#[derive(Clone)]
pub struct CacheInfo {
    #[pyo3(get)]
    pub path: String,
    #[pyo3(get)]
    pub size_bytes: u64,
    #[pyo3(get)]
    pub file_count: usize,
}

#[pymethods]
impl CacheInfo {
    /// Get human-readable size string.
    fn size_human(&self) -> String {
        const KB: u64 = 1024;
        const MB: u64 = KB * 1024;
        const GB: u64 = MB * 1024;

        if self.size_bytes >= GB {
            format!("{:.2} GB", self.size_bytes as f64 / GB as f64)
        } else if self.size_bytes >= MB {
            format!("{:.2} MB", self.size_bytes as f64 / MB as f64)
        } else if self.size_bytes >= KB {
            format!("{:.2} KB", self.size_bytes as f64 / KB as f64)
        } else {
            format!("{} bytes", self.size_bytes)
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "CacheInfo(path={:?}, size={}, files={})",
            self.path,
            self.size_human(),
            self.file_count
        )
    }
}

/// Get information about the audio conversion cache.
///
/// Returns:
///     CacheInfo with path, size, and file count.
///
/// Raises:
///     RuntimeError: If cache info cannot be retrieved.
#[pyfunction]
fn get_cache_info() -> PyResult<CacheInfo> {
    let info = core_get_cache_info()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get cache info: {}", e)))?;

    Ok(CacheInfo {
        path: info.path.display().to_string(),
        size_bytes: info.size_bytes,
        file_count: info.file_count,
    })
}

/// Clear the audio conversion cache.
///
/// Returns:
///     Number of files removed.
///
/// Raises:
///     RuntimeError: If cache cannot be cleared.
#[pyfunction]
fn clear_cache() -> PyResult<usize> {
    core_clear_cache()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to clear cache: {}", e)))
}

/// Enable debug logging for transcription operations.
///
/// This enables detailed logging of HTTP requests and responses,
/// which is useful for debugging API issues with cloud providers.
///
/// Call this before any transcription operations to see debug output.
/// Multiple calls are safe - only the first call has an effect.
#[pyfunction]
fn enable_debug_logging() {
    core_enable_debug_logging();
}

/// VoiceTranscription Python module.
#[pymodule]
fn voice_transcription(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Segment>()?;
    m.add_class::<TranscriptionResult>()?;
    m.add_class::<TranscriptionConfig>()?;
    m.add_class::<TranscriptionClient>()?;
    m.add_class::<SpeechTextAIOptions>()?;
    m.add_class::<CacheInfo>()?;
    m.add_function(wrap_pyfunction!(get_cache_info, m)?)?;
    m.add_function(wrap_pyfunction!(clear_cache, m)?)?;
    m.add_function(wrap_pyfunction!(enable_debug_logging, m)?)?;
    Ok(())
}
