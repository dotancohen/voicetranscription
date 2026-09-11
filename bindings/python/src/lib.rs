//! Python bindings for VoiceTranscription.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::sync::Arc;
use tokio::runtime::Runtime;

use ::voice_transcription::{
    audio::{get_cache_info as core_get_cache_info, clear_cache as core_clear_cache},
    enable_debug_logging as core_enable_debug_logging,
    BackendConfig, TranscriptionClient as CoreClient,
    TranscriptionConfig as CoreConfig, Segment as CoreSegment, TranscriptionResult as CoreResult,
    HasProviderSchema,
    schema::{
        OptionType as CoreOptionType, OptionValue as CoreOptionValue,
        ProviderOption as CoreProviderOption, ProviderSchema as CoreProviderSchema,
    },
};

#[cfg(feature = "local_whisper")]
use ::voice_transcription::backends::LocalWhisperBackend;

#[cfg(feature = "speechtext_ai")]
use ::voice_transcription::backends::{SpeechTextAIBackend, SpeechTextAIOptions as CoreSpeechTextAIOptions};

#[cfg(feature = "assemblyai")]
use ::voice_transcription::backends::AssemblyAIBackend;

#[cfg(feature = "google_cloud")]
use ::voice_transcription::backends::GoogleCloudBackend;
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
    beam_size: Option<u32>,
}

#[pymethods]
impl TranscriptionConfig {
    #[new]
    #[pyo3(signature = (language=None, speaker_count=None, word_timestamps=false, model=None, nocache=false, beam_size=None))]
    fn new(
        language: Option<String>,
        speaker_count: Option<u32>,
        word_timestamps: bool,
        model: Option<String>,
        nocache: bool,
        beam_size: Option<u32>,
    ) -> Self {
        Self {
            language,
            speaker_count,
            word_timestamps,
            model,
            nocache,
            beam_size,
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
        if let Some(n) = c.beam_size {
            config = config.with_beam_size(n);
        }
        config
    }
}

/// Options specific to SpeechText.AI transcription provider.
#[cfg(feature = "speechtext_ai")]
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

#[cfg(feature = "speechtext_ai")]
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
    #[cfg(feature = "local_whisper")]
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

    /// Create a new TranscriptionClient with the AssemblyAI backend.
    ///
    /// Args:
    ///     api_key: Your AssemblyAI API key.
    ///
    /// Returns:
    ///     A new TranscriptionClient instance.
    ///
    /// Raises:
    ///     RuntimeError: If the backend cannot be initialized.
    #[cfg(feature = "assemblyai")]
    #[staticmethod]
    #[pyo3(signature = (api_key))]
    fn with_assemblyai(api_key: &str) -> PyResult<Self> {
        let backend_config = BackendConfig::new().with_api_key(api_key);

        let backend = AssemblyAIBackend::new(backend_config)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create backend: {}", e)))?;

        let runtime = Runtime::new()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?;

        Ok(Self {
            client: Arc::new(CoreClient::new(backend)),
            runtime: Arc::new(runtime),
        })
    }

    /// Create a new TranscriptionClient with the Google Cloud Speech backend.
    ///
    /// Args:
    ///     access_token: Google Cloud access token (from `gcloud auth print-access-token`).
    ///     project_id: Your Google Cloud project ID.
    ///     location: Optional region (default: us-central1).
    ///     model: Optional model name (default: chirp).
    ///
    /// Returns:
    ///     A new TranscriptionClient instance.
    ///
    /// Raises:
    ///     RuntimeError: If the backend cannot be initialized.
    #[cfg(feature = "google_cloud")]
    #[staticmethod]
    #[pyo3(signature = (access_token, project_id, location=None, model=None))]
    fn with_google_cloud(
        access_token: &str,
        project_id: &str,
        location: Option<&str>,
        model: Option<&str>,
    ) -> PyResult<Self> {
        let mut backend_config = BackendConfig::new()
            .with_api_key(access_token)
            .with_option("project_id", project_id);

        if let Some(loc) = location {
            backend_config = backend_config.with_option("location", loc);
        }
        if let Some(m) = model {
            backend_config = backend_config.with_model_name(m);
        }

        let backend = GoogleCloudBackend::new(backend_config)
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
    #[cfg(feature = "speechtext_ai")]
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
        let config = TranscriptionConfig::new(Some(language.to_string()), None, false, None, false, None);
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

// ============================================================================
// Provider Schema Types
// ============================================================================

/// A value option for Select-type options.
#[pyclass]
#[derive(Clone)]
pub struct OptionValue {
    #[pyo3(get)]
    pub value: String,
    #[pyo3(get)]
    pub label: String,
}

#[pymethods]
impl OptionValue {
    fn __repr__(&self) -> String {
        format!("OptionValue(value={:?}, label={:?})", self.value, self.label)
    }
}

impl From<CoreOptionValue> for OptionValue {
    fn from(v: CoreOptionValue) -> Self {
        Self {
            value: v.value,
            label: v.label,
        }
    }
}

/// A configurable option for a transcription provider.
#[pyclass]
#[derive(Clone)]
pub struct ProviderOption {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub label: String,
    #[pyo3(get)]
    pub option_type: String,
    #[pyo3(get)]
    pub required: bool,
    #[pyo3(get)]
    pub default: String,
    #[pyo3(get)]
    pub values: Vec<OptionValue>,
    #[pyo3(get)]
    pub description: Option<String>,
}

#[pymethods]
impl ProviderOption {
    fn __repr__(&self) -> String {
        format!(
            "ProviderOption(id={:?}, label={:?}, type={:?})",
            self.id, self.label, self.option_type
        )
    }
}

impl From<CoreProviderOption> for ProviderOption {
    fn from(o: CoreProviderOption) -> Self {
        let option_type = match o.option_type {
            CoreOptionType::Text => "text",
            CoreOptionType::Number => "number",
            CoreOptionType::Select => "select",
            CoreOptionType::Checkbox => "checkbox",
            CoreOptionType::Path => "path",
        }
        .to_string();

        Self {
            id: o.id,
            label: o.label,
            option_type,
            required: o.required,
            default: o.default,
            values: o.values.into_iter().map(|v| v.into()).collect(),
            description: o.description,
        }
    }
}

/// Schema describing a transcription provider's configurable options.
#[pyclass]
#[derive(Clone)]
pub struct ProviderSchema {
    #[pyo3(get)]
    pub provider_id: String,
    #[pyo3(get)]
    pub provider_name: String,
    #[pyo3(get)]
    pub options: Vec<ProviderOption>,
}

#[pymethods]
impl ProviderSchema {
    fn __repr__(&self) -> String {
        format!(
            "ProviderSchema(id={:?}, name={:?}, options={})",
            self.provider_id,
            self.provider_name,
            self.options.len()
        )
    }
}

impl From<CoreProviderSchema> for ProviderSchema {
    fn from(s: CoreProviderSchema) -> Self {
        Self {
            provider_id: s.provider_id,
            provider_name: s.provider_name,
            options: s.options.into_iter().map(|o| o.into()).collect(),
        }
    }
}

/// Get all available provider schemas.
///
/// Returns:
///     List of ProviderSchema objects describing each available provider.
#[pyfunction]
fn get_provider_schemas() -> Vec<ProviderSchema> {
    let mut schemas = Vec::new();

    #[cfg(feature = "local_whisper")]
    schemas.push(LocalWhisperBackend::get_provider_schema().into());

    #[cfg(feature = "speechtext_ai")]
    schemas.push(SpeechTextAIBackend::get_provider_schema().into());

    #[cfg(feature = "assemblyai")]
    schemas.push(AssemblyAIBackend::get_provider_schema().into());

    #[cfg(feature = "google_cloud")]
    schemas.push(GoogleCloudBackend::get_provider_schema().into());

    schemas
}

/// Get a list of available backend names.
///
/// Returns:
///     List of backend identifiers that are available.
#[pyfunction]
fn get_available_backends() -> Vec<String> {
    let mut backends = Vec::new();

    #[cfg(feature = "local_whisper")]
    backends.push("local_whisper".to_string());

    #[cfg(feature = "speechtext_ai")]
    backends.push("speechtext_ai".to_string());

    #[cfg(feature = "assemblyai")]
    backends.push("assemblyai".to_string());

    #[cfg(feature = "google_cloud")]
    backends.push("google_cloud".to_string());

    backends
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
    // Transcription types
    m.add_class::<Segment>()?;
    m.add_class::<TranscriptionResult>()?;
    m.add_class::<TranscriptionConfig>()?;
    m.add_class::<TranscriptionClient>()?;

    #[cfg(feature = "speechtext_ai")]
    m.add_class::<SpeechTextAIOptions>()?;

    // Cache management
    m.add_class::<CacheInfo>()?;
    m.add_function(wrap_pyfunction!(get_cache_info, m)?)?;
    m.add_function(wrap_pyfunction!(clear_cache, m)?)?;

    // Provider schema types
    m.add_class::<OptionValue>()?;
    m.add_class::<ProviderOption>()?;
    m.add_class::<ProviderSchema>()?;
    m.add_function(wrap_pyfunction!(get_provider_schemas, m)?)?;
    m.add_function(wrap_pyfunction!(get_available_backends, m)?)?;
    m.add_function(wrap_pyfunction!(enable_debug_logging, m)?)?;

    Ok(())
}
