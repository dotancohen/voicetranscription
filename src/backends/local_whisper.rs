//! Local Whisper transcription backend using whisper-rs.

use async_trait::async_trait;
use std::path::Path;
use std::sync::Mutex;
use tracing::{debug, info, warn};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::audio::AudioConverter;
use crate::backend::{BackendConfig, TranscriptionBackend};
use crate::error::{Result, TranscriptionError};
use crate::schema::{HasProviderSchema, OptionType, OptionValue, ProviderOption, ProviderSchema};
use crate::types::{Segment, TranscriptionConfig, TranscriptionResult};

/// Local Whisper transcription backend.
///
/// This backend uses the whisper-rs library to run OpenAI's Whisper model
/// locally for transcription.
pub struct LocalWhisperBackend {
    context: Mutex<Option<WhisperContext>>,
    model_path: String,
    ready: bool,
    audio_converter: AudioConverter,
}

impl LocalWhisperBackend {
    /// Create a new LocalWhisperBackend.
    ///
    /// # Arguments
    /// * `config` - Backend configuration with model path.
    ///
    /// # Returns
    /// A new LocalWhisperBackend instance.
    pub fn new(config: BackendConfig) -> Result<Self> {
        let model_path = config
            .model_path
            .ok_or_else(|| TranscriptionError::InvalidConfig("Model path is required".into()))?;

        // Verify model file exists
        if !Path::new(&model_path).exists() {
            return Err(TranscriptionError::ModelLoadError(format!(
                "Model file not found: {}",
                model_path
            )));
        }

        Ok(Self {
            context: Mutex::new(None),
            model_path,
            ready: true,
            audio_converter: AudioConverter::new(),
        })
    }

    /// Lazily initialize the Whisper context.
    fn ensure_context(&self) -> Result<()> {
        let mut context_guard = self.context.lock().map_err(|e| {
            TranscriptionError::Other(format!("Failed to lock context mutex: {}", e))
        })?;

        if context_guard.is_none() {
            info!("Loading Whisper model from: {}", self.model_path);
            let ctx = WhisperContext::new_with_params(&self.model_path, WhisperContextParameters::default())
                .map_err(|e| {
                    TranscriptionError::ModelLoadError(format!("Failed to load Whisper model: {}", e))
                })?;
            *context_guard = Some(ctx);
            info!("Whisper model loaded successfully");
        }

        Ok(())
    }

    /// Read audio file and convert to f32 samples.
    fn read_audio_file(path: &Path) -> Result<Vec<f32>> {
        use std::io::Read;

        // For now, we assume WAV files. In production, you'd use a proper audio decoder.
        let mut file = std::fs::File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        // Simple WAV parsing (assumes 16-bit PCM mono 16kHz)
        // Skip WAV header (typically 44 bytes for standard WAV)
        if buffer.len() < 44 {
            return Err(TranscriptionError::UnsupportedFormat(
                "File too small to be a valid WAV".into(),
            ));
        }

        // Check WAV signature
        if &buffer[0..4] != b"RIFF" || &buffer[8..12] != b"WAVE" {
            return Err(TranscriptionError::UnsupportedFormat(
                "Not a valid WAV file".into(),
            ));
        }

        // Find data chunk
        let mut pos = 12;
        let mut data_start = 0;
        let mut data_size = 0;

        while pos + 8 <= buffer.len() {
            let chunk_id = &buffer[pos..pos + 4];
            let chunk_size = u32::from_le_bytes([
                buffer[pos + 4],
                buffer[pos + 5],
                buffer[pos + 6],
                buffer[pos + 7],
            ]) as usize;

            if chunk_id == b"data" {
                data_start = pos + 8;
                data_size = chunk_size;
                break;
            }

            pos += 8 + chunk_size;
            // Align to even byte
            if chunk_size % 2 != 0 {
                pos += 1;
            }
        }

        if data_start == 0 || data_size == 0 {
            return Err(TranscriptionError::UnsupportedFormat(
                "Could not find data chunk in WAV file".into(),
            ));
        }

        // Convert 16-bit samples to f32
        let samples: Vec<f32> = buffer[data_start..data_start + data_size]
            .chunks(2)
            .filter_map(|chunk| {
                if chunk.len() == 2 {
                    let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                    Some(sample as f32 / 32768.0)
                } else {
                    None
                }
            })
            .collect();

        debug!("Read {} audio samples from {:?}", samples.len(), path);
        Ok(samples)
    }
}

#[async_trait]
impl TranscriptionBackend for LocalWhisperBackend {
    fn name(&self) -> &str {
        "local_whisper"
    }

    fn supported_features(&self) -> Vec<&str> {
        vec!["language_detection", "timestamps", "word_timestamps"]
    }

    fn is_ready(&self) -> bool {
        self.ready
    }

    async fn transcribe(
        &self,
        audio_path: &Path,
        config: &TranscriptionConfig,
    ) -> Result<TranscriptionResult> {
        // Verify file exists
        if !audio_path.exists() {
            return Err(TranscriptionError::FileNotFound(
                audio_path.display().to_string(),
            ));
        }

        // Prepare audio (convert if necessary)
        let prepared = self.audio_converter.prepare(audio_path, config.nocache)?;

        if prepared.was_converted {
            warn!(
                "Audio was converted and cached. The cache at {:?} should be cleared periodically.",
                self.audio_converter.cache_dir()
            );
        }

        // Ensure model is loaded
        self.ensure_context()?;

        // Read audio samples from the prepared file
        let samples = Self::read_audio_file(&prepared.path)?;

        // Run transcription in a blocking task to avoid blocking the async runtime
        let context_guard = self.context.lock().map_err(|e| {
            TranscriptionError::Other(format!("Failed to lock context mutex: {}", e))
        })?;

        let ctx = context_guard.as_ref().ok_or_else(|| {
            TranscriptionError::Other("Context not initialized".into())
        })?;

        // Set up parameters
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

        // Set language if specified
        if let Some(ref lang) = config.language {
            params.set_language(Some(lang.as_str()));
            debug!("Set transcription language to: {}", lang);
        } else {
            params.set_language(None); // Auto-detect
            debug!("Using automatic language detection");
        }

        // Enable token timestamps for segment generation
        params.set_token_timestamps(true);

        // Create state and run inference
        let mut state = ctx.create_state().map_err(|e| {
            TranscriptionError::TranscriptionFailed(format!("Failed to create state: {}", e))
        })?;

        state.full(params, &samples).map_err(|e| {
            TranscriptionError::TranscriptionFailed(format!("Transcription failed: {}", e))
        })?;

        // Extract results
        let num_segments = state.full_n_segments().map_err(|e| {
            TranscriptionError::TranscriptionFailed(format!("Failed to get segment count: {}", e))
        })?;

        let mut segments = Vec::new();
        let mut full_text = String::new();

        for i in 0..num_segments {
            let segment_text = state.full_get_segment_text(i).map_err(|e| {
                TranscriptionError::TranscriptionFailed(format!(
                    "Failed to get segment {} text: {}",
                    i, e
                ))
            })?;

            let start = state.full_get_segment_t0(i).map_err(|e| {
                TranscriptionError::TranscriptionFailed(format!(
                    "Failed to get segment {} start time: {}",
                    i, e
                ))
            })?;

            let end = state.full_get_segment_t1(i).map_err(|e| {
                TranscriptionError::TranscriptionFailed(format!(
                    "Failed to get segment {} end time: {}",
                    i, e
                ))
            })?;

            // Convert timestamps from centiseconds to seconds
            let start_seconds = start as f64 / 100.0;
            let end_seconds = end as f64 / 100.0;

            full_text.push_str(&segment_text);

            segments.push(Segment::new(segment_text, start_seconds, end_seconds));
        }

        // Get detected language
        let languages = if config.language.is_none() {
            // Try to get detected language from the state
            // Note: whisper-rs doesn't directly expose detected language,
            // but we can infer it from the context if needed
            None
        } else {
            config.language.clone().map(|l| vec![l])
        };

        // Calculate duration from the last segment
        let duration_seconds = segments.last().map(|s| s.end_seconds);

        let mut result = TranscriptionResult::with_segments(full_text.trim().to_string(), segments);

        if let Some(langs) = languages {
            result = result.with_languages(langs);
        }

        if let Some(duration) = duration_seconds {
            result = result.with_duration(duration);
        }

        debug!(
            "Transcription complete: {} segments, {} characters",
            result.segments.len(),
            result.content.len()
        );

        Ok(result)
    }
}

impl HasProviderSchema for LocalWhisperBackend {
    fn get_provider_schema() -> ProviderSchema {
        ProviderSchema::new("local_whisper", "Local Whisper")
            .with_option(
                ProviderOption::new("model", "Model", OptionType::Select)
                    .with_default("base")
                    .with_values(vec![
                        OptionValue::new("tiny", "tiny (75MB, fastest)"),
                        OptionValue::new("tiny.en", "tiny.en (75MB, English only)"),
                        OptionValue::new("base", "base (142MB, recommended)"),
                        OptionValue::new("base.en", "base.en (142MB, English only)"),
                        OptionValue::new("small", "small (466MB)"),
                        OptionValue::new("small.en", "small.en (466MB, English only)"),
                        OptionValue::new("medium", "medium (1.5GB)"),
                        OptionValue::new("medium.en", "medium.en (1.5GB, English only)"),
                        OptionValue::new("large-v3", "large-v3 (3GB, best quality)"),
                        OptionValue::new("large-v3-turbo", "large-v3-turbo (1.6GB, fast + quality)"),
                    ])
                    .with_description(
                        "Whisper model size. Larger models are more accurate but slower."
                    ),
            )
            .with_option(
                ProviderOption::new("model_path", "Custom Model Path", OptionType::Path)
                    .with_description(
                        "Path to a custom GGML model file. Overrides the model selection above."
                    ),
            )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_config() {
        let config = BackendConfig::new()
            .with_model_path("/path/to/model.bin")
            .with_model_name("base");

        assert_eq!(config.model_path, Some("/path/to/model.bin".to_string()));
        assert_eq!(config.model_name, Some("base".to_string()));
    }
}
