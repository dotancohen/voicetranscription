//! AssemblyAI cloud transcription backend.
//!
//! This backend uses the AssemblyAI API for cloud-based transcription.
//! Features:
//! - High-quality transcription
//! - Speaker diarization
//! - Language detection
//! - Punctuation and formatting

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Duration;
use tracing::{debug, info};

use crate::backend::{BackendConfig, TranscriptionBackend};
use crate::error::{Result, TranscriptionError};
use crate::schema::{HasProviderSchema, OptionType, OptionValue, ProviderOption, ProviderSchema};
use crate::types::{Segment, TranscriptionConfig, TranscriptionResult};

const ASSEMBLYAI_API_BASE: &str = "https://api.assemblyai.com/v2";
const POLL_INTERVAL_MS: u64 = 1000;
const MAX_POLL_ATTEMPTS: u32 = 600; // 10 minutes max

/// AssemblyAI transcription backend.
pub struct AssemblyAIBackend {
    client: Client,
    api_key: String,
}

impl AssemblyAIBackend {
    /// Create a new AssemblyAI backend.
    ///
    /// # Arguments
    /// * `config` - Backend configuration containing API key.
    ///
    /// # Errors
    /// Returns an error if the API key is not provided.
    pub fn new(config: BackendConfig) -> Result<Self> {
        let api_key = config
            .api_key
            .ok_or_else(|| TranscriptionError::InvalidConfig("AssemblyAI API key required".into()))?;

        let client = Client::builder()
            .timeout(Duration::from_secs(300))
            .build()
            .map_err(|e| TranscriptionError::ApiError(format!("Failed to create HTTP client: {}", e)))?;

        info!("AssemblyAI backend initialized");
        Ok(Self { client, api_key })
    }

    /// Upload an audio file to AssemblyAI.
    async fn upload_file(&self, audio_path: &Path) -> Result<String> {
        debug!("Uploading file to AssemblyAI: {:?}", audio_path);

        let file_content = tokio::fs::read(audio_path)
            .await
            .map_err(|e| TranscriptionError::IoError(e))?;

        let response = self
            .client
            .post(format!("{}/upload", ASSEMBLYAI_API_BASE))
            .header("authorization", &self.api_key)
            .header("content-type", "application/octet-stream")
            .body(file_content)
            .send()
            .await
            .map_err(|e| TranscriptionError::ApiError(format!("Upload failed: {}", e)))?;

        if !response.status().is_success() {
            let error_text: String = response.text().await.unwrap_or_default();
            return Err(TranscriptionError::ApiError(format!(
                "Upload failed: {}",
                error_text
            )));
        }

        let upload_response: UploadResponse = response
            .json()
            .await
            .map_err(|e| TranscriptionError::ApiError(format!("Failed to parse upload response: {}", e)))?;

        debug!("File uploaded successfully: {}", upload_response.upload_url);
        Ok(upload_response.upload_url)
    }

    /// Start a transcription job.
    async fn start_transcription(
        &self,
        audio_url: &str,
        config: &TranscriptionConfig,
    ) -> Result<String> {
        debug!("Starting transcription job");

        let request = TranscriptRequest {
            audio_url: audio_url.to_string(),
            speaker_labels: config.speaker_count.map(|c| c > 1).unwrap_or(false),
            speakers_expected: config.speaker_count.filter(|&c| c > 1),
            language_detection: config.language.is_none(),
            language_code: config.language.clone(),
            punctuate: true,
            format_text: true,
        };

        let response = self
            .client
            .post(format!("{}/transcript", ASSEMBLYAI_API_BASE))
            .header("authorization", &self.api_key)
            .json(&request)
            .send()
            .await
            .map_err(|e| TranscriptionError::ApiError(format!("Failed to start transcription: {}", e)))?;

        if !response.status().is_success() {
            let error_text: String = response.text().await.unwrap_or_default();
            return Err(TranscriptionError::ApiError(format!(
                "Failed to start transcription: {}",
                error_text
            )));
        }

        let transcript_response: TranscriptResponse = response
            .json()
            .await
            .map_err(|e| TranscriptionError::ApiError(format!("Failed to parse response: {}", e)))?;

        debug!("Transcription job started: {}", transcript_response.id);
        Ok(transcript_response.id)
    }

    /// Poll for transcription completion.
    async fn poll_transcription(&self, transcript_id: &str) -> Result<TranscriptResponse> {
        debug!("Polling for transcription completion: {}", transcript_id);

        for attempt in 0..MAX_POLL_ATTEMPTS {
            let response = self
                .client
                .get(format!("{}/transcript/{}", ASSEMBLYAI_API_BASE, transcript_id))
                .header("authorization", &self.api_key)
                .send()
                .await
                .map_err(|e| TranscriptionError::ApiError(format!("Poll request failed: {}", e)))?;

            if !response.status().is_success() {
                let error_text: String = response.text().await.unwrap_or_default();
                return Err(TranscriptionError::ApiError(format!(
                    "Poll request failed: {}",
                    error_text
                )));
            }

            let transcript: TranscriptResponse = response
                .json()
                .await
                .map_err(|e| TranscriptionError::ApiError(format!("Failed to parse poll response: {}", e)))?;

            match transcript.status.as_str() {
                "completed" => {
                    debug!("Transcription completed after {} attempts", attempt + 1);
                    return Ok(transcript);
                }
                "error" => {
                    return Err(TranscriptionError::ApiError(format!(
                        "Transcription failed: {}",
                        transcript.error.unwrap_or_else(|| "Unknown error".into())
                    )));
                }
                status => {
                    debug!("Transcription status: {} (attempt {})", status, attempt + 1);
                    tokio::time::sleep(Duration::from_millis(POLL_INTERVAL_MS)).await;
                }
            }
        }

        Err(TranscriptionError::ApiTimeout(
            "Transcription timed out".into(),
        ))
    }

    /// Format the transcription result with speaker labels if available.
    fn format_result(&self, transcript: TranscriptResponse) -> TranscriptionResult {
        let text = if let Some(utterances) = &transcript.utterances {
            // Format with speaker labels
            utterances
                .iter()
                .map(|u| format!("Speaker {}: {}", u.speaker, u.text))
                .collect::<Vec<_>>()
                .join("\n")
        } else {
            transcript.text.unwrap_or_default()
        };

        // Convert words to segments if available
        let segments: Vec<Segment> = transcript
            .words
            .map(|words| {
                words
                    .iter()
                    .map(|w| {
                        Segment::new(
                            w.text.clone(),
                            w.start as f64 / 1000.0,
                            w.end as f64 / 1000.0,
                        )
                        .with_confidence(w.confidence as f64)
                    })
                    .collect()
            })
            .unwrap_or_default();

        let mut result = TranscriptionResult::with_segments(text, segments);

        if let Some(lang) = transcript.language_code {
            result = result.with_languages(vec![lang]);
        }

        if let Some(duration) = transcript.audio_duration {
            result = result.with_duration(duration as f64);
        }

        result
    }
}

#[async_trait]
impl TranscriptionBackend for AssemblyAIBackend {
    fn name(&self) -> &str {
        "assemblyai"
    }

    fn supported_features(&self) -> Vec<&str> {
        vec!["diarization", "language_detection", "punctuation"]
    }

    fn is_ready(&self) -> bool {
        true
    }

    async fn transcribe(
        &self,
        audio_path: &Path,
        config: &TranscriptionConfig,
    ) -> Result<TranscriptionResult> {
        info!("Transcribing with AssemblyAI: {:?}", audio_path);

        // Upload the file
        let audio_url = self.upload_file(audio_path).await?;

        // Start transcription
        let transcript_id = self.start_transcription(&audio_url, config).await?;

        // Poll for completion
        let transcript = self.poll_transcription(&transcript_id).await?;

        // Format and return result
        Ok(self.format_result(transcript))
    }
}

impl HasProviderSchema for AssemblyAIBackend {
    fn get_provider_schema() -> ProviderSchema {
        ProviderSchema::new("assemblyai", "AssemblyAI")
            .with_option(
                ProviderOption::new("api_key", "API Key", OptionType::Text)
                    .required()
                    .with_description("Your AssemblyAI API key from https://www.assemblyai.com/"),
            )
            .with_option(
                ProviderOption::new("speaker_labels", "Speaker Diarization", OptionType::Checkbox)
                    .with_default("false")
                    .with_description("Enable speaker diarization to identify different speakers"),
            )
            .with_option(
                ProviderOption::new("language", "Language", OptionType::Select)
                    .with_default("")
                    .with_values(vec![
                        OptionValue::new("", "Auto-detect"),
                        OptionValue::new("en", "English"),
                        OptionValue::new("es", "Spanish"),
                        OptionValue::new("fr", "French"),
                        OptionValue::new("de", "German"),
                        OptionValue::new("it", "Italian"),
                        OptionValue::new("pt", "Portuguese"),
                        OptionValue::new("nl", "Dutch"),
                        OptionValue::new("ja", "Japanese"),
                        OptionValue::new("ko", "Korean"),
                        OptionValue::new("zh", "Chinese"),
                        OptionValue::new("ru", "Russian"),
                        OptionValue::new("ar", "Arabic"),
                        OptionValue::new("he", "Hebrew"),
                        OptionValue::new("hi", "Hindi"),
                        OptionValue::new("tr", "Turkish"),
                        OptionValue::new("pl", "Polish"),
                        OptionValue::new("uk", "Ukrainian"),
                    ])
                    .with_description("Language of the audio (leave empty for auto-detection)"),
            )
    }
}

// API request/response types

#[derive(Debug, Deserialize)]
struct UploadResponse {
    upload_url: String,
}

#[derive(Debug, Serialize)]
struct TranscriptRequest {
    audio_url: String,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    speaker_labels: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    speakers_expected: Option<u32>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    language_detection: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    language_code: Option<String>,
    punctuate: bool,
    format_text: bool,
}

#[derive(Debug, Deserialize)]
struct TranscriptResponse {
    id: String,
    status: String,
    text: Option<String>,
    error: Option<String>,
    language_code: Option<String>,
    audio_duration: Option<u64>,
    words: Option<Vec<WordInfo>>,
    utterances: Option<Vec<Utterance>>,
}

#[derive(Debug, Deserialize)]
struct WordInfo {
    text: String,
    start: u64,
    end: u64,
    confidence: f32,
}

#[derive(Debug, Deserialize)]
struct Utterance {
    speaker: String,
    text: String,
    start: u64,
    end: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_config_requires_api_key() {
        let config = BackendConfig::new();
        let result = AssemblyAIBackend::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_provider_schema() {
        let schema = AssemblyAIBackend::get_provider_schema();
        assert_eq!(schema.provider_id, "assemblyai");
        assert_eq!(schema.provider_name, "AssemblyAI");
        assert!(!schema.options.is_empty());

        // Check that api_key is required
        let api_key_opt = schema.options.iter().find(|o| o.id == "api_key");
        assert!(api_key_opt.is_some());
        assert!(api_key_opt.unwrap().required);
    }
}
