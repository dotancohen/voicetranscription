//! Google Cloud Speech transcription backend.
//!
//! This backend uses the Google Cloud Speech-to-Text API (v2) with Chirp model.
//! Features:
//! - High-quality transcription with Chirp v3
//! - Multi-language support
//! - Automatic punctuation
//! - Speaker diarization (region dependent)

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Duration;
use tracing::{debug, info, warn};

use crate::backend::{BackendConfig, TranscriptionBackend};
use crate::error::{Result, TranscriptionError};
use crate::schema::{HasProviderSchema, OptionType, OptionValue, ProviderOption, ProviderSchema};
use crate::types::{Segment, TranscriptionConfig, TranscriptionResult};

const DEFAULT_LOCATION: &str = "us-central1";
const DEFAULT_MODEL: &str = "chirp";

/// Google Cloud Speech transcription backend.
pub struct GoogleCloudBackend {
    client: Client,
    access_token: String,
    project_id: String,
    location: String,
    model: String,
}

impl GoogleCloudBackend {
    /// Create a new Google Cloud Speech backend.
    ///
    /// # Arguments
    /// * `config` - Backend configuration containing credentials.
    ///
    /// # Required options:
    /// * `api_key` - Google Cloud access token or API key
    /// * `project_id` - Google Cloud project ID
    ///
    /// # Optional options:
    /// * `location` - Region (default: us-central1)
    /// * `model` - Model name (default: chirp)
    pub fn new(config: BackendConfig) -> Result<Self> {
        let access_token = config
            .api_key
            .ok_or_else(|| TranscriptionError::Configuration("Google Cloud access token required".into()))?;

        let project_id = config
            .options
            .get("project_id")
            .cloned()
            .ok_or_else(|| TranscriptionError::Configuration("Google Cloud project_id required".into()))?;

        let location = config
            .options
            .get("location")
            .cloned()
            .unwrap_or_else(|| DEFAULT_LOCATION.to_string());

        let model = config
            .model_name
            .unwrap_or_else(|| DEFAULT_MODEL.to_string());

        let client = Client::builder()
            .timeout(Duration::from_secs(300))
            .build()
            .map_err(|e| TranscriptionError::Backend(format!("Failed to create HTTP client: {}", e)))?;

        info!(
            "Google Cloud Speech backend initialized (project: {}, location: {}, model: {})",
            project_id, location, model
        );

        Ok(Self {
            client,
            access_token,
            project_id,
            location,
            model,
        })
    }

    /// Get the API endpoint URL.
    fn get_endpoint(&self) -> String {
        format!(
            "https://{}-speech.googleapis.com/v2/projects/{}/locations/{}/recognizers/_:recognize",
            self.location, self.project_id, self.location
        )
    }

    /// Build the recognition request.
    fn build_request(&self, audio_content: &[u8], config: &TranscriptionConfig) -> RecognizeRequest {
        let language_codes = config
            .language
            .as_ref()
            .map(|l| vec![l.clone()])
            .unwrap_or_else(|| vec!["en-US".to_string()]);

        let features = RecognitionFeatures {
            enable_automatic_punctuation: true,
            max_alternatives: 1,
            diarization_config: config.speaker_count.filter(|&c| c > 1).map(|c| {
                SpeakerDiarizationConfig {
                    min_speaker_count: 1,
                    max_speaker_count: c.min(6), // Google limit is 6
                }
            }),
        };

        RecognizeRequest {
            config: RecognitionConfig {
                auto_decoding_config: AutoDecodingConfig {},
                language_codes,
                model: self.model.clone(),
                features: Some(features),
            },
            content: base64::encode(audio_content),
        }
    }

    /// Parse the recognition response into a TranscriptionResult.
    fn parse_response(
        &self,
        response: RecognizeResponse,
        speaker_count: Option<u32>,
    ) -> TranscriptionResult {
        let mut full_text = String::new();
        let mut segments = Vec::new();

        for result in response.results.unwrap_or_default() {
            if let Some(alternatives) = result.alternatives {
                if let Some(alt) = alternatives.first() {
                    // Handle speaker diarization if available
                    if speaker_count.map(|c| c > 1).unwrap_or(false) {
                        if let Some(words) = &alt.words {
                            let formatted = self.format_diarized_text(words);
                            if !full_text.is_empty() {
                                full_text.push(' ');
                            }
                            full_text.push_str(&formatted);

                            // Convert words to segments
                            for word in words {
                                if let (Some(start), Some(end)) = (&word.start_offset, &word.end_offset) {
                                    segments.push(Segment {
                                        start: parse_duration(start),
                                        end: parse_duration(end),
                                        text: word.word.clone(),
                                        confidence: alt.confidence.unwrap_or(0.0),
                                    });
                                }
                            }
                        } else {
                            if !full_text.is_empty() {
                                full_text.push(' ');
                            }
                            full_text.push_str(&alt.transcript);
                        }
                    } else {
                        if !full_text.is_empty() {
                            full_text.push(' ');
                        }
                        full_text.push_str(&alt.transcript);
                    }
                }
            }
        }

        TranscriptionResult {
            text: full_text.trim().to_string(),
            segments: if segments.is_empty() {
                None
            } else {
                Some(segments)
            },
            language: None,
            duration: None,
        }
    }

    /// Format text with speaker labels.
    fn format_diarized_text(&self, words: &[WordInfo]) -> String {
        let mut result = String::new();
        let mut current_speaker: Option<&str> = None;
        let mut current_text = String::new();

        for word in words {
            let speaker = word.speaker_label.as_deref();

            if speaker != current_speaker {
                // New speaker, flush previous
                if !current_text.is_empty() {
                    if let Some(spk) = current_speaker {
                        result.push_str(&format!("Speaker {}: {}\n", spk, current_text.trim()));
                    } else {
                        result.push_str(&format!("{}\n", current_text.trim()));
                    }
                    current_text.clear();
                }
                current_speaker = speaker;
            }

            current_text.push_str(&word.word);
            current_text.push(' ');
        }

        // Flush remaining text
        if !current_text.is_empty() {
            if let Some(spk) = current_speaker {
                result.push_str(&format!("Speaker {}: {}", spk, current_text.trim()));
            } else {
                result.push_str(current_text.trim());
            }
        }

        result
    }
}

/// Parse a duration string like "1.5s" into seconds.
fn parse_duration(s: &str) -> f64 {
    s.trim_end_matches('s').parse().unwrap_or(0.0)
}

/// Base64 encoding helper (simple implementation).
mod base64 {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    pub fn encode(data: &[u8]) -> String {
        let mut result = String::new();
        let mut i = 0;

        while i < data.len() {
            let b0 = data[i] as u32;
            let b1 = if i + 1 < data.len() { data[i + 1] as u32 } else { 0 };
            let b2 = if i + 2 < data.len() { data[i + 2] as u32 } else { 0 };

            let triple = (b0 << 16) | (b1 << 8) | b2;

            result.push(ALPHABET[(triple >> 18) as usize & 0x3F] as char);
            result.push(ALPHABET[(triple >> 12) as usize & 0x3F] as char);

            if i + 1 < data.len() {
                result.push(ALPHABET[(triple >> 6) as usize & 0x3F] as char);
            } else {
                result.push('=');
            }

            if i + 2 < data.len() {
                result.push(ALPHABET[triple as usize & 0x3F] as char);
            } else {
                result.push('=');
            }

            i += 3;
        }

        result
    }
}

#[async_trait]
impl TranscriptionBackend for GoogleCloudBackend {
    fn name(&self) -> &str {
        "google_cloud"
    }

    fn supported_features(&self) -> Vec<&str> {
        vec!["punctuation", "diarization"]
    }

    fn is_ready(&self) -> bool {
        true
    }

    async fn transcribe(
        &self,
        audio_path: &Path,
        config: &TranscriptionConfig,
    ) -> Result<TranscriptionResult> {
        info!("Transcribing with Google Cloud Speech: {:?}", audio_path);

        // Read the audio file
        let audio_content = tokio::fs::read(audio_path)
            .await
            .map_err(|e| TranscriptionError::Io(e))?;

        // Check file size (Google has a 60-second limit for synchronous requests)
        // For longer files, we'd need to use async recognition with GCS
        if audio_content.len() > 10 * 1024 * 1024 {
            warn!("Audio file is large, transcription may fail. Consider using shorter files.");
        }

        // Build the request
        let request = self.build_request(&audio_content, config);

        debug!("Sending recognition request to {}", self.get_endpoint());

        // Send the request
        let response = self
            .client
            .post(&self.get_endpoint())
            .header("Authorization", format!("Bearer {}", self.access_token))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| TranscriptionError::Backend(format!("Request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(TranscriptionError::Backend(format!(
                "Recognition failed ({}): {}",
                status, error_text
            )));
        }

        let recognize_response: RecognizeResponse = response
            .json()
            .await
            .map_err(|e| TranscriptionError::Backend(format!("Failed to parse response: {}", e)))?;

        Ok(self.parse_response(recognize_response, config.speaker_count))
    }
}

impl HasProviderSchema for GoogleCloudBackend {
    fn get_provider_schema() -> ProviderSchema {
        ProviderSchema::new("google_cloud", "Google Cloud Speech")
            .with_option(
                ProviderOption::new("access_token", "Access Token", OptionType::Text)
                    .required()
                    .with_description(
                        "Google Cloud access token. Get with: gcloud auth print-access-token",
                    ),
            )
            .with_option(
                ProviderOption::new("project_id", "Project ID", OptionType::Text)
                    .required()
                    .with_description("Your Google Cloud project ID"),
            )
            .with_option(
                ProviderOption::new("location", "Location", OptionType::Select)
                    .with_default("us-central1")
                    .with_values(vec![
                        OptionValue::new("us-central1", "US Central (Iowa)"),
                        OptionValue::new("us-east1", "US East (South Carolina)"),
                        OptionValue::new("us-west1", "US West (Oregon)"),
                        OptionValue::new("europe-west1", "Europe West (Belgium)"),
                        OptionValue::new("europe-west2", "Europe West (London)"),
                        OptionValue::new("europe-west4", "Europe West (Netherlands)"),
                        OptionValue::new("asia-east1", "Asia East (Taiwan)"),
                        OptionValue::new("asia-northeast1", "Asia Northeast (Tokyo)"),
                        OptionValue::new("asia-southeast1", "Asia Southeast (Singapore)"),
                    ])
                    .with_description("Google Cloud region for Speech API"),
            )
            .with_option(
                ProviderOption::new("model", "Model", OptionType::Select)
                    .with_default("chirp")
                    .with_values(vec![
                        OptionValue::new("chirp", "Chirp (Best quality, latest)"),
                        OptionValue::new("chirp_2", "Chirp 2"),
                        OptionValue::new("long", "Long (For long audio)"),
                        OptionValue::new("short", "Short (For short audio)"),
                        OptionValue::new("telephony", "Telephony (Phone calls)"),
                        OptionValue::new("medical_dictation", "Medical Dictation"),
                        OptionValue::new("medical_conversation", "Medical Conversation"),
                    ])
                    .with_description("Speech recognition model to use"),
            )
            .with_option(
                ProviderOption::new("language", "Language", OptionType::Select)
                    .with_default("en-US")
                    .with_values(vec![
                        OptionValue::new("en-US", "English (US)"),
                        OptionValue::new("en-GB", "English (UK)"),
                        OptionValue::new("es-ES", "Spanish (Spain)"),
                        OptionValue::new("es-MX", "Spanish (Mexico)"),
                        OptionValue::new("fr-FR", "French"),
                        OptionValue::new("de-DE", "German"),
                        OptionValue::new("it-IT", "Italian"),
                        OptionValue::new("pt-BR", "Portuguese (Brazil)"),
                        OptionValue::new("ja-JP", "Japanese"),
                        OptionValue::new("ko-KR", "Korean"),
                        OptionValue::new("zh-CN", "Chinese (Simplified)"),
                        OptionValue::new("zh-TW", "Chinese (Traditional)"),
                        OptionValue::new("ru-RU", "Russian"),
                        OptionValue::new("ar-XA", "Arabic"),
                        OptionValue::new("he-IL", "Hebrew"),
                        OptionValue::new("hi-IN", "Hindi"),
                        OptionValue::new("tr-TR", "Turkish"),
                        OptionValue::new("pl-PL", "Polish"),
                        OptionValue::new("uk-UA", "Ukrainian"),
                        OptionValue::new("nl-NL", "Dutch"),
                        OptionValue::new("sv-SE", "Swedish"),
                    ])
                    .with_description("Language of the audio"),
            )
    }
}

// API request/response types

#[derive(Debug, Serialize)]
struct RecognizeRequest {
    config: RecognitionConfig,
    content: String, // Base64-encoded audio
}

#[derive(Debug, Serialize)]
struct RecognitionConfig {
    #[serde(rename = "autoDecodingConfig")]
    auto_decoding_config: AutoDecodingConfig,
    #[serde(rename = "languageCodes")]
    language_codes: Vec<String>,
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    features: Option<RecognitionFeatures>,
}

#[derive(Debug, Serialize)]
struct AutoDecodingConfig {}

#[derive(Debug, Serialize)]
struct RecognitionFeatures {
    #[serde(rename = "enableAutomaticPunctuation")]
    enable_automatic_punctuation: bool,
    #[serde(rename = "maxAlternatives")]
    max_alternatives: u32,
    #[serde(rename = "diarizationConfig", skip_serializing_if = "Option::is_none")]
    diarization_config: Option<SpeakerDiarizationConfig>,
}

#[derive(Debug, Serialize)]
struct SpeakerDiarizationConfig {
    #[serde(rename = "minSpeakerCount")]
    min_speaker_count: u32,
    #[serde(rename = "maxSpeakerCount")]
    max_speaker_count: u32,
}

#[derive(Debug, Deserialize)]
struct RecognizeResponse {
    results: Option<Vec<SpeechRecognitionResult>>,
}

#[derive(Debug, Deserialize)]
struct SpeechRecognitionResult {
    alternatives: Option<Vec<SpeechRecognitionAlternative>>,
}

#[derive(Debug, Deserialize)]
struct SpeechRecognitionAlternative {
    transcript: String,
    confidence: Option<f32>,
    words: Option<Vec<WordInfo>>,
}

#[derive(Debug, Deserialize)]
struct WordInfo {
    word: String,
    #[serde(rename = "startOffset")]
    start_offset: Option<String>,
    #[serde(rename = "endOffset")]
    end_offset: Option<String>,
    #[serde(rename = "speakerLabel")]
    speaker_label: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_config_requires_api_key() {
        let config = BackendConfig::new();
        let result = GoogleCloudBackend::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_backend_config_requires_project_id() {
        let config = BackendConfig::new().with_api_key("test-token");
        let result = GoogleCloudBackend::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_provider_schema() {
        let schema = GoogleCloudBackend::get_provider_schema();
        assert_eq!(schema.provider_id, "google_cloud");
        assert_eq!(schema.provider_name, "Google Cloud Speech");
        assert!(!schema.options.is_empty());

        // Check required options
        let required_ids: Vec<&str> = schema
            .options
            .iter()
            .filter(|o| o.required)
            .map(|o| o.id.as_str())
            .collect();

        assert!(required_ids.contains(&"access_token"));
        assert!(required_ids.contains(&"project_id"));
    }

    #[test]
    fn test_base64_encode() {
        assert_eq!(base64::encode(b"hello"), "aGVsbG8=");
        assert_eq!(base64::encode(b"a"), "YQ==");
        assert_eq!(base64::encode(b"ab"), "YWI=");
        assert_eq!(base64::encode(b"abc"), "YWJj");
    }

    #[test]
    fn test_parse_duration() {
        assert_eq!(parse_duration("1.5s"), 1.5);
        assert_eq!(parse_duration("10s"), 10.0);
        assert_eq!(parse_duration("0.123s"), 0.123);
    }
}
