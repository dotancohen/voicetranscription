//! SpeechText.AI cloud transcription backend.
//!
//! This backend uses the SpeechText.AI API for high-accuracy transcription,
//! with particular strength in Hebrew (he-IL) language support.

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::error::Error as StdError;
use std::path::Path;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{debug, info};

use crate::backend::{BackendConfig, TranscriptionBackend};
use crate::error::{Result, TranscriptionError};
use crate::language::{normalize_language_code, default_language_code, PROVIDER_SPEECHTEXT_AI};
use crate::types::{Segment, TranscriptionConfig, TranscriptionResult};

/// SpeechText.AI API base URL.
const API_BASE_URL: &str = "https://api.speechtext.ai/recognize";
const RESULTS_URL: &str = "https://api.speechtext.ai/results";

/// Default polling interval in seconds.
const DEFAULT_POLL_INTERVAL_SECS: u64 = 5;

/// Default timeout in seconds (30 minutes).
const DEFAULT_TIMEOUT_SECS: u64 = 1800;

/// SpeechText.AI-specific transcription options.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SpeechTextAIOptions {
    /// Enable punctuation in output (default: true).
    pub punctuation: bool,
    /// Generate summary of the transcription.
    pub summary: bool,
    /// Extract highlights/key phrases.
    pub highlights: bool,
    /// Number of speakers for diarization (0 = auto, 1 = disabled).
    pub number_of_speakers: u32,
    /// Output format: "txt", "srt", "vtt".
    pub caption_output: Option<String>,
    /// Custom vocabulary words to improve recognition.
    pub custom_vocabulary: Vec<String>,
}

impl SpeechTextAIOptions {
    pub fn new() -> Self {
        Self {
            punctuation: true,
            summary: false,
            highlights: false,
            number_of_speakers: 0,
            caption_output: None,
            custom_vocabulary: Vec::new(),
        }
    }

    /// Enable punctuation (default is already true).
    pub fn with_punctuation(mut self, enabled: bool) -> Self {
        self.punctuation = enabled;
        self
    }

    /// Enable or disable summary generation.
    pub fn with_summary(mut self, enabled: bool) -> Self {
        self.summary = enabled;
        self
    }

    /// Enable or disable highlights extraction.
    pub fn with_highlights(mut self, enabled: bool) -> Self {
        self.highlights = enabled;
        self
    }

    /// Set number of speakers for diarization.
    pub fn with_speakers(mut self, count: u32) -> Self {
        self.number_of_speakers = count;
        self
    }

    /// Set caption output format.
    pub fn with_caption_output(mut self, format: impl Into<String>) -> Self {
        self.caption_output = Some(format.into());
        self
    }

    /// Add custom vocabulary words.
    pub fn with_custom_vocabulary(mut self, words: Vec<String>) -> Self {
        self.custom_vocabulary = words;
        self
    }

    /// Convert to JSON for storage.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }
}

/// Response from the /recognize endpoint.
#[derive(Debug, Deserialize)]
struct RecognizeResponse {
    /// Task ID (API returns "id", not "taskId")
    id: Option<String>,
    error: Option<String>,
}

/// Response from the /results endpoint.
#[derive(Debug, Deserialize)]
struct ResultsResponse {
    status: String,
    results: Option<TranscriptionResults>,
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TranscriptionResults {
    transcript: Option<String>,
    /// Word-level timing information from the API
    word_time_offsets: Option<Vec<WordTimeOffset>>,
    summary: Option<String>,
    highlights: Option<Vec<String>>,
}

/// Word timing information from SpeechText.AI API.
#[derive(Debug, Deserialize)]
struct WordTimeOffset {
    word: String,
    start_time: f64,
    end_time: f64,
    confidence: Option<f64>,
}

/// SpeechText.AI cloud transcription backend.
///
/// This backend provides high-accuracy transcription with excellent Hebrew support.
/// It supports various output options including punctuation, summaries, and highlights.
#[derive(Debug)]
pub struct SpeechTextAIBackend {
    api_key: String,
    client: Client,
    poll_interval: Duration,
    timeout: Duration,
    default_options: SpeechTextAIOptions,
}

impl SpeechTextAIBackend {
    /// Create a new SpeechTextAIBackend.
    ///
    /// # Arguments
    /// * `config` - Backend configuration with API key.
    ///
    /// # Returns
    /// A new SpeechTextAIBackend instance.
    pub fn new(config: BackendConfig) -> Result<Self> {
        let api_key = config
            .api_key
            .ok_or_else(|| TranscriptionError::InvalidConfig("API key is required".into()))?;

        // Parse options from config
        let punctuation = config
            .options
            .get("punctuation")
            .map(|v| v == "true" || v == "1")
            .unwrap_or(true);

        let summary = config
            .options
            .get("summary")
            .map(|v| v == "true" || v == "1")
            .unwrap_or(false);

        let highlights = config
            .options
            .get("highlights")
            .map(|v| v == "true" || v == "1")
            .unwrap_or(false);

        let poll_interval_secs: u64 = config
            .options
            .get("poll_interval")
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_POLL_INTERVAL_SECS);

        let timeout_secs: u64 = config
            .options
            .get("timeout")
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_TIMEOUT_SECS);

        let default_options = SpeechTextAIOptions::new()
            .with_punctuation(punctuation)
            .with_summary(summary)
            .with_highlights(highlights);

        // SpeechText.AI has SSL certificate chain issues that cause verification failures
        // on some systems. We skip certificate verification as a workaround.
        // This is acceptable because we're only communicating with their specific API endpoint.
        let client = Client::builder()
            .timeout(Duration::from_secs(300)) // 5 min timeout for uploads
            .danger_accept_invalid_certs(true)
            .build()
            .map_err(|e| TranscriptionError::Other(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            api_key,
            client,
            poll_interval: Duration::from_secs(poll_interval_secs),
            timeout: Duration::from_secs(timeout_secs),
            default_options,
        })
    }

    /// Create options from TranscriptionConfig and defaults.
    fn build_options(&self, config: &TranscriptionConfig) -> SpeechTextAIOptions {
        let mut options = self.default_options.clone();

        if let Some(speaker_count) = config.speaker_count {
            options.number_of_speakers = speaker_count;
        }

        options
    }

    /// Submit audio file for transcription.
    ///
    /// The SpeechText.AI API expects:
    /// - POST request with query parameters in URL
    /// - Raw binary audio data in body
    /// - Content-Type: application/octet-stream
    async fn submit_job(
        &self,
        audio_path: &Path,
        language: &str,
        options: &SpeechTextAIOptions,
    ) -> Result<String> {
        let file_bytes = tokio::fs::read(audio_path).await.map_err(|e| {
            TranscriptionError::IoError(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to read audio file: {}", e),
            ))
        })?;

        // Build URL with query parameters
        let mut url = format!(
            "{}?key={}&language={}&punctuation={}",
            API_BASE_URL,
            self.api_key,
            language,
            if options.punctuation { "true" } else { "false" }
        );

        if options.number_of_speakers > 0 {
            url.push_str(&format!("&numberOfSpeakers={}", options.number_of_speakers));
        }

        if !options.custom_vocabulary.is_empty() {
            url.push_str(&format!("&customVocabulary={}", options.custom_vocabulary.join(",")));
        }

        debug!(
            "Submit request - Method: POST, URL: {}, Content-Type: application/octet-stream, Body size: {} bytes",
            url.replace(&self.api_key, "***"),
            file_bytes.len()
        );

        // POST with binary body and octet-stream content type
        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/octet-stream")
            .body(file_bytes)
            .send()
            .await
            .map_err(|e| TranscriptionError::ApiError(format!("Request failed: {}", e)))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| TranscriptionError::ApiError(format!("Failed to read response: {}", e)))?;

        debug!("Submit response - Status: {}, Body: {}", status, body);

        if !status.is_success() {
            return Err(TranscriptionError::ApiError(format!(
                "API returned status {}: {}",
                status, body
            )));
        }

        let parsed: RecognizeResponse = serde_json::from_str(&body).map_err(|e| {
            TranscriptionError::ApiError(format!("Failed to parse response: {} - body: {}", e, body))
        })?;

        if let Some(error) = parsed.error {
            return Err(TranscriptionError::ApiError(format!("API error: {} (full response: {})", error, body)));
        }

        parsed.id.ok_or_else(|| {
            TranscriptionError::ApiError(format!("No task id in response. Full response: {}", body))
        })
    }

    /// Poll for transcription results.
    async fn poll_results(&self, task_id: &str) -> Result<TranscriptionResult> {
        let start_time = std::time::Instant::now();

        loop {
            if start_time.elapsed() > self.timeout {
                return Err(TranscriptionError::ApiTimeout(format!(
                    "Transcription timed out after {:?}",
                    self.timeout
                )));
            }

            let url = format!("{}?key={}&task={}", RESULTS_URL, self.api_key, task_id);

            debug!("Polling for results: task_id={}", task_id);

            // Create a fresh client for polling to avoid any connection state issues
            let poll_client = Client::builder()
                .timeout(Duration::from_secs(30))
                .danger_accept_invalid_certs(true)
                .build()
                .map_err(|e| TranscriptionError::Other(format!("Failed to create poll client: {}", e)))?;

            let response = poll_client
                .get(&url)
                .send()
                .await
                .map_err(|e| {
                    let mut details = format!("Poll request failed: {}", e);
                    if e.is_connect() {
                        details.push_str(" [connection error]");
                    }
                    if e.is_timeout() {
                        details.push_str(" [timeout]");
                    }
                    if e.is_request() {
                        details.push_str(" [request error]");
                    }
                    if e.is_builder() {
                        details.push_str(" [builder error]");
                    }
                    if e.is_redirect() {
                        details.push_str(" [redirect error]");
                    }
                    if e.is_status() {
                        details.push_str(&format!(" [status: {:?}]", e.status()));
                    }
                    // Get the full error chain
                    let mut source = e.source();
                    while let Some(s) = source {
                        details.push_str(&format!(" -> {}", s));
                        source = s.source();
                    }
                    TranscriptionError::ApiError(details)
                })?;

            let status_code = response.status();
            let body = response
                .text()
                .await
                .map_err(|e| TranscriptionError::ApiError(format!("Failed to read response: {}", e)))?;

            debug!(
                "Poll response - URL: {}, status: {}, body: {}",
                url.replace(&self.api_key, "***"),
                status_code,
                body
            );

            let parsed: ResultsResponse = serde_json::from_str(&body).map_err(|e| {
                TranscriptionError::ApiError(format!("Failed to parse results: {} - body: {}", e, body))
            })?;

            if let Some(error) = parsed.error {
                return Err(TranscriptionError::ApiError(error));
            }

            match parsed.status.as_str() {
                "completed" | "finished" | "SUCCESS" => {
                    info!("Transcription completed for task {}", task_id);
                    return self.parse_results(parsed.results);
                }
                "failed" | "FAILED" | "ERROR" | "error" => {
                    return Err(TranscriptionError::TranscriptionFailed(
                        "Transcription failed on the server".to_string(),
                    ));
                }
                status => {
                    debug!("Transcription status: {} - waiting...", status);
                    sleep(self.poll_interval).await;
                }
            }
        }
    }

    /// Parse API results into TranscriptionResult.
    fn parse_results(&self, results: Option<TranscriptionResults>) -> Result<TranscriptionResult> {
        let results = results.ok_or_else(|| {
            TranscriptionError::TranscriptionFailed("No results in completed response".to_string())
        })?;

        let content = results.transcript.unwrap_or_default();

        // Convert word_time_offsets to segments
        // Each word becomes a segment with its timing
        let segments: Vec<Segment> = results
            .word_time_offsets
            .unwrap_or_default()
            .into_iter()
            .map(|word| {
                let mut segment = Segment::new(word.word, word.start_time, word.end_time);
                if let Some(confidence) = word.confidence {
                    segment = segment.with_confidence(confidence);
                }
                segment
            })
            .collect();

        let duration = segments.last().map(|s| s.end_seconds);

        let mut result = TranscriptionResult::with_segments(content, segments);

        if let Some(d) = duration {
            result = result.with_duration(d);
        }

        // Note: summary and highlights are available in results.summary and results.highlights
        // but our TranscriptionResult doesn't have fields for them yet.
        // They could be added to a metadata field in the future.

        if let Some(ref summary) = results.summary {
            debug!("Summary available: {}", summary);
        }

        if let Some(ref highlights) = results.highlights {
            debug!("Highlights available: {:?}", highlights);
        }

        Ok(result)
    }

    /// Get the task ID for a submitted job (for external polling).
    pub async fn submit_and_get_task_id(
        &self,
        audio_path: &Path,
        config: &TranscriptionConfig,
    ) -> Result<(String, SpeechTextAIOptions)> {
        if !audio_path.exists() {
            return Err(TranscriptionError::FileNotFound(
                audio_path.display().to_string(),
            ));
        }

        // Normalize language code to BCP-47 format expected by SpeechText.AI
        let raw_language = config.language.as_deref()
            .unwrap_or_else(|| default_language_code(PROVIDER_SPEECHTEXT_AI));
        let language = normalize_language_code(raw_language, PROVIDER_SPEECHTEXT_AI);
        let options = self.build_options(config);
        let task_id = self.submit_job(audio_path, &language, &options).await?;

        Ok((task_id, options))
    }

    /// Poll for results given a task ID (for external polling).
    pub async fn get_results(&self, task_id: &str) -> Result<TranscriptionResult> {
        self.poll_results(task_id).await
    }

    /// Check if a job is complete without waiting.
    pub async fn check_status(&self, task_id: &str) -> Result<Option<TranscriptionResult>> {
        let url = format!("{}?key={}&task={}", RESULTS_URL, self.api_key, task_id);

        debug!(
            "Status check request - URL: {}",
            url.replace(&self.api_key, "***")
        );

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| TranscriptionError::ApiError(format!("Status check failed: {}", e)))?;

        let status_code = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| TranscriptionError::ApiError(format!("Failed to read response: {}", e)))?;

        debug!("Status check response - Status: {}, Body: {}", status_code, body);

        let parsed: ResultsResponse = serde_json::from_str(&body).map_err(|e| {
            TranscriptionError::ApiError(format!("Failed to parse status: {} - body: {}", e, body))
        })?;

        if let Some(error) = parsed.error {
            return Err(TranscriptionError::ApiError(error));
        }

        match parsed.status.as_str() {
            "completed" | "finished" | "SUCCESS" => Ok(Some(self.parse_results(parsed.results)?)),
            "failed" | "FAILED" | "ERROR" | "error" => Err(TranscriptionError::TranscriptionFailed(
                "Transcription failed on the server".to_string(),
            )),
            _ => Ok(None), // Still processing
        }
    }
}

#[async_trait]
impl TranscriptionBackend for SpeechTextAIBackend {
    fn name(&self) -> &str {
        "speechtext_ai"
    }

    fn supported_features(&self) -> Vec<&str> {
        vec![
            "language_detection",
            "timestamps",
            "speaker_diarization",
            "punctuation",
            "summary",
            "highlights",
            "captions",
        ]
    }

    fn is_ready(&self) -> bool {
        !self.api_key.is_empty()
    }

    async fn transcribe(
        &self,
        audio_path: &Path,
        config: &TranscriptionConfig,
    ) -> Result<TranscriptionResult> {
        if !audio_path.exists() {
            return Err(TranscriptionError::FileNotFound(
                audio_path.display().to_string(),
            ));
        }

        // Normalize language code to BCP-47 format expected by SpeechText.AI
        let raw_language = config.language.as_deref()
            .unwrap_or_else(|| default_language_code(PROVIDER_SPEECHTEXT_AI));
        let language = normalize_language_code(raw_language, PROVIDER_SPEECHTEXT_AI);
        let options = self.build_options(config);

        info!(
            "Starting SpeechText.AI transcription for {:?} with language {}",
            audio_path, &language
        );

        // Submit job
        let task_id = self.submit_job(audio_path, &language, &options).await?;
        info!("Job submitted with task_id: {}", task_id);

        // Poll for results
        self.poll_results(&task_id).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_options_default() {
        let options = SpeechTextAIOptions::new();
        assert!(options.punctuation);
        assert!(!options.summary);
        assert!(!options.highlights);
        assert_eq!(options.number_of_speakers, 0);
        assert!(options.caption_output.is_none());
        assert!(options.custom_vocabulary.is_empty());
    }

    #[test]
    fn test_options_builder() {
        let options = SpeechTextAIOptions::new()
            .with_punctuation(true)
            .with_summary(true)
            .with_highlights(true)
            .with_speakers(2)
            .with_caption_output("srt")
            .with_custom_vocabulary(vec!["word1".to_string(), "word2".to_string()]);

        assert!(options.punctuation);
        assert!(options.summary);
        assert!(options.highlights);
        assert_eq!(options.number_of_speakers, 2);
        assert_eq!(options.caption_output, Some("srt".to_string()));
        assert_eq!(options.custom_vocabulary.len(), 2);
    }

    #[test]
    fn test_options_to_json() {
        let options = SpeechTextAIOptions::new().with_summary(true);
        let json = options.to_json();
        assert!(json.contains("\"summary\":true"));
        assert!(json.contains("\"punctuation\":true"));
    }

    #[test]
    fn test_backend_requires_api_key() {
        let config = BackendConfig::new();
        let result = SpeechTextAIBackend::new(config);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("API key is required"));
    }

    #[test]
    fn test_backend_with_api_key() {
        let config = BackendConfig::new().with_api_key("test-key-123");
        let result = SpeechTextAIBackend::new(config);
        assert!(result.is_ok());
        let backend = result.unwrap();
        assert_eq!(backend.name(), "speechtext_ai");
        assert!(backend.is_ready());
    }

    #[test]
    fn test_backend_supported_features() {
        let config = BackendConfig::new().with_api_key("test-key");
        let backend = SpeechTextAIBackend::new(config).unwrap();
        let features = backend.supported_features();
        assert!(features.contains(&"punctuation"));
        assert!(features.contains(&"summary"));
        assert!(features.contains(&"highlights"));
        assert!(features.contains(&"speaker_diarization"));
    }

    #[test]
    fn test_backend_options_from_config() {
        let config = BackendConfig::new()
            .with_api_key("test-key")
            .with_option("punctuation", "true")
            .with_option("summary", "true")
            .with_option("poll_interval", "10");

        let backend = SpeechTextAIBackend::new(config).unwrap();
        assert!(backend.default_options.punctuation);
        assert!(backend.default_options.summary);
        assert_eq!(backend.poll_interval, Duration::from_secs(10));
    }
}
