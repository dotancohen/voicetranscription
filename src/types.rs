//! Core types for transcription results.

use serde::{Deserialize, Serialize};

/// A segment of transcribed audio with timing information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    /// The transcribed text for this segment.
    pub text: String,
    /// Start time in seconds from the beginning of the audio.
    pub start_seconds: f64,
    /// End time in seconds from the beginning of the audio.
    pub end_seconds: f64,
    /// Speaker identifier if diarization is enabled.
    pub speaker: Option<String>,
    /// Confidence score for this segment (0.0 to 1.0).
    pub confidence: Option<f64>,
}

impl Segment {
    /// Create a new segment with required fields.
    pub fn new(text: String, start_seconds: f64, end_seconds: f64) -> Self {
        Self {
            text,
            start_seconds,
            end_seconds,
            speaker: None,
            confidence: None,
        }
    }

    /// Set the speaker for this segment.
    pub fn with_speaker(mut self, speaker: String) -> Self {
        self.speaker = Some(speaker);
        self
    }

    /// Set the confidence for this segment.
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = Some(confidence);
        self
    }
}

/// The result of a transcription operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionResult {
    /// The full transcribed text content.
    pub content: String,
    /// Individual segments with timing information.
    pub segments: Vec<Segment>,
    /// Detected languages in order of confidence.
    pub languages: Option<Vec<String>>,
    /// Total duration of the audio in seconds.
    pub duration_seconds: Option<f64>,
    /// Overall confidence score (0.0 to 1.0).
    pub confidence: Option<f64>,
    /// Number of distinct speakers detected.
    pub speaker_count: Option<u32>,
}

impl TranscriptionResult {
    /// Create a new transcription result with just the content.
    pub fn new(content: String) -> Self {
        Self {
            content,
            segments: Vec::new(),
            languages: None,
            duration_seconds: None,
            confidence: None,
            speaker_count: None,
        }
    }

    /// Create a transcription result with content and segments.
    pub fn with_segments(content: String, segments: Vec<Segment>) -> Self {
        Self {
            content,
            segments,
            languages: None,
            duration_seconds: None,
            confidence: None,
            speaker_count: None,
        }
    }

    /// Set the detected languages.
    pub fn with_languages(mut self, languages: Vec<String>) -> Self {
        self.languages = Some(languages);
        self
    }

    /// Set the audio duration.
    pub fn with_duration(mut self, duration_seconds: f64) -> Self {
        self.duration_seconds = Some(duration_seconds);
        self
    }

    /// Set the overall confidence.
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = Some(confidence);
        self
    }

    /// Set the speaker count.
    pub fn with_speaker_count(mut self, speaker_count: u32) -> Self {
        self.speaker_count = Some(speaker_count);
        self
    }
}

/// Configuration for a transcription request.
#[derive(Debug, Clone, Default)]
pub struct TranscriptionConfig {
    /// Language hint (ISO 639-1 code, e.g., "en", "he").
    /// If None, autodetection is used.
    pub language: Option<String>,
    /// Expected number of speakers for diarization.
    /// If None or 1, diarization is disabled.
    pub speaker_count: Option<u32>,
    /// Whether to include word-level timestamps.
    pub word_timestamps: bool,
    /// Model name or path (backend-specific).
    pub model: Option<String>,
    /// Skip audio conversion cache (always convert fresh).
    pub nocache: bool,
}

impl TranscriptionConfig {
    /// Create a new default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the language hint.
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Set the expected speaker count.
    pub fn with_speaker_count(mut self, count: u32) -> Self {
        self.speaker_count = Some(count);
        self
    }

    /// Enable word-level timestamps.
    pub fn with_word_timestamps(mut self) -> Self {
        self.word_timestamps = true;
        self
    }

    /// Set the model name or path.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Skip audio conversion cache (always convert fresh).
    pub fn with_nocache(mut self) -> Self {
        self.nocache = true;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segment_new() {
        let segment = Segment::new("Hello world".to_string(), 0.0, 1.5);
        assert_eq!(segment.text, "Hello world");
        assert_eq!(segment.start_seconds, 0.0);
        assert_eq!(segment.end_seconds, 1.5);
        assert!(segment.speaker.is_none());
        assert!(segment.confidence.is_none());
    }

    #[test]
    fn test_segment_with_speaker() {
        let segment = Segment::new("Hello".to_string(), 0.0, 1.0)
            .with_speaker("Speaker_A".to_string());
        assert_eq!(segment.speaker, Some("Speaker_A".to_string()));
    }

    #[test]
    fn test_segment_with_confidence() {
        let segment = Segment::new("Hello".to_string(), 0.0, 1.0)
            .with_confidence(0.95);
        assert_eq!(segment.confidence, Some(0.95));
    }

    #[test]
    fn test_segment_builder_chain() {
        let segment = Segment::new("Test".to_string(), 1.0, 2.5)
            .with_speaker("Speaker_B".to_string())
            .with_confidence(0.87);
        assert_eq!(segment.text, "Test");
        assert_eq!(segment.start_seconds, 1.0);
        assert_eq!(segment.end_seconds, 2.5);
        assert_eq!(segment.speaker, Some("Speaker_B".to_string()));
        assert_eq!(segment.confidence, Some(0.87));
    }

    #[test]
    fn test_transcription_result_new() {
        let result = TranscriptionResult::new("Hello world".to_string());
        assert_eq!(result.content, "Hello world");
        assert!(result.segments.is_empty());
        assert!(result.languages.is_none());
        assert!(result.duration_seconds.is_none());
        assert!(result.confidence.is_none());
        assert!(result.speaker_count.is_none());
    }

    #[test]
    fn test_transcription_result_with_segments() {
        let segments = vec![
            Segment::new("Hello".to_string(), 0.0, 0.5),
            Segment::new("world".to_string(), 0.5, 1.0),
        ];
        let result = TranscriptionResult::with_segments("Hello world".to_string(), segments);
        assert_eq!(result.content, "Hello world");
        assert_eq!(result.segments.len(), 2);
        assert_eq!(result.segments[0].text, "Hello");
        assert_eq!(result.segments[1].text, "world");
    }

    #[test]
    fn test_transcription_result_with_languages() {
        let result = TranscriptionResult::new("Test".to_string())
            .with_languages(vec!["en".to_string(), "de".to_string()]);
        assert_eq!(result.languages, Some(vec!["en".to_string(), "de".to_string()]));
    }

    #[test]
    fn test_transcription_result_with_duration() {
        let result = TranscriptionResult::new("Test".to_string())
            .with_duration(120.5);
        assert_eq!(result.duration_seconds, Some(120.5));
    }

    #[test]
    fn test_transcription_result_with_confidence() {
        let result = TranscriptionResult::new("Test".to_string())
            .with_confidence(0.92);
        assert_eq!(result.confidence, Some(0.92));
    }

    #[test]
    fn test_transcription_result_with_speaker_count() {
        let result = TranscriptionResult::new("Test".to_string())
            .with_speaker_count(3);
        assert_eq!(result.speaker_count, Some(3));
    }

    #[test]
    fn test_transcription_result_builder_chain() {
        let result = TranscriptionResult::new("Full test".to_string())
            .with_languages(vec!["he".to_string()])
            .with_duration(60.0)
            .with_confidence(0.88)
            .with_speaker_count(2);
        assert_eq!(result.content, "Full test");
        assert_eq!(result.languages, Some(vec!["he".to_string()]));
        assert_eq!(result.duration_seconds, Some(60.0));
        assert_eq!(result.confidence, Some(0.88));
        assert_eq!(result.speaker_count, Some(2));
    }

    #[test]
    fn test_transcription_config_default() {
        let config = TranscriptionConfig::default();
        assert!(config.language.is_none());
        assert!(config.speaker_count.is_none());
        assert!(!config.word_timestamps);
        assert!(config.model.is_none());
        assert!(!config.nocache);
    }

    #[test]
    fn test_transcription_config_new() {
        let config = TranscriptionConfig::new();
        assert!(config.language.is_none());
        assert!(config.speaker_count.is_none());
        assert!(!config.word_timestamps);
        assert!(config.model.is_none());
        assert!(!config.nocache);
    }

    #[test]
    fn test_transcription_config_with_language() {
        let config = TranscriptionConfig::new().with_language("en");
        assert_eq!(config.language, Some("en".to_string()));
    }

    #[test]
    fn test_transcription_config_with_speaker_count() {
        let config = TranscriptionConfig::new().with_speaker_count(4);
        assert_eq!(config.speaker_count, Some(4));
    }

    #[test]
    fn test_transcription_config_with_word_timestamps() {
        let config = TranscriptionConfig::new().with_word_timestamps();
        assert!(config.word_timestamps);
    }

    #[test]
    fn test_transcription_config_with_model() {
        let config = TranscriptionConfig::new().with_model("large-v3");
        assert_eq!(config.model, Some("large-v3".to_string()));
    }

    #[test]
    fn test_transcription_config_with_nocache() {
        let config = TranscriptionConfig::new().with_nocache();
        assert!(config.nocache);
    }

    #[test]
    fn test_transcription_config_builder_chain() {
        let config = TranscriptionConfig::new()
            .with_language("he")
            .with_speaker_count(2)
            .with_word_timestamps()
            .with_model("base")
            .with_nocache();
        assert_eq!(config.language, Some("he".to_string()));
        assert_eq!(config.speaker_count, Some(2));
        assert!(config.word_timestamps);
        assert_eq!(config.model, Some("base".to_string()));
        assert!(config.nocache);
    }

    #[test]
    fn test_segment_serialization() {
        let segment = Segment::new("Test".to_string(), 1.0, 2.0)
            .with_speaker("A".to_string())
            .with_confidence(0.9);
        let json = serde_json::to_string(&segment).unwrap();
        let deserialized: Segment = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.text, segment.text);
        assert_eq!(deserialized.start_seconds, segment.start_seconds);
        assert_eq!(deserialized.end_seconds, segment.end_seconds);
        assert_eq!(deserialized.speaker, segment.speaker);
        assert_eq!(deserialized.confidence, segment.confidence);
    }

    #[test]
    fn test_transcription_result_serialization() {
        let result = TranscriptionResult::new("Hello".to_string())
            .with_languages(vec!["en".to_string()])
            .with_duration(10.0);
        let json = serde_json::to_string(&result).unwrap();
        let deserialized: TranscriptionResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.content, result.content);
        assert_eq!(deserialized.languages, result.languages);
        assert_eq!(deserialized.duration_seconds, result.duration_seconds);
    }
}
