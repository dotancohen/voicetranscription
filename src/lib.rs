//! VoiceTranscription Core Library
//!
//! A modular, cross-platform transcription library with pluggable backends.
//!
//! # Features
//!
//! - Async API for non-blocking transcription
//! - Pluggable backend architecture
//! - Local Whisper support via whisper-rs
//! - Normalized output format with segments and timestamps
//!
//! # Example
//!
//! ```rust,no_run
//! use voice_transcription::{
//!     TranscriptionClient, TranscriptionConfig,
//!     backends::LocalWhisperBackend, BackendConfig,
//! };
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Configure the backend
//!     let backend_config = BackendConfig::new()
//!         .with_model_path("/path/to/ggml-base.bin");
//!
//!     // Create the backend
//!     let backend = LocalWhisperBackend::new(backend_config)?;
//!
//!     // Create the client
//!     let client = TranscriptionClient::new(backend);
//!
//!     // Transcribe with default settings
//!     let result = client.transcribe("/path/to/audio.wav").await?;
//!     println!("Transcription: {}", result.content);
//!
//!     // Or with custom configuration
//!     let config = TranscriptionConfig::new()
//!         .with_language("en")
//!         .with_speaker_count(2);
//!
//!     let result = client.transcribe_with_config("/path/to/audio.wav", &config).await?;
//!     println!("Languages: {:?}", result.languages);
//!     println!("Segments: {}", result.segments.len());
//!
//!     Ok(())
//! }
//! ```

pub mod audio;
pub mod backend;
pub mod backends;
pub mod client;
pub mod error;
pub mod schema;
pub mod language;
pub mod types;

use std::sync::Once;
use tracing_subscriber::EnvFilter;

static TRACING_INIT: Once = Once::new();

/// Initialize debug logging for transcription operations.
///
/// This enables detailed logging of HTTP requests and responses,
/// which is useful for debugging API issues.
///
/// Call this before any transcription operations to see debug output.
/// Multiple calls are safe - only the first call has an effect.
pub fn enable_debug_logging() {
    TRACING_INIT.call_once(|| {
        tracing_subscriber::fmt()
            .with_env_filter(
                EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| EnvFilter::new("voice_transcription=debug"))
            )
            .with_target(true)
            .with_thread_ids(false)
            .with_file(false)
            .with_line_number(false)
            .init();
    });
}

// Re-export main types at crate root for convenience
pub use backend::{BackendConfig, TranscriptionBackend};
pub use client::TranscriptionClient;
pub use error::{Result, TranscriptionError};
pub use schema::{HasProviderSchema, OptionType, OptionValue, ProviderOption, ProviderSchema};
pub use types::{Segment, TranscriptionConfig, TranscriptionResult};
