//! Transcription backend implementations.

#[cfg(feature = "local_whisper")]
mod local_whisper;

#[cfg(feature = "speechtext_ai")]
mod speechtext_ai;

#[cfg(feature = "assemblyai")]
mod assemblyai;

#[cfg(feature = "google_cloud")]
mod google_cloud;

#[cfg(feature = "local_whisper")]
pub use local_whisper::LocalWhisperBackend;

#[cfg(feature = "speechtext_ai")]
pub use speechtext_ai::{SpeechTextAIBackend, SpeechTextAIOptions};

#[cfg(feature = "assemblyai")]
pub use assemblyai::AssemblyAIBackend;

#[cfg(feature = "google_cloud")]
pub use google_cloud::GoogleCloudBackend;

