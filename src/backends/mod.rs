//! Transcription backend implementations.

#[cfg(feature = "local_whisper")]
mod local_whisper;

#[cfg(feature = "assemblyai")]
mod assemblyai;

#[cfg(feature = "google_cloud")]
mod google_cloud;

#[cfg(feature = "local_whisper")]
pub use local_whisper::LocalWhisperBackend;

#[cfg(feature = "assemblyai")]
pub use assemblyai::AssemblyAIBackend;

#[cfg(feature = "google_cloud")]
pub use google_cloud::GoogleCloudBackend;
