//! Transcription backend implementations.

mod local_whisper;
mod speechtext_ai;

pub use local_whisper::LocalWhisperBackend;
pub use speechtext_ai::{SpeechTextAIBackend, SpeechTextAIOptions};
