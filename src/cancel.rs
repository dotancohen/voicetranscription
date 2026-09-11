//! Stopping a transcription that is already running.
//!
//! Whisper runs inside one long call into native code, so the only way to
//! stop it is to tell it to stop from the inside: whisper.cpp asks an "abort
//! callback" between windows of audio, and returns early when the answer is
//! yes. That is what this module holds.
//!
//! The flag is one per process, not one per client, because the caller that
//! needs it (the phone) deliberately runs a single transcription at a time:
//! the model wants about a gigabyte of memory and every core, so a second
//! job would only slow the first one down. A caller that wants to run
//! several at once must not use this; it would stop all of them together.
//!
//! Whoever asks for a stop leaves the flag raised. The next batch lowers it
//! with [`clear_cancel`] before starting, so a stop asked for at the very
//! moment a job ends does not silently kill the following job.

use std::sync::atomic::{AtomicBool, Ordering};

static CANCEL_REQUESTED: AtomicBool = AtomicBool::new(false);

/// Ask the running transcription to stop as soon as it can.
///
/// It stops between windows of audio, so it takes about as long as one
/// window (a few hundred milliseconds of work), not the rest of the file.
/// The partial text is thrown away: a half-transcription is worse than
/// none, because it looks complete.
pub fn request_cancel() {
    CANCEL_REQUESTED.store(true, Ordering::SeqCst);
}

/// Lower the flag. Call this before starting work that should run.
pub fn clear_cancel() {
    CANCEL_REQUESTED.store(false, Ordering::SeqCst);
}

/// Whether a stop has been asked for and not yet cleared.
pub fn cancel_requested() -> bool {
    CANCEL_REQUESTED.load(Ordering::SeqCst)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The flag answers the question it was asked, and lowering it works.
    #[test]
    fn raised_then_lowered() {
        clear_cancel();
        assert!(!cancel_requested());
        request_cancel();
        assert!(cancel_requested());
        clear_cancel();
        assert!(!cancel_requested());
    }
}
