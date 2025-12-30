//! Audio format conversion and caching.
//!
//! Provides automatic conversion of audio files to WAV format (16kHz, mono, 16-bit PCM)
//! required by whisper.cpp, with caching to avoid repeated conversions.

use sha2::{Digest, Sha256};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::SystemTime;
use tracing::{debug, info, warn};

use crate::error::{Result, TranscriptionError};

/// Result of preparing an audio file for transcription.
#[derive(Debug)]
pub struct PreparedAudio {
    /// Path to the WAV file (either original or converted).
    pub path: PathBuf,
    /// Whether the file was converted (true) or used as-is (false).
    pub was_converted: bool,
}

/// Audio format converter with caching support.
pub struct AudioConverter {
    cache_dir: PathBuf,
}

impl Default for AudioConverter {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioConverter {
    /// Create a new AudioConverter with the default cache directory.
    ///
    /// Cache location: `~/.cache/voice-transcription/`
    pub fn new() -> Self {
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("/tmp"))
            .join("voice-transcription");
        Self { cache_dir }
    }

    /// Create an AudioConverter with a custom cache directory.
    pub fn with_cache_dir(cache_dir: PathBuf) -> Self {
        Self { cache_dir }
    }

    /// Get the cache directory path.
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Prepare an audio file for transcription.
    ///
    /// If the file is already a valid WAV, returns it directly.
    /// Otherwise, converts it to WAV format and caches the result.
    ///
    /// # Arguments
    /// * `audio_path` - Path to the input audio file.
    /// * `nocache` - If true, skip cache and always convert fresh.
    ///
    /// # Returns
    /// `PreparedAudio` with the path to use and whether conversion occurred.
    pub fn prepare(&self, audio_path: &Path, nocache: bool) -> Result<PreparedAudio> {
        // Verify input file exists
        if !audio_path.exists() {
            return Err(TranscriptionError::FileNotFound(
                audio_path.display().to_string(),
            ));
        }

        // Check if it's already a valid WAV file
        if self.is_valid_wav(audio_path)? {
            debug!("File is already a valid WAV: {:?}", audio_path);
            return Ok(PreparedAudio {
                path: audio_path.to_path_buf(),
                was_converted: false,
            });
        }

        // Need conversion - check cache first (unless nocache is set)
        let cache_path = self.get_cache_path(audio_path)?;

        if !nocache {
            if let Some(cached) = self.check_cache(audio_path, &cache_path)? {
                debug!("Using cached WAV: {:?}", cached);
                return Ok(PreparedAudio {
                    path: cached,
                    was_converted: false, // Not freshly converted
                });
            }
        }

        // Convert to WAV
        info!("Converting audio to WAV format: {:?}", audio_path);
        self.convert_to_wav(audio_path, &cache_path)?;

        Ok(PreparedAudio {
            path: cache_path,
            was_converted: true,
        })
    }

    /// Check if a file is a valid WAV in the format whisper expects.
    fn is_valid_wav(&self, path: &Path) -> Result<bool> {
        let mut file = match fs::File::open(path) {
            Ok(f) => f,
            Err(_) => return Ok(false),
        };

        let mut header = [0u8; 44];
        if file.read_exact(&mut header).is_err() {
            return Ok(false);
        }

        // Check RIFF header
        if &header[0..4] != b"RIFF" || &header[8..12] != b"WAVE" {
            return Ok(false);
        }

        // Check fmt chunk
        if &header[12..16] != b"fmt " {
            return Ok(false);
        }

        // Audio format (1 = PCM)
        let audio_format = u16::from_le_bytes([header[20], header[21]]);
        if audio_format != 1 {
            debug!("WAV is not PCM format: {}", audio_format);
            return Ok(false);
        }

        // Number of channels (should be 1 for mono)
        let channels = u16::from_le_bytes([header[22], header[23]]);

        // Sample rate (should be 16000)
        let sample_rate = u32::from_le_bytes([header[24], header[25], header[26], header[27]]);

        // Bits per sample (should be 16)
        let bits_per_sample = u16::from_le_bytes([header[34], header[35]]);

        let is_valid = channels == 1 && sample_rate == 16000 && bits_per_sample == 16;

        if !is_valid {
            debug!(
                "WAV format mismatch: channels={}, sample_rate={}, bits={}",
                channels, sample_rate, bits_per_sample
            );
        }

        Ok(is_valid)
    }

    /// Generate a cache path for an audio file.
    fn get_cache_path(&self, audio_path: &Path) -> Result<PathBuf> {
        // Get absolute path for consistent hashing
        let abs_path = audio_path.canonicalize().map_err(|e| {
            TranscriptionError::IoError(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Cannot resolve path {:?}: {}", audio_path, e),
            ))
        })?;

        // Get file modification time
        let metadata = fs::metadata(&abs_path)?;
        let mtime = metadata
            .modified()
            .unwrap_or(SystemTime::UNIX_EPOCH)
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Create hash from path + mtime
        let mut hasher = Sha256::new();
        hasher.update(abs_path.to_string_lossy().as_bytes());
        hasher.update(mtime.to_le_bytes());
        let hash = format!("{:x}", hasher.finalize());

        // Use first 16 chars of hash for filename
        let cache_filename = format!("{}.wav", &hash[..16]);

        Ok(self.cache_dir.join(cache_filename))
    }

    /// Check if a valid cached conversion exists.
    fn check_cache(&self, source_path: &Path, cache_path: &Path) -> Result<Option<PathBuf>> {
        if !cache_path.exists() {
            return Ok(None);
        }

        // Verify cache is newer than source
        let source_mtime = fs::metadata(source_path)?.modified()?;
        let cache_mtime = fs::metadata(cache_path)?.modified()?;

        if cache_mtime >= source_mtime {
            // Verify the cached file is valid
            if self.is_valid_wav(cache_path)? {
                return Ok(Some(cache_path.to_path_buf()));
            }
            // Invalid cache file, remove it
            warn!("Removing invalid cache file: {:?}", cache_path);
            let _ = fs::remove_file(cache_path);
        }

        Ok(None)
    }

    /// Convert an audio file to WAV format using ffmpeg.
    fn convert_to_wav(&self, input: &Path, output: &Path) -> Result<()> {
        // Ensure cache directory exists
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent)?;
        }

        // Run ffmpeg
        let result = Command::new("ffmpeg")
            .args([
                "-y",           // Overwrite output
                "-i",
                input.to_str().ok_or_else(|| {
                    TranscriptionError::ConversionFailed("Invalid input path".into())
                })?,
                "-ar", "16000", // Sample rate
                "-ac", "1",     // Mono
                "-c:a", "pcm_s16le", // 16-bit PCM
                "-f", "wav",    // WAV format
                output.to_str().ok_or_else(|| {
                    TranscriptionError::ConversionFailed("Invalid output path".into())
                })?,
            ])
            .output();

        match result {
            Ok(output_result) => {
                if output_result.status.success() {
                    info!("Audio converted successfully: {:?}", output);
                    Ok(())
                } else {
                    let stderr = String::from_utf8_lossy(&output_result.stderr);
                    Err(TranscriptionError::ConversionFailed(format!(
                        "ffmpeg failed: {}",
                        stderr.lines().last().unwrap_or("unknown error")
                    )))
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                Err(TranscriptionError::ConversionFailed(
                    "ffmpeg not found. Please install ffmpeg to convert audio files.".into(),
                ))
            }
            Err(e) => Err(TranscriptionError::ConversionFailed(format!(
                "Failed to run ffmpeg: {}",
                e
            ))),
        }
    }

    /// Get the total size of cached WAV files in bytes.
    pub fn cache_size(&self) -> Result<u64> {
        if !self.cache_dir.exists() {
            return Ok(0);
        }

        let mut total_size = 0u64;
        for entry in fs::read_dir(&self.cache_dir)? {
            if let Ok(entry) = entry {
                let path = entry.path();
                if path.is_file() && path.extension().map_or(false, |ext| ext == "wav") {
                    if let Ok(metadata) = entry.metadata() {
                        total_size += metadata.len();
                    }
                }
            }
        }

        Ok(total_size)
    }

    /// Get the number of cached WAV files.
    pub fn cache_file_count(&self) -> Result<usize> {
        if !self.cache_dir.exists() {
            return Ok(0);
        }

        let count = fs::read_dir(&self.cache_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| {
                let path = e.path();
                path.is_file() && path.extension().map_or(false, |ext| ext == "wav")
            })
            .count();

        Ok(count)
    }

    /// Clear the entire cache.
    ///
    /// Returns the number of files removed.
    pub fn clear_cache(&self) -> Result<usize> {
        if !self.cache_dir.exists() {
            return Ok(0);
        }

        let mut removed = 0;
        for entry in fs::read_dir(&self.cache_dir)? {
            if let Ok(entry) = entry {
                let path = entry.path();
                if path.is_file() && path.extension().map_or(false, |ext| ext == "wav") {
                    if fs::remove_file(&path).is_ok() {
                        removed += 1;
                        debug!("Removed cache file: {:?}", path);
                    }
                }
            }
        }

        info!("Cleared {} files from audio cache", removed);
        Ok(removed)
    }
}

/// Get information about the audio cache.
#[derive(Debug, Clone)]
pub struct CacheInfo {
    /// Path to the cache directory.
    pub path: PathBuf,
    /// Total size of cached files in bytes.
    pub size_bytes: u64,
    /// Number of cached files.
    pub file_count: usize,
}

impl CacheInfo {
    /// Get human-readable size string.
    pub fn size_human(&self) -> String {
        const KB: u64 = 1024;
        const MB: u64 = KB * 1024;
        const GB: u64 = MB * 1024;

        if self.size_bytes >= GB {
            format!("{:.2} GB", self.size_bytes as f64 / GB as f64)
        } else if self.size_bytes >= MB {
            format!("{:.2} MB", self.size_bytes as f64 / MB as f64)
        } else if self.size_bytes >= KB {
            format!("{:.2} KB", self.size_bytes as f64 / KB as f64)
        } else {
            format!("{} bytes", self.size_bytes)
        }
    }
}

/// Get information about the global audio cache.
pub fn get_cache_info() -> Result<CacheInfo> {
    let converter = AudioConverter::new();
    Ok(CacheInfo {
        path: converter.cache_dir().to_path_buf(),
        size_bytes: converter.cache_size()?,
        file_count: converter.cache_file_count()?,
    })
}

/// Clear the global audio cache.
///
/// Returns the number of files removed.
pub fn clear_cache() -> Result<usize> {
    let converter = AudioConverter::new();
    converter.clear_cache()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn create_test_wav(path: &Path) -> std::io::Result<()> {
        let mut file = fs::File::create(path)?;

        // Write a minimal valid WAV header (16kHz, mono, 16-bit)
        let data_size: u32 = 0;
        let file_size: u32 = 36 + data_size;

        file.write_all(b"RIFF")?;
        file.write_all(&file_size.to_le_bytes())?;
        file.write_all(b"WAVE")?;
        file.write_all(b"fmt ")?;
        file.write_all(&16u32.to_le_bytes())?; // fmt chunk size
        file.write_all(&1u16.to_le_bytes())?;  // audio format (PCM)
        file.write_all(&1u16.to_le_bytes())?;  // channels (mono)
        file.write_all(&16000u32.to_le_bytes())?; // sample rate
        file.write_all(&32000u32.to_le_bytes())?; // byte rate
        file.write_all(&2u16.to_le_bytes())?;  // block align
        file.write_all(&16u16.to_le_bytes())?; // bits per sample
        file.write_all(b"data")?;
        file.write_all(&data_size.to_le_bytes())?;

        Ok(())
    }

    #[test]
    fn test_is_valid_wav() {
        let temp_dir = TempDir::new().unwrap();
        let wav_path = temp_dir.path().join("test.wav");
        create_test_wav(&wav_path).unwrap();

        let converter = AudioConverter::with_cache_dir(temp_dir.path().join("cache"));
        assert!(converter.is_valid_wav(&wav_path).unwrap());
    }

    #[test]
    fn test_is_valid_wav_wrong_format() {
        let temp_dir = TempDir::new().unwrap();
        let txt_path = temp_dir.path().join("test.txt");
        fs::write(&txt_path, "not a wav file").unwrap();

        let converter = AudioConverter::with_cache_dir(temp_dir.path().join("cache"));
        assert!(!converter.is_valid_wav(&txt_path).unwrap());
    }

    #[test]
    fn test_cache_path_generation() {
        let temp_dir = TempDir::new().unwrap();
        let audio_path = temp_dir.path().join("test.mp3");
        fs::write(&audio_path, "fake audio").unwrap();

        let converter = AudioConverter::with_cache_dir(temp_dir.path().join("cache"));
        let cache_path = converter.get_cache_path(&audio_path).unwrap();

        assert!(cache_path.starts_with(temp_dir.path().join("cache")));
        assert!(cache_path.extension().map_or(false, |ext| ext == "wav"));
    }

    #[test]
    fn test_cache_size_empty() {
        let temp_dir = TempDir::new().unwrap();
        let converter = AudioConverter::with_cache_dir(temp_dir.path().join("nonexistent"));

        assert_eq!(converter.cache_size().unwrap(), 0);
        assert_eq!(converter.cache_file_count().unwrap(), 0);
    }

    #[test]
    fn test_clear_cache() {
        let temp_dir = TempDir::new().unwrap();
        let cache_dir = temp_dir.path().join("cache");
        fs::create_dir_all(&cache_dir).unwrap();

        // Create some fake cache files
        fs::write(cache_dir.join("abc123.wav"), "fake wav 1").unwrap();
        fs::write(cache_dir.join("def456.wav"), "fake wav 2").unwrap();
        fs::write(cache_dir.join("other.txt"), "not a wav").unwrap();

        let converter = AudioConverter::with_cache_dir(cache_dir);
        let removed = converter.clear_cache().unwrap();

        assert_eq!(removed, 2);
        assert_eq!(converter.cache_file_count().unwrap(), 0);
    }

    #[test]
    fn test_prepare_valid_wav() {
        let temp_dir = TempDir::new().unwrap();
        let wav_path = temp_dir.path().join("test.wav");
        create_test_wav(&wav_path).unwrap();

        let converter = AudioConverter::with_cache_dir(temp_dir.path().join("cache"));
        let prepared = converter.prepare(&wav_path, false).unwrap();

        assert_eq!(prepared.path, wav_path);
        assert!(!prepared.was_converted);
    }

    #[test]
    fn test_cache_info_human_size() {
        let info = CacheInfo {
            path: PathBuf::from("/tmp"),
            size_bytes: 1536,
            file_count: 1,
        };
        assert_eq!(info.size_human(), "1.50 KB");

        let info = CacheInfo {
            path: PathBuf::from("/tmp"),
            size_bytes: 1_500_000,
            file_count: 1,
        };
        assert_eq!(info.size_human(), "1.43 MB");
    }
}
