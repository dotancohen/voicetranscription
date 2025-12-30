# VoiceTranscription

A modular, cross-platform audio transcription library with pluggable backends.

## Features

- Async API for non-blocking transcription
- Pluggable backend architecture
- Local Whisper support via whisper-rs
- Normalized output format with segments and timestamps
- Python bindings via PyO3
- Android/Kotlin bindings via UniFFI

## Prerequisites

### System Requirements

- Rust 1.70+ (install from https://rustup.rs)
- cmake 3.14+
- A C/C++ compiler (gcc/clang)
- pkg-config

### Installing Prerequisites

#### Debian/Ubuntu

```bash
sudo apt-get update
sudo apt-get install -y cmake build-essential pkg-config clang libclang-dev
```

#### Red Hat/Fedora/CentOS

```bash
# Fedora
sudo dnf install -y cmake gcc gcc-c++ make pkg-config clang clang-devel

# RHEL/CentOS 8+
sudo dnf install -y cmake gcc gcc-c++ make pkgconfig clang clang-devel

# RHEL/CentOS 7
sudo yum install -y cmake3 gcc gcc-c++ make pkgconfig clang clang-devel
# Note: On CentOS 7, use 'cmake3' instead of 'cmake'
```

#### macOS

```bash
# Using Homebrew
brew install cmake pkg-config

# Xcode Command Line Tools (includes clang)
xcode-select --install
```

#### Windows

```powershell
# Using Chocolatey
choco install cmake visualstudio2022buildtools

# Or using winget
winget install Kitware.CMake
winget install Microsoft.VisualStudio.2022.BuildTools
```

### Installing Rust

If you don't have Rust installed:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

## Download

### Clone the Repository

```bash
git clone https://github.com/dotancohen/VoiceTranscription.git
cd VoiceTranscription
```

### Or Download a Release

Download the latest release from the [releases page](https://github.com/dotancohen/VoiceTranscription/releases).

## Building

### Build the Core Library

```bash
# Debug build
cargo build

# Release build (optimized)
cargo build --release
```

### Build Python Bindings

```bash
cd bindings/python
pip install maturin
maturin develop        # For development
maturin build --release  # For distribution
```

### Build Android Bindings

```bash
# Install Android targets
rustup target add aarch64-linux-android armv7-linux-androideabi

# Build for Android (requires NDK)
cd bindings/android
cargo build --release --target aarch64-linux-android
```

## Testing

### Run All Tests

```bash
cargo test
```

### Run Tests with Output

```bash
cargo test -- --nocapture
```

### Run Specific Test Module

```bash
# Test types module
cargo test types::tests

# Test client module
cargo test client::tests

# Test error module
cargo test error::tests

# Test backend config
cargo test backend::tests
```

### Run Tests with Coverage (requires cargo-tarpaulin)

```bash
cargo install cargo-tarpaulin
cargo tarpaulin --out Html
```

## Installation

### As a Rust Dependency

Add to your `Cargo.toml`:

```toml
[dependencies]
voice-transcription = { git = "https://github.com/dotancohen/VoiceTranscription.git" }
```

Or for a local path:

```toml
[dependencies]
voice-transcription = { path = "path/to/VoiceTranscription" }
```

### Python Package

```bash
cd bindings/python
pip install .
```

## Usage

### Rust

```rust
use voice_transcription::{
    TranscriptionClient, TranscriptionConfig,
    backends::LocalWhisperBackend, BackendConfig,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure the backend
    let backend_config = BackendConfig::new()
        .with_model_path("/path/to/ggml-base.bin");

    // Create the backend and client
    let backend = LocalWhisperBackend::new(backend_config)?;
    let client = TranscriptionClient::new(backend);

    // Transcribe with automatic language detection
    let result = client.transcribe("/path/to/audio.wav").await?;
    println!("Content: {}", result.content);
    println!("Languages: {:?}", result.languages);

    // Transcribe with specific language
    let result = client.transcribe_with_language("/path/to/audio.wav", "en").await?;

    // Transcribe with custom configuration
    let config = TranscriptionConfig::new()
        .with_language("he")
        .with_speaker_count(2);
    let result = client.transcribe_with_config("/path/to/audio.wav", &config).await?;

    for segment in &result.segments {
        println!("[{:.2}s - {:.2}s] {}",
            segment.start_seconds,
            segment.end_seconds,
            segment.text
        );
    }

    Ok(())
}
```

### Python

```python
from voice_transcription import TranscriptionClient, TranscriptionConfig

# Create client with local Whisper backend
client = TranscriptionClient.with_local_whisper("/path/to/ggml-base.bin")

# Simple transcription
result = client.transcribe("/path/to/audio.wav")
print(f"Content: {result.content}")
print(f"Languages: {result.languages}")

# With specific language
result = client.transcribe_with_language("/path/to/audio.wav", "en")

# With custom configuration
config = TranscriptionConfig(language="he", speaker_count=2)
result = client.transcribe("/path/to/audio.wav", config)

for segment in result.segments:
    print(f"[{segment.start_seconds:.2f}s - {segment.end_seconds:.2f}s] {segment.text}")
```

### Kotlin (Android)

```kotlin
import voice_transcription.*

// Create client
val client = createLocalWhisperClient("/path/to/ggml-base.bin")

// Simple transcription
val result = client.transcribe("/path/to/audio.wav")
println("Content: ${result.content}")
println("Languages: ${result.languages}")

// With specific language
val result = client.transcribeWithLanguage("/path/to/audio.wav", "en")

// With custom configuration
val config = TranscriptionConfig(
    language = "he",
    speakerCount = 2u,
    wordTimestamps = false,
    model = null
)
val result = client.transcribeWithConfig("/path/to/audio.wav", config)

result.segments.forEach { segment ->
    println("[${segment.startSeconds}s - ${segment.endSeconds}s] ${segment.text}")
}
```

## Example Applications

The `examples/` directory contains ready-to-use example applications with TOML configuration.

### Setup

1. Build the Python bindings:
   ```bash
   cd bindings/python
   maturin develop
   ```

2. Download a Whisper model and update `examples/config.toml`:
   ```toml
   [whisper]
   model_path = "/path/to/ggml-base.bin"
   ```

### Python Example

```bash
python examples/transcribe.py recording.wav
python examples/transcribe.py --config examples/config.toml recording.wav
```

### PHP Example

```bash
php examples/transcribe.php recording.wav
php examples/transcribe.php --config examples/config.toml recording.wav
```

**Note:** The PHP example invokes a Python helper script (`transcribe_helper.py`) as a subprocess, since native PHP bindings are not yet available. This approach works well for development and low-volume use cases.

**For production PHP deployments**, consider one of these alternatives:

1. **REST API**: Wrap VoiceTranscription in a lightweight HTTP server (see below)
2. **PHP FFI**: Create C-compatible bindings that PHP can call directly via FFI
3. **Message Queue**: Use a job queue where Python/Rust workers process transcription requests

### REST API Integration

VoiceTranscription can be wrapped in a REST API server, allowing any language to use it via HTTP. A Rust implementation using `axum`, `actix-web`, or `warp` would provide:

- Language-agnostic access (PHP, JavaScript, Go, Java, etc.)
- Horizontal scaling with multiple worker instances
- Easy deployment as a microservice
- Built-in request queuing and rate limiting

Example endpoint design:
```
POST /transcribe
  Body: multipart/form-data with audio file
  Query: ?language=en&speakers=2
  Response: JSON TranscriptionResult
```

This is the recommended approach for production systems that need to call VoiceTranscription from multiple languages or services.

## Downloading Whisper Models

The LocalWhisper backend requires a GGML-format Whisper model. Download from:

```bash
# Base model (~150MB, good balance of speed/accuracy)
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin

# Small model (~500MB, better accuracy)
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin

# Medium model (~1.5GB, even better accuracy)
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin

# Large model (~3GB, best accuracy)
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin
```

## API Reference

### TranscriptionResult

| Field | Type | Description |
|-------|------|-------------|
| `content` | `String` | Full transcribed text |
| `segments` | `Vec<Segment>` | Individual segments with timing |
| `languages` | `Option<Vec<String>>` | Detected languages in confidence order |
| `duration_seconds` | `Option<f64>` | Total audio duration |
| `confidence` | `Option<f64>` | Overall confidence (0.0-1.0) |
| `speaker_count` | `Option<u32>` | Number of detected speakers |

### Segment

| Field | Type | Description |
|-------|------|-------------|
| `text` | `String` | Segment text |
| `start_seconds` | `f64` | Start time in seconds |
| `end_seconds` | `f64` | End time in seconds |
| `speaker` | `Option<String>` | Speaker identifier |
| `confidence` | `Option<f64>` | Segment confidence |

### TranscriptionConfig

| Field | Type | Description |
|-------|------|-------------|
| `language` | `Option<String>` | Language hint (ISO 639-1) |
| `speaker_count` | `Option<u32>` | Expected speakers for diarization |
| `word_timestamps` | `bool` | Enable word-level timestamps |
| `model` | `Option<String>` | Model name/path (backend-specific) |

## Supported Backends

- **LocalWhisper**: Local transcription using OpenAI's Whisper model via whisper-rs

## Troubleshooting

### Build Errors

**"cmake not found"**: Install cmake using your package manager (see Prerequisites).

**"clang not found"**: Install clang/libclang-dev (Debian/Ubuntu) or clang-devel (Red Hat).

**"stdbool.h not found"**: Your C compiler headers are missing. Install build-essential (Debian/Ubuntu) or gcc (Red Hat).

### Runtime Errors

**"Model file not found"**: Ensure the path to your GGML model file is correct and the file exists.

**"Unsupported audio format"**: Currently only WAV files are supported. Convert your audio using ffmpeg:
```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

## License

MIT
