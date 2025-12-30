# VoiceTranscription Examples

Simple example applications demonstrating VoiceTranscription usage.

## Setup

1. Build the Python bindings:
```bash
cd ../bindings/python
pip install maturin
maturin develop
```

2. Download a Whisper model to the canonical location:
```bash
mkdir -p ~/.local/share/whisper
wget -P ~/.local/share/whisper/ https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin
```

## Python Example

### Basic Usage

```bash
# Show all options
python transcribe.py -h

# Basic transcription (auto-detects language)
python transcribe.py recording.wav

# Specify language (improves accuracy)
python transcribe.py --language he recording.wav
python transcribe.py --language en recording.wav

# Specify model
python transcribe.py --model small recording.wav
python transcribe.py --model /path/to/model.bin recording.wav

# Multiple speakers (diarization)
python transcribe.py --speaker_count 2 recording.wav

# Non-WAV files are automatically converted and cached
python transcribe.py recording.mp3

# Skip cache (always convert fresh)
python transcribe.py --nocache recording.mp3

# Cache management
python transcribe.py --cache-info
python transcribe.py --clear-cache

# List available models
python transcribe.py --list-models
```

### Options

Option names match the Rust `TranscriptionConfig` fields:

| Option | Description |
|--------|-------------|
| `--language CODE` | Language hint (ISO 639-1, e.g., `he`, `en`). Auto-detects if not specified. |
| `--speaker_count N` | Expected number of speakers for diarization |
| `--word_timestamps` | Enable word-level timestamps |
| `--model NAME` | Model file or name (e.g., `small`, `ggml-small.bin`, or full path) |
| `--nocache` | Skip audio conversion cache (always convert fresh) |
| `--config PATH` | Path to config.toml |

### Model Selection Priority

1. `--model` command-line argument (highest priority)
2. `model_path` from config.toml
3. Auto-select best available model from `~/.local/share/whisper/`

When auto-selecting, larger models are preferred for better multilingual support.

## PHP Example

The PHP example uses a Python subprocess since native PHP bindings are not yet available.

```bash
php transcribe.php recording.wav
php transcribe.php --config /path/to/config.toml recording.wav
```

## Configuration

Edit `config.toml` to configure defaults:

```toml
[whisper]
model_path = "~/.local/share/whisper/ggml-small.bin"

[transcription]
language = "he"  # ISO 639-1 code, or empty for auto-detect
speaker_count = 1  # Set > 1 to enable diarization
```

## Model Recommendations

| Language | Minimum Model | Recommended |
|----------|---------------|-------------|
| English | tiny | base |
| Hebrew | small | medium |
| Arabic | medium | large-v3 |
| Other | small | medium |
