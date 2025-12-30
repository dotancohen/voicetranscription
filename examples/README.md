# VoiceTranscription Examples

Simple example applications demonstrating VoiceTranscription usage.

## Setup

1. Build the Python bindings:
   ```bash
   cd ../bindings/python
   pip install maturin
   maturin develop
   ```

2. Download a Whisper model:
   ```bash
   wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin
   ```

3. Edit `config.toml` and set `model_path` to your downloaded model.

## Python Example

```bash
python transcribe.py recording.wav
python transcribe.py --config /path/to/config.toml recording.wav
```

## PHP Example

The PHP example uses a Python subprocess since native PHP bindings are not yet available.

```bash
php transcribe.php recording.wav
php transcribe.php --config /path/to/config.toml recording.wav
```

## Configuration

Edit `config.toml` to configure:

- `whisper.model_path` - Path to your GGML Whisper model
- `transcription.language` - Language hint (empty for auto-detection)
- `transcription.speaker_count` - Number of speakers for diarization
