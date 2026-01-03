# VoiceTranscription Tools

Command-line tools for audio transcription using the VoiceTranscription library.

## Overview

| Tool                   | Description                                       |
|------------------------|---------------------------------------------------|
| transcribe.py          | Main transcription CLI with multi-backend support |
| find_whisper_models.py | Find installed Whisper GGML models on your system |

## Installation

### Prerequisites

1. Install the VoiceTranscription Python bindings:

```bash
cd /path/to/VoiceTranscription/bindings/python

# Install with all backends
maturin develop --features all_backends

# Or install with specific backends only
maturin develop --features "local_whisper,assemblyai"
```

2. For faster-whisper:

```bash
pip install faster-whisper
```

3. For local Whisper transcription, download a model:

```bash
# Download the base model (recommended for most uses)
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin \
     -O ~/.local/share/whisper/ggml-base.bin

# Or download a larger model for better accuracy
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin \
     -O ~/.local/share/whisper/ggml-medium.bin
```

4. For cloud backends, set up credentials:

```bash
# AssemblyAI
export ASSEMBLYAI_API_KEY="your-api-key"

# Google Cloud Speech
gcloud auth login
export GOOGLE_CLOUD_PROJECT="your-project-id"
```

## transcribe.py

A unified CLI for transcribing audio files with multiple backends.

### Quick Start

```bash
# Transcribe with local Whisper (auto-detects installed models)
./transcribe.py recording.mp3

# Transcribe with a specific language
./transcribe.py -l he recording.mp3

# Transcribe with speaker diarization
./transcribe.py -s 2 meeting.mp3

# Output as JSON
./transcribe.py -f json recording.mp3 > transcript.json
```

### Backends

#### Local Whisper (default)

Fast, private, offline transcription using whisper.cpp.

```bash
# Auto-detect model
./transcribe.py recording.mp3

# Specify model path
./transcribe.py -m /path/to/ggml-large-v3.bin recording.mp3
```

**Pros:**
- No internet connection required
- No API costs
- Data stays on your machine

**Cons:**
- Requires downloading model files (75MB - 3GB)
- CPU-intensive for larger models

#### Faster Whisper

CTranslate2-based Whisper implementation supporting custom models.

```bash
# Use a specific model
./transcribe.py -b faster_whisper -m ivrit-ai/faster-whisper-v2-d4 -l he recording.mp3

# Use OpenAI Whisper models via faster-whisper
./transcribe.py -b faster_whisper -m large-v3 recording.mp3
```

**Hebrew Models (ivrit.ai):**

| Model | Description |
|-------|-------------|
| `ivrit-ai/faster-whisper-v2-d4` | Best accuracy (recommended) |
| `ivrit-ai/whisper-large-v3-turbo-ct2` | Faster, good accuracy |
| `ivrit-ai/whisper-large-v3-ct2` | Largest, highest quality |

**Pros:**
- Faster than original Whisper (CTranslate2 optimization)
- Models download automatically from HuggingFace
- Supports CUDA for GPU acceleration

**Cons:**
- Requires disk space for model download
- First run downloads model (one-time)

**GPU Acceleration:**
```bash
# Force CPU (slower but works everywhere)
./transcribe.py -b faster_whisper -m large-v3 --device cpu recording.mp3

# Force CUDA (requires NVIDIA GPU + CUDA)
./transcribe.py -b faster_whisper -m large-v3 --device cuda recording.mp3

# Use int8 quantization (faster, less VRAM)
./transcribe.py -b faster_whisper -m large-v3 --device cuda --compute-type int8 recording.mp3
```

#### AssemblyAI

Cloud-based transcription with advanced features.

```bash
# Using environment variable
export ASSEMBLYAI_API_KEY="your-key"
./transcribe.py -b assemblyai recording.mp3

# Or specify key directly
./transcribe.py -b assemblyai -k your-key recording.mp3
```

**Features:**
- Automatic punctuation
- Speaker diarization
- Language detection
- High accuracy

**Pricing:** See https://www.assemblyai.com/pricing

#### Google Cloud Speech

Enterprise-grade transcription with Chirp model.

```bash
# Using gcloud authentication
gcloud auth login
export GOOGLE_CLOUD_PROJECT="my-project"
./transcribe.py -b google_cloud recording.mp3

# Specify options directly
./transcribe.py -b google_cloud \
    -k $(gcloud auth print-access-token) \
    -p my-project \
    --location us-central1 \
    recording.mp3
```

**Features:**
- Chirp v3 model (latest)
- Multiple language support
- Speaker diarization (region-dependent)
- Enterprise reliability

**Pricing:** See https://cloud.google.com/speech-to-text/pricing

### Command-Line Options

```
usage: transcribe.py [-h] [-b {local_whisper,faster_whisper,assemblyai,google_cloud}]
                     [-m MODEL_PATH] [--device {auto,cuda,cpu}]
                     [--compute-type {auto,float16,int8,int8_float16}]
                     [-k API_KEY] [-p PROJECT_ID] [--location LOCATION]
                     [-l LANGUAGE] [-s SPEAKERS] [-f {text,json,segments}]
                     [-o OUTPUT] [--list-backends] [--show-options BACKEND]
                     [-v] [-q]
                     [audio_file]

positional arguments:
  audio_file            Path to the audio file to transcribe

options:
  -h, --help            show this help message and exit
  -b, --backend         Transcription backend (default: local_whisper)
  -m, --model-path      Path to Whisper model file or HuggingFace model ID
  --device              Device for faster_whisper: auto, cuda, cpu
  --compute-type        Compute type: auto, float16, int8, int8_float16
  -k, --api-key         API key for cloud backends
  -p, --project-id      Google Cloud project ID
  --location            Google Cloud region (default: us-central1)
  -l, --language        Language code (e.g., 'en', 'he', 'es')
  -s, --speakers        Number of speakers for diarization
  -f, --format          Output format: text, json, segments
  -o, --output          Output file path (default: stdout)
  --list-backends       List available transcription backends
  --show-options        Show configuration options for a backend
  -v, --version         show program's version number and exit
  -q, --quiet           Suppress status messages
```

## find_whisper_models.py

Utility to find installed Whisper GGML models on your system.

### Usage

```bash
# Human-readable output
./find_whisper_models.py

# JSON output (for scripting)
./find_whisper_models.py --json

# Rust code output (for embedding in code)
./find_whisper_models.py --rust

# Search additional directories
./find_whisper_models.py --dir /my/custom/models
```

### Default Search Paths

- `~/.local/share/whisper/`
- `~/.cache/whisper/`
- `~/.cache/huggingface/hub/`
- `/usr/share/whisper/`
- `/usr/local/share/whisper/`

## Troubleshooting

### "voice_transcription module not found"

Make sure you've installed the Python bindings:

```bash
cd bindings/python
maturin develop --features all_backends
```

### "No Whisper model found"

Download a model file:

```bash
mkdir -p ~/.local/share/whisper
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin \
     -O ~/.local/share/whisper/ggml-base.bin
```

Or specify a model path directly:

```bash
./transcribe.py -m /path/to/model.bin audio.mp3
```


## Language Codes

Common language codes for the `-l` option:

| Code | Language |
|------|----------|
| en | English |
| he | Hebrew |
| ar | Arabic |
| es | Spanish |
| fr | French |
| de | German |
| it | Italian |
| pt | Portuguese |
| ru | Russian |
| zh | Chinese |
| ja | Japanese |
| ko | Korean |

For Google Cloud, use full locale codes (e.g., `en-US`, `he-IL`).

## References

- [ivrit.ai on HuggingFace](https://huggingface.co/ivrit-ai)
- [Fine Tune Whisper the Right Way - ivrit.ai](https://www.ivrit.ai/en/2025/02/13/training-whisper/)
- [Fine-Tune Whisper For Multilingual ASR - Hugging Face](https://huggingface.co/blog/fine-tune-whisper)

