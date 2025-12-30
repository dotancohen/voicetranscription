#!/usr/bin/env python3
"""
Simple example of using VoiceTranscription in Python.

Usage:
    python transcribe.py <audio_file>
    python transcribe.py --language he <audio_file>
    python transcribe.py --model small --language en <audio_file>
"""

import argparse
import sys
import tomllib
from pathlib import Path

# Canonical location for Whisper models
WHISPER_MODELS_DIR = Path.home() / ".local" / "share" / "whisper"

# Model sizes in order of preference (larger = better for non-English)
MODEL_PRIORITY = ["ggml-large-v3.bin", "ggml-medium.bin", "ggml-small.bin", "ggml-base.bin", "ggml-tiny.bin"]


def find_available_models() -> list[Path]:
    """Find all available Whisper models in the canonical location."""
    if not WHISPER_MODELS_DIR.exists():
        return []
    return sorted(WHISPER_MODELS_DIR.glob("ggml-*.bin"))


def select_model(configured_path: Path | None, cli_model: str | None) -> Path | None:
    """
    Select a model with the following priority:
    1. CLI --model argument
    2. Config file model_path (if exists)
    3. Auto-select best available from canonical location
    """
    # 1. CLI argument takes priority
    if cli_model:
        # Check if it's a full path or just a filename
        cli_path = Path(cli_model).expanduser()
        if cli_path.exists():
            return cli_path
        # Try canonical location
        canonical_path = WHISPER_MODELS_DIR / cli_model
        if canonical_path.exists():
            return canonical_path
        # Try adding ggml- prefix if missing
        if not cli_model.startswith("ggml-"):
            canonical_path = WHISPER_MODELS_DIR / f"ggml-{cli_model}.bin"
            if canonical_path.exists():
                return canonical_path
        print(f"Warning: Specified model not found: {cli_model}")

    # 2. Config file path
    if configured_path and configured_path.exists():
        return configured_path

    # 3. Find available models
    available = find_available_models()
    if not available:
        return None

    if len(available) == 1:
        print(f"Using available model: {available[0].name}")
        return available[0]

    # Multiple models available - prefer larger models for better multilingual support
    for preferred in MODEL_PRIORITY:
        for model in available:
            if model.name == preferred:
                print(f"Auto-selected model: {model.name} (best available for multilingual)")
                return model

    # Fall back to first available
    return available[0]


def load_config(config_path: Path) -> dict:
    """Load configuration from TOML file."""
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def list_models():
    """List available models and exit."""
    available = find_available_models()
    if available:
        print(f"Available models in {WHISPER_MODELS_DIR}:")
        for model in available:
            size_mb = model.stat().st_size / (1024 * 1024)
            print(f"  {model.name} ({size_mb:.0f} MB)")
    else:
        print(f"No models found in {WHISPER_MODELS_DIR}")
        print("Download with: wget -P ~/.local/share/whisper/ https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin")
    sys.exit(0)


def show_cache_info():
    """Show cache information and exit."""
    try:
        from voice_transcription import get_cache_info
        info = get_cache_info()
        print(f"Cache directory: {info.path}")
        print(f"Cache size: {info.size_human()}")
        print(f"Cached files: {info.file_count}")
    except ImportError:
        print("Error: voice_transcription module not found.")
        print("Build it first: cd bindings/python && maturin develop")
        sys.exit(1)
    sys.exit(0)


def clear_cache():
    """Clear the cache and exit."""
    try:
        from voice_transcription import clear_cache as do_clear_cache
        removed = do_clear_cache()
        print(f"Removed {removed} cached file(s)")
    except ImportError:
        print("Error: voice_transcription module not found.")
        print("Build it first: cd bindings/python && maturin develop")
        sys.exit(1)
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using VoiceTranscription.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s recording.wav
  %(prog)s --language he recording.wav
  %(prog)s --model small --language en recording.wav
  %(prog)s --nocache recording.mp3

Model Selection Priority:
  1. --model argument (highest priority)
  2. model_path from config file
  3. Auto-select best available from ~/.local/share/whisper/
"""
    )

    # Transcription options (match Rust TranscriptionConfig field names)
    parser.add_argument("audio_file", nargs="?", help="Path to the audio file to transcribe")
    parser.add_argument("--language", metavar="CODE",
                        help="Language hint (ISO 639-1 code, e.g., 'en', 'he'). Auto-detects if not specified.")
    parser.add_argument("--speaker_count", type=int, metavar="N",
                        help="Expected number of speakers for diarization")
    parser.add_argument("--word_timestamps", action="store_true",
                        help="Enable word-level timestamps")
    parser.add_argument("--model", metavar="NAME",
                        help="Model file or name (e.g., 'small', 'ggml-small.bin', or full path)")
    parser.add_argument("--nocache", action="store_true",
                        help="Skip audio conversion cache (always convert fresh)")

    # Configuration
    parser.add_argument("--config", metavar="PATH",
                        help="Path to config.toml (default: examples/config.toml)")

    # Utility commands
    parser.add_argument("--list-models", action="store_true",
                        help="List available models and exit")
    parser.add_argument("--cache-info", action="store_true",
                        help="Show cache information and exit")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Clear the audio conversion cache and exit")

    args = parser.parse_args()

    # Handle utility commands
    if args.list_models:
        list_models()
    if args.cache_info:
        show_cache_info()
    if args.clear_cache:
        clear_cache()

    # Require audio file for transcription
    if not args.audio_file:
        parser.print_help()
        sys.exit(1)

    audio_file = Path(args.audio_file)
    if not audio_file.exists():
        print(f"Error: Audio file not found: {audio_file}")
        sys.exit(1)

    # Load configuration
    config_path = Path(args.config) if args.config else Path(__file__).parent / "config.toml"
    config = {}
    if config_path.exists():
        config = load_config(config_path)

    # Import voice_transcription (must be built first with maturin)
    try:
        from voice_transcription import TranscriptionClient, TranscriptionConfig
    except ImportError:
        print("Error: voice_transcription module not found.")
        print("Build it first: cd bindings/python && maturin develop")
        sys.exit(1)

    # Get configuration values (CLI args override config file)
    configured_model = config.get("whisper", {}).get("model_path", "")
    configured_path = Path(configured_model).expanduser() if configured_model else None

    # Language: CLI > config > None (auto-detect)
    language = args.language
    if not language:
        language = config.get("transcription", {}).get("language", "") or None

    # Speaker count: CLI > config > None
    speaker_count = args.speaker_count
    if speaker_count is None:
        speaker_count = config.get("transcription", {}).get("speaker_count", 1)
        if speaker_count <= 1:
            speaker_count = None

    # Select model
    model_path = select_model(configured_path, args.model)

    if not model_path:
        print(f"Error: No Whisper model found.")
        print(f"Models directory: {WHISPER_MODELS_DIR}")
        print()
        print("Download a model:")
        print("  wget -P ~/.local/share/whisper/ https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin")
        sys.exit(1)

    # Create client
    print(f"Loading model: {model_path}")
    client = TranscriptionClient.with_local_whisper(str(model_path))
    print(f"Backend: {client.backend_name()}")
    print(f"Features: {client.supported_features()}")

    # Create transcription config
    transcription_config = TranscriptionConfig(
        language=language,
        speaker_count=speaker_count,
        word_timestamps=args.word_timestamps,
        nocache=args.nocache,
    )

    # Transcribe
    print(f"\nTranscribing: {audio_file}")
    if language:
        print(f"Language: {language}")
    print("-" * 40)

    result = client.transcribe(str(audio_file), transcription_config)

    # Output results
    print(f"\nContent:\n{result.content}")

    if result.languages:
        print(f"\nDetected languages: {result.languages}")

    if result.duration_seconds:
        print(f"Duration: {result.duration_seconds:.2f}s")

    if result.segments:
        print(f"\nSegments ({len(result.segments)}):")
        for i, seg in enumerate(result.segments, 1):
            speaker = f" [{seg.speaker}]" if seg.speaker else ""
            print(f"  {i}. [{seg.start_seconds:.2f}s - {seg.end_seconds:.2f}s]{speaker} {seg.text}")


if __name__ == "__main__":
    main()
