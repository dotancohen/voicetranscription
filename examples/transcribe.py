#!/usr/bin/env python3
"""
Simple example of using VoiceTranscription in Python.

Usage:
    python transcribe.py <audio_file>
    python transcribe.py --config /path/to/config.toml <audio_file>
"""

import sys
import tomllib
from pathlib import Path


def load_config(config_path: Path) -> dict:
    """Load configuration from TOML file."""
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def main():
    # Parse arguments
    args = sys.argv[1:]
    config_path = Path(__file__).parent / "config.toml"

    if "--config" in args:
        idx = args.index("--config")
        config_path = Path(args[idx + 1])
        args = args[:idx] + args[idx + 2:]

    if not args:
        print("Usage: python transcribe.py [--config config.toml] <audio_file>")
        sys.exit(1)

    audio_file = Path(args[0])
    if not audio_file.exists():
        print(f"Error: Audio file not found: {audio_file}")
        sys.exit(1)

    # Load configuration
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)

    # Import voice_transcription (must be built first with maturin)
    try:
        from voice_transcription import TranscriptionClient, TranscriptionConfig
    except ImportError:
        print("Error: voice_transcription module not found.")
        print("Build it first: cd bindings/python && maturin develop")
        sys.exit(1)

    # Get configuration values
    model_path = config.get("whisper", {}).get("model_path", "")
    language = config.get("transcription", {}).get("language", "")
    speaker_count = config.get("transcription", {}).get("speaker_count", 1)

    if not model_path or not Path(model_path).exists():
        print(f"Error: Whisper model not found: {model_path}")
        print("Download a model from: https://huggingface.co/ggerganov/whisper.cpp/tree/main")
        sys.exit(1)

    # Create client
    print(f"Loading model: {model_path}")
    client = TranscriptionClient.with_local_whisper(model_path)
    print(f"Backend: {client.backend_name()}")
    print(f"Features: {client.supported_features()}")

    # Create transcription config
    transcription_config = TranscriptionConfig(
        language=language if language else None,
        speaker_count=speaker_count if speaker_count > 1 else None,
    )

    # Transcribe
    print(f"\nTranscribing: {audio_file}")
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
