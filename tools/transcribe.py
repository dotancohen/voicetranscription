#!/usr/bin/env python3
"""
VoiceTranscribers - Command-line audio transcription tool.

This tool provides a unified interface for transcribing audio files using
multiple transcription backends:

- Local Whisper: Fast, private, offline transcription using whisper.cpp
- Faster Whisper: CTranslate2-based Whisper with custom models
- AssemblyAI: Cloud-based transcription with speaker diarization
- Google Cloud Speech: High-quality cloud transcription with Chirp model

Examples:
    # Transcribe with local Whisper (default)
    python transcribe.py audio.mp3

    # Transcribe with faster-whisper and a specific model
    python transcribe.py -b faster_whisper -m ivrit-ai/faster-whisper-v2-d4 audio.mp3

    # Transcribe with AssemblyAI
    python transcribe.py --backend assemblyai --api-key YOUR_KEY audio.mp3

    # Transcribe with Google Cloud Speech
    python transcribe.py --backend google_cloud --project-id my-project audio.mp3

    # Transcribe with speaker diarization (2 speakers)
    python transcribe.py --speakers 2 audio.mp3

    # Transcribe in a specific language
    python transcribe.py --language he audio.mp3

    # List available backends
    python transcribe.py --list-backends

    # Show backend configuration options
    python transcribe.py --show-options local_whisper

See also:
    transcribe-hebrew-optimized.py - Wrapper for Hebrew with ivrit.ai model
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

def get_version():
    """Get the version string."""
    return "1.0.0"

def check_voice_transcription():
    """Check if voice_transcription module is available."""
    try:
        import voice_transcription
        return True
    except ImportError:
        return False

def list_backends():
    """List available transcription backends."""
    print("Available Transcription Backends:")
    print("=" * 50)

    # Check Rust-based backends
    try:
        from voice_transcription import get_available_backends, get_provider_schemas

        backends = get_available_backends()
        schemas = {s.provider_id: s for s in get_provider_schemas()}

        for backend_id in backends:
            schema = schemas.get(backend_id)
            if schema:
                print(f"\n  {schema.provider_id}")
                print(f"    Name: {schema.provider_name}")
                print(f"    Options: {len(schema.options)} configurable")
            else:
                print(f"\n  {backend_id}")
                print("    (No schema available)")
    except ImportError:
        print("\n  (Rust backends not available - run: maturin develop --features all_backends)")

    # Check faster-whisper backend
    try:
        import faster_whisper
        print(f"\n  faster_whisper")
        print(f"    Name: Faster Whisper (CTranslate2)")
        print(f"    Supports: ivrit.ai Hebrew models, OpenAI Whisper models")
        print(f"    See: transcribe-hebrew-optimized.py for Hebrew")
    except ImportError:
        print(f"\n  faster_whisper (not installed)")
        print(f"    Install with: pip install faster-whisper")

    print()
    return 0

def show_backend_options(backend_id: str):
    """Show configuration options for a specific backend."""
    try:
        from voice_transcription import get_provider_schemas

        schemas = {s.provider_id: s for s in get_provider_schemas()}
        schema = schemas.get(backend_id)

        if not schema:
            print(f"Error: Backend '{backend_id}' not found.", file=sys.stderr)
            print(f"Available backends: {', '.join(schemas.keys())}", file=sys.stderr)
            return 1

        print(f"Configuration Options for {schema.provider_name} ({schema.provider_id})")
        print("=" * 60)

        for opt in schema.options:
            required = " (REQUIRED)" if opt.required else ""
            print(f"\n  --{opt.id.replace('_', '-')}{required}")
            print(f"    Type: {opt.option_type}")
            if opt.default:
                print(f"    Default: {opt.default}")
            if opt.description:
                print(f"    Description: {opt.description}")
            if opt.values:
                print("    Allowed values:")
                for val in opt.values:
                    print(f"      - {val.value}: {val.label}")

        print()
        return 0

    except ImportError as e:
        print(f"Error: voice_transcription module not found: {e}", file=sys.stderr)
        return 1

def find_whisper_model() -> Optional[str]:
    """Find an installed Whisper model."""
    search_paths = [
        Path.home() / ".local" / "share" / "whisper",
        Path.home() / ".cache" / "whisper",
        Path.home() / ".cache" / "huggingface" / "hub",
        Path("/usr/share/whisper"),
        Path("/usr/local/share/whisper"),
    ]

    for search_path in search_paths:
        if search_path.exists():
            for model_file in search_path.rglob("*.bin"):
                if "ggml" in model_file.name.lower():
                    return str(model_file)

    return None

def transcribe_with_faster_whisper(
    audio_path: str,
    model_id: str,
    language: Optional[str] = None,
    output_format: str = "text",
    output_file: Optional[str] = None,
    device: str = "auto",
    compute_type: str = "auto",
) -> int:
    """Transcribe using faster-whisper backend."""
    try:
        from voice_transcription_py import FasterWhisperClient, TranscriptionConfig
    except ImportError:
        try:
            # Try importing directly if the package isn't installed
            import sys
            bindings_path = Path(__file__).parent.parent / "bindings" / "python"
            sys.path.insert(0, str(bindings_path))
            from voice_transcription_py import FasterWhisperClient, TranscriptionConfig
        except ImportError:
            print("Error: voice_transcription_py module not found.", file=sys.stderr)
            print("Make sure faster-whisper is installed: pip install faster-whisper", file=sys.stderr)
            return 1

    print(f"Transcribing: {audio_path}", file=sys.stderr)
    print(f"Backend: faster_whisper", file=sys.stderr)
    print(f"Model: {model_id}", file=sys.stderr)
    if language:
        print(f"Language: {language}", file=sys.stderr)

    try:
        client = FasterWhisperClient(model_id, device=device, compute_type=compute_type)
        config = TranscriptionConfig(language=language)
        result = client.transcribe(audio_path, config)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Format output
    if output_format == "json":
        output = json.dumps({
            "content": result.content,
            "segments": [
                {
                    "text": s.text,
                    "start": s.start_seconds,
                    "end": s.end_seconds,
                    "speaker": s.speaker,
                    "confidence": s.confidence,
                }
                for s in result.segments
            ],
            "languages": result.languages,
            "duration_seconds": result.duration_seconds,
            "confidence": result.confidence,
            "speaker_count": result.speaker_count,
        }, indent=2, ensure_ascii=False)
    elif output_format == "segments":
        lines = []
        for s in result.segments:
            speaker = f"[{s.speaker}] " if s.speaker else ""
            lines.append(f"[{s.start_seconds:.2f} - {s.end_seconds:.2f}] {speaker}{s.text}")
        output = "\n".join(lines)
    else:  # text
        output = result.content

    # Write output
    if output_file:
        Path(output_file).write_text(output, encoding="utf-8")
        print(f"Output written to: {output_file}", file=sys.stderr)
    else:
        print(output)

    return 0


def transcribe_file(
    audio_path: str,
    backend: str = "local_whisper",
    model_path: Optional[str] = None,
    api_key: Optional[str] = None,
    project_id: Optional[str] = None,
    location: Optional[str] = None,
    language: Optional[str] = None,
    speakers: Optional[int] = None,
    output_format: str = "text",
    output_file: Optional[str] = None,
    device: str = "auto",
    compute_type: str = "auto",
) -> int:
    """Transcribe an audio file."""
    # Handle faster_whisper backend separately (pure Python)
    if backend == "faster_whisper":
        model_id = model_path or "large-v3"  # Default to large-v3 if no model specified
        return transcribe_with_faster_whisper(
            audio_path, model_id, language, output_format, output_file, device, compute_type
        )

    try:
        from voice_transcription import TranscriptionClient, TranscriptionConfig

        # Create the appropriate client
        if backend == "local_whisper":
            if not model_path:
                model_path = find_whisper_model()
                if not model_path:
                    print("Error: No Whisper model found. Please specify --model-path.", file=sys.stderr)
                    print("You can download models from: https://huggingface.co/ggerganov/whisper.cpp", file=sys.stderr)
                    return 1
                print(f"Using model: {model_path}", file=sys.stderr)

            client = TranscriptionClient.with_local_whisper(model_path)

        elif backend == "assemblyai":
            if not api_key:
                api_key = os.environ.get("ASSEMBLYAI_API_KEY")
            if not api_key:
                print("Error: AssemblyAI API key required. Use --api-key or set ASSEMBLYAI_API_KEY.", file=sys.stderr)
                return 1

            client = TranscriptionClient.with_assemblyai(api_key)

        elif backend == "google_cloud":
            if not api_key:
                # Try to get access token from gcloud
                import subprocess
                try:
                    result = subprocess.run(
                        ["gcloud", "auth", "print-access-token"],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    api_key = result.stdout.strip()
                except (subprocess.CalledProcessError, FileNotFoundError):
                    print("Error: Google Cloud access token required.", file=sys.stderr)
                    print("Use --api-key or run: gcloud auth login", file=sys.stderr)
                    return 1

            if not project_id:
                project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
            if not project_id:
                print("Error: Google Cloud project ID required. Use --project-id or set GOOGLE_CLOUD_PROJECT.", file=sys.stderr)
                return 1

            client = TranscriptionClient.with_google_cloud(api_key, project_id, location)

        else:
            print(f"Error: Unknown backend '{backend}'", file=sys.stderr)
            return 1

        # Create configuration
        config = TranscriptionConfig(
            language=language,
            speaker_count=speakers,
            word_timestamps=True,
        )

        # Perform transcription
        print(f"Transcribing: {audio_path}", file=sys.stderr)
        print(f"Backend: {client.backend_name()}", file=sys.stderr)
        if language:
            print(f"Language: {language}", file=sys.stderr)
        if speakers:
            print(f"Speakers: {speakers}", file=sys.stderr)

        result = client.transcribe(audio_path, config)

        # Format output
        if output_format == "json":
            output = json.dumps({
                "content": result.content,
                "segments": [
                    {
                        "text": s.text,
                        "start": s.start_seconds,
                        "end": s.end_seconds,
                        "speaker": s.speaker,
                        "confidence": s.confidence,
                    }
                    for s in result.segments
                ],
                "languages": result.languages,
                "duration_seconds": result.duration_seconds,
                "confidence": result.confidence,
                "speaker_count": result.speaker_count,
            }, indent=2, ensure_ascii=False)
        elif output_format == "segments":
            lines = []
            for s in result.segments:
                speaker = f"[{s.speaker}] " if s.speaker else ""
                lines.append(f"[{s.start_seconds:.2f} - {s.end_seconds:.2f}] {speaker}{s.text}")
            output = "\n".join(lines)
        else:  # text
            output = result.content

        # Write output
        if output_file:
            Path(output_file).write_text(output, encoding="utf-8")
            print(f"Output written to: {output_file}", file=sys.stderr)
        else:
            print(output)

        return 0

    except ImportError as e:
        print(f"Error: voice_transcription module not found: {e}", file=sys.stderr)
        print("Please install it with: cd bindings/python && maturin develop --features all_backends", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="VoiceTranscribers - Audio transcription with multiple backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s audio.mp3                     Transcribe with local Whisper
  %(prog)s -b assemblyai -k KEY audio.mp3  Transcribe with AssemblyAI
  %(prog)s -b google_cloud -p PROJ audio.mp3  Transcribe with Google Cloud
  %(prog)s -s 2 audio.mp3                 Transcribe with 2 speakers
  %(prog)s -l he audio.mp3                 Transcribe in Hebrew
  %(prog)s -f json audio.mp3               Output as JSON
  %(prog)s --list-backends                 List available backends
  %(prog)s --show-options local_whisper    Show backend options

Environment Variables:
  ASSEMBLYAI_API_KEY     API key for AssemblyAI backend
  GOOGLE_CLOUD_PROJECT   Project ID for Google Cloud backend

For more information, see: https://github.com/dotancohen/VoiceTranscription
        """,
    )

    parser.add_argument(
        "audio_file",
        nargs="?",
        help="Path to the audio file to transcribe",
    )

    parser.add_argument(
        "-b", "--backend",
        choices=["local_whisper", "faster_whisper", "assemblyai", "google_cloud"],
        default="local_whisper",
        help="Transcription backend to use (default: local_whisper)",
    )

    parser.add_argument(
        "-m", "--model-path",
        help="Path to Whisper model file or HuggingFace model ID",
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device for faster_whisper (default: auto)",
    )

    parser.add_argument(
        "--compute-type",
        choices=["auto", "float16", "int8", "int8_float16"],
        default="auto",
        help="Compute type for faster_whisper (default: auto)",
    )

    parser.add_argument(
        "-k", "--api-key",
        help="API key for cloud backends",
    )

    parser.add_argument(
        "-p", "--project-id",
        help="Google Cloud project ID (for google_cloud backend)",
    )

    parser.add_argument(
        "--location",
        default="us-central1",
        help="Google Cloud region (default: us-central1)",
    )

    parser.add_argument(
        "-l", "--language",
        help="Language code (e.g., 'en', 'he', 'es'). Default: auto-detect",
    )

    parser.add_argument(
        "-s", "--speakers",
        type=int,
        help="Number of speakers for diarization",
    )

    parser.add_argument(
        "-f", "--format",
        choices=["text", "json", "segments"],
        default="text",
        help="Output format (default: text)",
    )

    parser.add_argument(
        "-o", "--output",
        help="Output file path (default: stdout)",
    )

    parser.add_argument(
        "--list-backends",
        action="store_true",
        help="List available transcription backends",
    )

    parser.add_argument(
        "--show-options",
        metavar="BACKEND",
        help="Show configuration options for a backend",
    )

    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"%(prog)s {get_version()}",
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress status messages",
    )

    args = parser.parse_args()

    # Handle special commands
    if args.list_backends:
        return list_backends()

    if args.show_options:
        return show_backend_options(args.show_options)

    # Require audio file for transcription
    if not args.audio_file:
        parser.print_help()
        print("\nError: audio_file is required for transcription", file=sys.stderr)
        return 1

    # Check if file exists
    if not Path(args.audio_file).exists():
        print(f"Error: File not found: {args.audio_file}", file=sys.stderr)
        return 1

    # Redirect stderr to /dev/null if quiet mode
    if args.quiet:
        sys.stderr = open(os.devnull, 'w')

    return transcribe_file(
        audio_path=args.audio_file,
        backend=args.backend,
        model_path=args.model_path,
        api_key=args.api_key,
        project_id=args.project_id,
        location=args.location,
        language=args.language,
        speakers=args.speakers,
        output_format=args.format,
        output_file=args.output,
        device=args.device,
        compute_type=args.compute_type,
    )

if __name__ == "__main__":
    sys.exit(main())
