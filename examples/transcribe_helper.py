#!/usr/bin/env python3
"""
Helper script for PHP integration.
Outputs transcription results as JSON for easy parsing.

Usage:
    python transcribe_helper.py --model <model_path> --audio <audio_file> --output json
"""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio to JSON")
    parser.add_argument("--model", required=True, help="Path to Whisper model")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--language", default=None, help="Language code")
    parser.add_argument("--speakers", type=int, default=1, help="Number of speakers")
    parser.add_argument("--output", choices=["json", "text"], default="json")
    args = parser.parse_args()

    try:
        from voice_transcription import TranscriptionClient, TranscriptionConfig
    except ImportError:
        print(json.dumps({"error": "voice_transcription module not found"}))
        sys.exit(1)

    try:
        # Create client
        client = TranscriptionClient.with_local_whisper(args.model)

        # Create config
        config = TranscriptionConfig(
            language=args.language,
            speaker_count=args.speakers if args.speakers > 1 else None,
        )

        # Transcribe
        result = client.transcribe(args.audio, config)

        # Build output
        output = {
            "content": result.content,
            "segments": [
                {
                    "text": seg.text,
                    "start_seconds": seg.start_seconds,
                    "end_seconds": seg.end_seconds,
                    "speaker": seg.speaker,
                    "confidence": seg.confidence,
                }
                for seg in result.segments
            ],
            "languages": result.languages,
            "duration_seconds": result.duration_seconds,
            "confidence": result.confidence,
            "speaker_count": result.speaker_count,
        }

        if args.output == "json":
            print(json.dumps(output))
        else:
            print(result.content)

    except Exception as e:
        if args.output == "json":
            print(json.dumps({"error": str(e)}))
        else:
            print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
