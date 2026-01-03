#!/usr/bin/env python3
"""
Find Whisper Models

Scans common directories for Whisper GGML model files and outputs
information about found models.

Common model locations:
- ~/.local/share/whisper/
- ~/.cache/whisper/
- ~/.cache/huggingface/hub/ (models downloaded via transformers)
- /usr/share/whisper/
- /usr/local/share/whisper/

Usage:
    python find_whisper_models.py           # Human-readable output
    python find_whisper_models.py --json    # JSON output
    python find_whisper_models.py --rust    # Rust code for static list
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple


class ModelInfo(NamedTuple):
    """Information about a found model."""
    name: str
    path: str
    size_bytes: int
    size_human: str


def human_size(size_bytes: int) -> str:
    """Convert bytes to human-readable size."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def get_search_directories() -> List[Path]:
    """Get list of directories to search for models."""
    home = Path.home()

    directories = [
        home / ".local" / "share" / "whisper",
        home / ".cache" / "whisper",
        home / ".cache" / "huggingface" / "hub",
        Path("/usr/share/whisper"),
        Path("/usr/local/share/whisper"),
    ]

    # Also check XDG_DATA_HOME if set
    xdg_data = os.environ.get("XDG_DATA_HOME")
    if xdg_data:
        directories.append(Path(xdg_data) / "whisper")

    # Check XDG_CACHE_HOME if set
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        directories.append(Path(xdg_cache) / "whisper")

    return directories


def find_models(directories: List[Path]) -> List[ModelInfo]:
    """Find all Whisper models in the given directories."""
    models: List[ModelInfo] = []
    seen_paths: set = set()

    for directory in directories:
        if not directory.exists():
            continue

        # Look for ggml-*.bin files (standard Whisper model format)
        for model_file in directory.rglob("ggml-*.bin"):
            # Resolve to handle symlinks
            resolved = model_file.resolve()
            if str(resolved) in seen_paths:
                continue
            seen_paths.add(str(resolved))

            # Extract model name from filename
            # e.g., "ggml-base.bin" -> "base"
            # e.g., "ggml-large-v3.bin" -> "large-v3"
            name = model_file.stem
            if name.startswith("ggml-"):
                name = name[5:]  # Remove "ggml-" prefix

            size = model_file.stat().st_size
            models.append(ModelInfo(
                name=name,
                path=str(resolved),
                size_bytes=size,
                size_human=human_size(size),
            ))

    # Sort by name
    return sorted(models, key=lambda m: m.name)


def output_human(models: List[ModelInfo], directories: List[Path]) -> None:
    """Output models in human-readable format."""
    print("Whisper Model Finder")
    print("=" * 60)
    print()
    print("Searched directories:")
    for d in directories:
        exists = "[exists]" if d.exists() else "[not found]"
        print(f"  {exists} {d}")
    print()

    if not models:
        print("No Whisper models found.")
        print()
        print("To download models, visit:")
        print("  https://huggingface.co/ggerganov/whisper.cpp")
        print()
        print("Or use whisper.cpp's download script:")
        print("  ./models/download-ggml-model.sh base")
        return

    print(f"Found {len(models)} model(s):")
    print()

    for model in models:
        print(f"  {model.name}")
        print(f"    Path: {model.path}")
        print(f"    Size: {model.size_human}")
        print()


def output_json(models: List[ModelInfo]) -> None:
    """Output models as JSON."""
    data = {
        "models": [
            {
                "name": m.name,
                "path": m.path,
                "size_bytes": m.size_bytes,
                "size_human": m.size_human,
            }
            for m in models
        ]
    }
    print(json.dumps(data, indent=2))


def output_rust(models: List[ModelInfo]) -> None:
    """Output models as Rust code for static inclusion."""
    print("// Auto-generated model list from find_whisper_models.py")
    print("// Update by running: python tools/find_whisper_models.py --rust")
    print()
    print("pub static INSTALLED_MODELS: &[(&str, &str)] = &[")
    for model in models:
        # Escape path for Rust string
        escaped_path = model.path.replace("\\", "\\\\").replace('"', '\\"')
        print(f'    ("{model.name}", "{escaped_path}"),')
    print("];")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find Whisper GGML models on the system.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "--rust",
        action="store_true",
        help="Output as Rust code",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        action="append",
        dest="extra_dirs",
        help="Additional directory to search (can be specified multiple times)",
    )

    args = parser.parse_args()

    directories = get_search_directories()
    if args.extra_dirs:
        directories.extend(args.extra_dirs)

    models = find_models(directories)

    if args.json:
        output_json(models)
    elif args.rust:
        output_rust(models)
    else:
        output_human(models, directories)

    return 0 if models else 1


if __name__ == "__main__":
    sys.exit(main())
