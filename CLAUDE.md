# Claude Code Instructions - VoiceTranscription

## Technical decisions

Decisions that apply to more than one project in the Voice Family — data rules,
time handling, thresholds, naming, interface conventions, testing rules — are in
`../TECHNICAL-DECISIONS.md`. Read it before changing behaviour that the other
projects share, and record new cross-project decisions there rather than in this
file. Two of its rules are absolute and are repeated here because they are easy
to break: **6.5 a failing test is never negotiated with** — never make a test
pass by writing data, loosening an assertion or changing the fixture's premise;
and **4.4 no ambiguous words** — name things for exactly what they do.


## Project Overview

VoiceTranscription is a modular, cross-platform audio transcription library written in Rust with bindings for Python (PyO3) and Android/Kotlin (UniFFI).

## Project Structure

```
VoiceTranscription/
├── src/                    # Core Rust library
│   ├── lib.rs              # Main library entry point
│   ├── types.rs            # TranscriptionResult, Segment, TranscriptionConfig
│   ├── error.rs            # Error types
│   ├── schema.rs           # Provider options schema for UI generation
│   ├── backend.rs          # TranscriptionBackend trait
│   ├── client.rs           # TranscriptionClient async wrapper
│   └── backends/           # Backend implementations
│       ├── mod.rs
│       └── local_whisper.rs  # Local Whisper (whisper-rs) backend
├── bindings/
│   ├── python/             # Python bindings (PyO3)
│   │   ├── src/lib.rs
│   │   ├── Cargo.toml
│   │   └── pyproject.toml
│   └── android/            # Android/Kotlin bindings (UniFFI)
│       ├── src/lib.rs
│       ├── src/voice_transcription.udl
│       ├── build.rs
│       └── Cargo.toml
├── tools/
│   └── find_whisper_models.py  # Script to find installed Whisper models
├── Cargo.toml              # Workspace + core library definition
└── CLAUDE.md               # This file
```

## Architecture

### Core Library (`voice-transcription`)

The core library provides:

1. **Types** (`types.rs`):
   - `TranscriptionResult`: Contains `content`, `segments`, `languages`, `duration_seconds`, `confidence`, `speaker_count`
   - `Segment`: Contains `text`, `start_seconds`, `end_seconds`, `speaker`, `confidence`
   - `TranscriptionConfig`: Configuration for transcription requests

2. **Provider Schema** (`schema.rs`):
   - `ProviderSchema`: Describes a provider's configurable options for UI generation
   - `ProviderOption`: A single configurable option (id, label, type, default, values)
   - `OptionType`: Text, Number, Select, Checkbox, or Path
   - `OptionValue`: Value/label pair for Select options
   - `HasProviderSchema`: Trait implemented by backends to expose their schema

3. **Backend Trait** (`backend.rs`):
   - `TranscriptionBackend`: Async trait for pluggable transcription backends
   - `BackendConfig`: Configuration for initializing backends

4. **Client** (`client.rs`):
   - `TranscriptionClient`: High-level async client wrapping a backend

5. **Backends** (`backends/`):
   - `LocalWhisperBackend`: Local transcription using whisper-rs

### Bindings

- **Python** (`bindings/python/`): Uses PyO3 with maturin for building
- **Android** (`bindings/android/`): Uses UniFFI for Kotlin code generation

## Building

### Core Library
```bash
cargo build --release
```

### Python Bindings
```bash
cd bindings/python
maturin develop  # For development
maturin build --release  # For distribution
```

### Android Bindings
The crate uses UniFFI proc-macros only (no `.udl`, no `build.rs`). whisper.cpp is built by CMake
from the `whisper-rs-sys` build script, which needs the NDK passed as `CMAKE_*` environment
variables in addition to `cargo ndk`; the full command is in README.md ("Build Android Bindings").
The library links `libc++_shared.so`, which the APK must ship. Generate Kotlin with
`uniffi-bindgen generate --library <the .so> --language kotlin`.

`TranscriptionConfig.beam_size` (core, Python and Android): `None`/1 = greedy (the default,
unchanged behaviour), n > 1 = whisper.cpp beam search with n beams (slower, more accurate on
Hebrew). `LocalWhisperBackend` uses every CPU core (`available_parallelism`, capped at 16).

## Adding New Backends

1. Create a new file in `src/backends/` (e.g., `google_cloud.rs`)
2. Implement the `TranscriptionBackend` trait
3. Implement the `HasProviderSchema` trait to expose configurable options
4. Add the backend to `src/backends/mod.rs`
5. Add factory methods to bindings if needed
6. Add the schema to `get_provider_schemas()` in Python bindings

### Provider Schema Example

```rust
impl HasProviderSchema for MyBackend {
    fn get_provider_schema() -> ProviderSchema {
        ProviderSchema::new("my_backend", "My Backend")
            .with_option(
                ProviderOption::new("api_key", "API Key", OptionType::Text)
                    .required()
                    .with_description("Your API key for authentication"),
            )
            .with_option(
                ProviderOption::new("model", "Model", OptionType::Select)
                    .with_default("standard")
                    .with_values(vec![
                        OptionValue::new("standard", "Standard"),
                        OptionValue::new("enhanced", "Enhanced"),
                    ]),
            )
    }
}
```

## Tools

### find_whisper_models.py

Scans the system for installed Whisper GGML models:

```bash
python tools/find_whisper_models.py           # Human-readable output
python tools/find_whisper_models.py --json    # JSON output
python tools/find_whisper_models.py --rust    # Rust code for static list
python tools/find_whisper_models.py --dir /custom/path  # Add custom search dir
```

Searched locations:
- `~/.local/share/whisper/`
- `~/.cache/whisper/`
- `~/.cache/huggingface/hub/`
- `/usr/share/whisper/`
- `/usr/local/share/whisper/`

## Cross-Platform Constraint - CRITICAL

VoiceTranscription is a **shared library** that MUST work on ALL target platforms:
- Desktop (via Python/PyO3 bindings)
- Android (via Kotlin/UniFFI bindings)

### What Belongs Here
- Rust code that compiles for all targets
- Platform-agnostic abstractions (traits, types, interfaces)
- Backends that can be implemented in Rust and cross-compiled (e.g., whisper-rs, ct2rs)

### What Does NOT Belong Here
- Python-only code or dependencies (e.g., faster-whisper Python library)
- Android-only code
- Platform-specific implementations that cannot work on other platforms

### Where Platform-Specific Code Goes
- **Hebrew-specific Python tools**: `~/Projects/Voice-Related/VoiceTranscriptionHebrew/`
- **Android-specific code**: `~/Projects/VoiceFamily/VoiceAndroid/`
- **Desktop Python app code**: `~/Projects/VoiceFamily/Voice/`

If you're tempted to add a "pure Python backend" or similar, STOP. That code belongs in a platform-specific repository, not here.

### CTranslate2 Models (ivrit.ai Hebrew models)

The ivrit.ai Hebrew-optimized models use CTranslate2 format, not GGML. Options:
- **ct2rs crate**: Direct Rust bindings to CTranslate2 C++ library - could enable cross-platform support if it compiles for Android (needs investigation)
- **faster-whisper-rs**: Wraps Python API, NOT cross-platform (requires Python runtime)

## Design Decisions

- **Async-first**: The core API is async for non-blocking operation
- **Modular backends**: Easy to swap or experiment with different transcription engines
- **No config files**: Host applications pass all configuration at initialization
- **Normalized output**: All backends return the same `TranscriptionResult` format
- **languages is plural**: The `languages` field can contain multiple detected languages in order of confidence
- **Schema-driven UI**: Backends expose their options via `HasProviderSchema` for dynamic UI generation
