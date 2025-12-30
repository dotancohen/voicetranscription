"""Type stubs for voice_transcription."""

from typing import List, Optional

class Segment:
    """A segment of transcribed audio."""

    text: str
    start_seconds: float
    end_seconds: float
    speaker: Optional[str]
    confidence: Optional[float]

class TranscriptionResult:
    """The result of a transcription operation."""

    content: str
    segments: List[Segment]
    languages: Optional[List[str]]
    duration_seconds: Optional[float]
    confidence: Optional[float]
    speaker_count: Optional[int]

class TranscriptionConfig:
    """Configuration for a transcription request."""

    def __init__(
        self,
        language: Optional[str] = None,
        speaker_count: Optional[int] = None,
        word_timestamps: bool = False,
        model: Optional[str] = None,
    ) -> None: ...

class TranscriptionClient:
    """Transcription client for audio transcription."""

    @staticmethod
    def with_local_whisper(model_path: str) -> "TranscriptionClient":
        """Create a new TranscriptionClient with a local Whisper backend.

        Args:
            model_path: Path to the Whisper model file (.bin).

        Returns:
            A new TranscriptionClient instance.

        Raises:
            RuntimeError: If the model cannot be loaded.
        """
        ...

    def backend_name(self) -> str:
        """Get the name of the current backend."""
        ...

    def supported_features(self) -> List[str]:
        """Get the features supported by the current backend."""
        ...

    def is_ready(self) -> bool:
        """Check if the backend is ready to transcribe."""
        ...

    def transcribe(
        self,
        audio_path: str,
        config: Optional[TranscriptionConfig] = None,
    ) -> TranscriptionResult:
        """Transcribe an audio file.

        Args:
            audio_path: Path to the audio file.
            config: Optional transcription configuration.

        Returns:
            TranscriptionResult with the transcription.

        Raises:
            RuntimeError: If transcription fails.
        """
        ...

    def transcribe_with_language(
        self,
        audio_path: str,
        language: str,
    ) -> TranscriptionResult:
        """Transcribe an audio file with a specific language.

        Args:
            audio_path: Path to the audio file.
            language: Language code (ISO 639-1, e.g., "en", "he").

        Returns:
            TranscriptionResult with the transcription.

        Raises:
            RuntimeError: If transcription fails.
        """
        ...
