"""Audio file validation and preprocessing utilities."""

import mimetypes
from pathlib import Path
from typing import Tuple, Optional
import io

import config


class AudioValidationError(Exception):
    """Custom exception for audio validation errors."""
    pass


def validate_file_size(file_size: int) -> None:
    """
    Validate that file size is within limits.

    Args:
        file_size: Size of file in bytes

    Raises:
        AudioValidationError: If file size exceeds limit
    """
    if file_size > config.MAX_FILE_SIZE_BYTES:
        raise AudioValidationError(
            f"File size {file_size / 1024 / 1024:.2f}MB exceeds maximum allowed size of {config.MAX_FILE_SIZE_MB}MB"
        )


def validate_file_format(filename: str, content_type: Optional[str] = None) -> None:
    """
    Validate that file format is supported.

    Args:
        filename: Name of the file
        content_type: MIME type of the file (optional)

    Raises:
        AudioValidationError: If file format is not supported
    """
    # Get file extension
    file_ext = Path(filename).suffix.lower().lstrip(".")

    # Check file extension
    if file_ext not in config.SUPPORTED_FORMATS:
        raise AudioValidationError(
            f"Unsupported file format '{file_ext}'. "
            f"Supported formats: {', '.join(config.SUPPORTED_FORMATS)}"
        )

    # Check MIME type if provided
    if content_type and content_type not in config.SUPPORTED_MIME_TYPES:
        # Try to guess MIME type from filename
        guessed_type, _ = mimetypes.guess_type(filename)
        if guessed_type not in config.SUPPORTED_MIME_TYPES:
            raise AudioValidationError(
                f"Unsupported MIME type '{content_type}'. "
                f"Expected one of: {', '.join(config.SUPPORTED_MIME_TYPES)}"
            )


def get_audio_duration(audio_bytes: bytes) -> float:
    """
    Get duration of audio file in seconds.

    Args:
        audio_bytes: Raw audio file bytes

    Returns:
        Duration in seconds

    Raises:
        AudioValidationError: If unable to read audio file
    """
    try:
        import ffmpeg
        import tempfile
        import os

        # Write to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".audio") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name

        try:
            # Get duration using ffprobe
            probe = ffmpeg.probe(tmp_path)
            duration = float(probe['format']['duration'])
            return duration
        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    except Exception as e:
        raise AudioValidationError(f"Failed to read audio file: {str(e)}")


def validate_audio_duration(audio_bytes: bytes) -> float:
    """
    Validate that audio duration is within limits.

    Args:
        audio_bytes: Raw audio file bytes

    Returns:
        Duration in seconds

    Raises:
        AudioValidationError: If duration exceeds limit
    """
    duration = get_audio_duration(audio_bytes)

    if duration > config.MAX_DURATION_SECONDS:
        raise AudioValidationError(
            f"Audio duration {duration:.1f}s exceeds maximum allowed duration of "
            f"{config.MAX_DURATION_SECONDS}s ({config.MAX_DURATION_SECONDS / 60:.1f} minutes)"
        )

    return duration


def preprocess_audio(audio_bytes: bytes) -> bytes:
    """
    Preprocess audio file for optimal transcription.
    - Convert to WAV format
    - Resample to 24kHz (Kyutai STT's native rate)
    - Convert to mono

    Args:
        audio_bytes: Raw audio file bytes

    Returns:
        Preprocessed audio bytes in WAV format

    Raises:
        AudioValidationError: If preprocessing fails
    """
    try:
        import ffmpeg
        import tempfile
        import os

        # Write input to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".audio") as tmp_file:
            tmp_file.write(audio_bytes)
            input_path = tmp_file.name

        # Create output temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            output_path = tmp_file.name

        try:
            # Use ffmpeg to convert: mono, 16kHz, WAV format
            (
                ffmpeg
                .input(input_path)
                .output(output_path, acodec='pcm_s16le', ac=1, ar=config.SAMPLE_RATE)
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True, quiet=True)
            )

            # Read the output file
            with open(output_path, 'rb') as f:
                return f.read()

        finally:
            # Cleanup
            if os.path.exists(input_path):
                os.remove(input_path)
            if os.path.exists(output_path):
                os.remove(output_path)

    except Exception as e:
        raise AudioValidationError(f"Failed to preprocess audio: {str(e)}")


def validate_audio_file(
    filename: str,
    file_size: int,
    audio_bytes: bytes,
    content_type: Optional[str] = None,
) -> Tuple[float, bytes]:
    """
    Complete validation and preprocessing of audio file.

    Args:
        filename: Name of the file
        file_size: Size of file in bytes
        audio_bytes: Raw audio file bytes
        content_type: MIME type of the file (optional)

    Returns:
        Tuple of (duration in seconds, preprocessed audio bytes)

    Raises:
        AudioValidationError: If validation or preprocessing fails
    """
    # Validate file size
    validate_file_size(file_size)

    # Validate file format
    validate_file_format(filename, content_type)

    # Validate duration and get it
    duration = validate_audio_duration(audio_bytes)

    # Preprocess audio
    preprocessed_bytes = preprocess_audio(audio_bytes)

    return duration, preprocessed_bytes
