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
    import ffmpeg
    import tempfile
    import os

    # Write to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".audio") as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_path = tmp_file.name

    wav_path = None
    try:
        # First, try to get duration using ffprobe directly
        probe_succeeded = False
        try:
            probe = ffmpeg.probe(tmp_path)
            probe_succeeded = True

            # Try to get duration from format first
            if 'format' in probe and 'duration' in probe['format']:
                return float(probe['format']['duration'])

            # Fallback: try to get duration from streams
            for stream in probe.get('streams', []):
                if 'duration' in stream:
                    return float(stream['duration'])

            # Last resort: calculate from frames/sample rate for audio streams
            for stream in probe.get('streams', []):
                if stream.get('codec_type') == 'audio':
                    # Try to calculate from nb_frames and time_base or sample rate
                    if 'nb_frames' in stream and 'sample_rate' in stream:
                        nb_frames = int(stream['nb_frames'])
                        sample_rate = int(stream['sample_rate'])
                        if sample_rate > 0:
                            return nb_frames / sample_rate
        except Exception as probe_error:
            # ffprobe failed - this is common with browser-recorded WebM
            # We'll try the WAV conversion fallback below
            print(f"ffprobe failed (will try WAV conversion): {probe_error}")

        # Fallback: convert to WAV and get duration from that
        # This handles webm files from browser recording that lack duration metadata
        # or have malformed headers that cause ffprobe to fail
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav_file:
            wav_path = wav_file.name

        (
            ffmpeg
            .input(tmp_path)
            .output(wav_path, acodec='pcm_s16le', ac=1, ar=16000)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True, quiet=True)
        )

        wav_probe = ffmpeg.probe(wav_path)
        if 'format' in wav_probe and 'duration' in wav_probe['format']:
            return float(wav_probe['format']['duration'])

        raise AudioValidationError("Could not determine audio duration from file metadata")

    except AudioValidationError:
        raise
    except Exception as e:
        # Extract stderr from ffmpeg.Error if available
        error_msg = str(e)
        if hasattr(e, 'stderr') and e.stderr:
            stderr = e.stderr.decode('utf-8', errors='ignore') if isinstance(e.stderr, bytes) else str(e.stderr)
            error_msg = f"{error_msg}\nFFmpeg stderr: {stderr}"
        raise AudioValidationError(f"Failed to read audio file: {error_msg}")
    finally:
        # Cleanup
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)


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


def preprocess_audio(audio_bytes: bytes, sample_rate: Optional[int] = None) -> bytes:
    """
    Preprocess audio file for optimal transcription/TTS.
    - Convert to WAV format
    - Resample to target sample rate
    - Convert to mono

    Args:
        audio_bytes: Raw audio file bytes
        sample_rate: Target sample rate (default: config.SAMPLE_RATE)

    Returns:
        Preprocessed audio bytes in WAV format

    Raises:
        AudioValidationError: If preprocessing fails
    """
    try:
        import ffmpeg
        import tempfile
        import os

        target_rate = sample_rate if sample_rate else config.SAMPLE_RATE

        # Write input to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".audio") as tmp_file:
            tmp_file.write(audio_bytes)
            input_path = tmp_file.name

        # Create output temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            output_path = tmp_file.name

        try:
            # Use ffmpeg to convert: mono, target sample rate, WAV format
            (
                ffmpeg
                .input(input_path)
                .output(output_path, acodec='pcm_s16le', ac=1, ar=target_rate)
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
    target_sample_rate: Optional[int] = None,
) -> Tuple[float, bytes]:
    """
    Complete validation and preprocessing of audio file.

    Args:
        filename: Name of the file
        file_size: Size of file in bytes
        audio_bytes: Raw audio file bytes
        content_type: MIME type of the file (optional)
        target_sample_rate: Target sample rate for preprocessing (optional)

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
    preprocessed_bytes = preprocess_audio(audio_bytes, target_sample_rate)

    return duration, preprocessed_bytes
