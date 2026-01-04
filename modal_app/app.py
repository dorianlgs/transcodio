"""Modal app with Whisper transcription service."""

import io
import json
from typing import Iterator, Dict, Any

import modal

from .image import whisper_image
import sys
from pathlib import Path

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Create Modal app
app = modal.App(config.MODAL_APP_NAME)

# Create Modal Volume for persistent model storage
volume = modal.Volume.from_name(config.MODAL_VOLUME_NAME, create_if_missing=True)


@app.cls(
    image=whisper_image,
    gpu=modal.gpu.L4(count=config.MODAL_GPU_COUNT),
    container_idle_timeout=config.MODAL_CONTAINER_IDLE_TIMEOUT,
    timeout=config.MODAL_TIMEOUT,
    memory=config.MODAL_MEMORY_MB,
    volumes={"/models": volume},
)
class WhisperModel:
    """Whisper model class for GPU-accelerated transcription."""

    @modal.enter()
    def load_model(self):
        """Load Whisper model once per container (runs on container startup)."""
        import whisper
        import torch

        print(f"Loading Whisper {config.WHISPER_MODEL} model...")

        # Load model with GPU acceleration
        self.model = whisper.load_model(
            config.WHISPER_MODEL,
            device=config.WHISPER_DEVICE,
            download_root="/models",
        )

        # Enable FP16 for faster inference if configured
        if config.WHISPER_FP16 and torch.cuda.is_available():
            self.model = self.model.half()

        print(f"Model loaded successfully on {config.WHISPER_DEVICE}")

    @modal.method()
    def transcribe(self, audio_bytes: bytes) -> Dict[str, Any]:
        """
        Transcribe audio file and return complete results.

        Args:
            audio_bytes: Raw audio file bytes

        Returns:
            Dictionary with transcription results
        """
        import whisper
        import tempfile
        import os

        # Write bytes to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".audio") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name

        try:
            # Transcribe audio
            result = self.model.transcribe(
                tmp_path,
                fp16=config.WHISPER_FP16,
                verbose=False,
            )

            return {
                "text": result["text"],
                "language": result.get("language", "unknown"),
                "segments": [
                    {
                        "id": seg["id"],
                        "start": seg["start"],
                        "end": seg["end"],
                        "text": seg["text"],
                    }
                    for seg in result.get("segments", [])
                ],
            }
        finally:
            # Cleanup temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @modal.method()
    def transcribe_stream(self, audio_bytes: bytes) -> Iterator[str]:
        """
        Transcribe audio file and yield segments as they complete.

        Args:
            audio_bytes: Raw audio file bytes

        Yields:
            JSON strings with segment data
        """
        import whisper
        import tempfile
        import os

        # Write bytes to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".audio") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name

        try:
            # Transcribe audio
            result = self.model.transcribe(
                tmp_path,
                fp16=config.WHISPER_FP16,
                verbose=False,
            )

            # Yield metadata first
            yield json.dumps({
                "type": "metadata",
                "language": result.get("language", "unknown"),
                "duration": result.get("segments", [{}])[-1].get("end", 0) if result.get("segments") else 0,
            })

            # Yield each segment as it's processed
            for segment in result.get("segments", []):
                yield json.dumps({
                    "type": "segment",
                    "id": segment["id"],
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"],
                })

            # Yield final complete text
            yield json.dumps({
                "type": "complete",
                "text": result["text"],
            })

        except Exception as e:
            # Yield error
            yield json.dumps({
                "type": "error",
                "error": str(e),
            })
        finally:
            # Cleanup temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


@app.local_entrypoint()
def main():
    """Local entrypoint for testing the Modal app."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: modal run modal_app/app.py <audio_file_path>")
        sys.exit(1)

    audio_path = sys.argv[1]

    # Read audio file
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    print(f"Transcribing {audio_path}...")
    print("-" * 50)

    # Test streaming transcription
    model = WhisperModel()
    for chunk in model.transcribe_stream.remote(audio_bytes):
        data = json.loads(chunk)
        if data["type"] == "segment":
            print(f"[{data['start']:.2f}s - {data['end']:.2f}s] {data['text']}")
        elif data["type"] == "complete":
            print("-" * 50)
            print(f"Complete: {data['text']}")
        elif data["type"] == "error":
            print(f"Error: {data['error']}")
