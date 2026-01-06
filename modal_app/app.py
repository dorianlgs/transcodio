"""Modal app with Whisper transcription service."""

import io
import json
from typing import Iterator, Dict, Any

import modal
import sys
from pathlib import Path

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Create the container image with all required dependencies
whisper_image = (
    modal.Image.debian_slim(python_version="3.11")
    # Install system dependencies for audio processing
    .apt_install(
        "ffmpeg",
        "libsndfile1",
    )
    # Install Python packages for ML and audio
    .pip_install(
        "openai-whisper>=20231117",
        "torch>=2.1.0",
        "torchaudio>=2.1.0",
        "ffmpeg-python>=0.2.0",
        "numpy>=1.24.0",
    )
    # Add config file to the image
    .add_local_file(
        str(Path(__file__).parent.parent / "config.py"),
        "/root/config.py"
    )
)

# Create Modal app
app = modal.App(config.MODAL_APP_NAME)

# Create Modal Volume for persistent model storage
volume = modal.Volume.from_name(config.MODAL_VOLUME_NAME, create_if_missing=True)

# Build decorator arguments based on optimization flags
decorator_kwargs = {
    "image": whisper_image,
    "gpu": config.MODAL_GPU_TYPE,
    "scaledown_window": (
        config.EXTENDED_IDLE_TIMEOUT_SECONDS
        if config.EXTENDED_IDLE_TIMEOUT
        else config.MODAL_CONTAINER_IDLE_TIMEOUT
    ),
    "timeout": config.MODAL_TIMEOUT,
    "memory": config.MODAL_MEMORY_MB,
    "volumes": {"/models": volume},
}

# Print configuration summary
print(f"GPU Type: {config.MODAL_GPU_TYPE}")
print(f"Memory: {config.MODAL_MEMORY_MB}MB")
print(f"Container idle timeout: {decorator_kwargs['scaledown_window']}s")

# Add CPU memory snapshot if enabled
if config.ENABLE_CPU_MEMORY_SNAPSHOT:
    decorator_kwargs["enable_memory_snapshot"] = True
    print(f"✓ CPU Memory Snapshots: ENABLED")

# Add GPU memory snapshot if enabled (requires CPU snapshots)
if config.ENABLE_GPU_MEMORY_SNAPSHOT:
    if not config.ENABLE_CPU_MEMORY_SNAPSHOT:
        raise ValueError(
            "ENABLE_GPU_MEMORY_SNAPSHOT requires ENABLE_CPU_MEMORY_SNAPSHOT to be True"
        )
    decorator_kwargs["experimental_options"] = {"enable_gpu_snapshot": True}
    print(f"✓ GPU Memory Snapshots: ENABLED (Experimental)")

@app.cls(**decorator_kwargs)
class WhisperModel:
    """Whisper model class for GPU-accelerated transcription."""

    @modal.enter(snap=config.ENABLE_CPU_MEMORY_SNAPSHOT or config.ENABLE_GPU_MEMORY_SNAPSHOT)
    def load_model(self):
        """Load Whisper model once per container (runs on container startup)."""
        import whisper
        import torch
        import time

        start_time = time.time()
        print(f"Loading Whisper {config.WHISPER_MODEL} model...")

        # Load model with GPU acceleration
        self.model = whisper.load_model(
            config.WHISPER_MODEL,
            device=config.WHISPER_DEVICE,
            download_root="/models",
        )

        load_time = time.time() - start_time
        print(f"Model loaded successfully on {config.WHISPER_DEVICE} in {load_time:.2f}s")

        # Optional: Warm up model with dummy forward pass
        # This compiles CUDA kernels which will be captured in GPU snapshots
        if config.ENABLE_MODEL_WARMUP:
            print("Warming up model with dummy forward pass...")
            warmup_start = time.time()

            import numpy as np
            # Create 1 second of silence at 16kHz (Whisper's native sample rate)
            dummy_audio = np.zeros(16000, dtype=np.float32)

            # Run transcription to compile kernels
            _ = self.model.transcribe(
                dummy_audio,
                fp16=config.WHISPER_FP16,
                verbose=False,
            )

            warmup_time = time.time() - warmup_start
            print(f"Model warm-up completed in {warmup_time:.2f}s")

        total_time = time.time() - start_time
        print(f"Total initialization time: {total_time:.2f}s")

        # Print optimization status
        if config.ENABLE_CPU_MEMORY_SNAPSHOT:
            print("→ This state will be captured in CPU memory snapshot")
        if config.ENABLE_GPU_MEMORY_SNAPSHOT:
            print("→ GPU state (including loaded model) will be captured in snapshot")
        if config.ENABLE_MODEL_WARMUP and config.ENABLE_GPU_MEMORY_SNAPSHOT:
            print("→ Compiled CUDA kernels will be captured in GPU snapshot")

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
    def transcribe_stream(self, audio_bytes: bytes, actual_duration: float = 0.0) -> Iterator[str]:
        """
        Transcribe audio file and yield segments as they complete.

        Args:
            audio_bytes: Raw audio file bytes
            actual_duration: Actual duration of audio in seconds (from ffmpeg)

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
            # Use actual_duration if provided, otherwise fallback to Whisper's last segment end time
            duration = actual_duration if actual_duration > 0 else (
                result.get("segments", [{}])[-1].get("end", 0) if result.get("segments") else 0
            )
            yield json.dumps({
                "type": "metadata",
                "language": result.get("language", "unknown"),
                "duration": duration,
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
