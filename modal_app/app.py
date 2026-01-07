"""Modal app with Kyutai STT transcription service."""

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
stt_image = (
    modal.Image.debian_slim(python_version="3.11")
    # Install system dependencies for audio processing
    .apt_install(
        "ffmpeg",
        "libsndfile1",
    )
    # Install Python packages for ML and audio
    .pip_install(
        "torch>=2.1.0",
        "torchaudio>=2.1.0",
        "transformers>=4.53.0",
        "accelerate>=0.30.0",
        "ffmpeg-python>=0.2.0",
        "numpy>=1.24.0",
        "soundfile>=0.12.0",
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
    "image": stt_image,
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
    print(f"CPU Memory Snapshots: ENABLED")

# Add GPU memory snapshot if enabled (requires CPU snapshots)
if config.ENABLE_GPU_MEMORY_SNAPSHOT:
    if not config.ENABLE_CPU_MEMORY_SNAPSHOT:
        raise ValueError(
            "ENABLE_GPU_MEMORY_SNAPSHOT requires ENABLE_CPU_MEMORY_SNAPSHOT to be True"
        )
    decorator_kwargs["experimental_options"] = {"enable_gpu_snapshot": True}
    print(f"GPU Memory Snapshots: ENABLED (Experimental)")


@app.cls(**decorator_kwargs)
class KyutaiSTTModel:
    """Kyutai STT model class for GPU-accelerated transcription."""

    @modal.enter(snap=config.ENABLE_CPU_MEMORY_SNAPSHOT or config.ENABLE_GPU_MEMORY_SNAPSHOT)
    def load_model(self):
        """Load Kyutai STT model once per container (runs on container startup)."""
        import torch
        import time
        from transformers import KyutaiSpeechToTextProcessor, KyutaiSpeechToTextForConditionalGeneration

        start_time = time.time()
        print(f"Loading Kyutai STT model: {config.STT_MODEL_ID}...")

        # Set cache directory to Modal volume
        cache_dir = "/models/huggingface"

        # Load processor and model
        self.processor = KyutaiSpeechToTextProcessor.from_pretrained(
            config.STT_MODEL_ID,
            cache_dir=cache_dir,
        )

        self.model = KyutaiSpeechToTextForConditionalGeneration.from_pretrained(
            config.STT_MODEL_ID,
            cache_dir=cache_dir,
            device_map=config.STT_DEVICE,
            torch_dtype=config.STT_DTYPE,
        )

        load_time = time.time() - start_time
        print(f"Model loaded successfully on {config.STT_DEVICE} in {load_time:.2f}s")

        # Optional: Warm up model with dummy forward pass
        if config.ENABLE_MODEL_WARMUP:
            print("Warming up model with dummy forward pass...")
            warmup_start = time.time()

            import numpy as np
            # Create 1 second of silence at 24kHz (Kyutai's native sample rate)
            dummy_audio = np.zeros(config.SAMPLE_RATE, dtype=np.float32)

            # Run transcription to compile kernels
            inputs = self.processor(
                dummy_audio,
                sampling_rate=config.SAMPLE_RATE,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            _ = self.model.generate(**inputs)

            warmup_time = time.time() - warmup_start
            print(f"Model warm-up completed in {warmup_time:.2f}s")

        total_time = time.time() - start_time
        print(f"Total initialization time: {total_time:.2f}s")

        # Print optimization status
        if config.ENABLE_CPU_MEMORY_SNAPSHOT:
            print("-> This state will be captured in CPU memory snapshot")
        if config.ENABLE_GPU_MEMORY_SNAPSHOT:
            print("-> GPU state (including loaded model) will be captured in snapshot")

    def _load_audio(self, audio_bytes: bytes) -> tuple:
        """Load audio from bytes and return numpy array with sample rate."""
        import soundfile as sf
        import numpy as np
        import io

        # Read audio file
        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))

        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        # Ensure float32
        audio_data = audio_data.astype(np.float32)

        return audio_data, sample_rate

    @modal.method()
    def transcribe(self, audio_bytes: bytes) -> Dict[str, Any]:
        """
        Transcribe audio file and return complete results.

        Args:
            audio_bytes: Raw audio file bytes (should be 24kHz WAV)

        Returns:
            Dictionary with transcription results
        """
        import torch

        # Load audio
        audio_data, sample_rate = self._load_audio(audio_bytes)

        # Calculate duration from audio length
        duration = len(audio_data) / sample_rate

        # Process audio through processor (single audio - no return_tensors needed)
        inputs = self.processor(audio_data)
        inputs = inputs.to(self.model.device)

        # Generate transcription
        with torch.no_grad():
            output_tokens = self.model.generate(**inputs)

        # Decode output
        transcription = self.processor.batch_decode(output_tokens, skip_special_tokens=True)
        text = transcription[0] if transcription else ""

        return {
            "text": text,
            "language": "en",  # Kyutai stt-2.6b-en is English only
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": duration,
                    "text": text,
                }
            ],
        }

    @modal.method()
    def transcribe_stream(self, audio_bytes: bytes, actual_duration: float = 0.0) -> Iterator[str]:
        """
        Transcribe audio file and yield segments as they complete.

        Args:
            audio_bytes: Raw audio file bytes (should be 24kHz WAV)
            actual_duration: Actual duration of audio in seconds (from ffmpeg)

        Yields:
            JSON strings with segment data
        """
        import torch

        try:
            # Load audio
            audio_data, sample_rate = self._load_audio(audio_bytes)

            # Calculate duration
            duration = actual_duration if actual_duration > 0 else len(audio_data) / sample_rate

            # Yield metadata first
            yield json.dumps({
                "type": "metadata",
                "language": "en",  # Kyutai stt-2.6b-en is English only
                "duration": duration,
            })

            # Process audio through processor
            inputs = self.processor(
                audio_data,
                sampling_rate=sample_rate,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate transcription
            with torch.no_grad():
                output_tokens = self.model.generate(**inputs)

            # Decode output
            transcription = self.processor.batch_decode(output_tokens, skip_special_tokens=True)
            text = transcription[0] if transcription else ""

            # Yield single segment (Kyutai doesn't provide segment-level timestamps by default)
            yield json.dumps({
                "type": "segment",
                "id": 0,
                "start": 0.0,
                "end": duration,
                "text": text,
            })

            # Yield final complete text
            yield json.dumps({
                "type": "complete",
                "text": text,
            })

        except Exception as e:
            # Yield error
            yield json.dumps({
                "type": "error",
                "error": str(e),
            })


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
    model = KyutaiSTTModel()
    for chunk in model.transcribe_stream.remote(audio_bytes):
        data = json.loads(chunk)
        if data["type"] == "segment":
            print(f"[{data['start']:.2f}s - {data['end']:.2f}s] {data['text']}")
        elif data["type"] == "complete":
            print("-" * 50)
            print(f"Complete: {data['text']}")
        elif data["type"] == "error":
            print(f"Error: {data['error']}")
