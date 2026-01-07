"""Modal app with NVIDIA Parakeet TDT transcription service."""

import io
import json
import os
import sys
from typing import Iterator, Dict, Any
from pathlib import Path

import modal

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Create the container image with all required dependencies
stt_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python="3.12"
    )
    .env({
        "HF_XET_HIGH_PERFORMANCE": "1",
        "HF_HOME": "/models",  # Use Modal volume for HuggingFace cache
        "DEBIAN_FRONTEND": "noninteractive",
        "CXX": "g++",
        "CC": "g++",
    })
    .apt_install("ffmpeg")
    .uv_pip_install(
        "hf_transfer==0.1.9",
        "huggingface-hub==0.36.0",
        "nemo_toolkit[asr]==2.3.0",
        "cuda-python==12.8.0",
        "numpy<2",
        "pydub==0.25.1",
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


class NoStdStreams:
    """Context manager to suppress NeMo's verbose stdout/stderr."""

    def __init__(self):
        self.devnull = open(os.devnull, "w")

    def __enter__(self):
        self._stdout, self._stderr = sys.stdout, sys.stderr
        self._stdout.flush()
        self._stderr.flush()
        sys.stdout, sys.stderr = self.devnull, self.devnull
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout, sys.stderr = self._stdout, self._stderr
        self.devnull.close()


@app.cls(**decorator_kwargs)
class ParakeetSTTModel:
    """NVIDIA Parakeet TDT model for streaming GPU-accelerated transcription."""

    @modal.enter(snap=config.ENABLE_CPU_MEMORY_SNAPSHOT or config.ENABLE_GPU_MEMORY_SNAPSHOT)
    def load_model(self):
        """Load Parakeet TDT model once per container (runs on container startup)."""
        import logging
        import nemo.collections.asr as nemo_asr
        import time

        start_time = time.time()

        # Suppress NeMo's verbose logging
        logging.getLogger("nemo_logger").setLevel(logging.CRITICAL)

        print(f"Loading Parakeet TDT model: {config.STT_MODEL_ID}...")

        # NeMo uses HuggingFace Hub internally, respects HF_HOME env var
        self.model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=config.STT_MODEL_ID
        )

        load_time = time.time() - start_time
        print(f"Model loaded successfully in {load_time:.2f}s")

        # Optional: Warm up model with dummy forward pass
        if config.ENABLE_MODEL_WARMUP:
            self._warmup_model()

        total_time = time.time() - start_time
        print(f"Total initialization time: {total_time:.2f}s")

        # Print optimization status
        if config.ENABLE_CPU_MEMORY_SNAPSHOT:
            print("-> This state will be captured in CPU memory snapshot")
        if config.ENABLE_GPU_MEMORY_SNAPSHOT:
            print("-> GPU state (including loaded model) will be captured in snapshot")

    def _warmup_model(self):
        """Warmup model with dummy audio to compile CUDA kernels."""
        import numpy as np
        import time

        print("Warming up model with dummy forward pass...")
        warmup_start = time.time()

        # Create 1 second of silence at 16kHz (Parakeet's native sample rate)
        dummy_audio = np.zeros(config.SAMPLE_RATE, dtype=np.float32)

        # Run transcription to compile kernels (suppress output)
        with NoStdStreams():
            self.model.transcribe([dummy_audio])

        warmup_time = time.time() - warmup_start
        print(f"Model warm-up completed in {warmup_time:.2f}s")

    @modal.method()
    def transcribe(self, audio_bytes: bytes) -> Dict[str, Any]:
        """
        Transcribe complete audio file (non-streaming).

        Args:
            audio_bytes: Raw audio bytes (16-bit PCM, 16kHz, mono WAV)

        Returns:
            Dict with transcription results
        """
        import numpy as np

        # Convert bytes to numpy array (int16 → float32)
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        duration = len(audio_data) / config.SAMPLE_RATE

        print(f"Transcribing audio: {len(audio_data)} samples ({duration:.2f}s)")

        # Transcribe with NeMo (suppress verbose logs)
        with NoStdStreams():
            output = self.model.transcribe([audio_data])

        # Extract text from NeMo output
        text = output[0].text if output and hasattr(output[0], 'text') else ""

        return {
            "text": text,
            "language": "en",  # Parakeet TDT 0.6B v3 is English-only
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
        Transcribe audio with real progressive streaming using silence detection.

        Args:
            audio_bytes: Raw audio bytes (16-bit PCM, 16kHz, mono WAV)
            actual_duration: Actual duration from FFmpeg

        Yields:
            JSON strings: metadata → segment(s) → complete
        """
        import numpy as np
        from pydub import AudioSegment, silence

        try:
            # Calculate duration
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            duration = actual_duration if actual_duration > 0 else len(audio_array) / config.SAMPLE_RATE

            # Yield metadata first
            yield json.dumps({
                "type": "metadata",
                "language": "en",
                "duration": duration,
            })

            # Create AudioSegment for silence detection
            audio_segment = AudioSegment(
                data=audio_bytes,
                channels=1,
                sample_width=2,  # 16-bit
                frame_rate=config.SAMPLE_RATE,
            )

            # Detect silent windows
            silent_windows = silence.detect_silence(
                audio_segment,
                min_silence_len=config.SILENCE_MIN_LENGTH_MS,
                silence_thresh=config.SILENCE_THRESHOLD_DB,
            )

            print(f"Detected {len(silent_windows)} silent windows")

            # If no silence detected, transcribe entire audio at once
            if not silent_windows:
                audio_float = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                with NoStdStreams():
                    output = self.model.transcribe([audio_float])
                text = output[0].text if output and hasattr(output[0], 'text') else ""

                yield json.dumps({
                    "type": "segment",
                    "id": 0,
                    "start": 0.0,
                    "end": duration,
                    "text": text,
                })
            else:
                # Transcribe segments progressively based on silence boundaries
                segment_id = 0
                current_pos = 0

                for window_start, window_end in silent_windows:
                    # Extract segment up to end of silence
                    segment_audio = audio_segment[current_pos:window_end]

                    # Skip very short segments (< 100ms)
                    if len(segment_audio) < 100:
                        continue

                    # Transcribe segment
                    segment_float = np.frombuffer(
                        segment_audio.raw_data,
                        dtype=np.int16
                    ).astype(np.float32)

                    with NoStdStreams():
                        output = self.model.transcribe([segment_float])

                    text = output[0].text if output and hasattr(output[0], 'text') else ""

                    # Only yield if there's actual text
                    if text.strip():
                        start_time = current_pos / 1000.0  # ms → seconds
                        end_time = window_end / 1000.0

                        yield json.dumps({
                            "type": "segment",
                            "id": segment_id,
                            "start": start_time,
                            "end": end_time,
                            "text": text,
                        })

                        segment_id += 1

                    current_pos = window_end

                # Process any remaining audio after the last silence
                if current_pos < len(audio_segment):
                    remaining_audio = audio_segment[current_pos:]
                    remaining_float = np.frombuffer(
                        remaining_audio.raw_data,
                        dtype=np.int16
                    ).astype(np.float32)

                    with NoStdStreams():
                        output = self.model.transcribe([remaining_float])

                    text = output[0].text if output and hasattr(output[0], 'text') else ""

                    if text.strip():
                        yield json.dumps({
                            "type": "segment",
                            "id": segment_id,
                            "start": current_pos / 1000.0,
                            "end": duration,
                            "text": text,
                        })

            # Yield completion
            yield json.dumps({
                "type": "complete",
                "text": "Transcription complete",
            })

        except Exception as e:
            import traceback
            print(f"Error: {e}")
            print(traceback.format_exc())
            yield json.dumps({
                "type": "error",
                "error": str(e),
            })


@app.local_entrypoint()
def main():
    """Test Parakeet TDT model locally."""
    if len(sys.argv) < 2:
        print("Usage: modal run modal_app/app.py <audio_file_path>")
        sys.exit(1)

    audio_path = sys.argv[1]

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    print(f"Transcribing {audio_path} with streaming...")
    print("-" * 60)

    model = ParakeetSTTModel()
    segments = []

    for chunk in model.transcribe_stream.remote(audio_bytes):
        data = json.loads(chunk)
        if data["type"] == "metadata":
            print(f"Duration: {data['duration']:.2f}s | Language: {data['language']}")
            print("-" * 60)
        elif data["type"] == "segment":
            print(f"[{data['start']:.2f}s - {data['end']:.2f}s] {data['text']}")
            segments.append(data['text'])
        elif data["type"] == "complete":
            print("-" * 60)
            print(f"Complete transcription:\n{' '.join(segments)}")
        elif data["type"] == "error":
            print(f"ERROR: {data['error']}")
