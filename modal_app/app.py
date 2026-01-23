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
        "scikit-learn>=1.3.0",  # For spectral clustering
        "soundfile>=0.12.1",    # For audio file I/O
    )
    # Add config file to the image
    .add_local_file(
        str(Path(__file__).parent.parent / "config.py"),
        "/root/config.py"
    )
)

# Anthropic API image for meeting minutes generation (no GPU needed)
anthropic_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("anthropic>=0.40.0")
    .add_local_file(
        str(Path(__file__).parent.parent / "config.py"),
        "/root/config.py"
    )
)

# TTS image for Qwen3-TTS voice cloning
qwen_tts_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python="3.12"
    )
    .env({
        "HF_HOME": "/models",
        "DEBIAN_FRONTEND": "noninteractive",
    })
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "qwen-tts",
        "torch",
        "transformers",
        "soundfile>=0.12.1",
    )
    .add_local_file(
        str(Path(__file__).parent.parent / "config.py"),
        "/root/config.py"
    )
)

# TTS image for Higgs Audio V2 voice cloning (using Replicate's working fork)
higgs_tts_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python="3.11"
    )
    .env({
        "HF_HOME": "/models",
        "DEBIAN_FRONTEND": "noninteractive",
    })
    .apt_install("ffmpeg", "libsndfile1", "git")
    .run_commands(
        # Clone the Replicate fork which has boson_multimodal bundled correctly
        "git clone https://github.com/lucataco/cog-higgs-audio.git /opt/higgs-audio",
    )
    .pip_install(
        # Install dependencies from the working Replicate implementation
        "torch",
        "torchaudio",
        "torchvision",
        "transformers>=4.45.1,<4.47.0",
        "descript-audio-codec",
        "librosa",
        "dacite",
        "pydantic",
        "vector_quantize_pytorch",
        "loguru",
        "pydub",
        "omegaconf",
        "langid",
        "jieba",
        "json_repair",
        "accelerate>=0.26.0",
        "soundfile>=0.12.1",
        "pandas",
    )
    .run_commands(
        # Install the package in editable mode
        "cd /opt/higgs-audio && pip install -e .",
    )
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


def align_speakers_to_segments(transcription_segments: list, speaker_timeline: list) -> list:
    """
    Assign speaker labels to transcription segments based on maximum overlap.

    Args:
        transcription_segments: List of dicts with {id, start, end, text}
        speaker_timeline: List of dicts with {start, end, speaker}

    Returns:
        List of segments with speaker field added
    """
    def calculate_overlap(range1, range2):
        """Calculate overlap duration between two time ranges."""
        start1, end1 = range1
        start2, end2 = range2
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        return max(0, overlap_end - overlap_start)

    for segment in transcription_segments:
        max_overlap = 0
        assigned_speaker = None

        for speaker_seg in speaker_timeline:
            overlap = calculate_overlap(
                (segment["start"], segment["end"]),
                (speaker_seg["start"], speaker_seg["end"])
            )

            if overlap > max_overlap:
                max_overlap = overlap
                assigned_speaker = speaker_seg["speaker"]

        # Assign speaker label (convert to "Speaker 1", "Speaker 2", etc.)
        if assigned_speaker is not None:
            segment["speaker"] = f"Speaker {assigned_speaker + 1}"
        else:
            segment["speaker"] = "Speaker 1"  # Fallback for no overlap

    return transcription_segments


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

            # Accumulate all segment texts for final transcription
            all_segments = []

            # If no silence detected, transcribe entire audio at once
            if not silent_windows:
                audio_float = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                with NoStdStreams():
                    output = self.model.transcribe([audio_float])
                text = output[0].text if output and hasattr(output[0], 'text') else ""

                if text.strip():
                    all_segments.append(text)

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

                        all_segments.append(text)

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
                        all_segments.append(text)

                        yield json.dumps({
                            "type": "segment",
                            "id": segment_id,
                            "start": current_pos / 1000.0,
                            "end": duration,
                            "text": text,
                        })

            # Yield completion with full transcription
            full_transcription = " ".join(all_segments)
            yield json.dumps({
                "type": "complete",
                "text": full_transcription,
            })

        except Exception as e:
            import traceback
            print(f"Error: {e}")
            print(traceback.format_exc())
            yield json.dumps({
                "type": "error",
                "error": str(e),
            })


@app.cls(**decorator_kwargs)
class SpeakerDiarizerModel:
    """NVIDIA TitaNet + clustering for speaker diarization."""

    @modal.enter(snap=config.ENABLE_CPU_MEMORY_SNAPSHOT or config.ENABLE_GPU_MEMORY_SNAPSHOT)
    def load_model(self):
        """Load TitaNet speaker embedding model."""
        import logging
        import nemo.collections.asr as nemo_asr
        import time

        start_time = time.time()
        logging.getLogger("nemo_logger").setLevel(logging.CRITICAL)

        print(f"Loading TitaNet model: {config.DIARIZATION_MODEL}...")

        self.embedding_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            model_name=config.DIARIZATION_MODEL
        )

        load_time = time.time() - start_time
        print(f"TitaNet model loaded in {load_time:.2f}s")

    @modal.method()
    def diarize(self, audio_bytes: bytes, duration: float) -> list:
        """
        Perform speaker diarization on audio.

        Args:
            audio_bytes: Raw audio bytes (16-bit PCM, 16kHz, mono WAV)
            duration: Audio duration in seconds

        Returns:
            List of speaker segments: [{"start": 0.0, "end": 5.2, "speaker": 0}, ...]
        """
        import numpy as np
        from sklearn.cluster import SpectralClustering
        import tempfile
        import soundfile as sf

        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            # Save to temporary WAV file (NeMo requires file input)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, audio_array, config.SAMPLE_RATE)
                audio_path = tmp.name

            # Extract single-scale embeddings (fixes multi-scale artifact issue)
            embeddings_list = []
            timestamps_list = []

            # Use only the first (longest) window length to avoid multi-scale artifacts
            window_length = config.DIARIZATION_WINDOW_LENGTHS[0]
            window_samples = int(window_length * config.SAMPLE_RATE)
            shift_samples = int(config.DIARIZATION_SHIFT_LENGTH * config.SAMPLE_RATE)

            num_windows = max(1, (len(audio_array) - window_samples) // shift_samples + 1)

            for i in range(num_windows):
                start_sample = i * shift_samples
                end_sample = min(start_sample + window_samples, len(audio_array))

                window_audio = audio_array[start_sample:end_sample]

                # Skip windows that are too short
                if len(window_audio) < window_samples * 0.5:
                    continue

                # Save window to temp file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_win:
                    sf.write(tmp_win.name, window_audio, config.SAMPLE_RATE)

                    # Get embedding
                    with NoStdStreams():
                        embedding = self.embedding_model.get_embedding(tmp_win.name)

                    embeddings_list.append(embedding.cpu().numpy().flatten())
                    timestamps_list.append({
                        "start": start_sample / config.SAMPLE_RATE,
                        "end": end_sample / config.SAMPLE_RATE
                    })

                    import os
                    os.unlink(tmp_win.name)

            # Cluster embeddings to identify speakers
            embeddings_matrix = np.array(embeddings_list)

            # Normalize embeddings for cosine similarity
            from sklearn.preprocessing import normalize
            embeddings_matrix = normalize(embeddings_matrix)

            # Determine optimal number of speakers using multiple metrics with complexity penalty
            from sklearn.metrics import silhouette_score, calinski_harabasz_score
            from sklearn.cluster import AgglomerativeClustering

            # Try different numbers of speakers and pick the best
            max_speakers_to_try = min(config.DIARIZATION_MAX_SPEAKERS, len(embeddings_matrix) // 5)
            min_speakers_to_try = config.DIARIZATION_MIN_SPEAKERS

            best_score = -np.inf
            best_n_speakers = 1
            best_labels = None

            # Only try clustering if we have enough segments
            if len(embeddings_matrix) >= 10 and max_speakers_to_try >= 2:
                for n in range(min_speakers_to_try, min(max_speakers_to_try + 1, 6)):  # Cap at 5 speakers
                    try:
                        # Use AgglomerativeClustering with cosine distance (better for speaker embeddings)
                        clustering = AgglomerativeClustering(
                            n_clusters=n,
                            metric='cosine',
                            linkage='average'
                        )
                        labels = clustering.fit_predict(embeddings_matrix)

                        # Calculate silhouette score (measures cluster quality)
                        silhouette = silhouette_score(embeddings_matrix, labels, metric='cosine')

                        # Calculate Calinski-Harabasz score (higher is better, measures cluster separation)
                        calinski = calinski_harabasz_score(embeddings_matrix, labels)

                        # Normalize calinski to 0-1 range (approximately)
                        calinski_norm = calinski / (calinski + 100)

                        # Apply complexity penalty: prefer fewer speakers (BIC-inspired)
                        # Penalty increases with number of speakers
                        complexity_penalty = 0.15 * (n - 1)  # Each additional speaker costs 0.15 points

                        # Combined score: weighted average of silhouette and calinski, minus penalty
                        combined_score = (0.6 * silhouette + 0.4 * calinski_norm) - complexity_penalty

                        print(f"Trying {n} speakers: silhouette={silhouette:.3f}, calinski_norm={calinski_norm:.3f}, penalty={complexity_penalty:.3f}, combined={combined_score:.3f}")

                        if combined_score > best_score:
                            best_score = combined_score
                            best_n_speakers = n
                            best_labels = labels
                    except Exception as e:
                        print(f"Failed to cluster with {n} speakers: {e}")
                        continue

                if best_labels is not None:
                    speaker_labels = best_labels
                    n_speakers = best_n_speakers
                    print(f"Selected {n_speakers} speakers with combined score {best_score:.3f}")
                else:
                    # Fallback: assume single speaker
                    n_speakers = 1
                    speaker_labels = np.zeros(len(embeddings_matrix), dtype=int)
            else:
                # Too few segments, assume single speaker
                n_speakers = 1
                speaker_labels = np.zeros(len(embeddings_matrix), dtype=int)

            # Create speaker timeline by merging consecutive windows with same speaker
            speaker_segments = []
            current_speaker = speaker_labels[0]
            current_start = timestamps_list[0]["start"]
            current_end = timestamps_list[0]["end"]

            for i in range(1, len(speaker_labels)):
                if speaker_labels[i] == current_speaker:
                    # Extend current segment
                    current_end = timestamps_list[i]["end"]
                else:
                    # Save previous segment and start new one
                    speaker_segments.append({
                        "start": current_start,
                        "end": current_end,
                        "speaker": int(current_speaker)
                    })
                    current_speaker = speaker_labels[i]
                    current_start = timestamps_list[i]["start"]
                    current_end = timestamps_list[i]["end"]

            # Add final segment
            speaker_segments.append({
                "start": current_start,
                "end": current_end,
                "speaker": int(current_speaker)
            })

            # Clean up temp file
            import os
            os.unlink(audio_path)

            print(f"Diarization complete: {len(speaker_segments)} speaker segments, {n_speakers} speakers")

            return speaker_segments

        except Exception as e:
            import traceback
            print(f"Diarization error: {e}")
            print(traceback.format_exc())
            return []


@app.cls(
    image=qwen_tts_image,
    gpu=config.TTS_MODELS["qwen"]["gpu_type"],
    scaledown_window=config.TTS_CONTAINER_IDLE_TIMEOUT,
    timeout=300,
    memory=config.TTS_MODELS["qwen"]["memory_mb"],
    volumes={"/models": volume},
)
class Qwen3TTSVoiceCloner:
    """Qwen3-TTS model for voice cloning."""

    @modal.enter()
    def load_model(self):
        """Load Qwen3-TTS model once per container."""
        import torch
        import time

        model_config = config.TTS_MODELS["qwen"]
        start_time = time.time()
        print(f"Loading Qwen3-TTS model: {model_config['model_id']}...")

        from qwen_tts import Qwen3TTSModel

        self.model = Qwen3TTSModel.from_pretrained(
            model_config["model_id"],
            device_map="cuda:0",
            dtype=torch.bfloat16,
        )

        load_time = time.time() - start_time
        print(f"Qwen3-TTS model loaded in {load_time:.2f}s")

    @modal.method()
    def generate_voice_clone(
        self,
        ref_audio_bytes: bytes,
        ref_text: str,
        target_text: str,
        language: str
    ) -> Dict[str, Any]:
        """
        Generate audio with cloned voice.

        Args:
            ref_audio_bytes: Reference audio bytes (WAV format)
            ref_text: Transcription of the reference audio
            target_text: Text to synthesize with cloned voice
            language: Target language

        Returns:
            Dict with audio_bytes and metadata
        """
        import soundfile as sf
        import tempfile
        import time
        import os

        try:
            start_time = time.time()

            # Save reference audio to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_ref:
                tmp_ref.write(ref_audio_bytes)
                ref_audio_path = tmp_ref.name

            print(f"Generating voice clone for {len(target_text)} chars in {language}...")

            # Generate audio with voice cloning
            wavs, sr = self.model.generate_voice_clone(
                text=target_text,
                language=language,
                ref_audio=ref_audio_path,
                ref_text=ref_text,
            )

            # Convert to bytes
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
                sf.write(tmp_out.name, wavs[0], sr)
                with open(tmp_out.name, "rb") as f:
                    audio_bytes = f.read()
                output_path = tmp_out.name

            # Cleanup temp files
            os.unlink(ref_audio_path)
            os.unlink(output_path)

            generation_time = time.time() - start_time
            duration = len(wavs[0]) / sr

            print(f"Voice clone generated in {generation_time:.2f}s, duration: {duration:.2f}s")

            return {
                "success": True,
                "audio_bytes": audio_bytes,
                "duration": duration,
                "sample_rate": sr,
            }

        except Exception as e:
            import traceback
            print(f"Voice clone error: {e}")
            print(traceback.format_exc())
            return {
                "success": False,
                "error": str(e),
            }


@app.cls(
    image=higgs_tts_image,
    gpu=config.TTS_MODELS["higgs"]["gpu_type"],
    scaledown_window=config.TTS_CONTAINER_IDLE_TIMEOUT,
    timeout=600,
    memory=config.TTS_MODELS["higgs"]["memory_mb"],
    volumes={"/models": volume},
)
class HiggsAudioVoiceCloner:
    """Higgs Audio V2 model for high-quality voice cloning."""

    @modal.enter()
    def load_model(self):
        """Load Higgs Audio V2 model once per container."""
        import sys
        import time
        import torch

        # Add the higgs-audio package to path
        sys.path.insert(0, "/opt/higgs-audio")

        start_time = time.time()
        model_config = config.TTS_MODELS["higgs"]
        print(f"Loading Higgs Audio V2 model: {model_config['model_id']}...")

        from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.engine = HiggsAudioServeEngine(
            model_config["model_id"],
            model_config["tokenizer_id"],
            device=device,
        )

        load_time = time.time() - start_time
        print(f"Higgs Audio V2 model loaded in {load_time:.2f}s")

    @modal.method()
    def generate_voice_clone(
        self,
        ref_audio_bytes: bytes,
        ref_text: str,
        target_text: str,
        language: str
    ) -> Dict[str, Any]:
        """
        Generate audio with cloned voice using Higgs Audio V2.

        Args:
            ref_audio_bytes: Reference audio bytes (WAV format)
            ref_text: Transcription of the reference audio
            target_text: Text to synthesize with cloned voice
            language: Target language

        Returns:
            Dict with audio_bytes and metadata
        """
        import sys
        sys.path.insert(0, "/opt/higgs-audio")

        import tempfile
        import time
        import os
        import torch
        import torchaudio

        try:
            start_time = time.time()

            from boson_multimodal.data_types import ChatMLSample, Message, AudioContent

            # Save reference audio to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_ref:
                tmp_ref.write(ref_audio_bytes)
                ref_audio_path = tmp_ref.name

            print(f"Generating voice clone for {len(target_text)} chars with Higgs Audio V2...")

            # Build system prompt
            system_prompt = (
                "Generate audio following instruction.\n\n<|scene_desc_start|>\n"
                "Audio is recorded from a quiet room.\n<|scene_desc_end|>"
            )

            # Load reference audio for voice cloning
            ref_audio_content = AudioContent(audio_path=ref_audio_path)

            # Build messages with reference audio
            messages = [
                Message(role="system", content=system_prompt),
                Message(
                    role="user",
                    content=[
                        "Clone the voice from this audio: ",
                        ref_audio_content,
                        f"\n\nReference text: {ref_text}\n\nNow generate: {target_text}",
                    ]
                ),
            ]

            # Generate audio
            output = self.engine.generate(
                chat_ml_sample=ChatMLSample(messages=messages),
                max_new_tokens=2048,
                temperature=0.3,
                top_p=0.95,
            )

            # Get audio and sample rate
            audio_array = output.audio
            sr = output.sampling_rate

            # Convert to bytes
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
                torchaudio.save(tmp_out.name, torch.from_numpy(audio_array)[None, :], sr)
                with open(tmp_out.name, "rb") as f:
                    audio_bytes = f.read()
                output_path = tmp_out.name

            # Cleanup temp files
            os.unlink(ref_audio_path)
            os.unlink(output_path)

            generation_time = time.time() - start_time
            duration = len(audio_array) / sr

            print(f"Higgs voice clone generated in {generation_time:.2f}s, duration: {duration:.2f}s")

            return {
                "success": True,
                "audio_bytes": audio_bytes,
                "duration": duration,
                "sample_rate": sr,
            }

        except Exception as e:
            import traceback
            print(f"Higgs voice clone error: {e}")
            print(traceback.format_exc())
            return {
                "success": False,
                "error": str(e),
            }


@app.cls(
    image=anthropic_image,
    secrets=[modal.Secret.from_name("anthropic-api-key")],
    scaledown_window=config.MINUTES_CONTAINER_IDLE_TIMEOUT,
    timeout=120,  # 2 minutes max for API call
)
class MeetingMinutesGenerator:
    """Meeting minutes generator using Anthropic Claude Haiku 4.5 API."""

    @modal.enter()
    def setup_client(self):
        """Initialize Anthropic client."""
        import anthropic
        import os

        self.client = anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"]
        )
        print(f"Anthropic client initialized for model: {config.ANTHROPIC_MODEL_ID}")

    @modal.method()
    def generate_minutes(self, transcription: str, speakers: list = None) -> Dict[str, Any]:
        """
        Generate meeting minutes from transcription using Claude Haiku 4.5.

        Args:
            transcription: Full transcription text
            speakers: Optional list of speaker-annotated segments

        Returns:
            Dict with structured meeting minutes
        """
        from datetime import datetime

        try:
            # Get current date for relative date calculations
            today = datetime.now()
            date_str = today.strftime("%d de %B de %Y")  # e.g., "21 de enero de 2025"
            weekday = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"][today.weekday()]

            # Build speaker context if available
            speaker_context = ""
            if speakers:
                unique_speakers = set(seg.get("speaker", "Speaker 1") for seg in speakers)
                speaker_context = f"\n\nParticipantes detectados: {', '.join(sorted(unique_speakers))}"

            # Build the prompt (Spanish)
            system_prompt = f"""Eres un experto generador de minutas de reunión. Analiza la transcripción y extrae la información clave en formato JSON estructurado.

FECHA DE HOY: {weekday}, {date_str}

Cuando en la transcripción se mencionen fechas relativas como "mañana", "pasado mañana", "la próxima semana", "el lunes", etc., DEBES calcular y mostrar la fecha exacta en formato "DD/MM/YYYY". Por ejemplo:
- Si hoy es martes 21/01/2025 y dicen "mañana" → "22/01/2025"
- Si dicen "la próxima semana" → mostrar la fecha del lunes de la próxima semana
- Si dicen "el viernes" → calcular el próximo viernes

Debes responder SOLO con JSON válido, sin ningún otro texto. El JSON debe tener exactamente esta estructura:
{{
  "executive_summary": "Un resumen de 2-3 oraciones de la reunión",
  "key_discussion_points": ["Punto 1", "Punto 2", ...],
  "decisions_made": ["Decisión 1", "Decisión 2", ...],
  "action_items": [{{"task": "Descripción de la tarea", "assignee": "Nombre de persona o Desconocido", "deadline": "DD/MM/YYYY o Por definir"}}],
  "participants_mentioned": ["Nombre 1", "Nombre 2", ...]
}}

Si una sección no tiene elementos, usa un array vacío []. Siempre incluye los cinco campos. Responde en español."""

            user_prompt = f"""Genera una minuta de reunión a partir de esta transcripción:{speaker_context}

TRANSCRIPCIÓN:
{transcription[:config.MINUTES_MAX_INPUT_TOKENS * 4]}

Recuerda: Responde SOLO con JSON válido, sin ningún otro texto. El contenido debe estar en español. Las fechas deben ser calculadas basándose en la fecha de hoy."""

            # Call Claude Haiku 4.5 API
            message = self.client.messages.create(
                model=config.ANTHROPIC_MODEL_ID,
                max_tokens=config.MINUTES_MAX_OUTPUT_TOKENS,
                temperature=config.MINUTES_TEMPERATURE,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )

            # Extract response text
            response = message.content[0].text.strip()
            print(f"Claude response: {response[:500]}...")

            # Parse JSON from response
            minutes = self._parse_json_response(response)

            return {
                "success": True,
                "minutes": minutes,
            }

        except Exception as e:
            import traceback
            print(f"Minutes generation error: {e}")
            print(traceback.format_exc())
            return {
                "success": False,
                "error": str(e),
                "minutes": self._empty_minutes(),
            }

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response, handling common issues."""
        import re

        # Try direct JSON parse first
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code block
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find JSON object in response
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # Return empty structure if parsing fails
        print(f"Failed to parse JSON from response: {response[:200]}...")
        return self._empty_minutes()

    def _empty_minutes(self) -> Dict[str, Any]:
        """Return empty minutes structure."""
        return {
            "executive_summary": "Unable to generate summary.",
            "key_discussion_points": [],
            "decisions_made": [],
            "action_items": [],
            "participants_mentioned": [],
        }


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
