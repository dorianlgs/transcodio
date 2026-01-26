"""Configuration constants for Transcodio transcription service."""

import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
TEMP_DIR = BASE_DIR / "temp"

# Audio file limits
MAX_FILE_SIZE_MB = 100
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_DURATION_SECONDS = 3600  # 60 minutes

# Supported audio formats
SUPPORTED_FORMATS = ["mp3", "wav", "m4a", "flac", "ogg", "webm", "mp4"]
SUPPORTED_MIME_TYPES = [
    "audio/mpeg",
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/mp4",
    "audio/x-m4a",
    "audio/flac",
    "audio/ogg",
    "audio/webm",
    "video/mp4",
]

# NVIDIA Parakeet TDT model configuration
STT_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"  # HuggingFace model ID
STT_DEVICE = "cuda"
STT_DTYPE = "auto"  # Use auto dtype selection (bfloat16 on supported GPUs)

# Modal configuration
MODAL_APP_NAME = "transcodio-app"
MODAL_VOLUME_NAME = "parakeet-models"  # Changed from kyutai-stt-models
MODAL_GPU_TYPE = "L4"  # NVIDIA L4
MODAL_GPU_COUNT = 1
MODAL_CONTAINER_IDLE_TIMEOUT = 120  # 2 minutes
MODAL_TIMEOUT = 3000  # 50 minutes max processing time (allows for 33min audio + diarization)
MODAL_MEMORY_MB = 8192  # 8GB RAM

# ============================================================================
# COLD START OPTIMIZATION FLAGS
# Enable these one at a time to measure impact on cold start performance
# ============================================================================

# Optimization 1: CPU Memory Snapshots
# Captures container state after initialization (excludes GPU state)
# Expected improvement: ~30-50% faster cold starts
ENABLE_CPU_MEMORY_SNAPSHOT = True

# Optimization 2: GPU Memory Snapshots (Experimental - requires CPU snapshots enabled)
# Captures full GPU state including loaded models and compiled kernels
# Expected improvement: 85-90% faster cold starts (34s -> 3-5s)
# NOTE: This is an alpha feature - test thoroughly before production
ENABLE_GPU_MEMORY_SNAPSHOT = True

# Optimization 3: Model Warm-up Pass
# Runs a dummy transcription during initialization to compile CUDA kernels
# Works best WITH GPU snapshots to capture compiled kernels
# Expected improvement: Minimal without snapshots, significant with GPU snapshots
ENABLE_MODEL_WARMUP = False

# Optimization 4: Extended Container Idle Timeout
# Keep containers warm longer to avoid cold starts
# Trade-off: Higher idle costs vs fewer cold starts
# Recommended values: 300 (5 min), 600 (10 min), 1200 (20 min)
# Set to 120 for baseline testing
EXTENDED_IDLE_TIMEOUT = False
EXTENDED_IDLE_TIMEOUT_SECONDS = 300  # Used when EXTENDED_IDLE_TIMEOUT = True

# API configuration
API_TITLE = "Transcodio Transcription API"
API_VERSION = "1.0.0"
API_DESCRIPTION = "Fast, cost-effective audio transcription with streaming results"

# CORS configuration
CORS_ORIGINS = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:3000",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
]

# Transcription settings
SAMPLE_RATE = 16000  # Parakeet TDT's native sample rate (16kHz)

# Silence detection configuration for streaming
# Lower threshold = more sensitive (detects softer pauses)
# Lower min_length = detects shorter pauses
SILENCE_THRESHOLD_DB = -40  # dB threshold for silence detection (balance between -45 and -35)
SILENCE_MIN_LENGTH_MS = 700  # Minimum silence duration in milliseconds (balance between 1000 and 400)

# Speaker Diarization Configuration
ENABLE_SPEAKER_DIARIZATION = True  # Feature flag
DIARIZATION_MODEL = "nvidia/speakerverification_en_titanet_large"  # TitaNet embeddings
DIARIZATION_MIN_SPEAKERS = 1  # Minimum speakers to detect
DIARIZATION_MAX_SPEAKERS = 5  # Maximum speakers to detect (uses silhouette score to find optimal)

# Embedding extraction configuration
# NOTE: Only the FIRST window length is used (single-scale to avoid multi-scale artifacts)
DIARIZATION_WINDOW_LENGTHS = [1.5, 1.0, 0.5]  # Window lengths in seconds (only [0] is used)
DIARIZATION_SHIFT_LENGTH = 0.75  # Shift length in seconds (window overlap)

# Meeting Minutes Generation Configuration (Anthropic Claude API)
ENABLE_MEETING_MINUTES = True  # Feature flag
ANTHROPIC_MODEL_ID = "claude-haiku-4-5-20251001"  # Claude Haiku 4.5 - fast and cost-effective
MINUTES_MAX_INPUT_TOKENS = 8000  # Maximum input tokens (transcription) - Haiku supports 200k context
MINUTES_MAX_OUTPUT_TOKENS = 2048  # Maximum output tokens (minutes)
MINUTES_TEMPERATURE = 0.3  # Low temperature for consistent, structured output
MINUTES_CONTAINER_IDLE_TIMEOUT = 60  # 1 minute (shorter than STT since less frequent)

# Voice Cloning Configuration
ENABLE_VOICE_CLONING = True
TTS_CONTAINER_IDLE_TIMEOUT = 120

# Available TTS Models
TTS_MODELS = {
    "qwen": {
        "name": "Qwen3-TTS",
        "model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "sample_rate": 24000,
        "gpu_type": "L4",
        "memory_mb": 8192,
        "description": "Fast, good quality (1.7B params)",
    },
    "higgs": {
        "name": "Higgs Audio V2",
        "model_id": "bosonai/higgs-audio-v2-generation-3B-base",
        "tokenizer_id": "bosonai/higgs-audio-v2-tokenizer",
        "sample_rate": 24000,
        "gpu_type": "L4",
        "memory_mb": 24576,
        "description": "High quality, expressive (3B params)",
    },
    "fish": {
        "name": "Fish Audio S1",
        "model_id": "fishaudio/openaudio-s1-mini",
        "sample_rate": 44100,
        "gpu_type": "L4",
        "memory_mb": 16384,
        "description": "SOTA voice cloning, multilingual (0.5B params)",
    },
}

# Default TTS model
DEFAULT_TTS_MODEL = "qwen"

# Voice Cloning Constraints
VOICE_CLONE_MIN_REF_DURATION = 3   # seconds
VOICE_CLONE_MAX_REF_DURATION = 60  # seconds
VOICE_CLONE_MAX_TARGET_TEXT = 500  # characters
VOICE_CLONE_SAMPLE_RATE = 24000

VOICE_CLONE_LANGUAGES = [
    "English", "Spanish", "Chinese", "Japanese", "Korean",
    "German", "French", "Russian", "Portuguese", "Italian"
]

# Environment variables
ENV_MODE = os.getenv("ENV", "development")
DEBUG = ENV_MODE == "development"
