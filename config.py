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
MAX_DURATION_SECONDS = 600  # 10 minutes

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

# Whisper model configuration
WHISPER_MODEL = "large"  # Options: tiny, base, small, medium, large
WHISPER_DEVICE = "cuda"
WHISPER_FP16 = True  # Enable FP16 for faster inference

# Modal configuration
MODAL_APP_NAME = "transcodio-app"
MODAL_VOLUME_NAME = "whisper-models"
MODAL_GPU_TYPE = "L4"  # NVIDIA L4
MODAL_GPU_COUNT = 1
MODAL_CONTAINER_IDLE_TIMEOUT = 120  # 2 minutes
MODAL_TIMEOUT = 600  # 10 minutes max processing time
MODAL_MEMORY_MB = 8192  # 8GB RAM

# ============================================================================
# COLD START OPTIMIZATION FLAGS
# Enable these one at a time to measure impact on cold start performance
# ============================================================================

# Optimization 1: CPU Memory Snapshots
# Captures container state after initialization (excludes GPU state)
# Expected improvement: ~30-50% faster cold starts
ENABLE_CPU_MEMORY_SNAPSHOT = False

# Optimization 2: GPU Memory Snapshots (Experimental - requires CPU snapshots enabled)
# Captures full GPU state including loaded models and compiled kernels
# Expected improvement: 85-90% faster cold starts (34s -> 3-5s)
# NOTE: This is an alpha feature - test thoroughly before production
ENABLE_GPU_MEMORY_SNAPSHOT = False

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
SAMPLE_RATE = 16000  # Whisper's native sample rate
CHUNK_LENGTH_SECONDS = 30  # Whisper processes in 30-second chunks

# Environment variables
ENV_MODE = os.getenv("ENV", "development")
DEBUG = ENV_MODE == "development"
