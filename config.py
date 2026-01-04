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
