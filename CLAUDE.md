# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Transcodio is a production-ready audio transcription service using Kyutai's STT 2.6B model deployed on Modal's serverless GPU infrastructure. The service provides real-time streaming transcription via Server-Sent Events (SSE).

Key features:
- GPU-accelerated transcription using NVIDIA L4 GPUs on Modal
- Real-time streaming results via SSE
- FastAPI web server with REST endpoints
- Audio preprocessing and validation using FFmpeg
- Support for multiple audio formats (MP3, WAV, M4A, FLAC, OGG, WebM)
- Subtitle export (SRT/VTT formats)

## Architecture

The application has a three-tier architecture:

1. **Frontend (static/)**: Browser-based UI with drag-and-drop and SSE streaming support
2. **API Layer (api/)**: FastAPI application handling uploads, validation, and SSE streaming
3. **GPU Backend (modal_app/)**: Modal serverless functions running Kyutai STT model

Data flow:
```
User uploads audio → FastAPI validates/preprocesses → Modal GPU transcribes → SSE streams results back
```

### Key Components

**modal_app/app.py**:
- Contains the `KyutaiSTTModel` class decorated with `@app.cls()`
- Runs on Modal's serverless GPU infrastructure (NVIDIA L4)
- Two main methods: `transcribe()` for complete results, `transcribe_stream()` for streaming
- Uses Modal volumes to cache the model at `/models` to avoid redownloading
- Container stays warm for `MODAL_CONTAINER_IDLE_TIMEOUT` seconds (default 120s) to reduce cold starts

**api/main.py**:
- FastAPI application with two transcription endpoints: `/api/transcribe` (non-streaming) and `/api/transcribe/stream` (SSE)
- Connects to Modal using `modal.Cls.from_name()` to lookup the deployed model class
- Audio validation happens before sending to Modal to save GPU costs
- SSE streaming converts Modal's synchronous generator to async for FastAPI

**utils/audio.py**:
- Audio validation pipeline: file size → format → duration → preprocessing
- Uses FFmpeg to preprocess audio (convert to mono, 24kHz WAV) for optimal Kyutai STT performance
- All validation happens locally before sending to Modal GPU

**config.py**:
- Centralized configuration for GPU settings, file limits, and API settings
- Key settings: `STT_MODEL_ID` (model ID), `MODAL_GPU_TYPE` (GPU type), `MAX_FILE_SIZE_MB`, `MAX_DURATION_SECONDS`

## Common Commands

### Development Setup

```bash
# Install dependencies using uv
uv sync

# Authenticate with Modal (required once)
py -m modal setup
```

### Deployment

```bash
# Deploy Modal GPU backend (required before running API)
py -m modal deploy modal_app/app.py

# Check Modal deployment status
py -m modal app list

# View Modal logs
py -m modal app logs transcodio-app
```

### Running the Service

```bash
# Start FastAPI server (after Modal is deployed)
uv run uvicorn api.main:app --reload

# Alternative: run directly
uv run python api/main.py
```

### Testing

```bash
# Test Modal function directly (bypasses FastAPI)
py -m modal run modal_app/app.py path/to/audio.mp3

# Test using the CLI tool
uv run transcribe_file.py path/to/audio.mp3

# Test API endpoint (requires server running)
curl -X POST "http://localhost:8000/api/transcribe" -F "file=@audio.mp3"

# Test streaming endpoint
curl -X POST "http://localhost:8000/api/transcribe/stream" -F "file=@audio.mp3"
```

## Critical Implementation Details

### Modal Integration

The FastAPI server connects to Modal using the deployed app name:
```python
STTModel = modal.Cls.from_name(config.MODAL_APP_NAME, "KyutaiSTTModel")
model = STTModel()
result = model.transcribe.remote(audio_bytes)
```

The Modal app MUST be deployed before running the FastAPI server, otherwise you'll get "Modal service unavailable" errors.

### Audio Preprocessing

All audio goes through preprocessing before transcription:
1. **Validation**: Check file size, format, and duration (utils/audio.py)
2. **Conversion**: FFmpeg converts to mono, 24kHz WAV (Kyutai STT's native format)
3. **Transmission**: Send preprocessed bytes to Modal GPU

This preprocessing happens in the FastAPI layer, not on Modal, to keep GPU time minimal and reduce costs.

### Streaming Architecture

The streaming endpoint uses a layered approach:
1. Modal's `transcribe_stream()` returns a synchronous generator yielding JSON strings
2. Each JSON contains: `{"type": "metadata|segment|complete|error", ...}`
3. FastAPI's `event_generator()` converts this to async and formats as SSE events
4. Browser receives SSE events: `metadata`, `progress`, `complete`, `error`

### Cost Optimization

GPU costs are ~$0.006 per minute of audio. Optimization strategies:
- `MODAL_CONTAINER_IDLE_TIMEOUT` keeps containers warm to reduce cold starts (20-30s)
- Audio preprocessing happens locally to minimize GPU time

## Configuration Changes

To change the STT model or GPU type, edit `config.py`:
```python
STT_MODEL_ID = "kyutai/stt-2.6b-en-trfs"  # HuggingFace model ID
MODAL_GPU_TYPE = "L4"    # Options: L4, A10G, T4
```

Then redeploy Modal: `py -m modal deploy modal_app/app.py`

## Dependencies

**System requirements**:
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer and runner
- FFmpeg (must be installed locally for audio processing)
- Modal account (free tier available)

**Critical Python packages**:
- `modal`: Serverless GPU infrastructure
- `fastapi` + `uvicorn`: Web framework and ASGI server
- `transformers>=4.53.0`: Kyutai STT model (runs on Modal GPU)
- `torch` + `torchaudio`: ML framework (runs on Modal GPU)
- `ffmpeg-python`: Audio preprocessing (runs locally)
- `sse-starlette`: Server-Sent Events support

## Troubleshooting

**"Modal service unavailable"**: The Modal app isn't deployed. Run `py -m modal deploy modal_app/app.py`.

**Slow first request**: Cold start takes 30-60s to download model and spin up GPU. Increase `MODAL_CONTAINER_IDLE_TIMEOUT` to keep containers warm longer.

**Audio validation errors**: Check that FFmpeg is installed locally (`ffmpeg -version`). The API server needs FFmpeg to preprocess audio.

**GPU out of memory**: Kyutai STT 2.6B needs ~12-15GB VRAM. Ensure you're using an L4 or better GPU.
