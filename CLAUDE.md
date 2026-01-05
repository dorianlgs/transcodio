# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Transcodio is a production-ready audio transcription service using OpenAI's Whisper Large model deployed on Modal's serverless GPU infrastructure. The service provides real-time streaming transcription via Server-Sent Events (SSE).

Key features:
- GPU-accelerated transcription using NVIDIA L4 GPUs on Modal
- Real-time streaming results via SSE
- FastAPI web server with REST endpoints
- Audio preprocessing and validation using FFmpeg
- Support for multiple audio formats (MP3, WAV, M4A, FLAC, OGG, WebM)

## Architecture

The application has a three-tier architecture:

1. **Frontend (static/)**: Browser-based UI with drag-and-drop and SSE streaming support
2. **API Layer (api/)**: FastAPI application handling uploads, validation, and SSE streaming
3. **GPU Backend (modal_app/)**: Modal serverless functions running Whisper model

Data flow:
```
User uploads audio → FastAPI validates/preprocesses → Modal GPU transcribes → SSE streams results back
```

### Key Components

**modal_app/app.py**:
- Contains the `WhisperModel` class decorated with `@app.cls()`
- Runs on Modal's serverless GPU infrastructure (NVIDIA L4)
- Two main methods: `transcribe()` for complete results, `transcribe_stream()` for streaming
- Uses Modal volumes to cache the Whisper model at `/models` to avoid redownloading
- Container stays warm for `MODAL_CONTAINER_IDLE_TIMEOUT` seconds (default 120s) to reduce cold starts

**api/main.py**:
- FastAPI application with two transcription endpoints: `/api/transcribe` (non-streaming) and `/api/transcribe/stream` (SSE)
- Connects to Modal using `modal.Cls.from_name()` to lookup the deployed `WhisperModel` class
- Audio validation happens before sending to Modal to save GPU costs
- SSE streaming converts Modal's synchronous generator to async for FastAPI

**utils/audio.py**:
- Audio validation pipeline: file size → format → duration → preprocessing
- Uses FFmpeg to preprocess audio (convert to mono, 16kHz WAV) for optimal Whisper performance
- All validation happens locally before sending to Modal GPU

**config.py**:
- Centralized configuration for GPU settings, file limits, and API settings
- Key settings: `WHISPER_MODEL` (model size), `MODAL_GPU_TYPE` (GPU type), `MAX_FILE_SIZE_MB`, `MAX_DURATION_SECONDS`

## Common Commands

### Development Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Authenticate with Modal (required once)
modal setup
```

### Deployment

```bash
# Deploy Modal GPU backend (required before running API)
modal deploy modal_app/app.py

# Check Modal deployment status
modal app list

# View Modal logs
modal app logs transcodio-app
```

### Running the Service

```bash
# Start FastAPI server (after Modal is deployed)
python -m uvicorn api.main:app --reload

# Alternative: run directly
python api/main.py
```

### Testing

```bash
# Test Modal function directly (bypasses FastAPI)
modal run modal_app/app.py path/to/audio.mp3

# Test API endpoint (requires server running)
curl -X POST "http://localhost:8000/api/transcribe" -F "file=@audio.mp3"

# Test streaming endpoint
curl -X POST "http://localhost:8000/api/transcribe/stream" -F "file=@audio.mp3"
```

## Critical Implementation Details

### Modal Integration

The FastAPI server connects to Modal using the deployed app name:
```python
WhisperModel = modal.Cls.from_name(config.MODAL_APP_NAME, "WhisperModel")
model = WhisperModel()
result = model.transcribe.remote(audio_bytes)
```

The Modal app MUST be deployed before running the FastAPI server, otherwise you'll get "Modal service unavailable" errors.

### Audio Preprocessing

All audio goes through preprocessing before transcription:
1. **Validation**: Check file size, format, and duration (utils/audio.py)
2. **Conversion**: FFmpeg converts to mono, 16kHz WAV (Whisper's native format)
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
- `WHISPER_FP16=True` enables half-precision for 2x faster processing
- Audio preprocessing happens locally to minimize GPU time
- Smaller models (base/small) can be used for less critical use cases

## Configuration Changes

To change Whisper model or GPU type, edit `config.py`:
```python
WHISPER_MODEL = "large"  # Options: tiny, base, small, medium, large
MODAL_GPU_TYPE = "L4"    # Options: L4, A10G, T4
```

Then redeploy Modal: `modal deploy modal_app/app.py`

## Dependencies

**System requirements**:
- Python 3.11+
- FFmpeg (must be installed locally for audio processing)
- Modal account (free tier available)

**Critical Python packages**:
- `modal`: Serverless GPU infrastructure
- `fastapi` + `uvicorn`: Web framework and ASGI server
- `openai-whisper`: Transcription model (runs on Modal GPU)
- `torch` + `torchaudio`: ML framework (runs on Modal GPU)
- `ffmpeg-python`: Audio preprocessing (runs locally)
- `sse-starlette`: Server-Sent Events support

## Troubleshooting

**"Modal service unavailable"**: The Modal app isn't deployed. Run `modal deploy modal_app/app.py`.

**Slow first request**: Cold start takes 20-30s to download model and spin up GPU. Increase `MODAL_CONTAINER_IDLE_TIMEOUT` to keep containers warm longer.

**Audio validation errors**: Check that FFmpeg is installed locally (`ffmpeg -version`). The API server needs FFmpeg to preprocess audio.

**GPU out of memory**: Whisper Large needs ~10GB VRAM. Switch to `medium` or `small` model in config.py if using smaller GPUs.
