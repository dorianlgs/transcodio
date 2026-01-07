# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Transcodio is a production-ready audio transcription service using NVIDIA's Parakeet TDT 0.6B v3 model deployed on Modal's serverless GPU infrastructure. The service provides real-time progressive streaming transcription via Server-Sent Events (SSE) with silence-based segmentation.

Key features:
- GPU-accelerated transcription using NVIDIA L4 GPUs on Modal
- **Real progressive streaming** with silence detection (yields segments as they complete)
- FastAPI web server with REST endpoints
- Audio preprocessing and validation using FFmpeg
- Support for multiple audio formats (MP3, WAV, M4A, FLAC, OGG, WebM)
- Subtitle export (SRT/VTT formats)
- Lightweight model (0.6B parameters) for faster inference and lower costs

## Architecture

The application has a three-tier architecture:

1. **Frontend (static/)**: Browser-based UI with drag-and-drop and SSE streaming support
2. **API Layer (api/)**: FastAPI application handling uploads, validation, and SSE streaming
3. **GPU Backend (modal_app/)**: Modal serverless functions running NVIDIA Parakeet TDT model with NeMo framework

Data flow:
```
User uploads audio → FastAPI validates/preprocesses → Modal GPU transcribes → SSE streams results back
```

### Key Components

**modal_app/app.py**:
- Contains the `ParakeetSTTModel` class decorated with `@app.cls()`
- Runs on Modal's serverless GPU infrastructure (NVIDIA L4)
- Uses **NVIDIA NeMo framework** for ASR (Automatic Speech Recognition)
- Two main methods: `transcribe()` for complete results, `transcribe_stream()` for progressive streaming
- **Real streaming**: Detects silence boundaries using pydub, transcribes segments progressively
- Uses Modal volumes to cache the model at `/models` to avoid redownloading
- Container stays warm for `MODAL_CONTAINER_IDLE_TIMEOUT` seconds (default 120s) to reduce cold starts
- **NoStdStreams** context manager suppresses NeMo's verbose logging

**api/main.py**:
- FastAPI application with two transcription endpoints: `/api/transcribe` (non-streaming) and `/api/transcribe/stream` (SSE)
- Connects to Modal using `modal.Cls.from_name()` to lookup the deployed model class (`ParakeetSTTModel`)
- Audio validation happens before sending to Modal to save GPU costs
- SSE streaming converts Modal's synchronous generator to async for FastAPI

**utils/audio.py**:
- Audio validation pipeline: file size → format → duration → preprocessing
- Uses FFmpeg to preprocess audio (convert to mono, **16kHz WAV**) for optimal Parakeet TDT performance
- All validation happens locally before sending to Modal GPU

**config.py**:
- Centralized configuration for GPU settings, file limits, and API settings
- Key settings:
  - `STT_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"` (HuggingFace model ID)
  - `SAMPLE_RATE = 16000` (Parakeet's native sample rate)
  - `MODAL_GPU_TYPE` (GPU type), `MAX_FILE_SIZE_MB`, `MAX_DURATION_SECONDS`
  - **Silence detection params**: `SILENCE_THRESHOLD_DB`, `SILENCE_MIN_LENGTH_MS`

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
STTModel = modal.Cls.from_name(config.MODAL_APP_NAME, "ParakeetSTTModel")
model = STTModel()
result = model.transcribe.remote(audio_bytes)
```

The Modal app MUST be deployed before running the FastAPI server, otherwise you'll get "Modal service unavailable" errors.

### Audio Preprocessing

All audio goes through preprocessing before transcription:
1. **Validation**: Check file size, format, and duration (utils/audio.py)
2. **Conversion**: FFmpeg converts to mono, **16kHz WAV** (Parakeet TDT's native format)
3. **Transmission**: Send preprocessed bytes to Modal GPU

This preprocessing happens in the FastAPI layer, not on Modal, to keep GPU time minimal and reduce costs.

### Streaming Architecture

The streaming endpoint uses **real progressive streaming** with silence detection:

1. **Modal's `transcribe_stream()`**:
   - Uses pydub to detect silence windows in audio
   - Configurable via `SILENCE_THRESHOLD_DB` (default: -40 dB) and `SILENCE_MIN_LENGTH_MS` (default: 700ms)
   - Transcribes each segment between silences independently
   - Yields JSON progressively: `{"type": "metadata|segment|complete|error", ...}`

2. **FastAPI's `event_generator()`**:
   - Converts Modal's synchronous generator to async
   - Formats as SSE events

3. **Browser receives SSE events**:
   - `metadata`: Audio duration and language
   - `progress`: Each transcribed segment (yields multiple times)
   - `complete`: Transcription finished
   - `error`: Any errors during processing

**Key Difference from Previous Implementation**:
- **Before (Kyutai)**: Transcribed entire audio, yielded single segment at end ("fake streaming")
- **Now (Parakeet)**: Detects natural pauses, yields segments progressively as they complete (**real streaming**)

### Silence Detection Tuning

Adjust these parameters in `config.py` to control segmentation granularity:

```python
SILENCE_THRESHOLD_DB = -40   # Lower = more sensitive (detects softer pauses)
SILENCE_MIN_LENGTH_MS = 700  # Lower = detects shorter pauses
```

**Guidelines**:
- **Fewer segments** (longer segments): Increase threshold to -45, increase min_length to 1000ms
- **More segments** (shorter segments): Decrease threshold to -35, decrease min_length to 400ms
- **Balanced** (3-5 segments per minute): -40 dB, 700ms (current default)

After changing these values, redeploy: `py -m modal deploy modal_app/app.py`

### Cost Optimization

GPU costs are ~$0.006 per minute of audio. Optimization strategies:
- **Parakeet TDT 0.6B** is much smaller than previous models (0.6B vs 2.6B params) = faster & cheaper
- `MODAL_CONTAINER_IDLE_TIMEOUT` keeps containers warm to reduce cold starts (20-30s)
- Audio preprocessing happens locally to minimize GPU time
- **Memory snapshots** can be enabled for 85-90% faster cold starts (see `ENABLE_GPU_MEMORY_SNAPSHOT` in config.py)

## Configuration Changes

### Changing the STT Model or GPU Type

Edit `config.py`:
```python
STT_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"  # Or another NeMo-compatible model
MODAL_GPU_TYPE = "L4"    # Options: L4, A10G, T4
SAMPLE_RATE = 16000      # Must match model's native sample rate
```

Then redeploy Modal: `py -m modal deploy modal_app/app.py`

### Adjusting Silence Detection

Edit `config.py`:
```python
SILENCE_THRESHOLD_DB = -40    # -50 (very conservative) to -30 (very sensitive)
SILENCE_MIN_LENGTH_MS = 700   # 300ms (granular) to 1500ms (conservative)
```

Redeploy Modal: `py -m modal deploy modal_app/app.py`

## Dependencies

**System requirements**:
- Python 3.12+ (Modal container uses Python 3.12)
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer and runner
- FFmpeg (must be installed locally for audio processing)
- Modal account (free tier available)

**Critical Python packages**:
- `modal`: Serverless GPU infrastructure
- `fastapi` + `uvicorn`: Web framework and ASGI server
- **`nemo_toolkit[asr]==2.3.0`**: NVIDIA NeMo framework for ASR (includes PyTorch & torchaudio)
- `pydub`: Silence detection for progressive streaming
- `numpy<2`: Required for NeMo compatibility
- `ffmpeg-python`: Audio preprocessing (runs locally)
- `sse-starlette`: Server-Sent Events support

**Modal Container Image**:
- Base: `nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04`
- Python: 3.12
- GPU libraries: CUDA 12.8, cuDNN

## Troubleshooting

**"Modal service unavailable"**: The Modal app isn't deployed. Run `py -m modal deploy modal_app/app.py`.

**Slow first request**: Cold start takes 30-60s to download model and spin up GPU. Enable memory snapshots in `config.py`:
```python
ENABLE_CPU_MEMORY_SNAPSHOT = True
ENABLE_GPU_MEMORY_SNAPSHOT = True  # Experimental, 85-90% faster cold starts
```

**Audio validation errors**: Check that FFmpeg is installed locally (`ffmpeg -version`). The API server needs FFmpeg to preprocess audio.

**GPU out of memory**: Parakeet TDT 0.6B needs ~3-4GB VRAM (much less than previous models). L4 (24GB VRAM) is over-provisioned; T4 (16GB) would also work.

**Too many/few segments in streaming**: Adjust silence detection parameters in `config.py`:
- Too many segments: Increase `SILENCE_THRESHOLD_DB` and `SILENCE_MIN_LENGTH_MS`
- Too few segments: Decrease `SILENCE_THRESHOLD_DB` and `SILENCE_MIN_LENGTH_MS`

**NeMo verbose logs**: The `NoStdStreams` context manager in `modal_app/app.py` suppresses NeMo's stdout/stderr during transcription. If you need to debug, temporarily remove the `with NoStdStreams():` context.

## Model Comparison

| Feature | Previous (Kyutai STT 2.6B) | Current (Parakeet TDT 0.6B v3) |
|---------|---------------------------|-------------------------------|
| Framework | HuggingFace Transformers | NVIDIA NeMo |
| Parameters | 2.6B | 0.6B (4.3x smaller) |
| Sample Rate | 24kHz | 16kHz |
| VRAM Usage | ~12-15GB | ~3-4GB |
| Streaming | Fake (single segment) | Real (silence-based) |
| Speed | Baseline | ~2-3x faster |
| Accuracy | High | High (comparable) |
