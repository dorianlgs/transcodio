# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Transcodio is a production-ready audio transcription service using NVIDIA's Parakeet TDT 0.6B v3 model deployed on Modal's serverless GPU infrastructure. The service provides real-time progressive streaming transcription via Server-Sent Events (SSE) with silence-based segmentation.

Key features:
- GPU-accelerated transcription using NVIDIA L4 GPUs on Modal
- **Real progressive streaming** with silence detection (yields segments as they complete)
- **Speaker diarization** using NVIDIA TitaNet for automatic speaker identification
- **Audio playback** with integrated player using session-based caching
- FastAPI web server with REST endpoints
- Audio preprocessing and validation using FFmpeg
- Support for multiple audio formats (MP3, WAV, M4A, FLAC, OGG, WebM)
- Subtitle export (SRT/VTT formats) with speaker labels
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
- **Speaker diarization**: `SpeakerDiarizerModel` class using NVIDIA TitaNet for speaker embeddings
  - Single-scale embedding extraction (1.5s windows by default, prevents multi-scale artifacts)
  - Automatic speaker count detection using combined score with complexity penalty
  - AgglomerativeClustering with cosine distance to identify and label speakers
  - Complexity penalty (0.15 per speaker) prevents over-segmentation
  - `align_speakers_to_segments()` function maps speaker labels to transcription segments
- Uses Modal volumes to cache the model at `/models` to avoid redownloading
- Container stays warm for `MODAL_CONTAINER_IDLE_TIMEOUT` seconds (default 120s) to reduce cold starts
- **NoStdStreams** context manager suppresses NeMo's verbose logging

**api/main.py**:
- FastAPI application with two transcription endpoints: `/api/transcribe` (non-streaming) and `/api/transcribe/stream` (SSE)
- **Audio session management**: `/api/audio/{session_id}` endpoint serves cached audio for playback
  - Generates UUID session IDs for each transcription
  - Caches original uploaded audio files for 1 hour
  - Automatic cleanup of expired audio cache entries
- Connects to Modal using `modal.Cls.from_name()` to lookup deployed models (`ParakeetSTTModel`, `SpeakerDiarizerModel`)
- Audio validation happens before sending to Modal to save GPU costs
- SSE streaming converts Modal's synchronous generator to async for FastAPI
- **Speaker diarization integration**: Runs after transcription completes, yields `speakers_ready` event with annotated segments

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
  - **Speaker diarization settings**:
    - `ENABLE_SPEAKER_DIARIZATION` (feature flag, default: True)
    - `DIARIZATION_MODEL = "nvidia/speakerverification_en_titanet_large"`
    - `DIARIZATION_MIN_SPEAKERS`, `DIARIZATION_MAX_SPEAKERS` (default: 1-5)
    - `DIARIZATION_WINDOW_LENGTHS` (multi-scale windows: [1.5, 1.0, 0.5] seconds)
    - `DIARIZATION_SHIFT_LENGTH` (default: 0.75 seconds)

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
   - `speakers_ready`: Speaker-annotated segments after diarization completes (optional, if enabled)
   - `complete`: Transcription finished (includes full transcription text and audio session ID for playback)
   - `error`: Any errors during processing

**Key Difference from Previous Implementation**:
- **Before (Kyutai)**: Transcribed entire audio, yielded single segment at end ("fake streaming")
- **Now (Parakeet)**: Detects natural pauses, yields segments progressively as they complete (**real streaming**)

**Recent Improvements**:
- **Full transcription in completion**: The final `complete` event now includes the complete transcription text assembled from all segments, making it easier to retrieve the entire result
- **Segment accumulation**: All segment texts are accumulated during streaming and returned together in the completion event
- **UI enhancements**: Copy button relocated inside the full text container for better user experience and more intuitive access
- **Speaker diarization**: Automatic speaker identification integrated into streaming pipeline
- **Audio playback**: Session-based caching enables audio player functionality

**Complete Streaming Flow with Speaker Diarization**:
```
1. User uploads audio → FastAPI validates & preprocesses
2. Generate session ID, cache original audio
3. Stream to Modal GPU → Parakeet transcribes with silence detection
4. Frontend receives progressive segments via SSE (`metadata` → `progress` events)
5. After transcription completes → TitaNet diarization runs in background
6. Speaker labels aligned to segments → `speakers_ready` event updates UI
7. Final `complete` event with full text & audio session ID
8. Audio player loads using session ID from cache
9. User can download transcription (TXT), subtitles (SRT/VTT), or listen to audio
```

This architecture ensures:
- **Progressive feedback**: Users see segments as they're transcribed (not blocked by diarization)
- **Non-blocking diarization**: Speaker identification happens asynchronously after transcription
- **Graceful degradation**: If diarization fails, transcription still succeeds
- **Audio playback**: Original audio available for listening without re-uploading

### Silence Detection Tuning

Adjust these parameters in `config.py` to control segmentation granularity:

```python
SILENCE_THRESHOLD_DB = -40   # Lower = more sensitive (detects softer pauses)
SILENCE_MIN_LENGTH_MS = 700  # Lower = detects shorter pauses
```

**Current Configuration**:
The default values have been optimized for balanced performance:
- **-40 dB threshold**: Provides a good balance between detecting natural pauses and avoiding over-segmentation
- **700ms minimum silence**: Catches most natural speech pauses without creating too many fragments
- These values were tuned from initial settings (-35 dB / 400ms) to reduce excessive segmentation while maintaining responsiveness

**Guidelines**:
- **Fewer segments** (longer segments): Increase threshold to -45, increase min_length to 1000ms
- **More segments** (shorter segments): Decrease threshold to -35, decrease min_length to 400ms
- **Balanced** (3-5 segments per minute): -40 dB, 700ms (current default)

After changing these values, redeploy: `py -m modal deploy modal_app/app.py`

### Speaker Diarization

The service includes **automatic speaker identification** that runs after transcription completes. This feature uses NVIDIA TitaNet embeddings combined with spectral clustering to identify and label different speakers.

**How it works:**
1. **Single-scale embedding extraction**: Audio is analyzed using 1.5s windows with 0.75s overlap (single-scale to avoid multi-scale artifacts)
2. **Speaker embedding**: TitaNet model generates normalized speaker embeddings for each audio window
3. **Automatic speaker detection**: Combined scoring with complexity penalty determines optimal number of speakers (1-5 by default)
4. **Clustering**: AgglomerativeClustering with cosine distance groups similar embeddings to identify unique speakers
5. **Alignment**: Speaker labels are mapped to transcription segments based on temporal overlap
6. **Frontend display**: Segments show speaker badges (e.g., "Speaker 1", "Speaker 2")
7. **Subtitle export**: Speaker labels are included in SRT/VTT downloads

**Configuration** (in `config.py`):
```python
ENABLE_SPEAKER_DIARIZATION = True  # Toggle feature on/off
DIARIZATION_MODEL = "nvidia/speakerverification_en_titanet_large"
DIARIZATION_MIN_SPEAKERS = 1  # Minimum speakers to detect
DIARIZATION_MAX_SPEAKERS = 5  # Maximum speakers (higher = slower)
DIARIZATION_WINDOW_LENGTHS = [1.5, 1.0, 0.5]  # Only first value used (1.5s window)
DIARIZATION_SHIFT_LENGTH = 0.75  # Window shift/overlap (seconds)
```

**Algorithm details:**
- **Clustering**: AgglomerativeClustering with cosine distance (better for speaker embeddings than euclidean)
- **Speaker selection**: Combined score = 60% silhouette + 40% Calinski-Harabasz - complexity penalty
- **Complexity penalty**: 0.15 per additional speaker (prefers simpler explanations per Occam's Razor)
- **Single-scale**: Uses only the first window length to avoid multi-scale artifacts (previous versions had 3-speaker bias from multi-scale)

**Performance considerations:**
- Diarization runs **after** transcription completes, so it doesn't block streaming
- Single-scale approach (1 window size) is faster and more accurate than multi-scale
- Complexity penalty prevents over-segmentation (fewer false positives)
- Processing time: ~1-3 seconds for a 1-minute audio file on L4 GPU
- If diarization fails, transcription still completes successfully (graceful degradation)

**Tuning tips:**
- **More speakers**: Increase `DIARIZATION_MAX_SPEAKERS` (trades speed for accuracy)
- **Longer windows**: Change `DIARIZATION_WINDOW_LENGTHS[0]` to `2.0` for better speaker characterization
- **Shorter windows**: Change to `1.0` for faster processing but less accurate speaker identification
- **More overlap**: Decrease `DIARIZATION_SHIFT_LENGTH` to `0.5` for smoother speaker transitions
- **Adjust complexity penalty**: Edit line 519 in `modal_app/app.py` (higher = prefers fewer speakers)
- **Disable entirely**: Set `ENABLE_SPEAKER_DIARIZATION = False` to skip diarization

After changing these values, redeploy: `py -m modal deploy modal_app/app.py`

### Audio Player

The frontend includes an **integrated audio player** that allows users to listen to the original uploaded audio alongside the transcription.

**How it works:**
1. **Session management**: Each transcription generates a unique UUID session ID
2. **Audio caching**: Original uploaded audio is cached server-side for 1 hour
3. **Playback endpoint**: `/api/audio/{session_id}` serves the cached audio file
4. **Automatic cleanup**: Expired audio sessions are cleaned up automatically
5. **Format preservation**: Audio is served in its original format (MP3, WAV, etc.) for best browser compatibility

**Key implementation details:**
- Audio cache is **in-memory** (simple dict) - suitable for low-traffic scenarios
- Cache stores **original uploaded files** (not preprocessed 16kHz WAV) for better quality
- Session expiry: 1 hour after transcription
- Browser audio player supports seeking, volume control, and playback speed
- Audio loads automatically when transcription completes

**Note**: For production with high traffic, consider replacing the in-memory cache with Redis or a similar persistent store.

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
- **`scikit-learn>=1.3.0`**: Spectral clustering for speaker diarization
- **`soundfile>=0.12.1`**: Audio file I/O for diarization

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

**Speaker diarization not working**: Check these common issues:
- Ensure `ENABLE_SPEAKER_DIARIZATION = True` in `config.py`
- Check Modal logs for diarization errors: `py -m modal app logs transcodio-app`
- Diarization requires at least ~5-10 seconds of audio with distinct speakers
- Single-speaker audio will show "Speaker 1" for all segments (this is expected)
- If diarization fails, transcription still completes (it's non-blocking)

**Audio player not loading**:
- Check that the audio session ID is being returned in the `complete` event
- Verify the `/api/audio/{session_id}` endpoint is accessible
- Audio cache expires after 1 hour - if playback fails, the session may have expired
- Check browser console for CORS or network errors

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
