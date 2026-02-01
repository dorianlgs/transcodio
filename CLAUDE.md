# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Transcodio is a production-ready audio transcription service using NVIDIA's Parakeet TDT 0.6B v3 model deployed on Modal's serverless GPU infrastructure. The service provides real-time progressive streaming transcription via Server-Sent Events (SSE) with silence-based segmentation.

Key features:
- GPU-accelerated transcription using NVIDIA L4 GPUs on Modal
- **Real progressive streaming** with silence detection (yields segments as they complete)
- **Speaker diarization** using NVIDIA TitaNet for automatic speaker identification
- **Audio playback** with integrated player using session-based caching
- **Image generation** using FLUX.1-schnell model for text-to-image
- **Voice cloning** using Qwen3-TTS with saved voice profiles
- **Saved voices** with persistent storage in Modal Volume
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
- **Embedded configuration**: All Modal-specific settings are embedded directly at the top of the file (not imported from config.py) to avoid import issues in Modal containers
- **Speaker diarization**: `SpeakerDiarizerModel` class using NVIDIA TitaNet for speaker embeddings
  - Single-scale embedding extraction (1.5s windows by default, prevents multi-scale artifacts)
  - Automatic speaker count detection using combined score with complexity penalty
  - AgglomerativeClustering with cosine distance to identify and label speakers
  - Complexity penalty (0.15 per speaker) prevents over-segmentation
  - `align_speakers_to_segments()` function maps speaker labels to transcription segments
- Uses Modal volumes to cache the model at `/models` to avoid redownloading
- Container stays warm for `MODAL_CONTAINER_IDLE_TIMEOUT` seconds (default 120s) to reduce cold starts
- **NoStdStreams** context manager suppresses NeMo's verbose logging
- **Image generation**: `FluxImageGenerator` class using FLUX.1-schnell for text-to-image generation
  - Uses diffusers library with bfloat16 precision
  - Sequential CPU offload for lower memory usage
  - Optimized for 4 inference steps (schnell mode)

**api/main.py**:
- FastAPI application with two transcription endpoints: `/api/transcribe` (non-streaming) and `/api/transcribe/stream` (SSE)
- **Audio session management**: `/api/audio/{session_id}` endpoint serves cached audio for playback
  - Generates UUID session IDs for each transcription
  - Caches original uploaded audio files for 1 hour
  - Automatic cleanup of expired audio cache entries
- **Image generation endpoints**: `/api/generate-image` (POST) and `/api/image/{session_id}` (GET)
  - `/api/generate-image`: Accepts prompt, width, height parameters; returns session ID
  - `/api/image/{session_id}`: Retrieves generated image as PNG
  - In-memory image cache with 1-hour expiry
- Connects to Modal using `modal.Cls.from_name()` to lookup deployed models (`ParakeetSTTModel`, `SpeakerDiarizerModel`, `FluxImageGenerator`, `VoiceStorage`, `Qwen3TTSVoiceCloner`)
- Audio validation happens before sending to Modal to save GPU costs
- SSE streaming converts Modal's synchronous generator to async for FastAPI
- **Speaker diarization integration**: Runs after transcription completes, yields `speakers_ready` event with annotated segments

**utils/audio.py**:
- Audio validation pipeline: file size → format → duration → preprocessing
- Uses FFmpeg to preprocess audio (convert to mono, **16kHz WAV**) for optimal Parakeet TDT performance
- All validation happens locally before sending to Modal GPU

**config.py**:
- Configuration for the **API layer** (FastAPI server) - file limits, API settings, feature flags
- **Note**: Modal-specific settings are duplicated/embedded in `modal_app/app.py` to avoid import issues in Modal containers. When changing Modal settings, update both files.
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
  - **Meeting minutes settings** (Anthropic Claude API):
    - `ENABLE_MEETING_MINUTES` (feature flag, default: True)
    - `ANTHROPIC_MODEL_ID = "claude-haiku-4-5-20251001"` (Claude Haiku 4.5)
    - `MINUTES_MAX_INPUT_TOKENS`, `MINUTES_MAX_OUTPUT_TOKENS`
    - `MINUTES_TEMPERATURE` (default: 0.3 for structured output)
  - **Image generation settings** (FLUX.1-schnell):
    - `ENABLE_IMAGE_GENERATION` (feature flag, default: True)
    - `IMAGE_GENERATION_MODEL = "black-forest-labs/FLUX.1-schnell"`
    - `IMAGE_GPU_TYPE`, `IMAGE_MEMORY_MB` (L4 GPU, 16GB memory)
    - `IMAGE_DEFAULT_WIDTH`, `IMAGE_DEFAULT_HEIGHT` (768x768 default)
    - `IMAGE_NUM_INFERENCE_STEPS` (4 steps, optimized for schnell)
    - `IMAGE_GUIDANCE_SCALE` (0.0 for schnell mode)
    - `IMAGE_CACHE_EXPIRY_HOURS` (1 hour cache)
  - **Saved voices settings**:
    - `VOICES_STORAGE_PATH` (path within Modal Volume)
    - `VOICES_INDEX_FILE` (index.json)
    - `MAX_SAVED_VOICES` (default: 50 voices)

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

# Test image generation endpoint
curl -X POST "http://localhost:8000/api/generate-image" -F "prompt=a beautiful sunset over mountains" -F "width=768" -F "height=768"
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
- **Embedded Modal configuration**: All Modal-specific settings are now embedded directly in `modal_app/app.py` instead of importing from `config.py`, avoiding import issues in Modal containers
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

Adjust these parameters in `modal_app/app.py` (embedded config at top of file) to control segmentation granularity:

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

**Configuration** (in `modal_app/app.py` - embedded config at top of file):
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
- **More speakers**: Increase `DIARIZATION_MAX_SPEAKERS` in `modal_app/app.py` (trades speed for accuracy)
- **Longer windows**: Change `DIARIZATION_WINDOW_LENGTHS[0]` to `2.0` for better speaker characterization
- **Shorter windows**: Change to `1.0` for faster processing but less accurate speaker identification
- **More overlap**: Decrease `DIARIZATION_SHIFT_LENGTH` to `0.5` for smoother speaker transitions
- **Adjust complexity penalty**: Edit line ~621 in `modal_app/app.py` (higher = prefers fewer speakers)
- **Disable entirely**: Set `ENABLE_SPEAKER_DIARIZATION = False` in `config.py` to skip diarization

After changing Modal settings, redeploy: `py -m modal deploy modal_app/app.py`

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

### Meeting Minutes Generation

The service includes **AI-powered meeting minutes generation** using Anthropic's Claude Haiku 4.5 API. This feature analyzes transcriptions and extracts structured information including summaries, key points, decisions, and action items.

**How it works:**
1. **Transcription completes**: After audio is fully transcribed
2. **API call**: Claude Haiku 4.5 analyzes the full transcription text
3. **Date awareness**: Current date is injected into the prompt for relative date calculation
4. **Structured output**: Returns JSON with executive summary, key points, decisions, action items, and participants
5. **Frontend display**: Minutes tab shows organized meeting information
6. **Download**: Export minutes as formatted TXT file

**Configuration** (in `modal_app/app.py` - embedded config at top of file):
```python
ANTHROPIC_MODEL_ID = "claude-haiku-4-5-20251001"  # Claude Haiku 4.5
MINUTES_MAX_INPUT_TOKENS = 8000  # Max transcription length
MINUTES_MAX_OUTPUT_TOKENS = 2048  # Max response length
MINUTES_TEMPERATURE = 0.3  # Low for consistent structured output
MINUTES_CONTAINER_IDLE_TIMEOUT = 60  # Container warm timeout
```

**Feature flag** (in `config.py` - API layer):
```python
ENABLE_MEETING_MINUTES = True  # Toggle feature on/off
```

**Modal Secret Setup** (required):
```bash
# Create the Anthropic API key secret in Modal
py -m modal secret create anthropic-api-key ANTHROPIC_API_KEY=sk-ant-...
```

**Output structure:**
```json
{
  "executive_summary": "2-3 sentence summary",
  "key_discussion_points": ["Point 1", "Point 2"],
  "decisions_made": ["Decision 1", "Decision 2"],
  "action_items": [
    {"task": "Task description", "assignee": "Person", "deadline": "DD/MM/YYYY"}
  ],
  "participants_mentioned": ["Name 1", "Name 2"]
}
```

**Key features:**
- **Relative date calculation**: "mañana", "la próxima semana" are converted to actual dates
- **Spanish language**: Prompts and output are in Spanish
- **No GPU required**: Uses Anthropic API (runs on CPU-only Modal container)
- **Fast**: Typical response time 2-5 seconds
- **Graceful degradation**: If minutes generation fails, transcription still succeeds

**Troubleshooting:**
- **"anthropic-api-key not found"**: Create the Modal secret with your API key
- **"model not found"**: Verify `ANTHROPIC_MODEL_ID` is correct
- **Empty minutes**: Check Modal logs for API errors

### Image Generation

The service includes **AI-powered image generation** using Black Forest Labs' FLUX.1-schnell model. This feature generates images from text prompts with fast inference times.

**How it works:**
1. **User submits prompt**: Text description of desired image (max 500 characters)
2. **Validation**: Prompt length and image dimensions are validated (512-1024px)
3. **GPU processing**: FLUX.1-schnell generates image on Modal GPU (L4)
4. **Caching**: Generated image is cached server-side with session ID
5. **Retrieval**: Frontend fetches image using session ID
6. **Display**: Image shown in dedicated tab with download option

**Configuration** (in `modal_app/app.py` - embedded config at top of file):
```python
IMAGE_GENERATION_MODEL = "black-forest-labs/FLUX.1-schnell"
IMAGE_GPU_TYPE = "L4"  # GPU type for generation
IMAGE_MEMORY_MB = 16384  # 16GB memory allocation
IMAGE_CONTAINER_IDLE_TIMEOUT = 120  # Keep container warm (seconds)
```

**Feature flag and API settings** (in `config.py` - API layer):
```python
ENABLE_IMAGE_GENERATION = True  # Toggle feature on/off
IMAGE_MAX_PROMPT_LENGTH = 500  # Maximum prompt length
IMAGE_DEFAULT_WIDTH = 768  # Default image width
IMAGE_DEFAULT_HEIGHT = 768  # Default image height
IMAGE_CACHE_EXPIRY_HOURS = 1  # Image cache expiry
```

**Modal Secret Setup** (required):
```bash
# Create the HuggingFace token secret in Modal (for gated model access)
py -m modal secret create hf-token HF_TOKEN=hf_...
```

**API endpoints:**
- `POST /api/generate-image`: Generate image from prompt
  - Parameters: `prompt` (required), `width` (512-1024), `height` (512-1024)
  - Returns: `ImageGenerationResponse` with `image_session_id`
- `GET /api/image/{session_id}`: Retrieve generated image as PNG

**Key features:**
- **Fast inference**: FLUX.1-schnell optimized for 4-step generation (~3-5 seconds)
- **Memory efficient**: Sequential CPU offload reduces VRAM usage
- **Flexible dimensions**: 512-1024px width/height (multiple of 8 recommended)
- **Session-based**: Images cached for 1 hour, retrieved via session ID
- **Graceful errors**: Clear error messages for invalid prompts or generation failures

**Performance considerations:**
- Cold start: ~30-60 seconds to download model and initialize
- Warm container: ~3-5 seconds per image generation
- Memory: Uses ~12-14GB VRAM with CPU offload enabled
- Model size: ~12GB (cached in Modal volume at `/models`)

**Troubleshooting:**
- **"hf-token not found"**: Create the Modal secret with your HuggingFace token
- **"Image generation service unavailable"**: Ensure Modal app is deployed
- **Out of memory**: Reduce image dimensions or enable CPU offload (already enabled by default)
- **Slow generation**: First request triggers cold start; subsequent requests faster

### Saved Voices (Voice Cloning)

The service supports **persistent voice storage** for voice cloning. Users can save reference audio + transcription as a "voice profile" and reuse it later without re-uploading the audio.

**How it works:**
1. **Create voice profile**: User uploads reference audio (3-60s) + transcription + name
2. **Save to Modal Volume**: Audio and metadata stored persistently in `/models/voices/`
3. **List saved voices**: Frontend displays all saved voices with name, language, and preview
4. **Synthesize with saved voice**: User selects voice + enters target text → generates audio
5. **Delete voice**: Remove voice profile from storage

**Storage Structure** (Modal Volume at `/models/voices/`):
```
/voices/
  ├── index.json              # List of all voices (id, name, language, created_at)
  └── {voice_id}/
      ├── ref_audio.wav       # Reference audio (24kHz mono WAV)
      └── metadata.json       # Full metadata (ref_text, language, etc.)
```

**API Endpoints:**
- `GET /api/voices`: List all saved voices
  - Returns: `SavedVoiceListResponse` with array of voices
- `POST /api/voices`: Save a new voice
  - Parameters: `name`, `ref_audio` (file), `ref_text`, `language`
  - Returns: `SaveVoiceResponse` with `voice_id`
- `DELETE /api/voices/{voice_id}`: Delete a saved voice
  - Returns: `{ success: true }`
- `POST /api/synthesize`: Synthesize audio with a saved voice
  - Parameters: `voice_id`, `target_text`
  - Returns: `SynthesizeResponse` with `audio_session_id`

**Configuration** (in `modal_app/app.py` - embedded config at top of file):
```python
VOICES_INDEX_FILE = "index.json"
MAX_SAVED_VOICES = 50  # Maximum number of saved voices
TTS_CONTAINER_IDLE_TIMEOUT = 120  # Container warm timeout
```

**Key features:**
- **Persistent storage**: Voices survive container restarts and redeploys
- **Shared access**: All users see the same voices (no authentication)
- **Quick synthesis**: No need to re-upload reference audio each time
- **Metadata preserved**: Language, transcription text stored with voice

**Frontend UI:**
- Voice Clone section has two tabs: "Voces Guardadas" and "Nueva Voz"
- Saved voices displayed as selectable cards with name, language, date
- "Guardar Voz" button saves current voice profile after entering a name
- Delete button removes voice from storage

**Troubleshooting:**
- **Voices not loading**: Ensure Modal app is deployed with VoiceStorage class
- **Save failed**: Check if MAX_SAVED_VOICES limit reached (default 50)
- **Duplicate name error**: Voice names must be unique (case-insensitive)

### Cost Optimization

GPU costs are ~$0.006 per minute of audio for transcription, ~$0.01-0.02 per image generation. Optimization strategies:
- **Parakeet TDT 0.6B** is much smaller than previous models (0.6B vs 2.6B params) = faster & cheaper
- `MODAL_CONTAINER_IDLE_TIMEOUT` keeps containers warm to reduce cold starts (20-30s)
- Audio preprocessing happens locally to minimize GPU time
- **Memory snapshots** are enabled by default for 85-90% faster cold starts (see `ENABLE_GPU_MEMORY_SNAPSHOT` in `modal_app/app.py`)
- **FLUX.1-schnell** uses only 4 inference steps (vs 20-50 for other models) = faster & cheaper
- Image generation uses sequential CPU offload to reduce VRAM requirements

## Configuration Changes

**Important**: Modal-specific settings are embedded directly in `modal_app/app.py` (not imported from config.py) to avoid import issues in Modal containers. When changing Modal settings, edit the constants at the top of `modal_app/app.py`.

### Changing the STT Model or GPU Type

Edit **`modal_app/app.py`** (top of file):
```python
STT_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"  # Or another NeMo-compatible model
MODAL_GPU_TYPE = "L4"    # Options: L4, A10G, T4
SAMPLE_RATE = 16000      # Must match model's native sample rate
```

Then redeploy Modal: `py -m modal deploy modal_app/app.py`

### Adjusting Silence Detection

Edit **`modal_app/app.py`** (top of file):
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
- **`anthropic>=0.40.0`**: Anthropic Claude API for meeting minutes generation
- **`diffusers>=0.30.0`**: HuggingFace diffusers for FLUX.1-schnell image generation
- **`accelerate>=0.30.0`**: Model acceleration and memory optimization

**Modal Container Image**:
- Base: `nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04`
- Python: 3.12
- GPU libraries: CUDA 12.8, cuDNN

## Troubleshooting

**"Modal service unavailable"**: The Modal app isn't deployed. Run `py -m modal deploy modal_app/app.py`.

**Slow first request**: Cold start takes 30-60s to download model and spin up GPU. Memory snapshots are enabled by default in `modal_app/app.py`:
```python
ENABLE_CPU_MEMORY_SNAPSHOT = True
ENABLE_GPU_MEMORY_SNAPSHOT = True  # Experimental, 85-90% faster cold starts
```

**Audio validation errors**: Check that FFmpeg is installed locally (`ffmpeg -version`). The API server needs FFmpeg to preprocess audio.

**GPU out of memory**: Parakeet TDT 0.6B needs ~3-4GB VRAM (much less than previous models). FLUX.1-schnell needs ~12-14GB VRAM with CPU offload. L4 (24GB VRAM) handles both comfortably.

**Too many/few segments in streaming**: Adjust silence detection parameters in `modal_app/app.py`:
- Too many segments: Increase `SILENCE_THRESHOLD_DB` and `SILENCE_MIN_LENGTH_MS`
- Too few segments: Decrease `SILENCE_THRESHOLD_DB` and `SILENCE_MIN_LENGTH_MS`

**NeMo verbose logs**: The `NoStdStreams` context manager in `modal_app/app.py` suppresses NeMo's stdout/stderr during transcription. If you need to debug, temporarily remove the `with NoStdStreams():` context.

**Speaker diarization not working**: Check these common issues:
- Ensure `ENABLE_SPEAKER_DIARIZATION = True` in `config.py` (API layer feature flag)
- Check Modal logs for diarization errors: `py -m modal app logs transcodio-app`
- Diarization requires at least ~5-10 seconds of audio with distinct speakers
- Single-speaker audio will show "Speaker 1" for all segments (this is expected)
- If diarization fails, transcription still completes (it's non-blocking)

**Audio player not loading**:
- Check that the audio session ID is being returned in the `complete` event
- Verify the `/api/audio/{session_id}` endpoint is accessible
- Audio cache expires after 1 hour - if playback fails, the session may have expired
- Check browser console for CORS or network errors

**Meeting minutes not generating**:
- Ensure Modal secret exists: `py -m modal secret list` should show `anthropic-api-key`
- Create the secret if missing: `py -m modal secret create anthropic-api-key ANTHROPIC_API_KEY=sk-ant-...`
- Check `ENABLE_MEETING_MINUTES = True` in `config.py` (API layer feature flag)
- Verify API key is valid and has credits
- Check Modal logs for API errors: `py -m modal app logs transcodio-app`

**Image generation not working**:
- Ensure Modal secret exists: `py -m modal secret list` should show `hf-token`
- Create the secret if missing: `py -m modal secret create hf-token HF_TOKEN=hf_...`
- Check `ENABLE_IMAGE_GENERATION = True` in `config.py` (API layer feature flag)
- Verify HuggingFace token has access to FLUX.1-schnell (may require accepting model terms)
- Check Modal logs for errors: `py -m modal app logs transcodio-app`
- First request may timeout due to cold start - retry after a few minutes

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
