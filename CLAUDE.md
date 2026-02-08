# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Transcodio is a production-ready AI platform deployed on Modal's serverless GPU infrastructure. It combines audio transcription, voice cloning, and image generation into a unified web service with real-time streaming.

Key features:
- **Real progressive streaming** transcription with silence detection (yields segments as they complete)
- **Speaker diarization** using NVIDIA TitaNet for automatic speaker identification
- **Meeting minutes** generation using Anthropic Claude Haiku 4.5
- **Voice cloning** using Qwen3-TTS with saved voice profiles and 10 language support
- **Saved voices** with persistent storage in Modal Volume
- **Image generation** using FLUX.1-schnell model for text-to-image
- **Audio playback** with integrated player using session-based caching
- GPU-accelerated transcription using NVIDIA Parakeet TDT 0.6B v3 on L4 GPUs
- FastAPI web server with REST endpoints and SSE streaming
- Audio preprocessing and validation using FFmpeg
- Support for multiple audio formats (MP3, WAV, M4A, FLAC, OGG, WebM, MP4)
- Subtitle export (SRT/VTT formats) with speaker labels
- **Internationalization (i18n)**: English/Spanish UI with language toggle, persisted via localStorage
- **Security hardening**: API key authentication, rate limiting, CSP headers, path traversal protection, XSS prevention

## Architecture

The application has a three-tier architecture:

1. **Frontend (static/)**: Browser-based UI with three modes — Transcription, Voice Cloning, Image Generation. Supports English (default) and Spanish via i18n system
2. **API Layer (api/)**: FastAPI application handling uploads, validation, SSE streaming, and session caching
3. **GPU Backend (modal_app/)**: Modal serverless functions with 6 classes across 4 container images

Data flow:
```
User uploads audio → FastAPI validates/preprocesses → Modal GPU processes → Results streamed back via SSE
```

### Key Components

**modal_app/app.py** — 6 Modal classes across 4 container images:

| Class | Image | GPU | Timeout | Purpose |
|-------|-------|-----|---------|---------|
| `ParakeetSTTModel` | `stt_image` (CUDA 12.8 + NeMo) | L4 | 3000s | Transcription (streaming + non-streaming) |
| `SpeakerDiarizerModel` | `stt_image` | L4 | 3000s | Speaker identification with TitaNet |
| `VoiceStorage` | `debian_slim` (no GPU) | — | 60s | Persistent voice profile management |
| `Qwen3TTSVoiceCloner` | `qwen_tts_image` (CUDA 12.8 + qwen-tts) | L4 | 300s | Voice cloning and synthesis |
| `FluxImageGenerator` | `flux_image` (CUDA 12.8 + diffusers) | L4 | 600s | Text-to-image generation |
| `MeetingMinutesGenerator` | `anthropic_image` (debian_slim) | — | 120s | Meeting minutes via Claude API |

Key implementation details:
- **Embedded configuration**: All Modal-specific settings are embedded directly at the top of the file (not imported from config.py) to avoid import issues in Modal containers
- **4 container images**: `stt_image` (NeMo + CUDA), `anthropic_image` (lightweight), `flux_image` (diffusers + CUDA), `qwen_tts_image` (qwen-tts + CUDA)
- **ParakeetSTTModel**: Uses NeMo framework for ASR. Two methods: `transcribe()` and `transcribe_stream()`. Real streaming detects silence boundaries using pydub.
- **SpeakerDiarizerModel**: Single-scale embedding extraction (1.5s windows), AgglomerativeClustering with cosine distance, complexity penalty (0.15/speaker)
- **VoiceStorage**: Manages voice profiles on Modal Volume at `/models/voices/`. Methods: `list_voices()`, `get_voice()`, `save_voice()`, `delete_voice()`. All methods validate `voice_id` as UUID format and verify resolved paths stay within the voices directory (path traversal protection).
- **Qwen3TTSVoiceCloner**: Loads `Qwen/Qwen3-TTS-12Hz-1.7B-Base` with bfloat16 on CUDA. Method: `generate_voice_clone(ref_audio_bytes, ref_text, target_text, language)`
- **FluxImageGenerator**: FLUX.1-schnell with sequential CPU offload. Requires `hf-token` Modal secret. Method: `generate_image(prompt, width, height, num_inference_steps, guidance_scale)`
- **MeetingMinutesGenerator**: Claude Haiku 4.5 API. Requires `anthropic-api-key` Modal secret. Spanish-language prompts with date awareness. Method: `generate_minutes(transcription, speakers)`
- **NoStdStreams** context manager suppresses NeMo's verbose logging
- **`align_speakers_to_segments()`** function maps speaker labels to transcription segments based on maximum temporal overlap
- Uses Modal Volume at `/models` to cache all models (Parakeet, TitaNet, Qwen3-TTS, FLUX.1-schnell)
- CPU and GPU memory snapshots enabled by default for faster cold starts

**api/main.py** — FastAPI application with all endpoints:

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/` | Serve web UI (index.html) |
| GET | `/health` | Health check |
| POST | `/api/transcribe` | Non-streaming transcription |
| POST | `/api/transcribe/stream` | Streaming transcription (SSE) with optional diarization & minutes |
| GET | `/api/audio/{session_id}` | Retrieve cached audio for playback |
| POST | `/api/voice-clone` | Clone voice and synthesize text (single-shot) |
| GET | `/api/voices` | List all saved voice profiles |
| POST | `/api/voices` | Save a new voice profile |
| DELETE | `/api/voices/{voice_id}` | Delete a saved voice |
| POST | `/api/synthesize` | Synthesize text with a saved voice |
| POST | `/api/generate-image` | Generate image from text prompt |
| GET | `/api/image/{session_id}` | Retrieve generated image as PNG |

Key implementation details:
- Connects to Modal using `modal.Cls.from_name()` to lookup deployed classes
- **Two in-memory caches**: `audio_cache` (audio sessions, 1hr expiry) and `image_cache` (images, 1hr expiry)
- Audio validation happens before sending to Modal to save GPU costs
- SSE streaming converts Modal's synchronous generator to async via `event_generator()`
- Speaker diarization runs after transcription completes, yields `speakers_ready` event
- Meeting minutes run after completion, yields `minutes_ready` or `minutes_error` event
- Voice clone endpoint supports model selection via `tts_model` param (currently only `qwen`)
- Reference audio preprocessing converts to 24kHz mono WAV for TTS (vs 16kHz for STT)
- **Security**: API key auth via `X-API-Key` header, rate limiting via `slowapi`, security response headers (CSP, X-Frame-Options, X-Content-Type-Options, Referrer-Policy), filename sanitization for Content-Disposition, UUID validation on all path parameters, generic error messages (no internal detail leakage)
- **CORS**: Restricted to `GET`, `POST`, `DELETE` methods and `Content-Type`, `X-API-Key` headers only

**api/models.py** — Pydantic response models:
- `TranscriptionResponse`, `TranscriptionSegment`, `TranscriptionStreamEvent`
- `VoiceCloneResponse`, `SavedVoice`, `SavedVoiceListResponse`, `SaveVoiceResponse`, `SynthesizeResponse`
- `ImageGenerationResponse`
- `ErrorResponse`, `HealthResponse`

**api/streaming.py** — SSE utilities:
- `create_sse_response()`: Wraps async generator as EventSourceResponse
- `format_sse_event()`: Formats data as SSE event string
- `transcription_event_stream()`: Converts Modal transcription segments into SSE events

**utils/audio.py**:
- Audio validation pipeline: file size → format → duration → preprocessing
- `validate_audio_file()`: Complete validation + preprocessing, accepts optional `target_sample_rate` param
- Uses FFmpeg to preprocess audio (convert to mono WAV at target sample rate)
- Default target: **16kHz** for STT, **24kHz** for TTS
- Handles browser-recorded WebM files with missing duration metadata (WAV conversion fallback)
- All validation happens locally before sending to Modal GPU

**config.py** — Configuration for the **API layer** (FastAPI server):
- **Note**: Modal-specific settings are duplicated/embedded in `modal_app/app.py` to avoid import issues in Modal containers. When changing Modal settings, update both files.
- Key settings:

```python
# Audio file limits
MAX_FILE_SIZE_MB = 100
MAX_DURATION_SECONDS = 3600  # 60 minutes
SUPPORTED_FORMATS = ["mp3", "wav", "m4a", "flac", "ogg", "webm", "mp4"]

# STT Model
STT_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
SAMPLE_RATE = 16000  # Parakeet's native sample rate

# Modal
MODAL_APP_NAME = "transcodio-app"
MODAL_VOLUME_NAME = "parakeet-models"
MODAL_GPU_TYPE = "L4"
MODAL_CONTAINER_IDLE_TIMEOUT = 120
MODAL_TIMEOUT = 3000  # 50 min max processing
MODAL_MEMORY_MB = 8192

# Cold start optimization
ENABLE_CPU_MEMORY_SNAPSHOT = True
ENABLE_GPU_MEMORY_SNAPSHOT = True  # 85-90% faster cold starts
ENABLE_MODEL_WARMUP = False
EXTENDED_IDLE_TIMEOUT = False

# Silence detection (streaming segmentation)
SILENCE_THRESHOLD_DB = -40
SILENCE_MIN_LENGTH_MS = 700

# Speaker diarization
ENABLE_SPEAKER_DIARIZATION = True
DIARIZATION_MODEL = "nvidia/speakerverification_en_titanet_large"
DIARIZATION_MIN_SPEAKERS = 1
DIARIZATION_MAX_SPEAKERS = 5
DIARIZATION_WINDOW_LENGTHS = [1.5, 1.0, 0.5]  # Only first value used (single-scale)
DIARIZATION_SHIFT_LENGTH = 0.75

# Meeting minutes (Anthropic Claude API)
ENABLE_MEETING_MINUTES = True
ANTHROPIC_MODEL_ID = "claude-haiku-4-5-20251001"
MINUTES_MAX_INPUT_TOKENS = 8000
MINUTES_MAX_OUTPUT_TOKENS = 2048
MINUTES_TEMPERATURE = 0.3
MINUTES_CONTAINER_IDLE_TIMEOUT = 60

# Voice cloning
ENABLE_VOICE_CLONING = True
TTS_MODELS = {
    "qwen": {
        "name": "Qwen3-TTS",
        "model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "sample_rate": 24000,
        "gpu_type": "L4",
        "memory_mb": 8192,
    },
}
DEFAULT_TTS_MODEL = "qwen"
TTS_CONTAINER_IDLE_TIMEOUT = 120
VOICE_CLONE_MIN_REF_DURATION = 3    # seconds
VOICE_CLONE_MAX_REF_DURATION = 300  # seconds (5 minutes)
VOICE_CLONE_MAX_TARGET_TEXT = 50000 # characters
VOICE_CLONE_SAMPLE_RATE = 24000
VOICE_CLONE_LANGUAGES = [
    "English", "Spanish", "Chinese", "Japanese", "Korean",
    "German", "French", "Russian", "Portuguese", "Italian"
]

# Saved voices
VOICES_STORAGE_PATH = "/voices"
VOICES_INDEX_FILE = "index.json"
MAX_SAVED_VOICES = 50

# Image generation (FLUX.1-schnell)
ENABLE_IMAGE_GENERATION = True
IMAGE_GENERATION_MODEL = "black-forest-labs/FLUX.1-schnell"
IMAGE_GPU_TYPE = "L4"
IMAGE_MEMORY_MB = 16384
IMAGE_CONTAINER_IDLE_TIMEOUT = 120
IMAGE_MAX_PROMPT_LENGTH = 500
IMAGE_DEFAULT_WIDTH = 768
IMAGE_DEFAULT_HEIGHT = 768
IMAGE_NUM_INFERENCE_STEPS = 4
IMAGE_GUIDANCE_SCALE = 0.0
IMAGE_CACHE_EXPIRY_HOURS = 1

# Security
API_KEY = os.getenv("TRANSCODIO_API_KEY", "")  # Set in production; empty = no auth (dev only)
RATE_LIMIT_TRANSCRIBE = "5/minute"
RATE_LIMIT_VOICE_CLONE = "10/minute"
RATE_LIMIT_IMAGE = "10/minute"
RATE_LIMIT_DEFAULT = "30/minute"
SAFE_AUDIO_MIME_TYPES = {
    "mp3": "audio/mpeg", "wav": "audio/wav", "m4a": "audio/mp4",
    "flac": "audio/flac", "ogg": "audio/ogg", "webm": "audio/webm", "mp4": "video/mp4",
}
```

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
# Note: Add -H "X-API-Key: YOUR_KEY" if TRANSCODIO_API_KEY is set
curl -X POST "http://localhost:8000/api/transcribe" -F "file=@audio.mp3"

# Test streaming endpoint
curl -X POST "http://localhost:8000/api/transcribe/stream" -F "file=@audio.mp3"

# Test voice cloning
curl -X POST "http://localhost:8000/api/voice-clone" \
  -F "ref_audio=@reference.wav" \
  -F "ref_text=Hello this is my voice" \
  -F "target_text=Text to synthesize" \
  -F "language=English"

# Test saved voices
curl "http://localhost:8000/api/voices"

# Test image generation
curl -X POST "http://localhost:8000/api/generate-image" \
  -F "prompt=a beautiful sunset over mountains" \
  -F "width=768" -F "height=768"

# Test with API key authentication (if TRANSCODIO_API_KEY is set)
curl -H "X-API-Key: your-api-key" "http://localhost:8000/api/voices"
```

## Critical Implementation Details

### Modal Integration

The FastAPI server connects to Modal using the deployed app name:
```python
STTModel = modal.Cls.from_name(config.MODAL_APP_NAME, "ParakeetSTTModel")
model = STTModel()
result = model.transcribe.remote(audio_bytes)
```

All 6 classes are looked up the same way:
```python
modal.Cls.from_name("transcodio-app", "ParakeetSTTModel")
modal.Cls.from_name("transcodio-app", "SpeakerDiarizerModel")
modal.Cls.from_name("transcodio-app", "VoiceStorage")
modal.Cls.from_name("transcodio-app", "Qwen3TTSVoiceCloner")
modal.Cls.from_name("transcodio-app", "FluxImageGenerator")
modal.Cls.from_name("transcodio-app", "MeetingMinutesGenerator")
```

The Modal app MUST be deployed before running the FastAPI server, otherwise you'll get "Modal service unavailable" errors.

### Audio Preprocessing

All audio goes through preprocessing before transcription or TTS:
1. **Validation**: Check file size, format, and duration (utils/audio.py)
2. **Conversion**: FFmpeg converts to mono WAV at target sample rate
   - **16kHz** for STT (Parakeet TDT's native format)
   - **24kHz** for TTS (Qwen3-TTS's native format)
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
   - Handles diarization and minutes generation after transcription completes

3. **Browser receives SSE events**:
   - `metadata`: Audio duration and language
   - `progress`: Each transcribed segment (yields multiple times)
   - `speakers_ready`: Speaker-annotated segments after diarization completes (optional)
   - `minutes_ready`: Structured meeting minutes (optional)
   - `minutes_error`: Minutes generation failed (non-fatal)
   - `complete`: Transcription finished (includes full transcription text and audio session ID)
   - `error`: Any errors during processing

**Complete Streaming Flow**:
```
1. User uploads audio → FastAPI validates & preprocesses
2. Generate session ID, cache original audio
3. Stream to Modal GPU → Parakeet transcribes with silence detection
4. Frontend receives progressive segments via SSE (metadata → progress events)
5. After transcription completes → TitaNet diarization runs (if enabled)
6. Speaker labels aligned to segments → speakers_ready event updates UI
7. Final complete event with full text & audio session ID
8. Meeting minutes generated → minutes_ready event (if enabled)
9. Audio player loads using session ID from cache
10. User can download transcription (TXT), subtitles (SRT/VTT), or listen to audio
```

This architecture ensures:
- **Progressive feedback**: Users see segments as they're transcribed (not blocked by diarization)
- **Non-blocking diarization**: Speaker identification happens after transcription
- **Non-blocking minutes**: Meeting minutes generated after completion event
- **Graceful degradation**: If diarization or minutes fail, transcription still succeeds
- **Audio playback**: Original audio available for listening without re-uploading

### Silence Detection Tuning

Adjust these parameters in `modal_app/app.py` (embedded config at top of file) to control segmentation granularity:

```python
SILENCE_THRESHOLD_DB = -40   # Lower = more sensitive (detects softer pauses)
SILENCE_MIN_LENGTH_MS = 700  # Lower = detects shorter pauses
```

**Guidelines**:
- **Fewer segments** (longer): Increase threshold to -45, increase min_length to 1000ms
- **More segments** (shorter): Decrease threshold to -35, decrease min_length to 400ms
- **Balanced** (3-5 segments per minute): -40 dB, 700ms (current default)

After changing these values, redeploy: `py -m modal deploy modal_app/app.py`

### Speaker Diarization

**How it works:**
1. **Single-scale embedding extraction**: Audio analyzed using 1.5s windows with 0.75s overlap
2. **Speaker embedding**: TitaNet generates normalized speaker embeddings per window
3. **Automatic speaker detection**: Combined scoring with complexity penalty (1-5 speakers)
4. **Clustering**: AgglomerativeClustering with cosine distance
5. **Alignment**: Speaker labels mapped to transcription segments via temporal overlap
6. **Frontend display**: Segments show speaker badges (e.g., "Speaker 1")

**Algorithm details:**
- **Clustering**: AgglomerativeClustering with cosine distance + average linkage
- **Speaker selection**: Combined score = 60% silhouette + 40% Calinski-Harabasz (normalized) - complexity penalty
- **Complexity penalty**: 0.15 per additional speaker (prefers fewer speakers per Occam's Razor)
- **Minimum segments**: Needs ≥10 embedding windows to attempt clustering; otherwise defaults to 1 speaker
- **Single-scale**: Uses only first window length (1.5s) to avoid multi-scale artifacts

**Configuration** (in `modal_app/app.py`):
```python
ENABLE_SPEAKER_DIARIZATION = True
DIARIZATION_MODEL = "nvidia/speakerverification_en_titanet_large"
DIARIZATION_MIN_SPEAKERS = 1
DIARIZATION_MAX_SPEAKERS = 5
DIARIZATION_WINDOW_LENGTHS = [1.5, 1.0, 0.5]  # Only first value used
DIARIZATION_SHIFT_LENGTH = 0.75
```

**Performance**: ~1-3 seconds for 1-minute audio. Runs after transcription (non-blocking). Graceful degradation on failure.

After changing settings, redeploy: `py -m modal deploy modal_app/app.py`

### Meeting Minutes Generation

Uses Anthropic Claude Haiku 4.5 API to generate structured meeting minutes in Spanish.

**How it works:**
1. Transcription completes → full text sent to Claude Haiku 4.5
2. Current date injected into prompt for relative date calculation
3. Returns JSON with executive summary, key points, decisions, action items, participants
4. Frontend displays in Minutes tab with download option

**Configuration** (in `modal_app/app.py`):
```python
ANTHROPIC_MODEL_ID = "claude-haiku-4-5-20251001"
MINUTES_MAX_INPUT_TOKENS = 8000
MINUTES_MAX_OUTPUT_TOKENS = 2048
MINUTES_TEMPERATURE = 0.3
MINUTES_CONTAINER_IDLE_TIMEOUT = 60
```

**Feature flag** (in `config.py`): `ENABLE_MEETING_MINUTES = True`

**Modal Secret** (required):
```bash
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
- **Relative date calculation**: "mañana", "la próxima semana" → actual dates
- **Spanish language**: Prompts and output in Spanish
- **No GPU required**: Runs on CPU-only Modal container
- **JSON parsing**: Handles markdown code blocks and malformed JSON gracefully

### Voice Cloning

Uses Qwen3-TTS (1.7B params, 24kHz) for voice cloning with zero-shot voice transfer.

**How it works:**
1. User uploads reference audio (3s–5min) + transcription text + target text
2. FastAPI validates and preprocesses reference audio to 24kHz mono WAV
3. Qwen3TTSVoiceCloner on Modal GPU generates audio with cloned voice
4. Generated audio cached with session ID for playback/download

**Two modes in the API:**
- **Single-shot** (`POST /api/voice-clone`): Upload reference + generate in one call. Supports `tts_model` param.
- **Saved voice** (`POST /api/synthesize`): Use a previously saved voice profile by `voice_id`.

**Configuration** (in `config.py`):
```python
ENABLE_VOICE_CLONING = True
VOICE_CLONE_MIN_REF_DURATION = 3    # seconds
VOICE_CLONE_MAX_REF_DURATION = 300  # seconds (5 minutes)
VOICE_CLONE_MAX_TARGET_TEXT = 50000 # characters
VOICE_CLONE_SAMPLE_RATE = 24000
VOICE_CLONE_LANGUAGES = [
    "English", "Spanish", "Chinese", "Japanese", "Korean",
    "German", "French", "Russian", "Portuguese", "Italian"
]
```

**Validation rules:**
- Reference audio: 3s–300s duration, max 15MB file size
- Target text: 1–50,000 characters
- Reference text: cannot be empty
- Language: must be in `VOICE_CLONE_LANGUAGES` list

### Saved Voices

Persistent voice storage on Modal Volume. Users save reference audio + transcription as a "voice profile" for reuse.

**Storage Structure** (Modal Volume at `/models/voices/`):
```
/voices/
  ├── index.json              # List of all voices (id, name, language, ref_text, created_at)
  └── {voice_id}/
      ├── ref_audio.wav       # Reference audio (24kHz mono WAV)
      └── metadata.json       # Full metadata (ref_text, language, etc.)
```

**API Endpoints:**
- `GET /api/voices`: List all saved voices
- `POST /api/voices`: Save new voice (params: `name`, `ref_audio`, `ref_text`, `language`)
- `DELETE /api/voices/{voice_id}`: Delete a saved voice
- `POST /api/synthesize`: Synthesize with saved voice (params: `voice_id`, `target_text`)

**Constraints:**
- Voice names: max 50 characters, must be unique (case-insensitive)
- Max saved voices: 50 (configurable via `MAX_SAVED_VOICES`)
- Volume commits after every write operation

**Frontend UI:**
- Voice Clone section has two tabs: "Saved Voices" and "New Voice" (translated via i18n)
- Saved voices displayed as selectable cards with name, language, date
- "Save Voice" button saves current voice profile
- Delete button removes voice from storage

### Image Generation

Uses FLUX.1-schnell for fast text-to-image generation.

**How it works:**
1. User submits text prompt (max 500 chars) + dimensions (512–1024px)
2. FLUX.1-schnell generates image on Modal L4 GPU with 4 inference steps
3. Image cached as PNG with session ID
4. Frontend retrieves and displays image

**Configuration** (in `modal_app/app.py`):
```python
IMAGE_GENERATION_MODEL = "black-forest-labs/FLUX.1-schnell"
IMAGE_GPU_TYPE = "L4"
IMAGE_MEMORY_MB = 16384  # 16GB
IMAGE_CONTAINER_IDLE_TIMEOUT = 120
```

**API settings** (in `config.py`):
```python
ENABLE_IMAGE_GENERATION = True
IMAGE_MAX_PROMPT_LENGTH = 500
IMAGE_DEFAULT_WIDTH = 768
IMAGE_DEFAULT_HEIGHT = 768
IMAGE_NUM_INFERENCE_STEPS = 4
IMAGE_GUIDANCE_SCALE = 0.0  # Required for schnell mode
IMAGE_CACHE_EXPIRY_HOURS = 1
```

**Modal Secret** (required):
```bash
py -m modal secret create hf-token HF_TOKEN=hf_...
```

**Performance:**
- Cold start: ~30-60s to download model
- Warm container: ~3-5s per image
- Memory: ~12-14GB VRAM with CPU offload
- Model: ~12GB (cached in Modal volume)
- `max_sequence_length=128` used to save memory

### Audio Player

Integrated audio player for listening to uploaded audio alongside transcription.

**Implementation:**
- Each transcription generates UUID session ID
- Original uploaded audio cached in-memory for 1 hour (`audio_cache` dict)
- `/api/audio/{session_id}` serves cached audio in original format
- Automatic cleanup of expired entries
- Voice clone output also cached via same mechanism (`audio/wav` content type)

**Note**: In-memory cache suitable for low-traffic. For production, consider Redis or persistent store.

### Internationalization (i18n)

Simple key-based i18n system with no external dependencies. Default language: English.

**Implementation** (in `static/app.js`):
- `translations` object at top of file with `en` and `es` keys containing ~100 translation strings each
- `t(key)` function returns the translated string for the current language
- `applyTranslations()` updates all DOM elements with `data-i18n` attributes
- `setLanguage(lang)` switches language, persists to `localStorage`, and re-renders
- Language preference stored in `localStorage` as `transcodio-lang`

**HTML attributes** (in `static/index.html`):
- `data-i18n="key"` — Translates element's `textContent`
- `data-i18n-placeholder="key"` — Translates element's `placeholder` attribute
- `data-i18n-alt="key"` — Translates element's `alt` attribute
- `data-i18n-title="key"` — Translates element's `title` attribute

**Language toggle**: `<button class="lang-toggle" id="langToggle">` in header, styled via `.lang-toggle` in `styles.css`. Displays "ES" when English is active (click to switch to Spanish) and "EN" when Spanish is active.

**Dynamic strings**: All `showToast()` messages, `confirm()` dialogs, error messages, status text, and dynamically generated HTML use `t('key')` calls instead of hardcoded strings.

**Adding a new language**:
1. Add a new key (e.g., `fr`) to the `translations` object in `app.js` with all ~100 keys translated
2. Update the language toggle logic in `setLanguage()` and `applyTranslations()` to cycle through languages or use a dropdown
3. Update `formatDate()` to include the new locale

**Translation key naming convention**: `section.descriptor` (e.g., `voice.save`, `toast.downloaded`, `minutes.title`)

### Security

The application implements multiple layers of security hardening:

**API Key Authentication** (`api/main.py`):
- Configured via `TRANSCODIO_API_KEY` environment variable (in `config.py`)
- When set, all `/api/*` endpoints require `X-API-Key` header
- When empty (default for dev), authentication is disabled
- Public routes (`/`, `/health`, `/static/*`) are always exempt
- Function: `verify_api_key(request)` called at the start of every API endpoint

**Rate Limiting** (`api/main.py`):
- Uses `slowapi` library with per-IP rate limits
- GPU-intensive endpoints: 5/min (transcribe), 10/min (voice-clone, image)
- Read endpoints: 30/min (list voices, etc.)
- Returns HTTP 429 when exceeded
- Configured in `config.py`: `RATE_LIMIT_TRANSCRIBE`, `RATE_LIMIT_VOICE_CLONE`, `RATE_LIMIT_IMAGE`, `RATE_LIMIT_DEFAULT`

**Security Response Headers** (`SecurityHeadersMiddleware` in `api/main.py`):
- `Content-Security-Policy`: `default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'`
- `X-Frame-Options: DENY` (clickjacking protection)
- `X-Content-Type-Options: nosniff` (MIME sniffing protection)
- `Referrer-Policy: strict-origin-when-cross-origin`

**Path Traversal Protection** (`modal_app/app.py` VoiceStorage class):
- All `voice_id` parameters validated as UUID format via regex (`_validate_voice_id()`)
- Resolved paths verified to stay within `/models/voices/` directory (`is_relative_to()`)
- Prevents attacks like `DELETE /api/voices/../../` that could delete model weights
- UUID validation also enforced in `api/main.py` via `_validate_uuid()` for all path parameters

**XSS Prevention** (`static/app.js`):
- Meeting minutes action items use `textContent` and DOM element creation (not `innerHTML`) to render server data
- Saved voices error handler uses `textContent` + `addEventListener` (not `innerHTML` + inline `onclick`)
- All server-provided values in saved voices rendering are passed through `escapeHtml()` (including `voice.id`, `voice.language`)
- `showToast()` uses `textContent` (safe by default)

**Content-Type / Filename Sanitization** (`api/main.py`):
- `_sanitize_filename()`: Strips path components, control characters, quotes, semicolons, newlines; limits to 100 chars
- `_safe_content_type()`: Maps file extension to hardcoded safe MIME type (from `SAFE_AUDIO_MIME_TYPES` config) — never trusts user-supplied `Content-Type`
- `Content-Disposition` headers use proper RFC 6266 quoting: `filename="sanitized_name.ext"`

**Error Message Hardening** (`api/main.py`):
- All error responses return generic messages (e.g., "An unexpected error occurred.") — no `str(e)` leakage
- Internal details (Modal class names, stack traces, file paths) are not exposed to clients
- Exceptions are logged server-side only

**CORS** (`api/main.py`):
- `allow_methods` restricted to `["GET", "POST", "DELETE"]` (not `["*"]`)
- `allow_headers` restricted to `["Content-Type", "X-API-Key"]` (not `["*"]`)
- Origins restricted to localhost variants (update for production)

### Cost Optimization

| Feature | Cost |
|---------|------|
| Transcription | ~$0.006 per minute of audio |
| Speaker Diarization | Included (same GPU) |
| Meeting Minutes | ~$0.001 per request (Haiku API) |
| Voice Cloning | ~$0.01-0.02 per synthesis |
| Image Generation | ~$0.01-0.02 per image |

Strategies:
- **Parakeet TDT 0.6B**: 4.3x smaller than previous model = faster & cheaper
- `MODAL_CONTAINER_IDLE_TIMEOUT` keeps containers warm (default 120s)
- Audio preprocessing happens locally to minimize GPU time
- **GPU memory snapshots** enabled for 85-90% faster cold starts
- **FLUX.1-schnell** uses only 4 inference steps (vs 20-50 for other models)
- Sequential CPU offload for image generation reduces VRAM requirements

## Configuration Changes

**Important**: Modal-specific settings are embedded directly in `modal_app/app.py` (not imported from config.py) to avoid import issues in Modal containers. When changing Modal settings, edit the constants at the top of `modal_app/app.py` AND update `config.py` for API-layer consistency.

### Changing the STT Model or GPU Type

Edit **`modal_app/app.py`** (top of file):
```python
STT_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"  # Or another NeMo-compatible model
MODAL_GPU_TYPE = "L4"    # Options: L4, A10G, T4
SAMPLE_RATE = 16000      # Must match model's native sample rate
```

Then redeploy: `py -m modal deploy modal_app/app.py`

### Adjusting Silence Detection

Edit **`modal_app/app.py`** (top of file):
```python
SILENCE_THRESHOLD_DB = -40    # -50 (very conservative) to -30 (very sensitive)
SILENCE_MIN_LENGTH_MS = 700   # 300ms (granular) to 1500ms (conservative)
```

Redeploy: `py -m modal deploy modal_app/app.py`

### Adding a New TTS Model

1. Add entry to `TTS_MODELS` dict in both `config.py` and `modal_app/app.py`
2. Create a new container image with required dependencies
3. Create a new Modal class with `generate_voice_clone()` method
4. Add model selection logic in `api/main.py` voice-clone endpoint
5. Redeploy: `py -m modal deploy modal_app/app.py`

## Dependencies

**System requirements**:
- Python 3.12+ (Modal containers use Python 3.12)
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer
- FFmpeg (must be installed locally for audio processing)
- Modal account (free tier available)

**Local Python packages** (requirements.txt):
- `modal>=1.3.0`: Serverless GPU infrastructure
- `fastapi>=0.128.0` + `uvicorn>=0.40.0`: Web framework and ASGI server
- `nemo_toolkit[asr]==2.3.0`: NVIDIA NeMo for ASR (includes PyTorch & torchaudio)
- `numpy>=1.24.0,<2`: Required for NeMo compatibility
- `hf_transfer==0.1.9` + `huggingface-hub>=0.36.0`: Fast model downloads
- `ffmpeg-python>=0.2.0`: Audio preprocessing
- `pydub>=0.25.1`: Silence detection for streaming
- `python-multipart>=0.0.21`: File upload handling
- `sse-starlette>=3.1.2`: Server-Sent Events
- `aiofiles>=25.1.0`: Async file I/O
- `anthropic>=0.40.0`: Claude API for meeting minutes
- `slowapi>=0.1.9`: Rate limiting for FastAPI
- `pydantic>=2.12.0`: Request/response models
- `python-dotenv>=1.2.0`: Environment variable management

**Modal Container packages** (installed only in containers, not locally):
- `stt_image`: nemo_toolkit, cuda-python, pydub, scikit-learn, soundfile
- `qwen_tts_image`: qwen-tts, torch, transformers, soundfile
- `flux_image`: torch, diffusers, transformers, accelerate, sentencepiece, protobuf
- `anthropic_image`: anthropic

**Modal Container Images**:
- **stt_image**: `nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04` + Python 3.12 + NeMo + FFmpeg
- **qwen_tts_image**: `nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04` + Python 3.12 + qwen-tts + FFmpeg + libsndfile
- **flux_image**: `nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04` + Python 3.12 + diffusers
- **anthropic_image**: `debian_slim` + Python 3.12 + anthropic (no GPU)

## Troubleshooting

**"Invalid or missing API key" (401)**: Set the `X-API-Key` header in your request. The key must match the `TRANSCODIO_API_KEY` environment variable. If unset, authentication is disabled (dev mode).

**"Rate limit exceeded" (429)**: You've exceeded the per-IP rate limit. Default limits: 5/min for transcription, 10/min for voice-clone and image generation, 30/min for other endpoints. Configurable in `config.py`.

**"Invalid voice ID format" (400)**: The `voice_id` or `session_id` must be a valid UUID (e.g., `550e8400-e29b-41d4-a716-446655440000`). Path traversal attempts are blocked.

**"Modal service unavailable"**: The Modal app isn't deployed. Run `py -m modal deploy modal_app/app.py`.

**Slow first request**: Cold start takes 30-60s. Memory snapshots are enabled by default:
```python
ENABLE_CPU_MEMORY_SNAPSHOT = True
ENABLE_GPU_MEMORY_SNAPSHOT = True  # 85-90% faster cold starts
```

**Audio validation errors**: Check that FFmpeg is installed locally (`ffmpeg -version`). The API server needs FFmpeg to preprocess audio.

**GPU out of memory**: Parakeet TDT 0.6B needs ~3-4GB VRAM. FLUX.1-schnell needs ~12-14GB with CPU offload. Qwen3-TTS needs ~8GB. L4 (24GB VRAM) handles all comfortably.

**Too many/few segments in streaming**: Adjust silence detection in `modal_app/app.py`:
- Too many: Increase `SILENCE_THRESHOLD_DB` and `SILENCE_MIN_LENGTH_MS`
- Too few: Decrease both values

**NeMo verbose logs**: `NoStdStreams` context manager in `modal_app/app.py` suppresses stdout/stderr. To debug, temporarily remove `with NoStdStreams():`.

**Speaker diarization not working**:
- Check `ENABLE_SPEAKER_DIARIZATION = True` in `config.py`
- Check Modal logs: `py -m modal app logs transcodio-app`
- Needs ≥5-10s of audio with distinct speakers
- Single-speaker audio shows "Speaker 1" for all segments (expected)
- Non-blocking: if diarization fails, transcription still completes

**Audio player not loading**:
- Check audio session ID in `complete` event
- Verify `/api/audio/{session_id}` accessible
- Cache expires after 1 hour
- Check browser console for CORS errors

**Meeting minutes not generating**:
- Check Modal secret: `py -m modal secret list` should show `anthropic-api-key`
- Create if missing: `py -m modal secret create anthropic-api-key ANTHROPIC_API_KEY=sk-ant-...`
- Check `ENABLE_MEETING_MINUTES = True` in `config.py`
- Check Modal logs for API errors

**Image generation not working**:
- Check Modal secret: `py -m modal secret list` should show `hf-token`
- Create if missing: `py -m modal secret create hf-token HF_TOKEN=hf_...`
- Check `ENABLE_IMAGE_GENERATION = True` in `config.py`
- HuggingFace token needs access to FLUX.1-schnell (may require accepting model terms)
- First request may timeout due to cold start

**Voice cloning not working**:
- Check `ENABLE_VOICE_CLONING = True` in `config.py`
- Ensure Modal app deployed with `Qwen3TTSVoiceCloner` and `VoiceStorage` classes
- Reference audio must be 3s–300s, max 15MB
- Check Modal logs: `py -m modal app logs transcodio-app`

**Saved voices not loading**:
- Ensure `VoiceStorage` class is deployed
- Check if `MAX_SAVED_VOICES` limit reached (default 50)
- Voice names must be unique (case-insensitive)
- Modal Volume must be accessible at `/models`
