# Transcodio

**AI-powered audio transcription, voice cloning, and image generation — all in one service**

Transcodio is a production-ready platform powered by NVIDIA's Parakeet TDT 0.6B v3 model, deployed on Modal's serverless GPU infrastructure. It combines real-time streaming transcription, speaker diarization, AI meeting minutes, voice cloning with saved profiles, and text-to-image generation into a unified web application.

## Features

- **Real-time Streaming Transcription**: Progressive results via SSE with silence-based segmentation using NVIDIA Parakeet TDT 0.6B v3
- **Speaker Diarization**: Automatic speaker identification using NVIDIA TitaNet embeddings + AgglomerativeClustering
- **Meeting Minutes**: AI-powered summaries with action items using Anthropic Claude Haiku 4.5
- **Voice Cloning**: Clone any voice with Qwen3-TTS — upload or record reference audio (up to 5 minutes), then synthesize up to 50,000 characters of text
- **Saved Voice Profiles**: Persistently store voice profiles in Modal Volume for reuse without re-uploading
- **Image Generation**: Text-to-image using FLUX.1-schnell with 4-step inference (~3-5 seconds per image)
- **Audio Playback**: Integrated player to listen to uploaded audio alongside transcription
- **Multiple Formats**: Supports MP3, WAV, M4A, FLAC, OGG, WebM, MP4
- **Subtitle Export**: Download transcriptions as SRT/VTT with speaker labels
- **Cost Effective**: ~$0.006 per minute of audio transcription on NVIDIA L4 GPUs
- **10 Languages for TTS**: Spanish, English, Chinese, Japanese, Korean, German, French, Russian, Portuguese, Italian

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) — Fast Python package installer and runner
- [Modal](https://modal.com) account (free tier available)
- FFmpeg installed locally
- Anthropic API key (for meeting minutes)
- HuggingFace token (for image generation — FLUX.1-schnell is a gated model)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/dorianlgs/transcodio.git
cd transcodio
```

2. Install dependencies:
```bash
uv sync
```

3. Set up Modal authentication:
```bash
py -m modal setup
```

4. Create Modal secrets:
```bash
# Anthropic API key (required for meeting minutes)
py -m modal secret create anthropic-api-key ANTHROPIC_API_KEY=sk-ant-...

# HuggingFace token (required for image generation)
py -m modal secret create hf-token HF_TOKEN=hf_...
```

### Deployment

1. Deploy the Modal backend:
```bash
py -m modal deploy modal_app/app.py
```

This deploys 6 Modal classes:
- **ParakeetSTTModel** — GPU transcription (streaming + non-streaming)
- **SpeakerDiarizerModel** — Speaker identification with TitaNet
- **MeetingMinutesGenerator** — Claude Haiku 4.5 meeting summaries
- **Qwen3TTSVoiceCloner** — Voice cloning and synthesis
- **VoiceStorage** — Persistent voice profile management
- **FluxImageGenerator** — Text-to-image generation

2. Start the FastAPI server:
```bash
uv run uvicorn api.main:app --reload
```

3. Open `http://localhost:8000`

## Usage

### Web Interface

The UI has three modes:

**Transcription**
1. Drag and drop an audio file or click to browse
2. Toggle optional features: Speaker Diarization, Meeting Minutes
3. Watch real-time transcription results appear segment by segment
4. View speaker-labeled segments and meeting minutes tabs
5. Copy text, download transcription (TXT), or export subtitles (SRT/VTT)
6. Listen to original audio with the integrated player

**Voice Cloning**
1. Choose a saved voice or create a new one
2. For new voices: upload reference audio (3s–5min) or record with your microphone
3. Enter the reference transcription and select the language
4. Enter target text to synthesize (up to 50,000 characters)
5. Click Generate — listen to and download the result
6. Optionally save the voice profile for reuse

**Image Generation**
1. Enter a text prompt (up to 500 characters)
2. Select dimensions (512x512, 768x768, or 1024x1024)
3. Click Generate — preview and download the image

### CLI Tool

```bash
# Basic transcription
uv run transcribe_file.py audio.mp3

# Save to file
uv run transcribe_file.py audio.mp3 -o transcript.txt

# Non-streaming mode
uv run transcribe_file.py audio.mp3 --no-stream

# Process multiple files
uv run transcribe_file.py *.mp3

# All options
uv run transcribe_file.py --help
```

### API Endpoints

#### Transcription

**POST /api/transcribe** — Complete transcription (non-streaming)

```bash
curl -X POST "http://localhost:8000/api/transcribe" -F "file=@audio.mp3"
```

```json
{
  "text": "Complete transcription...",
  "language": "en",
  "duration": 45.2,
  "segments": [
    { "id": 0, "start": 0.0, "end": 3.5, "text": "First segment..." }
  ]
}
```

**POST /api/transcribe/stream** — Streaming transcription (SSE)

```bash
curl -X POST "http://localhost:8000/api/transcribe/stream" \
  -F "file=@audio.mp3" \
  -F "enable_diarization=true" \
  -F "enable_minutes=true"
```

SSE events: `metadata` → `progress` (per segment) → `speakers_ready` → `minutes_ready` → `complete`

**GET /api/audio/{session_id}** — Retrieve cached audio for playback

#### Voice Cloning

**POST /api/voice-clone** — Clone a voice and synthesize text (single-shot)

```bash
curl -X POST "http://localhost:8000/api/voice-clone" \
  -F "ref_audio=@reference.wav" \
  -F "ref_text=Hello, this is my voice." \
  -F "target_text=Text to synthesize with the cloned voice." \
  -F "language=en"
```

**GET /api/voices** — List all saved voice profiles

**POST /api/voices** — Save a new voice profile

```bash
curl -X POST "http://localhost:8000/api/voices" \
  -F "name=My Voice" \
  -F "ref_audio=@reference.wav" \
  -F "ref_text=Reference transcription" \
  -F "language=en"
```

**DELETE /api/voices/{voice_id}** — Delete a saved voice

**POST /api/synthesize** — Synthesize text with a saved voice

```bash
curl -X POST "http://localhost:8000/api/synthesize" \
  -F "voice_id=abc123" \
  -F "target_text=Text to synthesize."
```

#### Image Generation

**POST /api/generate-image** — Generate image from text prompt

```bash
curl -X POST "http://localhost:8000/api/generate-image" \
  -F "prompt=a beautiful sunset over mountains" \
  -F "width=768" \
  -F "height=768"
```

**GET /api/image/{session_id}** — Retrieve generated image as PNG

#### Health

**GET /health** — Health check

## Architecture

```
┌─────────────────────┐
│      Web UI         │
│  (HTML/JS/CSS)      │
│  3 modes:           │
│  Transcription      │
│  Voice Cloning      │
│  Image Generation   │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│      FastAPI        │
│  Upload/Validation  │
│  SSE Streaming      │
│  Session Caching    │
└─────────┬───────────┘
          │
          ▼
┌───────────────────────────────────────────────────┐
│              Modal Serverless GPUs                 │
│                                                   │
│  ┌──────────────┐  ┌───────────────────────────┐ │
│  │ Parakeet STT │  │ TitaNet Diarization (GPU) │ │
│  │ (L4 GPU)     │  └───────────────────────────┘ │
│  └──────────────┘                                 │
│  ┌──────────────┐  ┌───────────────────────────┐ │
│  │ Qwen3-TTS    │  │ FLUX.1-schnell (L4 GPU)   │ │
│  │ Voice Clone  │  │ Image Generation          │ │
│  │ (L4 GPU)     │  └───────────────────────────┘ │
│  └──────────────┘                                 │
│  ┌──────────────┐  ┌───────────────────────────┐ │
│  │ Claude Haiku │  │ Voice Storage             │ │
│  │ Minutes (CPU)│  │ (Modal Volume)            │ │
│  └──────────────┘  └───────────────────────────┘ │
└───────────────────────────────────────────────────┘
```

## Configuration

Modal-specific settings are embedded in `modal_app/app.py`. API-layer settings are in `config.py`.

Key parameters:

```python
# Transcription
STT_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
SAMPLE_RATE = 16000
MODAL_GPU_TYPE = "L4"

# Silence Detection (streaming segmentation)
SILENCE_THRESHOLD_DB = -40
SILENCE_MIN_LENGTH_MS = 700

# Speaker Diarization
ENABLE_SPEAKER_DIARIZATION = True
DIARIZATION_MAX_SPEAKERS = 5

# Meeting Minutes
ENABLE_MEETING_MINUTES = True
ANTHROPIC_MODEL_ID = "claude-haiku-4-5-20251001"

# Voice Cloning
VOICE_CLONE_MAX_REF_DURATION = 300   # 5 minutes
VOICE_CLONE_MAX_TARGET_TEXT = 50000  # characters
MAX_SAVED_VOICES = 50

# Image Generation
ENABLE_IMAGE_GENERATION = True
IMAGE_NUM_INFERENCE_STEPS = 4

# File Limits
MAX_FILE_SIZE_MB = 100
MAX_DURATION_SECONDS = 3600  # 60 minutes

# Performance
MODAL_CONTAINER_IDLE_TIMEOUT = 120
ENABLE_GPU_MEMORY_SNAPSHOT = True  # 85-90% faster cold starts
```

## Supported Formats

- **Audio**: MP3, WAV, M4A, FLAC, OGG, WebM
- **Video**: MP4 (audio track extracted)
- **Max file size**: 100MB (transcription), 15MB (voice reference)
- **Max duration**: 60 minutes (transcription), 5 minutes (voice reference)

## Cost Analysis

| Feature | Cost |
|---------|------|
| Transcription | ~$0.006 per minute of audio |
| Speaker Diarization | Included (same GPU) |
| Meeting Minutes | ~$0.001 per request (Haiku API) |
| Voice Cloning | ~$0.01-0.02 per synthesis |
| Image Generation | ~$0.01-0.02 per image |

## Project Structure

```
transcodio/
├── modal_app/
│   ├── app.py             # 6 Modal classes (STT, diarization, TTS, image gen, minutes, storage)
│   └── image.py           # Image generation helper
├── api/
│   ├── main.py            # FastAPI endpoints
│   ├── models.py          # Pydantic response models
│   └── streaming.py       # SSE streaming utilities
├── static/
│   ├── index.html         # Web UI layout
│   ├── app.js             # Frontend logic
│   └── styles.css         # Styling
├── utils/
│   └── audio.py           # Audio validation pipeline
├── config.py              # Configuration constants
├── transcribe_file.py     # CLI transcription tool
├── requirements.txt       # Dependencies
├── CLAUDE.md              # Development guide
└── README.md
```

## Troubleshooting

**"Modal service unavailable"** — Deploy the Modal app first:
```bash
py -m modal deploy modal_app/app.py
```

**Slow first request** — Cold start takes 30-60s. GPU memory snapshots are enabled by default for 85-90% faster subsequent cold starts.

**Meeting minutes not working** — Check the Anthropic API key secret:
```bash
py -m modal secret list
py -m modal secret create anthropic-api-key ANTHROPIC_API_KEY=sk-ant-...
```

**Image generation not working** — Check the HuggingFace token secret:
```bash
py -m modal secret list
py -m modal secret create hf-token HF_TOKEN=hf_...
```

**Audio validation errors** — Ensure FFmpeg is installed:
```bash
ffmpeg -version
```

**Voice cloning fails** — Ensure Modal app is deployed with all classes. Check reference audio is 3s–5min and in a supported format.

## Acknowledgments

- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) — Parakeet TDT model
- [NVIDIA TitaNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/titanet_large) — Speaker embeddings
- [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS) — Voice cloning model
- [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) — Image generation model
- [Anthropic Claude](https://www.anthropic.com) — Meeting minutes generation
- [Modal](https://modal.com) — Serverless GPU infrastructure
- [FastAPI](https://fastapi.tiangolo.com) — Web framework

## License

MIT License — see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

- Open an issue on [GitHub](https://github.com/dorianlgs/transcodio/issues)
- Check the [Modal documentation](https://modal.com/docs)
