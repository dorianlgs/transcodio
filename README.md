# Transcodio

**Fast, cost-effective AI audio transcription service with real-time streaming, speaker diarization, and AI-powered meeting minutes**

Transcodio is a production-ready transcription service powered by NVIDIA's Parakeet TDT 0.6B v3 model, deployed on Modal's serverless GPU infrastructure. Upload an audio file and get streaming transcription results in real-time, with automatic speaker identification and AI-generated meeting minutes.

## Features

- **High Accuracy**: Uses NVIDIA Parakeet TDT 0.6B v3 for best-in-class transcription quality
- **Real-time Streaming**: See transcription results as they're generated via Server-Sent Events with silence-based segmentation
- **Speaker Diarization**: Automatic speaker identification using NVIDIA TitaNet
- **Meeting Minutes**: AI-powered meeting summaries using Anthropic Claude Haiku 4.5
- **Audio Playback**: Integrated player to listen to uploaded audio alongside transcription
- **Cost Effective**: ~$0.006 per minute of audio using NVIDIA L4 GPUs
- **Modern UI**: Beautiful, responsive web interface with drag-and-drop support
- **Multiple Formats**: Supports MP3, WAV, M4A, FLAC, OGG, WebM
- **Subtitle Export**: Download transcriptions as SRT/VTT with speaker labels
- **REST API**: Easy integration with your applications

## Quick Start

### Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer and runner
- [Modal](https://modal.com) account (free tier available)
- Modal CLI installed and authenticated
- FFmpeg installed locally
- Anthropic API key (for meeting minutes feature)

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

4. Create Modal secret for Anthropic API (required for meeting minutes):
```bash
py -m modal secret create anthropic-api-key ANTHROPIC_API_KEY=sk-ant-...
```

### Deployment

1. Deploy the Modal backend:
```bash
py -m modal deploy modal_app/app.py
```

This will:
- Create a Modal app named `transcodio-app`
- Set up NVIDIA L4 GPU instances for transcription and diarization
- Deploy the Anthropic-powered meeting minutes generator
- Download and cache the Parakeet TDT and TitaNet models

2. Start the FastAPI server locally:
```bash
uv run uvicorn api.main:app --reload
```


3. Open your browser to `http://localhost:8000`

## Usage

### Web Interface

1. Navigate to `http://localhost:8000`
2. Drag and drop an audio file or click to browse
3. Enable optional features:
   - **Speaker Diarization**: Identify different speakers
   - **Meeting Minutes**: Generate AI-powered meeting summary
4. Watch real-time transcription results appear
5. View speaker-labeled segments and meeting minutes
6. Copy, download transcription (TXT), or export subtitles (SRT/VTT)

### CLI Tool

Transcribe audio files directly from the command line:

```bash
# Basic usage
uv run transcribe_file.py audio.mp3

# Save to file
uv run transcribe_file.py audio.mp3 -o transcript.txt

# Non-streaming mode (faster for complete results)
uv run transcribe_file.py audio.mp3 --no-stream

# Process multiple files
uv run transcribe_file.py *.mp3

# View all options
uv run transcribe_file.py --help
```

### API Endpoints

#### POST /api/transcribe
Upload an audio file for complete transcription (non-streaming).

**Request:**
```bash
curl -X POST "http://localhost:8000/api/transcribe" \
  -F "file=@audio.mp3"
```

**Response:**
```json
{
  "text": "Complete transcription...",
  "language": "en",
  "duration": 45.2,
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 3.5,
      "text": "First segment..."
    }
  ]
}
```

#### POST /api/transcribe/stream
Upload an audio file for streaming transcription via Server-Sent Events.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/transcribe/stream" \
  -F "file=@audio.mp3" \
  -F "enable_diarization=true" \
  -F "enable_minutes=true"
```

**Response (SSE Stream):**
```
event: metadata
data: {"language": "en", "duration": 45.2}

event: progress
data: {"id": 0, "start": 0.0, "end": 3.5, "text": "First segment..."}

event: speakers_ready
data: {"segments": [{"id": 0, "speaker": "Speaker 1", ...}]}

event: minutes_ready
data: {"minutes": {"executive_summary": "...", "action_items": [...]}}

event: complete
data: {"text": "Complete transcription...", "audio_session_id": "uuid"}
```

#### GET /api/audio/{session_id}
Retrieve cached audio file for playback.

#### GET /health
Health check endpoint.

## Configuration

Edit `config.py` to customize settings:

```python
# STT Model Configuration
STT_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
SAMPLE_RATE = 16000

# GPU Configuration
MODAL_GPU_TYPE = "L4"  # Options: L4, A10G, T4

# File Limits
MAX_FILE_SIZE_MB = 100
MAX_DURATION_SECONDS = 3600  # 60 minutes

# Silence Detection (for streaming segmentation)
SILENCE_THRESHOLD_DB = -40
SILENCE_MIN_LENGTH_MS = 700

# Speaker Diarization
ENABLE_SPEAKER_DIARIZATION = True
DIARIZATION_MAX_SPEAKERS = 5

# Meeting Minutes (Anthropic Claude API)
ENABLE_MEETING_MINUTES = True
ANTHROPIC_MODEL_ID = "claude-haiku-4-5-20251001"
MINUTES_TEMPERATURE = 0.3

# Performance
MODAL_CONTAINER_IDLE_TIMEOUT = 120  # Keep containers warm
```

## Architecture

```
┌─────────────────┐
│    Web UI       │
│   (HTML/JS)     │
└───────┬─────────┘
        │
        ▼
┌─────────────────┐
│    FastAPI      │
│    (Python)     │
│  - Upload       │
│  - Validation   │
│  - SSE Stream   │
└───────┬─────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│           Modal Serverless              │
│                                         │
│  ┌─────────────┐  ┌─────────────────┐  │
│  │ Parakeet    │  │ TitaNet         │  │
│  │ STT (GPU)   │  │ Diarization     │  │
│  │ L4 24GB     │  │ (GPU)           │  │
│  └─────────────┘  └─────────────────┘  │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │ Claude Haiku 4.5 (API)          │   │
│  │ Meeting Minutes (No GPU)        │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

### Key Components

- **Frontend**: Modern responsive UI with drag-and-drop, SSE streaming, audio player
- **API Layer**: FastAPI with file upload, validation, and SSE support
- **STT Backend**: NVIDIA Parakeet TDT 0.6B on Modal GPU
- **Diarization**: NVIDIA TitaNet for speaker embeddings + AgglomerativeClustering
- **Minutes**: Anthropic Claude Haiku 4.5 API for meeting summaries

## Supported Formats

- **Audio**: MP3, WAV, M4A, FLAC, OGG, WebM
- **Video**: MP4 (audio track will be extracted)
- **Max file size**: 100MB
- **Max duration**: 60 minutes

## Cost Analysis

Using **NVIDIA L4 GPU** on Modal + **Claude Haiku 4.5** API:

| Feature | Cost |
|---------|------|
| Transcription | ~$0.006 per minute of audio |
| Speaker Diarization | Included (same GPU) |
| Meeting Minutes | ~$0.001 per request (Haiku API) |

**Cost optimization tips:**
- Use `MODAL_CONTAINER_IDLE_TIMEOUT` to keep containers warm between requests
- Enable GPU memory snapshots for faster cold starts
- Disable meeting minutes if not needed

## Troubleshooting

### "Modal service unavailable"
Make sure the Modal app is deployed:
```bash
py -m modal deploy modal_app/app.py
py -m modal app list  # Verify it's running
```

### Slow first request (cold start)
First request after idle takes 30-60s to load models. Enable memory snapshots in `config.py`:
```python
ENABLE_CPU_MEMORY_SNAPSHOT = True
ENABLE_GPU_MEMORY_SNAPSHOT = True
```

### Meeting minutes not working
Ensure the Anthropic API key secret exists:
```bash
py -m modal secret list  # Should show anthropic-api-key
py -m modal secret create anthropic-api-key ANTHROPIC_API_KEY=sk-ant-...
```

### Audio validation errors
Check that FFmpeg is installed locally:
```bash
ffmpeg -version
```

## Development

### Project Structure

```
transcodio/
├── modal_app/          # Modal GPU backend
│   └── app.py          # STT, Diarization, Minutes classes
├── api/                # FastAPI application
│   └── main.py         # API endpoints
├── static/             # Web interface
│   ├── index.html      # UI layout
│   ├── app.js          # Frontend logic
│   └── styles.css      # Modern styling
├── utils/              # Utilities
│   └── audio.py        # Audio validation
├── config.py           # Configuration
├── requirements.txt    # Dependencies
├── CLAUDE.md           # Development guide
└── README.md           # This file
```

### Testing Locally

Test the Modal function directly:
```bash
py -m modal run modal_app/app.py path/to/audio.mp3
```

Or use the CLI tool:
```bash
uv run transcribe_file.py path/to/audio.mp3
```

## License

MIT License - see [LICENSE](LICENSE) file for details

## Acknowledgments

- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) for the Parakeet TDT model
- [NVIDIA TitaNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/titanet_large) for speaker embeddings
- [Anthropic Claude](https://www.anthropic.com) for meeting minutes generation
- [Modal](https://modal.com) for serverless GPU infrastructure
- [FastAPI](https://fastapi.tiangolo.com) for the web framework

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions:
- Open an issue on [GitHub](https://github.com/dorianlgs/transcodio/issues)
- Check the [Modal documentation](https://modal.com/docs)

---

**Built with passion by developers, for developers**
