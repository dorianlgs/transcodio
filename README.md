# Transcodio

**Fast, cost-effective AI audio transcription service with real-time streaming results**

Transcodio is a production-ready transcription service powered by OpenAI's Whisper Large model, deployed on Modal's serverless GPU infrastructure. Upload an audio file and get streaming transcription results in real-time.

## Features

- **High Accuracy**: Uses Whisper Large model for best-in-class transcription quality
- **Real-time Streaming**: See transcription results as they're generated via Server-Sent Events
- **Cost Effective**: ~$0.006 per minute of audio using NVIDIA L4 GPUs
- **Modern UI**: Beautiful, responsive web interface with drag-and-drop support
- **Multiple Formats**: Supports MP3, WAV, M4A, FLAC, OGG, WebM
- **Fast Processing**: GPU-accelerated transcription with container warm-up optimization
- **REST API**: Easy integration with your applications

## Quick Start

### Prerequisites

- Python 3.11 or higher
- [Modal](https://modal.com) account (free tier available)
- Modal CLI installed and authenticated

### Installation

1. Clone the repository:
```bash
git clone https://github.com/dorianlgs/transcodio.git
cd transcodio
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Modal authentication:
```bash
modal setup
```

### Deployment

1. Deploy the Modal GPU backend:
```bash
modal deploy modal_app/app.py
```

This will:
- Create a Modal app named `transcodio-app`
- Set up NVIDIA L4 GPU instances
- Download and cache the Whisper Large model
- Deploy the transcription service

2. Start the FastAPI server locally:
```bash
python -m uvicorn api.main:app --reload
```

3. Open your browser to `http://localhost:8000`

## Usage

### Web Interface

1. Navigate to `http://localhost:8000`
2. Drag and drop an audio file or click to browse
3. Watch real-time transcription results appear
4. Copy or download the transcription

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
  -F "file=@audio.mp3"
```

**Response (SSE Stream):**
```
event: metadata
data: {"language": "en", "duration": 45.2}

event: progress
data: {"id": 0, "start": 0.0, "end": 3.5, "text": "First segment..."}

event: progress
data: {"id": 1, "start": 3.5, "end": 7.2, "text": "Second segment..."}

event: complete
data: {"text": "Complete transcription..."}
```

#### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

## Configuration

Edit `config.py` to customize settings:

```python
# GPU Configuration
MODAL_GPU_TYPE = "L4"  # Options: L4, A10G, T4
WHISPER_MODEL = "large"  # Options: tiny, base, small, medium, large

# File Limits
MAX_FILE_SIZE_MB = 100
MAX_DURATION_SECONDS = 600  # 10 minutes

# Performance
WHISPER_FP16 = True  # Enable FP16 for 2x speed
MODAL_CONTAINER_IDLE_TIMEOUT = 120  # Keep containers warm for 2 minutes
```

## Cost Analysis

Using **NVIDIA L4 GPU** on Modal:

| Audio Length | Processing Time | Cost per File | Monthly (100/day) |
|--------------|----------------|---------------|-------------------|
| 1 minute | ~30 seconds | $0.006 | $18 |
| 5 minutes | ~2.5 minutes | $0.029 | $87 |
| 10 minutes | ~5 minutes | $0.058 | $174 |

**Cost optimization tips:**
- Use `container_idle_timeout` to keep containers warm between requests
- Enable FP16 for 2x faster processing
- Consider smaller models (base, small) for less critical use cases
- Batch process multiple files when possible

## Architecture

```
┌─────────────┐
│  Web UI     │
│  (HTML/JS)  │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│   FastAPI       │
│   (Python)      │
│   - Upload      │
│   - Streaming   │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Modal GPU      │
│  (Serverless)   │
│  - Whisper Large│
│  - NVIDIA L4    │
└─────────────────┘
```

### Key Components

- **Frontend**: Modern responsive UI with drag-and-drop and SSE streaming
- **API Layer**: FastAPI with file upload, validation, and SSE support
- **GPU Backend**: Modal serverless functions with automatic scaling
- **Model**: OpenAI Whisper Large for high-accuracy transcription

## Supported Formats

- **Audio**: MP3, WAV, M4A, FLAC, OGG, WebM
- **Video**: MP4 (audio track will be extracted)
- **Max file size**: 100MB
- **Max duration**: 10 minutes

## Development

### Project Structure

```
transcodio/
├── modal_app/          # Modal GPU backend
│   ├── app.py         # Whisper model class
│   └── image.py       # Container image definition
├── api/               # FastAPI application
│   ├── main.py        # API endpoints
│   ├── models.py      # Pydantic schemas
│   └── streaming.py   # SSE utilities
├── static/            # Web interface
│   ├── index.html     # UI layout
│   ├── app.js         # Frontend logic
│   └── styles.css     # Modern styling
├── utils/             # Utilities
│   └── audio.py       # Audio validation
├── config.py          # Configuration
└── requirements.txt   # Dependencies
```

### Testing Locally

Test the Modal function directly:

```bash
modal run modal_app/app.py path/to/audio.mp3
```

### Running Tests

```bash
# Install dev dependencies
pip install pytest pytest-asyncio httpx

# Run tests (coming soon)
pytest tests/
```

## Troubleshooting

### "Modal service unavailable"
Make sure the Modal app is deployed:
```bash
modal deploy modal_app/app.py
modal app list  # Verify it's running
```

### Slow first request (cold start)
First request after idle takes 20-30s to load the model. Subsequent requests are fast. Adjust `MODAL_CONTAINER_IDLE_TIMEOUT` to keep containers warm longer.

### "File size exceeds limit"
Default limit is 100MB. Edit `MAX_FILE_SIZE_MB` in `config.py` to increase.

### GPU out of memory
Whisper Large needs ~10GB VRAM. If using smaller GPUs, switch to `medium` or `small` model in `config.py`.

## Roadmap

- [ ] Speaker diarization (identify different speakers)
- [ ] Multi-language support UI
- [ ] Export to SRT/VTT subtitle formats
- [ ] Batch processing for multiple files
- [ ] Authentication and API keys
- [ ] Usage dashboard and analytics
- [ ] Webhook support for async processing
- [ ] Model selection UI (choose between tiny/base/small/medium/large)

## License

MIT License - see [LICENSE](LICENSE) file for details

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for the transcription model
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