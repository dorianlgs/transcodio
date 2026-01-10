"""FastAPI application for transcription service."""

import sys
from pathlib import Path
from typing import Optional
import asyncio
import uuid
from datetime import datetime, timedelta

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, Response
from sse_starlette.sse import EventSourceResponse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from api.models import (
    TranscriptionResponse,
    ErrorResponse,
    HealthResponse,
)
from api.streaming import transcription_event_stream
from utils.audio import validate_audio_file, AudioValidationError

# Simple in-memory cache for audio (session_id -> (audio_bytes, content_type, filename, expiry_time))
audio_cache = {}

def cleanup_expired_audio():
    """Remove expired audio from cache."""
    now = datetime.now()
    expired_keys = [k for k, v in audio_cache.items() if v[3] < now]
    for key in expired_keys:
        del audio_cache[key]

# Create FastAPI app
app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description=config.API_DESCRIPTION,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = Path(__file__).parent.parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web UI."""
    index_path = Path(__file__).parent.parent / "static" / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse(
        content="<h1>Transcodio Transcription Service</h1><p>Upload an audio file to /api/transcribe</p>",
        status_code=200,
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=config.API_VERSION,
    )


@app.get("/api/audio/{session_id}")
async def get_audio(session_id: str):
    """
    Retrieve original audio file by session ID.

    Args:
        session_id: Session ID returned in the transcription completion event

    Returns:
        Original audio file in its uploaded format

    Raises:
        HTTPException: If session ID is invalid or expired
    """
    # Clean up expired entries first
    cleanup_expired_audio()

    if session_id not in audio_cache:
        raise HTTPException(
            status_code=404,
            detail="Audio session not found or expired"
        )

    audio_bytes, content_type, filename, expiry = audio_cache[session_id]

    # Return audio file with appropriate headers and original content type
    return Response(
        content=audio_bytes,
        media_type=content_type,
        headers={
            "Content-Disposition": f"inline; filename={filename}",
            "Accept-Ranges": "bytes",
        }
    )


@app.post("/api/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file to transcribe"),
):
    """
    Transcribe an audio file (non-streaming).

    Args:
        file: Uploaded audio file

    Returns:
        Complete transcription with segments

    Raises:
        HTTPException: If validation fails or transcription errors
    """
    try:
        # Read file
        audio_bytes = await file.read()
        file_size = len(audio_bytes)

        # Validate and preprocess audio
        try:
            duration, preprocessed_bytes = validate_audio_file(
                filename=file.filename,
                file_size=file_size,
                audio_bytes=audio_bytes,
                content_type=file.content_type,
            )
        except AudioValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Import Modal and lookup function
        try:
            import modal

            # Lookup the deployed class and method
            STTModel = modal.Cls.from_name(config.MODAL_APP_NAME, "ParakeetSTTModel")
            model = STTModel()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Modal service unavailable: {str(e)}. Make sure the Modal app is deployed.",
            )

        # Call Modal transcription
        try:
            result = model.transcribe.remote(preprocessed_bytes)
            result["duration"] = duration
            return TranscriptionResponse(**result)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Transcription failed: {str(e)}",
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}",
        )


@app.post("/api/transcribe/stream")
async def transcribe_audio_stream(
    file: UploadFile = File(..., description="Audio file to transcribe"),
):
    """
    Transcribe an audio file with streaming results via Server-Sent Events.

    Args:
        file: Uploaded audio file

    Returns:
        SSE stream of transcription segments

    Raises:
        HTTPException: If validation fails or transcription errors
    """
    try:
        # Read file
        audio_bytes = await file.read()
        file_size = len(audio_bytes)

        # Validate and preprocess audio
        try:
            duration, preprocessed_bytes = validate_audio_file(
                filename=file.filename,
                file_size=file_size,
                audio_bytes=audio_bytes,
                content_type=file.content_type,
            )
        except AudioValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Generate session ID and cache the ORIGINAL audio file
        session_id = str(uuid.uuid4())
        # Cache for 1 hour - store original audio for playback
        expiry_time = datetime.now() + timedelta(hours=1)
        audio_cache[session_id] = (
            audio_bytes,  # Original uploaded file
            file.content_type or "application/octet-stream",  # Original content type
            file.filename or "audio",  # Original filename
            expiry_time
        )
        # Clean up expired entries
        cleanup_expired_audio()

        # Import Modal and lookup function
        try:
            import modal
            import json

            # Lookup the deployed class
            STTModel = modal.Cls.from_name(config.MODAL_APP_NAME, "ParakeetSTTModel")
            model = STTModel()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Modal service unavailable: {str(e)}. Make sure the Modal app is deployed.",
            )

        # Create async generator for streaming
        async def event_generator():
            try:
                import asyncio
                print("Starting stream generation...")
                # Call Modal streaming transcription in thread pool to avoid blocking
                loop = asyncio.get_event_loop()

                def sync_generator():
                    for segment_json in model.transcribe_stream.remote_gen(preprocessed_bytes, duration):
                        yield segment_json

                # Process synchronous generator in async context
                for segment_json in sync_generator():
                    # Parse and yield as SSE event
                    print(f"Received segment: {segment_json[:100] if len(segment_json) > 100 else segment_json}...")
                    segment_data = json.loads(segment_json)
                    event_type = segment_data.get("type", "unknown")
                    print(f"Event type: {event_type}")

                    # Yield control to event loop
                    await asyncio.sleep(0)

                    if event_type == "metadata":
                        yield {
                            "event": "metadata",
                            "data": json.dumps({
                                "language": segment_data.get("language"),
                                "duration": segment_data.get("duration"),
                            })
                        }

                    elif event_type == "segment":
                        yield {
                            "event": "progress",
                            "data": json.dumps({
                                "id": segment_data.get("id"),
                                "start": segment_data.get("start"),
                                "end": segment_data.get("end"),
                                "text": segment_data.get("text"),
                            })
                        }

                    elif event_type == "complete":
                        yield {
                            "event": "complete",
                            "data": json.dumps({
                                "text": segment_data.get("text"),
                                "audio_session_id": session_id,
                            })
                        }

                    elif event_type == "error":
                        yield {
                            "event": "error",
                            "data": json.dumps({
                                "error": segment_data.get("error"),
                            })
                        }

            except Exception as e:
                yield {
                    "event": "error",
                    "data": json.dumps({
                        "error": f"Transcription error: {str(e)}",
                    })
                }

        # Return SSE response
        return EventSourceResponse(event_generator())

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=config.DEBUG,
    )
