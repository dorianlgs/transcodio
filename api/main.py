"""FastAPI application for transcription service."""

import sys
from pathlib import Path
from typing import Optional
import asyncio
import uuid
from datetime import datetime, timedelta

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
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
    VoiceCloneResponse,
    ImageGenerationResponse,
    SavedVoiceListResponse,
    SaveVoiceResponse,
    SynthesizeResponse,
)
from api.streaming import transcription_event_stream
from utils.audio import validate_audio_file, AudioValidationError

# Simple in-memory cache for audio (session_id -> (audio_bytes, content_type, filename, expiry_time))
audio_cache = {}

# Simple in-memory cache for images (session_id -> (image_bytes, content_type, expiry_time))
image_cache = {}

def cleanup_expired_audio():
    """Remove expired audio from cache."""
    now = datetime.now()
    expired_keys = [k for k, v in audio_cache.items() if v[3] < now]
    for key in expired_keys:
        del audio_cache[key]

def cleanup_expired_images():
    """Remove expired images from cache."""
    now = datetime.now()
    expired_keys = [k for k, v in image_cache.items() if v[2] < now]
    for key in expired_keys:
        del image_cache[key]

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
    enable_diarization: bool = Form(default=False, description="Enable speaker diarization"),
    enable_minutes: bool = Form(default=False, description="Generate meeting minutes"),
):
    """
    Transcribe an audio file with streaming results via Server-Sent Events.

    Args:
        file: Uploaded audio file
        enable_diarization: Whether to identify speakers in the audio
        enable_minutes: Whether to generate meeting minutes after transcription

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

                # Accumulate segments for diarization
                segments_data = []
                full_text = ""

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
                        # Accumulate segment for diarization
                        segments_data.append({
                            "id": segment_data.get("id"),
                            "start": segment_data.get("start"),
                            "end": segment_data.get("end"),
                            "text": segment_data.get("text"),
                        })

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
                        full_text = segment_data.get("text", "")

                        # Run speaker diarization if enabled and there are segments
                        if enable_diarization and config.ENABLE_SPEAKER_DIARIZATION and segments_data:
                            try:
                                print("Running speaker diarization...")
                                # Import diarizer from Modal
                                Diarizer = modal.Cls.from_name(config.MODAL_APP_NAME, "SpeakerDiarizerModel")
                                diarizer = Diarizer()

                                # Run diarization
                                speaker_timeline = diarizer.diarize.remote(preprocessed_bytes, duration)

                                if speaker_timeline:
                                    # Import alignment function
                                    import sys
                                    from pathlib import Path
                                    sys.path.insert(0, str(Path(__file__).parent.parent / "modal_app"))
                                    from app import align_speakers_to_segments

                                    # Align speakers with segments
                                    segments_with_speakers = align_speakers_to_segments(
                                        segments_data,
                                        speaker_timeline
                                    )

                                    # Yield updated segments with speaker labels
                                    yield {
                                        "event": "speakers_ready",
                                        "data": json.dumps({"segments": segments_with_speakers})
                                    }

                                    print(f"Speaker diarization complete: {len(speaker_timeline)} speaker segments")
                                else:
                                    print("Diarization returned no speaker segments")
                            except Exception as e:
                                print(f"Diarization failed (non-fatal): {e}")
                                import traceback
                                traceback.print_exc()
                                # Continue without speaker labels

                        # Yield completion event
                        yield {
                            "event": "complete",
                            "data": json.dumps({
                                "text": full_text,
                                "audio_session_id": session_id,
                            })
                        }

                        # Generate meeting minutes if enabled
                        if enable_minutes and config.ENABLE_MEETING_MINUTES and full_text:
                            try:
                                print("Generating meeting minutes...")
                                # Import minutes generator from Modal
                                MinutesGenerator = modal.Cls.from_name(
                                    config.MODAL_APP_NAME, "MeetingMinutesGenerator"
                                )
                                generator = MinutesGenerator()

                                # Generate minutes
                                minutes_result = generator.generate_minutes.remote(
                                    full_text,
                                    segments_data if segments_data else None
                                )

                                if minutes_result.get("success"):
                                    yield {
                                        "event": "minutes_ready",
                                        "data": json.dumps({
                                            "minutes": minutes_result.get("minutes", {})
                                        })
                                    }
                                    print("Meeting minutes generated successfully")
                                else:
                                    print(f"Minutes generation failed: {minutes_result.get('error')}")
                                    yield {
                                        "event": "minutes_error",
                                        "data": json.dumps({
                                            "error": minutes_result.get("error", "Unknown error")
                                        })
                                    }
                            except Exception as e:
                                print(f"Minutes generation failed (non-fatal): {e}")
                                import traceback
                                traceback.print_exc()
                                yield {
                                    "event": "minutes_error",
                                    "data": json.dumps({
                                        "error": str(e)
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
                import traceback
                traceback.print_exc()
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


@app.post("/api/voice-clone", response_model=VoiceCloneResponse)
async def voice_clone(
    ref_audio: UploadFile = File(..., description="Reference audio file (3-30 seconds)"),
    ref_text: str = Form(..., description="Transcription of the reference audio"),
    target_text: str = Form(..., description="Text to synthesize with cloned voice"),
    language: str = Form(default="Spanish", description="Target language"),
    tts_model: str = Form(default="qwen", description="TTS model to use: qwen, higgs, or neutts"),
):
    """
    Clone a voice and synthesize new text.

    Args:
        ref_audio: Reference audio file (3-30 seconds)
        ref_text: Transcription of the reference audio
        target_text: Text to synthesize with cloned voice
        language: Target language (Spanish, English, etc.)
        tts_model: TTS model to use (qwen, higgs, or neutts)

    Returns:
        VoiceCloneResponse with audio_session_id for playback/download

    Raises:
        HTTPException: If validation fails or generation errors
    """
    # Check if voice cloning is enabled
    if not config.ENABLE_VOICE_CLONING:
        raise HTTPException(
            status_code=503,
            detail="Voice cloning is disabled"
        )

    # Validate TTS model
    if tts_model not in config.TTS_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid TTS model. Available: {', '.join(config.TTS_MODELS.keys())}"
        )

    try:
        # Validate language
        if language not in config.VOICE_CLONE_LANGUAGES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language. Supported: {', '.join(config.VOICE_CLONE_LANGUAGES)}"
            )

        # Validate target text length
        if len(target_text) > config.VOICE_CLONE_MAX_TARGET_TEXT:
            raise HTTPException(
                status_code=400,
                detail=f"Target text too long. Maximum {config.VOICE_CLONE_MAX_TARGET_TEXT} characters."
            )

        if len(target_text.strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail="Target text cannot be empty."
            )

        # Validate reference text
        if len(ref_text.strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail="Reference text cannot be empty."
            )

        # Read reference audio
        ref_audio_bytes = await ref_audio.read()
        file_size = len(ref_audio_bytes)

        # Validate file size (max 10MB for reference audio)
        if file_size > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail="Reference audio too large. Maximum 10MB."
            )

        # Preprocess reference audio (convert to 24kHz mono WAV)
        try:
            ref_duration, preprocessed_ref = validate_audio_file(
                filename=ref_audio.filename,
                file_size=file_size,
                audio_bytes=ref_audio_bytes,
                content_type=ref_audio.content_type,
                target_sample_rate=config.VOICE_CLONE_SAMPLE_RATE,
            )
        except AudioValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Validate reference audio duration
        if ref_duration < config.VOICE_CLONE_MIN_REF_DURATION:
            raise HTTPException(
                status_code=400,
                detail=f"Reference audio too short. Minimum {config.VOICE_CLONE_MIN_REF_DURATION} seconds."
            )

        if ref_duration > config.VOICE_CLONE_MAX_REF_DURATION:
            raise HTTPException(
                status_code=400,
                detail=f"Reference audio too long. Maximum {config.VOICE_CLONE_MAX_REF_DURATION} seconds."
            )

        # Import Modal and lookup TTS model based on selection
        try:
            import modal

            # Select the appropriate model class
            if tts_model == "higgs":
                TTSModel = modal.Cls.from_name(config.MODAL_APP_NAME, "HiggsAudioVoiceCloner")
            elif tts_model == "neutts":
                TTSModel = modal.Cls.from_name(config.MODAL_APP_NAME, "NeuTTSVoiceCloner")
            else:
                TTSModel = modal.Cls.from_name(config.MODAL_APP_NAME, "Qwen3TTSVoiceCloner")
            model = TTSModel()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"TTS service unavailable: {str(e)}. Make sure the Modal app is deployed.",
            )

        # Generate voice clone
        try:
            result = model.generate_voice_clone.remote(
                preprocessed_ref,
                ref_text,
                target_text,
                language
            )

            if not result.get("success"):
                raise HTTPException(
                    status_code=500,
                    detail=f"Voice cloning failed: {result.get('error', 'Unknown error')}"
                )

            # Generate session ID and cache the generated audio
            session_id = str(uuid.uuid4())
            expiry_time = datetime.now() + timedelta(hours=1)
            audio_cache[session_id] = (
                result["audio_bytes"],
                "audio/wav",
                "voice_clone.wav",
                expiry_time
            )

            # Clean up expired entries
            cleanup_expired_audio()

            return VoiceCloneResponse(
                success=True,
                audio_session_id=session_id,
                duration=result.get("duration"),
            )

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Voice cloning failed: {str(e)}",
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}",
        )


@app.get("/api/voices", response_model=SavedVoiceListResponse)
async def list_voices():
    """
    List all saved voices.

    Returns:
        SavedVoiceListResponse with list of saved voices
    """
    if not config.ENABLE_VOICE_CLONING:
        raise HTTPException(status_code=503, detail="Voice cloning is disabled")

    try:
        import modal

        VoiceStorage = modal.Cls.from_name(config.MODAL_APP_NAME, "VoiceStorage")
        storage = VoiceStorage()
        voices = storage.list_voices.remote()
        return SavedVoiceListResponse(voices=voices)
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Voice storage service unavailable: {str(e)}"
        )


@app.post("/api/voices", response_model=SaveVoiceResponse)
async def save_voice(
    name: str = Form(..., description="Name for the voice"),
    ref_audio: UploadFile = File(..., description="Reference audio file (3-60 seconds)"),
    ref_text: str = Form(..., description="Transcription of the reference audio"),
    language: str = Form(default="Spanish", description="Voice language"),
):
    """
    Save a new voice for later use.

    Args:
        name: User-friendly name for the voice
        ref_audio: Reference audio file (3-60 seconds)
        ref_text: Exact transcription of the reference audio
        language: Language of the voice

    Returns:
        SaveVoiceResponse with voice_id
    """
    if not config.ENABLE_VOICE_CLONING:
        raise HTTPException(status_code=503, detail="Voice cloning is disabled")

    # Validate name
    if not name or len(name.strip()) == 0:
        raise HTTPException(status_code=400, detail="Voice name cannot be empty")
    if len(name) > 50:
        raise HTTPException(status_code=400, detail="Voice name too long (max 50 characters)")

    # Validate language
    if language not in config.VOICE_CLONE_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language. Supported: {', '.join(config.VOICE_CLONE_LANGUAGES)}"
        )

    # Validate ref_text
    if not ref_text or len(ref_text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Reference text cannot be empty")

    try:
        # Read and validate audio
        ref_audio_bytes = await ref_audio.read()
        file_size = len(ref_audio_bytes)

        if file_size > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Reference audio too large (max 10MB)")

        # Preprocess audio
        try:
            ref_duration, preprocessed_ref = validate_audio_file(
                filename=ref_audio.filename,
                file_size=file_size,
                audio_bytes=ref_audio_bytes,
                content_type=ref_audio.content_type,
                target_sample_rate=config.VOICE_CLONE_SAMPLE_RATE,
            )
        except AudioValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Validate duration
        if ref_duration < config.VOICE_CLONE_MIN_REF_DURATION:
            raise HTTPException(
                status_code=400,
                detail=f"Reference audio too short (min {config.VOICE_CLONE_MIN_REF_DURATION}s)"
            )
        if ref_duration > config.VOICE_CLONE_MAX_REF_DURATION:
            raise HTTPException(
                status_code=400,
                detail=f"Reference audio too long (max {config.VOICE_CLONE_MAX_REF_DURATION}s)"
            )

        # Save voice
        import modal

        VoiceStorage = modal.Cls.from_name(config.MODAL_APP_NAME, "VoiceStorage")
        storage = VoiceStorage()

        voice_id = str(uuid.uuid4())
        result = storage.save_voice.remote(
            voice_id=voice_id,
            name=name.strip(),
            ref_audio_bytes=preprocessed_ref,
            ref_text=ref_text.strip(),
            language=language,
        )

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to save voice"))

        return SaveVoiceResponse(success=True, voice_id=voice_id)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.delete("/api/voices/{voice_id}")
async def delete_voice(voice_id: str):
    """
    Delete a saved voice.

    Args:
        voice_id: ID of the voice to delete

    Returns:
        Success status
    """
    if not config.ENABLE_VOICE_CLONING:
        raise HTTPException(status_code=503, detail="Voice cloning is disabled")

    try:
        import modal

        VoiceStorage = modal.Cls.from_name(config.MODAL_APP_NAME, "VoiceStorage")
        storage = VoiceStorage()
        result = storage.delete_voice.remote(voice_id)

        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("error", "Voice not found"))

        return {"success": True}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.post("/api/synthesize", response_model=SynthesizeResponse)
async def synthesize_with_voice(
    voice_id: str = Form(..., description="ID of the saved voice to use"),
    target_text: str = Form(..., description="Text to synthesize"),
):
    """
    Synthesize text using a saved voice.

    Args:
        voice_id: ID of the saved voice
        target_text: Text to synthesize (max 500 characters)

    Returns:
        SynthesizeResponse with audio_session_id for playback/download
    """
    if not config.ENABLE_VOICE_CLONING:
        raise HTTPException(status_code=503, detail="Voice cloning is disabled")

    # Validate target text
    if not target_text or len(target_text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Target text cannot be empty")
    if len(target_text) > config.VOICE_CLONE_MAX_TARGET_TEXT:
        raise HTTPException(
            status_code=400,
            detail=f"Target text too long (max {config.VOICE_CLONE_MAX_TARGET_TEXT} characters)"
        )

    try:
        import modal

        # Get saved voice
        VoiceStorage = modal.Cls.from_name(config.MODAL_APP_NAME, "VoiceStorage")
        storage = VoiceStorage()
        voice_data = storage.get_voice.remote(voice_id)

        if not voice_data.get("success"):
            raise HTTPException(status_code=404, detail=voice_data.get("error", "Voice not found"))

        metadata = voice_data["metadata"]
        ref_audio_bytes = voice_data["audio_bytes"]

        # Generate audio with TTS
        TTSModel = modal.Cls.from_name(config.MODAL_APP_NAME, "Qwen3TTSVoiceCloner")
        model = TTSModel()

        result = model.generate_voice_clone.remote(
            ref_audio_bytes,
            metadata["ref_text"],
            target_text.strip(),
            metadata["language"],
        )

        if not result.get("success"):
            raise HTTPException(
                status_code=500,
                detail=f"Synthesis failed: {result.get('error', 'Unknown error')}"
            )

        # Cache generated audio
        session_id = str(uuid.uuid4())
        expiry_time = datetime.now() + timedelta(hours=1)
        audio_cache[session_id] = (
            result["audio_bytes"],
            "audio/wav",
            "synthesized.wav",
            expiry_time,
        )
        cleanup_expired_audio()

        return SynthesizeResponse(
            success=True,
            audio_session_id=session_id,
            duration=result.get("duration"),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.get("/api/image/{session_id}")
async def get_image(session_id: str):
    """
    Retrieve generated image by session ID.

    Args:
        session_id: Session ID returned in the image generation response

    Returns:
        Generated image as PNG

    Raises:
        HTTPException: If session ID is invalid or expired
    """
    # Clean up expired entries first
    cleanup_expired_images()

    if session_id not in image_cache:
        raise HTTPException(
            status_code=404,
            detail="Image session not found or expired"
        )

    image_bytes, content_type, expiry = image_cache[session_id]

    return Response(
        content=image_bytes,
        media_type=content_type,
        headers={
            "Content-Disposition": f"inline; filename=generated-image.png",
        }
    )


@app.post("/api/generate-image", response_model=ImageGenerationResponse)
async def generate_image(
    prompt: str = Form(..., description="Text prompt describing the image to generate"),
    width: int = Form(default=768, description="Image width (512-1024)"),
    height: int = Form(default=768, description="Image height (512-1024)"),
):
    """
    Generate an image from a text prompt using FLUX.1-schnell.

    Args:
        prompt: Text description of the image to generate
        width: Image width in pixels (512-1024)
        height: Image height in pixels (512-1024)

    Returns:
        ImageGenerationResponse with image_session_id for retrieval

    Raises:
        HTTPException: If validation fails or generation errors
    """
    # Check if image generation is enabled
    if not config.ENABLE_IMAGE_GENERATION:
        raise HTTPException(
            status_code=503,
            detail="Image generation is disabled"
        )

    try:
        # Validate prompt
        if not prompt or len(prompt.strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail="Prompt cannot be empty"
            )

        if len(prompt) > config.IMAGE_MAX_PROMPT_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Prompt too long. Maximum {config.IMAGE_MAX_PROMPT_LENGTH} characters."
            )

        # Validate dimensions
        width = max(512, min(1024, width))
        height = max(512, min(1024, height))

        # Import Modal and lookup function
        try:
            import modal

            ImageGenerator = modal.Cls.from_name(config.MODAL_APP_NAME, "FluxImageGenerator")
            generator = ImageGenerator()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Image generation service unavailable: {str(e)}. Make sure the Modal app is deployed.",
            )

        # Generate image
        try:
            result = generator.generate_image.remote(
                prompt=prompt.strip(),
                width=width,
                height=height,
                num_inference_steps=config.IMAGE_NUM_INFERENCE_STEPS,
                guidance_scale=config.IMAGE_GUIDANCE_SCALE,
            )

            if not result.get("success"):
                raise HTTPException(
                    status_code=500,
                    detail=f"Image generation failed: {result.get('error', 'Unknown error')}"
                )

            # Generate session ID and cache the generated image
            session_id = str(uuid.uuid4())
            expiry_time = datetime.now() + timedelta(hours=config.IMAGE_CACHE_EXPIRY_HOURS)
            image_cache[session_id] = (
                result["image_bytes"],
                "image/png",
                expiry_time
            )

            # Clean up expired entries
            cleanup_expired_images()

            return ImageGenerationResponse(
                success=True,
                image_session_id=session_id,
                width=result.get("width"),
                height=result.get("height"),
            )

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Image generation failed: {str(e)}",
            )

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
