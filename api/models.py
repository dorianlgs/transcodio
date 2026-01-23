"""Pydantic models for API requests and responses."""

from typing import Optional, List
from pydantic import BaseModel, Field


class TranscriptionSegment(BaseModel):
    """A single segment of transcribed text."""

    id: int = Field(..., description="Segment ID")
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    text: str = Field(..., description="Transcribed text for this segment")
    speaker: Optional[str] = Field(None, description="Speaker label (e.g., 'Speaker 1')")


class TranscriptionResponse(BaseModel):
    """Response model for transcription."""

    text: str = Field(..., description="Complete transcribed text")
    language: str = Field(..., description="Detected language code")
    segments: List[TranscriptionSegment] = Field(
        default_factory=list, description="Individual segments with timestamps"
    )
    duration: Optional[float] = Field(None, description="Audio duration in seconds")


class TranscriptionStreamEvent(BaseModel):
    """Server-Sent Event for streaming transcription."""

    type: str = Field(..., description="Event type: metadata, segment, complete, or error")
    data: dict = Field(..., description="Event data")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")


class VoiceCloneResponse(BaseModel):
    """Response model for voice cloning."""

    success: bool = Field(..., description="Whether generation succeeded")
    audio_session_id: Optional[str] = Field(None, description="Session ID to retrieve generated audio")
    duration: Optional[float] = Field(None, description="Generated audio duration in seconds")
    error: Optional[str] = Field(None, description="Error message if failed")
