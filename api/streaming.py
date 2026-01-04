"""Server-Sent Events (SSE) utilities for streaming transcription."""

import json
from typing import AsyncIterator, Dict, Any
from sse_starlette.sse import EventSourceResponse


async def create_sse_response(
    event_generator: AsyncIterator[Dict[str, Any]]
) -> EventSourceResponse:
    """
    Create an SSE response from an async event generator.

    Args:
        event_generator: Async generator that yields event dictionaries

    Returns:
        EventSourceResponse for FastAPI
    """
    return EventSourceResponse(event_generator)


async def format_sse_event(event_type: str, data: Dict[str, Any]) -> str:
    """
    Format data as an SSE event.

    Args:
        event_type: Type of event (metadata, segment, complete, error)
        data: Event data to send

    Returns:
        Formatted SSE event string
    """
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


async def transcription_event_stream(
    segments_iterator: AsyncIterator[str],
) -> AsyncIterator[Dict[str, Any]]:
    """
    Convert Modal transcription segments into SSE events.

    Args:
        segments_iterator: Async iterator of JSON strings from Modal

    Yields:
        SSE event dictionaries
    """
    try:
        async for segment_json in segments_iterator:
            # Parse the JSON segment from Modal
            segment_data = json.loads(segment_json)
            event_type = segment_data.get("type", "unknown")

            # Yield as SSE event
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
        # Yield error event if something goes wrong
        yield {
            "event": "error",
            "data": json.dumps({
                "error": f"Streaming error: {str(e)}",
            })
        }
