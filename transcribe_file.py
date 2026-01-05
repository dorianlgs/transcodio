"""Simple script to transcribe a specific audio file."""
import modal
import json

# Read the audio file
audio_path = "WhatsApp Ptt 2026-01-05 at 1.24.34 PM.ogg"

with open(audio_path, "rb") as f:
    audio_bytes = f.read()

print(f"Transcribing {audio_path}...")
print("-" * 50)

# Lookup the Modal class using from_name
WhisperModel = modal.Cls.from_name("transcodio-app", "WhisperModel")

# Create instance and call the streaming transcription
model = WhisperModel()
for chunk in model.transcribe_stream.remote_gen(audio_bytes):
    data = json.loads(chunk)
    if data["type"] == "segment":
        print(f"[{data['start']:.2f}s - {data['end']:.2f}s] {data['text']}")
    elif data["type"] == "complete":
        print("-" * 50)
        print(f"\nComplete transcription:\n{data['text']}")
    elif data["type"] == "error":
        print(f"Error: {data['error']}")
