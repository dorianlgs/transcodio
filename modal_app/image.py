"""Modal container image definition with all dependencies."""

import modal
from pathlib import Path

# Create the container image with all required dependencies
stt_image = (
    modal.Image.debian_slim(python_version="3.11")
    # Install system dependencies for audio processing
    .apt_install(
        "ffmpeg",
        "libsndfile1",
    )
    # Install Python packages for ML and audio
    .pip_install(
        "torch>=2.1.0",
        "torchaudio>=2.1.0",
        "transformers>=4.53.0",
        "accelerate>=0.30.0",
        "ffmpeg-python>=0.2.0",
        "numpy>=1.24.0",
        "soundfile>=0.12.0",
    )
    # Add config file to the image
    .add_local_file(
        str(Path(__file__).parent.parent / "config.py"),
        "/root/config.py"
    )
)
