"""CLI tool to transcribe audio files using Modal GPU backend."""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional
import modal


def validate_audio_file(file_path: Path) -> None:
    """Validate that the audio file exists and is readable."""
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")
    if file_path.stat().st_size == 0:
        raise ValueError(f"Audio file is empty: {file_path}")


def transcribe_streaming(model, audio_bytes: bytes, output_file: Optional[Path] = None) -> str:
    """Transcribe audio using streaming mode with real-time output."""
    full_transcription = ""

    for chunk in model.transcribe_stream.remote_gen(audio_bytes):
        data = json.loads(chunk)

        if data["type"] == "metadata":
            print(f"Audio duration: {data.get('duration', 'unknown')}s")
            print(f"Language: {data.get('language', 'unknown')}")
            print("-" * 70)
        elif data["type"] == "segment":
            segment_text = f"[{data['start']:>6.2f}s - {data['end']:>6.2f}s] {data['text']}"
            print(segment_text)
        elif data["type"] == "complete":
            full_transcription = data['text']
            print("-" * 70)
            print("\nComplete transcription:")
            print(full_transcription)
        elif data["type"] == "error":
            raise RuntimeError(f"Transcription error: {data['error']}")

    if output_file:
        output_file.write_text(full_transcription, encoding='utf-8')
        print(f"\nTranscription saved to: {output_file}")

    return full_transcription


def transcribe_non_streaming(model, audio_bytes: bytes, output_file: Optional[Path] = None) -> str:
    """Transcribe audio using non-streaming mode (faster for single result)."""
    result = model.transcribe.remote(audio_bytes)
    data = json.loads(result)

    if data["type"] == "error":
        raise RuntimeError(f"Transcription error: {data['error']}")

    transcription = data['text']
    metadata = data.get('metadata', {})

    print(f"Audio duration: {metadata.get('duration', 'unknown')}s")
    print(f"Language: {metadata.get('language', 'unknown')}")
    print("-" * 70)
    print(transcription)
    print("-" * 70)

    if output_file:
        output_file.write_text(transcription, encoding='utf-8')
        print(f"\nTranscription saved to: {output_file}")

    return transcription


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using Transcodio's Modal GPU backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transcribe a single file with streaming
  python transcribe_file.py audio.mp3

  # Transcribe without streaming (faster for complete results)
  python transcribe_file.py audio.mp3 --no-stream

  # Save output to file
  python transcribe_file.py audio.mp3 -o transcription.txt

  # Process multiple files
  python transcribe_file.py audio1.mp3 audio2.wav audio3.ogg

  # Use custom Modal app name
  python transcribe_file.py audio.mp3 --app-name my-custom-app
        """
    )

    parser.add_argument(
        'files',
        nargs='+',
        type=Path,
        help='Audio file(s) to transcribe (supports: MP3, WAV, M4A, FLAC, OGG, WebM)'
    )

    parser.add_argument(
        '-o', '--output',
        type=Path,
        help='Save transcription to file (for single file input only)'
    )

    parser.add_argument(
        '--no-stream',
        action='store_true',
        help='Use non-streaming mode (faster if you only need final result)'
    )

    parser.add_argument(
        '--app-name',
        default='transcodio-app',
        help='Modal app name (default: transcodio-app)'
    )

    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress progress output, only show final transcription'
    )

    args = parser.parse_args()

    # Validate arguments
    if args.output and len(args.files) > 1:
        parser.error("--output can only be used with a single input file")

    try:
        # Connect to Modal
        if not args.quiet:
            print(f"Connecting to Modal app: {args.app_name}...")
        WhisperModel = modal.Cls.from_name(args.app_name, "WhisperModel")
        model = WhisperModel()
        if not args.quiet:
            print("Connected successfully!\n")

        # Process each file
        for file_path in args.files:
            try:
                if len(args.files) > 1:
                    print(f"\n{'='*70}")
                    print(f"Processing: {file_path.name}")
                    print(f"{'='*70}")

                # Validate and read audio file
                validate_audio_file(file_path)

                if not args.quiet:
                    file_size_mb = file_path.stat().st_size / (1024 * 1024)
                    print(f"File: {file_path}")
                    print(f"Size: {file_size_mb:.2f} MB")

                with open(file_path, "rb") as f:
                    audio_bytes = f.read()

                # Determine output file path
                output_file = None
                if args.output:
                    output_file = args.output
                elif len(args.files) > 1:
                    # Auto-generate output filename for batch processing
                    output_file = file_path.with_suffix('.txt')

                # Transcribe
                if args.no_stream:
                    transcribe_non_streaming(model, audio_bytes, output_file)
                else:
                    transcribe_streaming(model, audio_bytes, output_file)

            except Exception as e:
                print(f"\nError processing {file_path}: {e}", file=sys.stderr)
                if len(args.files) == 1:
                    sys.exit(1)
                continue

        if len(args.files) > 1:
            print(f"\n{'='*70}")
            print(f"Completed processing {len(args.files)} file(s)")
            print(f"{'='*70}")

    except KeyboardInterrupt:
        print("\n\nTranscription cancelled by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\nFatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
