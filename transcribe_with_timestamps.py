#!/usr/bin/env python3
"""
transcribe_with_timestamps.py — same pipeline as transcribe.py, but returns
segments with start/end timestamps instead of one flat string.

All chunking/worker/caching logic lives in transcribe.transcribe_segments();
this file only formats its output as HH:MM:SS timestamps.

Usage:
    python transcribe_with_timestamps.py audio.wav
"""

import argparse
import json
import subprocess
import sys
import time

from transcribe import transcribe_segments


def _format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS, rounded to the nearest second."""
    total_seconds = round(seconds)
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def transcribe_with_timestamps(path: str) -> dict:
    """Transcribe the audio and return segments with HH:MM:SS timestamps."""
    result = transcribe_segments(path)
    segments = [
        {"start": _format_timestamp(s["start"]), "end": _format_timestamp(s["end"]), "text": s["text"]}
        for s in result["segments"]
    ]
    return {"language": result["language"], "segments": segments}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_file")
    args = parser.parse_args()

    try:
        start = time.perf_counter()
        result = transcribe_with_timestamps(args.audio_file)
        elapsed = time.perf_counter() - start
    except (ValueError, EnvironmentError, ImportError, subprocess.CalledProcessError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Transcription took {elapsed:.1f}s", file=sys.stderr)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
