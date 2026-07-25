#!/usr/bin/env python3
"""
accept_audio.py — validates that a file is genuinely readable audio, using
ffprobe rather than trusting the file extension.

Usage:
    python accept_audio.py audiofile
"""

import argparse
import json
import os
import shutil
import subprocess
import sys


def accept_audio(path: str) -> dict:
    """Validate the file is real, readable audio. Raises ValueError on any problem."""
    if not os.path.exists(path):
        raise ValueError(f"File not found: {path}")
    if os.path.getsize(path) == 0:
        raise ValueError("File is empty")
    if not shutil.which("ffprobe"):
        raise EnvironmentError("ffprobe not found. Install ffmpeg (provides ffprobe).")

    # Ask ffprobe what's actually inside the file — codec-level truth,
    # not a guess based on the file extension.
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise ValueError(f"Not a readable media file: {result.stderr.strip()}")

    probe = json.loads(result.stdout)
    streams = probe.get("streams", [])
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    video_streams = [s for s in streams if s.get("codec_type") == "video"]

    if not audio_streams:
        raise ValueError("No audio stream found in this file")
    if video_streams:
        raise ValueError("File contains video — only audio files are accepted")

    stream = audio_streams[0]
    fmt = probe.get("format", {})
    return {
        "path": path,
        "container": fmt.get("format_name"),
        "codec": stream.get("codec_name"),
        "duration_sec": round(float(fmt.get("duration", 0)), 2),
        "sample_rate": int(stream.get("sample_rate", 0)),
        "channels": stream.get("channels"),
        "size_kb": round(os.path.getsize(path) / 1024, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_file")
    args = parser.parse_args()

    try:
        info = accept_audio(args.audio_file)
    except (ValueError, EnvironmentError) as e:
        print(f"Rejected: {e}", file=sys.stderr)
        return 1

    print(f"Accepted: {info}")
    return 0


if __name__ == "__main__":
    sys.exit(main())