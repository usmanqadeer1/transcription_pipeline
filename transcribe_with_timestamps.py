#!/usr/bin/env python3
"""
transcribe_with_timestamps.py — same pipeline as transcribe.py, but returns
segments with start/end timestamps instead of one flat string.

Usage:
    python transcribe_with_timestamps.py audio.wav
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from multiprocessing import Pool, cpu_count

from accept_audio import accept_audio
from transcribe import _run_model, _split_into_chunks, _transcribe_chunk, _has_gpu, MIN_CHUNK_SECONDS


def _format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS, rounded to the nearest second."""
    total_seconds = round(seconds)
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _format_segments(segments: list) -> list:
    """Convert raw-seconds start/end (as produced by _run_model/_transcribe_chunk)
    into HH:MM:SS,mmm strings for the final output."""
    return [
        {"start": _format_timestamp(s["start"]), "end": _format_timestamp(s["end"]), "text": s["text"]}
        for s in segments
    ]


def transcribe_with_timestamps(path: str) -> dict:
    """Validate the audio, transcribe it (in parallel if long), and return
    segments with timestamps. See transcribe.transcribe() for the worker
    selection and chunking logic this reuses."""
    info = accept_audio(path)
    duration = info["duration_sec"]

    if _has_gpu():
        # CPU-process parallelism doesn't translate to a single GPU — see
        # transcribe.py's module docstring. Run the whole file in one pass.
        print(f"GPU detected — transcribing {duration:.1f}s of audio in a single pass...",
              file=sys.stderr)
        language, segments = _run_model(path)
        return {"language": language, "segments": _format_segments(segments)}

    auto_cap = max(1, cpu_count() // 2 - 1)
    workers = min(auto_cap, max(1, int(duration // MIN_CHUNK_SECONDS)))

    if workers <= 1:
        print(f"Transcribing {duration:.1f}s of audio...", file=sys.stderr)
        language, segments = _run_model(path)
        return {"language": language, "segments": _format_segments(segments)}

    chunk_length = duration / workers
    tmp_dir = tempfile.mkdtemp(prefix="transcribe_chunks_")
    try:
        chunks = _split_into_chunks(path, duration, chunk_length, tmp_dir)
        workers = min(workers, len(chunks))
        print(f"Transcribing {duration:.1f}s of audio across {len(chunks)} chunks "
              f"using {workers} worker process(es) ({cpu_count()} CPU cores available)...",
              file=sys.stderr)

        language = None
        all_segments = []
        with Pool(workers) as pool:
            for i, result in enumerate(pool.imap(_transcribe_chunk, chunks), start=1):
                print(f"  chunk {i}/{len(chunks)} done", file=sys.stderr)
                language = language or result["language"]
                all_segments.extend(result["segments"])

        return {"language": language, "segments": _format_segments(all_segments)}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


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
