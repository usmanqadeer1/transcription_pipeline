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
from transcribe import (
    _run_model, _split_into_chunks, _transcribe_chunk, _init_worker,
    _has_gpu, _cache_dir_for, MIN_CHUNK_SECONDS,
)


def _format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS, rounded to the nearest second."""
    total_seconds = round(seconds)
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _format_segments(segments: list) -> list:
    """Convert raw-seconds start/end (as produced by _run_model/_transcribe_chunk)
    into HH:MM:SS strings for the final output."""
    return [
        {"start": _format_timestamp(s["start"]), "end": _format_timestamp(s["end"]), "text": s["text"]}
        for s in segments
    ]


def transcribe_with_timestamps(path: str) -> dict:
    """Validate the audio, transcribe it (chunked if long), and return
    segments with timestamps. See transcribe.transcribe() for the worker
    selection and chunking logic this reuses."""
    info = accept_audio(path)
    duration = info["duration_sec"]

    if duration <= MIN_CHUNK_SECONDS:
        print(f"Transcribing {duration:.1f}s of audio...", file=sys.stderr)
        language, segments = _run_model(path)
        return {"language": language, "segments": _format_segments(segments)}

    if _has_gpu():
        workers = 1
    else:
        auto_cap = max(1, cpu_count() // 2 - 1)
        workers = min(auto_cap, max(1, int(duration // MIN_CHUNK_SECONDS)))
        # Not short-circuiting when this resolves to 1 worker — see
        # transcribe.transcribe() for why: the file is already long enough
        # (past MIN_CHUNK_SECONDS) that resumable chunking is worth it even
        # without a parallelism gain.

    # Chunk COUNT is not the same as worker count — see transcribe.transcribe()
    # for why: even one worker (GPU) should get many checkpointed chunks on a
    # long file, not one giant chunk with no resumability benefit.
    num_chunks = max(workers, int(duration // MIN_CHUNK_SECONDS))
    chunk_length = duration / num_chunks
    cache_dir = _cache_dir_for(path)
    tmp_dir = tempfile.mkdtemp(prefix="transcribe_chunks_")
    try:
        chunks = _split_into_chunks(path, duration, chunk_length, tmp_dir, cache_dir)
        workers = min(workers, len(chunks))
        mode = "1 GPU worker, model loaded once" if _has_gpu() else \
            f"{workers} CPU worker process(es) ({cpu_count()} cores available)"
        print(f"Transcribing {duration:.1f}s of audio across {len(chunks)} chunks using {mode}...",
              file=sys.stderr)

        language = None
        all_segments = []
        failed_ranges = []
        with Pool(workers, initializer=_init_worker) as pool:
            for i, result in enumerate(pool.imap(_transcribe_chunk, chunks), start=1):
                if result.get("error"):
                    failed_ranges.append((result["nominal_start"], result["nominal_end"], result["error"]))
                    print(f"  chunk {i}/{len(chunks)} FAILED: {result['error']}", file=sys.stderr)
                else:
                    print(f"  chunk {i}/{len(chunks)} done", file=sys.stderr)
                language = language or result.get("language")
                all_segments.extend(result["segments"])

        if failed_ranges:
            for start_s, end_s, err in failed_ranges:
                print(f"Warning: {start_s:.1f}s-{end_s:.1f}s failed and is missing "
                      f"from the transcript ({err}). Rerun the same file to retry just this part.",
                      file=sys.stderr)
        else:
            shutil.rmtree(cache_dir, ignore_errors=True)

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
