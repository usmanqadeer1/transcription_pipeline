#!/usr/bin/env python3
"""
transcribe.py — transcribes an audio file to plain text with faster-whisper.

Long files are split into chunks and transcribed via multiprocessing.Pool.
On CPU that pool has multiple workers running in parallel, for speed. On GPU
it's Pool(1) — one process, one model — since multiple processes contending
for a single GPU (each loading its own copy of the model into VRAM) isn't a
speedup. Both cases are the same code path; GPU is just "the CPU path with
exactly one worker," which is also how it's tested (see the test suite).
Chunking either way also gives fault isolation and resumable caching: a
failed chunk doesn't discard everything else, and a rerun of the same file
skips chunks that already succeeded.

Usage:
    python transcribe.py audio.wav
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from multiprocessing import Pool, cpu_count

from accept_audio import accept_audio

MIN_CHUNK_SECONDS = 150           # never split into chunks shorter than this — below this,
                                  # per-worker model-load overhead outweighs the parallelism gain
SNAP_SEARCH_SECONDS = 10         # how far to look for a nearby silence gap when placing a cut
READ_PAD_SECONDS = 1.0           # small safety margin on each chunk's audio read, so a sound
                                  # landing exactly on the cut sample isn't clipped
CHUNK_RETRIES = 2                # attempts per chunk before it's marked failed instead of
                                  # crashing the whole run

_worker_model = None   # set once per worker process by _init_worker(), reused for every
                        # chunk that worker processes


def _cache_dir_for(path: str) -> str:
    """A persistent, deterministic directory for this file's chunk results.

    Keyed off the file's absolute path, size, and modified time, so re-running
    the same file finds the same cache dir (and a different file, or an edited
    one, gets a fresh one). This is what makes resuming after a crash possible:
    each chunk's result is saved here as soon as it succeeds, so a rerun can
    skip chunks that already finished instead of re-transcribing the whole
    file from scratch.
    """
    stat = os.stat(path)
    key_input = f"{os.path.abspath(path)}:{stat.st_size}:{int(stat.st_mtime)}"
    key = hashlib.sha1(key_input.encode()).hexdigest()[:16]
    cache_dir = os.path.join(tempfile.gettempdir(), "transcribe_cache", key)
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _has_gpu() -> bool:
    """Detect a CUDA GPU via ctranslate2 (faster-whisper's inference backend) —
    no extra dependency needed, since ctranslate2 already ships with faster-whisper."""
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def _load_model():
    """Load a whisper model, GPU or CPU depending on what's available."""
    from faster_whisper import WhisperModel

    if _has_gpu():
        return WhisperModel("base", device="cuda", compute_type="float16")
    return WhisperModel("base", device="cpu", compute_type="int8")     # int8 = lighter, CPU-friendly


def _init_worker():
    """multiprocessing.Pool initializer: runs once when a worker process
    starts, before it's given any chunks. Loading the model here (into the
    module-level _worker_model, so _transcribe_chunk can find it) means each
    worker pays the ~5-6s model-load cost once for its whole lifetime, not
    once per chunk it happens to be assigned."""
    global _worker_model
    _worker_model = _load_model()


def _run_model(path: str, model=None):
    """Transcribe a single file/chunk. Returns (language, segments).
    If model is None, a fresh one is loaded just for this call."""
    if model is None:
        model = _load_model()
    segments, info = model.transcribe(path, vad_filter=True)  # vad_filter skips silence
    result = [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
               for s in segments]
    return info.language, result


def _find_silence_midpoints(path: str) -> list:
    """Run ffmpeg's silence detector once over the whole file (fast — this does
    not run the transcription model) to find natural pause points. Chunk
    boundaries are snapped to these instead of landing at a blind time mark,
    so a cut almost never falls in the middle of a sentence.

    d=1.0 (a full second of near-silence) is deliberately conservative: it's
    meant to catch pauses BETWEEN sentences, not the brief half-beat after a
    comma or between words, which would risk snapping a cut into the middle
    of a sentence instead of after one.
    """
    result = subprocess.run(
        ["ffmpeg", "-i", path, "-af", "silencedetect=noise=-30dB:d=1.0", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    midpoints = []
    start = None
    for line in result.stderr.splitlines():
        if "silence_start:" in line:
            start = float(line.split("silence_start:")[1].strip())
        elif "silence_end:" in line and start is not None:
            end = float(line.split("silence_end:")[1].split("|")[0].strip())
            midpoints.append((start + end) / 2)
            start = None
    return midpoints


def _snap_to_silence(boundary: float, silence_midpoints: list, max_shift: float) -> float:
    """Move a nominal chunk boundary to the nearest silence gap within max_shift
    seconds, if there is one. Otherwise leave it where it was."""
    nearby = [t for t in silence_midpoints if abs(t - boundary) <= max_shift]
    return min(nearby, key=lambda t: abs(t - boundary)) if nearby else boundary


def _split_into_chunks(path: str, duration: float, chunk_length: float, tmp_dir: str, cache_dir: str):
    """Cut a long file into overlapping chunks so they can be transcribed by
    the worker pool (parallel on CPU, sequential on GPU).

    """
    silence_midpoints = _find_silence_midpoints(path)
    raw_cuts = range(int(chunk_length), max(int(duration), 1), int(chunk_length))
    boundaries = [0.0]
    boundaries += [_snap_to_silence(t, silence_midpoints, SNAP_SEARCH_SECONDS) for t in raw_cuts]
    boundaries.append(duration)
    boundaries = sorted(set(boundaries))   # snapping can rarely produce duplicates/reordering

    chunks = []
    for nominal_start, nominal_end in zip(boundaries, boundaries[1:]):
        cache_file = os.path.join(cache_dir, f"chunk_{nominal_start:.2f}.json")
        read_start = max(nominal_start - READ_PAD_SECONDS, 0)

        if os.path.exists(cache_file):
            chunks.append((read_start, nominal_start, nominal_end, None, cache_file))
            continue

        read_end = min(nominal_end + READ_PAD_SECONDS, duration)
        chunk_path = os.path.join(tmp_dir, f"chunk_{nominal_start:.2f}.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-ss", str(read_start), "-i", path,
             "-t", str(read_end - read_start),
             chunk_path],
            check=True,
        )
        chunks.append((read_start, nominal_start, nominal_end, chunk_path, cache_file))
    return chunks


def _transcribe_chunk(chunk: tuple) -> dict:
    """Pool worker entry point: transcribe one chunk and keep only the
    segments this chunk "owns" — those whose midpoint falls inside its
    nominal (non-overlap) window. This is what prevents a sentence near a cut
    point from being emitted twice: both neighboring chunks hear it (thanks
    to the overlap), but only the one whose window contains its midpoint
    keeps it.
    """
    read_start, nominal_start, nominal_end, chunk_path, cache_file = chunk

    if chunk_path is None:
        with open(cache_file) as f:
            return json.load(f)

    error = None
    for attempt in range(1, CHUNK_RETRIES + 1):
        try:
            language, segments = _run_model(chunk_path, model=_worker_model)
            error = None
            break
        except Exception as e:
            error = str(e)

    if error is not None:
        # Deliberately NOT cached to disk: caching a failure would make it
        # permanent — a rerun would keep loading the same "failed" result
        # forever instead of getting a fresh chance to retry this chunk.
        return {"language": None, "segments": [], "error": error,
                "nominal_start": nominal_start, "nominal_end": nominal_end}

    kept = []
    for s in segments:
        abs_start = s["start"] + read_start
        abs_end = s["end"] + read_start
        midpoint = (abs_start + abs_end) / 2
        if nominal_start <= midpoint < nominal_end:
            kept.append({"start": round(abs_start, 2), "end": round(abs_end, 2), "text": s["text"]})
    result = {"language": language, "segments": kept}

    with open(cache_file, "w") as f:
        json.dump(result, f)
    return result


def transcribe_segments(path: str) -> dict:
    """Validate the audio, transcribe it (chunked if long), and return
    {"language": ..., "segments": [{"start": float, "end": float, "text": ...}]}
    with raw (unformatted) second-based timestamps.

    This is the single shared implementation of the whole pipeline — worker
    selection, chunking, the Pool loop, fault isolation, resumable caching.
    transcribe() below just joins the segments into one string; anything
    that wants timestamps (see transcribe_with_timestamps.py) calls this
    directly and formats the segments itself, instead of duplicating any of
    this orchestration.
    """
    info = accept_audio(path)
    duration = info["duration_sec"]

    if duration <= MIN_CHUNK_SECONDS:
        # Too short to be worth chunking either way.
        print(f"Transcribing {duration:.1f}s of audio...", file=sys.stderr)
        language, segments = _run_model(path)
        return {"language": language, "segments": segments}

    if _has_gpu():
        # One GPU, one worker
        workers = 1
    else:
        auto_cap = max(1, cpu_count() // 2 - 1)
        workers = min(auto_cap, max(1, int(duration // MIN_CHUNK_SECONDS)))

    num_chunks = max(workers, int(duration // MIN_CHUNK_SECONDS))
    chunk_length = duration / num_chunks
    cache_dir = _cache_dir_for(path)
    tmp_dir = tempfile.mkdtemp(prefix="transcribe_chunks_")
    try:
        # Timed separately from the pool below: silence detection and
        # chunk-splitting both run serially, on a single core, before any
        # worker starts — so they don't get faster just because you added
        # more workers, and can end up dominating the total time on long files.
        prep_start = time.perf_counter()
        chunks = _split_into_chunks(path, duration, chunk_length, tmp_dir, cache_dir)
        prep_elapsed = time.perf_counter() - prep_start

        # Snapping chunk boundaries to silence can occasionally merge two
        # planned chunks into one, so the actual chunk count may come in
        # slightly under the plan — never start more workers than chunks.
        workers = min(workers, len(chunks))
        mode = "1 GPU worker, model loaded once" if _has_gpu() else \
            f"{workers} CPU worker process(es) ({cpu_count()} cores available)"
        print(f"Prep (silence detection + splitting): {prep_elapsed:.1f}s", file=sys.stderr)
        print(f"Transcribing {duration:.1f}s of audio across {len(chunks)} chunks using {mode}...",
              file=sys.stderr)

        pool_start = time.perf_counter()
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
        pool_elapsed = time.perf_counter() - pool_start
        print(f"Transcription (pool): {pool_elapsed:.1f}s", file=sys.stderr)

        if failed_ranges:
            # Leave cache_dir in place on failure: the successful chunks are
            # saved, so rerunning this same file only retries what failed
            # instead of starting over from scratch.
            for start_s, end_s, err in failed_ranges:
                print(f"Warning: {start_s:.1f}s-{end_s:.1f}s failed and is missing "
                      f"from the transcript ({err}). Rerun the same file to retry just this part.",
                      file=sys.stderr)
        else:
            shutil.rmtree(cache_dir, ignore_errors=True)

        return {"language": language, "segments": all_segments}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def transcribe(path: str) -> str:
    """Validate the audio, transcribe it (chunked if long), and return the transcript text."""
    result = transcribe_segments(path)
    return " ".join(s["text"] for s in result["segments"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_file")
    args = parser.parse_args()

    try:
        start = time.perf_counter()
        text = transcribe(args.audio_file)
        elapsed = time.perf_counter() - start
    except (ValueError, EnvironmentError, ImportError, subprocess.CalledProcessError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Transcription took {elapsed:.1f}s", file=sys.stderr)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
