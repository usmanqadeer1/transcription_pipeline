#!/usr/bin/env python3
"""
transcribe.py — transcribes an audio file to plain text with faster-whisper.
On CPU, long files are chunked and transcribed in parallel across processes.
On GPU, chunking is skipped and the whole file goes through in one pass —
multiple processes fighting over one GPU (and each loading its own copy of
the model into VRAM) is not a speedup, so that strategy only applies to CPU.

Usage:
    python transcribe.py audio.wav
"""

import argparse
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


def _has_gpu() -> bool:
    """Detect a CUDA GPU via ctranslate2 (faster-whisper's inference backend) —
    no extra dependency needed, since ctranslate2 already ships with faster-whisper."""
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def _run_model(path: str):
    """Transcribe a single file/chunk in this process. Returns (language, segments)."""
    from faster_whisper import WhisperModel

    if _has_gpu():
        model = WhisperModel("base", device="cuda", compute_type="float16")
    else:
        model = WhisperModel("base", device="cpu", compute_type="int8")     # int8 = lighter, CPU-friendly
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


def _split_into_chunks(path: str, duration: float, chunk_length: float, tmp_dir: str):
    """Cut a long file into overlapping chunks so they can be transcribed in parallel.

    chunk_length is chosen by the caller based on how many CPU cores are
    available (see transcribe()), not a fixed constant — this splits the
    audio into roughly one chunk per worker.

    Interior cut points are snapped to nearby silence (searching up to
    SNAP_SEARCH_SECONDS away) so they land between sentences rather than
    through one. Each chunk then only needs a small READ_PAD_SECONDS margin
    on either side of its nominal [nominal_start, nominal_end) window, since
    the cut itself already sits in near-silence (clamped at the file's
    edges). Returns a list of (read_start, nominal_start, nominal_end, chunk_path).
    """
    silence_midpoints = _find_silence_midpoints(path)
    raw_cuts = range(int(chunk_length), max(int(duration), 1), int(chunk_length))
    boundaries = [0.0]
    boundaries += [_snap_to_silence(t, silence_midpoints, SNAP_SEARCH_SECONDS) for t in raw_cuts]
    boundaries.append(duration)
    boundaries = sorted(set(boundaries))   # snapping can rarely produce duplicates/reordering

    chunks = []
    for nominal_start, nominal_end in zip(boundaries, boundaries[1:]):
        read_start = max(nominal_start - READ_PAD_SECONDS, 0)
        read_end = min(nominal_end + READ_PAD_SECONDS, duration)
        chunk_path = os.path.join(tmp_dir, f"chunk_{nominal_start:.2f}.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-ss", str(read_start), "-i", path,
             "-t", str(read_end - read_start),
             chunk_path],
            check=True,
        )
        chunks.append((read_start, nominal_start, nominal_end, chunk_path))
    return chunks


def _transcribe_chunk(chunk: tuple) -> dict:
    """Worker entry point: transcribe one chunk and keep only the segments this
    chunk "owns" — those whose midpoint falls inside its nominal (non-overlap)
    window. This is what prevents a sentence near a cut point from being
    emitted twice: both neighboring chunks hear it (thanks to the overlap),
    but only the one whose window contains its midpoint keeps it.

    Because boundaries are snapped to silence first, a cut landing mid-sentence
    should be rare. It can still happen if no silence gap exists near a
    boundary (continuous speech for longer than one chunk) — a residual
    limitation of any fixed-size chunking scheme.
    """
    read_start, nominal_start, nominal_end, chunk_path = chunk
    language, segments = _run_model(chunk_path)

    kept = []
    for s in segments:
        abs_start = s["start"] + read_start
        abs_end = s["end"] + read_start
        midpoint = (abs_start + abs_end) / 2
        if nominal_start <= midpoint < nominal_end:
            kept.append({"start": round(abs_start, 2), "end": round(abs_end, 2), "text": s["text"]})
    return {"language": language, "segments": kept}


def transcribe(path: str) -> str:
    """Validate the audio, transcribe it (in parallel if long), and return the transcript text.
    Worker count is always chosen automatically — see the cap below."""
    info = accept_audio(path)
    duration = info["duration_sec"]

    if _has_gpu():
        # See the module docstring: CPU-process parallelism doesn't translate
        # to a single GPU, so just run the whole file through in one pass.
        print(f"GPU detected — transcribing {duration:.1f}s of audio in a single pass...",
              file=sys.stderr)
        _, segments = _run_model(path)
        return " ".join(s["text"] for s in segments)

    # Cap workers at ~physical cores minus one, not raw logical cpu_count():
    # whisper inference is CPU-bound native code, so hyperthreads buy little
    # extra throughput while doubling memory (each worker loads its own copy
    # of the model), and leaving one core free keeps the OS/ffmpeg responsive.
    # cpu_count() // 2 approximates physical cores (correct for the common
    # 2-way-hyperthreading case; a rough estimate otherwise). For long audio
    # this cap is what ends up binding — there's no upper limit on chunk
    # count worth raising it toward, since more workers than this just adds
    # contention rather than throughput.
    auto_cap = max(1, cpu_count() // 2 - 1)

    # Never split into chunks shorter than MIN_CHUNK_SECONDS — for a short
    # file that's 1 worker, i.e. the plain single-pass path below.
    workers = min(auto_cap, max(1, int(duration // MIN_CHUNK_SECONDS)))

    if workers <= 1:
        print(f"Transcribing {duration:.1f}s of audio...", file=sys.stderr)
        _, segments = _run_model(path)
        return " ".join(s["text"] for s in segments)

    chunk_length = duration / workers
    tmp_dir = tempfile.mkdtemp(prefix="transcribe_chunks_")
    try:
        # Timed separately from the parallel pool below: silence detection and
        # chunk-splitting both run serially, on a single core, before any
        # worker starts — so they don't get faster just because you added
        # more workers, and can end up dominating the total time on long files.
        prep_start = time.perf_counter()
        chunks = _split_into_chunks(path, duration, chunk_length, tmp_dir)
        prep_elapsed = time.perf_counter() - prep_start

        # Snapping chunk boundaries to silence can occasionally merge two
        # planned chunks into one, so the actual chunk count may come in
        # slightly under the plan — never start more workers than chunks.
        workers = min(workers, len(chunks))
        print(f"Prep (silence detection + splitting): {prep_elapsed:.1f}s", file=sys.stderr)
        print(f"Transcribing {duration:.1f}s of audio across {len(chunks)} chunks "
              f"using {workers} worker process(es) ({cpu_count()} CPU cores available)...",
              file=sys.stderr)

        pool_start = time.perf_counter()
        all_segments = []
        with Pool(workers) as pool:
            for i, result in enumerate(pool.imap(_transcribe_chunk, chunks), start=1):
                print(f"  chunk {i}/{len(chunks)} done", file=sys.stderr)
                all_segments.extend(result["segments"])
        pool_elapsed = time.perf_counter() - pool_start
        print(f"Parallel transcription: {pool_elapsed:.1f}s", file=sys.stderr)

        return " ".join(s["text"] for s in all_segments)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


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
