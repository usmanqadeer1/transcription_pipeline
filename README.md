# Audio Transcription Pipeline

A simple, script-based pipeline that validates an audio file, transcribes it to text, and returns per-segment timestamps — built with engineering judgment as the focus rather than model training.

## Overview

The pipeline is split into three small, independently runnable scripts, each responsible for one step:

| Script | Responsibility |
|---|---|
| [`accept_audio.py`](accept_audio.py) | Validates that a file is genuinely readable audio |
| [`transcribe.py`](transcribe.py) | Transcribes audio to plain text |
| [`transcribe_with_timestamps.py`](transcribe_with_timestamps.py) | Transcribes audio to text with per-segment `HH:MM:SS` timestamps |

`transcribe.py` and `transcribe_with_timestamps.py` both import `accept_audio()` from `accept_audio.py` rather than duplicating it, and `transcribe_with_timestamps.py` reuses the transcription/chunking internals from `transcribe.py` — there is a single implementation of each piece of logic.

## Setup

**Prerequisites**
- Python 3.9+
- [ffmpeg](https://ffmpeg.org/download.html) installed and available on `PATH` (provides `ffprobe`, used for audio validation, silence detection, and chunk splitting)

**Install dependencies**
```bash
pip install -r requirements.txt
```

## Usage

```bash
python accept_audio.py audio_files/recording01.m4a
python transcribe.py audio_files/recording01.m4a
python transcribe_with_timestamps.py audio_files/recording01.m4a
```

Sample audio files for testing are provided in [`audio_files/`](audio_files/).

`transcribe_with_timestamps.py` outputs JSON:
```json
{
  "language": "en",
  "segments": [
    {"start": "00:00:00", "end": "00:00:05", "text": "Good morning everyone, thank you for joining the call."},
    {"start": "00:00:05", "end": "00:00:08", "text": "Today we will discuss the quarterly results."}
  ]
}
```

## Design decisions

### Format handling
`accept_audio()` doesn't trust file extensions. It runs the file through `ffprobe` and inspects the actual stream data — rejecting anything without a genuine audio stream, and rejecting video files (their primary stream is video, not audio) even if they happen to carry an audio track. Because `ffprobe`/`ffmpeg` do the decoding, any format they support (WAV, MP3, M4A, FLAC, OGG, etc.) works without format-specific code.

### Transcription model
Both transcription scripts use [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2-backed Whisper) with the `base` checkpoint. `vad_filter=True` skips silence during decoding, which improves both speed and segment quality.

### Handling long audio
Whisper's own architecture already processes audio in internal 30-second windows and stitches the result into one continuous transcript — a single `model.transcribe()` call already handles multi-hour files correctly with no manual splitting required for *correctness*.

Chunking and parallelism were added on top of that for *speed*, since faster-whisper's internal windowing runs sequentially within a single process:

- **CPU path**: long files are split into as many chunks as there are usable CPU cores, and each chunk is transcribed in a separate process (`multiprocessing.Pool`).
  - Chunk boundaries are snapped to detected silence gaps (via `ffmpeg`'s `silencedetect`) rather than blind time marks, so a cut essentially never lands mid-sentence.
  - Each chunk keeps only the segments whose midpoint falls inside its own boundary window ("midpoint ownership"), which is what prevents a sentence near a cut point from being transcribed twice.
  - Worker count is chosen automatically as `max(1, cpu_count() // 2 - 1)` — approximating physical core count and leaving one core free — capped further so no chunk is shorter than `MIN_CHUNK_SECONDS`, since very short chunks pay a fixed per-worker model-load cost that outweighs the benefit of splitting.
- **GPU path**: if a CUDA GPU is detected (via `ctranslate2.get_cuda_device_count()`), chunking is skipped entirely and the whole file is transcribed in a single pass on the GPU. Multiple CPU processes contending for one GPU — each loading its own copy of the model into VRAM — is not a valid way to parallelize; that strategy only applies to CPU.

### What's out of scope
Speaker diarization, LLM-based transcript cleanup, and running transcription as an async background job behind an API are all reasonable next steps for a production service, but are outside the scope of this exercise, which focuses on the core accept → transcribe → timestamp pipeline and the reasoning behind the format/long-audio handling.
