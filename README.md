# Dutch Two-Speaker Diarization

Accuracy-first Dutch interview transcription and diarization for long files with exactly two speakers.

## What This Pipeline Does

- Normalizes the full audio file once before chunking to reduce volume-related language drift.
- Converts audio to a consistent mono 16 kHz WAV working format.
- Detects silences and splits long files into overlap-aware chunks.
- Runs speaker diarization with `pyannote` constrained to two speakers.
- Runs transcription with `Whisper large-v3` by default.
- Merges chunked transcription back into one speaker-labelled transcript and JSON result.

## Runtime Design

- Default ASR model: `Whisper large-v3`
- Default transcription backend: `faster-whisper`
- Default diarization backend: `pyannote/speaker-diarization-3.1`
- Default runtime target on Windows + AMD: CPU-safe first
- Optional acceleration path: `whisper.cpp` with Vulkan, while keeping `large-v3` as the target model

## Prerequisites

1. Install Python 3.11 or newer.
2. Install `ffmpeg` and ensure both `ffmpeg` and `ffprobe` are available on `PATH`.
3. Create and activate a virtual environment.
4. Install the package:

```bash
pip install -e .[dev]
```

5. Request access to the Hugging Face `pyannote/speaker-diarization-3.1` model and place your token in `.env`.

## Environment

Copy `.env.example` to `.env` and fill in the values you need. Important variables:

- `PYANNOTE_AUTH_TOKEN`: required for diarization.
- `TRANSCRIPTION_BACKEND`: `faster-whisper` or `whisper.cpp`.
- `WHISPER_MODEL`: keep this on `large-v3` for best accuracy.
- `WHISPER_DEVICE`: default `cpu`.
- `TARGET_CHUNK_SECONDS`: target chunk size before overlap.
- `CHUNK_OVERLAP_SECONDS`: overlap used to preserve sentence continuity.
- `ENABLE_GLOBAL_NORMALIZATION`: set to `false` to compare raw vs normalized runs.

If you later want to test `whisper.cpp`, also set:

- `WHISPER_CPP_PATH`
- `WHISPER_CPP_MODEL_PATH`

## Usage

Process a single file:

```bash
python -m src.cli "path/to/interview.wav"
```

Process a directory:

```bash
python -m src.cli "path/to/folder"
```

Useful overrides:

```bash
python -m src.cli "path/to/interview.wav" --chunk-seconds 480 --chunk-overlap 10
python -m src.cli "path/to/interview.wav" --skip-normalization
python -m src.cli "path/to/interview.wav" --backend whisper.cpp
```

## Output

For each input file the pipeline writes:

- `outputs/<stem>/transcript.txt`: human-readable speaker-labelled transcript
- `outputs/<stem>/result.json`: chunk manifest, diarization turns, and merged transcript segments
- `work/<stem>/...`: normalized audio, chunk WAVs, and intermediate artifacts

## Accuracy Notes

- Whole-file normalization is conservative by default: loudness normalization plus peak control, not heavy compression.
- Diarization runs on the normalized full file so speaker labels stay consistent across chunked transcription.
- Chunk overlap is used to avoid losing sentence starts/ends at boundaries.
- The default prompt biases transcription toward Dutch while allowing English business terms inside Dutch sentences.

## Testing

Run the unit tests with:

```bash
pytest
```

## Secret Handling

- Do not commit `.env`.
- Do not place real API keys or tokens in tracked files.
