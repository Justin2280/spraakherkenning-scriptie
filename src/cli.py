from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.audio.preprocess import prepare_audio
from src.config import PipelineConfig
from src.diarization.engine import PyannoteDiarizationEngine
from src.merge.align import (
    assign_speakers_to_segments,
    deduplicate_overlap_segments,
    merge_adjacent_segments,
    render_transcript,
)
from src.transcription.engine import create_transcription_engine

AUDIO_EXTENSIONS = {
    ".aac",
    ".flac",
    ".m4a",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".ogg",
    ".wav",
    ".webm",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Accuracy-first Dutch two-speaker diarization and transcription pipeline.",
    )
    parser.add_argument("input", type=Path, help="Audio file or directory to process.")
    parser.add_argument("--working-dir", type=Path, help="Intermediate work directory.")
    parser.add_argument("--output-dir", type=Path, help="Output directory for final transcripts.")
    parser.add_argument(
        "--backend",
        choices=["faster-whisper", "whisper.cpp"],
        help="Transcription backend runtime.",
    )
    parser.add_argument("--whisper-device", help="Whisper runtime device, for example `cpu`.")
    parser.add_argument("--diarization-device", help="Pyannote runtime device, for example `cpu`.")
    parser.add_argument("--chunk-seconds", type=int, help="Target chunk length in seconds.")
    parser.add_argument("--chunk-overlap", type=int, help="Chunk overlap in seconds.")
    parser.add_argument(
        "--skip-normalization",
        action="store_true",
        help="Skip full-audio loudness normalization and only standardize format.",
    )
    return parser


def collect_audio_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    return sorted(path for path in input_path.rglob("*") if path.suffix.lower() in AUDIO_EXTENSIONS)


def process_file(input_path: Path, config: PipelineConfig) -> tuple[Path, Path]:
    print(f"Preparing audio: {input_path}", flush=True)
    prepared = prepare_audio(input_path, config)
    diarization_engine = PyannoteDiarizationEngine(config)
    transcription_engine = create_transcription_engine(config)

    print("Running diarization...", flush=True)
    diarization_turns = diarization_engine.diarize(prepared.normalized_path)
    print("Running transcription...", flush=True)
    transcription_segments = transcription_engine.transcribe_chunks(
        prepared.chunks,
        prepared.chunk_dir / "transcription",
    )
    print("Merging diarization and transcript segments...", flush=True)
    deduplicated_segments = deduplicate_overlap_segments(
        transcription_segments,
        overlap_seconds=config.chunk_overlap_seconds,
    )
    speaker_segments = assign_speakers_to_segments(deduplicated_segments, diarization_turns)
    merged_segments = merge_adjacent_segments(
        sorted(speaker_segments, key=lambda item: (item.start_s, item.end_s)),
        max_gap_s=config.merge_gap_seconds,
    )

    output_dir = (config.output_dir / input_path.stem).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = output_dir / "transcript.txt"
    json_path = output_dir / "result.json"

    transcript_path.write_text(render_transcript(merged_segments), encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "input_file": str(input_path.resolve()),
                "prepared_audio": prepared.to_dict(),
                "diarization_turns": [turn.to_dict() for turn in diarization_turns],
                "segments": [segment.to_dict() for segment in merged_segments],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("Writing output files...", flush=True)
    return transcript_path, json_path


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = PipelineConfig.from_env()

    overrides: dict[str, object] = {}
    if args.working_dir:
        overrides["working_dir"] = args.working_dir
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.backend:
        overrides["transcription_backend"] = args.backend
    if args.whisper_device:
        overrides["whisper_device"] = args.whisper_device
    if args.diarization_device:
        overrides["diarization_device"] = args.diarization_device
    if args.chunk_seconds:
        overrides["target_chunk_seconds"] = args.chunk_seconds
    if args.chunk_overlap:
        overrides["chunk_overlap_seconds"] = args.chunk_overlap
    if args.skip_normalization:
        overrides["enable_global_normalization"] = False

    config = config.with_overrides(**overrides) if overrides else config
    files = collect_audio_files(args.input)
    if not files:
        raise FileNotFoundError(f"No audio files found at {args.input}")

    for file_path in files:
        transcript_path, json_path = process_file(file_path, config)
        print(f"Processed {file_path}")
        print(f"Transcript: {transcript_path}")
        print(f"Metadata:   {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
