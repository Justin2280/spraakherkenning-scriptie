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
from src.models import DiarizationTurn
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
    parser.add_argument(
        "--diarization",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run speaker diarization. Use --no-diarization to transcribe only.",
    )
    parser.add_argument(
        "--timestamps",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Prefix each paragraph in the transcript with its start time.",
    )
    parser.add_argument(
        "--speaker-labels",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Prefix each paragraph in the transcript with its speaker.",
    )
    parser.add_argument("--line-width", type=int, help="Wrap column for the transcript text.")
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
    transcription_engine = create_transcription_engine(config)

    diarization_turns: list[DiarizationTurn] = []
    if config.enable_diarization:
        print("Running diarization...", flush=True)
        diarization_turns = PyannoteDiarizationEngine(config).diarize(prepared.normalized_path)
    else:
        print("Skipping diarization (disabled).", flush=True)
    print("Running transcription...", flush=True)
    transcription_segments = transcription_engine.transcribe_chunks(
        prepared.chunks,
        prepared.chunk_dir / "transcription",
    )
    print("Merging transcript segments...", flush=True)
    deduplicated_segments = deduplicate_overlap_segments(
        transcription_segments,
        overlap_seconds=config.chunk_overlap_seconds,
    )
    ordered_segments = sorted(deduplicated_segments, key=lambda item: (item.start_s, item.end_s))
    if diarization_turns:
        speaker_segments = assign_speakers_to_segments(ordered_segments, diarization_turns)
        merged_segments = merge_adjacent_segments(
            speaker_segments,
            max_gap_s=config.merge_gap_seconds,
        )
    else:
        # Without speakers every segment would merge into one, so leave them granular and let
        # the renderer paragraph them by pause instead.
        merged_segments = ordered_segments

    output_dir = (config.output_dir / input_path.stem).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = output_dir / "transcript.txt"
    json_path = output_dir / "result.json"

    transcript_path.write_text(
        render_transcript(
            merged_segments,
            include_timestamps=config.include_timestamps,
            include_speaker_labels=config.include_speaker_labels,
            line_width=config.transcript_line_width,
        ),
        encoding="utf-8",
    )
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
    if args.diarization is not None:
        overrides["enable_diarization"] = args.diarization
    if args.timestamps is not None:
        overrides["include_timestamps"] = args.timestamps
    if args.speaker_labels is not None:
        overrides["include_speaker_labels"] = args.speaker_labels
    if args.line_width:
        overrides["transcript_line_width"] = args.line_width

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
