from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from src.audio.chunking import plan_chunks
from src.config import PipelineConfig
from src.models import PreparedAudio, SilenceInterval

SILENCE_START_RE = re.compile(r"silence_start:\s*(?P<value>-?\d+(?:\.\d+)?)")
SILENCE_END_RE = re.compile(r"silence_end:\s*(?P<value>-?\d+(?:\.\d+)?)")


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=True)


def _null_sink() -> str:
    return "NUL" if os.name == "nt" else "/dev/null"


def probe_audio(input_path: Path, config: PipelineConfig) -> dict[str, Any]:
    command = [
        config.ffprobe_bin,
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-of",
        "json",
        str(input_path),
    ]
    completed = _run_command(command)
    return json.loads(completed.stdout)


def get_duration_seconds(metadata: dict[str, Any]) -> float:
    format_info = metadata.get("format", {})
    duration = format_info.get("duration")
    if duration is not None:
        return float(duration)

    for stream in metadata.get("streams", []):
        stream_duration = stream.get("duration")
        if stream_duration is not None:
            return float(stream_duration)
    raise ValueError("Could not determine audio duration from ffprobe metadata.")


def _extract_loudnorm_stats(stderr_output: str) -> dict[str, Any]:
    lines = [line.strip() for line in stderr_output.splitlines() if line.strip()]
    json_lines: list[str] = []
    capturing = False
    brace_depth = 0
    for line in lines:
        if line.startswith("{") and not capturing:
            capturing = True
        if capturing:
            json_lines.append(line)
            brace_depth += line.count("{")
            brace_depth -= line.count("}")
            if brace_depth == 0:
                candidate = "\n".join(json_lines)
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    json_lines = []
                    capturing = False
    raise RuntimeError("ffmpeg loudnorm analysis did not return JSON stats.")


def analyze_loudness(input_path: Path, config: PipelineConfig) -> dict[str, Any]:
    filter_value = (
        f"loudnorm=I={config.loudness_target_lufs}:"
        f"LRA={config.loudness_lra}:"
        f"TP={config.loudness_true_peak}:"
        "print_format=json"
    )
    command = [
        config.ffmpeg_bin,
        "-hide_banner",
        "-i",
        str(input_path),
        "-af",
        filter_value,
        "-f",
        "null",
        _null_sink(),
    ]
    completed = _run_command(command)
    return _extract_loudnorm_stats(completed.stderr)


def standardize_audio(input_path: Path, output_path: Path, config: PipelineConfig) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        config.ffmpeg_bin,
        "-hide_banner",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ar",
        str(config.sample_rate),
        "-ac",
        str(config.channels),
        "-c:a",
        config.audio_codec,
        str(output_path),
    ]
    _run_command(command)


def normalize_audio(input_path: Path, output_path: Path, config: PipelineConfig) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = analyze_loudness(input_path, config)
    filter_value = (
        f"loudnorm=I={config.loudness_target_lufs}:"
        f"LRA={config.loudness_lra}:"
        f"TP={config.loudness_true_peak}:"
        f"measured_I={stats['input_i']}:"
        f"measured_LRA={stats['input_lra']}:"
        f"measured_TP={stats['input_tp']}:"
        f"measured_thresh={stats['input_thresh']}:"
        f"offset={stats['target_offset']}:"
        "linear=true:print_format=summary"
    )
    command = [
        config.ffmpeg_bin,
        "-hide_banner",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ar",
        str(config.sample_rate),
        "-ac",
        str(config.channels),
        "-af",
        filter_value,
        "-c:a",
        config.audio_codec,
        str(output_path),
    ]
    _run_command(command)


def detect_silences(input_path: Path, config: PipelineConfig) -> list[SilenceInterval]:
    command = [
        config.ffmpeg_bin,
        "-hide_banner",
        "-i",
        str(input_path),
        "-af",
        f"silencedetect=n={config.silence_threshold_db}dB:d={config.silence_min_duration_s}",
        "-f",
        "null",
        _null_sink(),
    ]
    completed = _run_command(command)

    silences: list[SilenceInterval] = []
    current_start: float | None = None
    for line in completed.stderr.splitlines():
        start_match = SILENCE_START_RE.search(line)
        if start_match:
            current_start = float(start_match.group("value"))
            continue

        end_match = SILENCE_END_RE.search(line)
        if end_match and current_start is not None:
            silences.append(SilenceInterval(start_s=current_start, end_s=float(end_match.group("value"))))
            current_start = None
    return silences


def export_chunks(
    normalized_path: Path,
    chunks: list,
    chunk_dir: Path,
    config: PipelineConfig,
) -> None:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    for chunk in chunks:
        chunk_path = chunk_dir / f"chunk_{chunk.index:04d}.wav"
        command = [
            config.ffmpeg_bin,
            "-hide_banner",
            "-y",
            "-ss",
            f"{chunk.start_s:.3f}",
            "-i",
            str(normalized_path),
            "-t",
            f"{chunk.duration_s:.3f}",
            "-ar",
            str(config.sample_rate),
            "-ac",
            str(config.channels),
            "-c:a",
            config.audio_codec,
            str(chunk_path),
        ]
        _run_command(command)
        chunk.path = chunk_path
        chunk.source_path = normalized_path


def write_chunk_manifest(
    manifest_path: Path,
    chunks: list,
    source_metadata: dict[str, Any],
    normalized_metadata: dict[str, Any],
    silence_intervals: list[SilenceInterval],
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "chunks": [chunk.to_dict() for chunk in chunks],
        "source_metadata": source_metadata,
        "normalized_metadata": normalized_metadata,
        "silence_intervals": [interval.to_dict() for interval in silence_intervals],
    }
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def prepare_audio(input_path: Path, config: PipelineConfig) -> PreparedAudio:
    source_path = input_path.resolve()
    job_dir = config.working_dir / source_path.stem
    audio_dir = job_dir / "audio"
    chunk_dir = job_dir / "chunks"
    normalized_path = audio_dir / "normalized.wav"
    manifest_path = job_dir / "chunk_manifest.json"

    source_metadata = probe_audio(source_path, config)
    if config.enable_global_normalization:
        normalize_audio(source_path, normalized_path, config)
    else:
        standardize_audio(source_path, normalized_path, config)

    normalized_metadata = probe_audio(normalized_path, config)
    total_duration_s = get_duration_seconds(normalized_metadata)
    silence_intervals = detect_silences(normalized_path, config)
    chunks = plan_chunks(
        total_duration_s=total_duration_s,
        silences=silence_intervals,
        target_chunk_s=config.target_chunk_seconds,
        overlap_s=config.chunk_overlap_seconds,
        min_chunk_s=config.min_chunk_seconds,
        max_chunk_s=config.max_chunk_seconds,
        preferred_split_window_s=config.preferred_split_window_seconds,
        source_path=normalized_path,
    )
    export_chunks(normalized_path, chunks, chunk_dir, config)
    write_chunk_manifest(
        manifest_path=manifest_path,
        chunks=chunks,
        source_metadata=source_metadata,
        normalized_metadata=normalized_metadata,
        silence_intervals=silence_intervals,
    )
    return PreparedAudio(
        source_path=source_path,
        normalized_path=normalized_path,
        chunk_dir=chunk_dir,
        manifest_path=manifest_path,
        chunks=chunks,
        silence_intervals=silence_intervals,
        source_metadata=source_metadata,
        normalized_metadata=normalized_metadata,
    )
