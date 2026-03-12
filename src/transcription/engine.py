from __future__ import annotations

import json
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from src.config import PipelineConfig
from src.models import AudioChunk, TranscriptSegment


class TranscriptionEngine(ABC):
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    @abstractmethod
    def transcribe_chunks(self, chunks: list[AudioChunk], work_dir: Path) -> list[TranscriptSegment]:
        raise NotImplementedError


class FasterWhisperTranscriptionEngine(TranscriptionEngine):
    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config)
        from faster_whisper import WhisperModel

        self.model = WhisperModel(
            model_size_or_path=config.whisper_model,
            device=config.whisper_device,
            compute_type=config.whisper_compute_type,
            cpu_threads=config.whisper_cpu_threads,
        )

    def transcribe_chunks(self, chunks: list[AudioChunk], work_dir: Path) -> list[TranscriptSegment]:
        all_segments: list[TranscriptSegment] = []
        for chunk in chunks:
            if chunk.path is None:
                raise RuntimeError(f"Chunk {chunk.index} has no file path.")

            segments, info = self.model.transcribe(
                str(chunk.path),
                task="transcribe",
                language=self.config.whisper_language,
                beam_size=self.config.whisper_beam_size,
                best_of=self.config.whisper_best_of,
                patience=self.config.whisper_patience,
                temperature=self.config.whisper_temperature,
                initial_prompt=self.config.whisper_initial_prompt,
                condition_on_previous_text=self.config.whisper_condition_on_previous_text,
                vad_filter=False,
                word_timestamps=False,
            )
            for segment in segments:
                text = segment.text.strip()
                if not text:
                    continue
                all_segments.append(
                    TranscriptSegment(
                        text=text,
                        start_s=round(chunk.start_s + float(segment.start), 3),
                        end_s=round(chunk.start_s + float(segment.end), 3),
                        chunk_index=chunk.index,
                        chunk_start_s=chunk.start_s,
                        chunk_end_s=chunk.end_s,
                        leading_overlap_s=chunk.leading_overlap_s,
                        avg_logprob=getattr(segment, "avg_logprob", None),
                        no_speech_prob=getattr(segment, "no_speech_prob", None),
                        language=getattr(info, "language", None),
                    )
                )
        return all_segments


class WhisperCppTranscriptionEngine(TranscriptionEngine):
    def transcribe_chunks(self, chunks: list[AudioChunk], work_dir: Path) -> list[TranscriptSegment]:
        if not self.config.whisper_cpp_path:
            raise RuntimeError("WHISPER_CPP_PATH must be set when using the whisper.cpp backend.")
        if not self.config.whisper_cpp_model_path:
            raise RuntimeError("WHISPER_CPP_MODEL_PATH must be set when using the whisper.cpp backend.")

        work_dir.mkdir(parents=True, exist_ok=True)
        all_segments: list[TranscriptSegment] = []
        for chunk in chunks:
            if chunk.path is None:
                raise RuntimeError(f"Chunk {chunk.index} has no file path.")

            output_prefix = work_dir / f"chunk_{chunk.index:04d}"
            command = [
                self.config.whisper_cpp_path,
                "-m",
                self.config.whisper_cpp_model_path,
                "-f",
                str(chunk.path),
                "-l",
                self.config.whisper_language,
                "-ojf",
                "-of",
                str(output_prefix),
            ]
            if self.config.whisper_initial_prompt:
                command.extend(["--prompt", self.config.whisper_initial_prompt])
            subprocess.run(command, capture_output=True, text=True, check=True)

            json_path = output_prefix.with_suffix(".json")
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            all_segments.extend(_parse_whisper_cpp_segments(payload, chunk))

        return all_segments


def _parse_whisper_cpp_segments(payload: dict[str, Any], chunk: AudioChunk) -> list[TranscriptSegment]:
    container = payload.get("result", payload)
    raw_segments = container.get("segments", payload.get("segments", []))
    language = container.get("language", payload.get("language"))

    parsed: list[TranscriptSegment] = []
    for segment in raw_segments:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue

        offsets = segment.get("offsets", {})
        if "from_ms" in offsets and "to_ms" in offsets:
            start_local_s = float(offsets["from_ms"]) / 1000.0
            end_local_s = float(offsets["to_ms"]) / 1000.0
        else:
            # whisper.cpp stores `t0`/`t1` in 10 ms units.
            start_local_s = float(segment.get("t0", 0.0)) / 100.0
            end_local_s = float(segment.get("t1", 0.0)) / 100.0

        parsed.append(
            TranscriptSegment(
                text=text,
                start_s=round(chunk.start_s + start_local_s, 3),
                end_s=round(chunk.start_s + end_local_s, 3),
                chunk_index=chunk.index,
                chunk_start_s=chunk.start_s,
                chunk_end_s=chunk.end_s,
                leading_overlap_s=chunk.leading_overlap_s,
                language=language,
            )
        )
    return parsed


def create_transcription_engine(config: PipelineConfig) -> TranscriptionEngine:
    backend = config.transcription_backend.strip().lower()
    if backend == "faster-whisper":
        return FasterWhisperTranscriptionEngine(config)
    if backend == "whisper.cpp":
        return WhisperCppTranscriptionEngine(config)
    raise ValueError(f"Unsupported transcription backend: {config.transcription_backend}")
