from __future__ import annotations

import json
import re
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from src.config import PipelineConfig
from src.models import AudioChunk, TranscriptSegment, TranscriptWord

# whisper.cpp control tokens such as [_BEG_] and [_TT_150]; they carry no spoken text.
WHISPER_CPP_SPECIAL_TOKEN_RE = re.compile(r"^\s*(\[_[^\]]*\]|<\|[^|]*\|>)\s*$")


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
                word_timestamps=self.config.whisper_word_timestamps,
            )
            for segment in segments:
                text = segment.text.strip()
                if not text:
                    continue
                words = [
                    TranscriptWord(
                        text=word.word.strip(),
                        start_s=round(chunk.start_s + float(word.start), 3),
                        end_s=round(chunk.start_s + float(word.end), 3),
                        probability=getattr(word, "probability", None),
                    )
                    for word in (getattr(segment, "words", None) or [])
                    if word.word.strip()
                ]
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
                        words=words,
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
                "-bs",
                str(self.config.whisper_beam_size),
                "-bo",
                str(self.config.whisper_best_of),
                # Full JSON includes per-token timestamps, which become word timestamps.
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
    # whisper.cpp writes {"result": {"language": ...}, "transcription": [...]}.
    raw_segments = payload.get("transcription", payload.get("segments", []))
    result = payload.get("result", {})
    language = result.get("language") if isinstance(result, dict) else None
    language = language or payload.get("language")

    parsed: list[TranscriptSegment] = []
    for segment in raw_segments:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue

        span = _whisper_cpp_span(segment)
        start_local_s, end_local_s = span if span else (0.0, 0.0)
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
                words=_whisper_cpp_words(segment.get("tokens", []), chunk.start_s),
            )
        )
    return parsed


def _whisper_cpp_span(item: dict[str, Any]) -> tuple[float, float] | None:
    offsets = item.get("offsets", {})
    if "from" in offsets and "to" in offsets:
        # whisper.cpp writes offsets in milliseconds.
        return float(offsets["from"]) / 1000.0, float(offsets["to"]) / 1000.0
    if "from_ms" in offsets and "to_ms" in offsets:
        return float(offsets["from_ms"]) / 1000.0, float(offsets["to_ms"]) / 1000.0
    if "t0" in item and "t1" in item:
        # whisper.cpp stores `t0`/`t1` in 10 ms units.
        return float(item["t0"]) / 100.0, float(item["t1"]) / 100.0
    return None


def _whisper_cpp_words(tokens: list[dict[str, Any]], offset_s: float) -> list[TranscriptWord]:
    """Join whisper.cpp tokens into words: a token without a leading space continues the last word."""
    words: list[TranscriptWord] = []
    for token in tokens:
        text = str(token.get("text", ""))
        span = _whisper_cpp_span(token)
        if not text.strip() or span is None or WHISPER_CPP_SPECIAL_TOKEN_RE.match(text):
            continue
        start_s = round(offset_s + span[0], 3)
        end_s = round(offset_s + span[1], 3)
        probability = token.get("p")
        if words and not text.startswith(" "):
            previous = words[-1]
            previous.text += text.strip()
            previous.end_s = max(previous.end_s, end_s)
            if probability is not None and previous.probability is not None:
                previous.probability = min(previous.probability, probability)
            continue
        words.append(TranscriptWord(text=text.strip(), start_s=start_s, end_s=end_s, probability=probability))
    return words


def create_transcription_engine(config: PipelineConfig) -> TranscriptionEngine:
    backend = config.transcription_backend.strip().lower()
    if backend == "faster-whisper":
        return FasterWhisperTranscriptionEngine(config)
    if backend == "whisper.cpp":
        return WhisperCppTranscriptionEngine(config)
    raise ValueError(f"Unsupported transcription backend: {config.transcription_backend}")
