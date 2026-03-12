from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class SilenceInterval:
    start_s: float
    end_s: float

    def to_dict(self) -> dict[str, float]:
        return {"start_s": self.start_s, "end_s": self.end_s}


@dataclass(slots=True)
class AudioChunk:
    index: int
    start_s: float
    end_s: float
    path: Path | None
    source_path: Path | None
    leading_overlap_s: float = 0.0

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["path"] = str(self.path) if self.path else None
        data["source_path"] = str(self.source_path) if self.source_path else None
        data["duration_s"] = self.duration_s
        return data


@dataclass(slots=True)
class DiarizationTurn:
    speaker: str
    start_s: float
    end_s: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TranscriptSegment:
    text: str
    start_s: float
    end_s: float
    speaker: str | None = None
    chunk_index: int | None = None
    chunk_start_s: float | None = None
    chunk_end_s: float | None = None
    leading_overlap_s: float = 0.0
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    language: str | None = None

    @property
    def midpoint_s(self) -> float:
        return self.start_s + max(0.0, self.end_s - self.start_s) / 2.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PreparedAudio:
    source_path: Path
    normalized_path: Path
    chunk_dir: Path
    manifest_path: Path
    chunks: list[AudioChunk]
    silence_intervals: list[SilenceInterval]
    source_metadata: dict[str, Any]
    normalized_metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": str(self.source_path),
            "normalized_path": str(self.normalized_path),
            "chunk_dir": str(self.chunk_dir),
            "manifest_path": str(self.manifest_path),
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "silence_intervals": [interval.to_dict() for interval in self.silence_intervals],
            "source_metadata": self.source_metadata,
            "normalized_metadata": self.normalized_metadata,
        }
