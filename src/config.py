from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from dotenv import load_dotenv


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


@dataclass(slots=True)
class PipelineConfig:
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    sample_rate: int = 16000
    channels: int = 1
    audio_codec: str = "pcm_s16le"
    working_dir: Path = Path("work")
    output_dir: Path = Path("outputs")
    enable_global_normalization: bool = True
    loudness_target_lufs: float = -16.0
    loudness_lra: float = 11.0
    loudness_true_peak: float = -1.5
    silence_threshold_db: float = -35.0
    silence_min_duration_s: float = 0.4
    target_chunk_seconds: int = 600
    chunk_overlap_seconds: int = 8
    min_chunk_seconds: int = 45
    max_chunk_seconds: int = 720
    preferred_split_window_seconds: int = 30
    enable_diarization: bool = True
    expected_speakers: int = 2
    diarization_model: str = "pyannote/speaker-diarization-3.1"
    diarization_device: str = "cpu"
    pyannote_auth_token: str | None = None
    transcription_backend: str = "faster-whisper"
    whisper_model: str = "large-v3"
    whisper_language: str = "nl"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    whisper_cpu_threads: int = max(1, os.cpu_count() or 1)
    whisper_beam_size: int = 8
    whisper_best_of: int = 8
    whisper_patience: float = 1.0
    whisper_temperature: float = 0.0
    whisper_condition_on_previous_text: bool = False
    whisper_initial_prompt: str = (
        "Dit is een Nederlands interview met twee sprekers. "
        "Behoud de transcriptie in het Nederlands, ook als er Engelstalige business termen voorkomen."
    )
    whisper_cpp_path: str | None = None
    whisper_cpp_model_path: str | None = None
    merge_gap_seconds: float = 0.35
    include_timestamps: bool = True
    include_speaker_labels: bool = True
    transcript_line_width: int = 100

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        load_dotenv(override=False)
        return cls(
            ffmpeg_bin=os.getenv("FFMPEG_BIN", "ffmpeg"),
            ffprobe_bin=os.getenv("FFPROBE_BIN", "ffprobe"),
            sample_rate=_get_int("AUDIO_SAMPLE_RATE", 16000),
            channels=_get_int("AUDIO_CHANNELS", 1),
            audio_codec=os.getenv("AUDIO_CODEC", "pcm_s16le"),
            working_dir=Path(os.getenv("WORKING_DIR", "work")),
            output_dir=Path(os.getenv("OUTPUT_DIR", "outputs")),
            enable_global_normalization=_get_bool("ENABLE_GLOBAL_NORMALIZATION", True),
            loudness_target_lufs=_get_float("LOUDNESS_TARGET_LUFS", -16.0),
            loudness_lra=_get_float("LOUDNESS_LRA", 11.0),
            loudness_true_peak=_get_float("LOUDNESS_TRUE_PEAK", -1.5),
            silence_threshold_db=_get_float("SILENCE_THRESHOLD_DB", -35.0),
            silence_min_duration_s=_get_float("SILENCE_MIN_DURATION_S", 0.4),
            target_chunk_seconds=_get_int("TARGET_CHUNK_SECONDS", 600),
            chunk_overlap_seconds=_get_int("CHUNK_OVERLAP_SECONDS", 8),
            min_chunk_seconds=_get_int("MIN_CHUNK_SECONDS", 45),
            max_chunk_seconds=_get_int("MAX_CHUNK_SECONDS", 720),
            preferred_split_window_seconds=_get_int("PREFERRED_SPLIT_WINDOW_SECONDS", 30),
            enable_diarization=_get_bool("ENABLE_DIARIZATION", True),
            expected_speakers=_get_int("EXPECTED_SPEAKERS", 2),
            diarization_model=os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1"),
            diarization_device=os.getenv("DIARIZATION_DEVICE", "cpu"),
            pyannote_auth_token=os.getenv("PYANNOTE_AUTH_TOKEN"),
            transcription_backend=os.getenv("TRANSCRIPTION_BACKEND", "faster-whisper"),
            whisper_model=os.getenv("WHISPER_MODEL", "large-v3"),
            whisper_language=os.getenv("WHISPER_LANGUAGE", "nl"),
            whisper_device=os.getenv("WHISPER_DEVICE", "cpu"),
            whisper_compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
            whisper_cpu_threads=_get_int("WHISPER_CPU_THREADS", max(1, os.cpu_count() or 1)),
            whisper_beam_size=_get_int("WHISPER_BEAM_SIZE", 8),
            whisper_best_of=_get_int("WHISPER_BEST_OF", 8),
            whisper_patience=_get_float("WHISPER_PATIENCE", 1.0),
            whisper_temperature=_get_float("WHISPER_TEMPERATURE", 0.0),
            whisper_condition_on_previous_text=_get_bool("WHISPER_CONDITION_ON_PREVIOUS_TEXT", False),
            whisper_initial_prompt=os.getenv(
                "WHISPER_INITIAL_PROMPT",
                "Dit is een Nederlands interview met twee sprekers. "
                "Behoud de transcriptie in het Nederlands, ook als er Engelstalige business termen voorkomen.",
            ),
            whisper_cpp_path=os.getenv("WHISPER_CPP_PATH"),
            whisper_cpp_model_path=os.getenv("WHISPER_CPP_MODEL_PATH"),
            merge_gap_seconds=_get_float("MERGE_GAP_SECONDS", 0.35),
            include_timestamps=_get_bool("INCLUDE_TIMESTAMPS", True),
            include_speaker_labels=_get_bool("INCLUDE_SPEAKER_LABELS", True),
            transcript_line_width=_get_int("TRANSCRIPT_LINE_WIDTH", 100),
        )

    def with_overrides(self, **kwargs: object) -> "PipelineConfig":
        return replace(self, **kwargs)
