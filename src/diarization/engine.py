from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any
import warnings

from src.config import PipelineConfig
from src.models import DiarizationTurn


class PyannoteDiarizationEngine:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def diarize(self, audio_path: Path) -> list[DiarizationTurn]:
        if not self.config.pyannote_auth_token:
            raise RuntimeError(
                "PYANNOTE_AUTH_TOKEN is required for diarization. "
                "Request access to the pyannote model on Hugging Face and set the token in your environment."
            )

        # We preload WAV audio ourselves, so pyannote's torchcodec warning is only noise here.
        warnings.filterwarnings(
            "ignore",
            message=r".*torchcodec is not installed correctly.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*degrees of freedom is <= 0.*",
            category=UserWarning,
        )

        import numpy as np
        import torch
        from pyannote.audio import Pipeline
        from scipy.io import wavfile

        pipeline = Pipeline.from_pretrained(
            self.config.diarization_model,
            token=self.config.pyannote_auth_token,
        )
        device = torch.device(self.config.diarization_device)
        pipeline.to(device)
        sample_rate, waveform_np = wavfile.read(str(audio_path))
        if waveform_np.ndim == 1:
            waveform_np = np.expand_dims(waveform_np, axis=0)
        else:
            waveform_np = waveform_np.T
        waveform = torch.from_numpy(waveform_np.astype(np.float32))
        if waveform.numel() and waveform.abs().max() > 1.0:
            waveform = waveform / 32768.0

        diarization = pipeline(
            {"waveform": waveform, "sample_rate": sample_rate},
            num_speakers=self.config.expected_speakers,
            min_speakers=self.config.expected_speakers,
            max_speakers=self.config.expected_speakers,
        )

        speaker_map: OrderedDict[str, str] = OrderedDict()
        turns: list[DiarizationTurn] = []
        annotation = _extract_annotation(diarization)
        for time_range, _, raw_label in annotation.itertracks(yield_label=True):
            if raw_label not in speaker_map:
                speaker_map[raw_label] = f"Spreker {len(speaker_map) + 1}"
            turns.append(
                DiarizationTurn(
                    speaker=speaker_map[raw_label],
                    start_s=round(float(time_range.start), 3),
                    end_s=round(float(time_range.end), 3),
                )
            )
        return _merge_adjacent_turns(turns)


def _merge_adjacent_turns(turns: list[DiarizationTurn], max_gap_s: float = 0.15) -> list[DiarizationTurn]:
    if not turns:
        return []

    merged: list[DiarizationTurn] = [turns[0]]
    for turn in turns[1:]:
        previous = merged[-1]
        if turn.speaker == previous.speaker and turn.start_s - previous.end_s <= max_gap_s:
            previous.end_s = max(previous.end_s, turn.end_s)
        else:
            merged.append(turn)
    return merged


def _extract_annotation(diarization: Any) -> Any:
    if hasattr(diarization, "itertracks"):
        return diarization

    for attribute_name in ("exclusive_speaker_diarization", "speaker_diarization"):
        annotation = getattr(diarization, attribute_name, None)
        if annotation is not None and hasattr(annotation, "itertracks"):
            return annotation

    raise TypeError(
        "Unsupported pyannote diarization output type. "
        f"Expected an Annotation-like object, got {type(diarization)!r}."
    )
