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
        if pipeline is None:
            # pyannote returns None instead of raising when the Hugging Face download is refused.
            raise RuntimeError(
                f"Could not load diarization model {self.config.diarization_model!r}. "
                "Accept the model's user conditions on Hugging Face with the account that owns "
                "PYANNOTE_AUTH_TOKEN, or set DIARIZATION_MODEL to a model you have access to."
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
            **speaker_count_kwargs(self.config),
        )

        annotation = _extract_annotation(diarization)
        raw_turns = [
            DiarizationTurn(
                speaker=str(raw_label),
                start_s=round(float(time_range.start), 3),
                end_s=round(float(time_range.end), 3),
            )
            for time_range, _, raw_label in annotation.itertracks(yield_label=True)
        ]
        # Filter before labelling, so a speaker that only appears in flicker gets no number.
        turns = drop_short_turns(raw_turns, self.config.min_turn_seconds)
        return _merge_adjacent_turns(_label_in_order_of_appearance(turns))


def speaker_count_kwargs(config: PipelineConfig) -> dict[str, int]:
    """Translate the speaker settings into pyannote arguments; no arguments means fully automatic."""
    if config.num_speakers is not None:
        return {"num_speakers": config.num_speakers}
    bounds = {"min_speakers": config.min_speakers, "max_speakers": config.max_speakers}
    return {name: value for name, value in bounds.items() if value is not None}


def drop_short_turns(turns: list[DiarizationTurn], min_turn_s: float) -> list[DiarizationTurn]:
    """Drop turns shorter than `min_turn_s`.

    Where two people talk at once, pyannote's exclusive output can flip between them every few
    tens of milliseconds. Those slivers are not real turns; words that fall in the gaps they leave
    are assigned to the nearest remaining turn instead.
    """
    if min_turn_s <= 0:
        return list(turns)
    kept = [turn for turn in turns if turn.end_s - turn.start_s >= min_turn_s]
    # A recording made up of nothing but tiny turns is better served unfiltered than empty.
    return kept or list(turns)


def _label_in_order_of_appearance(turns: list[DiarizationTurn]) -> list[DiarizationTurn]:
    speaker_map: OrderedDict[str, str] = OrderedDict()
    labelled: list[DiarizationTurn] = []
    for turn in sorted(turns, key=lambda item: (item.start_s, item.end_s)):
        if turn.speaker not in speaker_map:
            speaker_map[turn.speaker] = f"Spreker {len(speaker_map) + 1}"
        labelled.append(DiarizationTurn(speaker=speaker_map[turn.speaker], start_s=turn.start_s, end_s=turn.end_s))
    return labelled


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
