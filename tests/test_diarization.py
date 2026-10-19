from src.config import PipelineConfig
from src.diarization.engine import (
    _label_in_order_of_appearance,
    _merge_adjacent_turns,
    drop_short_turns,
    speaker_count_kwargs,
)
from src.models import DiarizationTurn


def test_speaker_count_kwargs_passes_a_known_count() -> None:
    assert speaker_count_kwargs(PipelineConfig(num_speakers=3)) == {"num_speakers": 3}


def test_speaker_count_kwargs_is_empty_for_fully_automatic_detection() -> None:
    assert speaker_count_kwargs(PipelineConfig()) == {}


def test_speaker_count_kwargs_passes_bounds_in_automatic_mode() -> None:
    config = PipelineConfig(min_speakers=2, max_speakers=4)

    assert speaker_count_kwargs(config) == {"min_speakers": 2, "max_speakers": 4}


def test_a_known_count_wins_over_bounds() -> None:
    config = PipelineConfig(num_speakers=2, min_speakers=3, max_speakers=5)

    assert speaker_count_kwargs(config) == {"num_speakers": 2}


def test_drop_short_turns_removes_flicker_but_keeps_short_replies() -> None:
    # Overlapping speech in exclusive output: slivers of 20-100 ms alternate between speakers.
    turns = [
        DiarizationTurn(speaker="A", start_s=0.0, end_s=2.35),
        DiarizationTurn(speaker="B", start_s=2.35, end_s=2.39),
        DiarizationTurn(speaker="A", start_s=2.39, end_s=2.49),
        DiarizationTurn(speaker="B", start_s=2.49, end_s=5.0),
        DiarizationTurn(speaker="A", start_s=5.0, end_s=5.3),
    ]

    kept = drop_short_turns(turns, min_turn_s=0.2)

    assert [(turn.speaker, turn.start_s) for turn in kept] == [("A", 0.0), ("B", 2.49), ("A", 5.0)]


def test_drop_short_turns_keeps_everything_rather_than_nothing() -> None:
    turns = [DiarizationTurn(speaker="A", start_s=0.0, end_s=0.1)]

    assert drop_short_turns(turns, min_turn_s=0.2) == turns


def test_a_speaker_seen_only_in_flicker_gets_no_number() -> None:
    turns = [
        DiarizationTurn(speaker="SPEAKER_01", start_s=0.0, end_s=0.05),
        DiarizationTurn(speaker="SPEAKER_00", start_s=0.05, end_s=4.0),
        DiarizationTurn(speaker="SPEAKER_01", start_s=4.0, end_s=8.0),
    ]

    labelled = _label_in_order_of_appearance(drop_short_turns(turns, min_turn_s=0.2))

    assert [turn.speaker for turn in labelled] == ["Spreker 1", "Spreker 2"]


def test_turns_split_by_removed_flicker_merge_again() -> None:
    turns = [
        DiarizationTurn(speaker="Spreker 1", start_s=0.0, end_s=2.0),
        DiarizationTurn(speaker="Spreker 1", start_s=2.05, end_s=4.0),
    ]

    assert [(turn.start_s, turn.end_s) for turn in _merge_adjacent_turns(turns)] == [(0.0, 4.0)]
