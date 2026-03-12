from src.audio.chunking import plan_chunks, speech_regions_from_silences
from src.models import SilenceInterval


def test_speech_regions_from_silences_builds_expected_ranges() -> None:
    silences = [
        SilenceInterval(start_s=5.0, end_s=6.0),
        SilenceInterval(start_s=10.0, end_s=12.0),
    ]

    assert speech_regions_from_silences(15.0, silences) == [
        (0.0, 5.0),
        (6.0, 10.0),
        (12.0, 15.0),
    ]


def test_plan_chunks_prefers_silence_boundaries_and_overlap() -> None:
    silences = [
        SilenceInterval(start_s=595.0, end_s=605.0),
        SilenceInterval(start_s=1188.0, end_s=1200.0),
    ]

    chunks = plan_chunks(
        total_duration_s=1500.0,
        silences=silences,
        target_chunk_s=600,
        overlap_s=8,
        min_chunk_s=45,
        max_chunk_s=720,
        preferred_split_window_s=30,
    )

    assert [(chunk.start_s, chunk.end_s) for chunk in chunks] == [
        (0.0, 600.0),
        (592.0, 1194.0),
        (1186.0, 1500.0),
    ]
    assert [chunk.leading_overlap_s for chunk in chunks] == [0.0, 8.0, 8.0]
