from src.merge.align import (
    assign_speakers_to_segments,
    deduplicate_overlap_segments,
    merge_adjacent_segments,
)
from src.models import DiarizationTurn, TranscriptSegment


def test_deduplicate_overlap_segments_drops_leading_overlap_from_next_chunk() -> None:
    segments = [
        TranscriptSegment(
            text="Welkom allemaal",
            start_s=0.0,
            end_s=7.5,
            chunk_index=0,
            chunk_start_s=0.0,
            chunk_end_s=8.0,
            leading_overlap_s=0.0,
        ),
        TranscriptSegment(
            text="Welkom allemaal",
            start_s=7.0,
            end_s=11.0,
            chunk_index=1,
            chunk_start_s=6.0,
            chunk_end_s=14.0,
            leading_overlap_s=2.0,
        ),
        TranscriptSegment(
            text="We gaan beginnen",
            start_s=11.0,
            end_s=14.0,
            chunk_index=1,
            chunk_start_s=6.0,
            chunk_end_s=14.0,
            leading_overlap_s=2.0,
        ),
    ]

    cleaned = deduplicate_overlap_segments(segments, overlap_seconds=2.0)

    assert [segment.text for segment in cleaned] == [
        "Welkom allemaal",
        "We gaan beginnen",
    ]


def test_assign_and_merge_segments_keeps_speaker_labels_consistent() -> None:
    turns = [
        DiarizationTurn(speaker="Spreker 1", start_s=0.0, end_s=8.0),
        DiarizationTurn(speaker="Spreker 2", start_s=8.0, end_s=20.0),
    ]
    segments = [
        TranscriptSegment(text="Goedemiddag", start_s=0.0, end_s=2.0),
        TranscriptSegment(text="hoe gaat het", start_s=2.1, end_s=4.0),
        TranscriptSegment(text="prima dank je", start_s=9.0, end_s=11.0),
    ]

    speaker_segments = assign_speakers_to_segments(segments, turns)
    merged = merge_adjacent_segments(speaker_segments, max_gap_s=0.25)

    assert [segment.speaker for segment in merged] == ["Spreker 1", "Spreker 2"]
    assert merged[0].text == "Goedemiddag hoe gaat het"
    assert merged[1].text == "prima dank je"
