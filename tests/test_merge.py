from src.merge.align import (
    assign_speakers_to_segments,
    deduplicate_overlap_segments,
    merge_adjacent_segments,
    render_transcript,
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


def _labelled_segments() -> list[TranscriptSegment]:
    return [
        TranscriptSegment(text="Goedemiddag, wat is jouw rol?", start_s=0.0, end_s=4.0, speaker="Spreker 1"),
        TranscriptSegment(text="Ik werk bij financien.", start_s=5.0, end_s=9.0, speaker="Spreker 2"),
    ]


def test_render_transcript_without_prefixes_produces_wrapped_paragraphs() -> None:
    rendered = render_transcript(
        _labelled_segments(),
        include_timestamps=False,
        include_speaker_labels=False,
        line_width=20,
    )

    assert "[" not in rendered
    assert "Spreker" not in rendered
    assert rendered.endswith("\n")
    paragraphs = rendered.strip().split("\n\n")
    assert len(paragraphs) == 2
    assert all(len(line) <= 20 for line in rendered.splitlines())


def test_render_transcript_prefixes_are_independently_toggleable() -> None:
    segments = _labelled_segments()

    timestamps_only = render_transcript(segments, include_speaker_labels=False)
    assert timestamps_only.startswith("[00:00:00] Goedemiddag")
    assert "Spreker" not in timestamps_only

    labels_only = render_transcript(segments, include_timestamps=False)
    assert labels_only.startswith("Spreker 1: Goedemiddag")
    assert "[" not in labels_only

    both = render_transcript(segments)
    assert both.startswith("[00:00:00] Spreker 1: Goedemiddag")
    assert "[00:00:05] Spreker 2: Ik werk bij financien." in both


def test_render_transcript_omits_labels_when_diarization_was_skipped() -> None:
    segments = [
        TranscriptSegment(text="Eerste zin.", start_s=0.0, end_s=4.0),
        TranscriptSegment(text="Tweede zin.", start_s=4.5, end_s=8.0),
    ]

    rendered = render_transcript(segments, include_timestamps=False)

    assert "Onbekend" not in rendered
    assert rendered == "Eerste zin. Tweede zin.\n"


def test_render_transcript_splits_paragraph_on_long_pause() -> None:
    segments = [
        TranscriptSegment(text="Eerste zin.", start_s=0.0, end_s=4.0),
        TranscriptSegment(text="Na een korte pauze.", start_s=5.0, end_s=8.0),
        TranscriptSegment(text="Na een lange pauze.", start_s=30.0, end_s=34.0),
    ]

    rendered = render_transcript(segments, include_timestamps=False, paragraph_gap_s=2.0)

    assert rendered.strip().split("\n\n") == [
        "Eerste zin. Na een korte pauze.",
        "Na een lange pauze.",
    ]


def test_render_transcript_splits_an_over_long_block_at_sentence_boundaries() -> None:
    long_text = " ".join(f"Dit is zin nummer {index}." for index in range(120))
    segments = [TranscriptSegment(text=long_text, start_s=0.0, end_s=600.0, speaker="Spreker 1")]

    rendered = render_transcript(segments, include_timestamps=False)
    paragraphs = rendered.strip().split("\n\n")

    assert len(paragraphs) > 1
    # Only the first paragraph repeats the label; the rest align underneath it.
    assert paragraphs[0].startswith("Spreker 1: ")
    assert all(paragraph.startswith(" " * len("Spreker 1: ")) for paragraph in paragraphs[1:])
    assert all(paragraph.rstrip().endswith(".") for paragraph in paragraphs)


def test_render_transcript_wrapping_uses_hanging_indent_under_the_prefix() -> None:
    segments = [
        TranscriptSegment(
            text="Dit is een tamelijk lange zin die zeker over meerdere regels moet worden afgebroken.",
            start_s=0.0,
            end_s=10.0,
            speaker="Spreker 1",
        )
    ]

    lines = render_transcript(segments, line_width=60).splitlines()

    assert len(lines) > 1
    prefix_width = len("[00:00:00] Spreker 1: ")
    assert all(line.startswith(" " * prefix_width) for line in lines[1:])
    assert all(len(line) <= 60 for line in lines)
