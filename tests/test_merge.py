from src.merge.align import (
    assign_speakers_to_segments,
    deduplicate_overlap_segments,
    merge_adjacent_segments,
    render_transcript,
)
from src.models import DiarizationTurn, TranscriptSegment, TranscriptWord


def _words(*items: tuple[str, float, float]) -> list[TranscriptWord]:
    return [TranscriptWord(text=text, start_s=start, end_s=end) for text, start, end in items]


def _segment_from_words(words: list[TranscriptWord], **kwargs: object) -> TranscriptSegment:
    return TranscriptSegment(
        text=" ".join(word.text for word in words),
        start_s=words[0].start_s,
        end_s=words[-1].end_s,
        words=words,
        **kwargs,
    )


def _speakers_and_text(segments: list[TranscriptSegment]) -> list[tuple[str | None, str]]:
    return [(segment.speaker, segment.text) for segment in segments]


def test_deduplicate_overlap_segments_drops_leading_overlap_from_next_chunk() -> None:
    # Segments without word timestamps (whisper.cpp fallback) are still dropped whole.
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


def test_deduplicate_keeps_the_words_of_a_segment_that_runs_past_the_chunk_boundary() -> None:
    # Chunk 1 takes over at 607 s. Its first segment starts inside the overlap but continues for
    # ten seconds; dropping it whole lost that speech from the transcript entirely.
    chunk = {"chunk_index": 1, "chunk_start_s": 599.0, "chunk_end_s": 1200.0, "leading_overlap_s": 8.0}
    straddling = _segment_from_words(
        _words(("de", 605.0, 605.3), ("wet.", 605.3, 605.9), ("En", 608.0, 608.2), ("dan", 608.2, 608.5)),
        **chunk,
    )
    after = _segment_from_words(_words(("Verder.", 610.0, 610.6)), **chunk)

    cleaned = deduplicate_overlap_segments([straddling, after], overlap_seconds=8.0)

    assert [segment.text for segment in cleaned] == ["En dan", "Verder."]
    assert cleaned[0].start_s == 608.0


def test_assign_speakers_splits_a_segment_where_the_speaker_changes() -> None:
    # The question's last words and the answer ended up in one Whisper segment.
    turns = [
        DiarizationTurn(speaker="Spreker 1", start_s=440.0, end_s=448.9),
        DiarizationTurn(speaker="Spreker 2", start_s=449.6, end_s=470.0),
    ]
    segment = _segment_from_words(
        _words(
            ("leveren", 445.8, 446.3),
            ("jullie", 446.3, 446.7),
            ("voor", 446.7, 447.0),
            ("de", 447.0, 447.1),
            ("rekenkamers?", 447.1, 448.0),
            ("Ja", 449.7, 450.0),
            ("zeker.", 450.0, 450.5),
        )
    )

    assigned = assign_speakers_to_segments([segment], turns)

    assert _speakers_and_text(assigned) == [
        ("Spreker 1", "leveren jullie voor de rekenkamers?"),
        ("Spreker 2", "Ja zeker."),
    ]
    assert all(word.speaker == "Spreker 2" for word in assigned[1].words)


def test_a_short_reply_inside_a_long_turn_gets_its_own_segment() -> None:
    turns = [
        DiarizationTurn(speaker="Spreker 2", start_s=0.0, end_s=5.0),
        DiarizationTurn(speaker="Spreker 1", start_s=5.0, end_s=5.4),
        DiarizationTurn(speaker="Spreker 2", start_s=5.4, end_s=10.0),
    ]
    segment = _segment_from_words(
        _words(("dus", 4.0, 4.5), ("zo", 4.5, 4.9), ("ja,", 5.05, 5.35), ("precies.", 5.5, 6.0))
    )

    assigned = assign_speakers_to_segments([segment], turns)

    assert _speakers_and_text(assigned) == [
        ("Spreker 2", "dus zo"),
        ("Spreker 1", "ja,"),
        ("Spreker 2", "precies."),
    ]


def test_a_word_straddling_a_turn_boundary_stays_with_its_sentence() -> None:
    # "snel." sits half in a pause; it finishes speaker 1's sentence rather than opening speaker 2's.
    turns = [
        DiarizationTurn(speaker="Spreker 1", start_s=0.0, end_s=4.0),
        DiarizationTurn(speaker="Spreker 2", start_s=4.3, end_s=8.0),
    ]
    segment = _segment_from_words(
        _words(("niet", 3.0, 3.4), ("heel", 3.4, 3.8), ("snel.", 3.8, 4.5), ("De", 4.5, 4.7), ("waan", 4.7, 5.0))
    )

    assigned = assign_speakers_to_segments([segment], turns)

    assert _speakers_and_text(assigned) == [("Spreker 1", "niet heel snel."), ("Spreker 2", "De waan")]


def test_an_unclear_word_after_a_finished_sentence_joins_the_next_speaker() -> None:
    turns = [
        DiarizationTurn(speaker="Spreker 1", start_s=0.0, end_s=4.0),
        DiarizationTurn(speaker="Spreker 2", start_s=4.3, end_s=8.0),
    ]
    segment = _segment_from_words(_words(("klaar?", 3.0, 3.8), ("Nou", 3.9, 4.6), ("ja.", 4.6, 5.0)))

    assigned = assign_speakers_to_segments([segment], turns)

    assert _speakers_and_text(assigned) == [("Spreker 1", "klaar?"), ("Spreker 2", "Nou ja.")]


def test_words_follow_speakers_across_segment_boundaries() -> None:
    turns = [
        DiarizationTurn(speaker="Spreker 1", start_s=0.0, end_s=2.0),
        DiarizationTurn(speaker="Spreker 2", start_s=2.0, end_s=6.0),
    ]
    first = _segment_from_words(_words(("Hoe", 0.0, 0.5), ("gaat", 0.5, 1.0), ("het?", 1.0, 1.9)))
    second = _segment_from_words(_words(("Goed,", 2.1, 2.6), ("dank", 2.6, 3.0), ("je.", 3.0, 3.4)))

    merged = merge_adjacent_segments(assign_speakers_to_segments([first, second], turns), max_gap_s=0.35)

    assert _speakers_and_text(merged) == [("Spreker 1", "Hoe gaat het?"), ("Spreker 2", "Goed, dank je.")]


def test_segment_without_words_goes_to_the_speaker_with_most_overlap() -> None:
    # A half-second reply on the segment's midpoint used to win from 9.5 s of the other speaker.
    turns = [
        DiarizationTurn(speaker="Spreker 1", start_s=0.0, end_s=4.8),
        DiarizationTurn(speaker="Spreker 2", start_s=4.8, end_s=5.3),
        DiarizationTurn(speaker="Spreker 1", start_s=5.3, end_s=10.0),
    ]
    segment = TranscriptSegment(text="lange zin", start_s=0.0, end_s=10.0)

    assert assign_speakers_to_segments([segment], turns)[0].speaker == "Spreker 1"


def test_segment_in_a_pause_goes_to_the_nearest_turn() -> None:
    turns = [
        DiarizationTurn(speaker="Spreker 1", start_s=0.0, end_s=2.0),
        DiarizationTurn(speaker="Spreker 2", start_s=10.0, end_s=12.0),
    ]
    segment = TranscriptSegment(text="hm", start_s=8.5, end_s=9.0)

    assert assign_speakers_to_segments([segment], turns)[0].speaker == "Spreker 2"


def test_merge_adjacent_segments_keeps_the_words() -> None:
    first = _segment_from_words(_words(("Eerste", 0.0, 0.5)), speaker="Spreker 1")
    second = _segment_from_words(_words(("tweede.", 0.6, 1.0)), speaker="Spreker 1")

    merged = merge_adjacent_segments([first, second], max_gap_s=0.35)

    assert [word.text for word in merged[0].words] == ["Eerste", "tweede."]


def test_words_in_the_listeners_short_turn_stay_with_the_talker() -> None:
    # The listener's "hm" while the talker keeps going gets its own turn, but Whisper only wrote
    # down the talker's words; "en er" must not be handed to the listener.
    turns = [
        DiarizationTurn(speaker="Spreker 1", start_s=0.0, end_s=4.25),
        DiarizationTurn(speaker="Spreker 2", start_s=4.25, end_s=4.86),
        DiarizationTurn(speaker="Spreker 1", start_s=5.24, end_s=10.0),
    ]
    segment = _segment_from_words(
        _words(
            ("er", 1.6, 2.2),
            ("wordt", 2.2, 2.4),
            ("veel", 2.4, 2.8),
            ("gemaakt", 2.8, 4.0),
            ("en", 4.0, 4.54),
            ("er", 4.54, 5.4),
            ("wordt", 5.4, 6.6),
            ("meer", 6.6, 7.0),
            ("gedeeld.", 7.0, 7.9),
        )
    )

    assigned = assign_speakers_to_segments([segment], turns)

    assert _speakers_and_text(assigned) == [("Spreker 1", "er wordt veel gemaakt en er wordt meer gedeeld.")]


def test_a_short_reply_in_the_middle_of_a_sentence_is_kept() -> None:
    turns = [
        DiarizationTurn(speaker="Spreker 1", start_s=0.0, end_s=4.3),
        DiarizationTurn(speaker="Spreker 2", start_s=4.4, end_s=5.8),
        DiarizationTurn(speaker="Spreker 1", start_s=6.0, end_s=9.0),
    ]
    segment = _segment_from_words(
        _words(
            ("Wie", 3.6, 3.9),
            ("weet,", 3.9, 4.2),
            ("ja", 4.5, 4.7),
            ("precies.", 4.7, 5.6),
            ("Nou", 6.1, 6.5),
            ("ja.", 6.5, 6.8),
        )
    )

    assigned = assign_speakers_to_segments([segment], turns)

    assert _speakers_and_text(assigned) == [
        ("Spreker 1", "Wie weet,"),
        ("Spreker 2", "ja precies."),
        ("Spreker 1", "Nou ja."),
    ]


def test_a_bare_reply_word_without_punctuation_is_kept() -> None:
    turns = [
        DiarizationTurn(speaker="Spreker 2", start_s=0.0, end_s=3.0),
        DiarizationTurn(speaker="Spreker 1", start_s=3.0, end_s=3.4),
        DiarizationTurn(speaker="Spreker 2", start_s=3.4, end_s=8.0),
    ]
    segment = _segment_from_words(
        _words(("principe.", 2.0, 2.9), ("Ja,", 3.05, 3.35), ("dat", 3.5, 3.7), ("klopt", 3.7, 4.0), ("wel.", 4.0, 4.3))
    )

    assigned = assign_speakers_to_segments([segment], turns)

    assert [speaker for speaker, _ in _speakers_and_text(assigned)] == ["Spreker 2", "Spreker 1", "Spreker 2"]


def test_a_change_just_before_the_end_of_a_sentence_moves_to_that_end() -> None:
    turns = [
        DiarizationTurn(speaker="Spreker 1", start_s=0.0, end_s=7.6),
        DiarizationTurn(speaker="Spreker 2", start_s=7.6, end_s=9.0),
    ]
    segment = _segment_from_words(
        _words(
            ("je", 5.0, 5.5),
            ("kent", 5.5, 5.8),
            ("die", 5.8, 6.0),
            ("cyclus", 6.0, 6.6),
            ("wel", 6.6, 7.0),
            ("volgens", 7.0, 7.55),
            ("mij.", 7.58, 7.72),
            ("Ja,", 7.78, 8.14),
            ("zeker.", 8.14, 8.4),
        )
    )

    assigned = assign_speakers_to_segments([segment], turns)

    assert _speakers_and_text(assigned) == [
        ("Spreker 1", "je kent die cyclus wel volgens mij."),
        ("Spreker 2", "Ja, zeker."),
    ]


def test_a_sentence_start_left_with_the_previous_speaker_moves_on() -> None:
    turns = [
        DiarizationTurn(speaker="Spreker 1", start_s=0.0, end_s=2.0),
        DiarizationTurn(speaker="Spreker 2", start_s=2.0, end_s=5.0),
    ]
    segment = _segment_from_words(
        _words(
            ("Ja,", 0.5, 0.8),
            ("klopt.", 0.8, 1.3),
            ("Niet", 1.4, 1.9),
            ("op", 2.1, 2.3),
            ("de", 2.3, 2.4),
            ("hype", 2.4, 2.8),
            ("meegaan.", 2.8, 3.5),
        )
    )

    assigned = assign_speakers_to_segments([segment], turns)

    assert _speakers_and_text(assigned) == [("Spreker 1", "Ja, klopt."), ("Spreker 2", "Niet op de hype meegaan.")]


def test_a_longer_interruption_is_not_treated_as_a_fragment() -> None:
    turns = [
        DiarizationTurn(speaker="Spreker 1", start_s=0.0, end_s=2.0),
        DiarizationTurn(speaker="Spreker 2", start_s=2.0, end_s=4.5),
        DiarizationTurn(speaker="Spreker 1", start_s=4.5, end_s=7.0),
    ]
    segment = _segment_from_words(
        _words(
            ("en", 1.0, 1.5),
            ("toen", 1.5, 1.9),
            ("maar", 2.1, 2.4),
            ("dat", 2.4, 2.6),
            ("was", 2.6, 2.9),
            ("toch", 2.9, 3.4),
            ("anders", 3.4, 4.3),
            ("zei", 4.6, 5.0),
            ("hij", 5.0, 5.3),
        )
    )

    assigned = assign_speakers_to_segments([segment], turns)

    assert [speaker for speaker, _ in _speakers_and_text(assigned)] == ["Spreker 1", "Spreker 2", "Spreker 1"]
