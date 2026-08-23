from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, replace

from src.models import DiarizationTurn, TranscriptSegment

# A block keeps growing until a speaker change, a long pause, or this many characters.
# The cap stops a long monologue from rendering as one unreadable wall of text.
MAX_BLOCK_CHARS = 700


def deduplicate_overlap_segments(
    segments: list[TranscriptSegment],
    overlap_seconds: float,
) -> list[TranscriptSegment]:
    if not segments:
        return []

    cleaned: list[TranscriptSegment] = []
    for segment in sorted(segments, key=lambda item: (item.start_s, item.end_s)):
        if segment.chunk_index and segment.leading_overlap_s > 0:
            overlap_limit_s = min(
                segment.leading_overlap_s,
                overlap_seconds,
            )
            boundary_s = (segment.chunk_start_s or 0.0) + overlap_limit_s
            if segment.start_s < boundary_s:
                continue
        cleaned.append(segment)
    return cleaned


def assign_speakers_to_segments(
    segments: list[TranscriptSegment],
    turns: list[DiarizationTurn],
) -> list[TranscriptSegment]:
    return [replace(segment, speaker=_pick_speaker(segment, turns)) for segment in segments]


def merge_adjacent_segments(
    segments: list[TranscriptSegment],
    max_gap_s: float,
) -> list[TranscriptSegment]:
    if not segments:
        return []

    merged: list[TranscriptSegment] = [segments[0]]
    for segment in segments[1:]:
        previous = merged[-1]
        if (
            previous.speaker == segment.speaker
            and segment.start_s - previous.end_s <= max_gap_s
        ):
            merged[-1] = replace(
                previous,
                text=f"{previous.text} {segment.text}".strip(),
                end_s=max(previous.end_s, segment.end_s),
                avg_logprob=_prefer_score(previous.avg_logprob, segment.avg_logprob),
                no_speech_prob=_prefer_score(previous.no_speech_prob, segment.no_speech_prob, lower_is_better=True),
            )
        else:
            merged.append(segment)
    return merged


@dataclass(slots=True)
class TranscriptBlock:
    """One rendered paragraph: a run of segments sharing a speaker and no long pause."""

    speaker: str | None
    start_s: float
    end_s: float
    text: str


def group_into_blocks(
    segments: list[TranscriptSegment],
    paragraph_gap_s: float = 2.0,
) -> list[TranscriptBlock]:
    blocks: list[TranscriptBlock] = []
    for segment in segments:
        text = " ".join(segment.text.split())
        if not text:
            continue

        current = blocks[-1] if blocks else None
        if (
            current is not None
            and current.speaker == segment.speaker
            and segment.start_s - current.end_s <= paragraph_gap_s
            and len(current.text) < MAX_BLOCK_CHARS
        ):
            current.text = f"{current.text} {text}"
            current.end_s = max(current.end_s, segment.end_s)
        else:
            blocks.append(
                TranscriptBlock(
                    speaker=segment.speaker,
                    start_s=segment.start_s,
                    end_s=segment.end_s,
                    text=text,
                )
            )
    return blocks


def render_transcript(
    segments: list[TranscriptSegment],
    *,
    include_timestamps: bool = True,
    include_speaker_labels: bool = True,
    line_width: int = 100,
    paragraph_gap_s: float = 2.0,
) -> str:
    blocks = group_into_blocks(segments, paragraph_gap_s=paragraph_gap_s)
    if not blocks:
        return ""

    # Without diarization no segment carries a speaker, so labels are omitted even when asked for.
    show_speakers = include_speaker_labels and any(block.speaker for block in blocks)

    paragraphs: list[str] = []
    for block in blocks:
        prefix = ""
        if include_timestamps:
            prefix += f"[{format_timestamp_short(block.start_s)}] "
        if show_speakers:
            prefix += f"{block.speaker or 'Onbekend'}: "
        paragraphs.append(_wrap_block(block.text, prefix, line_width))
    return "\n\n".join(paragraphs) + "\n"


def _wrap_block(text: str, prefix: str, line_width: int) -> str:
    width = max(line_width, len(prefix) + 20)
    indent = " " * len(prefix)
    return "\n\n".join(
        textwrap.fill(
            paragraph,
            width=width,
            initial_indent=prefix if index == 0 else indent,
            subsequent_indent=indent,
            break_long_words=False,
            break_on_hyphens=False,
        )
        for index, paragraph in enumerate(_split_long_text(text))
    )


def _split_long_text(text: str) -> list[str]:
    """Break an over-long block into paragraphs at sentence boundaries."""
    if len(text) <= MAX_BLOCK_CHARS:
        return [text]

    paragraphs: list[str] = []
    current = ""
    for sentence in re.split(r"(?<=[.!?…])\s+", text):
        if current and len(current) + len(sentence) + 1 > MAX_BLOCK_CHARS:
            paragraphs.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        paragraphs.append(current)
    return paragraphs


def format_timestamp_short(seconds: float) -> str:
    total_seconds = max(0, round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_timestamp(seconds: float) -> str:
    total_ms = max(0, round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


def _pick_speaker(segment: TranscriptSegment, turns: list[DiarizationTurn]) -> str:
    best_speaker = "Onbekend"
    best_overlap = -1.0
    midpoint = segment.midpoint_s
    closest_distance = float("inf")

    for turn in turns:
        overlap = max(0.0, min(segment.end_s, turn.end_s) - max(segment.start_s, turn.start_s))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = turn.speaker

        if turn.start_s <= midpoint <= turn.end_s:
            closest_distance = 0.0
            best_speaker = turn.speaker
            continue

        distance = min(abs(midpoint - turn.start_s), abs(midpoint - turn.end_s))
        if best_overlap <= 0 and distance < closest_distance:
            closest_distance = distance
            best_speaker = turn.speaker

    return best_speaker


def _prefer_score(
    left: float | None,
    right: float | None,
    lower_is_better: bool = False,
) -> float | None:
    if left is None:
        return right
    if right is None:
        return left
    if lower_is_better:
        return min(left, right)
    return max(left, right)
