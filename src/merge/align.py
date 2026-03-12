from __future__ import annotations

from dataclasses import replace

from src.models import DiarizationTurn, TranscriptSegment


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


def render_transcript(segments: list[TranscriptSegment]) -> str:
    lines: list[str] = []
    for segment in segments:
        speaker = segment.speaker or "Onbekend"
        lines.append(
            f"[{format_timestamp(segment.start_s)} - {format_timestamp(segment.end_s)}] "
            f"{speaker}: {segment.text}"
        )
    return "\n".join(lines)


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
