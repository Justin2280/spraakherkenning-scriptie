from __future__ import annotations

from pathlib import Path

from src.models import AudioChunk, SilenceInterval


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def sanitize_silences(total_duration_s: float, silences: list[SilenceInterval]) -> list[SilenceInterval]:
    cleaned: list[SilenceInterval] = []
    for interval in sorted(silences, key=lambda item: item.start_s):
        start_s = _clamp(interval.start_s, 0.0, total_duration_s)
        end_s = _clamp(interval.end_s, 0.0, total_duration_s)
        if end_s <= start_s:
            continue
        if cleaned and start_s <= cleaned[-1].end_s:
            cleaned[-1].end_s = max(cleaned[-1].end_s, end_s)
        else:
            cleaned.append(SilenceInterval(start_s=start_s, end_s=end_s))
    return cleaned


def speech_regions_from_silences(
    total_duration_s: float,
    silences: list[SilenceInterval],
) -> list[tuple[float, float]]:
    silences = sanitize_silences(total_duration_s, silences)
    if not silences:
        return [(0.0, total_duration_s)]

    regions: list[tuple[float, float]] = []
    cursor = 0.0
    for silence in silences:
        if silence.start_s > cursor:
            regions.append((cursor, silence.start_s))
        cursor = silence.end_s
    if cursor < total_duration_s:
        regions.append((cursor, total_duration_s))
    return regions


def plan_chunks(
    total_duration_s: float,
    silences: list[SilenceInterval],
    target_chunk_s: int,
    overlap_s: int,
    min_chunk_s: int,
    max_chunk_s: int,
    preferred_split_window_s: int,
    source_path: Path | None = None,
) -> list[AudioChunk]:
    if total_duration_s <= 0:
        return []

    cleaned_silences = sanitize_silences(total_duration_s, silences)
    split_candidates = [round((item.start_s + item.end_s) / 2.0, 3) for item in cleaned_silences]

    chunks: list[AudioChunk] = []
    start_s = 0.0
    index = 0

    while start_s < total_duration_s:
        remaining_s = total_duration_s - start_s
        if remaining_s <= max(target_chunk_s, min_chunk_s):
            end_s = total_duration_s
        else:
            target_end_s = min(total_duration_s, start_s + target_chunk_s)
            min_end_s = min(total_duration_s, start_s + min_chunk_s)
            max_end_s = min(total_duration_s, start_s + max_chunk_s)

            candidates = [point for point in split_candidates if min_end_s <= point <= max_end_s]
            preferred = [
                point for point in candidates if abs(point - target_end_s) <= preferred_split_window_s
            ]
            search_pool = preferred or candidates
            end_s = min(search_pool, key=lambda point: (abs(point - target_end_s), point)) if search_pool else target_end_s

        if end_s <= start_s:
            end_s = min(total_duration_s, start_s + target_chunk_s)

        leading_overlap_s = 0.0 if index == 0 else min(float(overlap_s), start_s)
        chunks.append(
            AudioChunk(
                index=index,
                start_s=round(start_s, 3),
                end_s=round(end_s, 3),
                path=None,
                source_path=source_path,
                leading_overlap_s=round(leading_overlap_s, 3),
            )
        )

        if end_s >= total_duration_s:
            break

        next_start_s = max(0.0, end_s - overlap_s)
        if next_start_s <= start_s:
            next_start_s = end_s
        start_s = round(next_start_s, 3)
        index += 1

    return chunks
