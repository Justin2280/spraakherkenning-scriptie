"""Check a result.json for speaker-label and chunk-boundary problems, without manual annotation.

The transcript's speaker labels are compared with the diarization turns from the same run. That
measures what gets lost between diarization and transcript; it cannot tell whether pyannote
itself heard the right speaker. Run it before and after a change to see whether the change helped:

    python -m src.diagnose outputs/<name>/result.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

# Turns shorter than this are reported as flicker rather than as real turns.
SHORT_TURN_S = 0.25
# Lost turns shorter than this are mostly a listener's "hm" under the talker's words, which the
# merge step gives back to the talker on purpose; only longer ones are listed.
LISTEN_TURN_S = 1.0
# Half-width of the window around each chunk boundary that is checked for missing text.
BOUNDARY_WINDOW_S = 12.0


def diagnose(result: dict[str, Any]) -> dict[str, Any]:
    turns = sorted(result.get("diarization_turns", []), key=lambda item: item["start_s"])
    segments = sorted(result.get("segments", []), key=lambda item: item["start_s"])
    chunks = result.get("prepared_audio", {}).get("chunks", [])

    per_speaker: dict[str, dict[str, float]] = defaultdict(lambda: {"turns": 0, "seconds": 0.0})
    for turn in turns:
        per_speaker[turn["speaker"]]["turns"] += 1
        per_speaker[turn["speaker"]]["seconds"] += turn["end_s"] - turn["start_s"]

    agreeing_s = 0.0
    mislabeled_s = 0.0
    for segment in segments:
        for turn in turns:
            overlap = _overlap(segment, turn)
            if overlap <= 0:
                continue
            if turn["speaker"] == segment.get("speaker"):
                agreeing_s += overlap
            else:
                mislabeled_s += overlap

    boundaries = [chunk["end_s"] for chunk in chunks[:-1]]
    boundary_gaps = [
        {
            "boundary_s": boundary,
            "untranscribed_speech_s": _untranscribed_speech(
                boundary - BOUNDARY_WINDOW_S, boundary + BOUNDARY_WINDOW_S, segments, turns
            ),
        }
        for boundary in boundaries
    ]
    elsewhere = sorted(
        _untranscribed_speech(start, start + 2 * BOUNDARY_WINDOW_S, segments, turns)
        for start in _window_starts(segments, turns)
        if all(abs(start + BOUNDARY_WINDOW_S - boundary) > 2 * BOUNDARY_WINDOW_S for boundary in boundaries)
    )

    return {
        "speakers": {speaker: dict(stats) for speaker, stats in sorted(per_speaker.items())},
        "turns": len(turns),
        "short_turns": sum(1 for turn in turns if turn["end_s"] - turn["start_s"] < SHORT_TURN_S),
        "diarization_speaker_changes": _count_changes(turns),
        "transcript_segments": len(segments),
        "transcript_speaker_changes": _count_changes(segments),
        "agreeing_seconds": agreeing_s,
        "mislabeled_seconds": mislabeled_s,
        "lost_turns": _lost_turns(turns, segments),
        "boundary_gaps": boundary_gaps,
        "untranscribed_speech_p90_elsewhere_s": elsewhere[int(len(elsewhere) * 0.9)] if elsewhere else None,
    }


def format_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    if not report["turns"]:
        return "No diarization turns in this result (diarization was off?); nothing to compare.\n"

    lines.append(f"Diarization: {report['turns']} turns, {report['diarization_speaker_changes']} speaker changes")
    for speaker, stats in report["speakers"].items():
        lines.append(f"  {speaker}: {stats['turns']} turns, {stats['seconds'] / 60:.1f} min")
    lines.append(f"  turns shorter than {SHORT_TURN_S} s (flicker): {report['short_turns']}")
    lines.append(
        f"Transcript: {report['transcript_segments']} segments, "
        f"{report['transcript_speaker_changes']} speaker changes"
    )

    labelled = report["agreeing_seconds"] + report["mislabeled_seconds"]
    share = report["mislabeled_seconds"] / labelled if labelled else 0.0
    lines.append(
        f"Speech under the wrong label (transcript vs. diarization): "
        f"{report['mislabeled_seconds']:.1f} s ({share:.1%})"
    )

    lost = report["lost_turns"]
    longer = [item for item in lost if item["end_s"] - item["start_s"] >= LISTEN_TURN_S]
    lines.append(f"Turns of {SHORT_TURN_S} s or longer that ended up under another speaker: {len(lost)}")
    lines.append(
        f"  shorter than {LISTEN_TURN_S:.0f} s: {len(lost) - len(longer)} "
        "(mostly a listener's 'hm' under the talker's words, kept with the talker on purpose)"
    )
    lines.append(f"  {LISTEN_TURN_S:.0f} s or longer, worth listening to: {len(longer)}")
    for item in longer:
        lines.append(
            f"  {_clock(item['start_s'])} {item['speaker']} ({item['end_s'] - item['start_s']:.1f} s), "
            f"labelled {item['transcript_speaker']}: {item['text']}"
        )

    p90 = report["untranscribed_speech_p90_elsewhere_s"]
    reference = f" (90th percentile elsewhere: {p90:.1f} s)" if p90 is not None else ""
    lines.append(f"Speech without text around chunk boundaries, +/-{BOUNDARY_WINDOW_S:.0f} s{reference}:")
    for gap in report["boundary_gaps"]:
        lines.append(f"  {_clock(gap['boundary_s'])}: {gap['untranscribed_speech_s']:.1f} s")
    return "\n".join(lines) + "\n"


def _lost_turns(turns: list[dict[str, Any]], segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lost: list[dict[str, Any]] = []
    for turn in turns:
        duration = turn["end_s"] - turn["start_s"]
        if duration < SHORT_TURN_S:
            continue
        by_label: dict[str | None, float] = defaultdict(float)
        for segment in segments:
            overlap = _overlap(segment, turn)
            if overlap > 0:
                by_label[segment.get("speaker")] += overlap
        other = {label: seconds for label, seconds in by_label.items() if label != turn["speaker"]}
        if sum(other.values()) > duration / 2:
            lost.append(
                {
                    "speaker": turn["speaker"],
                    "start_s": turn["start_s"],
                    "end_s": turn["end_s"],
                    "transcript_speaker": max(other, key=other.get),
                    "text": _text_during(turn, segments),
                }
            )
    return lost


def _text_during(turn: dict[str, Any], segments: list[dict[str, Any]], limit: int = 80) -> str:
    pieces: list[str] = []
    for segment in segments:
        if _overlap(segment, turn) <= 0:
            continue
        words = segment.get("words") or []
        if words:
            pieces.extend(word["text"] for word in words if _overlap(word, turn) > 0)
        else:
            # Leave room for the "≈…" and "…" markers within the limit.
            pieces.append(_approximate_excerpt(segment, turn, limit - 4))
    text = " ".join(pieces) or "(no words in this turn)"
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _approximate_excerpt(segment: dict[str, Any], turn: dict[str, Any], limit: int) -> str:
    """Without word timings, guess where the turn falls in the text by its relative position."""
    text = segment["text"]
    duration = segment["end_s"] - segment["start_s"]
    if len(text) <= limit or duration <= 0:
        return text
    midpoint = (max(turn["start_s"], segment["start_s"]) + min(turn["end_s"], segment["end_s"])) / 2
    centre = round((midpoint - segment["start_s"]) / duration * len(text))
    start = min(max(0, centre - limit // 2), len(text) - limit)
    lead = "…" if start > 0 else ""
    tail = "…" if start + limit < len(text) else ""
    return f"≈{lead}{text[start:start + limit].strip()}{tail}"


def _untranscribed_speech(
    start_s: float,
    end_s: float,
    segments: list[dict[str, Any]],
    turns: list[dict[str, Any]],
) -> float:
    covered = sorted(
        (max(start_s, segment["start_s"]), min(end_s, segment["end_s"]))
        for segment in segments
        if segment["end_s"] > start_s and segment["start_s"] < end_s
    )
    holes: list[tuple[float, float]] = []
    cursor = start_s
    for hole_end, covered_end in covered + [(end_s, end_s)]:
        if hole_end > cursor:
            holes.append((cursor, hole_end))
        cursor = max(cursor, covered_end)
    return sum(
        _overlap({"start_s": hole_start, "end_s": hole_end}, turn)
        for hole_start, hole_end in holes
        for turn in turns
    )


def _window_starts(segments: list[dict[str, Any]], turns: list[dict[str, Any]]) -> range:
    end_s = max([item["end_s"] for item in segments + turns], default=0.0)
    step = int(2 * BOUNDARY_WINDOW_S)
    return range(0, max(0, int(end_s) - step), step)


def _overlap(left: dict[str, Any], right: dict[str, Any]) -> float:
    return max(0.0, min(left["end_s"], right["end_s"]) - max(left["start_s"], right["start_s"]))


def _count_changes(items: list[dict[str, Any]]) -> int:
    return sum(1 for previous, current in zip(items, items[1:]) if previous.get("speaker") != current.get("speaker"))


def _clock(seconds: float) -> str:
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes):02d}:{secs:04.1f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path, nargs="+", help="One or more result.json files.")
    args = parser.parse_args(argv)
    for path in args.results:
        result = json.loads(path.read_text(encoding="utf-8"))
        print(f"== {path}")
        print(format_report(diagnose(result)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
