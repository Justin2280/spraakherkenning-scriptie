from __future__ import annotations

import re
import textwrap
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass, replace

from src.models import DiarizationTurn, TranscriptSegment, TranscriptWord

# A block keeps growing until a speaker change, a long pause, or this many characters.
# The cap stops a long monologue from rendering as one unreadable wall of text.
MAX_BLOCK_CHARS = 700

UNKNOWN_SPEAKER = "Onbekend"

# A word must lie at least this far inside one speaker's turns to be assigned on timing alone.
CLEAR_WORD_SHARE = 0.5

SENTENCE_END_RE = re.compile(r"[.!?…][\"'”’)\]]*$")
# Any punctuation at the end of a word: a speaker change right after it is plausible.
BREAK_RE = re.compile(r"[.!?…,;:][\"'”’)\]]*$")

# Label corrections only ever move this many words, spanning at most this long.
MAX_CORRECTION_WORDS = 4
MAX_CORRECTION_S = 1.5

# Short replies that may stand on their own in the middle of someone else's sentence.
BACKCHANNEL_WORDS = frozenset(
    {
        "ja", "jaja", "nee", "precies", "klopt", "zeker", "inderdaad", "oké", "oke", "ok", "okay",
        "hm", "hmm", "mhm", "mm", "uhu", "ah", "oh", "prima", "helder", "juist", "tuurlijk", "absoluut",
        "yes", "yeah", "exactly", "right", "sure",
    }
)


def deduplicate_overlap_segments(
    segments: list[TranscriptSegment],
    overlap_seconds: float,
) -> list[TranscriptSegment]:
    if not segments:
        return []

    cleaned: list[TranscriptSegment] = []
    for segment in sorted(segments, key=lambda item: (item.start_s, item.end_s)):
        boundary_s = _leading_boundary(segment, overlap_seconds)
        if boundary_s is None:
            cleaned.append(segment)
        elif segment.words:
            # Cut per word: a segment that starts inside the overlap can run well past it, and
            # dropping it whole loses that speech from both chunks.
            kept = [word for word in segment.words if word.midpoint_s >= boundary_s]
            if len(kept) == len(segment.words):
                cleaned.append(segment)
            elif kept:
                cleaned.append(_with_words(segment, kept))
        elif segment.start_s >= boundary_s:
            cleaned.append(segment)
    return cleaned


def assign_speakers_to_segments(
    segments: list[TranscriptSegment],
    turns: list[DiarizationTurn],
) -> list[TranscriptSegment]:
    """Label every segment with a speaker; expects segments in time order.

    With word timestamps each word gets its own speaker and a segment is split wherever the
    speaker changes, so a question and its answer inside one Whisper segment no longer share a
    label. Without words the speaker with the most overlap takes the whole segment.
    """
    index = _TurnIndex(turns)
    labels = iter(_label_words([word for segment in segments for word in segment.words], index))
    assigned: list[TranscriptSegment] = []
    for segment in segments:
        if not segment.words:
            speaker, _ = index.speaker_for(segment.start_s, segment.end_s)
            assigned.append(replace(segment, speaker=speaker))
            continue
        labelled = [replace(word, speaker=next(labels)) for word in segment.words]
        assigned.extend(_split_on_speaker_change(segment, labelled))
    return assigned


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
                words=previous.words + segment.words,
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


class _TurnIndex:
    """Diarization turns sorted by start time, for fast "who is speaking here" lookups."""

    def __init__(self, turns: list[DiarizationTurn]) -> None:
        self.turns = sorted(turns, key=lambda turn: (turn.start_s, turn.end_s))
        self.starts = [turn.start_s for turn in self.turns]
        # reach[i] points at the latest-ending turn among turns[0..i]. Turns may overlap, so a
        # backwards scan can only stop once nothing that early can still reach the span.
        self.reach: list[int] = []
        for position, turn in enumerate(self.turns):
            if self.reach and self.turns[self.reach[-1]].end_s >= turn.end_s:
                self.reach.append(self.reach[-1])
            else:
                self.reach.append(position)

    def speaker_for(self, start_s: float, end_s: float) -> tuple[str, float]:
        """Return the speaker covering most of the span and the share of the span they cover.

        A span outside every turn goes to the nearest turn with a share of 0.
        """
        if end_s - start_s < 0.02:
            # Zero-length words do occur; look at a sliver around them instead.
            start_s, end_s = start_s - 0.01, end_s + 0.01
        totals: defaultdict[str, float] = defaultdict(float)
        position = bisect_right(self.starts, end_s) - 1
        while position >= 0 and self.turns[self.reach[position]].end_s > start_s:
            turn = self.turns[position]
            overlap = min(end_s, turn.end_s) - max(start_s, turn.start_s)
            if overlap > 0:
                totals[turn.speaker] += overlap
            position -= 1
        if totals:
            speaker, overlap = max(totals.items(), key=lambda item: item[1])
            return speaker, overlap / (end_s - start_s)
        return self._nearest_speaker(start_s, end_s), 0.0

    def _nearest_speaker(self, start_s: float, end_s: float) -> str:
        if not self.turns:
            return UNKNOWN_SPEAKER
        after = bisect_left(self.starts, end_s)
        best: DiarizationTurn | None = None
        best_distance = float("inf")
        if after < len(self.turns):
            best = self.turns[after]
            best_distance = best.start_s - end_s
        if after > 0:
            before = self.turns[self.reach[after - 1]]
            if start_s - before.end_s <= best_distance:
                best = before
        return best.speaker if best else UNKNOWN_SPEAKER


def _label_words(words: list[TranscriptWord], index: _TurnIndex) -> list[str]:
    labels: list[str] = []
    shares: list[float] = []
    for word in words:
        speaker, share = index.speaker_for(word.start_s, word.end_s)
        labels.append(speaker)
        shares.append(share)

    # Word timestamps are only good to a few hundred milliseconds. A word that straddles a turn
    # boundary or sits in a pause therefore follows its sentence: it stays with the previous
    # word unless that word ended a sentence, in which case it joins the next one.
    for position, share in enumerate(shares):
        if share >= CLEAR_WORD_SHARE:
            continue
        if position > 0 and not SENTENCE_END_RE.search(words[position - 1].text):
            labels[position] = labels[position - 1]
        elif position + 1 < len(words):
            labels[position] = labels[position + 1]

    _absorb_fragments(words, labels)
    _snap_changes_to_sentences(words, labels)
    _absorb_fragments(words, labels)
    return labels


def _absorb_fragments(words: list[TranscriptWord], labels: list[str]) -> None:
    """Give short mid-sentence fragments back to the speaker around them.

    When the listener says "ja" or "hm" while the other person keeps talking, pyannote gives that
    moment to the listener, but Whisper only wrote down the talker's words. Those words, such as
    "en er" or "laat die", then land in the listener's turn. A short run of one speaker between
    two runs of the other is treated as such a fragment unless it finishes a sentence or consists
    of reply words only ("ja, precies").
    """
    changed = True
    while changed:
        changed = False
        runs = _runs(labels)
        candidates = []
        for index in range(1, len(runs) - 1):
            start, end = runs[index]
            surrounding = labels[runs[index - 1][0]]
            if surrounding != labels[runs[index + 1][0]]:
                continue
            fragment = words[start:end]
            if (
                _is_short(fragment)
                and not SENTENCE_END_RE.search(fragment[-1].text)
                and not _is_backchannel(fragment)
            ):
                candidates.append((_span_s(fragment), len(fragment), start, end, surrounding))
        # Shortest fragments first: absorbing one can turn its neighbour into a regular run.
        touched: set[int] = set()
        for _, _, start, end, surrounding in sorted(candidates):
            if touched & {start - 1, end}:
                continue
            labels[start:end] = [surrounding] * (end - start)
            touched.update(range(start, end))
            changed = True


def _snap_changes_to_sentences(words: list[TranscriptWord], labels: list[str]) -> None:
    """Move a speaker change that falls mid-sentence to a sentence boundary a few words away.

    "... die hypecyclus van Gartner wel volgens | mij. Ja, zeker." becomes
    "... wel volgens mij. | Ja, zeker." The smaller of the two possible moves wins.
    """
    runs = _runs(labels)
    position = 0
    while position < len(runs) - 1:
        (previous_start, previous_end), (next_start, next_end) = runs[position], runs[position + 1]
        position += 1
        if BREAK_RE.search(words[previous_end - 1].text):
            continue

        options = []
        last_forward = min(next_end, next_start + MAX_CORRECTION_WORDS)
        sentence_end = next(
            (index for index in range(next_start, last_forward) if SENTENCE_END_RE.search(words[index].text)),
            None,
        )
        if sentence_end is not None:
            options.append((sentence_end + 1 - next_start, 0, next_start, sentence_end + 1, labels[previous_start]))
        first_backward = max(previous_start, previous_end - 1 - MAX_CORRECTION_WORDS)
        sentence_end = next(
            (
                index
                for index in range(previous_end - 2, first_backward - 1, -1)
                if SENTENCE_END_RE.search(words[index].text)
            ),
            None,
        )
        if sentence_end is not None:
            options.append((previous_end - sentence_end - 1, 1, sentence_end + 1, previous_end, labels[next_start]))

        for _, _, start, end, speaker in sorted(options):
            moved = words[start:end]
            if _is_short(moved) and not _is_backchannel(moved):
                labels[start:end] = [speaker] * (end - start)
                runs = _runs(labels)
                break


def _runs(labels: list[str]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = 0
    for position in range(1, len(labels) + 1):
        if position == len(labels) or labels[position] != labels[start]:
            runs.append((start, position))
            start = position
    return runs


def _is_short(words: list[TranscriptWord]) -> bool:
    return len(words) <= MAX_CORRECTION_WORDS and _span_s(words) <= MAX_CORRECTION_S


def _span_s(words: list[TranscriptWord]) -> float:
    return words[-1].end_s - words[0].start_s


def _is_backchannel(words: list[TranscriptWord]) -> bool:
    return all(re.sub(r"[^\w'-]", "", word.text.lower()) in BACKCHANNEL_WORDS for word in words)


def _split_on_speaker_change(
    segment: TranscriptSegment,
    words: list[TranscriptWord],
) -> list[TranscriptSegment]:
    runs: list[list[TranscriptWord]] = []
    for word in words:
        if runs and runs[-1][-1].speaker == word.speaker:
            runs[-1].append(word)
        else:
            runs.append([word])
    if len(runs) == 1:
        # Keep Whisper's own text when nothing needs splitting.
        return [replace(segment, speaker=words[0].speaker, words=words)]
    return [replace(_with_words(segment, run), speaker=run[0].speaker) for run in runs]


def _leading_boundary(segment: TranscriptSegment, overlap_seconds: float) -> float | None:
    """Where this chunk takes over from the previous one, or None for the first chunk."""
    if not segment.chunk_index or segment.leading_overlap_s <= 0:
        return None
    return (segment.chunk_start_s or 0.0) + min(segment.leading_overlap_s, overlap_seconds)


def _with_words(segment: TranscriptSegment, words: list[TranscriptWord]) -> TranscriptSegment:
    return replace(
        segment,
        text=" ".join(word.text for word in words),
        start_s=words[0].start_s,
        end_s=words[-1].end_s,
        words=list(words),
    )


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
