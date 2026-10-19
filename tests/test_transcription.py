from src.models import AudioChunk
from src.transcription.engine import _parse_whisper_cpp_segments


def _token(text: str, start_ms: int, end_ms: int, p: float = 0.9) -> dict:
    return {"text": text, "offsets": {"from": start_ms, "to": end_ms}, "id": 1, "p": p}


def test_whisper_cpp_full_json_is_parsed_with_words() -> None:
    # Shape of `whisper-cli -ojf` output: segments live under "transcription", offsets are in ms.
    payload = {
        "result": {"language": "nl"},
        "transcription": [
            {
                "offsets": {"from": 1000, "to": 3000},
                "text": " Hallo allemaal.",
                "tokens": [
                    {"text": "[_BEG_]", "offsets": {"from": 1000, "to": 1000}, "id": 50365, "p": 1.0},
                    _token(" Hallo", 1000, 1500),
                    _token(" alle", 1600, 1900, p=0.8),
                    _token("maal", 1900, 2400, p=0.7),
                    _token(".", 2400, 2500),
                    {"text": "[_TT_150]", "offsets": {"from": 3000, "to": 3000}, "id": 50515, "p": 1.0},
                ],
            }
        ],
    }
    chunk = AudioChunk(index=1, start_s=600.0, end_s=1200.0, path=None, source_path=None, leading_overlap_s=8.0)

    segments = _parse_whisper_cpp_segments(payload, chunk)

    assert len(segments) == 1
    segment = segments[0]
    assert (segment.text, segment.start_s, segment.end_s, segment.language) == ("Hallo allemaal.", 601.0, 603.0, "nl")
    assert [(word.text, word.start_s, word.end_s) for word in segment.words] == [
        ("Hallo", 601.0, 601.5),
        ("allemaal.", 601.6, 602.5),
    ]
    assert segment.words[1].probability == 0.7


def test_whisper_cpp_json_without_tokens_still_gives_segments() -> None:
    payload = {"transcription": [{"offsets": {"from": 0, "to": 2000}, "text": " Goedemiddag."}]}
    chunk = AudioChunk(index=0, start_s=0.0, end_s=600.0, path=None, source_path=None)

    segments = _parse_whisper_cpp_segments(payload, chunk)

    assert [(segment.text, segment.end_s, segment.words) for segment in segments] == [("Goedemiddag.", 2.0, [])]
