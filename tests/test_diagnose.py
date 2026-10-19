from src.diagnose import diagnose, format_report


def _result() -> dict:
    return {
        "prepared_audio": {
            "chunks": [
                {"index": 0, "start_s": 0.0, "end_s": 60.0},
                {"index": 1, "start_s": 52.0, "end_s": 120.0},
            ]
        },
        "diarization_turns": [
            {"speaker": "Spreker 1", "start_s": 0.0, "end_s": 10.0},
            {"speaker": "Spreker 2", "start_s": 10.0, "end_s": 11.0},
            {"speaker": "Spreker 1", "start_s": 11.0, "end_s": 70.0},
        ],
        "segments": [
            # The reply at 10-11 s sits inside speaker 1's segment, and 60-68 s has no text.
            {"speaker": "Spreker 1", "start_s": 0.0, "end_s": 59.0, "text": "lange beurt"},
            {"speaker": "Spreker 1", "start_s": 68.0, "end_s": 70.0, "text": "verder"},
        ],
    }


def test_diagnose_finds_a_reply_under_the_wrong_speaker() -> None:
    report = diagnose(_result())

    assert report["mislabeled_seconds"] == 1.0
    assert [(turn["speaker"], turn["transcript_speaker"]) for turn in report["lost_turns"]] == [
        ("Spreker 2", "Spreker 1")
    ]


def test_diagnose_measures_speech_without_text_at_chunk_boundaries() -> None:
    report = diagnose(_result())

    assert report["boundary_gaps"] == [{"boundary_s": 60.0, "untranscribed_speech_s": 9.0}]


def test_report_without_diarization_says_so() -> None:
    assert "nothing to compare" in format_report(diagnose({"segments": []}))
