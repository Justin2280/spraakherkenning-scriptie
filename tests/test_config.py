import pytest

from src.config import PipelineConfig, parse_speaker_count

SPEAKER_VARIABLES = ("NUM_SPEAKERS", "EXPECTED_SPEAKERS", "MIN_SPEAKERS", "MAX_SPEAKERS")


@pytest.fixture
def clean_speaker_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    # Keep a developer's own .env out of these tests.
    monkeypatch.setattr("src.config.load_dotenv", lambda **_: None)
    for name in SPEAKER_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, None), ("", None), ("auto", None), ("AUTO", None), ("3", 3), (" 2 ", 2), (4, 4)],
)
def test_parse_speaker_count_accepts_numbers_and_auto(value: object, expected: int | None) -> None:
    assert parse_speaker_count(value) == expected


@pytest.mark.parametrize("value", ["0", "-1", "twee", "2.5"])
def test_parse_speaker_count_rejects_nonsense(value: str) -> None:
    with pytest.raises(ValueError):
        parse_speaker_count(value)


def test_speaker_count_defaults_to_automatic(clean_speaker_env: pytest.MonkeyPatch) -> None:
    config = PipelineConfig.from_env()

    assert (config.num_speakers, config.min_speakers, config.max_speakers) == (None, None, None)


def test_speaker_settings_are_read_from_the_environment(clean_speaker_env: pytest.MonkeyPatch) -> None:
    clean_speaker_env.setenv("NUM_SPEAKERS", "auto")
    clean_speaker_env.setenv("MIN_SPEAKERS", "2")
    clean_speaker_env.setenv("MAX_SPEAKERS", "5")

    config = PipelineConfig.from_env()

    assert (config.num_speakers, config.min_speakers, config.max_speakers) == (None, 2, 5)


def test_the_old_expected_speakers_variable_still_works(clean_speaker_env: pytest.MonkeyPatch) -> None:
    clean_speaker_env.setenv("EXPECTED_SPEAKERS", "2")

    assert PipelineConfig.from_env().num_speakers == 2


def test_num_speakers_wins_over_the_old_variable(clean_speaker_env: pytest.MonkeyPatch) -> None:
    clean_speaker_env.setenv("EXPECTED_SPEAKERS", "2")
    clean_speaker_env.setenv("NUM_SPEAKERS", "auto")

    assert PipelineConfig.from_env().num_speakers is None


def test_validate_rejects_inverted_bounds() -> None:
    with pytest.raises(ValueError, match="min_speakers"):
        PipelineConfig(min_speakers=4, max_speakers=2).validate()
