import pytest

from src.cli import apply_cli_overrides, build_parser, describe_speaker_setting
from src.config import PipelineConfig


def _configure(*flags: str, base: PipelineConfig | None = None) -> PipelineConfig:
    args = build_parser().parse_args(["opname.wav", *flags])
    return apply_cli_overrides(base or PipelineConfig(), args)


def test_a_known_number_of_speakers_is_passed_through() -> None:
    assert _configure("--speakers", "3").num_speakers == 3


def test_auto_overrides_a_fixed_count_from_env() -> None:
    config = _configure("--speakers", "auto", base=PipelineConfig(num_speakers=2))

    assert config.num_speakers is None
    assert describe_speaker_setting(config) == "auto"


def test_bounds_switch_to_automatic_detection() -> None:
    config = _configure("--min-speakers", "2", "--max-speakers", "4", base=PipelineConfig(num_speakers=2))

    assert (config.num_speakers, config.min_speakers, config.max_speakers) == (None, 2, 4)
    assert describe_speaker_setting(config) == "auto, 2-4"


def test_without_speaker_flags_the_env_setting_stays() -> None:
    assert _configure(base=PipelineConfig(num_speakers=2)).num_speakers == 2


def test_a_fixed_count_cannot_be_combined_with_bounds() -> None:
    with pytest.raises(ValueError):
        _configure("--speakers", "2", "--max-speakers", "4")


def test_inverted_bounds_are_rejected() -> None:
    with pytest.raises(ValueError):
        _configure("--min-speakers", "5", "--max-speakers", "2")


@pytest.mark.parametrize("flags", [["--speakers", "0"], ["--speakers", "veel"], ["--max-speakers", "auto"]])
def test_invalid_speaker_values_are_refused_by_the_parser(flags: list[str]) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["opname.wav", *flags])
