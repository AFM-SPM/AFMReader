"""Test the functioning of loading .asd files."""

from pathlib import Path

import pytest

from AFMReader import asd

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"


@pytest.mark.parametrize(
    ("file_name", "channel", "number_of_frames", "pixel_to_nm_scaling"),
    [
        pytest.param("sample_0.asd", "TP", 142, 0.78125, id="file type 0"),
        pytest.param("sample_1.asd", "TP", 197, 2.0, id="file type 1"),
    ],
)
def test_load_asd(file_name: str, channel: str, number_of_frames: int, pixel_to_nm_scaling: float) -> None:
    """Test the normal operation of loading a .asd file."""
    file_path = RESOURCES / file_name
    afm_load = asd.load_asd(file_path, channel)

    assert len(afm_load.image) == number_of_frames  # type: ignore
    assert afm_load.px2nm == pixel_to_nm_scaling
    assert isinstance(afm_load.metadata, dict)


def test_load_asd_file_not_found() -> None:
    """Ensure FileNotFound error is raised."""
    with pytest.raises(FileNotFoundError):
        asd.load_asd("nonexistant_file.asd", channel="TP")


@pytest.mark.parametrize(
    ("file_name", "expected_channels"),
    [
        pytest.param("sample_0.asd", ["TP", "PH"], id="sample_0.asd"),
        pytest.param("sample_1.asd", ["TP", "PH"], id="sample_1.asd"),
        pytest.param("extra_sample.asd", ["TP", "PH"], id="extra_sample.asd"),
    ],
)
def test_get_asd_channels(file_name: str, expected_channels: list[str]) -> None:
    """Test get_asd_channels."""
    file_path = RESOURCES / file_name
    channels = asd.get_asd_channels(file_path)
    assert sorted(channels) == sorted(expected_channels)
