"""Test the loading of ibw files."""

from pathlib import Path

import numpy as np
import pytest

from AFMReader import ibw

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"


@pytest.mark.parametrize(
    ("file_name", "channel", "pixel_to_nm_scaling", "image_shape", "image_dtype", "image_sum"),
    [pytest.param("sample_0.ibw", "HeightTracee", 1.5625, (512, 512), "f4", -218091520.0, id="test image 0")],
)
def test_load_ibw(
    file_name: str,
    channel: str,
    pixel_to_nm_scaling: float,
    image_shape: tuple[int, int],
    image_dtype: type,
    image_sum: float,
) -> None:
    """Test the normal operation of loading an .ibw file."""
    file_path = RESOURCES / file_name
    afm_load = ibw.load_ibw(file_path, channel)

    assert afm_load.pixel_to_nanometre_scaling == pytest.approx(pixel_to_nm_scaling)
    assert isinstance(afm_load.image, np.ndarray)
    assert afm_load.image.shape == image_shape
    assert afm_load.image.dtype == image_dtype
    assert afm_load.image.sum() == pytest.approx(image_sum)


def test_load_ibw_file_not_found() -> None:
    """Ensure FileNotFound error is raised."""
    with pytest.raises(FileNotFoundError):
        ibw.load_ibw("nonexistant_file.ibw", channel="TP")


@pytest.mark.parametrize(
    ("file_name", "expected_channels"),
    [
        pytest.param(
            "sample_0.ibw",
            [
                "HeightTracee",
                "HeightRetrace",
                "ZSensorTrace",
                "ZSensorRetrace",
                "UserIn0Trace",
                "UserIn0Retrace",
                "UserIn1Trace",
                "UserIn1Retrace",
            ],
            id="sample_0.ibw",
        ),
    ],
)
def test_get_ibw_channels(file_name: str, expected_channels: list[str]) -> None:
    """Test get_ibw_channels."""
    file_path = RESOURCES / file_name
    channels = ibw.get_ibw_channels(file_path)
    # The order might not be guaranteed, so sort before comparing
    assert sorted(channels) == sorted(expected_channels)
