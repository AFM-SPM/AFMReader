"""Test the loading of ibw files."""

from pathlib import Path
import pytest

import numpy as np

from topofileformats.ibw import load_ibw

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"


@pytest.mark.parametrize(
    ("file_name", "channel", "pixel_to_nm_scaling"),
    [pytest.param("sample_0.ibw", "HeightTracee", 1.5625, id="test image 0")],
)
def test_load_ibw(file_name: str, channel: str, pixel_to_nm_scaling: float) -> None:
    """Test the normal operation of loading an .ibw file."""
    result_image = np.ndarray
    result_pixel_to_nm_scaling = float

    file_path = RESOURCES / file_name
    result_image, result_pixel_to_nm_scaling = load_ibw(file_path, channel)

    assert result_pixel_to_nm_scaling == pixel_to_nm_scaling
    assert isinstance(result_image, np.ndarray)
    assert result_image.shape == (512, 512)
    assert result_image.dtype == "f4"
    assert result_image.sum() == -218091520.0
