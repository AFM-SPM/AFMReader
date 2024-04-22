"""Test the loading of spm files."""

from pathlib import Path
import pytest

import numpy as np

from topofileformats.spm import load_spm

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"


@pytest.mark.parametrize(
    ("file_name", "channel", "pixel_to_nm_scaling"),
    [
        pytest.param("sample_0.spm", "Height", 0.4940029296875, id="file type 0"),
    ],
)
def test_load_spm(file_name: str, channel: str, pixel_to_nm_scaling: float) -> None:
    """Test the normal operation of loading a .spm file."""
    result_image = np.ndarray
    result_pixel_to_nm_scaling = float

    file_path = RESOURCES / file_name
    result_image, result_pixel_to_nm_scaling = load_spm(file_path, channel=channel)

    assert result_pixel_to_nm_scaling == pixel_to_nm_scaling
    assert isinstance(result_image, np.ndarray)
    assert result_image.shape == (1024, 1024)
    assert result_image.dtype == np.float64
    assert result_image.sum() == 30695369.188316286
