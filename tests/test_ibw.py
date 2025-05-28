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
    result_image = np.ndarray
    result_pixel_to_nm_scaling = float

    file_path = RESOURCES / file_name
    result_image, result_pixel_to_nm_scaling = ibw.load_ibw(file_path, channel)  # type: ignore

    assert result_pixel_to_nm_scaling == pixel_to_nm_scaling
    assert isinstance(result_image, np.ndarray)
    assert result_image.shape == image_shape
    assert result_image.dtype == image_dtype
    assert result_image.sum() == image_sum


def test_load_ibw_file_not_found() -> None:
    """Ensure FileNotFound error is raised."""
    with pytest.raises(FileNotFoundError):
        ibw.load_ibw("nonexistant_file.ibw", channel="TP")
