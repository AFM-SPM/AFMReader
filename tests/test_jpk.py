"""Test the loading of jpk files."""

from pathlib import Path
import pytest

import numpy as np

from AFMReader.jpk import load_jpk

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"


@pytest.mark.parametrize(
    ("file_name", "channel", "pixel_to_nm_scaling", "image_shape", "image_dtype", "image_sum"),
    [
        pytest.param(
            "sample_0.jpk", "height_trace", 1.2770176335964876, (256, 256), float, 286598232.9308627, id="test image 0"
        )
    ],
)
def test_load_jpk(
    file_name: str,
    channel: str,
    pixel_to_nm_scaling: float,
    image_shape: tuple[int, int],
    image_dtype: type,
    image_sum: float,
) -> None:
    """Test the normal operation of loading a .jpk file."""
    result_image = np.ndarray
    result_pixel_to_nm_scaling = float

    file_path = RESOURCES / file_name
    result_image, result_pixel_to_nm_scaling = load_jpk(file_path, channel)

    assert result_pixel_to_nm_scaling == pixel_to_nm_scaling
    assert isinstance(result_image, np.ndarray)
    assert result_image.shape == image_shape
    assert result_image.dtype == image_dtype
    assert result_image.sum() == image_sum
