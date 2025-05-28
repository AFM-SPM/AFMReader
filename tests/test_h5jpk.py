"""Test the loading of 5h-jpk files."""

from pathlib import Path
import pytest

import numpy as np

from AFMReader import h5_jpk

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"


@pytest.mark.parametrize(
    ("file_name", "channel", "flip_image", "frame", "pixel_to_nm_scaling", "image_shape", "image_dtype", "image_sum"),
    [
        pytest.param(
            "sample_0.h5-jpk",
            "height_trace",
            True,
            0,
            1.171875,
            (128, 128),
            float,
            12014972.417998387,
            id="test image 0",
        )
    ],
)
def test_load_h5jpk(
    file_name: str,
    channel: str,
    flip_image: bool,
    frame: int,
    pixel_to_nm_scaling: float,
    image_shape: tuple[int, int],
    image_dtype: type[np.floating],
    image_sum: float,
) -> None:
    """Test the normal operation of loading a .h5-jpk file."""
    result_image, result_pixel_to_nm_scaling = h5_jpk.load_h5jpk(
        RESOURCES / file_name, channel, flip_image, frame
    )

    assert result_pixel_to_nm_scaling == pixel_to_nm_scaling
    assert isinstance(result_image, np.ndarray)
    assert result_image.shape == image_shape
    assert result_image.dtype == np.dtype(image_dtype)
    assert result_image.sum() == image_sum


def test_load_h5jpk_file_not_found() -> None:
    """Ensure FileNotFound error is raised."""
    with pytest.raises(FileNotFoundError):
        h5_jpk.load_h5jpk("nonexistant_file.h5-jpk", channel="TP")
