"""Test the loading of .5h-jpk files."""

from pathlib import Path

import numpy as np
import pytest

from AFMReader import h5_jpk

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"


@pytest.mark.parametrize(
    (
        "file_name",
        "channel",
        "flip_image",
        "pixel_to_nm_scaling",
        "image_shape",
        "image_dtype",
        "timestamps_dtype",
        "image_sum",
    ),
    [
        pytest.param(
            "sample_0.h5-jpk",
            "height_trace",
            True,
            1.171875,
            (4, 128, 128),
            float,
            dict,
            48525583.047271535,
            id="test image 0",
        )
    ],
)
def test_load_h5jpk(
    file_name: str,
    channel: str,
    flip_image: bool,
    pixel_to_nm_scaling: float,
    image_shape: tuple[int, int, int],
    image_dtype: type[np.floating],
    timestamps_dtype: type,
    image_sum: float,
) -> None:
    """Test the normal operation of loading a .h5-jpk file."""
    result_image, result_pixel_to_nm_scaling, results_timestamps = h5_jpk.load_h5jpk(
        RESOURCES / file_name, channel, flip_image
    )

    assert result_pixel_to_nm_scaling == pixel_to_nm_scaling
    assert isinstance(result_image, np.ndarray)
    assert result_image.shape == image_shape
    assert result_image.dtype == np.dtype(image_dtype)
    assert isinstance(results_timestamps, timestamps_dtype)
    assert result_image.sum() == image_sum
    assert len(results_timestamps) == result_image.shape[0]
    assert all(
        results_timestamps[f"frame {i}"] < results_timestamps[f"frame {i+1}"]
        for i in range(len(results_timestamps) - 1)
    )


def test_load_h5jpk_file_not_found() -> None:
    """Ensure FileNotFound error is raised."""
    with pytest.raises(FileNotFoundError):
        h5_jpk.load_h5jpk("nonexistant_file.h5-jpk", channel="TP")
