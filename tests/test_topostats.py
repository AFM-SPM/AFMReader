"""Test the loading of topostats (HDF5 format) files."""

from pathlib import Path
import pytest

import numpy as np

from AFMReader.topostats import load_topostats

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"


@pytest.mark.parametrize(
    ("file_name", "topostats_file_version", "image_shape", "pixel_to_nm_scaling", "data_keys", "image_sum"),
    [
        pytest.param(
            "sample_0_1.topostats",
            0.1,
            (64, 64),
            1.97601171875,
            {"topostats_file_version", "image", "pixel_to_nm_scaling"},
            112069.51332503435,
            id="version",
        ),
    ],
)
def test_load_topostats(
    file_name: str,
    topostats_file_version: float,
    image_shape: tuple[int, int],
    pixel_to_nm_scaling: float,
    data_keys: set[str],
    image_sum: float,
) -> None:
    """Test the normal operation of loading a .topostats (HDF5 format) file."""
    result_image = np.ndarray
    result_pixel_to_nm_scaling = float
    result_data = dict

    file_path = RESOURCES / file_name
    result_image, result_pixel_to_nm_scaling, result_data = load_topostats(file_path)

    assert result_pixel_to_nm_scaling == pixel_to_nm_scaling
    assert isinstance(result_image, np.ndarray)
    assert result_image.shape == image_shape
    assert set(result_data.keys()) == data_keys
    assert result_data["topostats_file_version"] == topostats_file_version
    assert result_image.sum() == image_sum
