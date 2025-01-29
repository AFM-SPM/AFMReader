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
            {
                "pixel_to_nm_scaling",
                "image",
                "topostats_file_version",
            },
            112069.51332503435,
            id="version 0.1",
        ),
        pytest.param(
            "sample_0_2.topostats",
            0.2,
            (64, 64),
            1.97601171875,
            {
                "disordered_traces",
                "filename",
                "grain_curvature_stats",
                "grain_masks",
                "height_profiles",
                "image",
                "image_original",
                "img_path",
                "nodestats",
                "ordered_traces",
                "pixel_to_nm_scaling",
                "splining",
                "topostats_file_version",
            },
            1176.601500471239,
            id="version 0.2",
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
    assert set(result_data.keys()) == data_keys  # type: ignore
    assert result_data["topostats_file_version"] == topostats_file_version
    assert result_image.sum() == image_sum
    if topostats_file_version >= 0.2:
        assert isinstance(result_data["img_path"], Path)
