"""Test the loading of .top files."""

from pathlib import Path

import numpy as np
import pytest

from AFMReader.top import load_top

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"


@pytest.mark.parametrize(
    (
        "file_name",
        "expected_pixel_to_nm_scaling",
        "expected_image_shape",
        "expected_image_dtype",
        "expected_image_sum",
    ),
    [
        pytest.param("sample_0.top", 0.9765625, (512, 512), float, 6034386.429246264),
        pytest.param("sample_1_um_scale.top", 3.90625, (512, 512), float, 125175.99999999997),
    ],
)
def test_load_top(
    file_name: str,
    expected_pixel_to_nm_scaling: float,
    expected_image_shape: tuple[int, int],
    expected_image_dtype: type,
    expected_image_sum: float,
) -> None:
    """Test the normal operation of loading a .top file."""
    file_path = RESOURCES / file_name
    result_image, result_pixel_to_nm_scaling = load_top(file_path=file_path)

    assert result_pixel_to_nm_scaling == pytest.approx(expected_pixel_to_nm_scaling)
    assert isinstance(result_image, np.ndarray)
    assert result_image.shape == expected_image_shape
    assert result_image.dtype == expected_image_dtype
    assert result_image.sum() == pytest.approx(expected_image_sum)
