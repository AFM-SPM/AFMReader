"""Test the loading of .stp files."""

from pathlib import Path
import pytest

import numpy as np

from AFMReader.stp import load_stp

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
    [pytest.param("sample_0.stp", 0.9765625, (512, 512), float, -15070620.440757688)],
)
def test_load_stp(
    file_name: str,
    expected_pixel_to_nm_scaling: float,
    expected_image_shape: tuple[int, int],
    expected_image_dtype: type,
    expected_image_sum: float,
) -> None:
    """Test the normal operation of loading a .stp file."""
    file_path = RESOURCES / file_name
    result_image, result_pixel_to_nm_scaling = load_stp(file_path=file_path)

    assert result_pixel_to_nm_scaling == expected_pixel_to_nm_scaling
    assert isinstance(result_image, np.ndarray)
    assert result_image.shape == expected_image_shape
    assert result_image.dtype == expected_image_dtype
    assert result_image.sum() == expected_image_sum
