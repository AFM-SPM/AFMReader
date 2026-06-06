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
        "expected_z_units",
    ),
    [
        pytest.param("sample_0.top", 0.9765625, (512, 512), float, 6110573.1148589, "nm"),
        pytest.param("sample_1_um_scale.top", 3.90625, (512, 512), float, 141800.9375, "nm"),
    ],
)
def test_load_top(
    file_name: str,
    expected_pixel_to_nm_scaling: float,
    expected_image_shape: tuple[int, int],
    expected_image_dtype: type,
    expected_image_sum: float,
    expected_z_units: str,
) -> None:
    """Test the normal operation of loading a .top file."""
    file_path = RESOURCES / file_name
    afm_load = load_top(file_path=file_path)

    assert afm_load.px2nm == pytest.approx(expected_pixel_to_nm_scaling)
    assert isinstance(afm_load.image, np.ndarray)
    assert afm_load.image.shape == expected_image_shape
    assert afm_load.image.dtype == expected_image_dtype
    assert afm_load.z_units == expected_z_units
    assert afm_load.image.sum() == pytest.approx(expected_image_sum)
