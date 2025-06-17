"""Test the loading of jpk files."""

from pathlib import Path
import pytest

import numpy as np

from AFMReader import jpk

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"


@pytest.mark.parametrize(
    ("file_name", "channel", "pixel_to_nm_scaling", "image_shape", "image_dtype", "image_sum"),
    [
        pytest.param(
            "sample_0.jpk", "height_trace", 1.2770176335964876, (256, 256), float, 219242202.8256843, id="test image 0"
        ),
        pytest.param(
            "sample_0.jpk-qi-image",
            "height_trace",
            4.999999999999986,
            (100, 100),
            float,
            31593146.16051172,
            id="qi-image 0; height_trace",
        ),
        pytest.param(
            "sample_0.jpk-qi-image",
            "slope_trace",
            4.999999999999986,
            (100, 100),
            float,
            626.3810326940231,
            id="qi-image 0; slope_trace",
        ),
        pytest.param(
            "sample_0.jpk-qi-image",
            "adhesion_trace",
            4.999999999999986,
            (100, 100),
            float,
            1.579363986493833e-06,
            id="qi-image 0; adhesion_trace",
        ),
        pytest.param(
            "sample_0.jpk-qi-image",
            "measuredHeight_trace",
            4.999999999999986,
            (100, 100),
            float,
            32181137.138706185,
            id="qi-image 0; measuredHeight_trace",
        ),
        pytest.param(
            "sample_0.jpk-qi-image",
            "vDeflection_trace",
            4.999999999999986,
            (100, 100),
            float,
            -1.3615033224242345e-05,
            id="qi-image 0; vDeflection_trace",
        ),
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
    result_image, result_pixel_to_nm_scaling = jpk.load_jpk(file_path, channel)  # type: ignore

    assert result_pixel_to_nm_scaling == pytest.approx(pixel_to_nm_scaling)
    assert isinstance(result_image, np.ndarray)
    assert result_image.shape == image_shape
    assert result_image.dtype == image_dtype
    assert result_image.sum() == pytest.approx(image_sum)


def test_load_jpk_file_not_found() -> None:
    """Ensure FileNotFound error is raised."""
    with pytest.raises(FileNotFoundError):
        jpk.load_jpk("nonexistant_file.jpk", channel="TP")
