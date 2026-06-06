"""Test the loading of jpk files."""

from pathlib import Path

import numpy as np
import pytest

from AFMReader import jpk

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"


# pylint: disable=too-many-arguments, too-many-positional-arguments
@pytest.mark.parametrize(
    ("file_name", "channel", "pixel_to_nm_scaling", "image_shape", "image_dtype", "image_sum", "unit"),
    [
        pytest.param(
            "sample_0.jpk",
            "height_trace",
            1.2770176335964876,
            (256, 256),
            float,
            219242202.8256843,
            "nm",
            id="test image 0",
        ),
        pytest.param(
            "sample_0.jpk-qi-image",
            "height_trace",
            4.999999999999986,
            (100, 100),
            float,
            31593146.16051172,
            "nm",
            id="qi-image 0; height_trace",
        ),
        pytest.param(
            "sample_0.jpk-qi-image",
            "slope_trace",
            4.999999999999986,
            (100, 100),
            float,
            626.3810326940231,
            "N/m",
            id="qi-image 0; slope_trace",
        ),
        pytest.param(
            "sample_0.jpk-qi-image",
            "adhesion_trace",
            4.999999999999986,
            (100, 100),
            float,
            1.579363986493833e-06,
            "N",
            id="qi-image 0; adhesion_trace",
        ),
        pytest.param(
            "sample_0.jpk-qi-image",
            "measuredHeight_trace",
            4.999999999999986,
            (100, 100),
            float,
            32181137.138706185,
            "nm",
            id="qi-image 0; measuredHeight_trace",
        ),
        pytest.param(
            "sample_0.jpk-qi-image",
            "vDeflection_trace",
            4.999999999999986,
            (100, 100),
            float,
            -1.3615033224242345e-05,
            "N",
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
    unit: str,
) -> None:
    """Test the normal operation of loading a .jpk file."""
    result_image = np.ndarray
    result_pixel_to_nm_scaling = float
    result_unit = str
    file_path = RESOURCES / file_name
    result_image, result_pixel_to_nm_scaling, result_unit = jpk.load_jpk(file_path, channel)  # type: ignore

    assert result_pixel_to_nm_scaling == pytest.approx(pixel_to_nm_scaling)
    assert isinstance(result_image, np.ndarray)
    assert result_image.shape == image_shape
    assert result_image.dtype == image_dtype
    assert result_image.sum() == pytest.approx(image_sum)
    assert result_unit == unit


def test_load_jpk_file_not_found() -> None:
    """Ensure FileNotFound error is raised."""
    with pytest.raises(FileNotFoundError):
        jpk.load_jpk("nonexistant_file.jpk", channel="TP")


@pytest.mark.parametrize(
    ("file_name", "expected"),
    [
        pytest.param(
            "sample_0.jpk",
            {
                "height_retrace": 1,
                "measuredHeight_retrace": 2,
                "amplitude_retrace": 3,
                "phase_retrace": 4,
                "error_retrace": 5,
                "height_trace": 6,
                "measuredHeight_trace": 7,
                "amplitude_trace": 8,
                "phase_trace": 9,
                "error_trace": 10,
            },
            id="sample_0.jpk",
        ),
        pytest.param(
            "sample_0.jpk-qi-image",
            {
                "measuredHeight_trace": 3,
                "vDeflection_trace": 2,
                "adhesion_trace": 4,
                "height_trace": 5,
                "slope_trace": 6,
            },
            id="sample_0.jpk-qi-image",
        ),
    ],
)
def test_get_jpk_channels(file_name: str, expected: dict[str, int]) -> None:
    """Test get_jpk_channels."""
    file_path = RESOURCES / file_name
    channels = jpk.get_jpk_channels(file_path)
    assert channels == expected
