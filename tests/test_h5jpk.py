"""Test the loading of .5h-jpk files."""

# mypy: disable-error-code="arg-type,index"

from pathlib import Path

import numpy as np
import pytest

from AFMReader import h5_jpk  # pylint: disable=no-name-in-module

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"

# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments


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
        "unit",
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
            "nm",
            id="test image 0",
        ),
        pytest.param(
            "sample_0.h5-jpk",
            "height_retrace",
            True,
            1.171875,
            (4, 128, 128),
            float,
            dict,
            48517762.77380567,
            "nm",
            id="test image 0",
        ),
        pytest.param(
            "sample_0.h5-jpk",
            "error_trace",
            True,
            1.171875,
            (4, 128, 128),
            float,
            dict,
            -360.7100517131785,
            "nm",
            id="test image 0",
        ),
        pytest.param(
            "sample_0.h5-jpk",
            "error_retrace",
            True,
            1.171875,
            (4, 128, 128),
            float,
            dict,
            367.81162274907103,
            "nm",
            id="test image 0",
        ),
        pytest.param(
            "sample_0.h5-jpk",
            "phase_retrace",
            True,
            1.171875,
            (4, 128, 128),
            float,
            dict,
            1741828.7412469066,
            "deg",
            id="test image 0",
        ),
        pytest.param(
            "sample_0.h5-jpk",
            "phase_trace",
            True,
            1.171875,
            (4, 128, 128),
            float,
            dict,
            1734511.5577225098,
            "deg",
            id="test image 0",
        ),
        pytest.param(
            "sample_0.h5-jpk",
            "amplitude_retrace",
            True,
            1.171875,
            (4, 128, 128),
            float,
            dict,
            275567.73614739266,
            "nm",
            id="test image 0",
        ),
        pytest.param(
            "sample_0.h5-jpk",
            "amplitude_trace",
            True,
            1.171875,
            (4, 128, 128),
            float,
            dict,
            276296.25732934737,
            "nm",
            id="test image 0",
        ),
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
    unit: str,
) -> None:
    """Test the normal operation of loading a .h5-jpk file."""
    result_image, result_pixel_to_nm_scaling, results_timestamps, result_unit = h5_jpk.load_h5jpk(  # type: ignore[misc]
        RESOURCES / file_name, channel, flip_image
    )

    assert result_pixel_to_nm_scaling == pytest.approx(pixel_to_nm_scaling)
    assert isinstance(result_image, np.ndarray)
    assert result_image.shape == image_shape
    assert result_image.dtype == np.dtype(image_dtype)
    assert isinstance(results_timestamps, timestamps_dtype)
    assert result_image.sum() == pytest.approx(image_sum)
    assert len(results_timestamps) == result_image.shape[0]
    assert all(
        results_timestamps[f"frame {i}"] < results_timestamps[f"frame {i + 1}"]
        for i in range(len(results_timestamps) - 1)
    )
    assert result_unit == unit


@pytest.mark.skip(reason="Test files are too large to store in the repo; a remote storage solution is needed.")
@pytest.mark.parametrize(
    (
        "file_name",
        "channel",
        "flip_image",
        "curve_coords",
        "curve_direction",
        "curve_targets",
    ),
    [
        pytest.param(
            "sample_0_curves.h5-jpk",
            "height_trace",
            True,
            (0, 0),
            "Segment_0",
            {
                "height": (31, 0.00019106604),
                "measuredHeight": (31, 0.00027384484),
                "smoothedMeasuredHeight": (31, -38066577408.0),
                "vDeflection": (31, 2.7409627e-07),
            },
            id="test curves 0",
        ),
    ],
)
def test_load_h5jpk_curves(
    file_name: str,
    channel: str,
    flip_image: bool,
    curve_coords: tuple[int, int],
    curve_direction: str,
    curve_targets: dict[str, tuple[int, float]],
) -> None:
    """
    Test loading of curve data from a .h5-jpk file.

    Parameters
    ----------
    file_name : str
        The name of the .h5-jpk file to load (should be located in the test resources directory).
    channel : str
        The channel to load curve data for.
    flip_image : bool
        Whether to flip the image vertically.
    curve_coords : tuple[int, int]
        The coordinates of the curve to load.
    curve_direction : str
        The direction of the curve to load.
    curve_targets : dict[str, tuple[int, float]]
        A dictionary mapping curve channels to their expected size and sum, used for validating the loaded curve data.
    """
    _, _, _, _, curve_dataset = h5_jpk.load_h5jpk(RESOURCES / file_name, channel, flip_image)  # type: ignore[misc]
    curve_at_coords = curve_dataset.get_default_volume()[curve_coords[0], curve_coords[1]]
    for curve_channel, (expected_size, expected_sum) in curve_targets.items():
        curve = curve_at_coords[curve_channel][curve_direction]
        assert curve.shape == (expected_size,)
        assert curve.sum() == pytest.approx(expected_sum)


def test_load_h5jpk_file_not_found() -> None:
    """Ensure FileNotFound error is raised."""
    with pytest.raises(FileNotFoundError):
        h5_jpk.load_h5jpk("nonexistant_file.h5-jpk", channel="TP")


@pytest.mark.parametrize(
    ("file_name", "expected_channels"),
    [
        pytest.param(
            "sample_0.h5-jpk",
            [
                "error_trace",
                "height_trace",
                "phase_retrace",
                "height_retrace",
                "measuredheight_trace",
                "error_retrace",
                "amplitude_trace",
                "amplitude_retrace",
                "phase_trace",
            ],
            id="sample_0.h5-jpk",
        ),
    ],
)
def test_get_h5jpk_channels(file_name: str, expected_channels: list[str]) -> None:
    """Test get_h5jpk_channels."""
    file_path = RESOURCES / file_name
    channels = h5_jpk.get_h5jpk_channels(file_path)
    # The order might not be guaranteed, so sort before comparing
    assert sorted(channels) == sorted(expected_channels)
