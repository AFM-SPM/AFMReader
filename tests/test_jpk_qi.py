"""Test the loading of jpk-qi-data files."""

from pathlib import Path

import numpy as np
import pytest

from AFMReader import jpk_qi

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"


@pytest.mark.skip(reason="Test files are too large to store in the repo; a remote storage solution is needed.")
@pytest.mark.parametrize(
    (
        "file_name",
        "channel",
        "pixel_to_nm_scaling",
        "image_shape",
        "image_dtype",
        "image_sum",
        "curve_coords",
        "curve_direction",
        "curve_targets",
    ),
    [
        pytest.param(
            "sample_0.jpk-qi-data",
            "height_trace",
            390.62499999999994,
            (256, 256),
            float,
            412271271.9961158,
            (0, 0),
            "Segment_0",
            {
                "height": (31, 0.00019106601492896875),
                "vDeflection": (31, 2.740962611337846e-07),
                "measuredHeight": (31, 0.00027384485398464497),
                "smoothedMeasuredHeight": (31, -38066578894.999535),
            },
            id="qi-data 0; height_trace",
        ),
        pytest.param(
            "sample_0.jpk-qi-data",
            "slope_trace",
            390.62499999999994,
            (256, 256),
            float,
            267675.3050073493,
            (0, 0),
            "Segment_0",
            {
                "height": (31, 0.00019106601492896875),
                "vDeflection": (31, 2.740962611337846e-07),
                "measuredHeight": (31, 0.00027384485398464497),
                "smoothedMeasuredHeight": (31, -38066578894.999535),
            },
            id="qi-data 0; slope_trace",
        ),
        pytest.param(
            "sample_0.jpk-qi-data",
            "adhesion_trace",
            390.62499999999994,
            (256, 256),
            float,
            0.0008930453784792601,
            (0, 0),
            "Segment_0",
            {
                "height": (31, 0.00019106601492896875),
                "vDeflection": (31, 2.740962611337846e-07),
                "measuredHeight": (31, 0.00027384485398464497),
                "smoothedMeasuredHeight": (31, -38066578894.999535),
            },
            id="qi-data 0; adhesion_trace",
        ),
        pytest.param(
            "sample_0.jpk-qi-data",
            "measuredHeight_trace",
            390.62499999999994,
            (256, 256),
            float,
            590908347.7454677,
            (0, 0),
            "Segment_0",
            {
                "height": (31, 0.00019106601492896875),
                "vDeflection": (31, 2.740962611337846e-07),
                "measuredHeight": (31, 0.00027384485398464497),
                "smoothedMeasuredHeight": (31, -38066578894.999535),
            },
            id="qi-data 0; measuredHeight_trace",
        ),
        pytest.param(
            "sample_0.jpk-qi-data",
            "vDeflection_trace",
            390.62499999999994,
            (256, 256),
            float,
            0.0004062236060247368,
            (0, 0),
            "Segment_0",
            {
                "height": (31, 0.00019106601492896875),
                "vDeflection": (31, 2.740962611337846e-07),
                "measuredHeight": (31, 0.00027384485398464497),
                "smoothedMeasuredHeight": (31, -38066578894.999535),
            },
            id="qi-data 0; vDeflection_trace",
        ),
    ],
)
def test_load_jpk_qi_data(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    file_name: str,
    channel: str,
    pixel_to_nm_scaling: float,
    image_shape: tuple[int, int],
    image_dtype: type,
    image_sum: float,
    curve_coords: tuple[int, int],
    curve_direction: str,
    curve_targets: dict[str, tuple[int, float]],
) -> None:
    """Test the normal operation of loading a .jpk-qi-data file."""
    result_image = np.ndarray
    result_pixel_to_nm_scaling = float
    file_path = RESOURCES / file_name
    jpk_qi_loader = jpk_qi.jpk_qi_loader(file_path, channel)
    result_image, result_pixel_to_nm_scaling, result_curve_data = jpk_qi_loader.load()  # type: ignore

    assert result_pixel_to_nm_scaling == pytest.approx(pixel_to_nm_scaling)
    assert isinstance(result_image, np.ndarray)
    assert result_image.shape == image_shape
    assert result_image.dtype == image_dtype
    assert result_image.sum() == pytest.approx(image_sum)

    # Test curve data for all targets
    all_curves, _, _ = result_curve_data
    curve_at_coords = all_curves[curve_coords[0]][curve_coords[1]]
    for curve_channel, (expected_size, expected_sum) in curve_targets.items():
        curve = curve_at_coords[curve_channel][curve_direction]
        assert curve.shape == (expected_size,)
        assert curve.sum() == pytest.approx(expected_sum)

    jpk_qi_loader.close()  # type: ignore


def test_load_jpk_data_file_not_found() -> None:
    """Ensure FileNotFound error is raised."""
    with pytest.raises(FileNotFoundError):
        jpk_qi.jpk_qi_loader("noexistant_file.jpk-qi-data", "TP")
