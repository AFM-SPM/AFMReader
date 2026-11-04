"""Test the loading of jpk-qi-data files."""

from pathlib import Path

import numpy as np
import pytest

from AFMReader import jpk_qi

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"


# @pytest.mark.skip(reason="Test files are too large to store in the repo; a remote storage solution is needed.")
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
            99.99999999999989,
            (256, 256),
            float,
            189899202.3380369,
            (0, 0),
            "Extend",
            {
                "height": (121, 0.0003540424940311566),
                "Vertical Deflection": (121, 1.6379227487083055e-07),
                "Height (Measured)": (121, 0.000420769736210223),
                "Lateral Deflection": (121, -53.68046313645267),
            },
            id="qi-data 0; height_trace",
        ),
        pytest.param(
            "sample_0.jpk-qi-data",
            "slope_trace",
            99.99999999999989,
            (256, 256),
            float,
            7370.169032247781,
            (0, 0),
            "Extend",
            {
                "height": (121, 0.0003540424940311566),
                "Vertical Deflection": (121, 1.6379227487083055e-07),
                "Height (Measured)": (121, 0.000420769736210223),
                "Lateral Deflection": (121, -53.68046313645267),
            },
            id="qi-data 0; slope_trace",
        ),
        pytest.param(
            "sample_0.jpk-qi-data",
            "adhesion_trace",
            99.99999999999989,
            (256, 256),
            float,
            1.550791819528104e-05,
            (0, 0),
            "Extend",
            {
                "height": (121, 0.0003540424940311566),
                "Vertical Deflection": (121, 1.6379227487083055e-07),
                "Height (Measured)": (121, 0.000420769736210223),
                "Lateral Deflection": (121, -53.68046313645267),
            },
            id="qi-data 0; adhesion_trace",
        ),
        pytest.param(
            "sample_0.jpk-qi-data",
            "measuredHeight_trace",
            99.99999999999989,
            (256, 256),
            float,
            223855050.55169588,
            (0, 0),
            "Extend",
            {
                "height": (121, 0.0003540424940311566),
                "Vertical Deflection": (121, 1.6379227487083055e-07),
                "Height (Measured)": (121, 0.000420769736210223),
                "Lateral Deflection": (121, -53.68046313645267),
            },
            id="qi-data 0; measuredHeight_trace",
        ),
        pytest.param(
            "sample_0.jpk-qi-data",
            "vDeflection_trace",
            99.99999999999989,
            (256, 256),
            float,
            0.00017690850877567513,
            (0, 0),
            "Extend",
            {
                "height": (121, 0.0003540424940311566),
                "Vertical Deflection": (121, 1.6379227487083055e-07),
                "Height (Measured)": (121, 0.000420769736210223),
                "Lateral Deflection": (121, -53.68046313645267),
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
    file_path = RESOURCES / file_name
    jpk_qi_loader = jpk_qi.JPKQILoader(file_path, channel)
    afm_load = jpk_qi_loader.load()

    assert afm_load.pixel_to_nanometre_scaling == pytest.approx(pixel_to_nm_scaling)
    assert isinstance(afm_load.image, np.ndarray)
    assert afm_load.image.shape == image_shape
    assert afm_load.image.dtype == image_dtype
    assert afm_load.image.sum() == pytest.approx(image_sum)

    # Test curve data for all targets
    curve_dataset = afm_load.curves_dataset
    assert curve_dataset is not None, "Curves data not found/ is None"
    curve_at_coords = curve_dataset.get_default_volume()[curve_coords[0], curve_coords[1]]
    for curve_channel, (expected_size, expected_sum) in curve_targets.items():
        print(curve_at_coords.keys())
        curve = curve_at_coords[curve_channel][curve_direction]
        assert curve.shape == (expected_size,)
        assert curve.sum() == pytest.approx(expected_sum)

    jpk_qi_loader.close()


def test_load_jpk_data_file_not_found() -> None:
    """Ensure FileNotFound error is raised."""
    with pytest.raises(FileNotFoundError):
        jpk_qi.JPKQILoader("noexistant_file.jpk-qi-data", "TP")
