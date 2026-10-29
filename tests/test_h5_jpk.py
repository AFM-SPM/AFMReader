"""Test the loading of .5h-jpk files."""

# mypy: disable-error-code="arg-type,index"

from contextlib import nullcontext as does_not_raise
from pathlib import Path

import h5py
import numpy as np
import pytest

from AFMReader import h5_jpk  # pylint: disable=no-name-in-module

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"

# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments


@pytest.mark.parametrize(
    ("channel_name", "expected", "expectation"),
    [
        pytest.param("height_trace", ("height", "trace"), does_not_raise(), id="height_trace"),
        pytest.param("depth_retrace", ("depth", "retrace"), does_not_raise(), id="retrace"),
        pytest.param(
            "height_other", None, pytest.raises(ValueError, match="Invalid trace type"), id="invalid trace type"
        ),
        pytest.param(
            "other", None, pytest.raises(ValueError, match="Invalid channel format"), id="invalid channel format"
        ),
    ],
)
def test_parse_channel_name(channel_name: str, expected: tuple[str] | None, expectation) -> None:
    with expectation:
        result = h5_jpk._parse_channel_name(channel_name)
        assert expected == result


@pytest.mark.parametrize(
    ("channel", "expected", "expectation"),
    [
        pytest.param(
            "height_trace",
            ("/Measurement_000/Channel_000", "/Measurement_000", "Height"),
            does_not_raise(),
            id="valid channel",
        ),
        pytest.param(
            "depth_trace",
            None,
            pytest.raises(ValueError, match="not found"),
            id="valid format but channel not in file",
        ),
        pytest.param(
            "other",
            None,
            pytest.raises(ValueError, match="Invalid channel format"),
            id="invalid channel format",
        ),
    ],
)
def test_get_channel_info(h5_file, monkeypatch, channel, expected, expectation) -> None:
    # Skip over _get_channel_info()'s call of _available_channels() for this test only
    monkeypatch.setattr(h5_jpk, "_available_channels", lambda _: {"height_trace": "Measurement_000/Channel_000"})

    with expectation:
        channel_group, measurement_group, dataset_name = h5_jpk._get_channel_info(h5_file, channel)
        assert (channel_group.name, measurement_group.name, dataset_name) == expected


REAL_LENGTH = "position-pattern.grid.ulength"
PX_LENGTH = "position-pattern.grid.ilength"

@pytest.mark.parametrize(
    ("attrs", "expected", "expectation"),
    [
        pytest.param({REAL_LENGTH: 1e-6, PX_LENGTH: 512}, 1.953125, does_not_raise(), id="1 um over 512 px"),
        pytest.param({REAL_LENGTH: 5e-7, PX_LENGTH: 100}, 5.0, does_not_raise(), id="500 nm over 100 px"),
        pytest.param({REAL_LENGTH: 1e-6, PX_LENGTH: 0}, None, pytest.raises(ValueError, match="zero"), id="zero pixels"),
        pytest.param({PX_LENGTH: 512}, None, pytest.raises(KeyError, match="ulength"), id="missing ulength"),
        pytest.param({REAL_LENGTH: 1e-6}, None, pytest.raises(KeyError, match="ilength"), id="missing ilength"),
    ],
)
def test_jpk_pixel_to_nm_scaling_h5(measurement_group, attrs, expected, expectation) -> None:
    measurement_group.attrs.update(attrs)

    with expectation:
        assert h5_jpk._jpk_pixel_to_nm_scaling_h5(measurement_group) == pytest.approx(expected)


MULTIPLIER = "net-encoder.scaling.multiplier"
OFFSET = "net-encoder.scaling.offset"
UNIT = "net-encoder.scaling.unit.unit"

@pytest.mark.parametrize(
        ("attrs", "expected"),
        [
            pytest.param({MULTIPLIER: 2.5e-9, OFFSET: -1e-9, UNIT: "m"}, (2.5e-9, -1e-9, "m"), id="all attributes present"),
            pytest.param({MULTIPLIER: 2.0, OFFSET: 0.5, UNIT: np.bytes_(b"nm")}, (2.0, 0.5, "nm"), id="unit stored as bytes"),
            pytest.param({MULTIPLIER: 2.0, OFFSET: 0.5}, (2.0, 0.5, "m"), id="unit missing defaults to m"),
            pytest.param({UNIT: "V"}, (1.0, 0.0, "V"), id="multiplier and offset missing"),
            pytest.param({}, (1.0, 0.0, "m"), id="no attributes"),
        ],
)
def test_get_z_scaling_h5(channel_group, attrs, expected) -> None:
    channel_group.attrs.update(attrs)
    result = h5_jpk._get_z_scaling_h5(channel_group)

    assert result == expected
    assert all (isinstance(value, float) for value in result[:2])


@pytest.mark.parametrize(
    ("attr", "expected"),
    [
        pytest.param("nm", "nm", id="str"),
        pytest.param(b"nm", "nm", id="bytes"),
        pytest.param(np.bytes_(b"nm"), "nm", id="numpy bytes (as h5py returns fixed-length strings)"),
        pytest.param(np.str_("nm"), "nm", id="numpy str"),
        pytest.param("µm".encode(), "µm", id="non-ascii utf-8 bytes"),
    ],
)
def test_decode_attr(attr, expected) -> None:
    result = h5_jpk._decode_attr(attr)

    assert result == expected
    assert isinstance(result, str)


@pytest.mark.parametrize("attr", [True, np.bool_(True), "true", " TRUE ", b"true", 1, np.int64(2)])
def test_attr_to_bool_true(attr) -> None:
    assert h5_jpk._attr_to_bool(attr) is True


@pytest.mark.parametrize("attr", [False, "false", np.bytes_(b"false"), 0, 0.0, "1", "yes"])
def test_attr_to_bool_false(attr) -> None:
    assert h5_jpk._attr_to_bool(attr) is False


def test_available_channels(jpk_h5_file) -> None:
    assert h5_jpk._available_channels(jpk_h5_file) == {
        "height_trace": "Measurement_000/Channel_000",
        "height_retrace": "Measurement_000/Channel_001",
        "error_trace": "Measurement_000/Channel_002",
    }


def test_available_channels_empty_file() -> None:
    with h5py.File("empty.h5", "w", driver="core", backing_store=False) as f:
        assert h5_jpk._available_channels(f) == {}


@pytest.mark.parametrize(
    ("attrs", "expected", "expectation"),
    [
        pytest.param({"timing-settings.scanRate": 2.5}, 2.5, does_not_raise(), id="scan rate present"),
        pytest.param({}, None, pytest.raises(KeyError, match="scanRate"), id="scan rate missing"),
    ],
)
def test_get_line_rate(measurement_group, attrs, expected, expectation) -> None:
    measurement_group.attrs.update(attrs)

    with expectation:
        assert h5_jpk._get_line_rate(measurement_group) == expected


@pytest.mark.parametrize(
    ("num_frames", "line_rate", "image_size", "expected"),
    [
        pytest.param(3, 2.0, 4, {"frame 0": 0.0, "frame 1": 2.0, "frame 2": 4.0}, id="2s per frame"),
        pytest.param(2, 512.0, 256, {"frame 0": 0.0, "frame 1": 0.5}, id="sub-second frames"),
        pytest.param(1, 1.0, 10, {"frame 0": 0.0}, id="single frame"),
        pytest.param(0, 1.0, 10, {}, id="no frames"),
    ],
)
def test_generate_timestamps(num_frames, line_rate, image_size, expected) -> None:
    assert h5_jpk.generate_timestamps(num_frames, line_rate, image_size) == pytest.approx(expected)


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
    afm_load = h5_jpk.load_h5jpk(RESOURCES / file_name, channel, flip_image)

    assert afm_load.pixel_to_nanometre_scaling == pytest.approx(pixel_to_nm_scaling)
    assert isinstance(afm_load.image, np.ndarray)
    assert afm_load.image.shape == image_shape
    assert afm_load.image.dtype == np.dtype(image_dtype)
    assert isinstance(afm_load.timestamps, timestamps_dtype)
    assert afm_load.image.sum() == pytest.approx(image_sum)
    assert afm_load.timestamps is not None
    assert len(afm_load.timestamps) == afm_load.image.shape[0]
    assert all(
        afm_load.timestamps[f"frame {i}"] < afm_load.timestamps[f"frame {i + 1}"]
        for i in range(len(afm_load.timestamps) - 1)
    )
    assert afm_load.z_units == unit


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
    afm_load = h5_jpk.load_h5jpk(RESOURCES / file_name, channel, flip_image)
    curve_dataset = afm_load.curves_dataset
    assert curve_dataset is not None
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
