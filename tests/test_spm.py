"""Test the loading of spm files."""

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pySPM
import pytest

from AFMReader import spm

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"

# pylint: disable=too-many-positional-arguments


@pytest.mark.parametrize(
    ("file_name", "channel", "pixel_to_nm_scaling", "image_shape", "image_dtype", "image_sum"),
    [
        pytest.param(
            "sample_0.spm",
            "Height",
            0.4940029296875,
            (1024, 1024),
            np.float64,
            30695369.188316286,
            id="bare name -> trace",
        ),
        pytest.param(
            "sample_0.spm",
            "Height trace",
            0.4940029296875,
            (1024, 1024),
            np.float64,
            30695369.188316286,
            id="explicit trace",
        ),
        pytest.param(
            "sample_0.spm",
            "Height Sensor retrace",
            0.4940029296875,
            (1024, 1024),
            np.float64,
            149782939.79099995,
            id="explicit retrace",
        ),
        pytest.param(
            "sample_0.spm",
            "Adhesion retrace ",
            0.4940029296875,
            (1024, 1024),
            np.float64,
            1767.6219379177087,
            id="trailing whitespace stripped",
        ),
    ],
)
def test_load_spm(
    file_name: str,
    channel: str,
    pixel_to_nm_scaling: float,
    image_shape: tuple[int, int],
    image_dtype: type,
    image_sum: float,
) -> None:
    """Test the normal operation of loading a .spm file."""
    result_image = np.ndarray
    result_pixel_to_nm_scaling = float

    file_path = RESOURCES / file_name
    result_image, result_pixel_to_nm_scaling = spm.load_spm(file_path, channel=channel)

    assert result_pixel_to_nm_scaling == pytest.approx(pixel_to_nm_scaling)
    assert isinstance(result_image, np.ndarray)
    assert result_image.shape == image_shape
    assert result_image.dtype == image_dtype
    assert result_image.sum() == pytest.approx(image_sum)


# A layer-dict factory keeps the parametrize table readable.
def _layer(name: bytes, direction: bytes) -> dict:
    """Build a minimal pySPM Bruker layer dict for a channel name and direction."""
    return {b"@2:Image Data": [b'ZS [ZS] "' + name + b'"'], b"Line Direction": [direction]}


@patch("pySPM.Bruker")
@pytest.mark.parametrize(
    ("layers", "requested", "expected_backward"),
    [
        pytest.param(
            [_layer(b"Height Sensor", b"Retrace")],
            "Height Sensor retrace",
            True,
            id="explicit retrace -> backward True",
        ),
        pytest.param(
            [_layer(b"Height Sensor", b"Trace")],
            "Height Sensor trace",
            False,
            id="explicit trace -> backward False",
        ),
        pytest.param(
            [_layer(b"Height Sensor", b"Retrace")],
            "HEIGHT SENSOR RETRACE",
            True,
            id="case-insensitive retrace -> backward True",
        ),
        pytest.param(
            [_layer(b"Height Sensor", b"Trace"), _layer(b"Height Sensor", b"Retrace")],
            "Height Sensor",
            False,
            id="bare name with both directions defaults to trace",
        ),
    ],
)
def test_direction_routes_to_get_channel(
    mock_bruker_cls: "MagicMock",
    layers: list[dict],
    requested: str,
    expected_backward: bool,
) -> None:
    """The requested direction must reach get_channel() as the correct backward flag."""
    scan = mock_bruker_cls.return_value
    scan.layers = layers
    # get_channel returns something array-like enough for the rest of load_spm.
    fake = MagicMock()
    fake.pixels = np.zeros((4, 4))
    fake.pxs.return_value = [(1, "nm"), (1, "nm")]
    scan.get_channel.return_value = fake

    spm.load_spm("whatever.spm", channel=requested)
    scan.get_channel.assert_called_once_with("Height Sensor", backward=expected_backward)


def test_bare_name_retrace_fallback_logs(caplog: pytest.LogCaptureFixture) -> None:
    """Bare 'Adhesion' (retrace-only in sample_0.spm) loads retrace and logs the fallback."""
    spm.load_spm(RESOURCES / "sample_0.spm", channel="Adhesion")
    assert "loading retrace" in caplog.text
    assert "Extracted channel Adhesion" in caplog.text


@patch("pySPM.SPM.SPM_image")
@pytest.mark.parametrize(
    ("filename", "size", "expected_px2nm"),
    [
        pytest.param(
            "square",
            {"pixels": {"x": 1024, "y": 1024}, "real": {"x": 505.859, "y": 505.859, "unit": "nm"}},
            0.4940029296875,
            id="square 0.494",
        ),
        pytest.param(
            "square",
            {"pixels": {"x": 2048, "y": 2048}, "real": {"x": 505.859, "y": 505.859, "unit": "nm"}},
            0.24700146484375,
            id="square 0.247",
        ),
    ],
)
def test_spm_pixel_to_nm_scaling_(
    mock_spm: "MagicMock",
    filename: str,
    size: dict[str, dict[str, int | str]],
    expected_px2nm: float,
) -> None:
    """Test obtaining scaling directly when ``pixel_to_nm_scale`` attribute is zero."""
    # Mock the pxs attribute to be zero which triggers derivation of sacling from the size attributes
    mock_spm.pxs.return_value = [(0, "nm"), (0, "nm")]
    mock_spm.size = size
    result = spm.spm_pixel_to_nm_scaling(filename, mock_spm)
    assert result == expected_px2nm


@patch("pySPM.SPM.SPM_image.pxs")
@pytest.mark.parametrize(
    ("filename", "unit", "x", "y", "expected_px2nm"),
    [
        pytest.param("square_mm", "mm", 0.01, 0.01, 10000, id="mm units; square"),
        pytest.param("square_um", "um", 1.5, 1.5, 1500, id="um units; square"),
        pytest.param("square_nm", "nm", 50, 50, 50, id="nm units; square"),
        pytest.param("square_pm", "pm", 233, 233, 0.233, id="pm units; square"),
        pytest.param("rectangle_thin_pm", "pm", 1, 512, 0.001, id="pm units; rectangular (thin)"),
        pytest.param("rectangle_tall_pm", "pm", 512, 1, 0.512, id="pm units; rectangular (tall)"),
    ],
)
def test__spm_pixel_to_nm_scaling(
    mock_pxs: "MagicMock",
    spm_channel_data: pySPM.SPM.SPM_image,
    filename: str,
    unit: str,
    x: int,
    y: int,
    expected_px2nm: float,
) -> None:
    """Test extraction of pixels to nanometer scaling."""
    mock_pxs.return_value = [(x, unit), (y, unit)]  # issue is that pxs is a func that returns the data
    result = spm.spm_pixel_to_nm_scaling(filename, spm_channel_data)
    assert result == expected_px2nm


def test_load_spm_file_not_found() -> None:
    """Ensure FileNotFound error is raised."""
    with pytest.raises(FileNotFoundError):
        spm.load_spm("nonexistant_file.spm", channel="TP")


def test_spm_channel_list() -> None:
    """spm_channel_list enumerates every channel with its direction and backward flag."""
    scan = pySPM.Bruker(RESOURCES / "sample_0.spm")
    channels = spm.spm_channel_list(scan)

    expected = {
        "Height Sensor retrace": ("Height Sensor", True),
        "Peak Force Error retrace": ("Peak Force Error", True),
        "DMTModulus retrace": ("DMTModulus", True),
        "LogDMTModulus retrace": ("LogDMTModulus", True),
        "Adhesion retrace": ("Adhesion", True),
        "Deformation retrace": ("Deformation", True),
        "Dissipation retrace": ("Dissipation", True),
        "Height trace": ("Height", False),
    }
    assert channels == expected


@pytest.mark.parametrize(
    ("channel", "message", "error"),
    [
        pytest.param(
            "Height",
            "Extracted channel Height",
            False,
            id="trace channel loads",
        ),
        pytest.param(
            "Might",
            "'Might' not in .spm channel list: ['Height Sensor retrace', 'Peak Force Error retrace', "
            + "'DMTModulus retrace', 'LogDMTModulus retrace', "
            + "'Adhesion retrace', 'Deformation retrace', 'Dissipation retrace', 'Height trace']",
            True,
            id="unknown channel raises with channel list",
        ),
    ],
)
def test_load_spm_channel_not_found(
    caplog: pytest.LogCaptureFixture,
    channel: str,
    message: str,
    error: bool,
) -> None:
    """Test the failure and success messages are correct."""
    if error:
        with pytest.raises(ValueError, match=re.escape(message)):
            spm.load_spm(RESOURCES / "sample_0.spm", channel)
    else:
        spm.load_spm(RESOURCES / "sample_0.spm", channel)
    assert message in caplog.text
