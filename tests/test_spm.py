"""Test the loading of spm files."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pySPM
import pytest
import re

from AFMReader import spm

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"

# pylint: disable=too-many-positional-arguments


@pytest.mark.parametrize(
    ("file_name", "channel", "pixel_to_nm_scaling", "image_shape", "image_dtype", "image_sum"),
    [
        pytest.param(
            "sample_0.spm", "Height", 0.4940029296875, (1024, 1024), np.float64, 30695369.188316286, id="file type 0"
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

    assert result_pixel_to_nm_scaling == pixel_to_nm_scaling
    assert isinstance(result_image, np.ndarray)
    assert result_image.shape == image_shape
    assert result_image.dtype == image_dtype
    assert result_image.sum() == image_sum


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


@pytest.mark.parametrize(
    ("channel", "message", "error"),
    [
        pytest.param("Height", "Extracted channel Height", False, id="Height found"),
        pytest.param("Height Sensor", "Extracted channel Height Sensor", False, id="Height Sensor found"),
        pytest.param("Peak Force Error", "Extracted channel Peak Force Error", False, id="Peak Force Error found"),
        pytest.param("DMTModulus", "Extracted channel DMTModulus", False, id="DMTModulus found"),
        pytest.param("LogDMTModulus", "Extracted channel LogDMTModulus", False, id="LogDMTModulus found"),
        pytest.param("Adhesion", "Extracted channel Adhesion", False, id="Adhesion found"),
        pytest.param("Deformation", "Extracted channel Deformation", False, id="Deformation found"),
        pytest.param("Dissipation", "Extracted channel Dissipation", False, id="Dissipation found"),
        pytest.param(
            "Might",
            "Might not in .spm channel list: ['Height Sensor', 'Peak Force Error', 'DMTModulus', 'LogDMTModulus', 'Adhesion', 'Deformation', 'Dissipation', 'Height']",
            True,
            id="Might not found",
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
