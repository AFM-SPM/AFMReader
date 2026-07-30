"""Test the general loader module."""

import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from AFMReader import general_loader

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"


@pytest.mark.parametrize(
    ("filepath", "channel", "error", "message"),
    [
        pytest.param(
            RESOURCES / "sample_0.asd",
            "TP",
            False,
            "Extracted image",
            id="'.asd' success.",
        ),
        pytest.param(
            RESOURCES / "sample_0.asd",
            "notherelol",
            True,
            "'notherelol' not found .asd channel list: TP, PH",
            id="'asd' channel not found.",
        ),
        pytest.param(
            RESOURCES / "sample_0.gwy",
            "ZSensor",
            False,
            "Extracted image",
            id="'.gwy' success.",
        ),
        pytest.param(
            RESOURCES / "sample_0.gwy",
            "SenZor",
            True,
            "'SenZor' not found in .gwy channel list: {'ZSensor': '0', 'Peak Force Error': '1', 'Stiffness': '2', "
            "'LogStiffness': '3', 'Adhesion': '4', 'Deformation': '5', 'Dissipation': '6', 'Height': '7'}",
            id="'.gwy' channel not found.",
        ),
        pytest.param(
            RESOURCES / "sample_0.ibw",
            "HeightTracee",
            False,
            "Extracted image",
            id="'.ibw' success.",
        ),
        pytest.param(
            RESOURCES / "sample_0.ibw",
            "Hight",
            True,
            "'Hight' not in .ibw channel list: ['HeightTracee', 'HeightRetrace', 'ZSensorTrace', 'ZSensorRetrace', "
            "'UserIn0Trace', 'UserIn0Retrace', 'UserIn1Trace', 'UserIn1Retrace']",
            id="'.ibw' channel not found.",
        ),
        pytest.param(
            RESOURCES / "sample_0.jpk",
            "height_trace",
            False,
            "Extracted image",
            id="'.jpk' success.",
        ),
        pytest.param(
            RESOURCES / "sample_0.jpk",
            "might_base",
            True,
            "'might_base' not in .jpk channel list: {'height_retrace': 1, 'measuredHeight_retrace': 2, "
            "'amplitude_retrace': 3, 'phase_retrace': 4, 'error_retrace': 5, 'height_trace': 6, "
            "'measuredHeight_trace': 7, 'amplitude_trace': 8, 'phase_trace': 9, 'error_trace': 10}",
            id="'.jpk' channel not found.",
        ),
        pytest.param(
            RESOURCES / "sample_0.spm",
            "Height",
            False,
            "Extracted channel Height",
            id="'.spm' success.",
        ),
        pytest.param(
            RESOURCES / "sample_0.spm",
            "Force",
            True,
            "'Force' not in .spm channel list: ['Height Sensor', 'Peak Force Error', 'DMTModulus', 'LogDMTModulus', "
            "'Adhesion', 'Deformation', 'Dissipation', 'Height']",
            id="'.spm' channel not found.",
        ),
        pytest.param(
            RESOURCES / "sample_0.stp",
            "",
            False,
            "Extracted image",
            id="'.stp' success.",
        ),
        pytest.param(
            RESOURCES / "sample_0.top",
            "",
            False,
            "Extracted image",
            id="'.top' success.",
        ),
        pytest.param(
            RESOURCES / "sample_0_1.topostats",
            "image",
            False,
            "Extracted .topostats dictionary.",
            id="'.topostats' success.",
        ),
        pytest.param(
            RESOURCES / "sample_0_1.topostats",
            "hgjswbweongp",
            True,
            "'hgjswbweongp' not in available image keys: ['image']",
            id="'.topostats' channel not found.",
        ),
        pytest.param(
            RESOURCES / "sample_0.xxx",
            "NotAChannel",
            True,
            "File type '.xxx' is not currently handled by AFMReader.",
            id="'.xxx' unsupported filetype.",
        ),
    ],
)
def test_load(capsys: pytest.CaptureFixture, filepath: Path, channel: str, error: bool, message: str) -> None:
    """Test loading of all (asd, gwy, ibw, jpk, spm, stp, top, topostats) filetypes."""
    loader = general_loader.LoadFile(filepath, channel)
    if error:
        with pytest.raises(ValueError, match=re.escape(message)):
            loader.load()
    else:
        afm_load = loader.load()
        assert isinstance(afm_load.image, np.ndarray)
        assert isinstance(afm_load.pixel_to_nanometre_scaling, float)
    # check output logs
    captured = capsys.readouterr()
    assert message in captured.err


@pytest.mark.parametrize(
    ("filepath"),
    [
        pytest.param(
            RESOURCES / "not_a_real_file.spm",
            id="File not found error raised.",
        ),
    ],
)
def test_load_filenotfounderror(filepath: Path) -> None:
    """Test that a file not found error is raise when filepath is wrong."""
    loader = general_loader.LoadFile(filepath, "channel")

    with pytest.raises(FileNotFoundError) as execinfo:  # noqa: PT012
        loader.load()
        assert "[not_a_real_file] FileNotFoundError" in execinfo.value


@pytest.mark.parametrize(
    ("file_name", "expected"),
    [
        pytest.param("sample_0.asd", ["TP", "PH"], id="asd"),
        pytest.param(
            "sample_0.gwy",
            [
                "ZSensor",
                "Peak Force Error",
                "Stiffness",
                "LogStiffness",
                "Adhesion",
                "Deformation",
                "Dissipation",
                "Height",
            ],
            id="gwy",
        ),
        pytest.param(
            "sample_0.ibw",
            [
                "HeightTracee",
                "HeightRetrace",
                "ZSensorTrace",
                "ZSensorRetrace",
                "UserIn0Trace",
                "UserIn0Retrace",
                "UserIn1Trace",
                "UserIn1Retrace",
            ],
            id="ibw",
        ),
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
            id="jpk",
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
            id="jpk-qi-image",
        ),
        pytest.param(
            "sample_0.spm",
            [
                "Height Sensor",
                "Peak Force Error",
                "DMTModulus",
                "LogDMTModulus",
                "Adhesion",
                "Deformation",
                "Dissipation",
                "Height",
            ],
            id="spm",
        ),
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
            id="h5-jpk sample_0",
        ),
        pytest.param(
            "sample_0_1.topostats",
            ["image", "image_original"],
            id="topostats 0.1",
        ),
        pytest.param(
            "sample_0_2.topostats",
            ["image", "image_original"],
            id="topostats 0.2",
        ),
    ],
)
def test_get_available_channels_all_formats(file_name: str, expected: Any) -> None:
    """Test get_available_channels for all formats."""
    file_path = RESOURCES / file_name
    loader = general_loader.LoadFile(file_path, channel="")
    channels = loader.get_available_channels()

    if isinstance(expected, list):
        assert sorted(channels) == sorted(expected)
    elif isinstance(expected, tuple) and len(expected) == 2:
        assert isinstance(channels, tuple)
        assert len(channels) == 2
        assert channels[0] == expected[0]
        assert channels[1] == expected[1]
    else:
        assert channels == expected
