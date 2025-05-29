"""Test the general loader module."""

from pathlib import Path

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
def test_load(caplog: pytest.LogCaptureFixture, filepath: Path, channel: str, error: bool, message: str) -> None:
    """Test loading of all (asd, gwy, ibw, jpk, spm, stp, top, topostats) filetypes."""
    loader = general_loader.LoadFile(filepath, channel)

    image, px2nm = loader.load()

    if not error:
        # check array and px2nm returned
        assert isinstance(image, np.ndarray)
        assert isinstance(px2nm, float)
    else:
        # check when channel wrong
        assert isinstance(image, ValueError)
        assert px2nm is None

    # check output logs
    assert message in caplog.text


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
        _, _ = loader.load()
        assert "[not_a_real_file] FileNotFoundError" in execinfo.value
