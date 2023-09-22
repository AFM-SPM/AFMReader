"""Test the functioning of loading .asd files."""

from topofileformats.asd import load_asd

from pathlib import Path
import pytest

import numpy as np

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"


@pytest.mark.parametrize(
    "file_name, channel, number_of_frames, pixel_to_nm_scaling",
    [
        # File type 0
        (
            "sample_0.asd",
            "TP",
            142,
            0.78125,
        ),
        # File type 1
        (
            "sample_1.asd",
            "TP",
            197,
            2.0,
        ),
    ],
)
def test_load_asd(file_name: str, channel: str, number_of_frames: int, pixel_to_nm_scaling: float) -> None:
    """Test the normal operation of loading a .asd file."""

    result_frames = list
    result_pixel_to_nm_scaling = float
    result_metadata = dict

    file_path = RESOURCES / file_name
    result_frames, result_pixel_to_nm_scaling, result_metadata = load_asd(file_path, channel)

    assert len(result_frames) == number_of_frames
    assert result_pixel_to_nm_scaling == pixel_to_nm_scaling
