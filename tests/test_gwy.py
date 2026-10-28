"""Test the loading of gwy files."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from AFMReader import gwy

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"


@pytest.mark.parametrize(
    ("file_name", "channel", "image_shape", "image_sum", "pixel_to_nm_scaling", "unit"),
    [
        pytest.param(
            "sample_0.gwy", "ZSensor", (512, 512), 33836850.232917726, 0.8468632812499975, "nm", id="test image 0"
        )
    ],
)
def test_load_gwy(
    file_name: str,
    channel: str,
    image_shape: tuple[int, int],
    image_sum: float,
    pixel_to_nm_scaling: float,
    unit: str,
) -> None:
    """Test the normal operation of loading a .gwy file."""
    file_path = RESOURCES / file_name
    afm_load = gwy.load_gwy(file_path, channel=channel)
    assert isinstance(afm_load.image, np.ndarray)
    assert afm_load.image.shape == image_shape
    assert afm_load.image.sum() == pytest.approx(image_sum)
    assert isinstance(afm_load.pixel_to_nanometre_scaling, float)
    assert afm_load.pixel_to_nanometre_scaling == pytest.approx(pixel_to_nm_scaling)
    assert afm_load.z_units == unit


def test_gwy_read_object() -> None:
    """Test reading an object of a `.gwy` file object from an open binary file."""
    with Path.open(RESOURCES / "IO_binary_file.bin", "rb") as open_binary_file:  # pylint: disable=unspecified-encoding
        open_binary_file.seek(19)
        test_dict: dict[Any, Any] = {}
        gwy.gwy_read_object(open_file=open_binary_file, data_dict=test_dict)

        assert list(test_dict.keys()) == ["test component", "test object component"]
        assert list(test_dict.values()) == [500, {"test nested component": 3}]


def test_gwy_read_component() -> None:
    """Tests reading a component of a `.gwy` file object from an open binary file."""
    with Path.open(RESOURCES / "IO_binary_file.bin", "rb") as open_binary_file:  # pylint: disable=unspecified-encoding
        open_binary_file.seek(56)
        test_dict: dict[Any, Any] = {}
        byte_size = gwy.gwy_read_component(initial_byte_pos=56, open_file=open_binary_file, data_dict=test_dict)
        assert byte_size == 73
        assert list(test_dict.keys()) == ["test object component"]
        assert list(test_dict.values()) == [{"test nested component": 3}]


@pytest.mark.parametrize(
    ("gwy_file_data", "expected_channel_ids"),
    [
        pytest.param(
            {
                "/0/data": "Height Channel Data",
                "/0/data/title": "Height",
                "/0/data/meta": "Height Channel Metadata",
                "/1/data": "Amplitude Channel Data",
                "/1/data/title": "Amplitude",
                "/1/data/meta": "Amplitude Channel Metadata",
                "/2/data": "Phase Channel Data",
                "/2/data/title": "Phase",
                "/2/data/meta": "Phase Channel Metadata",
                "/3/data": "Error Channel Data",
                "/3/data/title": "Error",
                "/3/data/meta": "Error Channel Metadata",
            },
            {
                "Height": "0",
                "Amplitude": "1",
                "Phase": "2",
                "Error": "3",
            },
            id="leading slash",
        ),
        pytest.param(
            {
                "0/data": "Height Channel Data",
                "0/data/title": "Height",
                "0/data/meta": "Height Channel Metadata",
                "1/data": "Amplitude Channel Data",
                "1/data/title": "Amplitude",
                "1/data/meta": "Amplitude Channel Metadata",
                "2/data": "Phase Channel Data",
                "2/data/title": "Phase",
                "2/data/meta": "Phase Channel Metadata",
                "3/data": "Error Channel Data",
                "3/data/title": "Error",
                "3/data/meta": "Error Channel Metadata",
            },
            {
                "Height": "0",
                "Amplitude": "1",
                "Phase": "2",
                "Error": "3",
            },
            id="no leading slash",
        ),
    ],
)
def test_gwy_get_channels(gwy_file_data: dict, expected_channel_ids: dict) -> None:
    """Tests getting the channels of a `.gwy` file."""
    channel_ids = gwy.gwy_get_channels(gwy_file_structure=gwy_file_data)

    assert channel_ids == expected_channel_ids


def test_read_gwy_component_dtype() -> None:
    """Test reading a data type of a `.gwy` file component from an open binary file."""
    with Path.open(RESOURCES / "IO_binary_file.bin", "rb") as open_binary_file:  # pylint: disable=unspecified-encoding
        open_binary_file.seek(19)
        value = gwy.read_gwy_component_dtype(open_binary_file)
        assert isinstance(value, str)
        assert value == "D"


def test_load_gwy_file_not_found() -> None:
    """Ensure FileNotFound error is raised."""
    with pytest.raises(FileNotFoundError):
        gwy.load_gwy("nonexistant_file.gwy", channel="TP")


@pytest.mark.parametrize(
    ("file_name", "expected_channels"),
    [
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
            id="sample_0.gwy",
        ),
    ],
)
def test_get_gwy_channels(file_name: str, expected_channels: list[str]) -> None:
    """Test get_gwy_channels."""
    file_path = RESOURCES / file_name
    channels = gwy.get_gwy_channels(file_path)
    # The order might not be guaranteed, so sort before comparing
    assert sorted(channels) == sorted(expected_channels)
