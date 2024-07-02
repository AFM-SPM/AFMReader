"""Test the loading of gwy files."""

from pathlib import Path


import pytest
import numpy as np
from AFMReader import gwy


BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"


def test_load_gwy():
    """Test the normal operation of loading a .gwy file."""
    channel = "ZSensor"
    file_path = RESOURCES / "sample_0.gwy"
    result_image, result_pixel_to_nm_scaling = gwy.load_gwy(file_path, channel=channel)
    assert isinstance(result_image, np.ndarray)
    assert result_image.shape == (512, 512)
    assert result_image.sum() == 33836850.232917726
    assert isinstance(result_pixel_to_nm_scaling, float)
    assert result_pixel_to_nm_scaling == 0.8468632812499975


def test_gwy_read_object() -> None:
    """Test reading an object of a `.gwy` file object from an open binary file."""
    with Path.open(RESOURCES / "IO_binary_file.bin", "rb") as open_binary_file:  # pylint: disable=unspecified-encoding
        open_binary_file.seek(19)
        test_dict = {}
        gwy.gwy_read_object(open_file=open_binary_file, data_dict=test_dict)

        assert list(test_dict.keys()) == ["test component", "test object component"]
        assert list(test_dict.values()) == [500, {"test nested component": 3}]


def test_gwy_read_component() -> None:
    """Tests reading a component of a `.gwy` file object from an open binary file."""
    with Path.open(RESOURCES / "IO_binary_file.bin", "rb") as open_binary_file:  # pylint: disable=unspecified-encoding
        open_binary_file.seek(56)
        test_dict = {}
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
