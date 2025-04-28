"""Test the loading of topostats (HDF5 format) files."""

from pathlib import Path
import pytest


from AFMReader import topostats

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"

# pylint: disable=too-many-positional-arguments


@pytest.mark.parametrize(
    (
        "file_name",
        "version_key",
        "version",
        "image_shape",
        "pixel_to_nm_scaling",
        "data_keys",
        "image_sum",
    ),
    [
        pytest.param(
            "sample_0_1.topostats",
            "topostats_file_version",
            "0.1",
            (64, 64),
            1.97601171875,
            {
                "pixel_to_nm_scaling",
                "image",
                "topostats_file_version",
            },
            112069.51332503435,
            id="version 0.1",
        ),
        pytest.param(
            "sample_0_2.topostats",
            "topostats_file_version",
            "0.2",
            (64, 64),
            1.97601171875,
            {
                "disordered_traces",
                "filename",
                "grain_curvature_stats",
                "grain_masks",
                "height_profiles",
                "image",
                "image_original",
                "img_path",
                "nodestats",
                "ordered_traces",
                "pixel_to_nm_scaling",
                "splining",
                "topostats_file_version",
            },
            1176.601500471239,
            id="version 0.2",
        ),
        pytest.param(
            "sample_2_3_2.topostats",
            "topostats_version",
            "2.3.2",
            (64, 64),
            1.97601171875,
            {
                "pixel_to_nm_scaling",
                "image",
                "topostats_version",
            },
            112069.51332503435,
            id="version 2.3.2",
            marks=pytest.mark.xfail(reason="Not yet implemented"),
        ),
    ],
)
def test_load_topostats(
    file_name: str,
    version_key: str,
    version: str,
    image_shape: tuple,
    pixel_to_nm_scaling: float,
    data_keys: set[str],
    image_sum: float,
) -> None:
    """Test the normal operation of loading a .topostats (HDF5 format) file."""
    file_path = RESOURCES / file_name
    topostats_data = topostats.load_topostats(file_path)

    assert set(topostats_data.keys()) == data_keys  # type: ignore
    if version_key == "topostats_file_version":
        assert topostats_data[version_key] == float(version)
    else:
        assert topostats_data[version_key] == version
    assert topostats_data["pixel_to_nm_scaling"] == pixel_to_nm_scaling
    assert topostats_data["image"].shape == image_shape
    assert topostats_data["image"].sum() == image_sum
    if version >= "0.2":
        assert isinstance(topostats_data["img_path"], Path)


def test_load_topostats_file_not_found() -> None:
    """Ensure FileNotFound error is raised."""
    with pytest.raises(FileNotFoundError):
        topostats.load_topostats("nonexistant_file.topostats")
