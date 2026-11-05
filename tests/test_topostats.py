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
            "minicircle_small.topostats",
            "topostats_version",
            "2.4",
            (64, 64),
            1.97601171875,
            {
                "pixel_to_nm_scaling",
                "image_original",
                "config",
                "filename",
                "img_path",
                "topostats_version",
                "image_statistics",
                "full_image_plots",
                "grain_crops",
                "image_name",
                "full_mask_tensor",
            },
            1176.601500617851,
            id="version 2.4",
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
    afm_scan = topostats.load_topostats(file_path, channel="image")

    expected_metadata_keys = data_keys - {"image", "pixel_to_nm_scaling"}
    assert afm_scan.metadata is not None
    assert set(afm_scan.metadata.keys()) == expected_metadata_keys
    if version_key == "topostats_file_version":
        assert afm_scan.metadata[version_key] == float(version)
    else:
        assert afm_scan.metadata[version_key] == version
    assert afm_scan.pixel_to_nanometre_scaling == pytest.approx(pixel_to_nm_scaling)
    assert afm_scan.image.shape == image_shape
    assert afm_scan.image.sum() == pytest.approx(image_sum)
    if version > "0.2":
        assert isinstance(afm_scan.metadata["img_path"], Path)


def test_load_topostats_file_not_found() -> None:
    """Ensure FileNotFound error is raised."""
    with pytest.raises(FileNotFoundError):
        topostats.load_topostats("nonexistant_file.topostats", channel="image")
