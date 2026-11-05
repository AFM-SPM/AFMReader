try:
    from datetime import UTC, datetime
except ImportError:  # Python < 3.11
    from datetime import datetime, timezone

    UTC = timezone.utc  # noqa: UP017
from pathlib import Path

import h5py
import numpy as np
import pytest

from AFMReader import __version__
from AFMReader.h5_saver import H5Saver, find_unused_filename, make_num_min_characters

# --- H5Saver class ---


@pytest.mark.parametrize(
    ("source", "file_exists", "expected_source"),
    [
        pytest.param(None, False, "new", id="new file"),
        pytest.param("jpk-qi-data", False, "jpk-qi-data", id="with source"),
        pytest.param(None, True, "new", id="existing file kept"),
    ],
)
def test_create_file(saver, source, file_exists, expected_source):
    if file_exists:
        with h5py.File(saver.filepath, "w") as f:
            f["existing"] = [1, 2, 3]

    before = datetime.now(UTC)
    h5file = saver.create_file(source=source)
    after = datetime.now(UTC)

    assert isinstance(h5file, h5py.File)
    assert h5file is saver.h5file
    h5file.close()

    with h5py.File(saver.filepath, "r") as f:
        assert f.attrs["AFMReader_version"] == __version__
        assert f.attrs["created_by"] == "AFMReader"
        assert f.attrs["source_type"] == expected_source
        assert before <= datetime.fromisoformat(f.attrs["created_on"]) <= after
        assert ("existing" in f) == file_exists


def test_create_file_without_filepath_raises():
    with pytest.raises(ValueError, match="Either filepath or h5file must be provided."):
        writer = H5Saver(filepath=None)
        writer.create_file()


@pytest.mark.parametrize(
    "channel_units",
    [
        pytest.param(None, id="no units"),
        pytest.param({"height": "m", "vDeflection": "N"}, id="with units"),
    ],
)
def test_setup_curves_group(saver, channel_units):
    saver.create_file()
    saver.setup_curves_group(channel_units=channel_units)

    assert saver.curve_data_group.name == "/Curve_Data"
    assert saver.global_meta_group.name == "/Curve_Data/Global_Metadata"

    unit_attrs = {key: value for key, value in saver.global_meta_group.attrs.items() if key.startswith("channel.unit.")}
    expected = {f"channel.unit.{name}": unit for name, unit in (channel_units or {}).items()}
    assert unit_attrs == expected


def test_setup_curves_group_twice_keeps_existing(saver):
    saver.create_file()
    saver.setup_curves_group({"height": "m"})
    saver.curve_data_group["existing"] = [1, 2, 3]

    saver.setup_curves_group({"vDeflection": "N"})

    assert "existing" in saver.curve_data_group
    assert saver.global_meta_group.attrs["channel.unit.height"] == "m"
    assert saver.global_meta_group.attrs["channel.unit.vDeflection"] == "N"


def test_setup_curves_group_without_file_raises(saver):
    with pytest.raises(AssertionError, match="create_file called before"):
        saver.setup_curves_group()


N_CURVES = 4
ALLOCATED_POINTS = 100

def test_complete_saving(saver, dummy_curves_volume):
    h5file = saver.create_file()
    saver.volume_datasets = {dummy_curves_volume.name: {}}
    saver.volume_points_saved = {dummy_curves_volume.name: {}}
    expected_points = {}

    for s, segment in enumerate(dummy_curves_volume.metadata.segment_names):
        saver.volume_datasets[dummy_curves_volume.name][segment] = {}
        saver.volume_points_saved[dummy_curves_volume.name][segment] = {}
        for c, channel in enumerate(dummy_curves_volume.metadata.channel_units):
            group = h5file.require_group(f"{segment}/{channel}")
            indices = group.create_dataset("Indices", data=np.zeros(N_CURVES + 1, dtype=np.int64))
            data = group.create_dataset("Data", data=np.arange(ALLOCATED_POINTS, dtype=float), maxshape=(None,))

            # A different count for each segment/channel, so mixing them up would be caught
            points_saved = 10 + 5 * s + c
            saver.volume_datasets[dummy_curves_volume.name][segment][channel] = {"Indices": indices, "Data": data}
            saver.volume_points_saved[dummy_curves_volume.name][segment][channel] = points_saved
            expected_points[(segment, channel)] = points_saved

    saver.complete_saving(dummy_curves_volume)

def test_complete_saving_without_file_raises(saver, dummy_curves_volume):
    with pytest.raises(AssertionError, match="create_file called before complete_saving"):
        saver.complete_saving(dummy_curves_volume)

@pytest.mark.parametrize(
        ("shape_x", "shape_y", "minimum_sample_size", "expected"),
        [
            pytest.param(8, 8, 3, range(0, 64, 21), id="standard"),
            pytest.param(2, 2, 2, range(4), id="min equal to shape dim"),
            pytest.param(2, 2, 8, range(4), id="min exceeds pixel count")
        ]
)
def test_get_curves_sample(saver, shape_x, shape_y, minimum_sample_size, expected):
    sample_nums = saver.get_curves_sample(shape_x, shape_y, minimum_sample_size)

    assert sample_nums == expected

# def test_predict_total_points(saver, dummy_curves_volume) -> None:
#     predicted_points_per_channel_segment = saver.predict_total_points(dummy_curves_volume)

#     print(predicted_points_per_channel_segment)


def test_setup_volume():
    pass


def test_save_curve_segments():
    pass


def test_save_curve():
    pass


def test_get_segment_search_terms():
    pass


def test_get_curve_search_terms():
    pass


def test_save_curve_meta_attr():
    pass


# -----------------------


@pytest.mark.parametrize(
    ("num", "min_chars", "expected"),
    [
        pytest.param(1, 3, "001", id="valid extension"),
        pytest.param(20, 2, "20", id="already at min"),
        pytest.param(100, 1, "100", id="already exceeds min"),
    ],
)
def test_mask_num_min_characters(num: int, min_chars: int, expected: str) -> None:
    result = make_num_min_characters(num, min_chars)

    assert expected == result


@pytest.mark.parametrize(
    ("existing", "temp", "expected"),
    [
        pytest.param([], False, "scan.h5-jpk", id="no clash"),
        pytest.param(["scan.h5-jpk"], False, "scan_0.h5-jpk", id="one clash"),
        pytest.param(["scan.h5-jpk", "scan_0.h5-jpk", "scan_1.h5-jpk"], False, "scan_2.h5-jpk", id="several clashes"),
        pytest.param(["scan.h5-jpk", "scan_1.h5-jpk"], False, "scan_0.h5-jpk", id="fills first gap"),
        pytest.param([], True, "temp_scan.h5-jpk", id="temp no clash"),
        pytest.param(["temp_scan.h5-jpk"], True, "temp_scan_0.h5-jpk", id="temp one clash"),
        pytest.param(["scan.h5-jpk"], True, "temp_scan.h5-jpk", id="temp ignores normal files"),
        pytest.param(["temp_scan.h5-jpk"], False, "scan.h5-jpk", id="normal ignores temp files"),
    ],
)
def test_find_unused_filename(tmp_path: Path, existing: list[str], temp: bool, expected: str) -> None:
    for name in existing:
        (tmp_path / name).touch()

    result = find_unused_filename(tmp_path / "scan.jpk-qi-data", temp=temp)

    assert result == tmp_path / expected
    assert not result.exists()


def test_find_unused_filename_keeps_dots_in_stem(tmp_path: Path) -> None:
    result = find_unused_filename(tmp_path / "Tap_ST4.0_00001.spm")

    assert result == tmp_path / "Tap_ST4.0_00001.h5-jpk"
