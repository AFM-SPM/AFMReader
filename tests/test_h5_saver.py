try:
    from datetime import UTC, datetime
except ImportError:  # Python < 3.11
    from datetime import datetime, timezone

    UTC = timezone.utc  # noqa: UP017
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from AFMReader import __version__
from AFMReader.data_classes import CurvesVolume
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
def test_create_file(saver, source, file_exists, expected_source) -> None:
    """Create an h5 file and then open it, checking the contents is what is expected."""
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


def test_create_file_without_filepath_raises() -> None:
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
def test_setup_curves_group(saver, channel_units) -> None:
    saver.create_file()
    saver.setup_curves_group(channel_units=channel_units)

    assert saver.curve_data_group.name == "/Curve_Data"
    assert saver.global_meta_group.name == "/Curve_Data/Global_Metadata"

    unit_attrs = {key: value for key, value in saver.global_meta_group.attrs.items() if key.startswith("channel.unit.")}
    expected = {f"channel.unit.{name}": unit for name, unit in (channel_units or {}).items()}
    assert unit_attrs == expected


def test_setup_curves_group_twice_keeps_existing(saver) -> None:
    saver.create_file()
    saver.setup_curves_group({"height": "m"})
    saver.curve_data_group["existing"] = [1, 2, 3]

    saver.setup_curves_group({"vDeflection": "N"})

    assert "existing" in saver.curve_data_group
    assert saver.global_meta_group.attrs["channel.unit.height"] == "m"
    assert saver.global_meta_group.attrs["channel.unit.vDeflection"] == "N"


def test_setup_curves_group_without_file_raises(saver) -> None:
    with pytest.raises(AssertionError, match="create_file called before"):
        saver.setup_curves_group()


class DummyCurvesVolume(CurvesVolume):
    """A 2x3 CurvesVolume where curve n has 10 * (n + 1) points, and pixels in `missing` raise KeyError."""

    def __init__(self, missing=()):
        metadata = SimpleNamespace(segment_names=["extend"], channel_units={"height": "m"})
        super().__init__("test", (2, 3), metadata, False)
        self.missing = set(missing)

    def __getitem__(self, position):
        if position in self.missing:
            raise KeyError(position)
        row, col = position
        return {"height": {"extend": np.zeros(10 * (row * 3 + col + 1))}}


N_CURVES = 4
ALLOCATED_POINTS = 100


def test_complete_saving(volume_saver):
    volume = DummyCurvesVolume()
    data_group = volume_saver.setup_volume(volume)
    volume_saver.volume_points_saved["test"]["extend"]["height"] = 30  # pretend 30 points were saved

    volume_saver.complete_saving(volume)

    assert data_group["extend/Indices/height"][-1] == 30
    assert data_group["extend/Data/height"].shape == (30,)


def test_complete_saving_without_file_raises(saver, dummy_curves_volume) -> None:
    with pytest.raises(AssertionError, match="create_file called before complete_saving"):
        saver.complete_saving(dummy_curves_volume)


@pytest.mark.parametrize(
    ("shape_x", "shape_y", "minimum_sample_size", "expected"),
    [
        pytest.param(8, 8, 3, range(0, 64, 21), id="standard"),
        pytest.param(2, 2, 2, range(4), id="min equal to shape dim"),
        pytest.param(2, 2, 8, range(4), id="min exceeds pixel count"),
    ],
)
def test_get_curves_sample(saver, shape_x, shape_y, minimum_sample_size, expected) -> None:
    sample_nums = saver.get_curves_sample(shape_x, shape_y, minimum_sample_size)

    assert sample_nums == expected


@pytest.mark.parametrize(
    ("sample", "missing", "expected"),
    [
        # Longest sampled curve is curve 4 (50 points): 50 + 10% = 55, times 6 curves
        pytest.param([0, 4], set(), 330, id="uses longest sampled curve"),
        # Curve 0 is missing, so curve 1 (20 points) is used: 20 + 10% = 22, times 6 curves
        pytest.param([0], {(0, 0)}, 132, id="missing curve moves to next one"),
        # Curve 5 is missing with nothing after it, so only curve 1 counts
        pytest.param([1, 5], {(1, 2)}, 132, id="missing last curve is skipped"),
    ],
)
def test_predict_total_points(saver, monkeypatch, sample, missing, expected) -> None:
    monkeypatch.setattr(saver, "get_curves_sample", lambda shape_x, shape_y: sample)

    result = saver.predict_total_points(DummyCurvesVolume(missing=missing))

    assert result == {"extend": {"height": expected}}


def test_predict_total_points_no_readable_curves(saver, monkeypatch) -> None:
    monkeypatch.setattr(saver, "get_curves_sample", lambda shape_x, shape_y: [5])

    with pytest.raises(ValueError):
        saver.predict_total_points(DummyCurvesVolume(missing={(1, 2)}))


def test_setup_volume_create_datasets(volume_saver) -> None:
    data_group = volume_saver.setup_volume(DummyCurvesVolume())

    assert data_group.name == "/Curve_Data/test_VOLM/Data"

    data = data_group["extend/Data/height"]
    assert data.shape == (50,)
    assert data.dtype == np.float32

    indices = data_group["extend/Indices/height"]
    assert indices.shape == (7,)
    assert indices.dtype == np.int32

    datasets = volume_saver.volume_datasets["test"]["extend"]["height"]
    assert datasets["Data"].name == data.name
    assert datasets["Indices"].name == indices.name
    assert volume_saver.volume_points_saved["test"] == {"extend": {"height": 0}}
    assert volume_saver.volume_points_read["test"] == {"extend": {"height": 0}}
    assert volume_saver.volumes_data_buffer["test"] == {"extend": {"height": {"Data": [], "Indices": []}}}
    assert volume_saver.volumes_dims["test"] == (2, 3)


def test_setup_volume_create_metadata_datasets(volume_saver) -> None:
    volume_saver.setup_volume(DummyCurvesVolume(), changing_curve_keys={"force"}, changing_segment_keys={"speed"})

    meta_group = volume_saver.h5file["Curve_Data/test_VOLM/Metadata"]
    assert meta_group["curve.force"].shape == (6,)
    assert meta_group["segment.speed"].shape == (6,)

    [(curve_key, curve_set, curve_buffer)] = volume_saver.curve_work["test"]
    assert curve_key == b"force="
    assert curve_set.name == meta_group["curve.force"].name
    assert curve_buffer == []
    assert volume_saver.curve_search_terms["test"] == [b"force="]

    [(segment_key, segment_set, segment_buffer)] = volume_saver.seg_work["test"]
    assert segment_key == b"speed="
    assert segment_set.name == meta_group["segment.speed"].name
    assert segment_buffer == []
    assert volume_saver.segment_search_terms["test"] == [b"speed="]


def test_save_curve_segments() -> None:
    pass


def test_save_curve() -> None:
    pass


def test_get_segment_search_terms() -> None:
    pass


def test_get_curve_search_terms() -> None:
    pass


def test_save_curve_meta_attr() -> None:
    pass


N_CURVES = 3
N_SEGMENTS = 2


def test_save_segment_meta_attr():
    pass


def test_save_global_meta_metadata(ready_saver, monkeypatch) -> None:
    monkeypatch.setattr("AFMReader.h5_saver.coerce_metadata_value", lambda value: value)
    global_meta = {"spring_constant": 0.1, "num_points": 500, "instrument": "JPK", "settings": {"a": 1}}

    ready_saver.save_global_meta(global_meta, size_x=3e-6, size_y=2e-6, shape_x=3, shape_y=2)

    attrs = ready_saver.global_meta_group.attrs
    assert attrs["spring_constant"] == 0.1
    assert attrs["num_points"] == 500
    assert attrs["instrument"] == "JPK"
    # h5py can't store a dict, fall back to the string form
    assert attrs["settings"] == str({"a": 1})


def test_save_global_meta_grid(ready_saver) -> None:
    ready_saver.save_global_meta({}, size_x=3e-6, size_y=2e-6, shape_x=3, shape_y=2)

    attrs = ready_saver.h5file["Measurement_000"].attrs
    assert attrs["position-pattern.grid.ulength"] == 3e-6
    assert attrs["position-pattern.grid.ilength"] == 3
    assert attrs["position-pattern.grid.vlength"] == 2e-6
    assert attrs["position-pattern.grid.jlength"] == 2
    assert attrs["timing-settings.scanRate"] == 1.0


@pytest.mark.parametrize(
    ("image_name", "channel_name", "is_retrace", "dataset_name"),
    [
        pytest.param("height_trace", "height", "false", "Height", id="trace"),
        pytest.param("height_retrace", "height", "true", "Height", id="retrace"),
        pytest.param("height_Retrace", "height", "true", "Height", id="retrace any case"),
        pytest.param("height", "height", "false", "Height", id="no direction"),
    ],
)
def test_save_image(saver, image_name, channel_name, is_retrace, dataset_name) -> None:
    saver.create_file()
    image = np.arange(6, dtype=float).reshape(2, 3)

    saver.save_image(image, image_name, z_unit="m", idx=0)

    chan_grp = saver.h5file[f"Measurement_000/Channel_{make_num_min_characters(0)}"]
    assert chan_grp.attrs["channel.name"] == channel_name
    assert chan_grp.attrs["retrace"] == is_retrace
    assert chan_grp.attrs["net-encoder.scaling.multiplier"] == 1.0
    assert chan_grp.attrs["net-encoder.scaling.offset"] == 0.0
    assert chan_grp.attrs["net-encoder.scaling.unit.unit"] == "m"
    # Image is stored flat
    np.testing.assert_array_equal(chan_grp[dataset_name][:], [[0], [1], [2], [3], [4], [5]])


def test_save_image_separate_channels(saver) -> None:
    saver.create_file()

    saver.save_image(np.zeros((2, 2)), "height_trace", z_unit="m", idx=0)
    saver.save_image(np.ones((2, 2)), "adhesion_trace", z_unit="N", idx=1)

    np.testing.assert_array_equal(
        saver.h5file[f"Measurement_000/Channel_{make_num_min_characters(0)}"]["Height"][:], np.zeros((4, 1))
    )

    np.testing.assert_array_equal(
        saver.h5file[f"Measurement_000/Channel_{make_num_min_characters(1)}"]["Adhesion"][:], np.ones((4, 1))
    )

    assert (
        saver.h5file[f"Measurement_000/Channel_{make_num_min_characters(1)}"].attrs["net-encoder.scaling.unit.unit"]
        == "N"
    )


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
