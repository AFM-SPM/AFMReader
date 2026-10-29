"""Fixtures for testing."""

from pathlib import Path

import h5py
import numpy as np
import pySPM
import pytest

from AFMReader import h5_jpk

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"


@pytest.fixture()
def volume_args():
    """Constants to be used in this script and across other volume related tests."""
    return {
        "shape": (2, 2),
        "segments": ["extend", "retract"],
        "channels": ["height", "vDeflection"]
    }


@pytest.fixture()
def spm_channel_data() -> pySPM.SPM.SPM_image:
    """Instantiate channel data from a LoadScans object."""
    scan = pySPM.Bruker(RESOURCES / "sample_0.spm")
    return scan.get_channel("Height")


@pytest.fixture
def h5_file():
    """Minimal HDF5 file with one measurement group containing one channel."""
    with h5py.File("in_memory.h5", "w", driver="core", backing_store=False) as f:
        f.create_group("Measurement_000/Channel_000")
        yield f


@pytest.fixture
def measurement_group():
    """Empty measurement group for an HDF5 file."""
    with h5py.File("in_memory.h5", "w", driver="core", backing_store=False) as f:
        yield f.create_group("Measurement_000")


@pytest.fixture
def channel_group():
    """Empty channel group for an HDF5 file."""
    with h5py.File("in_memory.h5", "w", driver="core", backing_store=False) as f:
        yield f.create_group("Measurement_000/Channel_001")

@pytest.fixture
def volume_data_group(h5_file, volume_args):
    """Volume group where curve k has k + 1 points, with values equal to their position in the flat data."""
    shape = volume_args["shape"]
    segments = volume_args["segments"]
    channels = volume_args["channels"]

    group = h5_file.create_group("Measurement_000/Volume")
    n_curves = shape[0] * shape[1]
    lengths = np.arange(1, n_curves + 1)  # different lengths per curve
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    for segment in segments:
        for channel in channels:
            group[f"Curves/{segment}/Indices/{channel}"] = offsets
            group[f"Curves/{segment}/Data/{channel}"] = np.arange(offsets[-1], dtype=np.float64)
    return group


@pytest.fixture
def jpk_h5_file():
    """HDF5 file laid out like a JPK file."""
    with h5py.File("in_memory.h5", "w", driver="core", backing_store=False) as f:

        def add_channel(path: str, name=None, retrace=None) -> None:
            group = f.create_group(path)
            if name is not None:
                group.attrs["channel.name"] = name
            if retrace is not None:
                group.attrs["retrace"] = retrace

        add_channel("Measurement_000/Channel_000", name="height", retrace="false")
        add_channel("Measurement_000/Channel_001", name="height", retrace="true")
        add_channel("Measurement_000/Channel_002", name=np.bytes_(b" Error "))
        add_channel("Measurement_000/Channel_003")
        add_channel("Measurement_000/Other_000", name="ignored")
        add_channel("Measurement_001/Channel_000", name="height", retrace="false")
        add_channel("Metadata/Channel_000", name="ignored")
        yield f


@pytest.fixture
def curve_meta_group(volume_args):
    """Curve metadata group whose per-curve and per-segment values equal their flat index."""
    shape = volume_args["shape"]
    segments = volume_args["segments"]

    with h5py.File("in_memory.h5", "w", driver="core", backing_store=False) as f:
        group = f.create_group("curve_metadata")
        n_curves = shape[0] * shape[1]
        group.create_dataset("curve.index", data=np.arange(n_curves))
        group.create_dataset("segment.index", data=np.arange(n_curves * len(segments)))
        group.attrs["curve.instrument"] = "JPK"  # constant value, stored as an attribute
        yield group


@pytest.fixture
def curve_metadata(h5_file, volume_args):
    """Metadata for the dummy volume, with an empty metadata group."""
    shape = volume_args["shape"]
    segments = volume_args["segments"]
    
    group = h5_file.create_group("Measurement_000/CurveMetadata")
    return h5_jpk.CurvesH5Metadata(
        group, shape=shape, channel_units={"height": "m", "vDeflection": "V"}, segment_names=segments, flip_image=True
    )


@pytest.fixture
def dummy_curves_h5_volumes(volume_args):
    """Dummy instance of the CurvesH5Volumes class."""
    shape = volume_args["shape"]

    return h5_jpk.CurvesH5Volume(
        name="dummy_curves_h5_volumes",
        shape=shape,
        volume_data_group=volume_data_group,
        metadata=curve_metadata,
        flip_image=True
    )