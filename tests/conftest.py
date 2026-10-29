"""Fixtures for testing."""

from pathlib import Path

import h5py
import numpy as np
import pySPM
import pytest

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"


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

SHAPE = (2, 3)
SEGMENTS = ["extend", "retract"]
@pytest.fixture
def curve_meta_group():
    """Curve metadata group whose per-curve and per-segment values equal their flat index."""
    with h5py.File("in_memory.h5", "w", driver="core", backing_store=False) as f:
        group = f.create_group("curve_metadata")
        n_curves = SHAPE[0] * SHAPE[1]
        group.create_dataset("curve.index", data=np.arange(n_curves))
        group.create_dataset("segment.index", data=np.arange(n_curves * len(SEGMENTS)))
        group.attrs["curve.instrument"] = "JPK"  # constant value, stored as an attribute
        yield group