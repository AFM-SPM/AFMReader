"""Test the reading and writing of data from / to files."""

from pathlib import Path

import numpy as np
import h5py


from AFMReader.io import unpack_hdf5


def test_unpack_hdf5_all_together_group_path_default(tmp_path: Path) -> None:
    """Test loading a nested dictionary with arrays from HDF5 format with group path as default."""
    to_save = {
        "a": 1,
        "b": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        "c": "test",
        "d": {"e": 1, "f": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), "g": "test"},
    }

    group_path = "/"

    # Manually save the dictionary to HDF5 format
    with h5py.File(tmp_path / "hdf5_file_nested_with_arrays_group_path_standard.hdf5", "w") as f:
        # Write the datasets and groups to the file without using the dict_to_hdf5 function
        f.create_dataset("a", data=to_save["a"])
        f.create_dataset("b", data=to_save["b"])
        f.create_dataset("c", data=to_save["c"])
        d = f.create_group("d")
        d.create_dataset("e", data=to_save["d"]["e"])
        d.create_dataset("f", data=to_save["d"]["f"])
        d.create_dataset("g", data=to_save["d"]["g"])

    # Load it back in and check if the dictionary is the same
    with h5py.File(tmp_path / "hdf5_file_nested_with_arrays_group_path_standard.hdf5", "r") as f:
        result = unpack_hdf5(open_hdf5_file=f, group_path=group_path)

    np.testing.assert_equal(result, to_save)


def test_unpack_hdf5_all_together_group_path_non_standard(tmp_path: Path) -> None:
    """Test loading a nested dictionary with arrays from HDF5 format with a non-standard group path."""
    to_save = {
        "a": 1,
        "b": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        "c": "test",
        "d": {"e": 1, "f": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), "g": "test"},
    }

    group_path = "/d/"

    expected = {
        "e": 1,
        "f": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        "g": "test",
    }

    # Manually save the dictionary to HDF5 format
    with h5py.File(tmp_path / "hdf5_file_all_together_group_path_nonstandard.hdf5", "w") as f:
        # Write the datasets and groups to the file without using the dict_to_hdf5 function
        f.create_dataset("a", data=to_save["a"])
        f.create_dataset("b", data=to_save["b"])
        f.create_dataset("c", data=to_save["c"])
        d = f.create_group("d")
        d.create_dataset("e", data=to_save["d"]["e"])
        d.create_dataset("f", data=to_save["d"]["f"])
        d.create_dataset("g", data=to_save["d"]["g"])

    # Load it back in and check if the dictionary is the same
    with h5py.File(tmp_path / "hdf5_file_all_together_group_path_nonstandard.hdf5", "r") as f:
        result = unpack_hdf5(open_hdf5_file=f, group_path=group_path)

    np.testing.assert_equal(result, expected)


def test_unpack_hdf5_int(tmp_path: Path) -> None:
    """Test loading a dictionary with an integer from HDF5 format."""
    to_save = {"a": 1, "b": 2}

    group_path = "/"

    # Manually save the dictionary to HDF5 format
    with h5py.File(tmp_path / "hdf5_file_int.hdf5", "w") as f:
        # Write the datasets and groups to the file without using the dict_to_hdf5 function
        f.create_dataset("a", data=to_save["a"])
        f.create_dataset("b", data=to_save["b"])

    # Load it back in and check if the dictionary is the same
    with h5py.File(tmp_path / "hdf5_file_int.hdf5", "r") as f:
        result = unpack_hdf5(open_hdf5_file=f, group_path=group_path)

    np.testing.assert_equal(result, to_save)


def test_unpack_hdf5_float(tmp_path: Path) -> None:
    """Test loading a dictionary with a float from HDF5 format."""
    to_save = {"a": 0.01, "b": 0.02}

    group_path = "/"

    # Manually save the dictionary to HDF5 format
    with h5py.File(tmp_path / "hdf5_file_float.hdf5", "w") as f:
        # Write the datasets and groups to the file without using the dict_to_hdf5 function
        f.create_dataset("a", data=to_save["a"])
        f.create_dataset("b", data=to_save["b"])

    # Load it back in and check if the dictionary is the same
    with h5py.File(tmp_path / "hdf5_file_float.hdf5", "r") as f:
        result = unpack_hdf5(open_hdf5_file=f, group_path=group_path)

    np.testing.assert_equal(result, to_save)


def test_unpack_hdf5_str(tmp_path: Path) -> None:
    """Test loading a dictionary with a string from HDF5 format."""
    to_save = {"a": "test", "b": "test2"}

    group_path = "/"

    # Manually save the dictionary to HDF5 format
    with h5py.File(tmp_path / "hdf5_file_str.hdf5", "w") as f:
        # Write the datasets and groups to the file without using the dict_to_hdf5 function
        f.create_dataset("a", data=to_save["a"])
        f.create_dataset("b", data=to_save["b"])

    # Load it back in and check if the dictionary is the same
    with h5py.File(tmp_path / "hdf5_file_str.hdf5", "r") as f:
        result = unpack_hdf5(open_hdf5_file=f, group_path=group_path)

    np.testing.assert_equal(result, to_save)


def test_unpack_hdf5_dict_nested_dict(tmp_path: Path) -> None:
    """Test loading a nested dictionary from HDF5 format."""
    to_save = {
        "a": 1,
        "b": 2,
        "c": {"d": 3, "e": 4},
    }

    group_path = "/"

    # Manually save the dictionary to HDF5 format
    with h5py.File(tmp_path / "hdf5_file_nested_dict.hdf5", "w") as f:
        # Write the datasets and groups to the file without using the dict_to_hdf5 function
        f.create_dataset("a", data=to_save["a"])
        f.create_dataset("b", data=to_save["b"])
        c = f.create_group("c")
        c.create_dataset("d", data=to_save["c"]["d"])
        c.create_dataset("e", data=to_save["c"]["e"])

    # Load it back in and check if the dictionary is the same
    with h5py.File(tmp_path / "hdf5_file_nested_dict.hdf5", "r") as f:
        result = unpack_hdf5(open_hdf5_file=f, group_path=group_path)

    np.testing.assert_equal(result, to_save)


def test_unpack_hdf5_nested_dict_group_path(tmp_path: Path) -> None:
    """Test loading a nested dictionary from HDF5 format with a non-standard group path."""
    to_save = {
        "a": 1,
        "b": 2,
        "c": {"d": 3, "e": 4},
    }

    group_path = "/c/"

    expected = {
        "d": 3,
        "e": 4,
    }

    # Manually save the dictionary to HDF5 format
    with h5py.File(tmp_path / "hdf5_file_nested_dict_group_path.hdf5", "w") as f:
        # Write the datasets and groups to the file without using the dict_to_hdf5 function
        f.create_dataset("a", data=to_save["a"])
        f.create_dataset("b", data=to_save["b"])
        c = f.create_group("c")
        c.create_dataset("d", data=to_save["c"]["d"])
        c.create_dataset("e", data=to_save["c"]["e"])

    # Load it back in and check if the dictionary is the same
    with h5py.File(tmp_path / "hdf5_file_nested_dict_group_path.hdf5", "r") as f:
        result = unpack_hdf5(open_hdf5_file=f, group_path=group_path)

    np.testing.assert_equal(result, expected)
