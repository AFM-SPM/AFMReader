"""Module to decode and load .bin AFM files into Python Numpy arrays."""

import math
from pathlib import Path

import numpy as np

from .logging import logger

# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,fixme

DTYPE_MAP = {
    "IEEE double": np.float64,
    "DBL": np.float64,
    "IEEE single": np.float32,
    "SGL": np.float32,
    "U32": np.uint32,
    "I32": np.int32,
    "U16": np.uint16,
    "I16": np.int16,
    "U8": np.uint8,
    "I8": np.int8,
    "float64": np.float64,
    "float32": np.float32,
    "int32": np.int32,
}


def load_bin(
    filepath: str | Path,
    data_type: str,
    offset_bytes: int,
    size_x: float | None = None,
    size_y: float | None = None,
    shape_x: int | None = None,
    shape_y: int | None = None,
    z_scaling: float = 1.0,
):
    """
    Load image from binary file. Parameters to interpret the binary file must be provided.

    Parameters
    ----------
    filepath : str | Path
        Path to the binary file.
    data_type : str
        Data type of the binary file.
    offset_bytes : int
        Number of bytes to skip at the beginning of the file.
    size_x : float, optional
        Size of the image in the x direction (default is None).
    size_y : float, optional
        Size of the image in the y direction (default is None).
    shape_x : int, optional
        Number of pixels in the x direction (default is None).
    shape_y : int, optional
        Number of pixels in the y direction (default is None).
    z_scaling : float, optional
        Scaling factor for the z values (default is 1.0).

    Returns
    -------
    image : np.ndarray
        2D array of shape (height, width) with image data.
    px2nm : float
        Scaling factor converting pixels to nanometers.
    """
    filepath = Path(filepath)
    dt_key = str(data_type).strip()
    shape_x = None if shape_x == 0 else shape_x
    shape_y = None if shape_y == 0 else shape_y

    if dt_key in DTYPE_MAP:
        np_dtype = DTYPE_MAP[dt_key]
    else:
        logger.warning(f"Unknown data type '{dt_key}'. Defaulting to float64.")
        np_dtype = np.float64
    with filepath.open("rb") as f:
        f.seek(offset_bytes)
        flat_data = np.fromfile(f, dtype=np_dtype)
    if None in [shape_x, shape_y]:
        dimension = int(math.sqrt(len(flat_data)))
        shape_x, shape_y = dimension, dimension
    assert shape_x is not None and shape_y is not None  # noqa: PT018
    assert size_x is not None and size_y is not None  # noqa: PT018
    if shape_x * shape_y != len(flat_data):
        logger.error(f"Loading binary file {filepath.stem} did not receive a shape and is not square")
    image = flat_data.reshape((shape_x, shape_y))
    image *= z_scaling
    pixel_to_nm_scaling_factor_x = size_x / shape_x if shape_x > 0 else 1.0
    pixel_to_nm_scaling_factor_y = size_y / shape_y if shape_y > 0 else 1.0
    px2nm = (pixel_to_nm_scaling_factor_x + pixel_to_nm_scaling_factor_y) / 2
    return image, px2nm


def get_bin_channels():
    """
    Get the list of channels available in the binary file.

    Since binary files do not have a standard structure,
    this function returns an empty list (as no standard channels are available) and the expected keyword
    arguments for loading a binary file.

    Returns
    -------
    list
        Empty list.
    dict
        Dictionary of expected keyword arguments for loading a binary file.
    """
    kwarg_types = {
        "data_type": (str, DTYPE_MAP.keys()),
        "offset_bytes": int,
        "size_x": float,
        "size_y": float,
        "shape_x": int,
        "shape_y": int,
        "z_scaling": float,
    }
    return [], kwarg_types
