"""For decoding and loading .topostats (HDF5 format) AFM file format into Python Nympy arrays."""

from __future__ import annotations
from pathlib import Path
from typing import Any

import h5py

from AFMReader.logging import logger
from AFMReader.io import unpack_hdf5

logger.enable(__package__)


def load_topostats(file_path: Path | str) -> dict[str, Any]:
    """
    Extract image and pixel to nm scaling from the .topostats (HDF5 format) file.

    Parameters
    ----------
    file_path : Path or str
        Path to the .topostats file.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the image, its pixel to nm scaling factor and nested Numpy arrays representing the
        analyses performed on the data.

    Raises
    ------
    OSError
        If the file is not found.

    Examples
    --------
    >>> image, pixel_to_nm_scaling = load_topostats("path/to/topostats_file.topostats")
    """
    logger.info(f"Loading image from : {file_path}")
    file_path = Path(file_path)
    filename = file_path.stem
    try:
        with h5py.File(file_path, "r") as f:
            data = unpack_hdf5(open_hdf5_file=f, group_path="/")
            if str(data["topostats_file_version"]) >= "0.2":
                data["img_path"] = Path(data["img_path"])
            file_version = data["topostats_file_version"]
            logger.info(f"[{filename}] TopoStats file version : {file_version}")

    except OSError as e:
        if "Unable to open file" in str(e):
            logger.error(f"[{filename}] File not found : {file_path}")
        raise e

    logger.info(f"[{filename}] : Extracted image.")
    return data
