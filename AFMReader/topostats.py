"""For decoding and loading .topostats (HDF5 format) AFM file format into Python Nympy arrays."""

from pathlib import Path
from typing import Any

import h5py

from packaging.version import parse as parse_version
from AFMReader.io import unpack_hdf5
from AFMReader.logging import logger

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
            # Handle different names for variables holding the file version (<=0.3) or the newer topostats version
            version = (
                data["topostats_file_version"]
                if "topostats_file_version" in data.keys()  # pylint: disable=consider-iterating-dictionary
                else data["topostats_version"]
            )
            if parse_version(str(version)) > parse_version("0.2"):
                data["img_path"] = Path(data["img_path"])
            logger.info(f"[{filename}] TopoStats file version : {version}")

    except OSError as e:
        if "Unable to open file" in str(e):
            logger.error(f"[{filename}] File not found : {file_path}")
        raise e

    logger.info(f"[{filename}] : Extracted .topostats dictionary.")
    return data
