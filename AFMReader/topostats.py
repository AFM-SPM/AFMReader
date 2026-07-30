"""For decoding and loading .topostats (HDF5 format) AFM file format into Python Nympy arrays."""

from pathlib import Path

import h5py

from packaging.version import parse as parse_version
from AFMReader.data_classes import AFMLoad
from AFMReader.io import unpack_hdf5
from AFMReader.logging import logger

logger.enable(__package__)


def load_topostats(file_path: Path | str, channel: str) -> AFMLoad:
    """
    Extract image and pixel to nm scaling from the .topostats (HDF5 format) file.

    Parameters
    ----------
    file_path : Path | str
        Path to the .topostats file.
    channel : str
        The channel to load.

    Returns
    -------
    AFMLoad
        An AFMLoad object containing the image, its pixel to nm scaling factor, and nested Numpy arrays representing the
        analyses performed on the data.

    Raises
    ------
    OSError
        If the file is not found.

    Examples
    --------
    >>> afm_load = load_topostats("path/to/topostats_file.topostats", channel="image")
    >>> image = afm_load.image
    >>> pixel_to_nm_scaling = afm_load.px2nm
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
    try:
        image = data.pop(channel)
        pixel_to_nanometre_scaling_factor = data.pop("pixel_to_nm_scaling")
    except KeyError as exc:
        image_keys = ["image", "image_original"]
        topostats_keys = list(data.keys())
        raise ValueError(
            f"'{channel}' not in available image keys: " f"{[im for im in image_keys if im in topostats_keys]}"
        ) from exc

    # Analyses are stored to metadata - this might be a bit clunky and potentially should be stored to their own attr
    return AFMLoad(image=image, pixel_to_nanometre_scaling=pixel_to_nanometre_scaling_factor, metadata=data)


def get_topostats_channels() -> list[str]:
    """
    Get the available channels for a .topostats file.

    Returns
    -------
    list[str]
        A list of available channels in the .topostats file.
    """
    return ["image", "image_original"]
