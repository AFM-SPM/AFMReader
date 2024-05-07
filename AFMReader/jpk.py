"""For decoding and loading .jpk AFM file format into Python Numpy arrays."""

from __future__ import annotations
from pathlib import Path

import numpy as np
import tifffile

from AFMReader.logging import logger

logger.enable(__package__)


def _jpk_pixel_to_nm_scaling(tiff_page: tifffile.tifffile.TiffPage) -> float:
    """
    Extract pixel to nm scaling from the JPK image metadata.

    Parameters
    ----------
    tiff_page : tifffile.tifffile.TiffPage
        An image file directory (IFD) of .jpk files.

    Returns
    -------
    float
        A value corresponding to the real length of a single pixel.
    """
    length = tiff_page.tags["32834"].value  # Grid-uLength (fast)
    width = tiff_page.tags["32835"].value  # Grid-vLength (slow)
    length_px = tiff_page.tags["32838"].value  # Grid-iLength (fast)
    width_px = tiff_page.tags["32839"].value  # Grid-jLength (slow)

    px_to_nm = (length / length_px, width / width_px)[0]

    return px_to_nm * 1e9


# pylint: disable=too-many-locals
def load_jpk(file_path: Path | str, channel: str) -> tuple[np.ndarray, float]:
    """
    Load image from JPK Instruments .jpk files.

    Parameters
    ----------
    file_path : Path | str
        Path to the .jpk file.
    channel : str
        The channel to extract from the .jpk file.

    Returns
    -------
    tuple[npt.NDArray, float]
        A tuple containing the image and its pixel to nanometre scaling value.

    Raises
    ------
    FileNotFoundError
        If the file is not found.
    KeyError
        If the channel is not found in the file.

    Examples
    --------
    Load height trace channel from the .jpk file. 'height_trace' is the default channel name.

    >>> from AFMReader.jpk import load_jpk
    >>> image, pixel_to_nanometre_scaling_factor = load_jpk(file_path="./my_jpk_file.jpk", channel="height_trace")
    """
    logger.info(f"Loading image from : {file_path}")
    file_path = Path(file_path)
    filename = file_path.stem
    try:
        tif = tifffile.TiffFile(file_path)
    except FileNotFoundError:
        logger.error(f"[{filename}] File not found : {file_path}")
        raise
    # Obtain channel list for all channels in file
    channel_list = {}
    for i, page in enumerate(tif.pages[1:]):  # [0] is thumbnail
        available_channel = page.tags["32848"].value  # keys are hexadecimal values
        if page.tags["32849"].value == 0:  # whether img is trace or retrace
            tr_rt = "trace"
        else:
            tr_rt = "retrace"
        channel_list[f"{available_channel}_{tr_rt}"] = i + 1
    try:
        channel_idx = channel_list[channel]
    except KeyError:
        logger.error(f"{channel} not in channel list: {channel_list}")
        raise

    # Get image and if applicable, scale it
    channel_page = tif.pages[channel_idx]
    image = channel_page.asarray()
    scaling_type = channel_page.tags["33027"].value
    if scaling_type == "LinearScaling":
        scaling = channel_page.tags["33028"].value
        offset = channel_page.tags["33029"].value
        image = (image * scaling) + offset
    elif scaling_type == "NullScaling":
        pass
    else:
        raise ValueError(f"Scaling type {scaling_type} is not 'NullScaling' or 'LinearScaling'")
    # Get page for common metadata between scans
    metadata_page = tif.pages[0]
    return (image * 1e9, _jpk_pixel_to_nm_scaling(metadata_page))
