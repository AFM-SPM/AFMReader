"""For decoding and loading .ibw AFM file format into Python Numpy arrays."""

from __future__ import annotations
from pathlib import Path

import numpy as np
from igor2 import binarywave

from AFMReader.logging import logger

logger.enable(__package__)


def _ibw_pixel_to_nm_scaling(scan: dict) -> float:
    """
    Extract pixel to nm scaling from the IBW image metadata.

    Parameters
    ----------
    scan : dict
        The loaded binary wave object.

    Returns
    -------
    float
        A value corresponding to the real length of a single pixel.
    """
    # Get metadata
    notes = {}
    for line in str(scan["wave"]["note"]).split("\\r"):
        if line.count(":"):
            key, val = line.split(":", 1)
            notes[key] = val.strip()
    # Has potential for non-square pixels but not yet implemented
    return (
        float(notes["SlowScanSize"]) / scan["wave"]["wData"].shape[0] * 1e9,  # as in m
        float(notes["FastScanSize"]) / scan["wave"]["wData"].shape[1] * 1e9,  # as in m
    )[0]


def load_ibw(file_path: Path | str, channel: str) -> tuple[np.ndarray, float]:
    """
    Load image from Asylum Research (Igor) .ibw files.

    Parameters
    ----------
    file_path : Path | str
        Path to the .ibw file.
    channel : str
        The channel to extract from the .ibw file.

    Returns
    -------
    tuple[np.ndarray, float]
        A tuple containing the image and its pixel to nanometre scaling value.

    Raises
    ------
    FileNotFoundError
        If the file is not found.
    ValueError
        If the channel is not found in the .ibw file.

    Examples
    --------
    Load the image and pixel to nanometre scaling factor - 'HeightTracee' is the default channel name (the extra 'e' is
    not a typo!).

    >>> from AFMReader.ibw import load_ibw
    >>> image, pixel_to_nanometre_scaling_factor = load_ibw(file_path="./my_ibw_file.ibw", channel="HeightTracee")
    """
    logger.info(f"Loading image from : {file_path}")
    file_path = Path(file_path)
    filename = file_path.stem
    try:
        scan = binarywave.load(file_path)
        logger.info(f"[{filename}] : Loaded image from : {file_path}")
        labels = []
        for label_list in scan["wave"]["labels"]:
            for label in label_list:
                if label:
                    labels.append(label.decode())
        channel_idx = labels.index(channel)
        image = scan["wave"]["wData"][:, :, channel_idx].T * 1e9  # Looks to be in m
        image = np.flipud(image)
        logger.info(f"[{filename}] : Extracted channel {channel}")
    except FileNotFoundError:
        logger.error(f"[{filename}] File not found : {file_path}")
    except ValueError:
        logger.error(f"[{filename}] : {channel} not in {file_path.suffix} channel list: {labels}")
        raise
    except Exception as e:
        logger.error(f"[{filename}] : {e}")
        raise e

    return (image, _ibw_pixel_to_nm_scaling(scan))
