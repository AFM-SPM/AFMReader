"""For decoding and loading .spm AFM file format into Python Numpy arrays."""

from __future__ import annotations
from pathlib import Path

import pySPM
import numpy as np

from AFMReader.logging import logger

logger.enable(__package__)


def spm_pixel_to_nm_scaling(filename: str, channel_data: pySPM.SPM.SPM_image) -> float:
    """
    Extract pixel to nm scaling from the SPM image metadata.

    Parameters
    ----------
    filename : str
        File name.
    channel_data : pySPM.SPM.SPM_image
        Channel data from PySPM.

    Returns
    -------
    float
        Pixel to nm scaling factor.
    """
    unit_dict = {
        "nm": 1,
        "um": 1e3,
    }
    px_to_real = channel_data.pxs()
    # Has potential for non-square pixels but not yet implimented
    pixel_to_nm_scaling = (
        px_to_real[0][0] * unit_dict[px_to_real[0][1]],
        px_to_real[1][0] * unit_dict[px_to_real[1][1]],
    )[0]
    if px_to_real[0][0] == 0 and px_to_real[1][0] == 0:
        pixel_to_nm_scaling = 1
        logger.info(f"[{filename}] : Pixel size not found in metadata, defaulting to 1nm")
    logger.info(f"[{filename}] : Pixel to nm scaling : {pixel_to_nm_scaling}")
    return pixel_to_nm_scaling


def load_spm(file_path: Path | str, channel: str) -> tuple:
    """
    Extract image and pixel to nm scaling from the Bruker .spm file.

    Parameters
    ----------
    file_path : Path or str
        Path to the .spm file.
    channel : str
        Channel name to extract from the .spm file.

    Returns
    -------
    tuple(np.ndarray, float)
        A tuple containing the image and its pixel to nanometre scaling value.

    Raises
    ------
    FileNotFoundError
        If the file is not found.
    ValueError
        If the channel is not found in the .spm file.

    Examples
    --------
    Load the image and pixel to nanometre scaling factor, available channels are 'Height', 'ZSensor' and 'Height
    Sensor'.

    >>> from AFMReader.spm import load_spm
    >>> image, pixel_to_nm = load_spm(file_path="path/to/file.spm", channel="Height")
    ```
    """
    logger.info(f"Loading image from : {file_path}")
    file_path = Path(file_path)
    filename = file_path.stem
    try:
        scan = pySPM.Bruker(file_path)
        logger.info(f"[{filename}] : Loaded image from : {file_path}")
        channel_data = scan.get_channel(channel)
        logger.info(f"[{filename}] : Extracted channel {channel}")
        image = np.flipud(np.array(channel_data.pixels))
    except FileNotFoundError:
        logger.error(f"[{filename}] File not found : {file_path}")
        raise
    except Exception as e:
        if "Channel" in str(e) and "not found" in str(e):
            # trying to return the error with options of possible channel values
            labels = []
            for channel_option in [layer[b"@2:Image Data"][0] for layer in scan.layers]:
                channel_name = channel_option.decode("latin1").split('"')[1][1:-1]
                labels.append(channel_name)
            logger.error(f"[{filename}] : {channel} not in {file_path.suffix} channel list: {labels}")
            raise ValueError(f"{channel} not in {file_path.suffix} channel list: {labels}") from e
        raise e

    return (image, spm_pixel_to_nm_scaling(filename, channel_data))
