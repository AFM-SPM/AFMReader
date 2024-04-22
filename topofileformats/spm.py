"""For decoding and loading .spm AFM file format into Python Numpy arrays."""

from pathlib import Path

import pySPM
import numpy as np


def spm_pixel_to_nm_scaling(filename: str, channel_data: pySPM.SPM.SPM_image) -> float:
    """Extract pixel to nm scaling from the SPM image metadata.

    Parameters
    ----------
    channel_data: pySPM.SPM.SPM_image
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
        print(f"[{filename}] : Pixel size not found in metadata, defaulting to 1nm")
    print(f"[{filename}] : Pixel to nm scaling : {pixel_to_nm_scaling}")
    return pixel_to_nm_scaling


def load_spm(file_path: Path | str, channel: str) -> tuple:
    """Extract image and pixel to nm scaling from the Bruker .spm file.

    Returns
    -------
    tuple(np.ndarray, float)
        A tuple containing the image and its pixel to nanometre scaling value.
    """
    print(f"Loading image from : {file_path}")
    file_path = Path(file_path)
    filename = file_path.stem
    try:
        scan = pySPM.Bruker(file_path)
        print(f"[{filename}] : Loaded image from : {file_path}")
        channel_data = scan.get_channel(channel)
        print(f"[{filename}] : Extracted channel {channel}")
        image = np.flipud(np.array(channel_data.pixels))
    except FileNotFoundError:
        print(f"[{filename}] File not found : {file_path}")
        raise
    except Exception as e:
        if "Channel" in str(e) and "not found" in str(e):
            # trying to return the error with options of possible channel values
            labels = []
            for channel_option in [layer[b"@2:Image Data"][0] for layer in scan.layers]:
                channel_name = channel_option.decode("latin1").split(" ")[1][1:-1]
                # channel_description = channel.decode('latin1').split('"')[1] # incase the blank field raises quesions?
                labels.append(channel_name)
            print(f"[{filename}] : {channel} not in {file_path.suffix} channel list: {labels}")
            raise ValueError(f"{channel} not in {file_path.suffix} channel list: {labels}") from e
        raise e

    return (image, spm_pixel_to_nm_scaling(filename, channel_data))
