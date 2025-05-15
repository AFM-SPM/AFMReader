"""For decoding and loading .top AFM file format into Python Numpy arrays."""

from pathlib import Path
import re

import numpy as np

from AFMReader.logging import logger
from AFMReader.io import read_int16

logger.enable(__package__)


# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=too-many-branches
def load_top(  # noqa: C901 (ignore too complex)
    file_path: Path | str, header_encoding: str = "latin-1"
) -> tuple[np.ndarray, float]:
    """
    Load image from TOP files.

    Parameters
    ----------
    file_path : Path | str
        Path to the .top file.
    header_encoding : str
        Encoding to use for the header of the file. Default is ''latin-1''.

    Returns
    -------
    tuple[np.ndarray, float]
        A tuple containing the image and its pixel to nanometre scaling value.

    Raises
    ------
    FileNotFoundError
        If the file is not found.
    ValueError
        If any of the required information are not found in the header.
    NotImplementedError
        If the image is non-square.
    """
    logger.info(f"Loading image from : {file_path}")
    file_path = Path(file_path)
    filename = file_path.stem
    try:
        with Path.open(file_path, "rb") as open_file:  # pylint: disable=unspecified-encoding
            # grab the beggining message, assume that it's within the first 150 bytes
            beginning_message = str(open_file.read(150))
            # find the header size in the beginning message
            header_size_match = re.search(r"Image header size: (\d+)", beginning_message)
            if header_size_match is None:
                raise ValueError(f"[{filename}] : 'Image header size' not found in image raw bytes.")
            header_size = int(header_size_match.group(1))

            # Return to start of file
            open_file.seek(0)
            # Read the header
            header = open_file.read(header_size)

            # decode the header bytes
            header_decoded = header.decode(header_encoding)

            # find num rows
            rows_match = re.search(r"Number of rows: (\d+)", header_decoded)
            if rows_match is None:
                raise ValueError(f"[{filename}] : 'rows' not found in file header.")
            rows = int(rows_match.group(1))
            cols_match = re.search(r"Number of columns: (\d+)", header_decoded)
            if cols_match is None:
                raise ValueError(f"[{filename}] : 'cols' not found in file header.")
            cols = int(cols_match.group(1))
            x_real_size_match = re.search(r"X Amplitude: (\d+\.?\d*) (µm|nm)", header_decoded)
            if x_real_size_match is None:
                raise ValueError(f"[{filename}] : 'X Amplitude' not found in file header.")
            x_real_size = float(x_real_size_match.group(1))
            x_units = x_real_size_match.group(2)
            x_real_size = x_real_size * 1000 if x_units == "µm" else x_real_size
            y_real_size_match = re.search(r"Y Amplitude: (\d+\.?\d*) (µm|nm)", header_decoded)
            if y_real_size_match is None:
                raise ValueError(f"[{filename}] : 'Y Amplitude' not found in file header.")
            y_real_size = float(y_real_size_match.group(1))
            y_units = y_real_size_match.group(2)
            y_real_size = y_real_size * 1000 if y_units == "µm" else y_real_size
            if x_real_size != y_real_size:
                raise NotImplementedError(
                    f"[{filename}] : X scan size (nm) does not equal Y scan size (nm) ({x_real_size}, {y_real_size})"
                    "we don't currently support non-square images."
                )
            Zrange_match = re.search(r"Z Amplitude: (\d+)", header_decoded)
            if Zrange_match is None:
                raise ValueError(f"[{filename}] : 'Z Amplitude' not found in file header.")
            Zrange = float(Zrange_match.group(1))

            # Calculate pixel to nm scaling
            pixel_to_nm_scaling = x_real_size / cols

            # Read R x C matrix of int16s
            image_list = []
            for _ in range(rows):
                row = []
                for _ in range(cols):
                    row.append(read_int16(open_file))
                image_list.append(row)
            image = np.array(image_list)
            image_int = np.array(image_list)
            Zmin = 0  # nm
            norm_image = (image_int - np.min(image_int)) / (np.max(image_int) - np.min(image_int))
            image = norm_image * (Zrange - Zmin) + Zmin

    except FileNotFoundError as e:
        logger.error(f"[{filename}] : File not found : {file_path}")
        raise e
    except Exception as e:
        logger.error(f"[{filename}] : {e}")
        raise e

    logger.info(f"[{filename}] : Extracted image.")
    return (image, pixel_to_nm_scaling)
