"""For decoding and loading .gwy AFM file format into Python Numpy arrays."""

from __future__ import annotations
import io
from pathlib import Path
import re


from loguru import logger
import numpy as np

from AFMReader.io import read_uint32, read_null_terminated_string, read_char, read_double


def load_gwy(file_path: Path | str, channel: str):
    """
    Extract image and pixel to nm scaling from the .gwy file.

    Parameters
    ----------
    file_path : Path or str
        Path to the .gwy file.
    channel : str
        Channel name to extract from the .gwy file.

    Returns
    -------
    tuple(np.ndarray, float)
        A tuple containing the image and its pixel to nanometre scaling value.

    Raises
    ------
    FileNotFoundError
        If the file is not found.
    ValueError
        If the channel is not found in the .gwy file.

    Examples
    --------
    Load the image and pixel to nanometre scaling factor, available channels are 'Height', 'ZSensor' and 'Height
    Sensor'.

    >>> from AFMReader.gwy import load_gwy
    >>> image, pixel_to_nm = load_gwy(file_path="path/to/file.gwy", channel="Height")
    ```
    """
    logger.info(f"Loading image from : {file_path}")
    filename = file_path.stem
    try:
        image_data_dict = {}
        with Path.open(file_path, "rb") as open_file:  # pylint: disable=unspecified-encoding
            # Read header
            header = open_file.read(4)
            logger.debug(f"Gwy file header: {header}")

            gwy_read_object(open_file, data_dict=image_data_dict)

        # For development - uncomment to have an indentation based nested
        # dictionary output showing the object - component structure and
        # available keys:
        # LoadScans._gwy_print_dict_wrapper(gwy_file_dict=image_data_dict)

        channel_ids = gwy_get_channels(gwy_file_structure=image_data_dict)

        if channel not in channel_ids:
            raise KeyError(f"Channel {channel} not found in {file_path.suffix} channel list: {channel_ids}")

        # Get the image data
        image = image_data_dict[f"/{channel_ids[channel]}/data"]["data"]
        units = image_data_dict[f"/{channel_ids[channel]}/data"]["si_unit_xy"]["unitstr"]
        # currently only support equal pixel sizes in x and y
        px_to_nm = image_data_dict[f"/{channel_ids[channel]}/data"]["xreal"] / image.shape[1]

        # Convert image heights to nanometresQ
        if units == "m":
            image = image * 1e9
            px_to_nm = px_to_nm * 1e9
        else:
            raise ValueError(
                f"Units '{units}' have not been added for .gwy files. Please add \
                an SI to nanometre conversion factor for these units in _gwy_read_component in \
                io.py."
            )

    except FileNotFoundError:
        logger.info(f"[{filename}] File not found : {file_path}")
        raise

    return (image, px_to_nm)


def gwy_read_object(open_file: io.TextIOWrapper, data_dict: dict) -> None:
    """
    Parse and extract data from a `.gwy` file object, starting at the current open file read position.

    Parameters
    ----------
    open_file : io.TextIOWrapper
        An open file object.
    data_dict : dict
        Dictionary of `.gwy` file image properties.
    """
    object_name = read_null_terminated_string(open_file=open_file)
    data_size = read_uint32(open_file)
    logger.debug(f"OBJECT | name: {object_name} | data_size: {data_size}")
    # Read components
    read_data_size = 0
    while read_data_size < data_size:
        component_data_size = gwy_read_component(
            open_file=open_file,
            initial_byte_pos=open_file.tell(),
            data_dict=data_dict,
        )
        read_data_size += component_data_size


def gwy_read_component(open_file: io.TextIOWrapper, initial_byte_pos: int, data_dict: dict) -> int:
    """
    Parse and extract data from a `.gwy` file object, starting at the current open file read position.

    Parameters
    ----------
    open_file : io.TextIOWrapper
        An open file object.
    initial_byte_pos : int
        Initial position, as byte.
    data_dict : dict
        Dictionary of `.gwy` file image properties.

    Returns
    -------
    int
        Size of the component in bytes.
    """
    component_name = read_null_terminated_string(open_file=open_file)
    data_type = read_gwy_component_dtype(open_file=open_file)

    if data_type == "o":
        logger.debug(f"component name: {component_name} | dtype: {data_type} |")
        sub_dict = {}
        gwy_read_object(open_file=open_file, data_dict=sub_dict)
        data_dict[component_name] = sub_dict
    elif data_type == "c":
        value = read_char(open_file=open_file)
        logger.debug(f"component name: {component_name} | dtype: {data_type} | value: {value}")
        data_dict[component_name] = value
    elif data_type == "i":
        value = read_uint32(open_file=open_file)
        logger.debug(f"component name: {component_name} | dtype: {data_type} | value: {value}")
        data_dict[component_name] = value
    elif data_type == "d":
        value = read_double(open_file=open_file)
        logger.debug(f"component name: {component_name} | dtype: {data_type} | value: {value}")
        data_dict[component_name] = value
    elif data_type == "s":
        value = read_null_terminated_string(open_file=open_file)
        logger.debug(f"component name: {component_name} | dtype: {data_type} | value: {value}")
        data_dict[component_name] = value
    elif data_type == "D":
        array_size = read_uint32(open_file=open_file)
        logger.debug(f"component name: {component_name} | dtype: {data_type}")
        logger.debug(f"array size: {array_size}")
        data = np.zeros(array_size)
        for index in range(array_size):
            data[index] = read_double(open_file=open_file)
        if "xres" in data_dict and "yres" in data_dict:
            data = data.reshape((data_dict["xres"], data_dict["yres"]))
        data_dict["data"] = data

    return open_file.tell() - initial_byte_pos


def gwy_get_channels(gwy_file_structure: dict) -> dict:
    """
    Extract a list of channels and their corresponding dictionary key ids from the `.gwy` file dictionary.

    Parameters
    ----------
    gwy_file_structure : dict
        Dictionary of the nested object / component structure of a `.gwy` file. Where the keys are object names
        and the values are dictionaries of the object's components.

    Returns
    -------
    dict
        Dictionary where the keys are the channel names and the values are the dictionary key ids.

    Examples
    --------
    # Using a loaded dictionary generated from a `.gwy` file:
    LoadScans._gwy_get_channels(gwy_file_structure=loaded_gwy_file_dictionary)
    """
    title_key_pattern = re.compile(r"\d+(?=/data/title)")
    channel_ids = {}

    for key, _ in gwy_file_structure.items():
        match = re.search(title_key_pattern, key)
        if match:
            channel = gwy_file_structure[key]
            channel_ids[channel] = match.group()

    return channel_ids


def read_gwy_component_dtype(open_file: io.TextIOWrapper) -> str:
    """
    Read the data type of a `.gwy` file component.

    Possible data types are as follows:

    - 'b': boolean
    - 'c': character
    - 'i': 32-bit integer
    - 'q': 64-bit integer
    - 'd': double
    - 's': string
    - 'o': `.gwy` format object

    Capitalised versions of some of these data types represent arrays of values of that data type. Arrays are stored as
    an unsigned 32 bit integer, describing the size of the array, followed by the unseparated array values:

    - 'C': array of characters
    - 'I': array of 32-bit integers
    - 'Q': array of 64-bit integers
    - 'D': array of doubles
    - 'S': array of strings
    - 'O': array of objects.

    Parameters
    ----------
    open_file : io.TextIOWrapper
        An open file object.

    Returns
    -------
    str
        Python string (one character long) of the data type of the component's value.
    """
    return open_file.read(1).decode("ascii")
