"""
Module to decode and load .h5-jpk AFM file format into Python NumPy arrays.

These files can contain multiple images that make up a video; here, single frames are read.
"""

from pathlib import Path

import numpy as np
import h5py

from AFMReader.logging import logger

logger.enable(__package__)


def _jpk_pixel_to_nm_scaling_h5(measurement_group: h5py.Group) -> float:
    """
    Extract pixel-to-nanometre scaling from an HDF5 JPK measurement group.

    This uses the fast scan axis (u/i) and converts the physical scan size to nanometres
    per pixel based on the scan length and pixel count.

    Parameters
    ----------
    measurement_group : h5py.Group
        HDF5 group corresponding to a Measurement (e.g. '/Measurement_000').

    Returns
    -------
    float
        Real-world size of a single pixel in nanometres.

    Raises
    ------
    KeyError
        If required attributes are missing in the measurement group.
    """
    try:
        ulength = measurement_group.attrs["position-pattern.grid.ulength"]  # physical length in meters
        ilength = measurement_group.attrs["position-pattern.grid.ilength"]  # number of pixels

        if ilength == 0:
            raise ValueError("Pixel count (ilength) is zero; cannot compute scaling.")

        return (ulength / ilength) * 1e9

    except KeyError as e:
        missing = e.args[0]
        raise KeyError(f"Missing required attribute '{missing}' in HDF5 measurement group.") from e


def _get_z_scaling_h5(channel_group: h5py.Group) -> tuple[float, float]:
    """
    Extract the Z scaling multiplier and offset from an HDF5 channel group.

    Parameters
    ----------
    channel_group : h5py.Group
        The HDF5 group corresponding to a specific channel (e.g. /Measurement_000/Channel_001).

    Returns
    -------
    tuple[float, float]
        A tuple containing the scaling multiplier and offset.

    Notes
    -----
    Defaults to (1.0, 0.0) if attributes are not present.
    """
    multiplier = float(channel_group.attrs.get("net-encoder.scaling.multiplier", 1.0))
    offset = float(channel_group.attrs.get("net-encoder.scaling.offset", 0.0))

    logger.debug(f"Z-scaling: multiplier = {multiplier}, offset = {offset}")
    return multiplier, offset


def _decode_attr(attr: bytes | str) -> str:
    """Decode an HDF5 attribute that might be bytes or str."""
    if isinstance(attr, bytes):
        return attr.decode("utf-8")
    return str(attr)


def _attr_to_bool(attr: bytes | str | bool | int | float) -> bool:
    """Convert an HDF5 attribute to bool, handling bytes, str, bool, int, float types."""
    if isinstance(attr, bytes | str):
        return _decode_attr(attr).strip().lower() == "true"
    return bool(attr)


def load_h5jpk(
    file_path: Path | str, channel: str, flip_image: bool = True, frame: int = 0
) -> tuple[np.ndarray, float]:
    """
    Load image from JPK Instruments .h5-jpk files.

    Parameters
    ----------
    file_path : Path | str
        Path to the .h5-jpk file.
    channel : str
        The channel to extract from the .h5-jpk file.
    flip_image : bool, optional
        Whether to flip the image vertically. Default is ``True``.
    frame : int, optional
        Which frame of the video to open. Default is 0 (first frame).


    Returns
    -------
    tuple[np.ndarray, float]
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

    >>> from AFMReader.jpk import load_h5jpk
    >>> image, pixel_to_nanometre_scaling_factor = load_h5jpk(file_path="./my_jpk_file.jpk",
    >>>                                                     channel="height_trace",
    >>>                                                     frame = 5,
    >>>                                                     flip_image=True)
    """
    logger.info(f"Loading H5-JPK file from : {file_path}")
    file_path = Path(file_path)
    filename = file_path.stem

    # Validate and process channel input
    if "_" not in channel:
        raise ValueError(f"Invalid channel format '{channel}'. Expected format: 'channel_trace' or 'channel_retrace'.")

    channel_type, trace_type = channel.rsplit("_", 1)
    trace_type = trace_type.lower()

    if trace_type not in ("trace", "retrace"):
        raise ValueError(f"Invalid trace type '{trace_type}'. Must be 'trace' or 'retrace'.")

    # Load HDF5 file
    with h5py.File(file_path, "r") as f:
        logger.debug(f"Opened HDF5 file structure: {list(f.keys())}")

        # Search for measurement groups and identify channels
        channel_list = {}
        for measurement_key, measurement_group in f.items():
            if not measurement_key.startswith("Measurement_"):
                continue
            for channel_key in measurement_group.keys():
                if not channel_key.startswith("Channel_"):
                    continue
                current_group = measurement_group[channel_key]

                # Check for required attributes
                if "channel.name" in current_group.attrs:
                    available_channel = _decode_attr(current_group.attrs["channel.name"])
                else:
                    continue  # skip channels with no name

                # Get retrace attribute safely
                retrace = _attr_to_bool(current_group.attrs.get("retrace", False))

                # Determine trace/retrace
                if retrace is True:
                    tr_rt = "retrace"
                else:
                    tr_rt = "trace"

                # Build channel key
                available_channel = available_channel.strip().lower()
                full_key = f"{available_channel}_{tr_rt}"
                full_path = f"{measurement_key}/{channel_key}"
                if full_key not in channel_list:
                    channel_list[full_key] = full_path

        if channel not in channel_list:
            logger.error(f"'{channel}' not in available channels: {channel_list}")
            raise ValueError(f"'{channel}' not found. Available channels: {list(channel_list)}")

        channel_path = channel_list[channel]
        channel_group = f[channel_path]
        measurement_group = f[channel_path.split("/")[0]]
        dataset_name = channel.split("_")[0].capitalize()  # "height_trace" -> "Height"

        # Load images and scaling factors from channel dataset
        images = channel_group[dataset_name][:]
        scaling, offset = _get_z_scaling_h5(channel_group)
        images = (images * scaling) + offset

        # Select and reshape a flattened frame
        image_size = measurement_group.attrs["position-pattern.grid.ilength"]   # number of pixels
        if frame < 0 or frame >= images.shape[1]:
            raise IndexError(f"Frame index {frame} out of range. Must be between 0 and {images.shape[1]-1}.")

        image = images[:, frame].reshape((image_size, image_size))

        # Flip image
        if flip_image:
            image = np.flipud(image)

        #Convert to nm
        if dataset_name.lower() in ("height", "measuredheight", "amplitude"):
            image = image * 1e9

        logger.info(f"[{filename}] : Extracted image.")
        return (image, _jpk_pixel_to_nm_scaling_h5(measurement_group))
