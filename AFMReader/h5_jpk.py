"""
Module to decode and load .h5-jpk AFM file format into 3D Python NumPy arrays.

It extracts scan channels, reshapes image frames, applies scaling, and generates
timestamps based on scan metadata.
"""

from pathlib import Path

import h5py
import numpy as np

from AFMReader.logging import logger

logger.enable(__package__)


def _parse_channel_name(channel: str) -> tuple[str, str]:
    """
    Validate and split a channel name into its type and trace direction.

    Parameters
    ----------
    channel : str
        The name of the channel, expected in the form 'name_trace' or 'name_retrace'.

    Returns
    -------
    tuple[str, str]
        A tuple containing the channel type and trace type.

    Raises
    ------
    ValueError
        If the format is invalid or the trace type is not 'trace' or 'retrace'.
    """
    if "_" not in channel:
        raise ValueError(f"Invalid channel format '{channel}'. Expected 'name_trace' or 'name_retrace'.")

    channel_type, trace_type = channel.rsplit("_", 1)
    trace_type = trace_type.lower()

    if trace_type not in ("trace", "retrace"):
        raise ValueError(f"Invalid trace type '{trace_type}'. Must be 'trace' or 'retrace'.")

    return channel_type, trace_type


def _get_channel_info(f: h5py.File, channel: str):
    """
    Retrieve channel-related HDF5 groups and dataset name.

    Parameters
    ----------
    f : h5py.File
        The open HDF5 file object.
    channel : str
        The name of the channel to retrieve.

    Returns
    -------
    tuple[h5py.Group, h5py.Group, str]
        The channel group, measurement group, and dataset name.

    Raises
    ------
    ValueError
        If the channel is not found.
    """
    _parse_channel_name(channel)  # just for validation

    channel_map = _discover_available_channels(f)
    if channel not in channel_map:
        raise ValueError(f"'{channel}' not found. Available channels: {list(channel_map)}")

    channel_path = channel_map[channel]
    channel_group = f[channel_path]
    measurement_group = f[channel_path.split("/")[0]]
    dataset_name = channel.split("_")[0].capitalize()

    return channel_group, measurement_group, dataset_name


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

    return multiplier, offset


def _decode_attr(attr: bytes | str) -> str:
    """
    Decode an attribute that may be bytes or a string.

    Parameters
    ----------
    attr : bytes or str
        The attribute to decode.

    Returns
    -------
    str
        The decoded string.
    """
    if isinstance(attr, bytes):
        return attr.decode("utf-8")
    return str(attr)


def _attr_to_bool(attr: bytes | str | bool | int | float) -> bool:
    """
    Convert an attribute to a boolean value.

    Parameters
    ----------
    attr : bytes, str, bool, int, or float
        The attribute to convert.

    Returns
    -------
    bool
        The boolean interpretation of the value.
    """
    if isinstance(attr, (bytes | str)):
        return _decode_attr(attr).strip().lower() == "true"
    return bool(attr)


def _discover_available_channels(f: h5py.File) -> dict[str, str]:
    """
    Discover all available scan channels in the HDF5 file.

    Parameters
    ----------
    f : h5py.File
        The open HDF5 file.

    Returns
    -------
    dict[str, str]
        Mapping of channel names (e.g. 'height_trace') to their full HDF5 path.
    """
    channel_map = {}
    for m_key, m_group in f.items():
        if not m_key.startswith("Measurement_"):
            continue
        for c_key in m_group.keys():
            if not c_key.startswith("Channel_"):
                continue

            c_group = m_group[c_key]
            name = c_group.attrs.get("channel.name")
            if name is None:
                continue

            retrace = _attr_to_bool(c_group.attrs.get("retrace", False))
            tr_rt = "retrace" if retrace else "trace"
            full_key = f"{_decode_attr(name).strip().lower()}_{tr_rt}"
            full_path = f"{m_key}/{c_key}"
            if full_key not in channel_map:
                channel_map[full_key] = full_path
    return channel_map


def _get_line_rate(measurement_group: h5py.Group) -> float:
    """
    Extract image line rate from an HDF5 JPK measurement group.

    The line rate is the scan speed in terms of lines per second,
    i.e. the speed of imaging in fast scan lines / second.

    Parameters
    ----------
    measurement_group : h5py.Group
        HDF5 group corresponding to a Measurement (e.g. '/Measurement_000').

    Returns
    -------
    float
        The line rate of imaging in lines per second.

    Raises
    ------
    KeyError
        If required attributes are missing in the measurement group.
    """
    try:
        return measurement_group.attrs["timing-settings.scanRate"]  # scan lines per second

    except KeyError as e:
        missing = e.args[0]
        raise KeyError(f"Missing required attribute '{missing}' in HDF5 measurement group.") from e


def generate_timestamps(num_frames: int, line_rate: float, image_size: int) -> dict:
    """
    Generate timestamps for a sequence of frames based on scan line rate and image size.

    Parameters
    ----------
    num_frames : int
        The total number of frames to generate timestamps for.
    line_rate : float
        The scan line rate in lines per second (Hz).
    image_size : int
        The number of horizontal lines per image (i.e., image height in pixels).

    Returns
    -------
    dict
        A dictionary mapping frame labels (e.g., "frame 0") to timestamps in seconds.
    """
    frame_interval = image_size / line_rate  # seconds per frame (height lines / lines per second)
    timestamps = np.arange(num_frames) * frame_interval
    # Compose a dictionary of timestamsps
    return {f"frame {i}": timestamp for i, timestamp in enumerate(timestamps)}


def load_h5jpk(
    file_path: Path | str, channel: str, flip_image: bool = True
) -> tuple[np.ndarray, float, dict[str, float]]:
    """
    Load image from JPK Instruments .h5-jpk files.

    Parameters
    ----------
    file_path : Path | str
        Path to the .h5-jpk file.
    channel : str
        The channel to extract from the .h5-jpk file.
    flip_image : bool, optional
        Whether to flip the images vertically. Default is ``True``.

    Returns
    -------
    image : np.ndarray
        3D array of shape (frames, height, width) with image data.
    pixel_to_nm_scaling : float
        Scaling factor converting pixels to nanometers.
    timestamps : dict[str, float]
        Dictionary mapping frame labels (e.g., "frame 0") to timestamp values in seconds.
    
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
    >>> frames, pixel_to_nanometre_scaling_factor, timestamps = load_h5jpk(file_path="./my_jpk_file.jpk",
    >>>                                                         channel="height_trace",
    >>>                                                         flip_image=True)
    """
    logger.info(f"Loading H5-JPK file from : {file_path}")
    file_path = Path(file_path)

    # Load HDF5 file
    with h5py.File(file_path, "r") as f:
        logger.info(f"Opened HDF5 file structure: {list(f.keys())}")

        channel_group, measurement_group, dataset_name = _get_channel_info(f, channel)

        # Load images and scaling factors from channel dataset
        images = channel_group[dataset_name][:]
        scaling, offset = _get_z_scaling_h5(channel_group)
        images = (images * scaling) + offset

        # Select and reshape a flattened frame
        image_size = measurement_group.attrs["position-pattern.grid.ilength"]  # number of pixels

        # Reshape each column vector (height, width) to get (num_frames, height, width)
        num_frames = images.shape[1]
        image_stack = np.empty((num_frames, image_size, image_size), dtype=images.dtype)
        for i in range(num_frames):
            frame = images[:, i].reshape((image_size, image_size))

            # Flip images
            if flip_image:
                frame = np.flipud(frame)
            image_stack[i] = frame

        # Convert to nm
        if dataset_name.lower() in ("height", "error", "measuredheight", "amplitude"):
            image_stack = image_stack * 1e9

        # Generate a dictionary of timestamps
        line_rate = _get_line_rate(measurement_group)
        timestamps = generate_timestamps(num_frames, line_rate, image_size)

        logger.info(f"[{file_path.stem}] : Extracted {num_frames} frames from channel '{channel}'")
        return (image_stack, _jpk_pixel_to_nm_scaling_h5(measurement_group), timestamps)
