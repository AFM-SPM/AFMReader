"""For decoding and loading .spm AFM file format into Python Numpy arrays."""

from pathlib import Path

import numpy as np
import pySPM

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
        "pm": 1e-3,
        "nm": 1,
        "um": 1e3,
        "mm": 1e6,
    }
    px_to_real = channel_data.pxs()
    # Has potential for non-square pixels but not yet implimented
    pixel_to_nm_scaling = (
        px_to_real[0][0] * unit_dict[px_to_real[0][1]],
        px_to_real[1][0] * unit_dict[px_to_real[1][1]],
    )[0]
    # ns-rse : Perhaps just switch to _always_ using the parameters from channel_data.size to calculate scaling?
    if px_to_real[0][0] == 0 and px_to_real[1][0] == 0:
        logger.info(
            f"[{filename}] : Pixel to nm scaling not directly available, calculating from 'channel_data.size['real']' "
            "and 'channel_data.size['pixels']'."
        )
        pixel_to_nm_scaling = (
            (channel_data.size["real"]["x"] / channel_data.size["pixels"]["x"])
            / unit_dict[channel_data.size["real"]["unit"]],
            (channel_data.size["real"]["y"] / channel_data.size["pixels"]["y"])
            / unit_dict[channel_data.size["real"]["unit"]],
        )[0]
    logger.info(f"[{filename}] : Pixel to nm scaling : {pixel_to_nm_scaling}")
    return pixel_to_nm_scaling


def _channel_name_from_layer(layer: dict, encoding: str = "latin1") -> str | None:
    """
    Extract the channel name from a pySPM Bruker layer's '@2:Image Data' field.

    Parameters
    ----------
    layer : dict
        A single entry from ``pySPM.Bruker.layers``.
    encoding : str
        Text encoding for decoding the raw bytes. Default "latin1".

    Returns
    -------
    str or None
        The channel name (the quoted portion of '@2:Image Data'), or None if the
        layer has no image-data field.
    """
    image_data = layer.get(b"@2:Image Data")
    if not image_data:
        return None
    # '@2:Image Data' looks like:  ZSensor [ZSensor] "Height Sensor"
    # the channel name is the quoted portion.
    text = image_data[0].decode(encoding)
    try:
        return text.split('"')[1]
    except IndexError:
        return None


def _layer_is_retrace(layer: dict, encoding: str = "latin1") -> bool:
    """
    Determine whether a pySPM Bruker layer is a retrace ('Line Direction': Retrace).

    Parameters
    ----------
    layer : dict
        A single entry from ``pySPM.Bruker.layers``.
    encoding : str
        Text encoding for decoding the raw bytes. Default "latin1".

    Returns
    -------
    bool
        True if the layer's line direction is 'Retrace', otherwise False (trace).
    """
    line_direction = layer.get(b"Line Direction")
    if not line_direction:
        return False
    return line_direction[0].decode(encoding).strip().lower() == "retrace"


def spm_channel_list(scan: "pySPM.Bruker", encoding: str = "latin1") -> dict[str, tuple[str, bool]]:
    """
    Build the available channel list for a Bruker .spm scan, including direction.

    Each Bruker layer carries a channel name ('@2:Image Data') and a line
    direction ('Line Direction'). The same channel name usually appears twice,
    once per direction, so the name alone is not unique. Keys are therefore
    formed as ``"<name> trace"`` / ``"<name> retrace"``.

    Parameters
    ----------
    scan : pySPM.Bruker
        An opened Bruker scan (header already parsed; no pixel data read).
    encoding : str
        Text encoding for decoding the raw header bytes. Default "latin1".

    Returns
    -------
    dict[str, tuple[str, bool]]
        Mapping of ``"<name> <trace|retrace>"`` to ``(channel_name, backward)``,
        where ``backward`` is the flag ``pySPM.Bruker.get_channel`` expects
        (True for retrace).
    """
    channel_list: dict[str, tuple[str, bool]] = {}
    # Keys are constructed as "<name> <direction>" and only ever matched whole
    # (never split back apart), so a channel name containing a space is safe.
    for layer in scan.layers:
        name = _channel_name_from_layer(layer, encoding)
        if name is None:
            continue
        backward = _layer_is_retrace(layer, encoding)
        tr_rt = "retrace" if backward else "trace"
        channel_list[f"{name} {tr_rt}"] = (name, backward)
    return channel_list


def load_spm(file_path: Path | str, channel: str) -> tuple:
    """
    Extract image and pixel to nm scaling from the Bruker .spm file.

    The ``channel`` argument accepts either a bare channel name (e.g.
    ``"Height Sensor"``), which loads the forward (trace) image for backwards
    compatibility (or retrace if that is the only available direction), or an
    explicit direction suffix (e.g. ``"Height Sensor trace"``
    / ``"Height Sensor retrace"``) to select a scan direction. Use
    :func:`spm_channel_list` (or catch the ValueError below) to see the available
    ``"<name> <direction>"`` keys for a file.

    Parameters
    ----------
    file_path : Path or str
        Path to the .spm file.
    channel : str
        Channel name to extract. Either ``"<name>"`` (defaults to trace) or
        ``"<name> trace"`` / ``"<name> retrace"``.

    Returns
    -------
    tuple(np.ndarray, float)
        A tuple containing the image and its pixel to nanometre scaling value.

    Raises
    ------
    FileNotFoundError
        If the file is not found.
    ValueError
        If the channel is not found in the .spm file. The error lists the
        available ``"<name> <direction>"`` channel keys.

    Examples
    --------
    Load the forward image (backwards-compatible bare name):

    >>> from AFMReader.spm import load_spm
    >>> image, pixel_to_nm = load_spm(file_path="path/to/file.spm", channel="Height Sensor")

    Load a specific scan direction:

    >>> image, pixel_to_nm = load_spm(file_path="path/to/file.spm", channel="Height Sensor retrace")
    """
    logger.info(f"Loading image from : {file_path}")
    file_path = Path(file_path)
    filename = file_path.stem
    try:
        scan = pySPM.Bruker(file_path)
        logger.info(f"[{filename}] : Loaded image from : {file_path}")

        # Build the direction-aware channel list from the parsed header.
        channel_list = spm_channel_list(scan)

        # Resolve the requested channel to a (name, backward) pair.
        # NB: match the requested string against whole keys only, don't split
        # on space to extract the direction, channel names contain spaces
        # (e.g. "Height Sensor"). All matching is case-insensitive.
        lookup = {key.lower(): value for key, value in channel_list.items()}
        request = channel.strip().lower()

        if request in lookup:
            # Exact "<name>" (with direction) match, case-insensitive.
            channel_name, backward = lookup[request]
        elif f"{request} trace" in lookup:
            # Bare name, no direction specified — default to trace, and say so.
            logger.info(f"[{filename}] : no direction specified for '{channel}'; loading trace.")
            channel_name, backward = lookup[f"{request} trace"]
        elif f"{request} retrace" in lookup:
            # Bare name and no trace image available — load retrace, and warn.
            logger.warning(
                f"[{filename}] : no direction specified for '{channel}' and no "
                f"trace image available; loading retrace."
            )
            channel_name, backward = lookup[f"{request} retrace"]
        else:
            logger.error(f"[{filename}] : '{channel}' not in {file_path.suffix} channel " f"list: {list(channel_list)}")
            raise ValueError(f"'{channel}' not in {file_path.suffix} channel list: " f"{list(channel_list)}")

        channel_data = scan.get_channel(channel_name, backward=backward)
        logger.info(f"[{filename}] : Extracted channel {channel_name} " f"({'retrace' if backward else 'trace'})")
        image = np.flipud(np.array(channel_data.pixels))
    except FileNotFoundError:
        logger.error(f"[{filename}] File not found : {file_path}")
        raise

    return (image, spm_pixel_to_nm_scaling(filename, channel_data))
