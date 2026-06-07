"""For decoding and loading .jpk AFM file format into Python Numpy arrays."""

from importlib import resources
from pathlib import Path
from io import BytesIO

import numpy as np
import tifffile

from AFMReader.io import read_yaml
from AFMReader.logging import logger
from AFMReader.data_classes import AFMLoad

logger.enable(__package__)

# pylint: disable=too-many-locals,too-many-positional-arguments,fixme


def _jpk_pixel_to_nm_scaling(tiff_page: tifffile.tifffile.TiffPage, jpk_tags: dict[str, int]) -> float:
    """
    Extract pixel to nm scaling from the JPK image metadata.

    Parameters
    ----------
    tiff_page : tifffile.tifffile.TiffPage
        An image file directory (IFD) of .jpk files.
    jpk_tags : dict[str, int]
        Dictionary of JPK tags.

    Returns
    -------
    float
        A value corresponding to the real length of a single pixel.
    """
    length = tiff_page.tags[jpk_tags["grid_ulength"]].value  # Grid-uLength (fast)
    width = tiff_page.tags[jpk_tags["grid_vlength"]].value  # Grid-vLength (slow)
    length_px = tiff_page.tags[jpk_tags["grid_ilength"]].value  # Grid-iLength (fast)
    width_px = tiff_page.tags[jpk_tags["grid_jlength"]].value  # Grid-jLength (slow)

    px_to_nm = (length / length_px, width / width_px)[0]

    return px_to_nm * 1e9


def _get_tag_value(page: tifffile.TiffPage, tag_name: str) -> str | int | float:
    """
    Retrieve the value of a specified tag from a TIFF page of a JPK file.

    Parameters
    ----------
    page : tifffile.TiffPage
        The TIFF page from which to retrieve the tag value.
    tag_name : str
        The name of the tag to retrieve.

    Returns
    -------
    Any
        The value of the specified tag.

    Raises
    ------
    KeyError
        If the tag is not found in the TIFF page.
    """
    try:
        return page.tags[tag_name].value
    except KeyError:
        logger.error(f"Missing tag in JPK file: {tag_name}")
        raise


def _get_number_of_slots(tif: tifffile.tifffile, channel_idx: int, jpk_tags: dict[str, int]) -> int:
    """
    Retrieve the number of slots from a JPK image channel.

    Parameters
    ----------
    tif : tifffile.TiffFile
        A TIFF file containing JPK images.
    channel_idx : int
        Numerical channel identifier used to navigate the TIFF file pages.
    jpk_tags : dict[str, int]
        Dictionary of JPK tags.

    Returns
    -------
    int
        The number of slots in the specified JPK image channel.

    Raises
    ------
    ValueError
        If the slot count cannot be converted to an integer.
    """
    try:
        n_slots = int(_get_tag_value(tif.pages[channel_idx], str(jpk_tags["n_slots"])))
    except ValueError as e:
        logger.error(f"n_slots is not a integer: {e}")
        raise
    return n_slots


def _get_z_scaling(tif: tifffile.tifffile, channel_idx: int, jpk_tags: dict[str, int]) -> tuple[float, float, str]:
    """
    Extract the z scaling factor and offset for a JPK image channel.

    Determined using JPKImageSpec.txt Version: 2.0:3fffffffffff supplied with JPK instrument.

    Parameters
    ----------
    tif : tifffile.tifffile
        A tiff file of .jpk images.
    channel_idx : int
        Numerical channel identifier used to navigate the tifffile pages.
    jpk_tags : dict[str, int]
        Dictionary of JPK tags.

    Returns
    -------
    tuple[float, float, str]
        A tuple contains values used to scale and offset raw data, and the unit of the z-axis.
    """
    # Create a dictionary of list for the differnt slots
    n_slots = _get_number_of_slots(tif, channel_idx, jpk_tags)
    slots: dict[int, list[str]] = {slot: [] for slot in range(n_slots)}

    # Extract the tags with numerical names in each slot
    while n_slots >= 0:
        for tag in tif.pages[channel_idx].tags:
            try:
                tag_name_int = int(tag.name)
                if tag_name_int >= (  # pylint: disable=chained-comparison
                    int(jpk_tags["first_slot_tag"]) + (n_slots * int(jpk_tags["slot_size"]))
                ) and tag_name_int < (int(jpk_tags["first_slot_tag"]) + ((n_slots + 1) * jpk_tags["slot_size"])):
                    slots[n_slots].append(tag.name)
            except ValueError:
                continue
        n_slots -= 1

    # Find the number of the default slot (selected in the instrument GUI)
    default_slot = tif.pages[channel_idx].tags[jpk_tags["default_slot"]]
    for slot, values in slots.items():
        for value in values:
            if tif.pages[channel_idx].tags[str(value)].value == default_slot.value:
                _default_slot = slot
    # pylint: disable=possibly-used-before-assignment
    # Determine if the default slot requires scaling and find scaling and offset values
    scaling_type = _get_tag_value(
        tif.pages[channel_idx],
        str(int(jpk_tags["first_scaling_type"]) + (jpk_tags["slot_size"] * (_default_slot))),
    )
    # pylint: enable=possibly-used-before-assignment
    if scaling_type == "LinearScaling":
        scaling_name = (
            tif.pages[channel_idx]
            .tags[str(int(jpk_tags["first_scaling_name"]) + (jpk_tags["slot_size"] * (_default_slot)))]
            .name
        )
        offset_name = (
            tif.pages[channel_idx]
            .tags[str(int(jpk_tags["first_offset_name"]) + (jpk_tags["slot_size"] * (_default_slot)))]
            .name
        )
        unit_name = (
            tif.pages[channel_idx]
            .tags[str(int(jpk_tags["first_unit_name"]) + (jpk_tags["slot_size"] * (_default_slot)))]
            .name
        )

        scaling = float(_get_tag_value(tif.pages[channel_idx], scaling_name))
        offset = float(_get_tag_value(tif.pages[channel_idx], offset_name))
        z_units = str(_get_tag_value(tif.pages[channel_idx], unit_name))
    elif scaling_type == "NullScaling":
        scaling = 1.0
        offset = 0.0
        z_units = "raw"
    else:
        raise ValueError(f"Scaling type {scaling_type} is not 'NullScaling' or 'LinearScaling'")
    return scaling, offset, z_units


def _get_jpk_channels(
    file: Path | BytesIO, filename: str, file_path: Path | str, config_path: Path | str | None = None
):
    """
    Retrieve the list of available channels from a JPK TIFF file.

    Parameters
    ----------
    file : Path | BytesIO
        Path to the JPK TIFF file.
    filename : str
        Name of the JPK TIFF file.
    file_path : Path | str
        Path to the JPK TIFF file.
    config_path : Path | str | None, optional
        Path to a configuration file. If ''None'' (default) then the packages
        default configuration is loaded from ''default_config.yaml''.

    Returns
    -------
    dict
        Dictionary of available channels with their corresponding page indices.
    """
    jpk_tags = _load_jpk_tags(config_path)
    try:
        tif = tifffile.TiffFile(file)
    except FileNotFoundError:
        logger.error(f"[{filename}] File not found : {file_path}")
        raise
    # Obtain channel list for all channels in file
    channel_list = {}
    for i, page in enumerate(tif.pages[1:]):  # [0] is thumbnail
        available_channel = page.tags[jpk_tags["channel_name"]].value  # keys are hexadecimal values
        if page.tags[jpk_tags["trace_retrace"]].value == 0:  # whether img is trace or retrace
            tr_rt = "trace"
        else:
            tr_rt = "retrace"
        channel_list[f"{available_channel}_{tr_rt}"] = i + 1
    return channel_list


def get_jpk_channels(file_path: Path | str, config_path: Path | str | None = None) -> list[str]:
    """
    Get the list of channels available in the .jpk file.

    Parameters
    ----------
    file_path : Path | str
        Path to the .jpk file.
    config_path : Path | str | None
        Path to a configuration file. If ''None'' (default) then the packages
        default configuration is loaded from ''default_config.yaml''.

    Returns
    -------
    list[str]
        List of available channels.
    """
    file_path = Path(file_path)
    filename = file_path.stem
    return _get_jpk_channels(file_path, filename, file_path, config_path)


def load_jpk(
    file_path: Path | str, channel: str, config_path: Path | str | None = None, flip_image: bool = True
) -> AFMLoad:
    """
    Load image from JPK Instruments .jpk files.

    Parameters
    ----------
    file_path : Path | str
        Path to the .jpk file.
    channel : str
        The channel to extract from the .jpk file.
    config_path : Path | str | None
        Path to a configuration file. If ''None'' (default) then the packages default configuration is loaded from
        ''default_config.yaml''.
    flip_image : bool
        Whether to flip the image vertically. Default is ``True``.

    Returns
    -------
    AFMLoad
        An AFMLoad object containing the image, its pixel to nanometre scaling value, and z-axis units.

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
    >>> afm_load = load_jpk(file_path="./my_jpk_file.jpk", channel="height_trace", flip_image=True)
    >>> image = afm_load.image
    >>> pixel_to_nanometre_scaling_factor = afm_load.px2nm
    >>> z_units = afm_load.z_units
    """
    logger.info(f"Loading image from : {file_path}")
    file_path = Path(file_path)
    filename = file_path.stem
    image, px2nm, units = _load_jpk(
        file=file_path,
        filename=filename,
        channel=channel,
        file_suffix=file_path.suffix,
        config_path=config_path,
        flip_image=flip_image,
    )
    return AFMLoad(image=image, px2nm=px2nm, z_units=units)


def _load_jpk(
    file: Path | BytesIO,
    filename: str,
    channel: str,
    file_suffix: str,
    config_path: Path | str | None = None,
    flip_image: bool = True,
    convert_to_nm: bool = True,
) -> tuple[np.ndarray, float, str]:
    """
    Load image data, pixel scaling, and z-axis units from a JPK TIFF file for a given channel.

    Parameters
    ----------
    file : Path | BytesIO
        Path to the JPK TIFF file.
    filename : str
        Name of the JPK TIFF file.
    channel : str
        The channel to extract from the JPK TIFF file.
    file_suffix : str
        The file suffix of the JPK TIFF file.
    config_path : Path | str | None, optional
        Path to a configuration file. If ''None'' (default) then the packages default configuration is
        loaded from ''default_config.yaml''.
    flip_image : bool, optional
        Whether to flip the image vertically. Default is True.
    convert_to_nm : bool, optional
        Whether to convert the image to nanometres. Default is True.

    Returns
    -------
    tuple[np.ndarray, float, str]
        A tuple containing the image, its pixel to nanometre scaling value, and the z-axis units.
    """
    jpk_tags = _load_jpk_tags(config_path)
    try:
        tif = tifffile.TiffFile(file)
    except FileNotFoundError:
        logger.error(f"[{filename}] File not found : {file}")
        raise
    # Obtain channel list for all channels in file
    channel_list = {}
    for i, page in enumerate(tif.pages[1:]):  # [0] is thumbnail
        available_channel = page.tags[jpk_tags["channel_name"]].value  # keys are hexadecimal values
        if page.tags[jpk_tags["trace_retrace"]].value == 0:  # whether img is trace or retrace
            tr_rt = "trace"
        else:
            tr_rt = "retrace"
        channel_list[f"{available_channel}_{tr_rt}"] = i + 1
    try:
        channel_idx = channel_list[channel]
    except KeyError as e:
        logger.error(f"'{channel}' not in {file_suffix} channel list: {channel_list}")
        raise ValueError(f"'{channel}' not in {file_suffix} channel list: {channel_list}") from e

    # Get image and if applicable, scale it
    channel_page = tif.pages[channel_idx]
    image = channel_page.asarray()
    scaling, offset, z_units = _get_z_scaling(tif, channel_idx, jpk_tags)
    image = (image * scaling) + offset
    if flip_image is True:
        image = np.flipud(image)

    if convert_to_nm and z_units == "m":
        image = image * 1e9
        z_units = "nm"

    # Get page for common metadata between scans
    metadata_page = tif.pages[0]

    logger.info(f"[{filename}] : Extracted image.")
    return (image, _jpk_pixel_to_nm_scaling(metadata_page, jpk_tags), z_units)


def _load_jpk_tags(config_path: str | Path | None = None) -> dict[str, int]:
    """
    Load the configuration file and extract JPK options.

    If no ''config_path'' is provided the ''default_config.yaml'' is loaded.

    Parameters
    ----------
    config_path : str | Path | None
        Path to a YAML configuration file.

    Returns
    -------
    dict[str, Any]
        Dictionary of JPK configuration options.
    """
    if config_path is None:
        config_path = resources.files(__package__) / "default_config.yaml"  # type: ignore[assignment]
    config = read_yaml(Path(config_path))  # type: ignore[arg-type]
    logger.info(f"Configuration loaded from : {config_path}")
    return config["jpk"]
