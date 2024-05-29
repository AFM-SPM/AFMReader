"""For decoding and loading .asd AFM file format into Python Numpy arrays."""

from __future__ import annotations
from pathlib import Path

from typing import BinaryIO


import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib import animation


from AFMReader.logging import logger
from AFMReader.io import (
    read_int32,
    read_int16,
    read_float,
    read_bool,
    read_hex_u32,
    read_ascii,
    read_uint8,
    read_null_separated_utf8,
    read_int8,
    read_double,
    skip_bytes,
)

logger.enable(__package__)


# pylint: disable=too-few-public-methods
class VoltageLevelConverter:
    """
    A class for converting arbitrary height levels from the AFM into real world nanometre heights.

    Different .asd files require different functions to perform this calculation based on many factors, hence why we
    need to define the correct function in each case.

    Parameters
    ----------
    analogue_digital_range : float
        The range of analogue voltage values.
    scaling_factor : float
        A scaling factor calculated elsewhere that scales the heightmap appropriately based on the type of channel
        and sensor parameters.
    resolution : int
        The vertical resolution of the instrument. Dependant on the number of bits used to store its
        values. Typically 12, hence 2^12 = 4096 sensitivity levels.
    """

    def __init__(self, analogue_digital_range: float, scaling_factor: float, resolution: int) -> None:
        """
        Convert arbitrary height levels from the AFM into real world nanometre heights.

        Parameters
        ----------
        analogue_digital_range : float
            The range of analogue voltage values.
        scaling_factor : float
            A scaling factor calculated elsewhere that scales the heightmap appropriately based on the type of channel
            and sensor parameters.
        resolution : int
            The vertical resolution of the instrumen. Dependant on the number of bits used to store its
            values. Typically 12, hence 2^12 = 4096 sensitivity levels.
        """
        self.ad_range = analogue_digital_range
        self.scaling_factor = scaling_factor
        self.resolution = resolution
        logger.info(
            f"created voltage converter. ad_range: {analogue_digital_range} -> {self.ad_range}, "
            f" scaling factor: {scaling_factor}, resolution: {resolution}"
        )


# pylint: disable=too-few-public-methods
class UnipolarConverter(VoltageLevelConverter):
    """A VoltageLevelConverter for unipolar encodings. (0 to +X Volts)."""

    def level_to_voltage(self, level: float) -> float:
        """
        Calculate the real world height scale in nanometres for an arbitrary level value.

        Parameters
        ----------
        level : float
            Arbitrary height measurement from the AFM that needs converting into real world
            length scale units.

        Returns
        -------
        float
            Real world nanometre height for the input height level.
        """
        multiplier = -self.ad_range / self.resolution * self.scaling_factor
        return level * multiplier


# pylint: disable=too-few-public-methods
class BipolarConverter(VoltageLevelConverter):
    """A VoltageLevelConverter for bipolar encodings. (-X to +X Volts)."""

    def level_to_voltage(self, level: float) -> float:
        """
        Calculate the real world height scale in nanometres for an arbitrary level value.

        Parameters
        ----------
        level : float
            Arbitrary height measurement from the AFM that needs converting into real world
            length scale units.

        Returns
        -------
        float
            Real world nanometre height for the input height level.
        """
        return (self.ad_range - 2 * level * self.ad_range / self.resolution) * self.scaling_factor


def calculate_scaling_factor(
    channel: str,
    z_piezo_gain: float,
    z_piezo_extension: float,
    scanner_sensitivity: float,
    phase_sensitivity: float,
) -> float:
    """
    Calculate the correct scaling factor.

    This function should be used in conjunction with the VoltageLevelConverter class to define the correct function and
    enables conversion between arbitrary level values from the AFM into real world nanometre height values.

    Parameters
    ----------
    channel : str
        The .asd channel being used.
    z_piezo_gain : float
        The z_piezo_gain listed in the header metadata for the .asd file.
    z_piezo_extension : float
        The z_piezo_extension listed in the header metadata for the .asd file.
    scanner_sensitivity : float
        The scanner_sensitivity listed in the header metadata for the .asd file.
    phase_sensitivity : float
        The phase_sensitivity listed in the heder metadata for the .asd file.

    Returns
    -------
    float
        The appropriate scaling factor to pass to a VoltageLevelConverter to convert arbitrary
        height levels to real world nanometre heights for the frame data in the specified channl
        in the .asd file.
    """
    if channel == "TP":
        logger.info(
            f"Scaling factor: Type: {channel} -> TP | piezo extension {z_piezo_gain} "
            f"* piezo gain {z_piezo_extension} = scaling factor {z_piezo_gain * z_piezo_extension}"
        )
        return z_piezo_gain * z_piezo_extension
    if channel == "ER":
        logger.info(
            f"Scaling factor: Type: {channel} -> ER | - scanner sensitivity {-scanner_sensitivity} "
            f"= scaling factor {-scanner_sensitivity}"
        )
        return -scanner_sensitivity
    if channel == "PH":
        logger.info(
            f"Scaling factor: Type: {channel} -> PH | - phase sensitivity {-phase_sensitivity} "
            f"= scaling factor {-phase_sensitivity}"
        )
        return -phase_sensitivity

    raise ValueError(f"channel {channel} not known for .asd file type.")


def load_asd(file_path: Path, channel: str):
    """
    Load a .asd file.

    Parameters
    ----------
    file_path : Path
        Path to the .asd file.
    channel : str
        Channel to load. Note that only three channels seem to be present in a single .asd file. Options: TP
        (Topograph), ER (Error) and PH (Phase).

    Returns
    -------
    npt.NDArray
        The .asd file frames data as a numpy 3D array N x W x H
        (Number of frames x Width of each frame x height of each frame).
    float
        The number of nanometres per pixel for the .asd file. (AKA the resolution).
        Enables converting between pixels and nanometres when working with the data, in order to use real-world length
        scales.
    dict
        Metadata for the .asd file. The number of entries is too long to list here, and changes based on the file
        version please either look into the `read_header_file_version_x` functions or print the keys too see what
        metadata is available.
    """
    # Ensure the file path is a Path object
    file_path = Path(file_path)
    # Open the file in binary mode
    with Path.open(file_path, "rb", encoding=None) as open_file:  # pylint: disable=unspecified-encoding
        file_version = read_file_version(open_file)

        if file_version == 0:
            header_dict = read_header_file_version_0(open_file)

        elif file_version == 1:
            header_dict = read_header_file_version_1(open_file)

        elif file_version == 2:
            header_dict = read_header_file_version_2(open_file)
        else:
            raise ValueError(
                f"File version {file_version} unknown. Please add support if you "
                "know how to decode this file version."
            )
        logger.debug(f"header dict: \n{header_dict}")

        pixel_to_nanometre_scaling_factor_x = header_dict["x_nm"] / header_dict["x_pixels"]
        pixel_to_nanometre_scaling_factor_y = header_dict["y_nm"] / header_dict["y_pixels"]
        if pixel_to_nanometre_scaling_factor_x != pixel_to_nanometre_scaling_factor_y:
            logger.warning(
                f"Resolution of image is different in x and y directions:"
                f"x: {pixel_to_nanometre_scaling_factor_x}"
                f"y: {pixel_to_nanometre_scaling_factor_y}"
            )
        pixel_to_nanometre_scaling_factor = pixel_to_nanometre_scaling_factor_x

        if channel == header_dict["channel1"]:
            logger.info(f"Requested channel {channel} matches first channel in file: {header_dict['channel1']}")
        elif channel == header_dict["channel2"]:
            logger.info(f"Requested channel {channel} matches second channel in file: " f"{header_dict['channel2']}")

            # Skip first channel data
            _size_of_frame_header = header_dict["frame_header_length"]
            # Remember that each value is two bytes (since signed int16)
            size_of_single_frame_plus_header = (
                header_dict["frame_header_length"] + header_dict["x_pixels"] * header_dict["y_pixels"] * 2
            )
            length_of_all_first_channel_frames = header_dict["num_frames"] * size_of_single_frame_plus_header
            _ = open_file.read(length_of_all_first_channel_frames)
        else:
            raise ValueError(
                f"Channel {channel} not found in this file's available channels: "
                f"{header_dict['channel1']}, {header_dict['channel2']}"
            )

        scaling_factor = calculate_scaling_factor(
            channel=channel,
            z_piezo_gain=header_dict["z_piezo_gain"],
            z_piezo_extension=header_dict["z_piezo_extension"],
            scanner_sensitivity=header_dict["scanner_sensitivity"],
            phase_sensitivity=header_dict["phase_sensitivity"],
        )

        analogue_digital_converter = create_analogue_digital_converter(
            analogue_digital_range=header_dict["analogue_digital_range"],
            scaling_factor=scaling_factor,
        )
        frames = read_channel_data(
            open_file=open_file,
            num_frames=header_dict["num_frames"],
            x_pixels=header_dict["x_pixels"],
            y_pixels=header_dict["y_pixels"],
            analogue_digital_converter=analogue_digital_converter,
        )

        frames = np.array(frames)

        return frames, pixel_to_nanometre_scaling_factor, header_dict


def read_file_version(open_file: BinaryIO) -> int:
    """
    Read the file version from an open asd file. File versions are 0, 1 and 2.

    Different file versions require different functions to read the headers as the formatting changes between them.

    Parameters
    ----------
    open_file : BinaryIO
        An open binary file object for a .asd file.

    Returns
    -------
    int
        Integer file version decoded from file.
    """
    file_version = read_int32(open_file)
    logger.info(f"file version: {file_version}")
    return file_version


# pylint: disable=too-many-statements
def read_header_file_version_0(open_file: BinaryIO) -> dict:
    """
    Read the header metadata for a .asd file using file version 0.

    Parameters
    ----------
    open_file : BinaryIO
        An open binary file object for a .asd file.

    Returns
    -------
    dict
        Dictionary of metadata decoded from the file header.
    """
    header_dict = {}

    # There only ever seem to be two channels available
    # Channel encoding are all in LITTLE ENDIAN format.
    # topology: 0x5054 decodes to 'TP' in ascii little endian
    # error: 0x5245 decodes to 'ER' in ascii little endian
    # phase: 0x4850 decodes to 'PH' in ascii little endian
    header_dict["channel1"] = read_ascii(open_file, 2)
    header_dict["channel2"] = read_ascii(open_file, 2)
    # length of file metadata header in bytes - so we can skip it to get to the data
    header_dict["header_length"] = read_int32(open_file)
    # Frame header is the length of the header for each frame to be skipped
    # before reading frame data.
    header_dict["frame_header_length"] = read_int32(open_file)
    # Length in bytes of the name given in the file
    header_dict["user_name_size"] = read_int32(open_file)
    header_dict["comment_offset_size"] = read_int32(open_file)
    # Length in bytes of the comment for the file
    header_dict["comment_size"] = read_int32(open_file)
    # x and y resolution (pixels)
    header_dict["x_pixels"] = read_int16(open_file)
    header_dict["y_pixels"] = read_int16(open_file)
    # x and y resolution (nm)
    header_dict["x_nm"] = read_int16(open_file)
    header_dict["y_nm"] = read_int16(open_file)
    # frame time
    header_dict["frame_time"] = read_float(open_file)
    # z piezo extension
    header_dict["z_piezo_extension"] = read_float(open_file)
    # z piezo gain
    header_dict["z_piezo_gain"] = read_float(open_file)
    # Range of analogue voltage values (for conversion to digital)
    header_dict["analogue_digital_range"] = read_hex_u32(open_file)
    # Number of bits of data for analogue voltage values (for conversion to digital)
    # aka the resolution of the instrument. Usually 12 bits, so 4096 sensitivity levels
    header_dict["analogue_digital_data_bits_size"] = read_int32(open_file)
    header_dict["analogue_digital_resolution"] = 2 ^ header_dict["analogue_digital_data_bits_size"]
    # Not sure, something to do with data averaging
    header_dict["is_averaged"] = read_bool(open_file)
    # Window for averaging the data
    header_dict["averaging_window"] = read_int32(open_file)
    # Some padding to ensure backwards compatilibilty I think
    _ = read_int16(open_file)
    # Date of creation
    header_dict["year"] = read_int16(open_file)
    header_dict["month"] = read_uint8(open_file)
    header_dict["day"] = read_uint8(open_file)
    header_dict["hour"] = read_uint8(open_file)
    header_dict["minute"] = read_uint8(open_file)
    header_dict["second"] = read_uint8(open_file)
    # Rounding degree?
    header_dict["rounding_degree"] = read_uint8(open_file)
    # Maximum x and y scanning range in real space, nm
    header_dict["max_x_scan_range"] = read_float(open_file)
    header_dict["max_y_scan_range"] = read_float(open_file)
    # No idea
    _ = read_int32(open_file)
    _ = read_int32(open_file)
    _ = read_int32(open_file)
    # Number of frames the file had when recorded
    header_dict["initial_frames"] = read_int32(open_file)
    # Actual number of frames
    header_dict["num_frames"] = read_int32(open_file)
    # ID of the AFM instrument
    header_dict["afm_id"] = read_int32(open_file)
    # ID of the file
    header_dict["file_id"] = read_int16(open_file)
    # Name of the user
    header_dict["user_name"] = read_null_separated_utf8(open_file, length_bytes=header_dict["user_name_size"])
    # Sensitivity of the scanner in nm / V
    header_dict["scanner_sensitivity"] = read_float(open_file)
    # Phase sensitivity
    header_dict["phase_sensitivity"] = read_float(open_file)
    # Direction of the scan
    header_dict["scan_direction"] = read_int32(open_file)
    # Skip bytes: comment offset size
    _ = skip_bytes(open_file, header_dict["comment_offset_size"])
    # Read a comment
    comment = []
    for _ in range(header_dict["comment_size"]):
        comment.append(chr(read_int8(open_file)))
    header_dict["comment_without_null"] = "".join([c for c in comment if c != "\x00"])

    return header_dict


# pylint: disable=too-many-statements
def read_header_file_version_1(open_file: BinaryIO):
    """
    Read the header metadata for a .asd file using file version 1.

    Parameters
    ----------
    open_file : BinaryIO
        An open binary file object for a .asd file.

    Returns
    -------
    dict
        Dictionary of metadata decoded from the file header.
    """
    header_dict = {}

    # length of file metadata header in bytes - so we can skip it to get to the data
    header_dict["header_length"] = read_int32(open_file)
    # Frame header is the length of the header for each frame to be skipped before
    # reading frame data.
    header_dict["frame_header_length"] = read_int32(open_file)
    # Encoding for strings
    header_dict["text_encoding"] = read_int32(open_file)
    # Length in bytes of the name given in the file
    header_dict["user_name_size"] = read_int32(open_file)
    # Length in bytes of the comment for the file
    header_dict["comment_size"] = read_int32(open_file)
    # There only ever seem to be two channels available
    # Channel encoding are all in LITTLE ENDIAN format.
    # topology: 0x5054 decodes to 'TP' in ascii little endian
    # error: 0x5245 decodes to 'ER' in ascii little endian
    # phase: 0x4850 decodes to 'PH' in ascii little endian
    header_dict["channel1"] = read_null_separated_utf8(open_file, length_bytes=4)
    header_dict["channel2"] = read_null_separated_utf8(open_file, length_bytes=4)
    # Number of frames the file had when recorded
    header_dict["initial_frames"] = read_int32(open_file)
    # Actual number of frames
    header_dict["num_frames"] = read_int32(open_file)
    # Direction of the scan
    header_dict["scan_direction"] = read_int32(open_file)
    # ID of the file
    header_dict["file_id"] = read_int32(open_file)
    # x and y resolution (pixels)
    header_dict["x_pixels"] = read_int32(open_file)
    header_dict["y_pixels"] = read_int32(open_file)
    # x and y resolution (nm)
    header_dict["x_nm"] = read_int32(open_file)
    header_dict["y_nm"] = read_int32(open_file)
    # Not sure, something to do with data averaging
    header_dict["is_averaged"] = read_bool(open_file)
    # Window for averaging the data
    header_dict["averaging_window"] = read_int32(open_file)
    # Date of creation
    header_dict["year"] = read_int32(open_file)
    header_dict["month"] = read_int32(open_file)
    header_dict["day"] = read_int32(open_file)
    header_dict["hour"] = read_int32(open_file)
    header_dict["minute"] = read_int32(open_file)
    header_dict["second"] = read_int32(open_file)
    # Rounding degree?
    header_dict["x_rounding_degree"] = read_int32(open_file)
    header_dict["y_rounding_degree"] = read_int32(open_file)
    # frame time
    header_dict["frame_time"] = read_float(open_file)
    # Sensitivity of the scanner in nm / V
    header_dict["scanner_sensitivity"] = read_float(open_file)
    # Phase sensitivity
    header_dict["phase_sensitivity"] = read_float(open_file)
    # Offset?
    header_dict["offset"] = read_int32(open_file)
    # Ignore 12 bytes
    _ = skip_bytes(open_file, 12)
    # ID of the AFM instrument
    header_dict["afm_id"] = read_int32(open_file)
    # Range of analogue voltage values (for conversion to digital)
    header_dict["analogue_digital_range"] = read_hex_u32(open_file)
    # Number of bits of data for analogue voltage values (for conversion to digital)
    # aka the resolution of the instrument. Usually 12 bits, so 4096 sensitivity levels
    header_dict["analogue_digital_data_bits_size"] = read_int32(open_file)
    header_dict["analogue_digital_resolution"] = 2 ^ header_dict["analogue_digital_data_bits_size"]
    # Maximum x and y scanning range in real space, nm
    header_dict["max_x_scan_range"] = read_float(open_file)
    header_dict["max_y_scan_range"] = read_float(open_file)
    # Piezo extensions
    header_dict["x_piezo_extension"] = read_float(open_file)
    header_dict["y_piezo_extension"] = read_float(open_file)
    header_dict["z_piezo_extension"] = read_float(open_file)
    # Piezo gain
    header_dict["z_piezo_gain"] = read_float(open_file)

    # Read the user name
    user_name = []
    for _ in range(header_dict["user_name_size"]):
        user_name.append(chr(read_int8(open_file)))
    header_dict["user_name"] = "".join([c for c in user_name if c != "\x00"])

    # Read a comment
    comment = []
    for _ in range(header_dict["comment_size"]):
        comment.append(chr(read_int8(open_file)))
    header_dict["comment_without_null"] = "".join([c for c in comment if c != "\x00"])

    return header_dict


# pylint: disable=too-many-statements
def read_header_file_version_2(open_file: BinaryIO) -> dict:
    """
    Read the header metadata for a .asd file using file version 2.

    Parameters
    ----------
    open_file : BinaryIO
        An open binary file object for a .asd file.

    Returns
    -------
    dict
        Dictionary of metadata decoded from the file header.
    """
    header_dict = {}

    # length of file metadata header in bytes - so we can skip it to get to the data
    header_dict["header_length"] = read_int32(open_file)
    # Frame header is the length of the header for each frame to be skipped before
    # reading frame data.
    header_dict["frame_header_length"] = read_int32(open_file)
    # Encoding for strings
    header_dict["text_encoding"] = read_int32(open_file)
    # Length in bytes of the name given in the file
    header_dict["user_name_size"] = read_int32(open_file)
    # Length in bytes of the comment for the file
    header_dict["comment_size"] = read_int32(open_file)
    # There only ever seem to be two channels available
    # Channel encoding are all in LITTLE ENDIAN format.
    # topology: 0x5054 decodes to 'TP' in ascii little endian
    # error: 0x5245 decodes to 'ER' in ascii little endian
    # phase: 0x4850 decodes to 'PH' in ascii little endian
    header_dict["channel1"] = read_null_separated_utf8(open_file, length_bytes=4)
    header_dict["channel2"] = read_null_separated_utf8(open_file, length_bytes=4)
    # Number of frames the file had when recorded
    header_dict["initial_frames"] = read_int32(open_file)
    # Actual number of frames
    header_dict["num_frames"] = read_int32(open_file)
    # Direction of the scan
    header_dict["scan_direction"] = read_int32(open_file)
    # ID of the file
    header_dict["file_id"] = read_int32(open_file)
    # x and y resolution (pixels)
    header_dict["x_pixels"] = read_int32(open_file)
    header_dict["y_pixels"] = read_int32(open_file)
    # x and y resolution (nm)
    header_dict["x_nm"] = read_int32(open_file)
    header_dict["y_nm"] = read_int32(open_file)
    # Not sure, something to do with data averaging
    header_dict["is_averaged"] = read_bool(open_file)
    # Window for averaging the data
    header_dict["averaging_window"] = read_int32(open_file)
    # Date of creation
    header_dict["year"] = read_int32(open_file)
    header_dict["month"] = read_int32(open_file)
    header_dict["day"] = read_int32(open_file)
    header_dict["hour"] = read_int32(open_file)
    header_dict["minute"] = read_int32(open_file)
    header_dict["second"] = read_int32(open_file)
    # Rounding degree?
    header_dict["x_rounding_degree"] = read_int32(open_file)
    header_dict["y_rounding_degree"] = read_int32(open_file)
    # frame time
    header_dict["frame_time"] = read_float(open_file)
    # Sensitivity of the scanner in nm / V
    header_dict["scanner_sensitivity"] = read_float(open_file)
    # Phase sensitivity
    header_dict["phase_sensitivity"] = read_float(open_file)
    # Offset?
    header_dict["offset"] = read_int32(open_file)
    # Ignore 12 bytes
    _ = skip_bytes(open_file, 12)
    # ID of the AFM instrument
    header_dict["afm_id"] = read_int32(open_file)
    # Range of analogue voltage values (for conversion to digital)
    header_dict["analogue_digital_range"] = read_hex_u32(open_file)
    # Number of bits of data for analogue voltage values (for conversion to digital)
    # aka the resolution of the instrument. Usually 12 bits, so 4096 sensitivity levels
    header_dict["analogue_digital_data_bits_size"] = read_int32(open_file)
    header_dict["analogue_digital_resolution"] = 2 ^ header_dict["analogue_digital_data_bits_size"]
    # Maximum x and y scanning range in real space, nm
    header_dict["max_x_scan_range"] = read_float(open_file)
    header_dict["max_y_scan_range"] = read_float(open_file)
    # Piezo extensions
    header_dict["x_piezo_extension"] = read_float(open_file)
    header_dict["y_piezo_extension"] = read_float(open_file)
    header_dict["z_piezo_extension"] = read_float(open_file)
    # Piezo gain
    header_dict["z_piezo_gain"] = read_float(open_file)

    # Read the user name
    user_name = []
    for _ in range(header_dict["user_name_size"]):
        user_name.append(chr(read_int8(open_file)))
    header_dict["user_name"] = "".join([c for c in user_name if c != "\x00"])

    # Read a comment
    comment = []
    for _ in range(header_dict["comment_size"]):
        comment.append(chr(read_int8(open_file)))
    header_dict["comment_without_null"] = "".join([c for c in comment if c != "\x00"])

    # No idea why this file type has the number of frames again. Storing it just in case.
    header_dict["number_of_frames"] = read_int32(open_file)
    # Feed forward parameter, no idea what it does.
    header_dict["is_x_feed_forward_integer"] = read_int32(open_file)
    # Feed forward parameter, no idea what it does.
    header_dict["is_x_feed_forward_double"] = read_double(open_file)
    # Minimum and maximum colour mapping values
    header_dict["max_colour_scale"] = read_int32(open_file)
    header_dict["min_colour_scale"] = read_int32(open_file)
    # RGB anchor point array sizes
    header_dict["length_red_anchor_points"] = read_int32(open_file)
    header_dict["length_green_anchor_points"] = read_int32(open_file)
    header_dict["length_blue_anchor_points"] = read_int32(open_file)
    # Coords of anchor points
    # Red
    coords_red = []
    for _ in range(header_dict["length_red_anchor_points"]):
        anchor_x = read_int32(open_file)
        anchor_y = read_int32(open_file)
        coords_red.append((anchor_x, anchor_y))
    # Green
    coords_green = []
    for _ in range(header_dict["length_green_anchor_points"]):
        anchor_x = read_int32(open_file)
        anchor_y = read_int32(open_file)
        coords_green.append((anchor_x, anchor_y))
    # Blue
    coords_blue = []
    for _ in range(header_dict["length_blue_anchor_points"]):
        anchor_x = read_int32(open_file)
        anchor_y = read_int32(open_file)
        coords_blue.append((anchor_x, anchor_y))

    return header_dict


def read_channel_data(
    open_file: BinaryIO,
    num_frames: int,
    x_pixels: int,
    y_pixels: int,
    analogue_digital_converter: VoltageLevelConverter,
) -> npt.NDArray:
    """
    Read frame data from an open .asd file, starting at the current position.

    Parameters
    ----------
    open_file : BinaryIO
        An open binary file object for a .asd file.
    num_frames : int
        The number of frames for this set of frame data.
    x_pixels : int
        The width of each frame in pixels.
    y_pixels : int
        The height of each frame in pixels.
    analogue_digital_converter : VoltageLevelConverter
        A VoltageLevelConverter instance for converting the raw level values to real world nanometre vertical heights.

    Returns
    -------
    np.ndarray
        The extracted frame heightmap data as a N x W x H 3D numpy array
        (number of frames x width of each frame x height of each frame). Units are nanometres.
    """
    # List to store the frames as numpy arrays
    frames = []
    # Dictionary to store all the variables together in case we want to return them.
    # Very useful for debugging!
    frame_header_dict = {}

    for _ in range(num_frames):
        frame_header_dict["frame_number"] = read_int32(open_file)
        frame_header_dict["max_data"] = read_int16(open_file)
        frame_header_dict["min_data"] = read_int16(open_file)
        frame_header_dict["x_offset"] = read_int16(open_file)
        frame_header_dict["y_offset"] = read_int16(open_file)
        frame_header_dict["x_tilt"] = read_float(open_file)
        frame_header_dict["y_tilt"] = read_float(open_file)
        frame_header_dict["is_stimulated"] = read_bool(open_file)
        _booked_1 = read_int8(open_file)
        _booked_2 = read_int16(open_file)
        _booked_3 = read_int32(open_file)
        _booked_4 = read_int32(open_file)

        frame_header_dict["total_size"] = x_pixels * y_pixels
        # Read frame byte data. Data is always stored as signed 2 byte integer form
        # so multiply the size of the array by 2
        frame_header_dict["total_byte_size"] = frame_header_dict["total_size"] * 2
        frame_data = open_file.read(frame_header_dict["total_size"] * 2)
        # Decode frame data from bytes. Data is always stored in signed 2 byte integer form
        frame_data = np.frombuffer(frame_data, dtype=np.int16)
        # Convert from Voltage to Real units
        frame_data = analogue_digital_converter.level_to_voltage(frame_data)
        # Reshape frame to 2D array
        frame_data = frame_data.reshape((y_pixels, x_pixels))
        frames.append(frame_data)

    return frames


def create_analogue_digital_converter(
    analogue_digital_range: float, scaling_factor: float, resolution: int = 4096
) -> VoltageLevelConverter:
    """
    Create an analogue to digital converter for a given range, scaling factor and resolution.

    Used for converting raw level values into real world height scales in nanometres.

    Parameters
    ----------
    analogue_digital_range : float
        The range of analogue voltage values.
    scaling_factor : float
        A scaling factor calculated elsewhere that scales the heightmap appropriately based on the type of channel and
        sensor parameters.
    resolution : int
        The vertical resolution of the instrumen. Dependant on the number of bits used to store its values. Typically
        12, hence 2^12 = 4096 sensitivity levels.

    Returns
    -------
    VoltageLevelConverter
        An instance of the VoltageLevelConverter class with a tailored function `level_to_voltage`
        which converts arbitrary level values into real world nanometre heights for the given .asd
        file. Note that this is file specific since the parameters will change between files.
    """
    # Analogue to digital hex conversion range encoding:
    # unipolar_1_00V : 0x00000001 +0.00 to +1.00 V
    # unipolar_2_50V : 0x00000002 +0.00 to +2.50 V
    # unipolar_9.99v : 0x00000003 +0.00 to +9.99 V
    # unipolar_5_00V : 0x00000004 +0.00 to +5.00 V
    # bipolar_1_00V  : 0x00010000 -1.00 to +1.00 V
    # bipolar_2_50V  : 0x00020000 -2.50 to +2.50 V
    # bipolar_5_00V  : 0x00040000 -5.00 to +5.00 V

    converter: VoltageLevelConverter

    if analogue_digital_range == hex(0x00000001):
        # unipolar 1.0V
        mapping = (0.0, 1.0)
        converter = UnipolarConverter(
            analogue_digital_range=1.0,
            resolution=resolution,
            scaling_factor=scaling_factor,
        )
    elif analogue_digital_range == hex(0x00000002):
        # unipolar 2.5V
        mapping = (0.0, 2.5)
        converter = UnipolarConverter(
            analogue_digital_range=2.5,
            resolution=resolution,
            scaling_factor=scaling_factor,
        )
    elif analogue_digital_range == hex(0x00000003):
        mapping = (0, 9.99)
        converter = UnipolarConverter(
            analogue_digital_range=9.99,
            resolution=resolution,
            scaling_factor=scaling_factor,
        )
    elif analogue_digital_range == hex(0x00000004):
        # unipolar 5.0V
        mapping = (0.0, 5.0)
        converter = UnipolarConverter(
            analogue_digital_range=5.0,
            resolution=resolution,
            scaling_factor=scaling_factor,
        )
    elif analogue_digital_range == hex(0x00010000):
        # bipolar 1.0V
        mapping = (-1.0, 1.0)
        converter = BipolarConverter(
            analogue_digital_range=1.0,
            resolution=resolution,
            scaling_factor=scaling_factor,
        )
    elif analogue_digital_range == hex(0x00020000):
        # bipolar 2.5V
        mapping = (-2.5, 2.5)
        converter = BipolarConverter(
            analogue_digital_range=2.5,
            resolution=resolution,
            scaling_factor=scaling_factor,
        )
    elif analogue_digital_range == hex(0x00040000):
        # bipolar 5.0V
        mapping = (-5.0, 5.0)
        converter = BipolarConverter(
            analogue_digital_range=5.0,
            resolution=resolution,
            scaling_factor=scaling_factor,
        )
    else:
        raise ValueError(
            f"Analogue to digital range hex value {analogue_digital_range} has no known " "analogue-digital mapping."
        )
    logger.info(f"Analogue to digital mapping | Range: {analogue_digital_range} -> {mapping}")
    logger.info(f"Converter: {converter}")
    return converter


def create_animation(file_name: str, frames: npt.NDArray, file_format: str = ".gif") -> None:
    """
    Create animation from a numpy array of frames (2d numpy arrays).

    File format can be specified, defaults to .gif.

    Parameters
    ----------
    file_name : str
        Name of the file to save.
    frames : npt.NDArray
        Numpy array of frames of shape (N x W x H) where N is the number of frames,
        W is the width of the frames and H is the height of the frames.
    file_format : str
        Optional string for the file format to save as. Formats currently available: .mp4, .gif.
    """
    fig, axis = plt.subplots()

    def update(frame: npt.NDArray):
        """
        Update the image with the latest frame.

        Parameters
        ----------
        frame : npt.NDArray
            Single frame to add to the image.

        Returns
        -------
        axis
            Matplotlib axis.
        """
        axis.imshow(frames[frame])
        return axis

    # Create the animation object
    ani = animation.FuncAnimation(fig, update, frames=frames.shape[0], interval=200)

    if file_format == ".mp4":
        ani.save(f"{file_name}.mp4", writer="ffmpeg")
    elif file_format == ".gif":
        ani.save(f"{file_name}.gif", writer="imagemagick")
    else:
        raise ValueError(f"{file_format} format not supported yet.")
