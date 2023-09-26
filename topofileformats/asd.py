"""For decoding and loading .asd AFM file format into Python Numpy arrays"""
from topofileformats.io import (
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
    read_uint32,
    skip_bytes,
)

from pathlib import Path
from typing import BinaryIO, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class VoltageLevelConverter:
    def __init__(self, analogue_digital_range, max_voltage, scaling_factor, resolution):
        self.ad_range = int(analogue_digital_range, 16)
        self.max_voltage = max_voltage
        self.scaling_factor = scaling_factor
        self.resolution = resolution
        print(
            f"created voltage converter. ad_range: {analogue_digital_range} -> {self.ad_range}, max voltage: {max_voltage}, scaling factor: {scaling_factor}, resolution: {resolution}"
        )


class UnipolarConverter(VoltageLevelConverter):
    def level_to_voltage(self, level):
        return (self.ad_range * level / self.resolution) * self.scaling_factor


class BipolarConverter(VoltageLevelConverter):
    def level_to_voltage(self, level):
        return (self.ad_range - 2 * level * self.ad_range / self.resolution) * self.scaling_factor


def calculate_scaling_factor(channel, z_piezo_gain, z_piezo_extension, scanner_sensitivity, phase_sensitivity):
    if channel == "TP":
        print(
            f"Scaling factor: Type: {channel} -> TP | piezo extension {z_piezo_gain} * piezo gain {z_piezo_extension} = scaling factor {z_piezo_gain * z_piezo_extension}"
        )
        return z_piezo_gain * z_piezo_extension
    elif channel == "ER":
        print(
            f"Scaling factor: Type: {channel} -> ER | - scanner sensitivity {-scanner_sensitivity} = scaling factor {-scanner_sensitivity}"
        )
        return -scanner_sensitivity
    elif channel == "PH":
        print(
            f"Scaling factor: Type: {channel} -> PH | - phase sensitivity {-phase_sensitivity} = scaling factor {-phase_sensitivity}"
        )
        return -phase_sensitivity
    else:
        raise ValueError(f"channel {channel} not known for .asd file type.")


def load_asd(file_path: Union[Path, str], channel: str):
    with open(file_path, "rb") as f:
        file_version = read_file_version(f)

        if file_version == 0:
            header_dict = read_header_file_version_0(f)

        elif file_version == 1:
            header_dict = read_header_file_version_1(f)

        elif file_version == 2:
            header_dict = read_header_file_version_2(f)
        else:
            raise ValueError(
                f"File version {file_version} unknown. Please add support if you know how to decode this file version."
            )
        print(header_dict)

        pixel_to_nanometre_scaling_factor_x = header_dict["x_nm"] / header_dict["x_pixels"]
        pixel_to_nanometre_scaling_factor_y = header_dict["y_nm"] / header_dict["y_pixels"]
        if pixel_to_nanometre_scaling_factor_x != pixel_to_nanometre_scaling_factor_y:
            print(
                f"WARNING: Resolution of image is different in x and y directions: x: {pixel_to_nanometre_scaling_factor_x}\
 y: {pixel_to_nanometre_scaling_factor_y}"
            )
        pixel_to_nanometre_scaling_factor = pixel_to_nanometre_scaling_factor_x

        if channel == header_dict["channel1"]:
            print(f"Requested channel {channel} matches first channel in file: {header_dict['channel1']}")
        elif channel == header_dict["channel2"]:
            print(f"Requested channel {channel} matches second channel in file: {header_dict['channel2']}")

            # Skip first channel data
            size_of_frame_header = header_dict["frame_header_length"]
            # Remember that each value is two bytes (since signed int16)
            size_of_single_frame_plus_header = (
                header_dict["frame_header_length"] + header_dict["x_pixels"] * header_dict["y_pixels"] * 2
            )
            length_of_all_first_channel_frames = header_dict["num_frames"] * size_of_single_frame_plus_header
            _ = f.read(length_of_all_first_channel_frames)
        else:
            raise ValueError(
                f"Channel {channel} not found in this file's available channels: {header_dict['channel1']}, {header_dict['channel2']}"
            )

        scaling_factor = calculate_scaling_factor(
            channel=channel,
            z_piezo_gain=header_dict["z_piezo_gain"],
            z_piezo_extension=header_dict["z_piezo_extension"],
            scanner_sensitivity=header_dict["scanner_sensitivity"],
            phase_sensitivity=header_dict["phase_sensitivity"],
        )
        analogue_digital_converter = create_analogue_digital_converter(
            analogue_digital_range=header_dict["analogue_digital_range"], scaling_factor=scaling_factor
        )
        frames = read_channel_data(
            f=f,
            num_frames=header_dict["num_frames"],
            x_pixels=header_dict["x_pixels"],
            y_pixels=header_dict["y_pixels"],
            analogue_digital_converter=analogue_digital_converter,
        )

        frames = np.array(frames)

        return frames, pixel_to_nanometre_scaling_factor, header_dict


def read_file_version(open_file):
    """Read the file version from an open asd file. File versions are 0, 1 and 2.
    Different file versions require different functions to read the headers as
    the formatting changes between them.

    Parameters
    ----------
    open_file: BinaryIO
        An open binary file object for a .asd file.

    Returns
    -------
    int
        Integer file version decoded from file.
    """
    file_version = read_int32(open_file)
    print(f"file version: {file_version}")
    return file_version


def read_header_file_version_0(f):
    header_dict = {}

    # There only ever seem to be two channels available
    # Channel encoding are all in LITTLE ENDIAN format.
    # topology: 0x5054 decodes to 'TP' in ascii little endian
    # error: 0x5245 decodes to 'ER' in ascii little endian
    # phase: 0x4850 decodes to 'PH' in ascii little endian
    header_dict["channel1"] = read_ascii(f, 2)
    header_dict["channel2"] = read_ascii(f, 2)
    # length of file metadata header in bytes - so we can skip it to get to the data
    header_dict["header_length"] = read_int32(f)
    # Frame header is the length of the header for each frame to be skipped before reading frame data.
    header_dict["frame_header_length"] = read_int32(f)
    # Length in bytes of the name given in the file
    header_dict["user_name_size"] = read_int32(f)
    header_dict["comment_offset_size"] = read_int32(f)
    # Length in bytes of the comment for the file
    header_dict["comment_size"] = read_int32(f)
    # x and y resolution (pixels)
    header_dict["x_pixels"] = read_int16(f)
    header_dict["y_pixels"] = read_int16(f)
    # x and y resolution (nm)
    header_dict["x_nm"] = read_int16(f)
    header_dict["y_nm"] = read_int16(f)
    # frame time
    header_dict["frame_time"] = read_float(f)
    # z piezo extension
    header_dict["z_piezo_extension"] = read_float(f)
    # z piezo gain
    header_dict["z_piezo_gain"] = read_float(f)
    # Range of analogue voltage values (for conversion to digital)
    header_dict["analogue_digital_range"] = read_hex_u32(f)
    # Number of bits of data for analogue voltage values (for conversion to digital)
    # aka the resolution of the instrument. Usually 12 bits, so 4096 sensitivity levels
    header_dict["analogue_digital_data_bits_size"] = read_int32(f)
    header_dict["analogue_digital_resolution"] = 2 ^ header_dict["analogue_digital_data_bits_size"]
    # Not sure, something to do with data averaging
    header_dict["is_averaged"] = read_bool(f)
    # Window for averaging the data
    header_dict["averaging_window"] = read_int32(f)
    # Some padding to ensure backwards compatilibilty I think
    _ = read_int16(f)
    # Date of creation
    header_dict["year"] = read_int16(f)
    header_dict["month"] = read_uint8(f)
    header_dict["day"] = read_uint8(f)
    header_dict["hour"] = read_uint8(f)
    header_dict["minute"] = read_uint8(f)
    header_dict["second"] = read_uint8(f)
    # Rounding degree?
    header_dict["rounding_degree"] = read_uint8(f)
    # Maximum x and y scanning range in real space, nm
    header_dict["max_x_scan_range"] = read_float(f)
    header_dict["max_y_scan_range"] = read_float(f)
    # No idea
    _ = read_int32(f)
    _ = read_int32(f)
    _ = read_int32(f)
    # Number of frames the file had when recorded
    header_dict["initial_frames"] = read_int32(f)
    # Actual number of frames
    header_dict["num_frames"] = read_int32(f)
    # ID of the AFM instrument
    header_dict["afm_id"] = read_int32(f)
    # ID of the file
    header_dict["file_id"] = read_int16(f)
    # Name of the user
    header_dict["user_name"] = read_null_separated_utf8(f, length_bytes=header_dict["user_name_size"])
    # Sensitivity of the scanner in nm / V
    header_dict["scanner_sensitivity"] = read_float(f)
    # Phase sensitivity
    header_dict["phase_sensitivity"] = read_float(f)
    # Direction of the scan
    header_dict["scan_direction"] = read_int32(f)
    # Skip bytes: comment offset size
    _ = skip_bytes(f, header_dict["comment_offset_size"])
    # Read a comment
    comment = []
    for i in range(header_dict["comment_size"]):
        comment.append(chr(read_int8(f)))
    header_dict["comment_without_null"] = "".join([c for c in comment if c != "\x00"])

    return header_dict


def read_header_file_version_1(f):
    header_dict = {}

    # length of file metadata header in bytes - so we can skip it to get to the data
    header_dict["header_length"] = read_int32(f)
    # Frame header is the length of the header for each frame to be skipped before reading frame data.
    header_dict["frame_header_length"] = read_int32(f)
    # Encoding for strings
    header_dict["text_encoding"] = read_int32(f)
    # Length in bytes of the name given in the file
    header_dict["user_name_size"] = read_int32(f)
    # Length in bytes of the comment for the file
    header_dict["comment_size"] = read_int32(f)
    # There only ever seem to be two channels available
    # Channel encoding are all in LITTLE ENDIAN format.
    # topology: 0x5054 decodes to 'TP' in ascii little endian
    # error: 0x5245 decodes to 'ER' in ascii little endian
    # phase: 0x4850 decodes to 'PH' in ascii little endian
    header_dict["channel1"] = read_null_separated_utf8(f, length_bytes=4)
    header_dict["channel2"] = read_null_separated_utf8(f, length_bytes=4)
    # Number of frames the file had when recorded
    header_dict["initial_frames"] = read_int32(f)
    # Actual number of frames
    header_dict["num_frames"] = read_int32(f)
    # Direction of the scan
    header_dict["scan_direction"] = read_int32(f)
    # ID of the file
    header_dict["file_id"] = read_int32(f)
    # x and y resolution (pixels)
    header_dict["x_pixels"] = read_int32(f)
    header_dict["y_pixels"] = read_int32(f)
    # x and y resolution (nm)
    header_dict["x_nm"] = read_int32(f)
    header_dict["y_nm"] = read_int32(f)
    # Not sure, something to do with data averaging
    header_dict["is_averaged"] = read_bool(f)
    # Window for averaging the data
    header_dict["averaging_window"] = read_int32(f)
    # Date of creation
    header_dict["year"] = read_int32(f)
    header_dict["month"] = read_int32(f)
    header_dict["day"] = read_int32(f)
    header_dict["hour"] = read_int32(f)
    header_dict["minute"] = read_int32(f)
    header_dict["second"] = read_int32(f)
    # Rounding degree?
    header_dict["x_rounding_degree"] = read_int32(f)
    header_dict["y_rounding_degree"] = read_int32(f)
    # frame time
    header_dict["frame_time"] = read_float(f)
    # Sensitivity of the scanner in nm / V
    header_dict["scanner_sensitivity"] = read_float(f)
    # Phase sensitivity
    header_dict["phase_sensitivity"] = read_float(f)
    # Offset?
    header_dict["offset"] = read_int32(f)
    # Ignore 12 bytes
    _ = skip_bytes(f, 12)
    # ID of the AFM instrument
    header_dict["afm_id"] = read_int32(f)
    # Range of analogue voltage values (for conversion to digital)
    header_dict["analogue_digital_range"] = read_hex_u32(f)
    # Number of bits of data for analogue voltage values (for conversion to digital)
    # aka the resolution of the instrument. Usually 12 bits, so 4096 sensitivity levels
    header_dict["analogue_digital_data_bits_size"] = read_int32(f)
    header_dict["analogue_digital_resolution"] = 2 ^ header_dict["analogue_digital_data_bits_size"]
    # Maximum x and y scanning range in real space, nm
    header_dict["max_x_scan_range"] = read_float(f)
    header_dict["max_y_scan_range"] = read_float(f)
    # Piezo extensions
    header_dict["x_piezo_extension"] = read_float(f)
    header_dict["y_piezo_extension"] = read_float(f)
    header_dict["z_piezo_extension"] = read_float(f)
    # Piezo gain
    header_dict["z_piezo_gain"] = read_float(f)

    # Read the user name
    user_name = []
    for i in range(header_dict["user_name_size"]):
        user_name.append(chr(read_int8(f)))
    header_dict["user_name"] = "".join([c for c in user_name if c != "\x00"])

    # Read a comment
    comment = []
    for i in range(header_dict["comment_size"]):
        comment.append(chr(read_int8(f)))
    header_dict["comment_without_null"] = "".join([c for c in comment if c != "\x00"])

    return header_dict


def read_header_file_version_2(f):
    header_dict = {}

    # length of file metadata header in bytes - so we can skip it to get to the data
    header_dict["header_length"] = read_int32(f)
    # Frame header is the length of the header for each frame to be skipped before reading frame data.
    header_dict["frame_header_length"] = read_int32(f)
    # Encoding for strings
    header_dict["text_encoding"] = read_int32(f)
    # Length in bytes of the name given in the file
    header_dict["user_name_size"] = read_int32(f)
    # Length in bytes of the comment for the file
    header_dict["comment_size"] = read_int32(f)
    # There only ever seem to be two channels available
    # Channel encoding are all in LITTLE ENDIAN format.
    # topology: 0x5054 decodes to 'TP' in ascii little endian
    # error: 0x5245 decodes to 'ER' in ascii little endian
    # phase: 0x4850 decodes to 'PH' in ascii little endian
    header_dict["channel1"] = read_null_separated_utf8(f, length_bytes=4)
    header_dict["channel2"] = read_null_separated_utf8(f, length_bytes=4)
    # Number of frames the file had when recorded
    header_dict["initial_frames"] = read_int32(f)
    # Actual number of frames
    header_dict["num_frames"] = read_int32(f)
    # Direction of the scan
    header_dict["scan_direction"] = read_int32(f)
    # ID of the file
    header_dict["file_id"] = read_int32(f)
    # x and y resolution (pixels)
    header_dict["x_pixels"] = read_int32(f)
    header_dict["y_pixels"] = read_int32(f)
    # x and y resolution (nm)
    header_dict["x_nm"] = read_int32(f)
    header_dict["y_nm"] = read_int32(f)
    # Not sure, something to do with data averaging
    header_dict["is_averaged"] = read_bool(f)
    # Window for averaging the data
    header_dict["averaging_window"] = read_int32(f)
    # Date of creation
    header_dict["year"] = read_int32(f)
    header_dict["month"] = read_int32(f)
    header_dict["day"] = read_int32(f)
    header_dict["hour"] = read_int32(f)
    header_dict["minute"] = read_int32(f)
    header_dict["second"] = read_int32(f)
    # Rounding degree?
    header_dict["x_rounding_degree"] = read_int32(f)
    header_dict["y_rounding_degree"] = read_int32(f)
    # frame time
    header_dict["frame_time"] = read_float(f)
    # Sensitivity of the scanner in nm / V
    header_dict["scanner_sensitivity"] = read_float(f)
    # Phase sensitivity
    header_dict["phase_sensitivity"] = read_float(f)
    # Offset?
    header_dict["offset"] = read_int32(f)
    # Ignore 12 bytes
    _ = skip_bytes(f, 12)
    # ID of the AFM instrument
    header_dict["afm_id"] = read_int32(f)
    # Range of analogue voltage values (for conversion to digital)
    header_dict["analogue_digital_range"] = read_hex_u32(f)
    # Number of bits of data for analogue voltage values (for conversion to digital)
    # aka the resolution of the instrument. Usually 12 bits, so 4096 sensitivity levels
    header_dict["analogue_digital_data_bits_size"] = read_int32(f)
    header_dict["analogue_digital_resolution"] = 2 ^ header_dict["analogue_digital_data_bits_size"]
    # Maximum x and y scanning range in real space, nm
    header_dict["max_x_scan_range"] = read_float(f)
    header_dict["max_y_scan_range"] = read_float(f)
    # Piezo extensions
    header_dict["x_piezo_extension"] = read_float(f)
    header_dict["y_piezo_extension"] = read_float(f)
    header_dict["z_piezo_extension"] = read_float(f)
    # Piezo gain
    header_dict["z_piezo_gain"] = read_float(f)

    # Read the user name
    user_name = []
    for i in range(header_dict["user_name_size"]):
        user_name.append(chr(read_int8(f)))
    header_dict["user_name"] = "".join([c for c in user_name if c != "\x00"])

    # Read a comment
    comment = []
    for i in range(header_dict["comment_size"]):
        comment.append(chr(read_int8(f)))
    header_dict["comment_without_null"] = "".join([c for c in comment if c != "\x00"])

    # No idea why this file type has the number of frames again. Storing it just in case.
    header_dict["number_of_frames"] = read_int32(f)
    # Feed forward parameter, no idea what it does.
    header_dict["is_x_feed_forward_integer"] = read_int32(f)
    # Feed forward parameter, no idea what it does.
    header_dict["is_x_feed_forward_double"] = read_double(f)
    # Minimum and maximum colour mapping values
    header_dict["max_colour_scale"] = read_int32(f)
    header_dict["min_colour_scale"] = read_int32(f)
    # RGB anchor point array sizes
    header_dict["length_red_anchor_points"] = read_int32(f)
    header_dict["length_green_anchor_points"] = read_int32(f)
    header_dict["length_blue_anchor_points"] = read_int32(f)
    # Coords of anchor points
    # Red
    coords_red = []
    for i in range(header_dict["length_red_anchor_points"]):
        x = read_int32(f)
        y = read_int32(f)
        coords_red.append((x, y))
    # Green
    coords_green = []
    for i in range(header_dict["length_green_anchor_points"]):
        x = read_int32(f)
        y = read_int32(f)
        coords_green.append((x, y))
    # Blue
    coords_blue = []
    for i in range(header_dict["length_blue_anchor_points"]):
        x = read_int32(f)
        y = read_int32(f)
        coords_blue.append((x, y))

    return header_dict


def read_channel_data(f, num_frames, x_pixels, y_pixels, analogue_digital_converter):
    # List to store the frames as numpy arrays
    frames = []
    # Dictionary to store all the variables together in case we want to return them.
    # Very useful for debugging!
    frame_header_dict = {}

    for i in range(num_frames):
        frame_header_dict["frame_number"] = read_int32(f)
        frame_header_dict["max_data"] = read_int16(f)
        frame_header_dict["min_data"] = read_int16(f)
        frame_header_dict["x_offset"] = read_int16(f)
        frame_header_dict["y_offset"] = read_int16(f)
        frame_header_dict["x_tilt"] = read_float(f)
        frame_header_dict["y_tilt"] = read_float(f)
        frame_header_dict["is_stimulated"] = read_bool(f)
        _booked_1 = read_int8(f)
        _booked_2 = read_int16(f)
        _booked_3 = read_int32(f)
        _booked_4 = read_int32(f)

        frame_header_dict["total_size"] = x_pixels * y_pixels
        # Read frame byte data. Data is always stored as signed 2 byte integer form
        # so multiply the size of the array by 2
        frame_header_dict["total_byte_size"] = frame_header_dict["total_size"] * 2
        frame_data = f.read(frame_header_dict["total_size"] * 2)
        # Decode frame data from bytes. Data is always stored in signed 2 byte integer form
        frame_data = np.frombuffer(frame_data, dtype=np.int16)
        # Convert from Voltage to Real units
        frame_data = analogue_digital_converter.level_to_voltage(frame_data)
        # Reshape frame to 2D array
        frame_data = frame_data.reshape((y_pixels, x_pixels))
        frames.append(frame_data)

    return frames


def create_analogue_digital_converter(analogue_digital_range, scaling_factor, resolution=4096):
    # Analogue to digital hex conversion range encoding:
    # unipolar_1_0V : 0x00000001 +0.0 to +1.0 V
    # unipolar_2_5V : 0x00000002 +0.0 to +2.5 V
    # unipolar_5_0V : 0x00000004 +0.0 to +5.0 V
    # bipolar_1_0V  : 0x00010000 -1.0 to +1.0 V
    # bipolar_2_5V  : 0x00020000 -2.5 to +2.5 V
    # bipolar_5_0V  : 0x00040000 -5.0 to +5.0 V

    if analogue_digital_range == hex(0x00000001):
        # unipolar 1.0V
        mapping = (0.0, 1.0)
        converter = UnipolarConverter(
            analogue_digital_range=analogue_digital_range,
            max_voltage=1.0,
            resolution=resolution,
            scaling_factor=scaling_factor,
        )
    elif analogue_digital_range == hex(0x00000002):
        # unipolar 2.5V
        mapping = (0.0, 2.5)
        converter = UnipolarConverter(
            analogue_digital_range=analogue_digital_range,
            max_voltage=2.0,
            resolution=resolution,
            scaling_factor=scaling_factor,
        )
    elif analogue_digital_range == hex(0x00000004):
        # unipolar 5.0V
        mapping = (0.0, 5.0)
        converter = UnipolarConverter(
            analogue_digital_range=analogue_digital_range,
            max_voltage=5.0,
            resolution=resolution,
            scaling_factor=scaling_factor,
        )
    elif analogue_digital_range == hex(0x00010000):
        # bipolar 1.0V
        mapping = (-1.0, 1.0)
        converter = BipolarConverter(
            analogue_digital_range=analogue_digital_range,
            max_voltage=1.0,
            resolution=resolution,
            scaling_factor=scaling_factor,
        )
    elif analogue_digital_range == hex(0x00020000):
        # bipolar 2.5V
        mapping = (-2.5, 2.5)
        converter = BipolarConverter(
            analogue_digital_range=analogue_digital_range,
            max_voltage=2.0,
            resolution=resolution,
            scaling_factor=scaling_factor,
        )
    elif analogue_digital_range == hex(0x00040000):
        # bipolar 5.0V
        mapping = (-5.0, 5.0)
        converter = BipolarConverter(
            analogue_digital_range=analogue_digital_range,
            max_voltage=5.0,
            resolution=resolution,
            scaling_factor=scaling_factor,
        )
    else:
        raise ValueError(
            f"Analogue to digital range hex value {analogue_digital_range} has no known analogue-digital mapping."
        )
    print(f"Analogue to digital mapping | Range: {analogue_digital_range} -> {mapping}")
    print(f"Converter: {converter}")
    return converter


def create_animation(file_name: str, frames: np.ndarray, file_format: str = ".gif") -> None:
    """Create animation from a numpy array of frames (2d numpy arrays). File format can be specified, defaults to .gif.

    Parameters
    ----------
    file_name: str
        Name of the file to save
    frames: np.ndarray
        Numpy array of frames of shape (N x W x H) where N is the number of frames, W is the width of the frames and 
        H is the height of the frames.
    file_format: str
        Optional string for the file format to save as. Formats currently available: .mp4, .gif.
    """
    fig, ax = plt.subplots()

    def update(frame):
        ax.imshow(frames[frame])
        return ax

    # Create the animation object
    ani = animation.FuncAnimation(fig, update, frames=frames.shape[0], interval=200)

    if file_format == ".mp4":
        ani.save(f"{file_name}.mp4", writer="ffmpeg")
    elif file_format == ".gif":
        ani.save(f"{file_name}.gif", writer="imagemagick")
    else:
        raise ValueError(f"{file_format} format not supported yet.")
