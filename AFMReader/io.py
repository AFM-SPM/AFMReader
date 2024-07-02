"""For reading and writing data from / to files."""

import io
import struct
from typing import BinaryIO
import h5py
from loguru import logger


def read_uint8(open_file: BinaryIO) -> int:
    """
    Read an unsigned 8 bit integer from an open binary file.

    Parameters
    ----------
    open_file : BinaryIO
        An open binary file object.

    Returns
    -------
    int
        Integer decoded value.
    """
    return int.from_bytes(open_file.read(1), byteorder="little")


def read_int8(open_file: BinaryIO) -> int:
    """
    Read a signed 8 bit integer from an open binary file.

    Parameters
    ----------
    open_file : BinaryIO
        An open binary file object.

    Returns
    -------
    int
        Integer decoded value.
    """
    return struct.unpack("b", open_file.read(1))[0]


def read_int16(open_file: BinaryIO) -> int:
    """
    Read a signed 16 bit integer from an open binary file.

    Parameters
    ----------
    open_file : BinaryIO
        An open binary file object.

    Returns
    -------
    int
        Integer decoded value.
    """
    return struct.unpack("h", open_file.read(2))[0]


def read_int32(open_file: BinaryIO) -> int:
    """
    Read a signed 32 bit integer from an open binary file.

    Parameters
    ----------
    open_file : BinaryIO
        An open binary file object.

    Returns
    -------
    int
        Integer decoded value.
    """
    return struct.unpack("i", open_file.read(4))[0]


def read_uint32(open_file: BinaryIO) -> int:
    """
    Read an unsigned 32 bit integer from an open binary file.

    Parameters
    ----------
    open_file : BinaryIO
        An open binary file object.

    Returns
    -------
    int
        Integer decoded value.
    """
    return struct.unpack("<I", open_file.read(4))[0]


def read_hex_u32(open_file: BinaryIO) -> str:
    """
    Read a hex encoded unsigned 32 bit integer value from an open binary file.

    Parameters
    ----------
    open_file : BinaryIO
        An open binary file object.

    Returns
    -------
    str
        String representing a hexadecimal encoded integer value.
    """
    return hex(struct.unpack("<I", open_file.read(4))[0])


def read_float(open_file: BinaryIO) -> float:
    """
    Read a float from an open binary file.

    Parameters
    ----------
    open_file : BinaryIO
        An open binary file object.

    Returns
    -------
    float
        Float decoded value.
    """
    return struct.unpack("f", open_file.read(4))[0]


def read_bool(open_file: BinaryIO) -> bool:
    """
    Read a boolean from an open binary file.

    Parameters
    ----------
    open_file : BinaryIO
        An open binary file object.

    Returns
    -------
    bool
        Boolean decoded value.
    """
    return bool(int.from_bytes(open_file.read(1), byteorder="little"))


def read_double(open_file: BinaryIO) -> float:
    """
    Read an 8 byte double from an open binary file.

    Parameters
    ----------
    open_file : BinaryIO
        An open binary file object.

    Returns
    -------
    float
        Float decoded from the double value.
    """
    return struct.unpack("d", open_file.read(8))[0]


def read_ascii(open_file: BinaryIO, length_bytes: int = 1) -> str:
    """
    Read an ASCII string of defined length from an open binary file.

    Parameters
    ----------
    open_file : BinaryIO
        An open binary file object.
    length_bytes : int
        Length of the ASCII string in bytes that should be read. Default: 1 byte (1 character).

    Returns
    -------
    str
        ASCII text decoded from file.
    """
    return open_file.read(length_bytes).decode("ascii")


def read_null_separated_utf8(open_file: BinaryIO, length_bytes: int = 2) -> str:
    r"""
    Read an ASCII string of defined length from an open binary file.

    Each character is separated by a null byte. This encoding is known as UTF-16LE (Little Endian).
    Eg: b'\x74\x00\x6f\x00\x70\x00\x6f' would decode to 'topo' in this format.

    Parameters
    ----------
    open_file : BinaryIO
        An open binary file object.
    length_bytes : int
        Length of the ASCII string in bytes that should be read. Default: 2 bytes (1 UTF-16LE character).

    Returns
    -------
    str
        ASCII text decoded from file.
    """
    return open_file.read(length_bytes).replace(b"\x00", b"").decode("ascii")


def skip_bytes(open_file: BinaryIO, length_bytes: int = 1) -> bytes:
    """
    Skip a specified number of bytes when reading an open binary file.

    Parameters
    ----------
    open_file : BinaryIO
        An open binary file object.
    length_bytes : int
        Number of bytes to skip.

    Returns
    -------
    bytes
        The bytes that were skipped.
    """
    return open_file.read(length_bytes)


def unpack_hdf5(open_hdf5_file: h5py.File, group_path: str = "/") -> dict:
    """
    Read a dictionary from an open hdf5 file.

    Parameters
    ----------
    open_hdf5_file : h5py.File
        An open hdf5 file object.
    group_path : str
        Path to the group in the hdf5 file to start reading the data from.

    Returns
    -------
    dict
        Dictionary containing the data from the hdf5 file.

    Examples
    --------
    Read the data from the root group of the hdf5 file.
    >>> with h5py.File("path/to/file.h5", "r") as f:
    >>>     data = unpack_hdf5(open_hdf5_file=f, group_path="/")
    Read data from a particular dataset in the hdf5 file.
    >>> with h5py.File("path/to/file.h5", "r") as f:
    >>>     data = unpack_hdf5(open_hdf5_file=f, group_path="/dataset_name")
    """
    data = {}
    for key, item in open_hdf5_file[group_path].items():
        if isinstance(item, h5py.Group):
            # Incur recursion for nested groups
            data[key] = unpack_hdf5(open_hdf5_file, f"{group_path}/{key}")
        # Decode byte strings to utf-8. The data type "O" is a byte string.
        elif isinstance(item, h5py.Dataset) and item.dtype == "O":
            # Byte string
            data[key] = item[()].decode("utf-8")
        else:
            # Another type of dataset
            data[key] = item[()]
    return data


def read_null_terminated_string(open_file: io.TextIOWrapper, encoding: str = "utf-8") -> str:
    """
    Read an open file from the current position in the open binary file, until the next null value.

    Parameters
    ----------
    open_file : io.TextIOWrapper
        An open file object.
    encoding : str
        Encoding to use when decoding the bytes.

    Returns
    -------
    str
        String of the ASCII decoded bytes before the next null byte.

    Examples
    --------
    >>> with open("test.txt", "rb") as f:
    ...     print(read_null_terminated_string(f), encoding="utf-8")
    """
    byte = open_file.read(1)
    value = b""
    while byte != b"\x00":
        value += byte
        byte = open_file.read(1)
    # Sometimes encodings cannot decode a byte that is not defined in the encoding.
    # Try 'latin1' in this case as it is able to handle symbols such as micro (µ).
    try:
        return str(value.decode(encoding=encoding))
    except UnicodeDecodeError as e:
        if "codec can't decode byte" in str(e):
            bad_byte = str(e).split("byte ")[1].split(":")[0]
            logger.debug(
                f"Decoding error while reading null terminated string. Encoding {encoding} encountered"
                f" a byte that could not be decoded: {bad_byte}. Trying 'latin1' encoding."
            )
            return str(value.decode(encoding="latin1"))
        raise e


def read_char(open_file: io.TextIOWrapper) -> str:
    """
    Read a character from an open binary file.

    Parameters
    ----------
    open_file : io.TextIOWrapper
        An open file object.

    Returns
    -------
    str
        A string type cast from the decoded character.
    """
    return open_file.read(1).decode("ascii")
