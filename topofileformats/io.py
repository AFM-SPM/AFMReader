"""For reading and writing data from / to files."""

import struct
from typing import BinaryIO


def read_uint8(open_file: BinaryIO) -> int:
    """Read an unsigned 8 bit integer from an open binary file.

    Parameters
    ----------
    open_file: BinaryIO
        An open binary file object.

    Returns
    -------
    int
        Integer decoded value.
    """
    return int.from_bytes(open_file.read(1), byteorder="little")


def read_int8(open_file: BinaryIO) -> int:
    """Read a signed 8 bit integer from an open binary file.

    Parameters
    ----------
    open_file: BinaryIO
        An open binary file object.

    Returns
    -------
    int
        Integer decoded value.
    """
    return struct.unpack("b", open_file.read(1))[0]


def read_int16(open_file: BinaryIO) -> int:
    """Read a signed 16 bit integer from an open binary file.

    Parameters
    ----------
    open_file: BinaryIO
        An open binary file object.

    Returns
    -------
    int
        Integer decoded value.
    """
    return struct.unpack("h", open_file.read(2))[0]


def read_int32(open_file: BinaryIO) -> int:
    """Read a signed 32 bit integer from an open binary file.

    Parameters
    ----------
    open_file: BinaryIO
        An open binary file object.

    Returns
    -------
    int
        Integer decoded value.
    """
    return struct.unpack("i", open_file.read(4))[0]


def read_uint32(open_file: BinaryIO) -> int:
    """Read an unsigned 32 bit integer from an open binary file.

    Parameters
    ----------
    open_file: BinaryIO
        An open binary file object.

    Returns
    -------
    int
        Integer decoded value.
    """
    return struct.unpack("<I", open_file.read(4))[0]


def read_hex_u32(open_file: BinaryIO) -> str:
    """Read a hex encoded unsigned 32 bit integer value from an open binary file.

    Parameters
    ----------
    open_file: BinaryIO
        An open binary file object.

    Returns
    -------
    str
        String representing a hexadecimal encoded integer value.
    """
    return hex(struct.unpack("<I", open_file.read(4))[0])


def read_float(open_file: BinaryIO) -> float:
    """Read a float from an open binary file.

    Parameters
    ----------
    open_file: BinaryIO
        An open binary file object.

    Returns
    -------
    float
    Float decoded value.
    """
    return struct.unpack("f", open_file.read(4))[0]


def read_bool(open_file: BinaryIO) -> bool:
    """Read a boolean from an open binary file.

    Parameters
    ----------
    open_file: BinaryIO
        An open binary file object.

    Returns
    -------
    bool
        Boolean decoded value
    """
    return bool(int.from_bytes(open_file.read(1), byteorder="little"))


def read_double(open_file: BinaryIO) -> float:
    """Read an 8 byte double from an open binary file.

    Parameters
    ----------
    open_file: BinaryIO
        An open binary file object.

    Returns
    -------
    float
        Float decoded from the double value
    """
    return struct.unpack("d", open_file.read(8))[0]


def read_ascii(open_file: BinaryIO, length_bytes: int = 1) -> str:
    """Read an ascii string of defined length from an open binary file.

    Parameters
    ----------
    open_file: BinaryIO
        An open binary file object.
    length_bytes: int
        Length of the ascii string in bytes that should be read. Default: 1 byte (1 character).

    Returns
    -------
    str
        Ascii text decoded from file.
    """
    return open_file.read(length_bytes).decode("ascii")


def read_null_separated_utf8(open_file: BinaryIO, length_bytes: int = 2) -> str:
    r"""Read an ASCII string of defined length from an open binary file.

    Each character is separated by a null byte. This encoding is known as UTF-16LE (Little Endian).
    Eg: b'\x74\x00\x6f\x00\x70\x00\x6f' would decode to 'topo' in this format.

    Parameters
    ----------
    open_file: BinaryIO
        An open binary file object.
    length_bytes: int
        Length of the ascii string in bytes that should be read. Default: 2 bytes
        (1 UTF-16LE character)

    Returns
    -------
    str
        Ascii text decoded from file.
    """
    return open_file.read(length_bytes).replace(b"\x00", b"").decode("ascii")


def skip_bytes(open_file: BinaryIO, length_bytes: int = 1) -> bytes:
    """Skip a specified number of bytes when reading an open binary file.

    Parameters
    ----------
    open_file: BinaryIO
        An open binary file object.
    length_bytes: int
        Number of bytes to skip.

    Returns
    -------
    bytes
        The bytes that were skipped.
    """
    return open_file.read(length_bytes)
