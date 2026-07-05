"""A module for parsing ARDF files."""

# Copyright (C) 2026 TopoStats Team
# Copyright (C) Richard J. Sheridan
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# See <https://www.gnu.org/licenses/> for a copy of the GNU General Public License.
#
# This file includes code copied and adapted from Magic AFM
# (https://github.com/richardsheridan/magic-afm).

import mmap
import struct
from pathlib import Path
from bisect import bisect_left
from collections.abc import Collection, Iterable
from typing import TypeAlias, Any

from tqdm import tqdm
from AFMReader.data_classes import AFMLoad, CurvesMetadata, CurvesVolume, CurvesDataset
from AFMReader.h5_saver import H5Saver, find_unused_filename
from AFMReader.io import coerce_metadata_dict, load_config
from AFMReader.logging import logger

try:
    from collections.abc import Buffer
except ImportError:
    Buffer: TypeAlias = mmap.mmap | memoryview | bytes | bytearray

import numpy as np
from attrs import frozen, field

NANOMETER_UNIT_CONVERSION = 1e9  # maybe we can intelligently read this from the file someday
NANCURVE = np.full(shape=(2, 2, 100), fill_value=np.nan, dtype=np.float32)
NANCURVE.setflags(write=False)
TOC_STRUCT = struct.Struct("<QLL")

# ###############################################
# ############### Typing stuff ##################
# ###############################################


Index: TypeAlias = tuple[int, ...]
ZDArrays: TypeAlias = Collection[np.ndarray]
ChanMap: TypeAlias = dict[str, tuple[int, "ARDFVchan"]]
StepInfo: TypeAlias = tuple[tuple[float, str], ...]


# ###############################################
# ################## Helpers ####################
# ###############################################


def mmap_path_read_only(path: Any) -> mmap.mmap:
    """
    Create a read-only memory map of the file.

    Parameters
    ----------
    path : Any
        The path of the file to memory map.

    Returns
    -------
    mmap.mmap
        The memory map of the file.
    """
    with Path(path).open(mode="rb", buffering=0) as file:
        return mmap.mmap(file.fileno(), length=0, access=mmap.ACCESS_READ)


def decode_cstring(cstring: bytes) -> str:
    """
    Decode a null-terminated C string.

    Parameters
    ----------
    cstring : bytes
        The byte string to decode.

    Returns
    -------
    str
        The decoded string.
    """
    return cstring.rstrip(b"\0").decode("windows-1252")


def parse_ar_note(note: Iterable[str]) -> dict[str, Any]:
    """
    Parse keys and values from notes.

    Parameters
    ----------
    note : Iterable[str]
        The lines of notes to parse.

    Returns
    -------
    dict[str, Any]
        A dictionary mapping keys to values.
    """
    # The notes have a very regular key-value structure
    # convert to dict for later access
    return coerce_metadata_dict(dict(line.split(":", 1) for line in note if ":" in line and "@Line:" not in line))


@frozen
class ARDFHeader:
    """
    Header of an ARDF section.

    This class parses and validates section headers in ARDF files.
    """

    data: Buffer
    offset: int
    crc: int = field(repr=hex)
    size: int
    name: bytes
    flags: int = field(repr=hex)
    _struct = struct.Struct("<LL4sL")

    @classmethod
    def unpack(cls, data: Buffer, offset: int) -> "ARDFHeader":
        """
        Unpack a header from the buffer.

        Parameters
        ----------
        data : Buffer
            The data buffer.
        offset : int
            The offset to unpack from.

        Returns
        -------
        ARDFHeader
            The unpacked header.
        """
        return cls(data, offset, *cls._struct.unpack_from(data, offset))

    def validate(self) -> bool:
        """
        Validate the section CRC.

        Returns
        -------
        bool
            True if validation succeeds.
        """
        # ImHex poly 0x4c11db7 init 0xffffffff xor out 0xffffffff reflect in and out
        import zlib

        crc = zlib.crc32(memoryview(self.data)[self.offset + 4 : self.offset + self.size])
        if self.crc != crc:
            raise ValueError(f"Invalid section. Expected {self.crc:X}, got {crc:X}.", self)
        return True


@frozen
class ARDFTableOfContents:
    """
    Table of contents of an ARDF file.

    Represents a TOC section that references multiple headers.
    """

    data: Buffer
    offset: int
    size: int
    entries: list[tuple[ARDFHeader, int]]
    _entry_struct = struct.Struct("<LL4sLQ")

    @classmethod
    def unpack(cls, header: ARDFHeader) -> "ARDFTableOfContents":
        """
        Unpack the table of contents.

        Parameters
        ----------
        header : ARDFHeader
            The TOC section header.

        Returns
        -------
        ARDFTableOfContents
            The unpacked table of contents.
        """
        if header.size != 32:
            raise ValueError("Malformed table of contents", header)
        header.validate()
        data = header.data
        offset = header.offset
        size, nentries, stride = TOC_STRUCT.unpack_from(data, offset + 16)
        assert stride == 24
        assert size - 32 == nentries * stride, (size, nentries, stride)
        entries = []
        for toc_offset in range(offset + 32, offset + size, stride):
            *header, pointer = cls._entry_struct.unpack_from(data, toc_offset)
            if not pointer:
                break  # rest is null padding
            entry_header = ARDFHeader(data, toc_offset, *header)
            if entry_header.name not in {b"IMAG", b"VOLM", b"NEXT", b"THMB", b"NSET"}:
                raise ValueError("Malformed table of contents.", entry_header)
            entry_header.validate()
            entries.append((entry_header, pointer))
        return cls(data, offset, size, entries)


@frozen
class ARDFTextTableOfContents:
    """
    Text table of contents section.

    Contains pointers to text sections in an ARDF file.
    """

    data: Buffer
    offset: int
    size: int
    entries: list[tuple[ARDFHeader, int]]
    _entry_struct = struct.Struct("<LL4sLQQ")

    @classmethod
    def unpack(cls, header: ARDFHeader) -> "ARDFTextTableOfContents":
        """
        Unpack the text table of contents.

        Parameters
        ----------
        header : ARDFHeader
            The TTOC section header.

        Returns
        -------
        ARDFTextTableOfContents
            The unpacked text table of contents.
        """
        if header.size != 32 or header.name != b"TTOC":
            raise ValueError("Malformed text table of contents.", header)
        header.validate()
        offset = header.offset
        data = header.data
        size, nentries, stride = TOC_STRUCT.unpack_from(data, offset + 16)
        assert stride == 32, stride
        assert size - 32 == nentries * stride, (size, nentries, stride)
        entries = []
        for toc_offset in range(offset + 32, offset + size, stride):
            *header, _, pointer = cls._entry_struct.unpack_from(data, toc_offset)
            if not pointer:
                break  # rest is null padding
            entry_header = ARDFHeader(data, toc_offset, *header)
            if entry_header.name != b"TOFF":
                raise ValueError("Malformed text table entry.", entry_header)
            entry_header.validate()
            entries.append((entry_header, pointer))
        return cls(data, offset, size, entries)

    def decode_entry(self, index: int) -> str:
        """
        Decode a text entry by its index.

        Parameters
        ----------
        index : int
            The index of the entry to decode.

        Returns
        -------
        str
            The decoded text.
        """
        # maybe could be another TEXT class but we'll read it straight in
        entry_header, pointer = self.entries[index]
        text_header = ARDFHeader.unpack(self.data, pointer)
        if text_header.name != b"TEXT":
            raise ValueError("Malformed text section.", text_header)
        text_header.validate()
        i, text_len = struct.unpack_from("<LL", self.data, pointer + 16)
        assert i == index, (i, index)
        offset = text_header.offset + 24
        assert text_len < text_header.size - 24, (text_len, text_header)
        text = memoryview(self.data)[offset : offset + text_len]
        return bytes(text).replace(b"\r", b"\n").decode("windows-1252")


@frozen
class ARDFVolumeTableOfContents:
    """
    Volume table of contents section.

    Contains lines and pointers for a volume in an ARDF file.
    """

    offset: int
    size: int
    lines: list[int]
    pointers: list[int]
    _voff_struct = struct.Struct("<LLQQ")

    @classmethod
    def unpack(cls, header: ARDFHeader) -> "ARDFVolumeTableOfContents":
        """
        Unpack a volume table of contents.

        Parameters
        ----------
        header : ARDFHeader
            The VTOC section header.

        Returns
        -------
        ARDFVolumeTableOfContents
            The unpacked volume table of contents.
        """
        # cant reuse exact ARDFTableOfContents for VTOC, but structure is similar
        data = header.data
        offset = header.offset
        if header.size != 32 or header.name != b"VTOC":
            raise ValueError("Malformed volume table of contents.", header)
        size, nentries, stride = TOC_STRUCT.unpack_from(data, offset + 16)
        assert stride == 40, stride
        assert size - 32 == nentries * stride, (size, nentries, stride)
        lines, pointers = [], []
        for _i, toc_offset in enumerate(range(header.offset + 32, header.offset + size, stride)):
            entry_header = ARDFHeader.unpack(data, toc_offset)
            if not entry_header.crc:
                # scan was interrupted, and vtoc is zero-filled
                break
            if entry_header.name != b"VOFF":
                raise ValueError("Malformed volume table entry.", entry_header)
            entry_header.validate()
            force_index, line, point, pointer = cls._voff_struct.unpack_from(data, toc_offset + 16)
            lines.append(line)
            pointers.append(pointer)
        return cls(offset, size, lines, pointers)


@frozen
class ARDFVchan:
    """
    Volume channel definition.

    Represents a channel with a name and unit.
    """

    name: str
    unit: str
    _struct = struct.Struct("<32s32s")

    @classmethod
    def unpack(cls, header: ARDFHeader) -> "ARDFVchan":
        """
        Unpack a channel definition.

        Parameters
        ----------
        header : ARDFHeader
            The VCHN section header.

        Returns
        -------
        ARDFVchan
            The unpacked channel definition.
        """
        if header.size != 80 or header.name != b"VCHN":
            raise ValueError("Malformed channel definition.", header)
        header.validate()
        name, unit = cls._struct.unpack_from(header.data, header.offset + 16)
        return cls(decode_cstring(name), decode_cstring(unit))


@frozen
class ARDFXdef:
    """
    Experiment definition section.

    Stores configuration values for the AFM experiment.
    """

    offset: int
    size: int
    xdef: list[str]
    _struct = struct.Struct("<LL")

    @classmethod
    def unpack(cls, header: ARDFHeader) -> "ARDFXdef":
        """
        Unpack an experiment definition.

        Parameters
        ----------
        header : ARDFHeader
            The XDEF section header.

        Returns
        -------
        ARDFXdef
            The unpacked experiment definition.
        """
        if header.size != 96 or header.name != b"XDEF":
            raise ValueError("Malformed experiment definition.", header)
        header.validate()
        # Experiment definition is just a string
        _, nchars = cls._struct.unpack_from(header.data, header.offset + 16)
        assert _ == 0, _
        if nchars > header.size:
            raise ValueError("Experiment definition too long.", header, nchars)
        xdef = memoryview(header.data)[header.offset + 24 : header.offset + 24 + nchars]
        xdef = bytes(xdef).decode("windows-1252").split(";")[:-1]
        return cls(header.offset, header.size, xdef)


@frozen
class ARDFVset:
    """
    Volume set metadata header.

    Points to individual volume data locations.
    """

    data: Buffer
    offset: int
    size: int
    force_index: int
    line: int
    point: int
    # vtype seems to differ between different FV modes.
    # FMAP with ext;ret;dwell shows 0b10 = 2 everywhere.
    # FFM with just trace or just retrace shows 0b101 = 5 everywhere.
    # FFM storing both shows 0b1010 = 10 for trace
    # and 0b1011 = 11 for retrace.
    vtype: int = field(repr=bin)
    prev_vset_offset: int
    next_vset_offset: int
    _struct = struct.Struct("<LLLLQQ")

    @classmethod
    def unpack(cls, vset_header: ARDFHeader) -> "ARDFVset":
        """
        Unpack a VSET header.

        Parameters
        ----------
        vset_header : ARDFHeader
            The VSET section header.

        Returns
        -------
        ARDFVset
            The unpacked volume set header.
        """
        if vset_header.name != b"VSET":
            raise ValueError("malformed VSET header", vset_header)
        vset_header.validate()
        data = vset_header.data
        offset = vset_header.offset
        size = cls._struct.size + 16
        return cls(data, offset, size, *cls._struct.unpack_from(data, offset + 16))


@frozen
class ARDFVdata:
    """
    Volume data segment metadata.

    Provides offsets and segments for a single force curve.
    """

    data: Buffer
    offset: int
    force_index: int
    line: int
    point: int
    nfloats: int
    channel: int
    seg_offsets: tuple[int, ...]
    _struct = struct.Struct("<10L")

    @classmethod
    def unpack(cls, header: ARDFHeader) -> "ARDFVdata":
        """
        Unpack volume data metadata.

        Parameters
        ----------
        header : ARDFHeader
            The VDAT section header.

        Returns
        -------
        ARDFVdata
            The unpacked volume data metadata.
        """
        data = header.data
        offset = header.offset
        (
            force_index,
            line,
            point,
            nfloats,
            channel,
            *seg_offsets,
        ) = cls._struct.unpack_from(data, offset + 16)
        return cls(data, offset, force_index, line, point, nfloats, channel, seg_offsets)

    @property
    def array_offset(self) -> int:
        """
        Get the byte offset of the data array.

        Returns
        -------
        int
            The byte offset.
        """
        return self.offset + self._struct.size + 16

    @property
    def next_offset(self) -> int:
        """
        Get the byte offset of the next section.

        Returns
        -------
        int
            The byte offset.
        """
        return self.array_offset + self.nfloats * 4

    def get_ndarray(self) -> np.ndarray:
        """
        Read the data segment as a numpy array.

        Returns
        -------
        np.ndarray
            The parsed data array in the unit reported by the ARDF channel metadata.
        """
        with memoryview(self.data) as v:  # assert data is open, and hold it open
            return np.ndarray(
                shape=self.nfloats,
                dtype="<f4",
                buffer=v,
                offset=self.array_offset,
            ).astype("f4", copy=True)


@frozen
class ARDFImage:
    """
    An image section in an ARDF file.

    Parses and represents an image channel.
    """

    data: Buffer
    ibox_offset: int
    name: str
    shape: Index
    units: str
    step_info: StepInfo
    _struct = struct.Struct("<LLQQdd32s32s32s32s")

    @classmethod
    def parse_imag(cls, imag_header: ARDFHeader) -> "ARDFImage":
        """
        Parse an image section from its header.

        Parameters
        ----------
        imag_header : ARDFHeader
            The IMAG section header.

        Returns
        -------
        ARDFImage
            The parsed image representation.
        """
        if imag_header.name != b"IMAG":
            raise ValueError("Malformed image header.", imag_header)
        imag_toc = ARDFTableOfContents.unpack(imag_header)

        # don't use NEXT or THMB data, step over
        ttoc_header = ARDFHeader.unpack(imag_toc.data, imag_toc.offset + imag_toc.size)
        ttoc = ARDFTextTableOfContents.unpack(ttoc_header)

        # don't use TTOC or TOFF, step over
        idef_header = ARDFHeader.unpack(ttoc.data, ttoc.offset + ttoc.size)
        if idef_header.name != b"IDEF" or idef_header.size != cls._struct.size + 16:
            raise ValueError("Malformed image definition.", idef_header)
        idef_header.validate()
        points, lines, _, __, x_step, y_step, *cstrings = cls._struct.unpack_from(
            idef_header.data,
            idef_header.offset + 16,
        )
        assert not _, (_, __)
        assert not __, (_, __)
        x_unit, y_unit, name, units = list(map(decode_cstring, cstrings))

        return cls(
            imag_header.data,
            idef_header.offset + idef_header.size,
            name,
            (points, lines),
            units,
            ((x_step, x_unit), (y_step, y_unit)),
        )

    def get_image(self) -> np.ndarray:
        """
        Retrieve the full image array.

        Returns
        -------
        np.ndarray
            The image data.
        """
        ibox_header = ARDFHeader.unpack(self.data, self.ibox_offset)
        if ibox_header.size != 32 or ibox_header.name != b"IBOX":
            raise ValueError("Malformed image layout.", ibox_header)
        ibox_header.validate()
        data_offset = ibox_header.offset + ibox_header.size + 16  # past IDAT header
        ibox_size, lines, stride = TOC_STRUCT.unpack_from(ibox_header.data, ibox_header.offset + 16)
        points = (stride - 16) // 4  # less IDAT header
        assert (points, lines) == self.shape
        # elide image data validation and map into an array directly
        with memoryview(self.data) as v:  # assert data is open, and hold it open
            arr = np.ndarray(
                shape=(lines, points),
                dtype="<f4",
                buffer=v,
                offset=data_offset,
                strides=(stride, 4),
            ).astype("f4", copy=True)
        gami_header = ARDFHeader.unpack(ibox_header.data, ibox_header.offset + ibox_size)
        if gami_header.size != 16 or gami_header.name != b"GAMI":
            raise ValueError("Malformed image layout.", gami_header)
        gami_header.validate()
        return arr


@frozen
class ARDFFFMReader:
    """
    Fast Force Mapping (FFM) reader.

    Reads force curves efficiently when files are regularly spaced.
    """

    data: Buffer
    array_view: np.ndarray = field(repr=False)
    array_offset: int  # hard to recover from views
    channels: list[int]  # [z, d]
    # seg_offsets is weird. you'd think it would contain the starting index
    # for each segment. However, it always has a trailing value of 1-nfloats,
    # and nonexistent segments get a zero. For regular/FFM data, we'll just
    # assume that the second offset maps to our "split" concept.
    seg_offsets: tuple
    up: bool
    trace: bool
    vtype: int
    vchns: list[ARDFVchan]
    curve_height_offset: float = field(repr=False, default=0.0)

    @classmethod
    def parse(
        cls,
        first_vset_header: ARDFHeader,
        points: int,
        lines: int,
        channels: Any,
        vchns: list[ARDFVchan],
        curve_height_offset: float = 0.0,
    ) -> "ARDFFFMReader":
        """
        Parse FFM metadata.

        Parameters
        ----------
        first_vset_header : ARDFHeader
            The header of the first VSET.
        points : int
            Number of points per line.
        lines : int
            Number of lines.
        channels : Any
            The channel mapping.
        vchns : list[ARDFVchan]
            The list of channel definitions.

        Returns
        -------
        ARDFFFMReader
            The parsed FFM reader.
        """
        data = first_vset_header.data
        # just walk past these first headers to find our data_offset
        first_vset = ARDFVset.unpack(first_vset_header)
        vset_stride = first_vset.next_vset_offset - first_vset_header.offset
        if first_vset.vtype & 0x2:
            line_stride = vset_stride * points * 2
        else:
            # better be "both" setting see ARDFVset
            line_stride = vset_stride * points
        up = first_vset.line == 0
        trace = first_vset.point == 0

        first_vnam_header = ARDFHeader.unpack(data, first_vset_header.offset + first_vset_header.size)
        if first_vnam_header.name != b"VNAM":
            raise ValueError("Malformed volume name", first_vnam_header)
        first_vnam_header.validate()
        first_vdat_header = ARDFHeader.unpack(data, first_vnam_header.offset + first_vnam_header.size)
        if first_vdat_header.name != b"VDAT":
            raise ValueError("Malformed volume data")
        first_vdat = ARDFVdata.unpack(first_vdat_header)
        return cls(
            data=data,
            array_view=np.ndarray(
                shape=(lines, points, len(channels), first_vdat.nfloats),
                dtype="<f4",
                buffer=data,
                offset=first_vdat.array_offset,
                strides=(
                    line_stride,
                    vset_stride,
                    first_vdat_header.size,
                    4,
                ),
                # match up array and image coordinates
            )[:: 1 if up else -1, :: 1 if trace else -1],
            array_offset=first_vdat.array_offset,
            channels=[channel[0] for channel in channels.values()],
            seg_offsets=first_vdat.seg_offsets,
            up=up,
            trace=trace,
            vtype=first_vset.vtype,
            vchns=vchns,
            curve_height_offset=curve_height_offset,
        )

    def get_curve(self, r: int, c: int, reverse_curve_points: bool = False) -> dict[str, dict[str, np.ndarray]]:
        """
        Efficiently get a specific curve from disk.

        Parameters
        ----------
        r : int
            The row index.
        c : int
            The column index.
        reverse_curve_points : bool, optional
            Whether to reverse the points in each curve segment.

        Returns
        -------
        dict[str, dict[str, np.ndarray]]
            The curve data.
        """
        with memoryview(self.data):  # assert data is open, and hold it open
            x = self.array_view[r, c, self.channels]  # advanced indexing copies
        x = x.astype("f4", copy=False)
        num_phases = 2 if (self.vtype & 0x2) else 1
        reshaped = x.reshape((len(self.channels), num_phases, -1))

        if num_phases == 2:
            seg_keys = ["Segment_0", "Segment_1"]
        else:
            seg_keys = ["Segment_1"] if self.is_retrace else ["Segment_0"]

        curve_dict = {}
        for idx, chan_idx in enumerate(self.channels):
            chan_name = self.vchns[chan_idx].name
            curve_dict[chan_name] = {}
            for phase_idx, seg_name in enumerate(seg_keys):
                # Map channel and segment name to its respective slice
                if reverse_curve_points and chan_name == "Raw":
                    curve_dict[chan_name][seg_name] = self.curve_height_offset - reshaped[idx, phase_idx]
                else:
                    curve_dict[chan_name][seg_name] = reshaped[idx, phase_idx]

        return curve_dict

    def iter_indices(self) -> Iterable[Index]:
        """
        Iterate over force curve indices in on-disk order.

        Returns
        -------
        Iterable[Index]
            An iterable of indices.
        """
        # undo the array reversals in the constructor method
        lines, points = self.array_view.shape[:2]
        lines_iter = range(lines)
        if not self.up:
            lines_iter = reversed(lines_iter)
        for line in lines_iter:
            points_iter = range(points)
            if not self.trace:
                points_iter = reversed(points_iter)
            for point in points_iter:
                yield line, point

    def iter_curves(self) -> Iterable[dict[str, dict[str, np.ndarray]]]:
        """
        Iterate over curves lazily in on-disk order.

        Returns
        -------
        Iterable[dict[str, dict[str, np.ndarray]]]
            An iterable yielding curve data.
        """
        # TODO: cleverly use np.nditer?
        for index in self.iter_indices():
            yield self.get_curve(*index)

    def get_all_curves(self) -> ZDArrays:
        """
        Eagerly load all curves into memory.

        Returns
        -------
        ZDArrays
            All curves in memory.
        """
        with memoryview(self.data):  # assert data is open, and hold it open
            # advanced indexing triggers a copy
            loaded_data = self.array_view[:, :, self.channels, :]
        # reshape assuming equal points on extend and retract
        loaded_data = loaded_data.reshape(loaded_data.shape[:-1] + (2, -1))
        # make it look like z, d
        return np.moveaxis(loaded_data, 2, 0)


@frozen
class ARDFForceMapReader:
    """
    Force map reader for non-regular files.

    Used when lines are not evenly spaced or scan was stopped early.
    """

    data: Buffer
    vtoc: ARDFVolumeTableOfContents
    lines: int
    points: int
    vtype: int
    channels: ChanMap
    _seen_vsets: dict[Index, ARDFVset] = field(init=False, factory=dict)

    @property
    def zname(self) -> str:
        """
        Get the name of the Z channel.

        Returns
        -------
        str
            The Z channel name.
        """
        return "Raw" if "Raw" in self.channels else "ZSnsr"

    def traverse_vsets(self, pointer: int) -> Iterable[ARDFVset]:
        """
        Traverse volume set headers from a starting pointer.

        Parameters
        ----------
        pointer : int
            The byte offset to start traversing from.

        Returns
        -------
        Iterable[ARDFVset]
            An iterable of volume sets.
        """
        while True:
            header = ARDFHeader.unpack(self.data, pointer)
            if header.name != b"VSET":
                break
            vset = ARDFVset.unpack(header)
            index = (vset.line, vset.point, vset.vtype)
            if index not in self._seen_vsets:
                self._seen_vsets[index] = vset
            yield vset  # .line, vset.point, vset.
            pointer = vset.next_vset_offset

    def traverse_vdats(self, pointer: int) -> Iterable[ARDFVdata]:
        """
        Traverse volume data segments from a starting pointer.

        Parameters
        ----------
        pointer : int
            The byte offset to start traversing from.

        Returns
        -------
        Iterable[ARDFVdata]
            An iterable of volume data segments.
        """
        vnam_header = ARDFHeader.unpack(self.data, pointer)
        if vnam_header.name != b"VNAM":
            raise ValueError("Malformed volume name", vnam_header)
        vnam_header.validate()
        pointer = vnam_header.offset + vnam_header.size
        while True:
            vdat_header = ARDFHeader.unpack(self.data, pointer)
            if vdat_header.name != b"VDAT":
                break
            vdat_header.validate()  # opportunity to read data with gil released
            yield ARDFVdata.unpack(vdat_header)
            pointer = vdat_header.offset + vdat_header.size

    def get_curve(self, r: int, c: int) -> ZDArrays:
        """
        Efficiently get a specific curve from disk.

        Parameters
        ----------
        r : int
            The row index.
        c : int
            The column index.

        Returns
        -------
        ZDArrays
            The curve data.
        """
        if not (0 <= r < self.lines and 0 <= c < self.points):
            raise ValueError("Invalid index:", (self.lines, self.points), (r, c))

        index = r, c, self.vtype
        if index not in self._seen_vsets:
            # bisect row pointer
            if self.vtoc.lines[0] > self.vtoc.lines[-1]:
                # probably reversed
                sl = np.s_[::-1]
            else:
                sl = np.s_[:]
            i = bisect_left(self.vtoc.lines[sl], r)
            if i >= len(self.vtoc.lines) or r != int(self.vtoc.lines[sl][i]):
                return NANCURVE
            # read entire line of the vtoc
            for vset in self.traverse_vsets(int(self.vtoc.pointers[sl][i])):
                if vset.line != r:
                    break
                # traverse_vsets implicitly fills in seen_vsets

        try:
            vset = self._seen_vsets[index]
        except KeyError:
            # index was missing from the line, probably due to "Stop"
            return NANCURVE

        zxr, dxr = NANCURVE
        for vdat in self.traverse_vdats(vset.offset + vset.size):
            _, ext, *_ = vdat.seg_offsets
            ret = 2 * ext
            if vdat.channel == self.channels[self.zname][0]:
                z = vdat.get_ndarray()
                zxr = z[:ret].reshape(2, ext)
            elif vdat.channel == self.channels["Defl"][0]:
                d = vdat.get_ndarray()
                dxr = d[:ret].reshape(2, ext)
        return zxr, dxr

    def iter_indices(self) -> Iterable[Index]:
        """
        Iterate over force curve indices in on-disk order.

        Returns
        -------
        Iterable[Index]
            An iterable of indices.
        """
        for vset in self.traverse_vsets(self.vtoc.pointers[0]):
            if vset.vtype != self.vtype:
                continue
            yield vset.line, vset.point

    def iter_curves(self) -> Iterable[tuple[Index, ZDArrays]]:
        """
        Iterate over curves lazily in on-disk order.

        Returns
        -------
        Iterable[tuple[Index, ZDArrays]]
            An iterable yielding curve indices and data.
        """
        zname = self.zname
        for vset in self.traverse_vsets(self.vtoc.pointers[0]):
            if vset.vtype != self.vtype:
                continue
            zxr, dxr = NANCURVE
            for vdat in self.traverse_vdats(vset.offset + vset.size):
                _, ext, *_ = vdat.seg_offsets
                ret = 2 * ext
                if vdat.channel == self.channels[zname][0]:
                    z = vdat.get_ndarray()
                    zxr = z[:ret].reshape(2, ext)
                elif vdat.channel == self.channels["Defl"][0]:
                    d = vdat.get_ndarray()
                    dxr = d[:ret].reshape(2, ext)
            yield (vset.line, vset.point), (zxr, dxr)

    def get_all_curves(self) -> ZDArrays:
        """
        Eagerly load all curves into memory.

        Returns
        -------
        ZDArrays
            All curves in memory.
        """
        minext = 0xFFFFFFFF
        vdats = {}
        for vset in self.traverse_vsets(self.vtoc.pointers[0]):
            if vset.vtype != self.vtype:
                continue
            x = vdats[vset.line, vset.point] = [None, None]
            for vdat in self.traverse_vdats(vset.offset + vset.size):
                if vdat.channel == self.channels[self.zname][0]:
                    x[0] = vdat  # zvdat
                elif vdat.channel == self.channels["Defl"][0]:
                    x[1] = vdat  # dvdat
            assert None not in x, f"missing vdat channel: {x}, {self.channels}"
            minext = min(minext, vdat.seg_offsets[1])
        del vset, vdat, x
        minfloats = 2 * minext
        x = np.full((self.lines, self.points, 2, minfloats), np.nan, dtype=np.float32)
        for (r, c), (zvdat, dvdat) in vdats.items():
            # code elsewhere assumes split is halfway through
            floats = zvdat.seg_offsets[1] * 2
            halfextra = (floats - minfloats) // 2  # even - even -> even
            sl = np.s_[halfextra : minfloats + halfextra]
            # TODO: verify turnaround point against iter_curves
            x[r, c, :, :] = zvdat.get_ndarray()[sl], dvdat.get_ndarray()[sl]
        return np.moveaxis(x.reshape(x.shape[:-1] + (2, -1)), 2, 0)


class ARDFVolume(CurvesVolume):
    """
    A volume section in an ARDF file.

    Represents a full force volume containing multiple force curves.

    Parameters
    ----------
    name : str
        The name of the volume.
    shape_x : int
        The number of columns in the image.
    shape_y : int
        The number of rows in the image.
    reader : ARDFForceMapReader | ARDFFFMReader
        The reader used to load curve data.
    channel_units : dict[str, str]
        A dictionary mapping channel names to their units.
    step_info : StepInfo
        Step size and unit information for each axis.
    flip_image : bool, optional
        Whether to flip the image vertically. Default is True.
    reverse_curve_points : bool, optional
        Whether to reverse the points in each curve segment. Default is True.
    """

    volm_offset: int
    name: str
    shape: Index
    step_info: StepInfo
    _reader: ARDFForceMapReader | ARDFFFMReader
    _struct = struct.Struct("<LL24sddd32s32s32s32sQ")

    def __init__(
        self,
        name: str,
        shape_x: int,
        shape_y: int,
        reader: ARDFForceMapReader | ARDFFFMReader,
        channel_units: dict[str, str],
        step_info: StepInfo,
        flip_image: bool = True,
        reverse_curve_points: bool = True,
    ):
        """
        Initialise ARDFVolume.

        Parameters
        ----------
        name : str
            The name of the volume.
        shape_x : int
            The number of columns in the image.
        shape_y : int
            The number of rows in the image.
        reader : ARDFForceMapReader | ARDFFFMReader
            The reader used to load curve data.
        channel_units : dict[str, str]
            A dictionary mapping channel names to their units.
        step_info : StepInfo
            Step size and unit information for each axis.
        flip_image : bool, optional
            Whether to flip the image vertically. Default is True.
        reverse_curve_points : bool, optional
            Whether to reverse the points in each curve segment. Default is True.
        """
        super().__init__(
            name=name, shape_x=shape_x, shape_y=shape_y, channel_units=channel_units, flip_image=flip_image
        )
        self.reverse_curve_points = reverse_curve_points
        self.step_info = step_info
        self._reader = reader

    def get_curve(self, y: int, x: int, flip_image: bool | None = None) -> dict[str, dict[str, np.ndarray]]:
        """
        Efficiently get a specific curve from disk.

        Parameters
        ----------
        y : int
            The row index.
        x : int
            The column index.
        flip_image : bool, optional
            Whether to flip the image vertically. If None, uses the instance's flip_image attribute.

        Returns
        -------
        dict[str, dict[str, np.ndarray]]
            The curve data.
        """
        if flip_image is None:
            flip_image = self.flip_image
        if flip_image:
            y = self.shape_y - 1 - y
        return self._reader.get_curve(y, x, reverse_curve_points=self.reverse_curve_points)

    def iter_indices(self) -> Iterable[Index]:
        """
        Iterate over force curve indices in on-disk order.

        Returns
        -------
        Iterable[Index]
            An iterable of indices.
        """
        return self._reader.iter_indices()

    def __iter__(self):
        """
        Iterate over curves using the instance's flip setting.

        Returns
        -------
        Iterable[dict[str, dict[str, np.ndarray]]]
            An iterable yielding curve data.
        """
        return self.iter_curves()


class ARDFDataset(CurvesDataset):
    """
    A dataset containing multiple ARDF volumes.

    Parameters
    ----------
    volumes : dict[str, ARDFVolume]
        A dictionary of ARDFVolume instances keyed by their names.
    metadata : CurvesMetadata
        Metadata associated with the dataset.
    mmap_file : mmap.mmap
        The memory map backing the ARDF data.
    default_volume_name : str | None
        The name of the default volume to use when accessing curve data.
    """

    def __init__(
        self,
        volumes: dict[str, ARDFVolume],
        metadata: CurvesMetadata,
        mmap_file: mmap.mmap,
        default_volume_name: str | None = None,
    ):
        """
        Initialise ARDFDataset.

        Parameters
        ----------
        volumes : dict[str, ARDFVolume]
            A dictionary of ARDFVolume instances keyed by their names.
        metadata : CurvesMetadata
            Metadata associated with the dataset.
        mmap_file : mmap.mmap
            The memory map backing the ARDF data.
        default_volume_name : str | None
            The name of the default volume to use when accessing curve data.
        """
        super().__init__(volumes=volumes, metadata=metadata, default_volume_name=default_volume_name)
        self.mmap_file = mmap_file

    def close(self):
        """Close the underlying ARDF memory map to free resources."""
        if self.mmap_file is not None:
            self.mmap_file.close()
            self.mmap_file = None


def parse_volm(volm_header: ARDFHeader, flip_image: bool = True, curve_height_offset: float = 0.0) -> ARDFVolume:
    """
    Parse a volume section from its header.

    Parameters
    ----------
    volm_header : ARDFHeader
        The VOLM section header.
    flip_image : bool
        Whether to flip the image vertically.

    Returns
    -------
    ARDFVolume
        The parsed volume.
    """
    data = volm_header.data
    if volm_header.size != 32 or volm_header.name != b"VOLM":
        raise ValueError("Malformed volume header.", volm_header)
    # the next headers look a lot like VSET is a table of contents, but I've only
    # seen NEXT and NSET headers. NEXT shows up if "both" trace and retrace data
    # are inside. I'll assume the last entry is always NSET, the number of VSETs.
    volm_toc = ARDFTableOfContents.unpack(volm_header)
    nset_header, nsets = volm_toc.entries[-1]
    nset_header.validate()
    ttoc_header = ARDFHeader.unpack(data, volm_toc.offset + volm_toc.size)
    ttoc = ARDFTextTableOfContents.unpack(ttoc_header)

    # don't use TTOC or TOFF, step over

    # cls essentially represents VDEF plus its linkage down to VSET
    # so this unpacking is intentionally inlined here.
    vdef_header = ARDFHeader.unpack(data, ttoc.offset + ttoc.size)
    if vdef_header.name != b"VDEF" or vdef_header.size != ARDFVolume._struct.size + 16:
        raise ValueError("Malformed volume definition.", vdef_header)
    vdef_header.validate()
    unpack_from = ARDFVolume._struct.unpack_from  # line wrapping
    points, lines, _, x_step, y_step, t_step, *cstrings, nseg = unpack_from(vdef_header.data, vdef_header.offset + 16)
    assert sum(_) == 0, _
    complete = points * lines == nsets
    x_unit, y_unit, t_unit, seg_names = list(map(decode_cstring, cstrings))
    seg_names = seg_names.split(";")[:-1]
    assert nseg == len(seg_names)

    # Implicit table of channels here smh
    offset = vdef_header.offset + vdef_header.size
    channels: ChanMap = {}
    vchn_list: list[ARDFVchan] = []
    for _i in range(5):
        header = ARDFHeader.unpack(data, offset)
        if header.name != b"VCHN":
            break
        vchn = ARDFVchan.unpack(header)
        offset = header.offset + header.size
        vchn_list.append(vchn)
    else:
        raise RuntimeError("Got too many channels.", channels)

    channels: ChanMap = {vchn.name: (i, vchn) for i, vchn in enumerate(vchn_list)}

    xdef = ARDFXdef.unpack(header)
    vtoc_header = ARDFHeader.unpack(data, xdef.offset + xdef.size)
    vtoc = ARDFVolumeTableOfContents.unpack(vtoc_header)

    mlov_header = ARDFHeader.unpack(data, vtoc.offset + vtoc.size)
    if mlov_header.size != 16 or mlov_header.name != b"MLOV":
        raise ValueError("Malformed volume table of contents.", mlov_header)
    mlov_header.validate()

    channel_units = {channel_name: channel.unit for channel_name, (i, channel) in channels.items()}
    logger.debug(f"Channel units: {channel_units}")

    # Check if each offset is regularly spaced
    # optimize for LARGE regular case (FMaps are SMALL)
    first_vset_header = ARDFHeader.unpack(data, vtoc.pointers[0])
    if complete and not np.any(np.diff(np.diff(vtoc.pointers))):
        reader = ARDFFFMReader.parse(
            first_vset_header, points, lines, channels, vchn_list, curve_height_offset=curve_height_offset
        )
        name = "Trace" if reader.trace else "Retrace"
    else:
        first_vset = ARDFVset.unpack(first_vset_header)
        reader = ARDFForceMapReader(
            data,
            vtoc,
            lines,
            points,
            first_vset.vtype,
            channels,
            vchn_list,
        )
        name = "FMAP"

    return ARDFVolume(
        name=name,
        shape_x=points,
        shape_y=lines,
        reader=reader,
        channel_units=channel_units,
        step_info=((x_step, x_unit), (y_step, y_unit), (t_step, t_unit)),
        flip_image=flip_image,
        reverse_curve_points=True,
    )


class ARDFReader:
    """
    A reader for ARDF files.

    Parameters
    ----------
    filepath : str | Path
        The path to the ARDF file.
    channel : str | None, optional
        The channel to load by default.
    flip_image : bool, optional
        Whether to flip the image vertically. Default is True.
    """

    def __init__(
        self,
        filepath: str | Path,
        channel: str | None = None,
        flip_image: bool = True,
        config_path: Path | str | None = None,
    ):
        """
        Initialise ARDFReader.

        Parameters
        ----------
        filepath : str | Path
            The path to the ARDF file.
        channel : str | None, optional
            The channel to load by default.
        flip_image : bool, optional
            Whether to flip the image vertically. Default is True.
        """
        self.filepath = Path(filepath)
        self.channel = channel
        self.flip_image = flip_image
        self.config_path = config_path
        self.mmap_file = mmap_path_read_only(filepath)
        file_header = self.check_type(self.mmap_file)
        ftoc_header = ARDFHeader.unpack(self.mmap_file, offset=file_header.size)
        if ftoc_header.name != b"FTOC":
            raise ValueError("Malformed ARDF file table of contents.", ftoc_header)
        ftoc = ARDFTableOfContents.unpack(ftoc_header)
        ttoc_header = ARDFHeader.unpack(self.mmap_file, offset=ftoc.offset + ftoc.size)
        ttoc = ARDFTextTableOfContents.unpack(ttoc_header)
        assert len(ttoc.entries) == 1
        self.metadata = parse_ar_note(ttoc.decode_entry(0).splitlines())
        self.images: dict[str, ARDFImage] = {}
        self.volumes: dict[str, ARDFVolume] = {}
        TRIGGER_HEIGHT_KEYS = ["TriggerRawZSensor", "ForceDist"]
        curve_height_offset = 0
        for key in TRIGGER_HEIGHT_KEYS:
            if key in self.metadata:
                curve_height_offset = float(self.metadata[key])
                break

        for item, pointer in ftoc.entries:
            item.validate()
            item = ARDFHeader.unpack(self.mmap_file, pointer)
            if item.name == b"IMAG":
                item = ARDFImage.parse_imag(item)
                self.images[item.name] = item
            elif item.name == b"VOLM":
                item = parse_volm(item, flip_image=self.flip_image, curve_height_offset=curve_height_offset)
                self.volumes[item.name] = item
            else:
                raise RuntimeError(f"Unknown TOC entry {item.name}.", item)

        self.essential_metadata = self.filter_essential_metadata(self.metadata)

        self.size_x = float(self.metadata["FastScanSize"]) * NANOMETER_UNIT_CONVERSION
        self.size_y = float(self.metadata["SlowScanSize"]) * NANOMETER_UNIT_CONVERSION
        if self.images:
            first_image = next(iter(self.images.values()))
            self.shape_x, self.shape_y = first_image.shape
        self.px2nm = self.size_x / self.shape_x

    def filter_essential_metadata(self, raw_metadata: dict[str, Any]) -> dict[str, Any]:
        """
        Extract canonical essential metadata from raw global metadata.

        Parameters
        ----------
        raw_metadata : dict[str, Any]
            The raw global metadata dictionary to filter.

        Returns
        -------
        dict[str, Any]
            A dictionary keyed by canonical metadata names.
        """
        essential_key_options = load_config(self.config_path).get("ardf", {}).get("essential_metadata_keys", {})
        filtered_metadata = {}
        for target_name, source_keys in essential_key_options.items():
            for source_key in source_keys:
                if source_key in raw_metadata:
                    filtered_metadata[target_name] = raw_metadata[source_key]
                    break
        if "read_sample_rate" not in filtered_metadata:
            if "global.time_step" in self.metadata:
                filtered_metadata["read_sample_rate"] = 1.0 / float(self.metadata["global.time_step"])
            else:
                first_volume = next(iter(self.volumes.values()))
                filtered_metadata["read_sample_rate"] = 1.0 / first_volume.step_info[-1][0]
        return filtered_metadata

    def save_to_h5(self):
        """
        Save the ARDF data to an HDF5 file.

        The output file will have the same name as the input file but with a .h5 extension.
        """
        # Determine the path for the H5 file, ensuring it does not overwrite an existing file
        self.h5_path = find_unused_filename(self.filepath)

        h5_saver = H5Saver(self.h5_path)

        with h5_saver.create_file(source=self.filepath.suffix):
            # Save metadata
            h5_saver.setup_curves_group(channel_units=next(iter(self.volumes.values())).channel_units)
            size_x_m = self.size_x / NANOMETER_UNIT_CONVERSION
            size_y_m = self.size_y / NANOMETER_UNIT_CONVERSION
            h5_saver.save_global_meta(
                self.metadata | {f"essential.{key}": value for key, value in self.essential_metadata.items()},
                size_x=size_x_m,
                size_y=size_y_m,
                shape_x=self.shape_x,
                shape_y=self.shape_y,
            )

            # Save volumes
            for volume_name, ardf_volume in self.volumes.items():
                h5_saver.setup_volume(ardf_volume)
                logger.debug(f"Volume channel units before saving: {ardf_volume.channel_units}")
                for curve_idx, curve in enumerate(
                    tqdm(
                        ardf_volume.iter_curves(flip_image=False),
                        total=ardf_volume.shape_x * ardf_volume.shape_y,
                        desc=f"Saving curves for volume '{volume_name}'",
                    )
                ):
                    h5_saver.save_curve(curve, curve_idx, ardf_volume.shape_x * ardf_volume.shape_y, volume_name)

                h5_saver.complete_saving(ardf_volume)
                logger.debug(f"Channel units after saving: {ardf_volume.channel_units}")

            # Save images
            for idx, (image_name, ardf_image) in enumerate(self.images.items()):
                h5_saver.save_image(
                    image_data=ardf_image.get_image(), image_name=image_name, z_unit=ardf_image.units, idx=idx
                )

    def check_type(self, data: Buffer) -> ARDFHeader:
        """
        Check if the buffer contains a valid ARDF file.

        Parameters
        ----------
        data : Buffer
            The data buffer.

        Returns
        -------
        ARDFHeader
            The parsed file header.
        """
        file_header = ARDFHeader.unpack(data, 0)
        if file_header.size != 16 or file_header.name != b"ARDF":
            raise ValueError("Not an ARDF file.", file_header)
        file_header.validate()
        return file_header

    def load(self, channel: str | None = None, flip_image: bool = True) -> AFMLoad:
        """
        Load the ARDF file data.

        Parameters
        ----------
        channel : str | None
            The channel to load. If None, the default channel will be used.
        flip_image : bool
            Whether to flip the image vertically to match typical AFM orientation.

        Returns
        -------
        AFMLoad
            The loaded AFM data.
        """
        if channel is not None:
            self.channel = channel
        if flip_image is not None:
            self.flip_image = flip_image
        for volume in self.volumes.values():
            volume.flip_image = self.flip_image
        self.trace = "retrace" not in self.channel.lower()
        logger.info(
            f"Loading ARDF file: {self.filepath}, channel: {self.channel}, trace: {self.trace}, flip_image: {self.flip_image}"
        )
        self.shape_x, self.shape_y = self.images[self.channel].shape
        default_volume_name = "Trace" if self.trace else "Retrace"
        self.metadata["global.time_step"] = self.volumes[default_volume_name].step_info[-1][0]
        self.essential_metadata = self.filter_essential_metadata(self.metadata)
        curves_metadata = CurvesMetadata(
            self.metadata,
            self.essential_metadata,
            self.shape_x,
            self.shape_y,
            self.flip_image,
        )
        curves_dataset = ARDFDataset(
            self.volumes,
            curves_metadata,
            self.mmap_file,
            default_volume_name=default_volume_name,
        )

        image = self.images[self.channel].get_image()
        z_units = self.images[self.channel].units
        if self.flip_image:
            image = np.flipud(image)

        return AFMLoad(
            image=image,
            px2nm=self.px2nm,
            z_units=z_units,
            curves_dataset=curves_dataset,
        )

    def get_available_channels(self) -> list[str]:
        """
        Get a list of available channels in the ARDF file.

        Returns
        -------
        list[str]
            A list of channel names.
        """
        return list(self.images.keys())

    def close(self):
        """Close the underlying ARDF memory map to free resources."""
        if self.mmap_file is not None:
            self.mmap_file.close()
            self.mmap_file = None


def load_ardf(filepath: str | Path, channel: str, cached_data: dict) -> AFMLoad:
    """
    Load the ARDF file data.

    Parameters
    ----------
    filepath : str | Path
        Path to the ARDF file.
    channel : str | None
        The channel to load from the file. If None, the default channel will be used.
    cached_data : dict
        Cached data to avoid reloading heavy data.

    Returns
    -------
    AFMLoad
        The loaded AFM data.
    """
    if "ardf_loader" not in cached_data:
        cached_data["ardf_loader"] = ARDFReader(filepath=filepath, channel=channel)
    return cached_data["ardf_loader"].load(channel=channel)


def get_ardf_channels(filepath: str | Path, cached_data: dict) -> list[str]:
    """
    Get the available channels in the ARDF file.

    Parameters
    ----------
    filepath : str | Path
        Path to the ARDF file.
    cached_data : dict
        Cached data to avoid reloading heavy data.

    Returns
    -------
    list[str]
        A list of available channels in the ARDF file.
    """
    if "ardf_loader" not in cached_data:
        cached_data["ardf_loader"] = ARDFReader(filepath=filepath)
    return cached_data["ardf_loader"].get_available_channels()


def get_ardf_params() -> dict:
    """
    Get the parameters of the ARDF file.

    Returns
    -------
    dict
        A dictionary containing the parameters of the ARDF file.
    """
    return {"save_as_h5": bool}


def save_ardf_to_h5(filepath: str | Path, cached_data: dict) -> Path:
    """
    Save the ARDF data to an HDF5 file.

    Parameters
    ----------
    filepath : str | Path
        Path to the ARDF file.
    cached_data : dict
        Cached data to avoid reloading heavy data.

    Returns
    -------
    Path
        The path to the saved HDF5 file.
    """
    if "ardf_loader" not in cached_data:
        cached_data["ardf_loader"] = ARDFReader(filepath=filepath)
    cached_data["ardf_loader"].save_to_h5()
    cached_data["ardf_loader"].close()
    return cached_data["ardf_loader"].h5_path
