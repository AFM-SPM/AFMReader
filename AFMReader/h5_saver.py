# pylint: disable=too-many-instance-attributes,too-many-positional-arguments,too-many-locals
"""Module for saving AFM reader data to HDF5 files."""

from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from AFMReader import __version__
from AFMReader.logging import logger


class H5Saver:
    """
    A class to handle saving AFM curve and volume data to HDF5 files.

    Parameters
    ----------
    filepath : Path
        The path to the h5 file.
    h5file : h5py.File
        The h5 file object that will be written to.
    """

    def __init__(self, filepath: Path, h5file: h5py.File | None = None):
        """
        Initialise H5Saver.

        Parameters
        ----------
        filepath : Path
            The path to the h5 file.
        h5file : h5py.File | None
            The h5 file object that will be written to.
        """
        # The path to the h5 file
        self.filepath = filepath
        # The h5 file object that will be written to
        self.h5file = h5file

        # A nested dictionary to hold curve volume float data in memory before writing to the h5 file.
        # This is structured as [volume_name][segment][channel][data/indices]. Each list can get no larger
        # than BUFFER_SIZE before being written to the h5 file and cleared from memory
        self.volumes_data_buffer: dict[str, dict[str, dict[str, dict[str, list]]]] = {}

        # A nested dictionary to hold the h5 datasets for each volume to be saved, structured
        # as [volume_name][segment][channel][data/indices]
        self.volume_datasets: dict[str, dict[str, dict[str, dict[str, h5py.Dataset]]]] = {}
        self.volumes_dims: dict[str, tuple[int, int]] = {}

        self.curves_meta_buffer: dict[str, list] = {}
        self.curves_meta_datasets: dict[str, h5py.Dataset] = {}

        # The number of points saved so far for each channel and segment, structured as [volume_name][segment][channel]
        self.volume_points_saved: dict[str, dict[str, dict[str, int]]] = {}

        # The number of points read so far for each channel and segment, structured as [volume_name][segment][channel]
        # This is similar to volume_points_saved but tracks how many points have been read from the original data and
        # hence includes the points currently in the buffer.
        self.volume_points_read: dict[str, dict[str, dict[str, int]]] = {}

        # Initialise attributes to empty values, will be assigned in setup functions
        self.curve_data_group: h5py.Group | None = None
        self.global_meta_group: h5py.Group | None = None
        self.curves_meta_group: h5py.Group | None = None
        self.curve_work: list[tuple[bytes, h5py.Dataset, list[str]]] = []
        self.curve_search_terms: list[bytes] = []
        self.seg_work: list[tuple[bytes, h5py.Dataset, list[str]]] = []
        self.segment_search_terms: list[bytes] = []

        # Chunk size for H5 datasets
        self.DATA_CHUNKSIZE = 512 * 1024
        # Chunk size for indices datasets
        self.INDICES_CHUNKSIZE = 64 * 1024
        # Chunk size for metadata datasets (if needed)
        self.META_CHUNKSIZE = 64 * 1024
        # Maximum number of curves to check for changing metadata keys (to avoid checking every curve)
        self.MAX_CURVE_CHECKS = 20
        # Number of curves to hold in buffer
        self.BUFFER_SIZE = 500

    def create_file(self) -> h5py.File:
        """
        Create the h5 file and write initial global attributes.

        Returns
        -------
        h5py.File
            The created h5 file object.
        """
        self.h5file = h5py.File(self.filepath, "a")
        self.h5file.attrs["AFMReader_version"] = __version__
        self.h5file.attrs["created_by"] = "AFMReader"
        self.h5file.attrs["created_on"] = datetime.now().isoformat()
        return self.h5file

    def setup_curve_metadata_structure(self, changing_curve_keys: set, changing_segment_keys: set, num_of_curves: int):
        """
        Set up structure in the h5 file for saving curve data and metadata.

        Parameters
        ----------
        changing_curve_keys : set
            Keys of metadata that change across curves.
        changing_segment_keys : set
            Keys of metadata that change across segments.
        num_of_curves : int
            Total number of curves.
        """
        assert (
            self.h5file is not None
        ), "existing h5 file must be passed or create_file called before setup_curve_metadata_structure"
        # Create the main group for the curve data that all the curve data will be in
        self.curve_data_group = self.h5file.require_group("Curve_Data")

        # Establish empty groups for global metadata and curve metadata
        self.global_meta_group = self.curve_data_group.require_group("Global_Metadata")
        self.curves_meta_group = self.curve_data_group.require_group("Curve_Metadata")

        for key in changing_curve_keys:
            self.curves_meta_datasets[f"curve.{key}"] = self.curves_meta_group.create_dataset(
                name=f"curve.{key}",
                shape=(num_of_curves,),
                maxshape=(None,),
                chunks=self.META_CHUNKSIZE,
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            self.curves_meta_buffer[f"curve.{key}"] = []
        self.curve_work = [
            (f"{k}=".encode(), self.curves_meta_datasets[f"curve.{k}"], self.curves_meta_buffer[f"curve.{k}"])
            for k in changing_curve_keys
        ]
        self.curve_search_terms = [f"{k}=".encode() for k in changing_curve_keys]
        for key in changing_segment_keys:
            self.curves_meta_datasets[f"segment.{key}"] = self.curves_meta_group.create_dataset(
                name=f"segment.{key}",
                shape=(num_of_curves * 2,),
                maxshape=(None,),
                chunks=self.META_CHUNKSIZE,
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            self.curves_meta_buffer[f"segment.{key}"] = []
        self.seg_work = [
            (f"{k}=".encode(), self.curves_meta_datasets[f"segment.{k}"], self.curves_meta_buffer[f"segment.{k}"])
            for k in changing_segment_keys
        ]
        self.segment_search_terms = [f"{k}=".encode() for k in changing_segment_keys]

    def complete_saving(self, volume_channels: list[dict[str, str]]):
        """
        Resize datasets and finalize the saved file.

        Parameters
        ----------
        volume_channels : list[dict[str, str]]
            The list of channel dictionaries containing information about each channel.
        """
        # Add the last index to the indices datasets to mark the end of the last curve
        for volume_name, segments in self.volume_datasets.items():
            for direction in range(2):
                seg_name = f"Segment_{direction}"
                for chan in volume_channels:
                    chan_name = chan["name"]
                    current_dataset = segments[seg_name][chan_name]["Data"]
                    indices_dataset = segments[seg_name][chan_name]["Indices"]
                    indices_dataset[-1] = current_dataset.shape[0]

                    segments[seg_name][chan_name]["Data"].resize(
                        (self.volume_points_saved[volume_name][seg_name][chan_name],)
                    )

    def setup_volume(
        self,
        volume_name: str,
        predicted_points_per_channel_segment: dict[int, dict[str, int]],
        volume_dims: tuple[int, int],
        volume_channels: list[dict[str, str]],
    ):
        """
        Set up a dataset in the h5 file for saving volume data.

        Parameters
        ----------
        volume_name : str
            The name of the volume dataset to be created.
        predicted_points_per_channel_segment : dict[int, dict[str, int]]
            Predicted number of points per channel and segment.
        volume_dims : tuple[int, int]
            Dimensions of the volume.
        volume_channels : list[dict[str, str]]
            The list of channel dictionaries containing information about each channel.
        """
        assert self.h5file is not None, "existing h5 file must be passed or create_file called before setup_volume"
        self.curve_data_group = self.h5file.require_group("Curve_Data")
        volume_data_group = self.curve_data_group.require_group(f"{volume_name}_VOLM")
        self.volumes_dims[volume_name] = volume_dims
        curve_groups: dict[str, dict[str, h5py.Group]] = {"Data": {}, "Indices": {}}
        self.volume_datasets[volume_name] = {}
        self.volumes_data_buffer[volume_name] = {}
        self.volume_points_saved[volume_name] = {}
        self.volume_points_read[volume_name] = {}

        for direction in range(2):
            # For each segment direction, establish necessary group structure that will contain each channel dataset
            seg_name = f"Segment_{direction}"
            dir_group = volume_data_group.require_group(seg_name)
            self.volume_datasets[volume_name][seg_name] = {}
            self.volumes_data_buffer[volume_name][seg_name] = {}
            self.volume_points_saved[volume_name][seg_name] = {}
            self.volume_points_read[volume_name][seg_name] = {}
            # Create the Data and Indices subfolders and store their references
            curve_groups["Data"][seg_name] = dir_group.require_group("Data")
            curve_groups["Indices"][seg_name] = dir_group.require_group("Indices")
            for chan in volume_channels:
                self.volume_datasets[volume_name][seg_name][chan["name"]] = {}
                # For each channel, create an empty dataset
                self.volume_datasets[volume_name][seg_name][chan["name"]]["Data"] = curve_groups["Data"][
                    seg_name
                ].create_dataset(
                    name=chan["name"],
                    shape=(predicted_points_per_channel_segment[direction][chan["name"]],),
                    maxshape=(None,),
                    chunks=(self.DATA_CHUNKSIZE,),
                    dtype=np.float32,
                )
                self.volume_datasets[volume_name][seg_name][chan["name"]]["Indices"] = curve_groups["Indices"][
                    seg_name
                ].create_dataset(
                    name=chan["name"],
                    shape=(volume_dims[0] * volume_dims[1] + 1,),
                    maxshape=(None,),
                    chunks=(self.INDICES_CHUNKSIZE,),
                    dtype=np.int32,
                )
                self.volumes_data_buffer[volume_name][seg_name][chan["name"]] = {"Data": [], "Indices": []}
                self.volume_points_saved[volume_name][seg_name][chan["name"]] = 0
                self.volume_points_read[volume_name][seg_name][chan["name"]] = 0

    def save_curve_segment(
        self,
        volume_name: str,
        segment_data: np.ndarray,
        curve_num: int,
        direction: int,
        channel_name: str,
        num_of_curves: int,
    ) -> None:
        """
        Save a curve segment's data and indices to the h5 file.

        Parameters
        ----------
        volume_name : str
            The name of the volume dataset to which the curve segment belongs.
        segment_data : np.ndarray
            The curve segment's data to be saved.
        curve_num : int
            The number of the curve being saved.
        direction : int
            The direction of the curve segment (0 for trace, 1 for retrace).
        channel_name : str
            The name of the channel that the curve segment belongs to.
        num_of_curves : int
            The total number of curves in the dataset (used to determine when to flush buffer).
        """
        seg_name = f"Segment_{direction}"
        buf = self.volumes_data_buffer[volume_name][seg_name][channel_name]
        indices_set = self.volume_datasets[volume_name][seg_name][channel_name]["Indices"]

        # The starting index for this curve segment's data in the dataset is the
        # number of points already read for this channel and segment
        start_offset = self.volume_points_read[volume_name][seg_name][channel_name]

        buf["Data"].append(segment_data.astype(np.float32))
        self.volume_points_read[volume_name][seg_name][channel_name] += len(segment_data)

        # If the buffer is full or if this is the last curve segment, empty buffer into the dataset
        if len(buf["Data"]) >= self.BUFFER_SIZE or curve_num == num_of_curves - 1:
            filled_size = self.volume_points_saved[volume_name][seg_name][channel_name]
            data_set = self.volume_datasets[volume_name][seg_name][channel_name]["Data"]

            buffered_data = np.concatenate(buf["Data"])
            required_size = filled_size + len(buffered_data)
            if required_size > data_set.shape[0]:
                # Fetch and resize the existing dataset for this channel and segment to fit the new data
                data_set.resize((required_size,))

            # Add the buffer to the dataset
            data_set[filled_size:required_size] = buffered_data
            # Update the filled size for this channel and segment
            self.volume_points_saved[volume_name][seg_name][channel_name] = required_size
            # Clear the buffer
            buf["Data"].clear()

        # Append the new index to the indices buffer: note this is the STARTING index of the saved data
        buf["Indices"].append(start_offset)

        # If the indices buffer is full add it to the indices dataset and clear the buffer
        if len(buf["Indices"]) > 0 and len(buf["Indices"]) % self.BUFFER_SIZE == 0:
            indices_set[curve_num - self.BUFFER_SIZE + 1 : curve_num + 1] = buf["Indices"]
            buf["Indices"].clear()

        # Or if this is the last curve and there are still indices in the buffer
        elif len(buf["Indices"]) > 0 and curve_num == num_of_curves - 1:
            # Add the remaining indices to the indices dataset and clear the buffer
            items_in_buffer = len(buf["Indices"])
            indices_set[curve_num - items_in_buffer + 1 : curve_num + 1] = buf["Indices"]
            buf["Indices"].clear()

    def get_segment_search_terms(self) -> list[bytes]:
        """
        Get the list of segment search terms.

        Returns
        -------
        list[bytes]
            List of segment search terms.
        """
        return self.segment_search_terms

    def get_curve_search_terms(self) -> list[bytes]:
        """
        Get the list of curve search terms.

        Returns
        -------
        list[bytes]
            List of curve search terms.
        """
        return self.curve_search_terms

    def save_curve_meta_attr(self, curve_num: int, attr_idx: int, value: str, num_of_curves: int) -> None:
        """
        Save a curve metadata attribute.

        Parameters
        ----------
        curve_num : int
            The number of the curve.
        attr_idx : int
            The index of the attribute.
        value : str
            The value of the attribute to save.
        num_of_curves : int
            The total number of curves.
        """
        attr_name, meta_set, meta_buffer = self.curve_work[attr_idx]
        if meta_buffer is not None:
            meta_buffer.append(value)
            if len(meta_buffer) >= self.BUFFER_SIZE or curve_num == num_of_curves - 1:
                meta_set[curve_num - len(meta_buffer) + 1 : curve_num + 1] = meta_buffer
                meta_buffer.clear()
        else:
            logger.error(
                f"Metadata dataset for key {attr_name.decode('utf-8')} not found when trying to save "
                f"metadata for curve {curve_num}"
            )

    def save_segment_meta_attr(
        self, curve_num: int, direction: int, attr_idx: int, value: str, num_of_curves: int
    ) -> None:
        """
        Save a segment metadata attribute.

        Parameters
        ----------
        curve_num : int
            The number of the curve.
        direction : int
            The direction of the segment (0 or 1).
        attr_idx : int
            The index of the attribute.
        value : str
            The value of the attribute to save.
        num_of_curves : int
            The total number of curves.
        """
        attr_name, meta_set, meta_buffer = self.seg_work[attr_idx]
        if meta_buffer is not None:
            meta_buffer.append(value)
            if len(meta_buffer) >= self.BUFFER_SIZE or curve_num == num_of_curves - 1:
                idx = curve_num * 2 + direction
                meta_set[idx - len(meta_buffer) + 1 : idx + 1] = meta_buffer
                meta_buffer.clear()
        else:
            logger.error(
                f"Metadata dataset for key {attr_name.decode('utf-8')} not found when trying to save "
                f"metadata for curve {curve_num}, direction {direction}"
            )

    def save_global_meta(self, global_meta: dict[str, Any], size_x: float, size_y: float, shape_x: int, shape_y: int):
        """
        Save global metadata attributes.

        Parameters
        ----------
        global_meta : dict[str, Any]
            Dictionary of global metadata.
        size_x : float
            The size of the image in the x-direction.
        size_y : float
            The size of the image in the y-direction.
        shape_x : int
            The number of pixels in the x-direction.
        shape_y : int
            The number of pixels in the y-direction.
        """
        assert self.global_meta_group is not None, "setup_curve_data_structure must be called first"
        assert self.h5file is not None, "existing h5 file must be passed or create_file called before setup"
        for key, value in global_meta.items():
            self.global_meta_group.attrs[key] = str(value).encode("utf-8")

        # Save data required for reading the h5 file as a normal image file
        meas_grp = self.h5file.require_group("Measurement_000")
        # Save dimensions data
        meas_grp.attrs["position-pattern.grid.ulength"] = size_x
        meas_grp.attrs["position-pattern.grid.ilength"] = shape_x
        meas_grp.attrs["position-pattern.grid.vlength"] = size_y
        meas_grp.attrs["position-pattern.grid.jlength"] = shape_y
        meas_grp.attrs["timing-settings.scanRate"] = 1.0  # Dummy value to satisfy reader

    def save_image(self, image_data: np.ndarray, image_name: str, z_unit: str, idx: int):
        """
        Save an image dataset to the h5 file.

        Parameters
        ----------
        image_data : np.ndarray
            The image data to be saved.
        image_name : str
            The name of the image dataset.
        z_unit : str
            The z-axis unit of the image.
        idx : int
            The index of the image channel.
        """
        assert (
            self.h5file is not None
        ), "existing h5 file must be passed or create_file called before setup_curve_data_structure"
        meas_grp = self.h5file.require_group("Measurement_000")
        chan_grp = meas_grp.require_group(f"Channel_{make_num_min_characters(idx)}")
        # Extract name and retrace information from the channel name
        if image_name and "_" in str(image_name):
            base_name, trace_dir = str(image_name).rsplit("_", 1)
            is_retrace = "true" if trace_dir.lower() == "retrace" else "false"
        else:
            base_name = image_name
            is_retrace = "false"

        # Add the necessary attributes to the channel group
        chan_grp.attrs["channel.name"] = base_name.encode("utf-8")
        chan_grp.attrs["retrace"] = is_retrace.encode("utf-8")
        chan_grp.attrs["net-encoder.scaling.multiplier"] = 1.0
        chan_grp.attrs["net-encoder.scaling.offset"] = 0.0

        # Format name and reshape image (flattened frame stack)
        dataset_name = image_name.split("_")[0].capitalize()
        # Include all the channels including the calculated channel
        chan_grp.attrs["net-encoder.scaling.unit.unit"] = z_unit.encode("utf-8")
        frame_stack = image_data.flatten().reshape(-1, 1)

        # Update/ replace the channels dataset
        if dataset_name in chan_grp:
            del chan_grp[dataset_name]
        chan_grp.create_dataset(dataset_name, data=frame_stack)


def make_num_min_characters(num: int, min_chars: int = 3):
    """
    Zero-pad an integer to a minimum number of characters.

    Parameters
    ----------
    num : int
        The integer to pad.
    min_chars : int
        The minimum number of characters the resulting string should have. Default is 3.

    Returns
    -------
    str
        The zero-padded string.
    """
    string_num = str(num)
    if len(string_num) >= min_chars:
        return string_num
    return "0" * (min_chars - len(string_num)) + string_num
