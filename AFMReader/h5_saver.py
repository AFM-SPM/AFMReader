# ruff: noqa: C901
# pylint: disable=too-many-instance-attributes,too-many-positional-arguments,too-many-locals
"""Module for saving AFM reader data to HDF5 files."""

from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from AFMReader import __version__
from AFMReader.logging import logger
from AFMReader.data_classes import CurvesVolume
from AFMReader.io import coerce_metadata_value


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

    def __init__(self, filepath: Path | None = None, h5file: h5py.File | None = None):
        """
        Initialise H5Saver.

        Parameters
        ----------
        filepath : Path | None
            The path to the h5 file.
        h5file : h5py.File | None
            The h5 file object that will be written to.
        """
        # The path to the h5 file
        self.filepath = filepath
        # The h5 file object that will be written to
        self.h5file = h5file

        if self.filepath is None and self.h5file is None:
            raise ValueError("Either filepath or h5file must be provided.")

        # A nested dictionary to hold curve volume float data in memory before writing to the h5 file.
        # This is structured as [volume_name][segment][channel][data/indices]. Each list can get no larger
        # than BUFFER_SIZE before being written to the h5 file and cleared from memory
        self.volumes_data_buffer: dict[str, dict[str, dict[str, dict[str, list]]]] = {}

        # A nested dictionary to hold the h5 datasets for each volume to be saved, structured
        # as [volume_name][segment][channel][data/indices]
        self.volume_datasets: dict[str, dict[str, dict[str, dict[str, h5py.Dataset]]]] = {}
        self.volumes_dims: dict[str, tuple[int, int]] = {}

        self.curves_meta_buffer: dict[str, dict[str, list]] = {}
        self.curves_meta_datasets: dict[str, dict[str, h5py.Dataset]] = {}

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
        self.curve_work: dict[str, list[tuple[bytes, h5py.Dataset, list[str]]]] = {}
        self.curve_search_terms: dict[str, list[bytes]] = {}
        self.seg_work: dict[str, list[tuple[bytes, h5py.Dataset, list[str]]]] = {}
        self.segment_search_terms: dict[str, list[bytes]] = {}

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

    def create_file(self, source: str | None = None) -> h5py.File:
        """
        Create the h5 file and write initial global attributes.

        Parameters
        ----------
        source : str | None
            The source type of the data (the original file type if converting).

        Returns
        -------
        h5py.File
            The created h5 file object.
        """
        assert self.filepath is not None, "Filepath must be provided to create an h5 file."
        self.h5file = h5py.File(self.filepath, "a")
        self.h5file.attrs["AFMReader_version"] = __version__
        self.h5file.attrs["created_by"] = "AFMReader"
        self.h5file.attrs["created_on"] = datetime.now().isoformat()
        self.h5file.attrs["source_type"] = source if source is not None else "new"
        return self.h5file

    def setup_curves_group(self, channel_units: dict[str, str] | None = None):
        """
        Set up the HDF5 groups used to store curve data.

        Parameters
        ----------
        channel_units : dict[str, str] | None, optional
            A dictionary mapping channel names to their units.
        """
        assert (
            self.h5file is not None
        ), "existing h5 file must be passed or create_file called before setup_curves_group"
        # Create the main group for the curve data that all the curve data will be in
        self.curve_data_group = self.h5file.require_group("Curve_Data")

        # Establish empty groups for global metadata
        self.global_meta_group = self.curve_data_group.require_group("Global_Metadata")

        if channel_units is not None:
            for channel_name, unit in channel_units.items():
                self.global_meta_group.attrs[f"channel.unit.{channel_name}"] = unit

    def complete_saving(self, volume: CurvesVolume):
        """
        Resize datasets and finalize the saved file.

        Parameters
        ----------
        volume : CurvesVolume
            The CurvesVolume instance containing the curve data for each pixel.
        """
        assert self.h5file is not None, "existing h5 file must be passed or create_file called before complete_saving"
        # Add the last index to the indices datasets to mark the end of the last curve
        for segment_name in volume.metadata.segment_names:
            for channel_name in volume.metadata.channel_units:
                indices_dataset = self.volume_datasets[volume.name][segment_name][channel_name]["Indices"]
                indices_dataset[-1] = self.volume_points_saved[volume.name][segment_name][channel_name]

                self.volume_datasets[volume.name][segment_name][channel_name]["Data"].resize(
                    (self.volume_points_saved[volume.name][segment_name][channel_name],)
                )
        self.h5file.flush()

    def get_curves_sample(self, shape_x: int, shape_y: int, minimum_sample_size: int = 20):
        """
        Get a sample of curve numbers distributed evenly across the dataset.

        Parameters
        ----------
        shape_x : int
            The number of columns in the image.
        shape_y : int
            The number of rows in the image.
        minimum_sample_size : int, optional
            The minimum number of curves to sample. Default is 20.

        Returns
        -------
        range:
            A range object representing the sampled curve numbers.
        """
        num_of_curves = shape_x * shape_y
        # Check evenly spaced curves in the dataset to sample metadata without having to load every curve
        step = 1 if num_of_curves <= minimum_sample_size else num_of_curves // minimum_sample_size
        # If the step is equal to a shape dimension, we might just go down the row or column
        while step in [shape_x, shape_y] and step > 1:
            # So make the step slightly smaller (more checks) to ensure we get a good sample
            step -= 1
        return range(0, num_of_curves, step)

    def predict_total_points(self, curves_volume: CurvesVolume) -> dict[str, dict[str, int]]:
        """
        Predict the total number of points for each channel and segment.

        This is done by sampling a subset of curves and extrapolating based on the maximum number
        of points found in the sample.

        Parameters
        ----------
        curves_volume : CurvesVolume
            The CurvesVolume instance containing the curve data for each pixel.

        Returns
        -------
        dict:
            A dictionary containing the predicted total points for each channel and segment.
        """
        # Get a sample of curve (indices)
        shape_y, shape_x = curves_volume.shape
        curves_to_check = self.get_curves_sample(shape_x, shape_y)
        num_of_curves = shape_y * shape_x
        points_for_channel_segment: dict[str, dict[str, list[int]]] = {}

        # Iterate through the segments, channels and our curve indices
        for segment_name in curves_volume.metadata.segment_names:
            points_for_channel_segment[segment_name] = {}
            for channel in curves_volume.metadata.channel_units:
                points_for_channel_segment[segment_name][channel] = []
        for curve_num in curves_to_check:
            # Loop until we successfully retrieve some data
            while True:
                try:
                    # Count points in extracted data
                    sampled_curve = curves_volume[curve_num // shape_x, curve_num % shape_x]
                    for segment_name in curves_volume.metadata.segment_names:
                        for channel in points_for_channel_segment[segment_name]:
                            raw_array = sampled_curve[channel][segment_name]
                            points_for_channel_segment[segment_name][channel].append(len(raw_array))
                    break

                except KeyError:
                    # If the file doesn't exist for this curve, check the next curve so we don't just get
                    # a smaller sample
                    if curve_num + 1 >= num_of_curves:
                        # If we've gone past the number of curves, stop checking
                        break
                    curve_num += 1
                    continue
        predicted_points_per_channel_segment: dict[str, dict[str, int]] = {}
        # Calculate a prediction for total number of points based on maximum number of points then assuming
        # maximum points throughout data is no more than 10% higher
        for segment_name in curves_volume.metadata.segment_names:
            predicted_points_per_channel_segment[segment_name] = {}
            for channel in points_for_channel_segment[segment_name]:
                predicted_points_per_channel_segment[segment_name][channel] = (
                    int(np.max(points_for_channel_segment[segment_name][channel]) * 1.1) * num_of_curves
                )
        return predicted_points_per_channel_segment

    def setup_volume(
        self,
        curves_volume: CurvesVolume,
        changing_curve_keys: set[str] | None = None,
        changing_segment_keys: set[str] | None = None,
    ) -> h5py.Group:
        """
        Set up a dataset in the h5 file for saving volume data.

        Parameters
        ----------
        curves_volume : CurvesVolume
            The CurvesVolume instance containing the curve data for each pixel.
        changing_curve_keys : set[str] | None
            Curve metadata keys that can vary between curves.
        changing_segment_keys : set[str] | None
            Segment metadata keys that can vary between segments.

        Returns
        -------
        h5py.Group
            The HDF5 group created or retrieved for the volume data.
        """
        if changing_curve_keys is None:
            changing_curve_keys = set()
        if changing_segment_keys is None:
            changing_segment_keys = set()
        assert self.h5file is not None, "existing h5 file must be passed or create_file called before setup_volume"
        self.curve_data_group = self.h5file.require_group("Curve_Data")
        volume_group = self.curve_data_group.require_group(f"{curves_volume.name}_VOLM")

        volume_meta_group = volume_group.require_group("Metadata")
        volume_data_group = volume_group.require_group("Data")
        self.curves_meta_datasets[curves_volume.name] = {}
        self.curves_meta_buffer[curves_volume.name] = {}

        for key in changing_curve_keys:
            self.curves_meta_datasets[curves_volume.name][f"curve.{key}"] = volume_meta_group.create_dataset(
                name=f"curve.{key}",
                shape=(len(curves_volume),),
                maxshape=(None,),
                chunks=self.META_CHUNKSIZE,
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            self.curves_meta_buffer[curves_volume.name][f"curve.{key}"] = []
        self.curve_work[curves_volume.name] = [
            (
                f"{k}=".encode(),
                self.curves_meta_datasets[curves_volume.name][f"curve.{k}"],
                self.curves_meta_buffer[curves_volume.name][f"curve.{k}"],
            )
            for k in changing_curve_keys
        ]
        self.curve_search_terms[curves_volume.name] = [f"{k}=".encode() for k in changing_curve_keys]
        for key in changing_segment_keys:
            self.curves_meta_datasets[curves_volume.name][f"segment.{key}"] = volume_meta_group.create_dataset(
                name=f"segment.{key}",
                shape=(len(curves_volume) * len(curves_volume.metadata.segment_names),),
                maxshape=(None,),
                chunks=self.META_CHUNKSIZE,
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            self.curves_meta_buffer[curves_volume.name][f"segment.{key}"] = []
        self.seg_work[curves_volume.name] = [
            (
                f"{k}=".encode(),
                self.curves_meta_datasets[curves_volume.name][f"segment.{k}"],
                self.curves_meta_buffer[curves_volume.name][f"segment.{k}"],
            )
            for k in changing_segment_keys
        ]
        self.segment_search_terms[curves_volume.name] = [f"{k}=".encode() for k in changing_segment_keys]

        self.volumes_dims[curves_volume.name] = curves_volume.shape
        curve_groups: dict[str, dict[str, h5py.Group]] = {"Data": {}, "Indices": {}}
        self.volume_datasets[curves_volume.name] = {}
        self.volumes_data_buffer[curves_volume.name] = {}
        self.volume_points_saved[curves_volume.name] = {}
        self.volume_points_read[curves_volume.name] = {}

        predicted_points_per_channel_segment = self.predict_total_points(curves_volume)
        shape_y, shape_x = curves_volume.shape

        for segment_name in curves_volume.metadata.segment_names:
            # For each segment, establish the group structure that will contain each channel dataset.
            dir_group = volume_data_group.require_group(segment_name)
            self.volume_datasets[curves_volume.name][segment_name] = {}
            self.volumes_data_buffer[curves_volume.name][segment_name] = {}
            self.volume_points_saved[curves_volume.name][segment_name] = {}
            self.volume_points_read[curves_volume.name][segment_name] = {}
            # Create the Data and Indices subfolders and store their references
            curve_groups["Data"][segment_name] = dir_group.require_group("Data")
            curve_groups["Indices"][segment_name] = dir_group.require_group("Indices")
            for chan in curves_volume.metadata.channel_units:
                self.volume_datasets[curves_volume.name][segment_name][chan] = {}
                # For each channel, create an empty dataset
                self.volume_datasets[curves_volume.name][segment_name][chan]["Data"] = curve_groups["Data"][
                    segment_name
                ].create_dataset(
                    name=chan,
                    shape=(predicted_points_per_channel_segment[segment_name][chan],),
                    maxshape=(None,),
                    chunks=(self.DATA_CHUNKSIZE,),
                    dtype=np.float32,
                )
                self.volume_datasets[curves_volume.name][segment_name][chan]["Indices"] = curve_groups["Indices"][
                    segment_name
                ].create_dataset(
                    name=chan,
                    shape=(shape_y * shape_x + 1,),
                    maxshape=(None,),
                    chunks=(self.INDICES_CHUNKSIZE,),
                    dtype=np.int32,
                )
                self.volumes_data_buffer[curves_volume.name][segment_name][chan] = {"Data": [], "Indices": []}
                self.volume_points_saved[curves_volume.name][segment_name][chan] = 0
                self.volume_points_read[curves_volume.name][segment_name][chan] = 0
        return volume_data_group

    def save_curve_segment(
        self,
        volume_name: str,
        segment_data: np.ndarray,
        curve_num: int,
        segment_name: str,
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
        segment_name : str
            The name of the segment that the curve segment belongs to.
        channel_name : str
            The name of the channel that the curve segment belongs to.
        num_of_curves : int
            The total number of curves in the dataset (used to determine when to flush buffer).
        """
        buf = self.volumes_data_buffer[volume_name][segment_name][channel_name]
        indices_set = self.volume_datasets[volume_name][segment_name][channel_name]["Indices"]

        # The starting index for this curve segment's data in the dataset is the
        # number of points already read for this channel and segment
        start_offset = self.volume_points_read[volume_name][segment_name][channel_name]

        buf["Data"].append(segment_data.astype(np.float32))
        self.volume_points_read[volume_name][segment_name][channel_name] += len(segment_data)

        # If the buffer is full or if this is the last curve segment, empty buffer into the dataset
        if len(buf["Data"]) >= self.BUFFER_SIZE or curve_num == num_of_curves - 1:
            filled_size = self.volume_points_saved[volume_name][segment_name][channel_name]
            data_set = self.volume_datasets[volume_name][segment_name][channel_name]["Data"]

            buffered_data = np.concatenate(buf["Data"])
            required_size = filled_size + len(buffered_data)
            if required_size > data_set.shape[0]:
                # Fetch and resize the existing dataset for this channel and segment to fit the new data
                data_set.resize((required_size,))

            # Add the buffer to the dataset
            data_set[filled_size:required_size] = buffered_data
            # Update the filled size for this channel and segment
            self.volume_points_saved[volume_name][segment_name][channel_name] = required_size
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

    def save_curve(
        self,
        curve_data: dict[str, dict[str, np.ndarray]],
        curve_num: int,
        num_of_curves: int,
        volume_name: str,
        segment_names: list[str],
    ):
        """
        Save a curve's data and metadata to the h5 file.

        Parameters
        ----------
        curve_data : dict
            The curve data to be saved.
        curve_num : int
            The number of the curve being saved.
        num_of_curves : int
            The total number of curves in the dataset (used to determine when to flush buffer).
        volume_name : str
            The name of the volume to which the curve belongs.
        segment_names : list[str]
            Segment names to save for the curve.
        """
        for segment_name in segment_names:
            for channel_name, segment_data in curve_data.items():
                self.save_curve_segment(
                    volume_name=volume_name,
                    segment_data=segment_data[segment_name],
                    curve_num=curve_num,
                    segment_name=segment_name,
                    channel_name=channel_name,
                    num_of_curves=num_of_curves,
                )

    def get_segment_search_terms(self, volume_name: str) -> list[bytes]:
        """
        Get the list of segment search terms.

        Parameters
        ----------
        volume_name : str
            The name of the volume.

        Returns
        -------
        list[bytes]
            List of segment search terms.
        """
        return self.segment_search_terms[volume_name]

    def get_curve_search_terms(self, volume_name: str) -> list[bytes]:
        """
        Get the list of curve search terms.

        Parameters
        ----------
        volume_name : str
            The name of the volume.

        Returns
        -------
        list[bytes]
            List of curve search terms.
        """
        return self.curve_search_terms[volume_name]

    def save_curve_meta_attr(
        self, curve_num: int, attr_idx: int, value: Any, volume_name: str, num_of_curves: int
    ) -> None:
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
        volume_name : str
            The name of the volume to which the curve belongs.
        num_of_curves : int
            The total number of curves.
        """
        attr_name, meta_set, meta_buffer = self.curve_work[volume_name][attr_idx]
        if meta_buffer is not None:
            meta_buffer.append(str(value))
            if len(meta_buffer) >= self.BUFFER_SIZE or curve_num == num_of_curves - 1:
                meta_set[curve_num - len(meta_buffer) + 1 : curve_num + 1] = meta_buffer
                meta_buffer.clear()
        else:
            logger.error(
                f"Metadata dataset for key {attr_name.decode('utf-8')} not found when trying to save "
                f"metadata for curve {curve_num}"
            )

    # pylint: disable-next=too-many-arguments
    def save_segment_meta_attr(
        self,
        curve_num: int,
        segment_idx: int,
        attr_idx: int,
        value: Any,
        volume_name: str,
        num_of_curves: int,
        num_of_segments: int,
    ) -> None:
        """
        Save a segment metadata attribute.

        Parameters
        ----------
        curve_num : int
            The number of the curve.
        segment_idx : int
            The index of the segment.
        attr_idx : int
            The index of the attribute.
        value : str
            The value of the attribute to save.
        volume_name : str
            The name of the volume to which the segment belongs.
        num_of_curves : int
            The total number of curves.
        num_of_segments : int
            The number of segments per curve.
        """
        attr_name, meta_set, meta_buffer = self.seg_work[volume_name][attr_idx]
        if meta_buffer is not None:
            meta_buffer.append(str(value))
            if len(meta_buffer) >= self.BUFFER_SIZE or curve_num == num_of_curves - 1:
                idx = curve_num * num_of_segments + segment_idx
                meta_set[idx - len(meta_buffer) + 1 : idx + 1] = meta_buffer
                meta_buffer.clear()
        else:
            logger.error(
                f"Metadata dataset for key {attr_name.decode('utf-8')} not found when trying to save "
                f"metadata for curve {curve_num}, segment {segment_idx}"
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
            value = coerce_metadata_value(value)
            try:
                self.global_meta_group.attrs[key] = value
            except TypeError:
                self.global_meta_group.attrs[key] = str(value)

        # Save data required for reading the h5 file as a normal image file
        meas_grp = self.h5file.require_group("Measurement_000")
        # Save dimensions data
        meas_grp.attrs["position-pattern.grid.ulength"] = size_x
        meas_grp.attrs["position-pattern.grid.ilength"] = shape_x
        meas_grp.attrs["position-pattern.grid.vlength"] = size_y
        meas_grp.attrs["position-pattern.grid.jlength"] = shape_y
        meas_grp.attrs["timing-settings.scanRate"] = 1.0  # Dummy value to satisfy reader

    def close_file(self):
        """Close the h5 file if it is open."""
        if self.h5file is not None:
            if self.h5file:
                self.h5file.close()
            self.h5file = None

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


def find_unused_filename(original_path: Path, temp: bool = False) -> Path:
    """
    Find an unused filename by appending a number to the base name.

    Parameters
    ----------
    original_path : Path
        The original file path used to derive the HDF5 file name.
    temp : bool
        Whether to create a temporary file name. Default is False.

    Returns
    -------
    Path
        An unused HDF5 file path.
    """
    if temp:
        h5_path = original_path.parent / f"temp_{original_path.stem}.h5-jpk"
    else:
        h5_path = original_path.parent / f"{original_path.stem}.h5-jpk"
    # Determine the path for the H5 file, ensuring it does not overwrite an existing file
    i = 0
    while h5_path.exists():
        if temp:
            h5_path = original_path.parent / f"temp_{original_path.stem}_{i}.h5-jpk"
        else:
            h5_path = original_path.parent / f"{original_path.stem}_{i}.h5-jpk"
        i += 1
    return h5_path
