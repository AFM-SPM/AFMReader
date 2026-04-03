import os
from pathlib import Path
from contextlib import nullcontext
import io
import zipfile
import time

import numpy as np
import javaproperties
import h5py
import psutil
from pympler import asizeof

from AFMReader.logging import logger
from AFMReader import jpk


class LazyCurveData:
    """A proxy class that behaves like a 2D list but fetches .dat files on demand."""

    def __init__(self, filepath, shape_x, shape_y, channel_scaling, archive, flip_image: bool = True):
        self.filepath = filepath
        self.shape_x = shape_x
        self.shape_y = shape_y
        self.channel_scaling = channel_scaling
        self.archive = archive
        self.flip_image = flip_image

    def __getitem__(self, y: int):
        # Return a row proxy to handle the second index [x]
        class RowProxy:
            def __init__(self, parent, y):
                self.parent = parent
                self.y = y

            def __getitem__(self, x: int):
                return self.parent._fetch_curve(self.y, x)

        return RowProxy(self, y)

    def __iter__(self):
        for y in range(self.shape_y):
            for x in range(self.shape_x):
                yield self._fetch_curve(y, x)

    def _fetch_curve(self, y: int, x: int):
        if y < 0 or y >= self.shape_y or x < 0 or x >= self.shape_x:
            raise IndexError(f"Curve index out of bounds: ({x}, {y})")
        if self.flip_image:
            y = self.shape_y - 1 - y
        curve_num = y * self.shape_x + x
        curve_data = {}

        for chan_name, scale in self.channel_scaling.items():
            curve_data[chan_name] = {}
            for direction in (0, 1):
                dat_path = f"index/{curve_num}/segments/{direction}/channels/{chan_name}.dat"
                try:
                    # Access the file directly without re-parsing the ZIP directory
                    with self.archive.open(dat_path) as f:
                        raw_array = np.frombuffer(f.read(), dtype=">i4")
                        curve_data[chan_name][f"Segment_{direction}"] = (raw_array * scale["multiplier"]) + scale[
                            "offset"
                        ]
                except KeyError:
                    pass  # File doesn't exist for this segment

        return curve_data

    def load_all_curves(self):
        """
        Eagerly loads all curve data into a 2D list structure matching the image dimensions.

        This can be used if the user wants to have all the curve data available at once, but it is not recommended
        for large datasets as it will consume a lot of memory. In this case, it is not notably faster as the zip
        structure means each curve is effectively loaded individually anyway
        """
        all_curve_data = [[None for _ in range(self.shape_x)] for _ in range(self.shape_y)]
        for y in range(self.shape_y):
            for x in range(self.shape_x):
                all_curve_data[y][x] = self._fetch_curve(y, x)
        # TODO may be good to just return self here as not faster and lots of memory
        # return self
        return all_curve_data

    def close(self):
        self.archive.close()


class LazyCurveMetadata:
    """A proxy class that fetches header.properties files on demand."""

    def __init__(self, filepath, top_level_meta, archive, shape_x: int, shape_y: int, flip_image: bool = True):
        self.filepath = filepath
        # Expose top_level so the frontend can still do `raw_metadata["top_level"]`
        self.top_level = top_level_meta
        self.archive = archive
        self.shape_x = shape_x
        self.shape_y = shape_y
        self.flip_image = flip_image

    def __getitem__(self, key):
        if key == "top_level":
            return self.top_level
        elif key == "curves":
            return LazyMetaProxy(self.filepath, "curve", self.archive, self.shape_x, self.shape_y, self.flip_image)
        elif key == "segments":
            return LazyMetaProxy(self.filepath, "segment", self.archive, self.shape_x, self.shape_y, self.flip_image)
        raise KeyError(key)


class LazyMetaProxy:
    def __init__(self, filepath, meta_type, archive, shape_x: int, shape_y: int, flip_image: bool = True):
        self.filepath = filepath
        self.meta_type = meta_type
        self.archive = archive
        self.shape_x = shape_x
        self.shape_y = shape_y
        self.flip_image = flip_image

    def __getitem__(self, y: int):
        class RowProxy:
            def __init__(self, parent, y):
                self.parent = parent
                self.y = y

            def __getitem__(self, x):
                if self.parent.meta_type == "curve":
                    return self.parent._fetch_meta(self.y, x)
                elif self.parent.meta_type == "segment":

                    class SegmentMetaProxy:
                        def __init__(self, parent, y, x):
                            self.parent = parent
                            self.y = y
                            self.x = x

                        def __getitem__(self, direction):
                            return self.parent.parent._fetch_meta(self.y, self.x, direction)

                    return SegmentMetaProxy(self, self.y, x)

        return RowProxy(self, y)

    def _fetch_meta(self, y: int, x: int, direction: int = None):
        if y < 0 or y >= self.shape_y or x < 0 or x >= self.shape_x:
            raise IndexError(f"Curve index out of bounds: ({x}, {y})")
        if self.flip_image:
            y = self.shape_y - 1 - y
        idx = (y * self.shape_x) + x
        if self.meta_type == "curve":
            path = f"index/{idx}/header.properties"
        else:
            if direction is None:
                raise ValueError("Direction must be provided for segment metadata")
            path = f"index/{idx}/segments/{direction}/segment-header.properties"

        try:
            with self.archive.open(path) as f:
                meta_dict = {".".join(k.split(".")[1:]): v for k, v in javaproperties.load(f).items()}
        except KeyError:
            meta_dict = {}

        return meta_dict


def _get_channel_scaling(props, channel_index):
    """
    Parses the JPK properties dictionary to find the cumulative multiplier
    and offset for a specific channel index (e.g., '1' for vDeflection).

    Parameters
    ----------
    props : dict
        The properties dictionary loaded from the JPK file.
    channel_index : str
        The index of the channel to find the scaling for (e.g., '1' for vDeflection).

    Returns
    -------
    final_multiplier : float
        The cumulative multiplier for the specified channel.
    final_offset : float
        The cumulative offset for the specified channel.
    unit : str
        The unit of the channel.
    """
    prefix = f"lcd-info.{channel_index}."

    current_slot = props.get(f"{prefix}conversion-set.conversions.default")

    if not current_slot:
        mult = float(props.get(f"{prefix}encoder.scaling.multiplier", "1.0"))
        off = float(props.get(f"{prefix}encoder.scaling.offset", "0.0"))
        unit = props.get(f"{prefix}encoder.scaling.unit.unit", "Unknown")
        return mult, off, unit

    cumulative_multiplier = 1.0
    cumulative_offset = 0.0
    unit = props.get(f"{prefix}conversion-set.conversion.{current_slot}.scaling.unit.unit")

    while current_slot:
        slot_prefix = f"{prefix}conversion-set.conversion.{current_slot}."

        if f"{slot_prefix}scaling.multiplier" in props:
            m = float(props[f"{slot_prefix}scaling.multiplier"])
            c = float(props[f"{slot_prefix}scaling.offset"])

            cumulative_offset = (cumulative_multiplier * c) + cumulative_offset
            cumulative_multiplier *= m

            current_slot = props.get(f"{slot_prefix}base-calibration-slot")

            if current_slot == props.get(f"{prefix}conversion-set.conversions.base"):
                break
        else:
            break

    enc_m = float(props.get(f"{prefix}encoder.scaling.multiplier", "1.0"))
    enc_c = float(props.get(f"{prefix}encoder.scaling.offset", "0.0"))

    final_multiplier = cumulative_multiplier * enc_m
    final_offset = (cumulative_multiplier * enc_c) + cumulative_offset
    if not unit:
        unit = props.get(f"{prefix}encoder.scaling.unit.unit", "Unknown")

    return final_multiplier, final_offset, unit


class jpk_qi_loader:
    """Class for readability and improving modularity in the load jpk qi data function"""

    def __init__(
        self,
        filepath: Path | str,
        channel: str | None = None,
        config_path: Path | str | None = None,
        flip_image: bool | None = True,
        save_as_h5: bool = False,
    ):
        """
        Initializes the loader with the provided parameters.

        Parameters
        ----------
        filepath : Path | str
            The path to the .jpk-qi file to be loaded.
        channel : str | None, optional
            The specific channel to be extracted from the file (e.g., "measuredHeight"). Default is None.
        config_path : Path | str | None, optional
            The path to the configuration file, if any. Default is None.
        flip_image : bool | None, optional
            Whether to flip the image vertically. Default is True.
        save_as_h5 : bool, optional
            Whether to save the loaded data as an H5 file. Default is False.
        """

        self.filepath = Path(filepath)
        self.channel = channel
        self.config_path = config_path
        self.flip_image = flip_image
        self.save_as_h5 = save_as_h5
        logger.debug(f"Initialized JPK QI loader for file {self.filepath} with channel {self.channel}")
        # Open the ZIP archive once and keep it open for the duration of the loading process to improve performance when accessing multiple files within the archive
        self.qi_archive = zipfile.ZipFile(self.filepath, "r")
        logger.debug(f"Opened JPK QI archive at {self.filepath}")
        self.namelist = self.qi_archive.namelist()
        # Set path to the .jpk-qi-image file within the archive for later use
        self.path_to_image = None

        # Chunk size for H5 datasets
        self.DATA_CHUNKSIZE = 512 * 1024
        # Chunk size for indicies datasets
        self.INDICIES_CHUNKSIZE = 64 * 1024
        # Chunk size for metadata datasets (if needed)
        self.META_CHUNKSIZE = 64 * 1024
        # Maximum number of curves to check for changing metadata keys (to avoid checking every curve in large datasets)
        self.MAX_CURVE_CHECKS = 20
        # Number of curves to hold in buffer
        self.BUFFER_SIZE = 500

        # Initialize key attributes that will be returned / accessed frequently

        # Just the top level metadata extracted from the header files
        self.top_level_meta = {}
        # A dictionary containing all metadata, splitting the top level metadata and the metadata for each curve and segment
        self.full_metadata = {}
        # A 2D list of curve data dictionaries, where each dictionary contains the data for all channels and segments
        self.all_curve_data = None
        # A lookup for channel name to unit to be returned
        self.channels_units = {}
        # The list of channels for the segments with their scaling information extracted from the shared header
        self.segment_channels = []
        self.curve_meta = {}
        self.segment_meta = {}
        # Define the image shape and size attributes
        self.size_x, self.size_y, self.shape_x, self.shape_y = None, None, None, None
        self.failed_curves = set()

        # Timing counters for performance monitoring
        self.t_load_data = 0.0
        self.t_proc_data = 0.0
        self.t_save_data = 0.0
        self.t_load_meta = 0.0
        self.t_proc_meta = 0.0
        self.t_save_meta = 0.0
        self.t_changing_keys = 0.0

        # Instantiate containers for data to be saved (so an exception is not caused if not saving)
        self.curve_groups = None

    def get_available_channels(self):
        """
        Retrieves the available channels from the .jpk-qi-image file within the archive, and adds any additional calculated channels.

        Returns
        -------
        channels : list
            A list of available channels including the calculated channels.
        metadata_options : dict
            A dictionary of options for what metadata to return
        """

        # Look for the jpk-qi-image file in the archive
        if self.path_to_image is None:
            for file_name in self.namelist:
                if file_name.endswith(".jpk-qi-image"):
                    self.path_to_image = file_name

        # Add the channels which exist in the jpk-qi-image file
        with self.qi_archive.open(self.path_to_image, "r") as image_file:
            channels = jpk._get_jpk_channels(
                file=image_file, filename=self.filepath.stem, file_path=self.filepath / Path(self.path_to_image)
            )
        return channels, {"save_as_h5": bool}

    def load(
        self,
        channel: str | None = None,
        config_path: Path | str | None = None,
        flip_image: bool | None = True,
        save_as_h5: bool | None = None,
    ) -> tuple[np.ndarray, float, dict] | tuple[np.ndarray, float]:
        """
        Loads the .jpk-qi-data file

        Parameters
        ----------
        channel : str | None, optional
            The specific channel to be extracted from the file (e.g., "measuredHeight"). If None, the default channel will be used. Default is None.
        config_path : Path | str | None, optional
            Path to the configuration file. If None, the default configuration will be used. Default is None.
        flip_image : bool | None, optional
            Whether to flip the image. If None, the default behavior will be used. Default is True.
        save_as_h5 : bool, optional
            Whether to save the data as an H5 file. Default is False.

        Returns
        -------
        tuple
            A tuple containing the image data (numpy.ndarray), the pixel to nanometre scaling factor (float), and optionally the curve data (dict) if available.
        """

        # Update instance attributes based on provided parameters, largely so loader can be called to get channels without setting a channel
        self.channel = channel if channel else self.channel
        self.config_path = config_path if config_path else self.config_path
        self.flip_image = flip_image if flip_image is not None else self.flip_image
        self.save_as_h5 = save_as_h5 if save_as_h5 is not None else self.save_as_h5

        logger.info(f"Loading JPK QI data from {self.filepath} with channel {self.channel}")
        self.extract_global_metadata()

        self.parse_dimension_data()

        # Setup H5 Data structures if needed
        if self.save_as_h5:
            self.save_to_h5()

        # Establish the lazy loading structures for curve data and metadata. Note how lazy structure is used even if
        # all the data has been accessed and saved to H5 to prevent excessive memory usage
        self.full_metadata = LazyCurveMetadata(
            self.filepath, self.top_level_meta, self.qi_archive, self.shape_x, self.shape_y, flip_image=self.flip_image
        )
        self.all_curve_data = LazyCurveData(
            self.filepath, self.shape_x, self.shape_y, self.channel_scaling, self.qi_archive, flip_image=self.flip_image
        )

        # Load the image
        self.image, _ = self.get_image()

        # Save a lite form of the images (precalculated) if saving to a file
        if self.save_as_h5:
            self.save_lite_data()

        if self.all_curve_data:
            return (self.image, self.px2nm, (self.all_curve_data, self.channels_units, self.full_metadata))

        return self.image, self.px2nm

    def output_summary(self):
        """
        Outputs a summary of the loading process, including any failed curve loads and their details.
        """
        if self.failed_curves:
            logger.warning(f"Failed to load {len(self.failed_curves)} files.")
            logger.warning("Summary of missing files (up to 10 shown):")

            # Output the first 10 failed loads with details
            for i, (curve_num, direction, chan_name) in enumerate(self.failed_curves):
                if i < 10:
                    if chan_name:
                        logger.warning(
                            f"Failed to load data file for curve {curve_num}, direction {direction}, channel {chan_name}"
                        )
                    else:
                        if direction is not None:
                            logger.warning(
                                f"Failed to load segment meta file for curve {curve_num}, direction {direction}"
                            )
                        else:
                            logger.warning(f"Failed to load curve meta file for curve {curve_num}")
                else:
                    break
        else:
            # If there are no failed loads, log that all data was loaded successfully
            logger.info("Successfully loaded all curve data without any missing files.")

        # Output the performance summary for the loading process
        summary = (
            f"\n--- Performance Summary ---\n"
            f"Changing Keys Detection: {self.t_changing_keys:.2f}s\n"
            f"Raw Data - Loading: {self.t_load_data:.2f}s | Processing: {self.t_proc_data:.2f}s | Saving: {self.t_save_data:.2f}s\n"
            f"Metadata - Loading: {self.t_load_meta:.2f}s | Processing: {self.t_proc_meta:.2f}s | Saving: {self.t_save_meta:.2f}s\n"
            f"---------------------------"
        )
        logger.info(summary)

    def extract_data_to_h5(
        self, h5_datasets, h5_meta_datasets, h5_datasets_buffer, h5_meta_datasets_buffer, include_metadata: bool = True
    ):
        """
        Load all curve data and optionally metadata from the JPK QI archive into HDF5 datasets.

        Parameters
        ----------
        h5_datasets : dict
            Dictionary of HDF5 datasets for storing curve data.
        h5_meta_datasets : dict
            Dictionary of HDF5 datasets for storing metadata.
        h5_datasets_buffer : dict
            Dictionary of buffers for HDF5 curve data.
        h5_meta_datasets_buffer : dict
            Dictionary of buffers for HDF5 metadata.
        include_metadata : bool, optional
            Whether to include metadata in the loading process, by default True.
        """

        logger.info(
            f"Loading all curve data from JPK QI archive with {len(self.namelist)} files {'' if include_metadata else 'not '}including metadata"
        )
        progress_counter = 0
        process = psutil.Process(os.getpid())
        if include_metadata:
            # Prepare keys for metadata to speed up processing
            curve_work = [
                (f"{k}=".encode("utf-8"), h5_meta_datasets[f"curve.{k}"], h5_meta_datasets_buffer[f"curve.{k}"])
                for k in self.changing_curve_keys
            ]
            seg_work = [
                (f"{k}=".encode("utf-8"), h5_meta_datasets[f"segment.{k}"], h5_meta_datasets_buffer[f"segment.{k}"])
                for k in self.changing_segment_keys
            ]
        for curve_num in range(self.num_of_curves):
            # Output progress every 1000 curves to give some indication of how long the loading is taking
            if progress_counter % 1000 == 0:
                mem = process.memory_info().rss / 1024 / 1024
                logger.info(
                    f"Progress: {progress_counter}/{self.num_of_curves} curves processed, Memory usage: {mem:.2f} MB"
                )
            progress_counter += 1

            for direction in range(2):
                for chan in self.segment_channels:
                    # Save the actual curve data to the h5 datasets
                    self.extract_dat_file(
                        h5_datasets=h5_datasets,
                        h5_datasets_buffer=h5_datasets_buffer,
                        curve_num=curve_num,
                        direction=direction,
                        chan_name=chan["name"],
                    )

                if include_metadata:
                    # Extract and store the segment metadata for later saving
                    self.extract_segment_metadata(curve_num=curve_num, direction=direction, seg_work=seg_work)

            if include_metadata:
                # Extract and store the curve metadata for later saving
                self.extract_curve_metadata(curve_num=curve_num, curve_work=curve_work)

        # Add the last index to the indicies datasets to mark the end of the last curve
        for direction in range(2):
            seg_name = f"Segment_{direction}"
            for chan in self.segment_channels:
                chan_name = chan["name"]
                current_dataset = h5_datasets[seg_name][chan_name]["Data"]
                indicies_dataset = h5_datasets[seg_name][chan_name]["Indicies"]
                indicies_dataset[-1] = current_dataset.shape[0]


    def save_to_h5(
        self,
        include_metadata: bool = True,
    ):
        """
        Saves the data as an H5 file. If include_metadata is False, only the curve data will be saved.

        Parameters
        ----------
        include_metadata : bool, optional
            If True, metadata will be included in the saved H5 file. Default is True.
        """
        with self.get_saving_context() as file:

            t0 = time.perf_counter()

            # Sample curves in dataset to make a best guess for the meta keys that need to be extracted from each curve
            self.changing_curve_keys, self.changing_segment_keys = self.get_changing_keys()
            self.points_for_channel_segment = self.predict_total_points()
            self.t_changing_keys = time.perf_counter() - t0

            # Setup H5 structure for saving the data, creating datasets for curve data and metadata as needed
            global_meta_group, h5_datasets, h5_meta_datasets, h5_datasets_buffer, h5_meta_datasets_buffer = (
                self.setup_h5_structure(file)
            )

            # Set up current_offsets to keep track of how many points have been read
            self.current_offsets = {}
            for direction in range(2):
                self.current_offsets[direction] = {}
                for chan in self.segment_channels:
                    self.current_offsets[direction][chan["name"]] = 0

                    # Reset the points for channel segment to 0 so it can be used to store the actual number of points held in each dataset
                    self.points_for_channel_segment[direction][chan["name"]] = 0

            # Extract data from the JPK QI archive and save to H5 datasets, optionally including metadata
            self.extract_data_to_h5(
                h5_datasets,
                h5_meta_datasets,
                h5_datasets_buffer,
                h5_meta_datasets_buffer,
                include_metadata=include_metadata,
            )
            # Resize the datasets to the actual number of points read
            for direction in range(2):
                for chan in self.segment_channels:
                    logger.debug(
                        f"Resizing dataset for channel {chan['name']} in segment {direction} from {h5_datasets[f'Segment_{direction}'][chan['name']]['Data'].shape[0]} to final size {self.points_for_channel_segment[direction][chan['name']]}"
                    )
                    h5_datasets[f"Segment_{direction}"][chan["name"]]["Data"].resize(
                        (self.points_for_channel_segment[direction][chan["name"]],)
                    )

            self.output_summary()

            if include_metadata:
                # Save the global metadata to the h5 file
                for key, value in self.get_collated_metadata().items():
                    global_meta_group.attrs[key] = str(value).encode("utf-8")

            logger.info(f"QI data copied to h5 data {file.filename}")

    def get_curves_sample(self):
        """
        Get a sample of curve numbers distrubuted evenly across the dataset

        Returns
        -------
        range:
            A range object representing the sampled curve numbers.
        """

        # Check evenly spaced curves in the dataset to sample metadata without having to load every curve
        step = 1 if self.num_of_curves <= self.MAX_CURVE_CHECKS else self.num_of_curves // self.MAX_CURVE_CHECKS
        # If the step is equal to a shape dimension, we might just go down the row or column
        while step in [self.shape_x, self.shape_y] and step > 1:
            # So make the step slightly smaller (more checks) to ensure we get a good sample
            step -= 1
        return range(0, self.num_of_curves, step)

    def predict_total_points(self):
        """
        Predicts the total number of points for each channel and segment by sampling a subset of curves
        and extrapolating based on the maximum number of points found in the sample.

        Returns
        -------
        dict:
            A dictionary containing the predicted total points for each channel and segment.
        """

        # Get a sample of curve (indicies)
        curves_to_check = self.get_curves_sample()
        points_for_channel_segment = {}

        # Iterate through the segments, channels and our curve indicies
        for direction in range(2):
            points_for_channel_segment[direction] = {}
            for channel in self.segment_channels:
                points_for_channel_segment[direction][channel["name"]] = []
                for curve_num in curves_to_check:
                    # Loop until we successfully retrieve some data
                    while True:
                        dat_path = f"index/{curve_num}/segments/{direction}/channels/{channel['name']}.dat"
                        try:
                            # Count points in extracted data
                            with self.qi_archive.open(dat_path) as f:
                                raw_array = np.frombuffer(f.read(), dtype=">i4")
                                points_for_channel_segment[direction][channel["name"]].append(len(raw_array))
                                break

                        except KeyError:
                            # If the file doesn't exist for this curve, check the next curve so we don't just get a smaller sample
                            if curve_num + 1 >= self.num_of_curves:
                                # If we've gone past the number of curves, stop checking
                                break
                            curve_num += 1
                            continue
                # Calculate a prediction for total number of points based on maximum number of points then assuming
                # maximum points throughout data is no more than 10% higher
                points_for_channel_segment[direction][channel["name"]] = (
                    int(np.max(points_for_channel_segment[direction][channel["name"]]) * 1.1) * self.num_of_curves
                )
        return points_for_channel_segment

    def get_changing_keys(self):
        """
        Checks a sample of curves to see which metadata keys change across curves and segments,
        so we can extract only the changing keys for each curve and segment.

        None changing keys are moved to the top level metadata and not extracted for each curve/segment.

        Returns
        -------
        tuple:
            A tuple containing two sets: changing_curve_keys and changing_segment_keys.
        """

        curve_meta_dict = {}
        segment_meta_dict = {}
        curves_to_check = self.get_curves_sample()
        for curve_num in curves_to_check:
            for direction in range(2):
                while True:
                    meta_path = f"index/{curve_num}/segments/{direction}/segment-header.properties"
                    try:
                        with self.qi_archive.open(meta_path) as f:
                            meta_dict = {".".join(k.split(".")[1:]): v for k, v in javaproperties.load(f).items()}
                            for k, v in meta_dict.items():
                                if k not in segment_meta_dict:
                                    segment_meta_dict[k] = []
                                segment_meta_dict[k].append(v)
                        break

                    except KeyError:
                        if curve_num + 1 >= self.num_of_curves:
                            break  # If we've gone past the number of curves, stop checking
                        curve_num += 1
                        continue
            meta_path = f"index/{curve_num}/header.properties"
            while True:
                try:
                    with self.qi_archive.open(meta_path) as f:
                        meta_dict = {".".join(k.split(".")[1:]): v for k, v in javaproperties.load(f).items()}
                        for k, v in meta_dict.items():
                            if k not in curve_meta_dict:
                                curve_meta_dict[k] = []
                            curve_meta_dict[k].append(v)
                        break
                except KeyError:
                    if curve_num + 1 >= self.num_of_curves:
                        break  # If we've gone past the number of curves, stop checking
                    curve_num += 1
                    continue

        changing_curve_keys, changing_segment_keys = set(), set()
        for key, values in curve_meta_dict.items():
            if len(set(v for v in values if v is not None)) > 1:
                changing_curve_keys.add(key)
            else:
                # If the key does not change across curves, move it to the top level metadata
                self.top_level_meta[f"curve.{key}"] = values[0]
        for key, values in segment_meta_dict.items():
            if len(set(v for v in values if v is not None)) > 1:
                changing_segment_keys.add(key)
            else:
                # If the key does not change across segments, move it to the top level metadata
                self.top_level_meta[f"segment.{key}"] = values[0]
        return changing_curve_keys, changing_segment_keys

    def get_collated_metadata(self):
        """
        Collates the metadata from being split by curve, to being split by attribute so data can be saved more efficiently

        Returns
        -------
        collated_meta : dict
            A dictionary containing the collated metadata.
        """
        collated_meta = {}
        for seg_chan in self.segment_channels:
            collated_meta[f"channel.unit.{seg_chan['name']}"] = seg_chan["unit"]
        for key, value in self.top_level_meta.items():
            collated_meta[key] = value
        return collated_meta

    def get_image(
        self, overide_channel: str | None = None, convert_to_nm: bool = True, flip_image: bool | None = None
    ) -> tuple[np.ndarray, float]:
        """
        Processes the flat curve data dictionary into a 2D list structure matching the image dimensions.

        Returns
        -------
        image : np.ndarray
            A 2D array representing the image data.
        """

        # Get channel and flip_image parameters, defaulting to the instance attributes if not provided
        if overide_channel:
            channel = overide_channel
        else:
            channel = self.channel

        if flip_image is None:
            flip_image = self.flip_image

        # Search through the namelist to find the .jpk-qi-image file
        path_to_image = None
        for file_name in self.namelist:
            if file_name.endswith(".jpk-qi-image"):
                path_to_image = file_name
        if path_to_image is None:
            raise FileNotFoundError(f"{path_to_image} not found in JPK archive")

        # Read the .jpk-qi-image file as bytes and load it using the existing jpk loading function
        tif_bytes = self.qi_archive.read(path_to_image)

        virtual_file = io.BytesIO(tif_bytes)
        logger.info(f"Looking for channel {channel} in {path_to_image}")
        return jpk._load_jpk(
            virtual_file,
            path_to_image,
            channel=channel,
            file_suffix=".jpk-qi-data",
            config_path=self.config_path,
            convert_to_nm=convert_to_nm,
            flip_image=flip_image,
        )

    def save_lite_data(self):
        """
        Saves a lite form of the data (e.g., the calculated image data) to the appropriate format based on the save_as_h5 attribute.
        """
        with h5py.File(self.filepath.parent / f"{self.filepath.stem}.h5-jpk", "a") as h5file:
            # Save data required for reading the h5 file as a normal image file
            meas_grp = h5file.require_group("Measurement_000")
            # Save dimensions data
            meas_grp.attrs["position-pattern.grid.ulength"] = self.size_x
            meas_grp.attrs["position-pattern.grid.ilength"] = self.shape_x
            meas_grp.attrs["position-pattern.grid.vlength"] = self.size_y
            meas_grp.attrs["position-pattern.grid.jlength"] = self.shape_y
            meas_grp.attrs["timing-settings.scanRate"] = 1.0  # Dummy value to satisfy reader

            logger.info(f"Saving a hdf5 copy of the data {self.filepath.parent / f'{self.filepath.stem}.h5-jpk'}")

            h5_channels = [self.channel]
            # Look for the jpk-qi-image file in the archive
            for file_name in self.namelist:
                if file_name.endswith(".jpk-qi-image"):
                    path_to_image = file_name
                    break
            # Add the channels which exist in the jpk-qi-image file
            with self.qi_archive.open(path_to_image, "r") as image_file:
                h5_channels += jpk._get_jpk_channels(
                    file=image_file, filename=self.filepath.stem, file_path=self.filepath / Path(path_to_image)
                )
            for i, h5_channel in enumerate(h5_channels):
                # For each available channel, save the required data to the h5 file
                # TODO make sure this metadata is accurate for the channels coming from the .jpk-qi-image file
                chan_grp = meas_grp.require_group(f"Channel_{_make_num_min_characters(i)}")
                # Extract name and retrace information from the channel name
                if "_" in h5_channel:
                    base_name, trace_dir = h5_channel.rsplit("_", 1)
                    is_retrace = "true" if trace_dir.lower() == "retrace" else "false"
                else:
                    base_name = h5_channel
                    is_retrace = "false"

                # Add the necessary attributes to the channel group
                chan_grp.attrs["channel.name"] = base_name.encode("utf-8")
                chan_grp.attrs["retrace"] = is_retrace.encode("utf-8")
                chan_grp.attrs["net-encoder.scaling.multiplier"] = 1.0
                chan_grp.attrs["net-encoder.scaling.offset"] = 0.0

                # Format name and reshape image (flattened frame stack)
                dataset_name = h5_channel.split("_")[0].capitalize()
                # Include all the channels including the calculated channel
                # TODO make this slightly faster by remembering we have load a channel already but difficult cause of scaling
                channel_image, _ = self.get_image(overide_channel=h5_channel, convert_to_nm=False, flip_image=False)
                frame_stack = channel_image.flatten().reshape(-1, 1)

                # Update/ replace the channels dataset
                if dataset_name in chan_grp:
                    del chan_grp[dataset_name]
                chan_grp.create_dataset(dataset_name, data=frame_stack)

    def extract_dat_file(self, h5_datasets, h5_datasets_buffer, curve_num: int, direction: int, chan_name: str):
        """
        Extracts the data from a .dat file in the JPK QI archive, applies the appropriate scaling, and saves it to the internal data structure and h5 dataset if required.

        Parameters
        ----------
        h5_datasets : dict
            A dictionary containing the h5 datasets for each channel and segment direction, used for saving the data
        h5_datasets_buffer : dict
            A dictionary containing the buffer for each h5 dataset, used for temporary storage before writing to the dataset
        curve_num : int
            The curve number associated with the .dat file, parsed from the filename.
        direction : int
            The segment direction (0 or 1) associated with the .dat file, parsed from the filename.
        chan_name : str
            The channel name associated with the .dat file, parsed from the filename.
        """
        if chan_name in self.channel_scaling:
            # Get data structures for this channel and segment
            scale = self.channel_scaling[chan_name]
            dat_path = f"index/{curve_num}/segments/{direction}/channels/{chan_name}.dat"
            data_set = h5_datasets[f"Segment_{direction}"][chan_name]["Data"]
            indicies_set = h5_datasets[f"Segment_{direction}"][chan_name]["Indicies"]
            data_size = data_set.shape[0]
            buf = h5_datasets_buffer[f"Segment_{direction}"][chan_name]
            filled_size = self.points_for_channel_segment[direction][chan_name]
            start_offset = self.current_offsets[direction][chan_name]

            try:
                t0 = time.perf_counter()
                with self.qi_archive.open(dat_path) as f:
                    # Read the binary data as big-endian 32-bit integers
                    raw_bytes = f.read()
                    self.t_load_data += time.perf_counter() - t0

                    t1 = time.perf_counter()
                    raw_array = np.frombuffer(raw_bytes, dtype=">i4")

                    # Apply the scaling to convert raw values into real world values
                    segment_array = (raw_array * scale["multiplier"]) + scale["offset"]

                    # Update the current offset so it include the length of the data we have just read
                    self.current_offsets[direction][chan_name] += len(segment_array)

                    buf["Data"].append(segment_array)
                    if len(buf["Data"]) >= self.BUFFER_SIZE or curve_num == self.num_of_curves - 1:
                        if self.points_for_channel_segment[direction][chan_name] > data_size:
                            # Fetch and resize the existing dataset for this channel and segment to fit the new data
                            data_set.resize((self.points_for_channel_segment[direction][chan_name],))

                        buffered_data = np.concatenate(buf["Data"])

                        self.t_proc_data += time.perf_counter() - t1
                        start_time = time.perf_counter()

                        # Add the buffer to the dataset
                        data_set[filled_size : filled_size + len(buffered_data)] = buffered_data
                        # Update the filled size for this channel and segment
                        self.points_for_channel_segment[direction][chan_name] += len(buffered_data)
                        # Clear the buffer
                        buf["Data"].clear()
                        self.t_save_data += time.perf_counter() - start_time

            except KeyError:
                self.failed_curves.add((curve_num, direction, chan_name))

                # Limit the number of warnings to avoid spamming the logs
                if len(self.failed_curves) < 10:
                    logger.warning(
                        f"Data file {dat_path} not found in archive. Skipping data for curve {curve_num}, direction {direction}, channel {chan_name}."
                    )
                elif len(self.failed_curves) == 10:
                    logger.warning(
                        "Lots of missing files, further warnings will be suppressed. View summary at the end."
                    )

            start_time = time.perf_counter()
            # Append the new index to the indicies buffer
            buf["Indicies"].append(start_offset)

            # If the indicies buffer is full add it to the indicies dataset and clear the buffer
            if len(buf["Indicies"]) > 0 and len(buf["Indicies"]) % self.BUFFER_SIZE == 0:
                indicies_set[curve_num - self.BUFFER_SIZE + 1 : curve_num + 1] = buf["Indicies"]
                buf["Indicies"].clear()

            # Or if this is the last curve and there are still indicies in the buffer, add them to the indicies dataset and clear the buffer
            elif len(buf["Indicies"]) > 0 and curve_num == self.num_of_curves - 1:
                items_in_buffer = len(buf["Indicies"])
                indicies_set[curve_num - items_in_buffer + 1 : curve_num + 1] = buf["Indicies"]
                buf["Indicies"].clear()

            self.t_save_data += time.perf_counter() - start_time

        else:
            # Log if curve failed
            self.failed_curves.add((curve_num, direction, chan_name))
            if len(self.failed_curves) < 10:  # Limit the number of warnings to avoid spamming the logs
                logger.warning(
                    f"Channel {chan_name} not found in scaling information. Skipping data for curve {curve_num}, direction {direction}."
                )

    def extract_curve_metadata(self, curve_num: int, curve_work):
        """
        Extracts the curve metadata from its header.properties file in the JPK QI archive and save to h5

        Parameters
        ----------
        curve_num : int
            The curve number associated with the metadata, parsed from the filename.
        curve_work : list
            A list of tuples containing the search term for the metadata, the h5 dataset to save to, and the buffer for that dataset.
        """

        meta_path = f"index/{curve_num}/header.properties"
        raw_bytes = b""
        try:
            start_time = time.perf_counter()
            # Read metadata file as raw bytes
            with self.qi_archive.open(meta_path) as f:
                raw_bytes = f.read()
            self.t_load_meta += time.perf_counter() - start_time
        except KeyError:
            self.failed_curves.add((curve_num, None, None))
            # Limit the number of warnings to avoid spamming the logs
            if len(self.failed_curves) < 10:
                logger.warning(
                    f"Metadata file {meta_path} not found in archive. Skipping metadata for curve {curve_num}."
                )
            elif len(self.failed_curves) == 10:
                logger.warning("Lots of missing files, further warnings will be suppressed. View summary at the end.")

        for search_term, meta_set, meta_buffer in curve_work:
            start_time = time.perf_counter()
            # Find the location of the metadata value in the raw bytes
            start = raw_bytes.find(search_term)
            # If found, extract the actual value
            if start != -1:
                start += len(search_term)
                end = raw_bytes.find(b"\n", start)
                value = raw_bytes[start:end].decode("utf-8").strip()
            # Save a no data value if the search term is not found in the metadata file
            else:
                value = "No data"
            self.t_proc_meta += time.perf_counter() - start_time
            if meta_buffer is not None:
                start_time = time.perf_counter()
                meta_buffer.append(value)
                if len(meta_buffer) >= self.BUFFER_SIZE or curve_num == self.num_of_curves - 1:
                    meta_set[curve_num - len(meta_buffer) + 1 : curve_num + 1] = meta_buffer
                    meta_buffer.clear()
                self.t_save_meta += time.perf_counter() - start_time
            else:
                logger.error(
                    f"Metadata dataset for key {search_term.decode('utf-8')} not found when trying to save metadata for curve {curve_num}"
                )

    def extract_segment_metadata(self, curve_num: int, direction: int, seg_work):
        """
        Extracts the segment metadata from its header.properties file in the JPK QI archive and saves it to the internal data structure.

        Parameters
        ----------
        curve_num : int
            The curve number associated with the metadata, parsed from the filename.
        direction : int
            The segment direction (0 or 1) associated with the metadata, parsed from the filename.
        """
        meta_path = f"index/{curve_num}/segments/{direction}/segment-header.properties"
        raw_content = b""
        try:
            start_time = time.perf_counter()
            with self.qi_archive.open(meta_path) as f:
                raw_content = f.read()
            self.t_load_meta += time.perf_counter() - start_time
        except KeyError:
            self.failed_curves.add((curve_num, direction, None))
            if len(self.failed_curves) < 10:  # Limit the number of warnings to avoid spamming the logs
                logger.warning(
                    f"Metadata file {meta_path} not found in archive. Skipping metadata for curve {curve_num}, direction {direction}."
                )
            elif len(self.failed_curves) == 10:
                logger.warning("Lots of missing files, further warnings will be suppressed. View summary at the end.")
        start_time = time.perf_counter()
        for search_term, meta_set, meta_buffer in seg_work:
            start = raw_content.find(search_term)
            if start != -1:
                start += len(search_term)
                end = raw_content.find(b"\n", start)
                value = raw_content[start:end].decode("utf-8").strip()
            else:
                value = "No data"
            if meta_buffer is not None:
                meta_buffer.append(value)
                if len(meta_buffer) >= self.BUFFER_SIZE or curve_num == self.num_of_curves - 1:
                    idx = curve_num * 2 + direction
                    meta_set[idx - len(meta_buffer) + 1 : idx + 1] = meta_buffer
                    meta_buffer.clear()
            else:
                logger.error(
                    f"Metadata dataset for key {search_term.decode('utf-8')} not found when trying to save metadata for curve {curve_num}, direction {direction}"
                )
        self.t_proc_meta += time.perf_counter() - start_time

    def setup_h5_structure(self, h5file):
        """
        Sets up the structure in the h5 file for saving the curve data and metadata, and returns the datasets and metadata groups for later use.

        Parameters
        ----------
        h5file : h5py.File
            The h5 file in which to set up the structure for saving the curve data and metadata.
        Returns
        -------
        curve_groups : dict
            A dictionary containing the group structure for each segment direction.
        global_meta_group : h5py.Group
            The h5 group for storing global metadata.
        curves_meta_group : h5py.Group
            The h5 group for storing curve metadata.
        """

        # Create the main group for the QI curve data that all the curve data will be in
        qi_group = h5file.require_group("QI_Curve_Data")

        # Establish empty groups for global metadata and curve metadata
        global_meta_group = qi_group.require_group("Global_Metadata")
        curves_meta_group = qi_group.require_group("Curve_Metadata")
        curves_group = qi_group.require_group("Curves")

        curve_groups = {"Data": {}, "Indicies": {}}
        h5_datasets = {}
        h5_meta_datasets = {}
        h5_datasets_buffer = {}
        h5_meta_datasets_buffer = {}
        for key in self.changing_curve_keys:
            h5_meta_datasets[f"curve.{key}"] = curves_meta_group.create_dataset(
                name=f"curve.{key}",
                shape=(self.num_of_curves,),
                maxshape=(None,),
                chunks=self.META_CHUNKSIZE,
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            h5_meta_datasets_buffer[f"curve.{key}"] = []
        for key in self.changing_segment_keys:
            h5_meta_datasets[f"segment.{key}"] = curves_meta_group.create_dataset(
                name=f"segment.{key}",
                shape=(self.num_of_curves * 2,),
                maxshape=(None,),
                chunks=self.META_CHUNKSIZE,
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            h5_meta_datasets_buffer[f"segment.{key}"] = []

        for direction in range(2):
            # For each segment direction, establish the necessary group structure that will contain each channel dataset
            seg_name = f"Segment_{direction}"
            dir_group = curves_group.require_group(seg_name)
            h5_datasets[seg_name] = {}
            h5_datasets_buffer[seg_name] = {}
            # Create the Data and Indicies subfolders and store their references
            curve_groups["Data"][seg_name] = dir_group.require_group("Data")
            curve_groups["Indicies"][seg_name] = dir_group.require_group("Indicies")
            for chan in self.segment_channels:
                h5_datasets[seg_name][chan["name"]] = {}
                # For each channel, create an empty dataset for the curve data and indicies with the appropriate name and data type
                h5_datasets[seg_name][chan["name"]]["Data"] = curve_groups["Data"][seg_name].create_dataset(
                    name=chan["name"],
                    shape=(self.points_for_channel_segment[direction][chan["name"]],),
                    maxshape=(None,),
                    chunks=(self.DATA_CHUNKSIZE,),
                    dtype=np.float32,
                )
                h5_datasets[seg_name][chan["name"]]["Indicies"] = curve_groups["Indicies"][seg_name].create_dataset(
                    name=chan["name"],
                    shape=(self.num_of_curves + 1,),
                    maxshape=(None,),
                    chunks=(self.INDICIES_CHUNKSIZE,),
                    dtype=np.int32,
                )
                h5_datasets_buffer[seg_name][chan["name"]] = {"Data": [], "Indicies": []}
        return global_meta_group, h5_datasets, h5_meta_datasets, h5_datasets_buffer, h5_meta_datasets_buffer

    def get_saving_context(self):
        """
        Returns the appropriate context manager for saving the data based on the save_as_h5 attribute.
        If save_as_h5 is True, it returns a context manager for an h5 file. Otherwise, it returns a null context.

        Returns
        -------
        contextlib.AbstractContextManager
            The context manager for saving the data.
        """
        if self.save_as_h5:
            return h5py.File(self.filepath.parent / f"{self.filepath.stem}.h5-jpk", "a")
        else:
            return nullcontext()

    def parse_dimension_data(self):
        """
        Parses the dimension data from the provided properties dictionary and calculates the pixel to nanometer scaling factor.
        """
        # Extract both real size and pixel dimensions from the metadata
        for key, value in self.top_level_meta.items():
            if key.endswith(".ulength"):
                self.size_x = float(value)
            if key.endswith(".vlength"):
                self.size_y = float(value)
            if key.endswith(".ilength"):
                self.shape_x = int(value)
            if key.endswith(".jlength"):
                self.shape_y = int(value)

        # Log an error if any of these do not exist
        if None in [self.size_x, self.size_y, self.shape_x, self.shape_y]:
            logger.error(f"Incomplete dimension data in {self.filepath}")

        # Calculate the pixel to nano metre scaling as an average of the scale for each direction
        pixel_to_nm_scaling_factor_x = self.size_x / self.shape_x * 1e9 if self.shape_x > 0 else 1.0
        pixel_to_nm_scaling_factor_y = self.size_y / self.shape_y * 1e9 if self.shape_y > 0 else 1.0
        self.px2nm = (pixel_to_nm_scaling_factor_x + pixel_to_nm_scaling_factor_y) / 2

        # Establish number of curves
        self.num_of_curves = self.shape_x * self.shape_y

    def extract_global_metadata(self):
        """Extracts the global metadata from the JPK QI archive and populates the top level metadata dictionary and segment channels list."""

        # Load the metadata from the global properties file
        if "header.properties" in self.namelist:
            with self.qi_archive.open("header.properties") as archive_meta_file:
                props = javaproperties.load(archive_meta_file)

                # Add all the data from the main header to the top level metadata with a prefix to avoid key clashes
                for key, value in props.items():
                    self.top_level_meta[f"main-header.{key}"] = value
        else:
            logger.error(f"File {self.filepath} does not contain essential metadata and cannot be loaded")

        # Load the metadata from the shared header and parse the channel information for the segments
        if "shared-data/header.properties" in self.namelist:
            with self.qi_archive.open("shared-data/header.properties") as shared_data_file:
                shared_meta = javaproperties.load(shared_data_file)
                channel_idx = 0

                # Add all the data from the shared header to the top level metadata with a prefix to avoid key clashes
                for key, value in shared_meta.items():
                    self.top_level_meta[f"shared-data.{key}"] = value

                # Collect channel data from the shared metadata
                while f"lcd-info.{channel_idx}.channel.name" in shared_meta:
                    channel_dict = {}
                    # Calculate and store the offset and multiplier to convert raw values into real world values
                    multiplier, offset, unit = _get_channel_scaling(shared_meta, channel_idx)
                    channel_dict["name"] = shared_meta[f"lcd-info.{channel_idx}.channel.name"]
                    channel_dict["offset"] = offset
                    channel_dict["multiplier"] = multiplier
                    channel_dict["unit"] = unit
                    # Add the channel dict to the list
                    self.segment_channels.append(channel_dict)
                    # Increment the channel index to look for the next channel
                    channel_idx += 1
        else:
            logger.error(f"File {self.filepath} does not contain essential channel metadata and cannot be loaded")

        if len(self.segment_channels) == 0:
            logger.error("Could not find channels for segments")

        # Create a lookup for channel name to unit to be returned
        self.channels_units = {seg_chan["name"]: seg_chan["unit"] for seg_chan in self.segment_channels}
        # Lookup map for binary scaling
        self.channel_scaling = {chan["name"]: chan for chan in self.segment_channels}

    def close(self):
        """Closes the ZIP archive when done to free up system resources."""
        self.qi_archive.close()
        self.image = None
        self.all_curve_data = None
        self.curve_meta = {}
        self.segment_meta = {}
        self.top_level_meta = {}
        self.full_metadata = {}
        self.failed_curves = set()
        self.points_for_channel_segment = {}
        self.namelist = []


def load_fdcurves_from_h5(file_path: Path | str):
    file_path = Path(file_path)

    with h5py.File(file_path, "r") as h5file:
        meas_grp = h5file["Measurement_000"]
        shape_x = meas_grp.attrs["position-pattern.grid.ilength"]
        shape_y = meas_grp.attrs["position-pattern.grid.jlength"]
        num_of_curves = shape_x * shape_y
        qi_data_group = h5file["QI_Curve_Data"]
        all_curve_data = []
        for i in range(num_of_curves):
            curve_data = {}
            for direction, direction_group in qi_data_group.items():
                for channel, channel_group in direction_group.items():
                    if channel not in curve_data:
                        curve_data[channel] = {}
                    curve_data[channel][direction] = channel_group[str(i)]
            all_curve_data.append(curve_data)

        return all_curve_data


def _make_num_min_characters(num: int, min_chars: int = 3):
    string_num = str(num)
    if len(string_num) >= min_chars:
        return string_num
    string_num = "0" * (min_chars - len(string_num)) + string_num
    return string_num
