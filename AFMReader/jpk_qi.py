from pathlib import Path
from contextlib import nullcontext
import io
import re
import time
import zipfile

import numpy as np
import javaproperties
import h5py

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
        # Open the ZIP archive once and keep it open for the duration of the loading process to improve performance when accessing multiple files within the archive
        self.qi_archive = zipfile.ZipFile(self.filepath, "r")
        # Set path to the .jpk-qi-image file within the archive for later use
        self.path_to_image = None

        self.time_loading_data = 0.0
        self.time_loading_metadata = 0.0
        self.time_saving_h5 = 0.0
        self.time_with_regex = 0.0
        self.time_collating_data = 0.0

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
        # Dictionary of the metadata for each curve, indexed by curve number
        self.curve_meta_dict = {}
        # Dictionary of the metadata for each segment, indexed by segement number (calculated as curve_num * 2 + direction)
        self.segment_meta_dict = {}
        # The keys that exist in the curve metadata across all curves, used to determine which keys to move to the top level metadata
        self.all_curve_keys = set()
        # The keys that exist in the segment metadata across all segments, used to determine which keys to move to the top level metadata
        self.all_segment_keys = set()
        self.curve_meta = {}
        self.segment_meta = {}
        # Define the image shape and size attributes
        self.size_x, self.size_y, self.shape_x, self.shape_y = None, None, None, None

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
            for file_name in self.qi_archive.namelist():
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
            self.load_all_data()
            start_time = time.perf_counter()
            self.save_to_h5(collated_curve_data=self.collated_curve_data, indicies=self.indicies, collated_metadata=self.collated_metadata)
            self.time_saving_h5 += time.perf_counter() - start_time

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
            start_time = time.perf_counter()
            self.save_lite_data()
            self.time_saving_h5 += time.perf_counter() - start_time

        logger.info(f"Finished loading JPK QI data from {self.filepath} in {self.time_loading_data + self.time_loading_metadata + self.time_collating_data + self.time_saving_h5:.2f} seconds \n"
                    f"Data loading: {self.time_loading_data:.2f}s\n"
                    f"Metadata loading: {self.time_loading_metadata:.2f}s\n"
                    f"Time spent in regex matching: {self.time_with_regex:.2f}s\n"
                    f"Collating data: {self.time_collating_data:.2f}s\n"
                    f"Saving H5: {self.time_saving_h5:.2f}s")

        if self.all_curve_data:
            return (self.image, self.px2nm, (self.all_curve_data, self.channels_units, self.full_metadata))

        return self.image, self.px2nm

    def load_all_data(self, include_metadata: bool = True):
        # Compile Regexes
        dat_regex = re.compile(r"index/(\d+)/segments/(\d+)/channels/([^/]+)\.dat")
        if include_metadata:
            curve_meta_regex = re.compile(r"index/(\d+)/header\.properties")
            segment_meta_regex = re.compile(r"index/(\d+)/segments/(\d+)/segment-header\.properties")
        archive_infolist = self.qi_archive.infolist()
        logger.info(f"Loading all curve data from JPK QI archive with {len(archive_infolist)} files {'' if include_metadata else 'not '}including metadata")
        progress_counter = 0
        for file_info in archive_infolist:
            filename = file_info.filename
            if progress_counter % 10000 == 0:
                logger.info(f"Progress: {progress_counter}/{len(archive_infolist)} files processed")
            progress_counter += 1

            # Check Binary Data
            start_time = time.perf_counter()
            dat_match = dat_regex.match(filename)
            if dat_match:
                # If file is a .dat file, extract the curve number, segment direction and channel name from the filename
                curve_num, direction, chan_name = int(dat_match.group(1)), int(dat_match.group(2)), dat_match.group(3)
                # Then load the data from the file
                self.time_with_regex += time.perf_counter() - start_time
                start_time = time.perf_counter()
                self.extract_dat_file(file_info, curve_num, direction, chan_name)
                self.time_loading_data += time.perf_counter() - start_time
                continue

            if include_metadata:
                # Check Segment Metadata
                segment_meta_match = segment_meta_regex.match(filename)
                if segment_meta_match:
                    # If file is a segment metadata file, extract the curve number and segment direction from the filename
                    curve_num, direction = int(segment_meta_match.group(1)), int(segment_meta_match.group(2))
                    self.time_with_regex += time.perf_counter() - start_time
                    # Then load the segment metadata from the file
                    start_time = time.perf_counter()
                    self.extract_segment_metadata(file_info, curve_num, direction)
                    self.time_loading_metadata += time.perf_counter() - start_time
                    continue

                # Check Curve Metadata
                curve_meta_match = curve_meta_regex.match(filename)
                if curve_meta_match:
                    # If file is a curve metadata file, extract the curve number from the filename
                    curve_num = int(curve_meta_match.group(1))
                    self.time_with_regex += time.perf_counter() - start_time
                    # Then load the metadata from the file
                    start_time = time.perf_counter()

                    self.extract_curve_metadata(file_info, curve_num)
                    self.time_loading_metadata += time.perf_counter() - start_time
                    continue

        start_time = time.perf_counter()
        # If saving, need to collate the curve data into a format that can be easily saved to the h5 file (a dataset per channel per segment direction)
        self.collated_curve_data, self.indicies = self.get_collated_curves()

        # TODO can we remove curve_meta_dict and just use curve_meta or is the non duplicating necessary
        if include_metadata:
            self.curve_meta = [self.curve_meta_dict.get(i, {}) for i in range(self.num_of_curves)]
            self.segment_meta = [self.segment_meta_dict.get(i, {}) for i in range(self.num_of_curves * 2)]
            self.full_metadata = self.construct_full_metadata()
            self.collated_metadata = self.get_collated_metadata()

        self.time_collating_data += time.perf_counter() - start_time

    def save_to_h5(
        self,
        include_metadata: bool = True,
        collated_curve_data: dict | None = None,
        indicies: dict | None = None,
        collated_metadata: dict | None = None,
    ):
        """Saves the data as an H5 file. If include_metadata is False, only the curve data will be saved."""
        with self.get_saving_context() as file:

            curve_groups, global_meta_group, curves_meta_group = self.setup_h5_structure(file)

            # Save the curve data to the appropriate datasets in the h5 file
            for chan_name, chan_data in collated_curve_data.items():
                for direction in range(2):
                    # Save the curve data and indicies to the appropriate dataset in the h5 file
                    seg_name = f"Segment_{direction}"
                    curve_groups["Data"][seg_name].create_dataset(
                        name=chan_name,
                        data=chan_data[seg_name],
                        dtype=np.float32,
                    )
                    curve_groups["Indicies"][seg_name].create_dataset(
                        name=chan_name,
                        data=indicies[chan_name][seg_name],
                        dtype=np.int32,
                    )

            if include_metadata:
                # Save the global metadata to the h5 file
                vlen_str_dt = h5py.string_dtype(encoding="utf-8")
                for key, value in collated_metadata.items():
                    if isinstance(value, list):
                        # If the key is a changing key, save as a dataset with one entry per curve/ segment
                        curves_meta_group.create_dataset(name=key, data=value, dtype=vlen_str_dt)
                    else:
                        global_meta_group.attrs[key] = str(value).encode("utf-8")

            logger.info(f"QI data copied to h5 data {file.filename}")

    def get_collated_curves(self):
        """
        Collates the curve data from the flat structure it is extracted in into a structure grouped by channel and segment for easier saving to h5.

        Returns
        -------
        collated_curve_data : dict
            A dictionary containing the curve data collated by channel and segment, with the structure:
                {
                    "channel_name": {
                        "Segment_0": [...],
                        "Segment_1": [...],
                        ...
                    },
                    ...
                }
            indicies : dict
            A dictionary containing the indexes of the curve data within each segment, with the structure:
                {
                    "channel_name": {
                        "Segment_0": [...],
                        "Segment_1": [...],
                        ...
                    },
                    ...
                }
        """
        collated_curve_data = {}
        indicies = {}

        for curve_data in self.flat_curve_data:
            for chan_name, chan_data in curve_data.items():
                for seg_name, seg_data in chan_data.items():
                    if chan_name not in collated_curve_data:
                        collated_curve_data[chan_name] = {}
                        indicies[chan_name] = {}

                    if seg_name not in collated_curve_data[chan_name]:
                        collated_curve_data[chan_name][seg_name] = []
                        indicies[chan_name][seg_name] = [0]

                    # Append the segment data as an array to the list (creates a 2D list)
                    collated_curve_data[chan_name][seg_name].append(seg_data)

                    last_index = indicies[chan_name][seg_name][-1]
                    next_index = last_index + len(seg_data)

                    indicies[chan_name][seg_name].append(next_index)

        for chan_name, segments in collated_curve_data.items():
            for seg_name in segments:
                # Flattens the list of arrays into one massive 1D array for more efficiency
                collated_curve_data[chan_name][seg_name] = np.concatenate(collated_curve_data[chan_name][seg_name])

                # Converts the indices list into a standard fixed-length integer array
                indicies[chan_name][seg_name] = np.array(indicies[chan_name][seg_name], dtype=np.int32)

        return collated_curve_data, indicies

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
        for curve_dict in self.full_metadata["curves"]:
            for key, value in curve_dict.items():
                if f"curve.{key}" not in collated_meta:
                    collated_meta[f"curve.{key}"] = []
                collated_meta[f"curve.{key}"].append(value)
        for segment_dict in self.full_metadata["segments"]:
            for key, value in segment_dict.items():
                if f"segment.{key}" not in collated_meta:
                    collated_meta[f"segment.{key}"] = []
                collated_meta[f"segment.{key}"].append(value)
        return collated_meta

    def get_image(self, overide_channel: str | None = None, convert_to_nm: bool = True):
        """
        Processes the flat curve data dictionary into a 2D list structure matching the image dimensions.

        Returns
        -------
        image : np.ndarray
            A 2D array representing the image data.
        """

        if overide_channel:
            channel = overide_channel
        else:
            channel = self.channel

        path_to_image = None
        for file_name in self.qi_archive.namelist():
            if file_name.endswith(".jpk-qi-image"):
                path_to_image = file_name
        if path_to_image is None:
            raise FileNotFoundError(f"{path_to_image} not found in JPK archive")

        tif_bytes = self.qi_archive.read(path_to_image)

        virtual_file = io.BytesIO(tif_bytes)
        logger.info(f"Looking for channel {channel} in {path_to_image}")
        return jpk._load_jpk(
            virtual_file, path_to_image, channel=channel, file_suffix=".jpk-qi-data", config_path=self.config_path, convert_to_nm=convert_to_nm
        )

    def save_lite_data(self):
        """Saves a lite form of the data (e.g., the calculated image data) to the appropriate format based on the save_as_h5 attribute."""
        if self.save_as_h5:
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
                for file_name in self.qi_archive.namelist():
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
                    channel_image, _ = self.get_image(overide_channel=h5_channel, convert_to_nm=False)
                    frame_stack = channel_image.flatten().reshape(-1, 1)

                    # Update/ replace the channels dataset
                    if dataset_name in chan_grp:
                        del chan_grp[dataset_name]
                    chan_grp.create_dataset(dataset_name, data=frame_stack)

    def save_metadata(self):
        """Saves the metadata to the appropriate format based on the save_as_h5 attribute."""
        if self.save_as_h5:
            for seg_chan in self.segment_channels:
                self.global_meta_group.attrs[f"channel.unit.{seg_chan['name']}"] = seg_chan["unit"]
            for key, value in self.top_level_meta.items():
                self.global_meta_group.attrs[key] = str(value).encode("utf-8")
            for i, c_meta in enumerate(self.curve_meta):
                curve_meta_group = self.curves_meta_group.require_group(f"{i}")
                for key, value in c_meta.items():
                    curve_meta_group.attrs[key] = str(value).encode("utf-8")

                for d in range(2):
                    segment_meta_group = curve_meta_group.require_group(f"{d}")
                    for key, value in self.segment_meta[i * 2 + d].items():
                        segment_meta_group.attrs[key] = str(value).encode("utf-8")

    def construct_full_metadata(self):
        """
        Constructs the full metadata dictionary by determining which keys in the curve and segment metadata change across curves/segments and which do not, moving the non-changing keys to the top level metadata, and then combining everything into a single dictionary.

        Returns
        -------
        dict
            The full metadata dictionary with the structure:
            {
                "top_level": { ... },
                "curves": [ { ... }, { ... }, ... ],
                "segments": [ { ... }, { ... }, ... ]
            }
        """

        # Find keys that change across curves/segments
        changing_curve_keys = {
            k
            for k in self.all_curve_keys
            if any(self.curve_meta[i].get(k) != self.curve_meta[0].get(k) for i in range(1, self.num_of_curves))
        }
        changing_segment_keys = {
            k
            for k in self.all_segment_keys
            if any(self.segment_meta[i].get(k) != self.segment_meta[0].get(k) for i in range(1, len(self.segment_meta)))
        }

        # Move duplicated meta to top level for both segments and curves
        for key in self.all_curve_keys - changing_curve_keys:
            if self.curve_meta and key in self.curve_meta[0]:
                self.top_level_meta[f"curve.{key}"] = self.curve_meta[0][key]
        for key in self.all_segment_keys - changing_segment_keys:
            if self.segment_meta and key in self.segment_meta[0]:
                self.top_level_meta[f"segment.{key}"] = self.segment_meta[0][key]

        # Strip duplicated keys from individual curve/segment dicts
        for c_meta in self.curve_meta:
            for k in self.all_curve_keys - changing_curve_keys:
                c_meta.pop(k, None)
        for s_meta in self.segment_meta:
            for k in self.all_segment_keys - changing_segment_keys:
                s_meta.pop(k, None)

        # Construct full metadata dict from subdicts
        full_metadata = {"top_level": self.top_level_meta, "curves": self.curve_meta, "segments": self.segment_meta}
        return full_metadata

    def extract_dat_file(self, file_info: zipfile.ZipInfo, curve_num: int, direction: int, chan_name: str):
        """
        Extracts the data from a .dat file in the JPK QI archive, applies the appropriate scaling, and saves it to the internal data structure and h5 dataset if required.

        Parameters
        ----------
        file_info : zipfile.ZipInfo
            The ZipInfo object corresponding to the .dat file to be extracted.
        curve_num : int
            The curve number associated with the .dat file, parsed from the filename.
        direction : int
            The segment direction (0 or 1) associated with the .dat file, parsed from the filename.
        chan_name : str
            The channel name associated with the .dat file, parsed from the filename.
        """
        if chan_name in self.channel_scaling:
            scale = self.channel_scaling[chan_name]
            with self.qi_archive.open(file_info) as f:
                # Read the binary data as big-endian 32-bit integers
                raw_array = np.frombuffer(f.read(), dtype=">i4")
                # Apply the scaling to convert raw values into real world values
                segment_array = (raw_array * scale["multiplier"]) + scale["offset"]

            # If the channel doesn't exist in the curve data for this curve, add it as a new entry
            if chan_name not in self.flat_curve_data[curve_num]:
                self.flat_curve_data[curve_num][chan_name] = {}
            # Add the segment data to the curve data under the appropriate channel and segment direction
            self.flat_curve_data[curve_num][chan_name][f"Segment_{direction}"] = segment_array

        else:
            logger.warning(
                f"Channel {chan_name} not found in scaling information. Skipping data for curve {curve_num}, direction {direction}."
            )

    def extract_curve_metadata(self, file_info: zipfile.ZipInfo, curve_num: int):
        """
        Extracts the curve metadata from its header.properties file in the JPK QI archive and saves it to the internal data structure.

        Parameters
        ----------
        file_info : zipfile.ZipInfo
            The ZipInfo object corresponding to the header.properties file to be extracted.
        curve_num : int
            The curve number associated with the metadata, parsed from the filename.
        """
        # with self.qi_archive.open(file_info) as f:
        #     cleaned_meta = {".".join(k.split(".")[1:]): v for k, v in javaproperties.load(f).items()}
        #     self.curve_meta_dict[curve_num] = cleaned_meta
        #     self.all_curve_keys.update(cleaned_meta.keys())

        with self.qi_archive.open(file_info) as f:
            cleaned_meta = {".".join(k.split(".")[1:]): v for k, v in javaproperties.load(f).items()}
            for key, value in cleaned_meta.items():
                if key not in self.curve_meta:
                    self.curve_meta[key] = [None for _ in range(len(self.curve_meta))]
                self.curve_meta[key].append(value)
            for key in self.curve_meta.keys():
                if key not in cleaned_meta:
                    self.curve_meta[key].append(None)
            self.all_curve_keys.update(cleaned_meta.keys())

    def extract_segment_metadata(self, file_info: zipfile.ZipInfo, curve_num: int, direction: int):
        """
        Extracts the segment metadata from its header.properties file in the JPK QI archive and saves it to the internal data structure.

        Parameters
        ----------
        file_info : zipfile.ZipInfo
            The ZipInfo object corresponding to the header.properties file to be extracted.
        curve_num : int
            The curve number associated with the metadata, parsed from the filename.
        direction : int
            The segment direction (0 or 1) associated with the metadata, parsed from the filename.
        """
        idx = curve_num * 2 + direction
        with self.qi_archive.open(file_info) as f:
            cleaned_meta = {".".join(k.split(".")[1:]): v for k, v in javaproperties.load(f).items()}
            for key, value in cleaned_meta.items():
                if key not in self.segment_meta:
                    self.segment_meta[key] = [None for _ in range(len(self.segment_meta))]
                self.segment_meta[key].append(value)
            for key in self.segment_meta.keys():
                if key not in cleaned_meta:
                    self.segment_meta[key].append(None)
            self.all_se.update(cleaned_meta.keys())
            # self.segment_meta_dict[idx] = cleaned_meta
            # self.all_segment_keys.update(cleaned_meta.keys())

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
        vlen_type = h5py.vlen_dtype(np.float32)

        # Create the main group for the QI curve data that all the curve data will be in
        qi_group = h5file.require_group("QI_Curve_Data")

        # Establish empty groups for global metadata and curve metadata
        global_meta_group = qi_group.require_group("Global_Metadata")
        curves_meta_group = qi_group.require_group("Curve_Metadata")
        curves_group = qi_group.require_group("Curves")

        curve_groups = {"Data": {}, "Indicies": {}}

        for direction in range(2):
            # For each segment direction, establish the necessary group structure that will contain each channel dataset
            seg_name = f"Segment_{direction}"
            dir_group = curves_group.require_group(seg_name)
            # Create the Data and Indicies subfolders and store their references
            curve_groups["Data"][seg_name] = dir_group.require_group("Data")
            curve_groups["Indicies"][seg_name] = dir_group.require_group("Indicies")
        return curve_groups, global_meta_group, curves_meta_group

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

        # Setup the data structure to hold the curve data as it's read in
        self.flat_curve_data = [{} for _ in range(self.num_of_curves)]

    def extract_global_metadata(self):
        """Extracts the global metadata from the JPK QI archive and populates the top level metadata dictionary and segment channels list."""

        # Load the metadata from the global properties file
        if "header.properties" in self.qi_archive.namelist():
            with self.qi_archive.open("header.properties") as archive_meta_file:
                props = javaproperties.load(archive_meta_file)

                # Add all the data from the main header to the top level metadata with a prefix to avoid key clashes
                for key, value in props.items():
                    self.top_level_meta[f"main-header.{key}"] = value
        else:
            logger.error(f"File {self.filepath} does not contain essential metadata and cannot be loaded")

        # Load the metadata from the shared header and parse the channel information for the segments
        if "shared-data/header.properties" in self.qi_archive.namelist():
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
