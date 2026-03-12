from pathlib import Path
from contextlib import nullcontext
import io
import re
import zipfile

import numpy as np
import javaproperties
import h5py

from AFMReader.logging import logger
from AFMReader import jpk


ADDITIONAL_CHANNELS = ["contactPoint", "manualTriggerPoint"]
ADDITIONAL_CHANNELS_IN_M = ["contactPoint", "manualTriggerPoint"]

def _get_channel_scaling(props, channel_index):
    """
    Parses the JPK properties dictionary to find the cumulative multiplier
    and offset for a specific channel index (e.g., '1' for vDeflection).
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

def _load_preprocessed_image(qi_archive, channel, config_path=None):
    path_to_image = None
    for file_name in qi_archive.namelist():
        if file_name.endswith(".jpk-qi-image"):
            path_to_image = file_name
    if path_to_image is None:
        raise FileNotFoundError(f"{path_to_image} not found in JPK archive")

    tif_bytes = qi_archive.read(path_to_image)

    virtual_file = io.BytesIO(tif_bytes)
    logger.info(f"Looking for channel {channel} in ")
    return jpk._load_jpk(virtual_file, path_to_image, channel=channel, file_suffix=".jpk-qi-data", config_path=config_path, flip_image=False)

class jpk_qi_loader:
    """Class for readability and improving modularity in the load jpk qi data function"""
    def __init__(self,
        filepath: Path | str,
        channel: str,
        config_path: Path | str | None = None,
        flip_image: bool | None = True,
        save_as: str | None = None):
        """
        Initializes the loader with the provided parameters.

        Parameters
        ----------
        filepath : Path | str
            The path to the .jpk-qi file to be loaded.
        channel : str
            The specific channel to be extracted from the file (e.g., "measuredHeight")."
        config_path : Path | str | None, optional
            The path to the configuration file, if any. Default is None.
        flip_image : bool | None, optional
            Whether to flip the image vertically. Default is True.
        save_as : str | None, optional
            The format to save the loaded data. Default is None.
        """

        self.filepath = Path(filepath)
        self.channel = channel
        self.config_path = config_path
        self.flip_image = flip_image
        self.save_as = save_as

        # Initialize key attributes that will be returned / accessed frequently

        # Just the top level metadata extracted from the header files
        self.top_level_meta = {}
        # A dictionary containing all metadata, splitting the top level metadata and the metadata for each curve and segment
        self.full_metadata = {}

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
        # Define the image shape and size attributes
        self.size_x, self.size_y, self.shape_x, self.shape_y = None, None, None, None

        # Instantiate containers for data to be saved (so an exception is not caused if not saving)
        self.curve_datasets = None

    def load(self):
        with zipfile.ZipFile(self.filepath, "r") as qi_archive:
            self.extract_global_metadata(qi_archive)

            self.parse_dimension_data()

            # Pre-allocate the image array based on the dimensions parsed from the metadata
            self.image = np.zeros((self.shape_y, self.shape_x), dtype=np.float32)

            # Access the curve data and metadata, and save to given file format
            with self.get_saving_context() as file:

                # Compile Regexes
                dat_regex = re.compile(r"index/(\d+)/segments/(\d+)/channels/([^/]+)\.dat")
                curve_meta_regex = re.compile(r"index/(\d+)/header\.properties")
                segment_meta_regex = re.compile(r"index/(\d+)/segments/(\d+)/segment-header\.properties")

                # Setup H5 Data structures if needed
                if self.save_as == "h5":
                    self.curve_datasets, self.global_meta_group, self.curves_meta_group = self.setup_h5_structure(file)

                for file_info in qi_archive.infolist():
                    filename = file_info.filename

                    # Check Binary Data
                    dat_match = dat_regex.match(filename)
                    if dat_match:
                        # If file is a .dat file, extract the curve number, segment direction and channel name from the filename
                        curve_num, direction, chan_name = int(dat_match.group(1)), int(dat_match.group(2)), dat_match.group(3)
                        # Then load the data from the file
                        self.extract_dat_file(qi_archive, file_info, curve_num, direction, chan_name)
                        continue

                    # Check Curve Metadata
                    curve_meta_match = curve_meta_regex.match(filename)
                    if curve_meta_match:
                        # If file is a curve metadata file, extract the curve number from the filename
                        curve_num = int(curve_meta_match.group(1))
                        # Then load the metadata from the file
                        self.extract_curve_metadata(qi_archive, file_info, curve_num)
                        continue

                    # Check Segment Metadata
                    segment_meta_match = segment_meta_regex.match(filename)
                    if segment_meta_match:
                        # If file is a segment metadata file, extract the curve number and segment direction from the filename
                        curve_num, direction = int(segment_meta_match.group(1)), int(segment_meta_match.group(2))
                        # Then load the segment metadata from the file
                        self.extract_segment_metadata(qi_archive, file_info, curve_num, direction)
                        continue

                self.all_curve_data = self.process_flat_curve_data()

                # TODO can we remove curve_meta_dict and just use curve_meta or is the non duplicating necessary
                self.curve_meta = [self.curve_meta_dict.get(i, {}) for i in range(self.num_of_curves)]
                self.segment_meta = [self.segment_meta_dict.get(i, {}) for i in range(self.num_of_curves * 2)]
                self.full_metadata = self.construct_full_metadata()

                # Save the full metadata to appropiate format if saving
                self.save_metadata()


            # Convert to nanometers if in meters
            if self.channel in ADDITIONAL_CHANNELS_IN_M:
                self.image = self.image * 1e9

            # Save a lite form of the images (precalculated) if saving to a file
            if self.save_as is not None:
                self.save_lite_data(qi_archive)

            # Need to include flip image as _load_jpk flip image is set to false
            if self.flip_image:
                self.image = np.flipud(self.image)
        if self.all_curve_data:
            return (self.image, self.px2nm, (self.all_curve_data, self.channels_units, self.full_metadata))

        return self.image, self.px2nm

    def process_flat_curve_data(self):
        all_curve_data = []
        for y in range(self.shape_y):
            row = []
            for x in range(self.shape_x):
                curve_num = y * self.shape_x + x
                curve_data = self.flat_curve_data[curve_num]
                row.append(curve_data)

                # Calculate on-the-fly image data if required
                if self.channel in ADDITIONAL_CHANNELS:
                    seg_0_dict = {c: data["Segment_0"] for c, data in curve_data.items() if "Segment_0" in data}
                    if self.channel == "contactPoint":
                        self.image[y, x] = _find_contact_point(seg_0_dict)
                    elif self.channel == "manualTriggerPoint":
                        self.image[y, x] = _find_trigger_point(seg_0_dict)
            all_curve_data.append(row)
        return all_curve_data

    def save_lite_data(self, qi_archive):
        """
        Saves a lite form of the data (e.g., the calculated image data) to the appropriate format based on the save_as attribute.

        Parameters
        ----------
        qi_archive : zipfile.ZipFile
            The archive containing the .jpk-qi-image file.
        """
        if self.save_as == "h5":
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
                for file_name in qi_archive.namelist():
                    if file_name.endswith(".jpk-qi-image"):
                        path_to_image = file_name
                        break
                # Add the channels which exist in the jpk-qi-image file
                with qi_archive.open(path_to_image, "r") as image_file:
                    h5_channels += jpk._get_jpk_channels(file=image_file, filename=self.filepath.stem, file_path=self.filepath / Path(path_to_image))
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
                    if h5_channel == self.channel:
                        channel_image = self.image
                    else:
                        channel_image, _ = _load_preprocessed_image(qi_archive=qi_archive, channel=h5_channel, config_path=self.config_path)
                    frame_stack = channel_image.flatten().reshape(-1, 1)

                    # Update/ replace the channels dataset
                    if dataset_name in chan_grp:
                        del chan_grp[dataset_name]
                    chan_grp.create_dataset(dataset_name, data=frame_stack)


    def save_metadata(self):
        """Saves the metadata to the appropriate format based on the save_as attribute."""
        if self.save_as == "h5":
            for seg_chan in self.segment_channels:
                self.global_meta_group.attrs[f"channel.unit.{seg_chan['name']}"] = seg_chan['unit']
            for key, value in self.top_level_meta.items():
                self.global_meta_group.attrs[key] = str(value).encode('utf-8')
            for i, c_meta in enumerate(self.curve_meta):
                curve_meta_group = self.curves_meta_group.require_group(f"{i}")
                for key, value in c_meta.items():
                    curve_meta_group.attrs[key] = str(value).encode('utf-8')

                for d in range(2):
                    segment_meta_group = curve_meta_group.require_group(f"{d}")
                    for key, value in self.segment_meta[i*2+d].items():
                        segment_meta_group.attrs[key] = str(value).encode('utf-8')


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
        changing_curve_keys = {k for k in self.all_curve_keys if any(self.curve_meta[i].get(k) != self.curve_meta[0].get(k) for i in range(1, self.num_of_curves))}
        changing_segment_keys = {k for k in self.all_segment_keys if any(self.segment_meta[i].get(k) != self.segment_meta[0].get(k) for i in range(1, len(self.segment_meta)))}

        # Move duplicated meta to top level for both segments and curves
        for key in self.all_curve_keys - changing_curve_keys:
            if self.curve_meta and key in self.curve_meta[0]:
                self.top_level_meta[f"curve.{key}"] = self.curve_meta[0][key]
        for key in self.all_segment_keys - changing_segment_keys:
            if self.segment_meta and key in self.segment_meta[0]:
                self.top_level_meta[f"segment.{key}"] = self.segment_meta[0][key]

        # Strip duplicated keys from individual curve/segment dicts
        for c_meta in self.curve_meta:
            for k in self.all_curve_keys - changing_curve_keys: c_meta.pop(k, None)
        for s_meta in self.segment_meta:
            for k in self.all_segment_keys - changing_segment_keys: s_meta.pop(k, None)

        # Construct full metadata dict from subdicts
        full_metadata = {
            "top_level": self.top_level_meta,
            "curves": self.curve_meta,
            "segments": self.segment_meta
        }
        return full_metadata

    def extract_dat_file(self, qi_archive: zipfile.ZipFile, file_info: zipfile.ZipInfo, curve_num: int, direction: int, chan_name: str):
        """
        Extracts the data from a .dat file in the JPK QI archive, applies the appropriate scaling, and saves it to the internal data structure and h5 dataset if required.

        Parameters
        ----------
        qi_archive : zipfile.ZipFile
            The JPK QI archive from which to extract the .dat file.
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
            with qi_archive.open(file_info) as f:
                # Read the binary data as big-endian 32-bit integers
                raw_array = np.frombuffer(f.read(), dtype='>i4')
                # Apply the scaling to convert raw values into real world values
                segment_array = (raw_array * scale["multiplier"]) + scale["offset"]

            # If the channel doesn't exist in the curve data for this curve, add it as a new entry
            if chan_name not in self.flat_curve_data[curve_num]:
                self.flat_curve_data[curve_num][chan_name] = {}
            # Add the segment data to the curve data under the appropriate channel and segment direction
            self.flat_curve_data[curve_num][chan_name][f"Segment_{direction}"] = segment_array

            # Update the dataset if saving as h5
            if self.curve_datasets is not None:
                self.curve_datasets[f"{direction}_{chan_name}"][curve_num] = segment_array
        else:
            logger.warning(f"Channel {chan_name} not found in scaling information. Skipping data for curve {curve_num}, direction {direction}.")


    def extract_curve_metadata(self, qi_archive: zipfile.ZipFile, file_info: zipfile.ZipInfo, curve_num: int):
        """
        Extracts the curve metadata from its header.properties file in the JPK QI archive and saves it to the internal data structure.

        Parameters
        ----------
        qi_archive : zipfile.ZipFile
            The JPK QI archive from which to extract the curve metadata.
        file_info : zipfile.ZipInfo
            The ZipInfo object corresponding to the header.properties file to be extracted.
        curve_num : int
            The curve number associated with the metadata, parsed from the filename.
        """
        with qi_archive.open(file_info) as f:
            cleaned_meta = {".".join(k.split(".")[1:]): v for k, v in javaproperties.load(f).items()}
            self.curve_meta_dict[curve_num] = cleaned_meta
            self.all_curve_keys.update(cleaned_meta.keys())


    def extract_segment_metadata(self, qi_archive: zipfile.ZipFile, file_info: zipfile.ZipInfo, curve_num: int, direction: int):
        """
        Extracts the segment metadata from its header.properties file in the JPK QI archive and saves it to the internal data structure.

        Parameters
        ----------
        qi_archive : zipfile.ZipFile
            The JPK QI archive from which to extract the segment metadata.
        file_info : zipfile.ZipInfo
            The ZipInfo object corresponding to the header.properties file to be extracted.
        curve_num : int
            The curve number associated with the metadata, parsed from the filename.
        direction : int
            The segment direction (0 or 1) associated with the metadata, parsed from the filename.
        """
        idx = curve_num * 2 + direction
        with qi_archive.open(file_info) as f:
            cleaned_meta = {".".join(k.split(".")[1:]): v for k, v in javaproperties.load(f).items()}
            self.segment_meta_dict[idx] = cleaned_meta
            self.all_segment_keys.update(cleaned_meta.keys())

    def setup_h5_structure(self, h5file):
        """
        Sets up the structure in the h5 file for saving the curve data and metadata, and returns the datasets and metadata groups for later use.

        Parameters
        ----------
        h5file : h5py.File
            The h5 file in which to set up the structure for saving the curve data and metadata.
        Returns
        -------
        curve_datasets : dict
            A dictionary containing the datasets for each curve and segment direction.
        global_meta_group : h5py.Group
            The h5 group for storing global metadata.
        curves_meta_group : h5py.Group
            The h5 group for storing curve metadata.
        """
        vlen_type = h5py.vlen_dtype(np.float32)

        # Create the main group for the QI curve data that all the curve data will be in
        qi_group = h5file.require_group("QI_Curve_Data")
        curve_datasets = {}

        # Establish empty groups for global metadata and curve metadata
        global_meta_group = qi_group.require_group("Global_Metadata")
        curves_meta_group = qi_group.require_group("Curve_Metadata")

        for direction in range(2):
            # For each segment direction, establish a group and datasets for each channel in the segment channels list
            dir_group = qi_group.require_group(f"Segment_{direction}")
            for seg_chan in self.segment_channels:
                ds_name = seg_chan["name"]
                if ds_name not in dir_group:
                    # Create the dataset for the given channel and segment direction if it doesn't already exist
                    curve_datasets[f"{direction}_{ds_name}"] = dir_group.create_dataset(
                        ds_name, shape=(self.num_of_curves,), dtype=vlen_type
                    )
                else:
                    # If the dataset already exists, just add it to the curve datasets dictionary for later use
                    curve_datasets[f"{direction}_{ds_name}"] = dir_group[ds_name]
        return curve_datasets, global_meta_group, curves_meta_group

    def get_saving_context(self):
        """
        Returns the appropriate context manager for saving the data based on the save_as attribute.
        If save_as is "h5", it returns a context manager for an h5 file. Otherwise, it returns a null context.

        Returns
        -------
        contextlib.AbstractContextManager
            The context manager for saving the data.
        """
        if self.save_as == "h5":
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

    def extract_global_metadata(self, qi_archive):
        """
        Extracts the global metadata from the JPK QI archive and populates the top level metadata dictionary and segment channels list.

        Parameters
        ----------
        qi_archive : zipfile.ZipFile
            The JPK QI archive from which to extract metadata.
        """

        # Load the metadata from the global properties file
        if "header.properties" in qi_archive.namelist():
            with qi_archive.open("header.properties") as archive_meta_file:
                props = javaproperties.load(archive_meta_file)

                # Add all the data from the main header to the top level metadata with a prefix to avoid key clashes
                for key, value in props.items():
                    self.top_level_meta[f"main-header.{key}"] = value
        else:
            logger.error(f"File {self.filepath} does not contain essential metadata and cannot be loaded")

        # Load the metadata from the shared header and parse the channel information for the segments
        if "shared-data/header.properties" in qi_archive.namelist():
            with qi_archive.open("shared-data/header.properties") as shared_data_file:
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
        self.channels_units = {seg_chan['name'] : seg_chan['unit'] for seg_chan in self.segment_channels}
        # Lookup map for binary scaling
        self.channel_scaling = {chan["name"]: chan for chan in self.segment_channels}



def load_jpk_qi(
    file_path: Path | str,
    channel: str,
    config_path: Path | str | None = None,
    flip_image: bool | None = True,
    save_as_h5: bool | None = False
) -> tuple[np.ndarray, float]:

    jpk_loader = jpk_qi_loader(filepath=file_path, channel=channel, config_path=config_path, flip_image=flip_image, save_as="h5" if save_as_h5 else None)
    return jpk_loader.load()


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


def _make_num_min_characters(num : int, min_chars: int = 3):
    string_num = str(num)
    if len(string_num) >= min_chars:
        return string_num
    string_num = "0" * (min_chars - len(string_num)) + string_num
    return string_num

def get_jpk_qi_channels(file_path: Path | str):
    logger.debug("Starting to get jpk qi data channels")
    file_path = Path(file_path)
    channels = []
    with zipfile.ZipFile(file_path, "r") as qi_archive:
        for file_name in qi_archive.namelist():
            if file_name.endswith(".jpk-qi-image"):
                path_to_image = file_name
        with qi_archive.open(path_to_image, "r") as image_file:
            channels += jpk._get_jpk_channels(file=image_file, filename=file_path.stem, file_path=file_path / Path(path_to_image))
    channels += ADDITIONAL_CHANNELS
    logger.debug("Got jpk qi data channels")
    return channels, {"save_as_h5": bool}

def _find_contact_point(curve):
    # find contact point in vertical deflection by peak in first derivative
    vdef = curve["vDeflection"]
    if len(vdef) < 2:
        return np.nan
    derivative_vert_deflection = np.diff(vdef)
    # Doesn't look like this line is needed: peak_derivative_value = np.max(derivative_vert_deflection)
    peak_derivative_index = np.argmax(derivative_vert_deflection)

    # find corresponding height value
    corresponding_height_at_peak = curve["measuredHeight"][peak_derivative_index]

    return corresponding_height_at_peak

def _find_trigger_point(curve):
    trigger_point = curve["measuredHeight"][-1]
    return trigger_point

def _max_points_buffer(curves_data, samples=20, points_buffer=1.2):

    step = len(curves_data) // samples
    max_points = np.max(len(curves_data[i]["segment"]) for i in range(0, len(curves_data), step))
    return max_points * points_buffer



