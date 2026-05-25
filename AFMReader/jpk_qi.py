"""
Module to decode and load JPK QI (Quantitative Imaging) data files.

It provides lazy loading for curve data and metadata to minimize memory usage,
and supports exporting to HDF5 format.
"""

# pylint: disable=too-many-lines,too-many-positional-arguments,too-few-public-methods,too-many-instance-attributes
# pylint: disable=too-many-locals,too-many-branches,protected-access,attribute-defined-outside-init,fixme

import io
import zipfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import javaproperties
import h5py
from tqdm import tqdm

from AFMReader.lazy_data_classes import LazyMetadata, LazyMetaProxy, LazyQiData
from AFMReader.logging import logger
from AFMReader import jpk


class LazyJpkQiData(LazyQiData):
    """
    A proxy class that behaves like a 2D list of shape (shape_y, shape_x) but fetches .dat file data on demand.

    Parameters
    ----------
    filepath : str
        Path to the .jpk file.
    shape_x : int
        Number of columns in the image.
    shape_y : int
        Number of rows in the image.
    channel_scaling : dict
        Dictionary containing scaling information for each channel.
    archive : zipfile.ZipFile
        The opened ZIP archive containing the .dat files.
    flip_image : bool, optional
        Whether to flip the image vertically. Default is ``True``.
    """

    def __init__(self, filepath, shape_x: int, shape_y: int, channel_scaling, archive, flip_image: bool = True):
        """
        Initialize the LazyJpkQiData instance.

        Parameters
        ----------
        filepath : str
            Path to the .jpk file.
        shape_x : int
            Number of columns in the image.
        shape_y : int
            Number of rows in the image.
        channel_scaling : dict
            Dictionary containing scaling information for each channel.
        archive : zipfile.ZipFile
            The opened ZIP archive containing the .dat files.
        flip_image : bool, optional
            Whether to flip the image vertically. Default is ``True``.
        """
        super().__init__(shape_x, shape_y, flip_image)
        self.filepath = filepath
        self.channel_scaling = channel_scaling
        self.archive = archive

    def __iter__(self):
        """Yield the curve data for each pixel in the image, iterating in row-major order (y first, then x)."""
        for y in range(self.shape_y):
            for x in range(self.shape_x):
                yield self._fetch_curve(y, x)

    def _fetch_curve(self, y: int, x: int):
        """
        Fetch the curve data for a specific pixel.

        Parameters
        ----------
        y : int
            Row index of the pixel.
        x : int
            Column index of the pixel.

        Returns
        -------
        dict
            Dictionary containing the curve data for the specified pixel.
        """
        if y < 0 or y >= self.shape_y or x < 0 or x >= self.shape_x:
            raise IndexError(f"Curve index out of bounds: ({x}, {y})")
        if self.flip_image:
            y = self.shape_y - 1 - y
        curve_num = y * self.shape_x + x
        curve_data: dict[str, Any] = {}

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

        Returns
        -------
        list
            A 2D list containing dictionaries with curve data for each pixel.
        """
        all_curve_data = [[None for _ in range(self.shape_x)] for _ in range(self.shape_y)]
        for y in range(self.shape_y):
            for x in range(self.shape_x):
                all_curve_data[y][x] = self._fetch_curve(y, x)
        # TODO may be good to just return self here as not faster and lots of memory
        # return self
        return all_curve_data

    def close(self):
        """Close the ZIP archive when done to free up resources."""
        self.archive.close()


class LazyQiMetadata(LazyMetadata):
    """
    A proxy class that fetches header.properties files on demand.

    It behaves like a 2D array of shape (shape_y, shape_x) where each element
    is a dictionary containing the requested metadata for that pixel.

    Parameters
    ----------
    filepath : str
        Path to the .jpk file.
    top_level_meta : dict
        Dictionary containing the top-level metadata extracted from the header files.
    archive : zipfile.ZipFile
        The opened ZIP archive containing the JPK file contents.
    shape_x : int
        The number of columns in the image.
    shape_y : int
        The number of rows in the image.
    flip_image : bool, optional
        Whether to flip the image vertically. Default is True.
    """

    def __init__(self, filepath, top_level_meta, archive, shape_x: int, shape_y: int, flip_image: bool = True):
        """
        Initialize the LazyQiMetadata instance.

        Parameters
        ----------
        filepath : str
            Path to the .jpk file.
        top_level_meta : dict
            Dictionary containing the top-level metadata extracted from the header files.
        archive : zipfile.ZipFile
            The opened ZIP archive containing the JPK file contents.
        shape_x : int
            The number of columns in the image.
        shape_y : int
            The number of rows in the image.
        flip_image : bool, optional
            Whether to flip the image vertically. Default is True.
        """
        self.filepath = filepath
        # Expose top_level so the frontend can still do `raw_metadata["top_level"]`
        self.archive = archive
        super().__init__(top_level_meta, shape_x, shape_y, flip_image)

    def __getitem__(self, key):
        """
        Fetch requested metadata based on key.

        If the key is 'top_level', it returns the top-level metadata.
        If the key is 'curves' or 'segments', it returns a LazyQiMetaProxy
        that can be used to fetch curve or segment metadata on demand.

        Parameters
        ----------
        key : str
            The key to fetch metadata for.

        Returns
        -------
        dict or LazyQiMetaProxy
            The requested metadata.
        """
        if key == "top_level":
            return self.top_level
        if key == "curves":
            return LazyQiMetaProxy(self.filepath, "curve", self.archive, self.shape_x, self.shape_y, self.flip_image)
        if key == "segments":
            return LazyQiMetaProxy(self.filepath, "segment", self.archive, self.shape_x, self.shape_y, self.flip_image)
        raise KeyError(key)


class LazyQiMetaProxy(LazyMetaProxy):
    """
    A proxy class to represent curve and segment metadata.

    It behaves like a 2D list of shape (shape_y, shape_x) but fetches header.properties files on demand for
    curves or segments.

    Parameters
    ----------
    filepath : str
        Path to the .jpk file.
    meta_type : str
        The type of metadata to fetch ('curve' or 'segment').
    archive : zipfile.ZipFile
        The opened ZIP archive containing the JPK file contents.
    shape_x : int
        The number of columns in the image.
    shape_y : int
        The number of rows in the image.
    flip_image : bool, optional
        Whether to flip the image vertically. Default is True.
    """

    def __init__(self, filepath, meta_type, archive, shape_x: int, shape_y: int, flip_image: bool = True):
        """
        Initialize the LazyQiMetaProxy instance.

        Parameters
        ----------
        filepath : str
            Path to the .jpk file.
        meta_type : str
            The type of metadata to fetch ('curve' or 'segment').
        archive : zipfile.ZipFile
            The opened ZIP archive containing the JPK file contents.
        shape_x : int
            The number of columns in the image.
        shape_y : int
            The number of rows in the image.
        flip_image : bool, optional
            Whether to flip the image vertically. Default is True.
        """
        self.filepath = filepath
        self.archive = archive
        super().__init__(meta_type, shape_x, shape_y, flip_image)

    def _fetch_meta(self, y: int, x: int, direction: int | None = None):
        """
        Fetch the metadata for a specific curve or segment.

        Parameters
        ----------
        y : int
            Row index of the curve or segment.
        x : int
            Column index of the curve or segment.
        direction : int, optional
            The direction index for segment metadata. Required if meta_type is 'segment'.

        Returns
        -------
        dict
            The metadata dictionary for the specified curve or segment.
        """
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
    Parse the JPK properties dictionary to find cumulative multiplier and offset for a specific channel index.

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
    """
    Class for readability and improving modularity in the load jpk qi data function.

    Parameters
    ----------
    filepath : Path | str
        The path to the .jpk-qi file to be loaded.
    channel : str | None, optional
        The specific channel to be extracted (e.g., "measuredHeight"). Default is None.
    config_path : Path | str | None, optional
        The path to the configuration file, if any. Default is None.
    flip_image : bool | None, optional
        Whether to flip the image vertically. Default is True.
    save_as_h5 : bool, optional
        Whether to save the loaded data as an H5 file. Default is False.
    """

    def __init__(
        self,
        filepath: Path | str,
        channel: str | None = None,
        config_path: Path | str | None = None,
        flip_image: bool | None = True,
        save_as_h5: bool = False,
    ):
        """
        Initialize the loader with the provided parameters.

        Parameters
        ----------
        filepath : Path | str
            The path to the .jpk-qi file to be loaded.
        channel : str | None, optional
            The specific channel to be extracted (e.g., "measuredHeight"). Default is None.
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

        # Open the ZIP archive once and keep it open for the duration of the loading process
        self.qi_archive = zipfile.ZipFile(self.filepath, "r")  # pylint: disable=consider-using-with
        logger.info(f"Opened JPK QI archive at {self.filepath}")
        self.namelist = self.qi_archive.namelist()
        # Set path to the .jpk-qi-image file within the archive for later use
        self.path_to_image = None

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

        # Initialize key attributes that will be returned / accessed frequently

        # Just the top level metadata extracted from the header files
        self.top_level_meta: dict[str, Any] = {}
        # A lazy reference containing all metadata
        self.full_metadata: LazyQiMetadata | None = None
        # A 2D list of curve data dictionaries
        self.curve_data: Any = None
        # A lookup for channel name to unit to be returned
        self.channels_units: dict[str, str] = {}
        # The list of channels for the segments with their scaling information extracted from the shared header
        self.segment_channels: list[dict[str, Any]] = []
        self.curve_meta: dict[str, Any] = {}
        self.segment_meta: dict[str, Any] = {}

        # Define the image shape and size attributes
        self.size_x: float | None = None
        self.size_y: float | None = None
        self.shape_x: int | None = None
        self.shape_y: int | None = None
        self.failed_curves: set[tuple[int, int | None, str | None]] = set()

        # Instantiate containers for data to be saved (so an exception is not caused if not saving)
        self.curve_groups = None
        self.saved_to_h5 = False

        self.extract_global_metadata()

        self.parse_dimension_data()

    def get_available_channels(self):
        """
        Retrieve available channels from the .jpk-qi-image file within the archive.

        Returns
        -------
        channels : list
            A list of available channels including the calculated channels.
        metadata_options : dict
            A dictionary of options for what metadata to return.
        """
        # Look for the jpk-qi-image file in the archive
        if self.path_to_image is None:
            for file_name in self.namelist:
                if file_name.endswith(".jpk-qi-image"):
                    self.path_to_image = file_name

        # Add the channels which exist in the jpk-qi-image file
        with self.qi_archive.open(self.path_to_image, "r") as image_file:
            return jpk._get_jpk_channels(
                file=image_file, filename=self.filepath.stem, file_path=self.filepath / Path(self.path_to_image)
            )

    def get_additional_params(self) -> dict[str, type]:
        """
        Get additional parameters that can be passed to the load function.

        Returns
        -------
        dict
            A dictionary of additional parameters with their types.
        """
        return {"save_as_h5": bool}

    def load(
        self,
        channel: str | None = None,
        config_path: Path | str | None = None,
        flip_image: bool | None = True,
        save_as_h5: bool = False,
    ) -> tuple[np.ndarray, float, str, tuple[LazyQiData, dict[str, str], LazyMetadata]]:
        """
        Load the .jpk-qi-data file.

        Parameters
        ----------
        channel : str | None, optional
            The specific channel to be extracted. Default is None.
        config_path : Path | str | None, optional
            Path to the configuration file. Default is None.
        flip_image : bool | None, optional
            Whether to flip the image. Default is True.
        save_as_h5 : bool, optional
            Whether to save the loaded data as an H5 file. Default is False.

        Returns
        -------
        tuple[np.ndarray, float, str, tuple[LazyQiData, dict[str, str], LazyMetadata]]
            A tuple containing image data, scaling factor, z-axis unit, and curve data.
        """
        # Update instance attributes based on provided parameters
        self.channel = channel if channel else self.channel
        self.config_path = config_path if config_path else self.config_path
        self.flip_image = flip_image if flip_image is not None else self.flip_image
        self.save_as_h5 = save_as_h5

        # TODO add a save to h5 option here?

        logger.info(f"Loading JPK QI data from {self.filepath} with channel {self.channel}")

        # Establish the lazy loading structures for curve data and metadata.
        self.full_metadata = LazyQiMetadata(
            self.filepath,
            self.top_level_meta,
            self.qi_archive,
            self.shape_x or 0,
            self.shape_y or 0,
            flip_image=bool(self.flip_image),
        )
        self.curve_data = LazyJpkQiData(
            self.filepath,
            self.shape_x or 0,
            self.shape_y or 0,
            self.channel_scaling,
            self.qi_archive,
            flip_image=bool(self.flip_image),
        )

        # Load the image
        self.image, _, self.z_unit = self.get_image()

        return (self.image, self.px2nm, self.z_unit, (self.curve_data, self.channels_units, self.full_metadata))

    def output_summary(self):
        """Output a summary of the loading process, including any failed curve loads and their details."""
        if self.failed_curves:
            logger.warning(f"Failed to load {len(self.failed_curves)} files.")
            logger.warning("Summary of missing files (up to 10 shown):")

            # Output the first 10 failed loads with details
            for i, (curve_num, direction, chan_name) in enumerate(self.failed_curves):
                if i < 10:
                    if chan_name:
                        logger.warning(
                            f"Failed to load data for curve {curve_num}, direction {direction}, channel {chan_name}"
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
            f"Loading all curve data from JPK QI archive with {len(self.namelist)} files "
            f"{'' if include_metadata else 'not '}including metadata"
        )
        if include_metadata:
            # Prepare keys for metadata to speed up processing
            curve_work = [
                (f"{k}=".encode(), h5_meta_datasets[f"curve.{k}"], h5_meta_datasets_buffer[f"curve.{k}"])
                for k in self.changing_curve_keys
            ]
            seg_work = [
                (f"{k}=".encode(), h5_meta_datasets[f"segment.{k}"], h5_meta_datasets_buffer[f"segment.{k}"])
                for k in self.changing_segment_keys
            ]
        for curve_num in tqdm(range(self.num_of_curves)):

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

        # Add the last index to the indices datasets to mark the end of the last curve
        for direction in range(2):
            seg_name = f"Segment_{direction}"
            for chan in self.segment_channels:
                chan_name = chan["name"]
                current_dataset = h5_datasets[seg_name][chan_name]["Data"]
                indices_dataset = h5_datasets[seg_name][chan_name]["Indices"]
                indices_dataset[-1] = current_dataset.shape[0]

    def save_to_h5(
        self,
        include_metadata: bool = True,
    ) -> Path:
        """
        Save data as an H5 file. If include_metadata is False, only curve data is saved.

        Parameters
        ----------
        include_metadata : bool, optional
            If True, metadata will be included in the saved H5 file. Default is True.

        Returns
        -------
        Path
            The path to the saved H5 file.
        """
        # Determine the path for the H5 file, ensuring it does not overwrite an existing file
        self.h5_path = self.filepath.parent / f"{self.filepath.stem}.h5-jpk"
        i = 0
        while self.h5_path.exists():
            self.h5_path = self.filepath.parent / f"{self.filepath.stem}_{i}.h5-jpk"
            i += 1

        with self.get_saving_context() as file:

            t0 = time.perf_counter()

            # Sample curves in dataset to make a best guess for the meta keys
            self.changing_curve_keys, self.changing_segment_keys = self.get_changing_keys()
            self.points_for_channel_segment = self.predict_total_points()
            self.t_changing_keys = time.perf_counter() - t0

            # Setup H5 structure for saving the data
            global_meta_group, h5_datasets, h5_meta_datasets, h5_datasets_buffer, h5_meta_datasets_buffer = (
                self.setup_h5_structure(file)
            )

            # Set up current_offsets to keep track of how many points have been read
            self.current_offsets: dict[int, dict[str, int]] = {}
            for direction in range(2):
                self.current_offsets[direction] = {}
                for chan in self.segment_channels:
                    self.current_offsets[direction][chan["name"]] = 0

                    # Reset points_for_channel_segment to 0 to store actual number of points
                    self.points_for_channel_segment[direction][chan["name"]] = 0

            # Extract data from the JPK QI archive and save to H5 datasets
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
                    h5_datasets[f"Segment_{direction}"][chan["name"]]["Data"].resize(
                        (self.points_for_channel_segment[direction][chan["name"]],)
                    )

            self.output_summary()

            if include_metadata:
                # Save the global metadata to the h5 file
                for key, value in self.get_collated_metadata().items():
                    global_meta_group.attrs[key] = str(value).encode("utf-8")

            self.save_lite_data()

            logger.info(f"QI data copied to h5 data {file.filename}")
            return self.h5_path

    def get_curves_sample(self):
        """
        Get a sample of curve numbers distrubuted evenly across the dataset.

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
        Predict the total number of points for each channel and segment.

        This is done by sampling a subset of curves and extrapolating based on the maximum number
        of points found in the sample.

        Returns
        -------
        dict:
            A dictionary containing the predicted total points for each channel and segment.
        """
        # Get a sample of curve (indices)
        curves_to_check = self.get_curves_sample()
        points_for_channel_segment = {}

        # Iterate through the segments, channels and our curve indices
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
                            # If the file doesn't exist for this curve, check the next curve so we don't just get
                            # a smaller sample
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

    def get_changing_keys(self):  # noqa: C901
        """
        Check a sample of curves to see which metadata keys change across curves and segments.

        This allows us to extract only the changing keys for each curve and segment.
        Non-changing keys are moved to the top-level metadata and not extracted for each curve/segment.

        Returns
        -------
        tuple:
            A tuple containing two sets: changing_curve_keys and changing_segment_keys.
        """
        curve_meta_dict: dict[str, list[Any]] = {}
        segment_meta_dict: dict[str, list[Any]] = {}
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
            if len({v for v in values if v is not None}) > 1:
                changing_curve_keys.add(key)
            else:
                # If the key does not change across curves, move it to the top level metadata
                self.top_level_meta[f"curve.{key}"] = values[0]
        for key, values in segment_meta_dict.items():
            if len({v for v in values if v is not None}) > 1:
                changing_segment_keys.add(key)
            else:
                # If the key does not change across segments, move it to the top level metadata
                self.top_level_meta[f"segment.{key}"] = values[0]
        return changing_curve_keys, changing_segment_keys

    def get_collated_metadata(self):
        """
        Collate metadata from being split by curve to being split by attribute.

        Returns
        -------
        dict
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
    ) -> tuple[np.ndarray, float, str]:
        """
        Process the flat curve data dictionary into a 2D list structure matching the image dimensions.

        Parameters
        ----------
        overide_channel : str | None, optional
            Channel name to use instead of the instance default. Default is None.
        convert_to_nm : bool, optional
            Whether to convert the image data to nanometres. Default is True.
        flip_image : bool | None, optional
            Whether to flip the image vertically. Defaults to the instance setting if None.

        Returns
        -------
        tuple[np.ndarray, float, str]
            A 2D array representing the image data, the pixel-to-nm scaling factor, and the unit of the z-axis.
        """
        # Get channel and flip_image parameters
        channel = str(overide_channel) if overide_channel else str(self.channel)

        if flip_image is None:
            flip_image = bool(self.flip_image)

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
            flip_image=bool(flip_image),
        )

    def save_lite_data(self):
        """Save a lite form of the data (e.g., the calculated image data) to H5."""
        with h5py.File(self.h5_path, "a") as h5file:
            # Save data required for reading the h5 file as a normal image file
            meas_grp = h5file.require_group("Measurement_000")
            # Save dimensions data
            meas_grp.attrs["position-pattern.grid.ulength"] = self.size_x
            meas_grp.attrs["position-pattern.grid.ilength"] = self.shape_x
            meas_grp.attrs["position-pattern.grid.vlength"] = self.size_y
            meas_grp.attrs["position-pattern.grid.jlength"] = self.shape_y
            meas_grp.attrs["timing-settings.scanRate"] = 1.0  # Dummy value to satisfy reader

            logger.info(f"Saving a hdf5 copy of the data {self.h5_path}")

            # Look for the jpk-qi-image file in the archive
            path_to_image = None
            for file_name in self.namelist:
                if file_name.endswith(".jpk-qi-image"):
                    path_to_image = file_name
                    break
            # Add the channels which exist in the jpk-qi-image file
            h5_channels = []
            if path_to_image:
                with self.qi_archive.open(path_to_image, "r") as image_file:
                    h5_channels = jpk._get_jpk_channels(
                        file=image_file, filename=self.filepath.stem, file_path=self.filepath / Path(path_to_image)
                    )
            else:
                logger.warning(
                    f"No image data found in {self.filepath}. Cannot save image data to H5."
                    f"Please check the file and channel name."
                )
                return

            for i, h5_channel in enumerate(h5_channels):
                # For each available channel, save the required data to the h5 file
                # TODO make sure this metadata is accurate for the channels coming from the .jpk-qi-image file
                chan_grp = meas_grp.require_group(f"Channel_{_make_num_min_characters(i)}")
                # Extract name and retrace information from the channel name
                if h5_channel and "_" in str(h5_channel):
                    base_name, trace_dir = str(h5_channel).rsplit("_", 1)
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
                # TODO make this slightly faster by remembering we have load a channel already but
                # difficult cause of scaling
                channel_image, _, z_unit = self.get_image(
                    overide_channel=h5_channel, convert_to_nm=False, flip_image=False
                )
                chan_grp.attrs["net-encoder.scaling.unit.unit"] = z_unit.encode("utf-8")
                frame_stack = channel_image.flatten().reshape(-1, 1)

                # Update/ replace the channels dataset
                if dataset_name in chan_grp:
                    del chan_grp[dataset_name]
                chan_grp.create_dataset(dataset_name, data=frame_stack)

    def extract_dat_file(self, h5_datasets, h5_datasets_buffer, curve_num: int, direction: int, chan_name: str):
        """
        Extract the data from a .dat file in the JPK QI archive.

        Applies the appropriate scaling and saves it to the internal data structure and h5 dataset if required.

        Parameters
        ----------
        h5_datasets : dict
            A dictionary containing the h5 datasets for each channel and segment direction, used for saving the
            data.
        h5_datasets_buffer : dict
            A dictionary containing the buffer for each h5 dataset, used for temporary storage before writing to
            the dataset.
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
            indices_set = h5_datasets[f"Segment_{direction}"][chan_name]["Indices"]
            data_size = data_set.shape[0]
            buf = h5_datasets_buffer[f"Segment_{direction}"][chan_name]
            filled_size = self.points_for_channel_segment[direction][chan_name]
            start_offset = self.current_offsets[direction][chan_name]

            try:
                with self.qi_archive.open(dat_path) as f:
                    # Read binary data as big-endian 32-bit integers
                    raw_bytes = f.read()

                    raw_array = np.frombuffer(raw_bytes, dtype=">i4")

                    # Apply scaling to convert raw values into real world values
                    segment_array = (raw_array * scale["multiplier"]) + scale["offset"]

                    # Update the current offset so it include the length of the data we have just read
                    self.current_offsets[direction][chan_name] += len(segment_array)

                    buf["Data"].append(segment_array)
                    if len(buf["Data"]) >= self.BUFFER_SIZE or curve_num == self.num_of_curves - 1:
                        if self.points_for_channel_segment[direction][chan_name] > data_size:
                            # Fetch and resize the existing dataset for this channel and segment to fit the new data
                            data_set.resize((self.points_for_channel_segment[direction][chan_name],))

                        buffered_data = np.concatenate(buf["Data"])

                        # Add the buffer to the dataset
                        data_set[filled_size : filled_size + len(buffered_data)] = buffered_data
                        # Update the filled size for this channel and segment
                        self.points_for_channel_segment[direction][chan_name] += len(buffered_data)
                        # Clear the buffer
                        buf["Data"].clear()

            except KeyError:
                self.failed_curves.add((curve_num, direction, chan_name))

                # Limit the number of warnings to avoid spamming the logs
                if len(self.failed_curves) < 10:
                    logger.warning(
                        f"Data file {dat_path} not found in archive. Skipping data for curve {curve_num}, "
                        f"direction {direction}, channel {chan_name}."
                    )
                elif len(self.failed_curves) == 10:
                    logger.warning(
                        "Lots of missing files, further warnings will be suppressed. View summary at the end."
                    )

            # Append the new index to the indices buffer
            buf["Indices"].append(start_offset)

            # If the indices buffer is full add it to the indices dataset and clear the buffer
            if len(buf["Indices"]) > 0 and len(buf["Indices"]) % self.BUFFER_SIZE == 0:
                indices_set[curve_num - self.BUFFER_SIZE + 1 : curve_num + 1] = buf["Indices"]
                buf["Indices"].clear()

            # Or if this is the last curve and there are still indices in the buffer
            elif len(buf["Indices"]) > 0 and curve_num == self.num_of_curves - 1:
                # Add the remaining indices to the indices dataset and clear the buffer
                items_in_buffer = len(buf["Indices"])
                indices_set[curve_num - items_in_buffer + 1 : curve_num + 1] = buf["Indices"]
                buf["Indices"].clear()

        else:
            # Log if curve failed
            self.failed_curves.add((curve_num, direction, chan_name))
            if len(self.failed_curves) < 10:  # Limit the number of warnings to avoid spamming the logs
                logger.warning(
                    f"Channel {chan_name} not found in scaling information. Skipping data for curve {curve_num}, "
                    f"direction {direction}."
                )

    def extract_curve_metadata(self, curve_num: int, curve_work):
        """
        Extract the curve metadata from its header.properties file in the JPK QI archive and save to h5.

        Parameters
        ----------
        curve_num : int
            The curve number associated with the metadata.
        curve_work : list
            A list of tuples containing the search term for the metadata, the h5 dataset to save to,
            and the buffer for that dataset.
        """
        meta_path = f"index/{curve_num}/header.properties"
        raw_bytes = b""
        try:
            # Read metadata file as raw bytes
            with self.qi_archive.open(meta_path) as f:
                raw_bytes = f.read()
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
            if meta_buffer is not None:
                meta_buffer.append(value)
                if len(meta_buffer) >= self.BUFFER_SIZE or curve_num == self.num_of_curves - 1:
                    meta_set[curve_num - len(meta_buffer) + 1 : curve_num + 1] = meta_buffer
                    meta_buffer.clear()
            else:
                logger.error(
                    f"Metadata dataset for key {search_term.decode('utf-8')} not found when trying to save "
                    f"metadata for curve {curve_num}"
                )

    def extract_segment_metadata(self, curve_num: int, direction: int, seg_work):
        """
        Extract segment metadata from its header.properties file.

        Parameters
        ----------
        curve_num : int
            The curve number associated with the metadata.
        direction : int
            The segment direction (0 or 1) associated with the metadata.
        seg_work : list
            A list of tuples containing metadata extraction information.
        """
        meta_path = f"index/{curve_num}/segments/{direction}/segment-header.properties"
        raw_content = b""
        try:
            with self.qi_archive.open(meta_path) as f:
                raw_content = f.read()
        except KeyError:
            self.failed_curves.add((curve_num, direction, None))
            if len(self.failed_curves) < 10:  # Limit the number of warnings to avoid spamming the logs
                logger.warning(
                    f"Metadata file {meta_path} not found in archive. Skipping metadata for curve {curve_num}, "
                    f"direction {direction}."
                )
            elif len(self.failed_curves) == 10:
                logger.warning("Lots of missing files, further warnings will be suppressed. View summary at the end.")
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
                    f"Metadata dataset for key {search_term.decode('utf-8')} not found when trying to save "
                    f"metadata for curve {curve_num}, direction {direction}"
                )

    def setup_h5_structure(self, h5file):
        """
        Set up structure in the h5 file for saving curve data and metadata.

        Parameters
        ----------
        h5file : h5py.File
            The h5 file in which to set up the structure.

        Returns
        -------
        global_meta_group : h5py.Group
            The h5 group for storing global metadata.
        h5_datasets : dict
            A dictionary containing the h5 datasets for storing curve data.
        h5_meta_datasets : dict
            A dictionary containing the h5 datasets for storing metadata.
        h5_datasets_buffer : dict
            A dictionary containing buffers for the curve data datasets for temporary pre-writing storage.
        h5_meta_datasets_buffer : dict
            A dictionary containing buffers for the metadata datasets for temporary pre-writing storage.
        """
        # Create the main group for the QI curve data that all the curve data will be in
        qi_group = h5file.require_group("QI_Curve_Data")

        # Establish empty groups for global metadata and curve metadata
        global_meta_group = qi_group.require_group("Global_Metadata")
        curves_meta_group = qi_group.require_group("Curve_Metadata")
        curves_group = qi_group.require_group("Curves")

        curve_groups = {"Data": {}, "Indices": {}}
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
            # For each segment direction, establish necessary group structure that will contain each channel dataset
            seg_name = f"Segment_{direction}"
            dir_group = curves_group.require_group(seg_name)
            h5_datasets[seg_name] = {}
            h5_datasets_buffer[seg_name] = {}
            # Create the Data and Indices subfolders and store their references
            curve_groups["Data"][seg_name] = dir_group.require_group("Data")
            curve_groups["Indices"][seg_name] = dir_group.require_group("Indices")
            for chan in self.segment_channels:
                h5_datasets[seg_name][chan["name"]] = {}
                # For each channel, create an empty dataset
                h5_datasets[seg_name][chan["name"]]["Data"] = curve_groups["Data"][seg_name].create_dataset(
                    name=chan["name"],
                    shape=(self.points_for_channel_segment[direction][chan["name"]],),
                    maxshape=(None,),
                    chunks=(self.DATA_CHUNKSIZE,),
                    dtype=np.float32,
                )
                h5_datasets[seg_name][chan["name"]]["Indices"] = curve_groups["Indices"][seg_name].create_dataset(
                    name=chan["name"],
                    shape=(self.num_of_curves + 1,),
                    maxshape=(None,),
                    chunks=(self.INDICES_CHUNKSIZE,),
                    dtype=np.int32,
                )
                h5_datasets_buffer[seg_name][chan["name"]] = {"Data": [], "Indices": []}
        return global_meta_group, h5_datasets, h5_meta_datasets, h5_datasets_buffer, h5_meta_datasets_buffer

    def get_saving_context(self):
        """
        Return the appropriate context manager for saving the data based on the save_as_h5 attribute.

        If save_as_h5 is True, it returns a context manager for an h5 file. Otherwise, it returns a null context.

        Returns
        -------
        contextlib.AbstractContextManager
            The context manager for saving the data.
        """
        return h5py.File(self.h5_path, "a")

    def parse_dimension_data(self):
        """Parse dimension data and calculate the pixel to nanometer scaling factor."""
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
        """Extract global metadata and populate top level metadata dictionary and segment channels list."""
        # Load the metadata from the global properties file
        if "header.properties" in self.namelist:
            with self.qi_archive.open("header.properties") as archive_meta_file:
                props = javaproperties.load(archive_meta_file)

                # Add data from the main header to the top level metadata with a prefix to avoid key clashes
                for key, value in props.items():
                    self.top_level_meta[f"main-header.{key}"] = value
        else:
            logger.error(f"File {self.filepath} does not contain essential metadata and cannot be loaded")

        # Load the metadata from the shared header
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
        """Close the ZIP archive when done to free up system resources."""
        self.qi_archive.close()
        self.namelist = []


def _make_num_min_characters(num: int, min_chars: int = 3):
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
