"""
Module to decode and load JPK QI (Quantitative Imaging) data files.

It provides lazy loading for curve data and metadata to minimize memory usage,
and supports exporting to HDF5 format.
"""

# pylint: disable=too-many-lines,too-many-positional-arguments,too-few-public-methods,too-many-instance-attributes
# pylint: disable=too-many-locals,too-many-branches,protected-access,attribute-defined-outside-init,fixme
# pylint: disable=too-many-arguments

import io
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import javaproperties
from tqdm import tqdm

from AFMReader.data_classes import AFMLoad, CurvesVolume, CurvesDataset, CurvesVolumeMetadata
from AFMReader.h5_saver import H5Saver, find_unused_filename
from AFMReader.io import coerce_metadata_dict, coerce_metadata_value, load_config
from AFMReader.logging import logger
from AFMReader import jpk


def _get_standard_segment_mapping(
    seg_names: list[str], segment_names_map: dict[str, list[str]] | None
) -> dict[str, str]:
    """
    Map source segment names to canonical segment names.

    Parameters
    ----------
    seg_names : list[str]
        Source segment names present in the curve data.
    segment_names_map : dict[str, list[str]] | None
        Dictionary mapping canonical segment names to their accepted aliases.

    Returns
    -------
    dict[str, str]
        Dictionary mapping source segment names to canonical segment names.
    """
    if segment_names_map is None:
        return {}
    new_segment_mapping = {}
    for key, names in segment_names_map.items():
        for name in names:
            if name in seg_names:
                new_segment_mapping[name] = key
                break
    return new_segment_mapping


def _get_standard_channel_mapping(
    channel_names: list[str], channel_names_map: dict[str, list[str]] | None
) -> dict[str, str]:
    """
    Map at most one source channel name to each canonical channel name.

    Parameters
    ----------
    channel_names : list[str]
        Source channel names present in the curve data.
    channel_names_map : dict[str, list[str]] | None
        Dictionary mapping canonical channel names to their accepted aliases.

    Returns
    -------
    dict[str, str]
        Dictionary mapping source channel names to canonical channel names.
    """
    if channel_names_map is None:
        return {}
    channel_mapping = {}
    for key, names in channel_names_map.items():
        for name in names:
            if name in channel_names:
                channel_mapping[name] = key
                break
    return channel_mapping


class CurvesJPKDataset(CurvesDataset):
    """
    A dataset class for JPK QI data that holds the raw data as well as metadata.

    Parameters
    ----------
    volumes : dict[str, CurvesVolume]
        A dictionary mapping curve names to CurvesVolume instances that
        provide access to the curve data for each pixel.
    metadata : dict
        Global metadata for the curve dataset.
    essential_metadata : dict
        Essential global metadata for the curve dataset.
    archive : zipfile.ZipFile
        The ZIP archive containing the JPK data.
    """

    def __init__(
        self, volumes: dict[str, CurvesVolume], metadata: dict, essential_metadata: dict, archive: zipfile.ZipFile
    ):
        """
        Initialise CurvesJPKDataset.

        Parameters
        ----------
        volumes : dict[str, CurvesVolume]
            A dictionary mapping curve names to CurvesVolume instances that
            provide access to the curve data for each pixel.
        metadata : dict
            A dictionary containing all metadata for the dataset.
        essential_metadata : dict
            A dictionary containing essential metadata for the dataset.
        archive : zipfile.ZipFile
            The ZIP archive containing the JPK data.
        """
        super().__init__(volumes, metadata, essential_metadata)
        self.archive = archive

    def close(self):
        """Close the ZIP archive when done to free up resources."""
        self.archive.close()
        self.archive = None


class CurvesJPKMetadata(CurvesVolumeMetadata):
    """
    A metadata class for JPK QI data that provides lazy loading of pixel metadata.

    Parameters
    ----------
    archive : zipfile.ZipFile
        The ZIP archive containing the JPK data.
    shape : tuple[int, int]
        The shape of the image as (rows, columns).
    channel_units : dict[str, str]
        A dictionary mapping channel names to their units.
    segment_names : list[str]
        The names of the curve segments available in this volume.
    flip_image : bool, optional
        Whether to flip the image vertically. Default is True.
    """

    def __init__(
        self,
        archive: zipfile.ZipFile,
        shape: tuple[int, int],
        channel_units: dict[str, str],
        segment_names: list[str],
        flip_image: bool = True,
    ):
        """
        Initialise the CurvesJPKMetadata instance.

        Parameters
        ----------
        archive : zipfile.ZipFile
            The ZIP archive containing the JPK data.
        shape : tuple[int, int]
            The shape of the image as (rows, columns).
        channel_units : dict[str, str]
            A dictionary mapping channel names to their units.
        segment_names : list[str]
            The names of the curve segments available in this volume.
        flip_image : bool, optional
            Whether to flip the image vertically. Default is True.
        """
        super().__init__(shape, channel_units, segment_names, flip_image)
        self.archive = archive

    def get_point_metadata(self, y: int, x: int, segment_name: str | None = None):
        """
        Fetch the metadata for a specific pixel or segment.

        Parameters
        ----------
        y : int
            Row index of the pixel.
        x : int
            Column index of the pixel.
        segment_name : str, optional
            The name of the segment to fetch metadata for. If None, returns metadata for the entire pixel.

        Returns
        -------
        dict
            The metadata for the specified pixel (or segment, if provided).
        """
        if y < 0 or y >= self.shape[0] or x < 0 or x >= self.shape[1]:
            raise IndexError(f"Curve index out of bounds: ({x}, {y})")
        if self.flip_image:
            y = self.shape[0] - 1 - y
        idx = (y * self.shape[1]) + x
        if segment_name is None:
            path = f"index/{idx}/header.properties"
        else:
            segment_number = self.segment_names.index(segment_name)
            path = f"index/{idx}/segments/{segment_number}/segment-header.properties"

        try:
            with self.archive.open(path) as f:
                meta_dict = coerce_metadata_dict(
                    {".".join(k.split(".")[1:]): v for k, v in javaproperties.load(f).items()}
                )
        except KeyError:
            meta_dict = {}

        return meta_dict


class CurvesJPKVolume(CurvesVolume):
    """
    A CurvesVolume implementation for JPK QI curve data that provides lazy loading of curve data for each pixel.

    Parameters
    ----------
    name : str
        The name of the curve volume.
    shape : tuple[int, int]
        The shape of the image as (rows, columns).
    archive : zipfile.ZipFile
        The ZIP archive containing the JPK data.
    metadata : CurvesJPKMetadata
        Metadata associated with this JPK curve volume.
    channel_scaling : dict[str, dict[str, float]]
        A dictionary mapping channel names to their scaling factors.
    segment_mapping : dict[str, str]
        A dictionary mapping source segment names to canonical segment names.
    channel_mapping : dict[str, str]
        A dictionary mapping source channel names to canonical channel names.
    flip_image : bool, optional
        Whether to flip the image vertically. Default is True.
    """

    def __init__(
        self,
        name: str,
        shape: tuple[int, int],
        archive: zipfile.ZipFile,
        metadata: CurvesJPKMetadata,
        channel_scaling: dict[str, dict[str, float]],
        segment_mapping: dict[str, str],
        channel_mapping: dict[str, str],
        flip_image: bool = True,
    ):
        """
        Initialise CurvesJPKVolume.

        Parameters
        ----------
        name : str
            The name of the curve volume.
        shape : tuple[int, int]
            The shape of the image as (rows, columns).
        archive : zipfile.ZipFile
            The ZIP archive containing the JPK data.
        metadata : CurvesJPKMetadata
            Metadata associated with this JPK curve volume.
        channel_scaling : dict[str, dict[str, float]]
            A dictionary mapping channel names to their scaling factors.
        segment_mapping : dict[str, str]
            A dictionary mapping source segment names to canonical segment names.
        channel_mapping : dict[str, str]
            A dictionary mapping source channel names to canonical channel names.
        flip_image : bool, optional
            Whether to flip the image vertically. Default is True.
        """
        super().__init__(
            name=name,
            shape=shape,
            metadata=metadata,
            flip_image=flip_image,
        )
        self.archive = archive
        self.channel_scaling = channel_scaling
        self.segment_mapping = segment_mapping
        self.channel_mapping = channel_mapping

    def get_curve(self, y: int, x: int, flip_image: bool | None = None):
        """
        Fetch the curve data for a specific pixel.

        Parameters
        ----------
        y : int
            Row index of the pixel.
        x : int
            Column index of the pixel.
        flip_image : bool, optional
            Whether to flip the image vertically. If None, uses the instance's flip_image attribute.

        Returns
        -------
        dict
            Dictionary containing the curve data for the specified pixel.
        """
        if y < 0 or y >= self.shape[0] or x < 0 or x >= self.shape[1]:
            raise IndexError(f"Curve index out of bounds: ({x}, {y})")
        if flip_image is None:
            flip_image = self.flip_image
        if flip_image:
            y = self.shape[0] - 1 - y
        curve_num = y * self.shape[1] + x
        curve_data: dict[str, Any] = {}

        for channel_name, scale in self.channel_scaling.items():
            standard_channel_name = self.channel_mapping.get(channel_name, channel_name)
            curve_data[standard_channel_name] = {}
            for segment_idx, segment_name in enumerate(self.metadata.segment_names):
                dat_path = f"index/{curve_num}/segments/{segment_idx}/channels/{channel_name}.dat"
                try:
                    # Access the file directly without re-parsing the ZIP directory
                    with self.archive.open(dat_path) as f:
                        raw_array = np.frombuffer(f.read(), dtype=">i4")
                        standard_segment_name = self.segment_mapping.get(segment_name, segment_name)
                        curve_data[standard_channel_name][standard_segment_name] = (
                            raw_array * scale["multiplier"]
                        ) + scale["offset"]
                except KeyError as e:
                    raise KeyError(
                        f"Internal data file missing for pixel ({x}, {y}), "
                        f"segment {segment_name}, channel {channel_name}"
                    ) from e

        return curve_data


def _get_channel_scaling(props: dict, channel_index: str) -> tuple[float, float, str]:
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


class JPKQILoader:
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
        Initialise the loader with the provided parameters.

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

        self.config = load_config(self.config_path)

        # Open the ZIP archive once and keep it open for the duration of the loading process
        self.qi_archive = zipfile.ZipFile(self.filepath, "r")  # pylint: disable=consider-using-with
        logger.info(f"Opened JPK QI archive at {self.filepath}")
        # Store the list of all paths in the archive to avoid having to call namelist() multiple times
        self.list_of_all_paths = self.qi_archive.namelist()
        # For holding the reference to where the actual .jqk-qi image is (not the metadata).
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

        # Initialise key attributes that will be returned / accessed frequently

        # Just the top level metadata extracted from the header files
        self.top_level_meta: dict[str, Any] = {}
        # A lazy reference containing all metadata
        self.volume_metadata: CurvesJPKMetadata | None = None
        # A 2D list of curve data dictionaries
        self.curves_volume: CurvesJPKVolume | None = None
        # A lookup for channel name to unit to be returned
        self.channels_units: dict[str, str] = {}
        # The list of channels for the segments with their scaling information extracted from the shared header
        self.segment_channels: list[dict[str, Any]] = []
        self.curve_meta: dict[str, Any] = {}
        self.segment_meta: dict[str, Any] = {}
        self.segment_names: list[str] = []
        self.segment_mapping: dict[str, str] = {}
        self.channel_mapping: dict[str, str] = {}

        # Define the image shape and size attributes
        self.size_x: float = np.nan
        self.size_y: float = np.nan
        self.shape_x: int = 0
        self.shape_y: int = 0
        self.failed_curves: set[tuple[int, int | None, str | None]] = set()

        # Instantiate containers for data to be saved (so an exception is not caused if not saving)
        self.curve_groups = None
        self.saved_to_h5 = False

        self.extract_global_metadata()

        self.parse_dimension_data()
        self.essential_metadata = self.filter_essential_metadata(self.top_level_meta)

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
            for file_name in self.list_of_all_paths:
                if file_name.endswith(".jpk-qi-image"):
                    self.path_to_image = file_name

        # Add the channels which exist in the jpk-qi-image file
        with self.qi_archive.open(self.path_to_image, "r") as image_file:
            return jpk._get_jpk_channels(
                file=image_file, filename=self.filepath.stem, file_path=self.filepath / Path(self.path_to_image)
            )

    def load(
        self,
        channel: str | None = None,
        config_path: Path | str | None = None,
        flip_image: bool | None = True,
        save_as_h5: bool | None = None,
    ) -> AFMLoad:
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
        AFMLoad
            An AFMLoad object containing the image, its pixel to nanometre
            scaling value, z-axis unit, and curves dataset.
        """
        # Update instance attributes based on provided parameters
        self.channel = channel if channel else self.channel
        self.config_path = config_path if config_path else self.config_path
        self.flip_image = flip_image if flip_image is not None else self.flip_image
        self.save_as_h5 = save_as_h5 if save_as_h5 is not None else self.save_as_h5

        # TODO add a save to h5 option here?

        logger.info(f"Loading JPK QI data from {self.filepath} with channel {self.channel}")

        # Setup H5 Data structures if needed
        if self.save_as_h5 and not self.saved_to_h5:
            self.save_to_h5()

        # Establish the lazy loading structures for curve data and metadata. Note how lazy structure is used even if
        # all the data has been accessed and saved to H5 to prevent excessive memory usage
        self.volume_metadata = CurvesJPKMetadata(
            self.qi_archive,
            shape=(self.shape_y, self.shape_x),
            channel_units=self.channels_units,
            segment_names=self.segment_names,
            flip_image=bool(self.flip_image),
        )
        self.curves_volume = CurvesJPKVolume(
            name="Trace",
            shape=(self.shape_y, self.shape_x),
            archive=self.qi_archive,
            metadata=self.volume_metadata,
            channel_scaling=self.channel_scaling,
            segment_mapping=self.segment_mapping,
            channel_mapping=self.channel_mapping,
            flip_image=bool(self.flip_image),
        )
        self.curves_dataset = CurvesJPKDataset(
            volumes={"Trace": self.curves_volume},
            metadata=self.top_level_meta,
            essential_metadata=self.essential_metadata,
            archive=self.qi_archive,
        )

        # Load the image
        self.image, _, self.z_unit = self.get_image()

        return AFMLoad(image=self.image, px2nm=self.px2nm, z_units=self.z_unit, curves_dataset=self.curves_dataset)

    def output_summary(self):
        """Output a summary of the loading process, including any failed curve loads and their details."""
        if self.failed_curves:
            logger.warning(f"Failed to load {len(self.failed_curves)} files.")
            logger.warning("Summary of missing files (up to 10 shown):")

            # Output the first 10 failed loads with details
            for i, (curve_num, segment, chan_name) in enumerate(self.failed_curves):
                if i < 10:
                    if chan_name:
                        logger.warning(
                            f"Failed to load data for curve {curve_num}, segment {segment}, channel {chan_name}"
                        )
                    else:
                        if segment is not None:
                            logger.warning(f"Failed to load segment meta file for curve {curve_num}, segment {segment}")
                        else:
                            logger.warning(f"Failed to load curve meta file for curve {curve_num}")
                else:
                    break
        else:
            # If there are no failed loads, log that all data was loaded successfully
            logger.info("Successfully loaded all curve data without any missing files.")

    def extract_data_to_h5(self, h5_saver: H5Saver, include_metadata: bool = True):
        """
        Load all curve data and optionally metadata from the JPK QI archive into HDF5 datasets.

        Parameters
        ----------
        h5_saver : H5Saver
            H5Saver instance for managing HDF5 datasets and buffers.
        include_metadata : bool, optional
            Whether to include metadata in the loading process, by default True.
        """
        logger.info(
            f"Loading all curve data from JPK QI archive with {len(self.list_of_all_paths)} files "
            f"{'' if include_metadata else 'not '}including metadata"
        )
        curve_search_terms = h5_saver.get_curve_search_terms("Trace")
        segment_search_terms = h5_saver.get_segment_search_terms("Trace")
        num_of_segments = len(self.segment_names)
        for curve_num in tqdm(range(self.num_of_curves)):
            for segment_idx, segment_name in enumerate(self.segment_names):
                for chan in self.segment_channels:
                    # Save the actual curve data to the h5 datasets
                    self.extract_dat_file(
                        h5_saver=h5_saver,
                        volume_name="Trace",
                        curve_num=curve_num,
                        segment_idx=segment_idx,
                        segment_name=segment_name,
                        chan_name=chan["name"],
                    )

                if include_metadata:
                    # Extract and store the segment metadata for later saving
                    self.extract_segment_metadata(
                        h5_saver=h5_saver,
                        curve_num=curve_num,
                        segment_idx=segment_idx,
                        search_terms=segment_search_terms,
                        volume_name="Trace",
                        num_of_segments=num_of_segments,
                    )

            if include_metadata:
                # Extract and store the curve metadata for later saving
                self.extract_curve_metadata(
                    h5_saver=h5_saver, curve_num=curve_num, search_terms=curve_search_terms, volume_name="Trace"
                )

    def save_to_h5(
        self,
        include_per_curve_metadata: bool = True,
    ) -> Path:
        """
        Save data as an H5 file. If include_per_curve_metadata is False, only curve data is saved.

        Parameters
        ----------
        include_per_curve_metadata : bool, optional
            If True, per-curve metadata will be included in the saved H5 file. Default is True.

        Returns
        -------
        Path
            The path to the saved H5 file.
        """
        self.volume_metadata = CurvesJPKMetadata(
            archive=self.qi_archive,
            shape=(self.shape_y, self.shape_x),
            channel_units=self.channels_units,
            segment_names=self.segment_names,
            flip_image=bool(self.flip_image),
        )
        self.curves_volume = CurvesJPKVolume(
            name="Trace",
            shape=(self.shape_y, self.shape_x),
            archive=self.qi_archive,
            metadata=self.volume_metadata,
            channel_scaling=self.channel_scaling,
            segment_mapping=self.segment_mapping,
            channel_mapping=self.channel_mapping,
            flip_image=bool(self.flip_image),
        )
        # Determine the path for the H5 file, ensuring it does not overwrite an existing file
        self.h5_path = find_unused_filename(self.filepath)

        h5_saver = H5Saver(self.h5_path)
        with h5_saver.create_file(source=self.filepath.suffix) as file:

            # Sample curves in dataset to make a best guess for the meta keys
            self.changing_curve_keys, self.changing_segment_keys = self.get_changing_keys(h5_saver)
            h5_saver.setup_curves_group()

            h5_saver.setup_volume(
                curves_volume=self.curves_volume,
                changing_curve_keys=self.changing_curve_keys,
                changing_segment_keys=self.changing_segment_keys,
            )

            # Extract data from the JPK QI archive and save to H5 datasets
            self.extract_data_to_h5(
                h5_saver,
                include_metadata=include_per_curve_metadata,
            )
            # Resize the datasets to the actual number of points read
            h5_saver.complete_saving(self.curves_volume)

            # Save the global metadata to the h5 file
            h5_saver.save_global_meta(
                self.get_collated_metadata(), self.size_x, self.size_y, self.shape_x, self.shape_y
            )

            logger.info(f"QI data copied to h5 data {file.filename}")

            # Save a lite form of the images (precalculated) if saving to a file
            self.save_lite_data(h5_saver)

            self.output_summary()
            self.saved_to_h5 = True
            return self.h5_path

    def get_changing_keys(self, h5_saver: H5Saver):  # noqa: C901
        """
        Check a sample of curves to see which metadata keys change across curves and segments.

        This allows us to extract only the changing keys for each curve and segment.
        Non-changing keys are moved to the top-level metadata and not extracted for each curve/segment.

        Parameters
        ----------
        h5_saver : H5Saver
            The H5Saver instance to use for sampling curves.

        Returns
        -------
        tuple:
            A tuple containing two sets: changing_curve_keys and changing_segment_keys.
        """
        curve_meta_dict: dict[str, list[Any]] = {}
        segment_meta_dict: dict[str, list[Any]] = {}
        curves_to_check = h5_saver.get_curves_sample(self.shape_x, self.shape_y, self.MAX_CURVE_CHECKS)
        for curve_num in curves_to_check:
            for segment_idx in range(len(self.segment_names)):
                while True:
                    meta_path = f"index/{curve_num}/segments/{segment_idx}/segment-header.properties"
                    try:
                        with self.qi_archive.open(meta_path) as f:
                            meta_dict = coerce_metadata_dict(
                                {".".join(k.split(".")[1:]): v for k, v in javaproperties.load(f).items()}
                            )
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
                        meta_dict = coerce_metadata_dict(
                            {".".join(k.split(".")[1:]): v for k, v in javaproperties.load(f).items()}
                        )
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
        for key, value in self.essential_metadata.items():
            collated_meta[f"essential.{key}"] = value
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
        for file_name in self.list_of_all_paths:
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

    def save_lite_data(self, h5_saver: H5Saver):
        """
        Save a lite form of the data (e.g., the calculated image data) to H5.

        Parameters
        ----------
        h5_saver : H5Saver
            An instance of the H5Saver class used for saving data to the H5 file.
        """
        logger.info(f"Saving a hdf5 copy of the data {self.h5_path}")

        # Look for the jpk-qi-image file in the archive
        path_to_image = None
        for file_name in self.list_of_all_paths:
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
            channel_image, _, z_unit = self.get_image(overide_channel=h5_channel, convert_to_nm=False, flip_image=False)
            h5_saver.save_image(channel_image, image_name=h5_channel, z_unit=z_unit, idx=i)

    def extract_dat_file(
        self, h5_saver: H5Saver, volume_name: str, curve_num: int, segment_idx: int, segment_name: str, chan_name: str
    ):
        """
        Extract the data from a .dat file in the JPK QI archive.

        Applies the appropriate scaling and saves it to the internal data structure and h5 dataset if required.

        Parameters
        ----------
        h5_saver : H5Saver
            An instance of the H5Saver class used for saving data to the H5 file.
        volume_name : str
            The name of the volume to which the curve belongs.
        curve_num : int
            The curve number associated with the .dat file, parsed from the filename.
        segment_idx : int
            The segment index associated with the .dat file, parsed from the filename.
        segment_name : str
            The segment name associated with the .dat file, parsed from the filename.
        chan_name : str
            The channel name associated with the .dat file, parsed from the filename.
        """
        if chan_name in self.channel_scaling:
            # Get data structures for this channel and segment
            scale = self.channel_scaling[chan_name]
            dat_path = f"index/{curve_num}/segments/{segment_idx}/channels/{chan_name}.dat"

            try:
                with self.qi_archive.open(dat_path) as f:
                    # Read binary data as big-endian 32-bit integers
                    raw_bytes = f.read()

                    raw_array = np.frombuffer(raw_bytes, dtype=">i4")

                    # Apply scaling to convert raw values into real world values
                    segment_array = (raw_array * scale["multiplier"]) + scale["offset"]

            except KeyError:
                self.failed_curves.add((curve_num, segment_idx, chan_name))

                # Limit the number of warnings to avoid spamming the logs
                if len(self.failed_curves) < 10:
                    logger.warning(
                        f"Data file {dat_path} not found in archive. Skipping data for curve {curve_num}, "
                        f"segment {segment_name}, channel {chan_name}."
                    )
                elif len(self.failed_curves) == 10:
                    logger.warning(
                        "Lots of missing files, further warnings will be suppressed. View summary at the end."
                    )
                segment_array = np.empty(0, dtype=np.float32)
            h5_saver.save_curve_segment(
                volume_name=volume_name,
                segment_data=segment_array,
                curve_num=curve_num,
                segment_name=segment_name,
                channel_name=self.channel_mapping.get(chan_name, chan_name),
                num_of_curves=self.num_of_curves,
            )

        else:
            # Log if curve failed
            self.failed_curves.add((curve_num, segment_idx, chan_name))
            if len(self.failed_curves) < 10:  # Limit the number of warnings to avoid spamming the logs
                logger.warning(
                    f"Channel {chan_name} not found in scaling information. Skipping data for curve {curve_num}, "
                    f"segment {segment_name}."
                )

    def extract_curve_metadata(self, h5_saver: H5Saver, curve_num: int, search_terms: list[bytes], volume_name: str):
        """
        Extract the curve metadata from its header.properties file in the JPK QI archive and save to h5.

        Parameters
        ----------
        h5_saver : H5Saver
            The H5Saver instance to save the metadata to.
        curve_num : int
            The curve number associated with the metadata.
        search_terms : list[bytes]
            A list of search terms to look for in the metadata file.
        volume_name : str
            The name of the volume.
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

        for attr_idx, search_term in enumerate(search_terms):
            # Find the location of the metadata value in the raw bytes
            start = raw_bytes.find(search_term)
            # If found, extract the actual value
            if start != -1:
                start += len(search_term)
                end = raw_bytes.find(b"\n", start)
                value = coerce_metadata_value(raw_bytes[start:end].decode("utf-8").strip())
            # Save a no data value if the search term is not found in the metadata file
            else:
                value = "No data"
            h5_saver.save_curve_meta_attr(
                curve_num=curve_num,
                attr_idx=attr_idx,
                value=value,
                volume_name=volume_name,
                num_of_curves=self.num_of_curves,
            )

    def extract_segment_metadata(
        self,
        h5_saver: H5Saver,
        curve_num: int,
        segment_idx: int,
        search_terms: list[bytes],
        volume_name: str,
        num_of_segments: int,
    ):
        """
        Extract segment metadata from its header.properties file.

        Parameters
        ----------
        h5_saver : H5Saver
            The H5Saver instance to save the metadata to.
        curve_num : int
            The curve number associated with the metadata.
        segment_idx : int
            The segment index associated with the metadata.
        search_terms : list[bytes]
            A list of search terms to look for in the metadata file.
        volume_name : str
            The name of the volume.
        num_of_segments : int
            The number of segments in each curve.
        """
        meta_path = f"index/{curve_num}/segments/{segment_idx}/segment-header.properties"
        raw_content = b""
        try:
            with self.qi_archive.open(meta_path) as f:
                raw_content = f.read()
        except KeyError:
            self.failed_curves.add((curve_num, segment_idx, None))
            if len(self.failed_curves) < 10:  # Limit the number of warnings to avoid spamming the logs
                logger.warning(
                    f"Metadata file {meta_path} not found in archive. Skipping metadata for curve {curve_num}, "
                    f"segment {self.segment_names[segment_idx]}."
                )
            elif len(self.failed_curves) == 10:
                logger.warning("Lots of missing files, further warnings will be suppressed. View summary at the end.")
        for attr_idx, search_term in enumerate(search_terms):
            start = raw_content.find(search_term)
            if start != -1:
                start += len(search_term)
                end = raw_content.find(b"\n", start)
                value = coerce_metadata_value(raw_content[start:end].decode("utf-8").strip())
            else:
                value = "No data"
            h5_saver.save_segment_meta_attr(
                curve_num=curve_num,
                segment_idx=segment_idx,
                attr_idx=attr_idx,
                value=value,
                volume_name=volume_name,
                num_of_curves=self.num_of_curves,
                num_of_segments=num_of_segments,
            )

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
        if np.isnan(self.size_x) or np.isnan(self.size_y) or 0 in [self.shape_x, self.shape_y]:
            logger.error(f"Incomplete dimension data in {self.filepath}")

        # Calculate the pixel to nano metre scaling as an average of the scale for each axis
        pixel_to_nm_scaling_factor_x = self.size_x / self.shape_x * 1e9 if self.shape_x > 0 else 1.0
        pixel_to_nm_scaling_factor_y = self.size_y / self.shape_y * 1e9 if self.shape_y > 0 else 1.0
        self.px2nm = (pixel_to_nm_scaling_factor_x + pixel_to_nm_scaling_factor_y) / 2

        # Establish number of curves
        self.num_of_curves = self.shape_x * self.shape_y

    def extract_global_metadata(self):
        """Extract global metadata and populate top level metadata dictionary and segment channels list."""
        # Load the metadata from the global properties file
        if "header.properties" in self.list_of_all_paths:
            with self.qi_archive.open("header.properties") as archive_meta_file:
                props = coerce_metadata_dict(javaproperties.load(archive_meta_file))

                # Add data from the main header to the top level metadata with a prefix to avoid key clashes
                for key, value in props.items():
                    self.top_level_meta[f"main-header.{key}"] = value
        else:
            logger.error(f"File {self.filepath} does not contain essential metadata and cannot be loaded")

        # Load the metadata from the shared header
        if "shared-data/header.properties" in self.list_of_all_paths:
            with self.qi_archive.open("shared-data/header.properties") as shared_data_file:
                shared_meta = coerce_metadata_dict(javaproperties.load(shared_data_file))
                channel_idx = 0
                segment_count = int(shared_meta.get("force-segment-header-infos.count", 0))
                self.segment_names = [
                    shared_meta.get(f"force-segment-header-info.{seg_idx}.settings.style", f"Segment {seg_idx}")
                    for seg_idx in range(segment_count)
                ]
                self.segment_mapping = _get_standard_segment_mapping(
                    self.segment_names,
                    self.config.get("standard_curve", {}).get("segment_names", {}),
                )
                self.segment_names = [
                    self.segment_mapping.get(segment_name, segment_name) for segment_name in self.segment_names
                ]

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
        self.channel_mapping = _get_standard_channel_mapping(
            list(self.channel_scaling),
            self.config.get("standard_curve", {}).get("channel_names", {}),
        )
        self.channels_units = {
            self.channel_mapping.get(channel_name, channel_name): unit
            for channel_name, unit in self.channels_units.items()
        }

    def filter_essential_metadata(self, raw_metadata: dict[str, Any]) -> dict[str, Any]:
        """
        Extract canonical essential metadata from raw global metadata.

        Parameters
        ----------
        raw_metadata : dict
            The raw metadata dictionary to filter.

        Returns
        -------
        dict[str, Any]
            A dictionary containing only the essential metadata keys.
        """
        essential_key_options = self.config.get("jpk_qi", {}).get("essential_metadata_keys", {})
        filtered_metadata = {}
        for target_name, source_keys in essential_key_options.items():
            for source_key in source_keys:
                if source_key in raw_metadata:
                    filtered_metadata[target_name] = raw_metadata[source_key]
                    break
        if "read_sample_rate" not in filtered_metadata:
            if "extend_num_points" in filtered_metadata and "extend_duration" in filtered_metadata:
                filtered_metadata["read_sample_rate"] = (
                    filtered_metadata["extend_num_points"] / filtered_metadata["extend_duration"]
                )
        return filtered_metadata

    def close(self):
        """Close the ZIP archive when done to free up system resources."""
        self.qi_archive.close()
        self.list_of_all_paths = []


def load_jpk_data(filepath: str | Path, channel: str, cached_data: dict) -> AFMLoad:
    """
    Load the JPK QI data using the JPKQILoader.

    Parameters
    ----------
    filepath : str | Path
        Path to the JPK QI file.
    channel : str
        The channel to load from the file.
    cached_data : dict
        Cached data to avoid reloading heavy data.

    Returns
    -------
    AFMLoad
        The loaded JPK QI data.
    """
    if "jpk_qi_loader" not in cached_data:
        cached_data["jpk_qi_loader"] = JPKQILoader(filepath=filepath, channel=channel)
    return cached_data["jpk_qi_loader"].load(channel=channel)


def get_jpk_data_channels(filepath: str | Path, cached_data: dict) -> list[str]:
    """
    Get the available channels in the JPK QI data.

    Parameters
    ----------
    filepath : str | Path
        Path to the JPK QI file.
    cached_data : dict
        Cached data to avoid reloading heavy data.

    Returns
    -------
    list[str]
        A list of available channels in the JPK QI data.
    """
    if "jpk_qi_loader" not in cached_data:
        cached_data["jpk_qi_loader"] = JPKQILoader(filepath=filepath)
    return cached_data["jpk_qi_loader"].get_available_channels()


def get_jpk_data_params() -> dict:
    """
    Get any additional parameters for the JPK QI data.

    Returns
    -------
    dict
        A dictionary containing any additional parameters for the JPK QI data.
    """
    return {"save_as_h5": bool}


def save_jpk_data_to_h5(filepath: str | Path, cached_data: dict | None = None) -> Path:
    """
    Save the JPK QI data as an h5 file for faster future loading.

    Parameters
    ----------
    filepath : str | Path
        Path to the JPK QI file.
    cached_data : dict
        Cached data to avoid reloading heavy data.

    Returns
    -------
    Path
        The path to the saved h5 file.
    """
    if cached_data is None:
        cached_data = {}
    if "jpk_qi_loader" not in cached_data:
        cached_data["jpk_qi_loader"] = JPKQILoader(filepath=filepath)
    h5_path = cached_data["jpk_qi_loader"].save_to_h5()
    cached_data["jpk_qi_loader"].close()
    cached_data.pop("jpk_qi_loader")
    return h5_path
