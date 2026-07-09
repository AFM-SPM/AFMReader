"""
Data classes for lazy loading of curve data and metadata from files.

These classes provide a consistent interface for accessing curve data and metadata in a
lazy manner (i.e. loading data on demand rather than all at once) across different file
formats. This is necessary for handling large datasets with massive memory consumption.
"""

import numpy as np

# pylint: disable=too-few-public-methods,fixme


class CurvesVolumeMetadata:
    """
    A class representing metadata for a curve volume, providing lazy loaded access to point metadata.

    This is a parent class that should be subclassed for specific file formats to implement
    the get_point_metadata method, which defines how the metadata is retrieved from the
    underlying data source.

    Parameters
    ----------
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
        shape: tuple[int, int],
        channel_units: dict[str, str],
        segment_names: list[str],
        flip_image: bool = True,
    ):
        """
        Initialise CurvesVolumeMetadata.

        Parameters
        ----------
        shape : tuple[int, int]
            The shape of the image as (rows, columns).
        channel_units : dict[str, str]
            A dictionary mapping channel names to their units.
        segment_names : list[str]
            The names of the curve segments available in this volume.
        flip_image : bool, optional
            Whether to flip the image vertically. Default is True.
        """
        self.shape = shape
        self.channel_units = channel_units
        self.segment_names = segment_names
        self.flip_image = flip_image

    def __getitem__(self, keys):
        """
        Fetch the metadata for a specific pixel or segment.

        For example, metadata[y, x] would return the metadata for the pixel at row y and column x, while
        metadata[y, x, segment_name] would return the metadata for specifically the segment of that pixel
        if segment metadata is available.

        Parameters
        ----------
        keys : tuple
            A tuple of (y, x) or (y, x, segment_name) representing the indices.

        Returns
        -------
        dict
            The metadata for the specified pixel or segment.
        """
        if isinstance(keys, tuple) and len(keys) == 2:
            y, x = keys
            return self.get_point_metadata(y, x)
        if isinstance(keys, tuple) and len(keys) == 3:
            y, x, segment_name = keys
            return self.get_point_metadata(y, x, segment_name)
        raise IndexError(
            f"Invalid indexing. Expected (y, x) or (y, x, segment_name) for point metadata indexing. Got {keys}."
        )

    # pylint: disable=unused-argument
    def get_point_metadata(self, y: int, x: int, segment_name: str | None = None):
        """
        Fetch the metadata for a specific pixel/point, optionally for a specific segment.

        Should be implemented by subclasses if there exists per point metadata to define how the metadata is retrieved
        from the underlying data source. If there is no per point metadata, this can simply return an empty dict.

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
            The metadata for the specified pixel or segment.
        """
        return {}


class CurvesVolume:
    """
    A class representing a 2D map or volume of curves, providing lazy loaded access to curve data.

    An individual curve can be accessed using volume[y, x], which will load the curve data for that pixel on demand.

    Parameters
    ----------
    name : str
        The name of the curve volume.
    shape : tuple[int, int]
        The shape of the image as (rows, columns).
    metadata : CurvesVolumeMetadata
        Metadata associated with this curve volume.
    flip_image : bool, optional
        Whether to flip the image vertically. Default is True.
    """

    def __init__(
        self,
        name: str,
        shape: tuple[int, int],
        metadata: CurvesVolumeMetadata,
        flip_image: bool = True,
    ):
        """
        Initialise CurvesVolume.

        Parameters
        ----------
        name : str
            The name of the curve volume.
        shape : tuple[int, int]
            The shape of the image as (rows, columns).
        metadata : CurvesVolumeMetadata
            Metadata associated with this curve volume.
        flip_image : bool, optional
            Whether to flip the image vertically. Default is True.
        """
        self.name = name
        self.shape = shape
        self.metadata = metadata
        self.flip_image = flip_image

        # Store analysis results in a dict, with the values being numpy arrays of the results for each pixel.
        self.analysis_results: dict[str, np.ndarray] = {}

    def __len__(self):
        """
        Return the total number of pixels in the image.

        Returns
        -------
        int
            The total number of pixels in the image.
        """
        return self.shape[0] * self.shape[1]

    def __getitem__(self, keys):
        """
        Allow numpy style indexing to fetch curve data for a specific pixel.

        Parameters
        ----------
        keys : tuple
            A tuple of (y, x) representing the row and column indices of the pixel.

        Returns
        -------
        dict
            The curve data for the specified pixel.
        """
        if not isinstance(keys, tuple) or len(keys) != 2:
            raise IndexError(f"Invalid indexing. Expected (y, x) for pixel indexing. Got {keys}.")
        y, x = keys
        return self.get_curve(y, x)

    def get_curve(self, y: int, x: int, flip_image: bool | None = None) -> dict:
        """
        Purely overridable method to fetch the curve data for a specific pixel.

        Should be implemented by subclasses to define how the curve data is retrieved from the underlying data source.

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
            The curve data for the specified pixel.
        """
        raise NotImplementedError("This method should be implemented by subclasses to fetch curve data on demand.")

    def iter_curves(self, flip_image: bool | None = None):
        """
        Iterate over all pixels in the image, yielding the curve data for each pixel.

        Parameters
        ----------
        flip_image : bool, optional
            Whether to flip the image vertically during iteration. If None, uses the instance's flip_image attribute.

        Yields
        ------
        dict
            The QI curve data for each pixel in row-major order (y first, then x).
        """
        if flip_image is None:
            flip_image = self.flip_image
        for y in range(self.shape[0]):
            for x in range(self.shape[1]):
                yield self.get_curve(y, x, flip_image=flip_image)

    def get_analysis_results(self, y: int, x: int, flip_image: bool | None = None) -> dict:
        """
        Fetch the analysis results for a specific pixel.

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
            The analysis results for the specified pixel.
        """
        if flip_image is None:
            flip_image = self.flip_image
        if flip_image:
            y = self.shape[0] - 1 - y  # Flip the y index if needed
        return {key: value[y, x] for key, value in self.analysis_results.items() if value is not None}

    def __iter__(self):
        """
        Iterate over all pixels in the image.

        Returns
        -------
        Iterator
            An iterator over the curve data for each pixel.
        """
        return self.iter_curves()


class CurvesDataset:
    """
    A dataset containing multiple curve volumes and associated metadata.

    Parameters
    ----------
    volumes : dict[str, CurvesVolume]
        A dictionary mapping curve names to CurvesVolume instances that
        provide access to the curve data for each pixel.
    metadata : dict
        Global metadata for the curve dataset.
    essential_metadata : dict
        Essential global metadata for the curve dataset.
    default_volume_name : str, optional
        The name of the default volume to use when accessing curve data.
        If None, the first volume in the dictionary is used.
    """

    def __init__(
        self,
        volumes: dict[str, CurvesVolume],
        metadata: dict,
        essential_metadata: dict,
        default_volume_name: str | None = None,
    ):
        """
        Initialise CurvesDataset.

        Parameters
        ----------
        volumes : dict[str, CurvesVolume]
            A dictionary mapping curve names to CurvesVolume instances that
            provide access to the curve data for each pixel.
        metadata : dict
            A dictionary containing metadata for each curve.
        essential_metadata : dict
            A dictionary containing essential metadata for each curve.
        default_volume_name : str | None, optional
            The name of the default volume to use when accessing curve data.
            If None, the first volume in the dictionary is used.
        """
        self.volumes: dict[str, CurvesVolume] = volumes
        self.metadata: dict = metadata
        self.essential_metadata: dict = essential_metadata
        self.default_volume_name: str = default_volume_name or next(
            iter(volumes)
        )  # Use the first volume as default if not specified

    def add_volume(self, name: str, volume: CurvesVolume, default: bool = False):
        """
        Add a CurvesVolume to the dataset.

        Parameters
        ----------
        name : str
            The name of the curve to add.
        volume : CurvesVolume
            The CurvesVolume instance containing the curve data for each pixel.
        default : bool, optional
            Whether to set this volume as the default volume. Default is False.
        """
        self.volumes[name] = volume
        if default:
            self.default_volume_name = name

    def get_default_volume(self) -> CurvesVolume:
        """
        Get the default CurvesVolume for this dataset.

        Returns
        -------
        CurvesVolume
            The default CurvesVolume instance for this dataset.
        """
        return self.volumes[self.default_volume_name]

    def get_volume(self, name: str) -> CurvesVolume | None:
        """
        Get a specific CurvesVolume by name.

        Parameters
        ----------
        name : str
            The name of the curve to retrieve.

        Returns
        -------
        CurvesVolume | None
            The CurvesVolume instance for the specified curve name, or None if not found.
        """
        return self.volumes.get(name)

    def get_volume_names(self) -> list[str]:
        """
        Get a list of all volume names in the dataset.

        Returns
        -------
        list[str]
            A list of all volume names in the dataset.
        """
        return list(self.volumes.keys())

    def close(self):
        """
        Close the dataset and release any resources.

        This method should be called when the dataset is no longer needed to free up memory and release files.
        """
        raise NotImplementedError(
            "This method should be implemented by subclasses to close the dataset and release resources."
        )


class AFMLoad:
    """
    A class representing the loaded AFM data, including the image and scaling factors.

    Parameters
    ----------
    image : np.ndarray
        The image data.
    px2nm : float
        The pixel to nanometer scaling factor.
    z_units : str
        The units of the z-axis (e.g. 'm', 'nm').
    timestamps : dict | None, optional
        Timestamps associated with the data. Default is None.
    metadata : dict | None, optional
        Metadata associated with the data. Default is None.
    curves_dataset : CurvesDataset | None, optional
        Curves dataset associated with the data. Default is None.
    """

    image: np.ndarray
    px2nm: float
    z_units: str
    timestamps: dict | None = None
    metadata: dict | None = None
    curves_dataset: CurvesDataset | None = None

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        image: np.ndarray,
        px2nm: float,
        z_units: str,
        timestamps: dict | None = None,
        metadata: dict | None = None,
        curves_dataset: CurvesDataset | None = None,
    ):
        """
        Initialise AFMLoad.

        Parameters
        ----------
        image : np.ndarray
            The image data.
        px2nm : float
            The pixel to nanometer scaling factor.
        z_units : str
            The units of the z-axis (e.g. 'm', 'nm').
        timestamps : dict | None, optional
            Timestamps associated with the data. Default is None.
        metadata : dict | None, optional
            Metadata associated with the data. Default is None.
        curves_dataset : CurvesDataset | None, optional
            Curves dataset associated with the data. Default is None.
        """
        self.image = image
        self.px2nm = px2nm
        self.z_units = z_units
        self.timestamps = timestamps
        self.metadata = metadata if metadata is not None else {}
        self.curves_dataset = curves_dataset
