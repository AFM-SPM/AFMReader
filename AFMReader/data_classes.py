"""
Data classes for lazy loading of curve data and metadata from files.

These classes provide a consistent interface for accessing curve data and metadata in a
lazy manner (i.e. loading data on demand rather than all at once) across different file
formats. This is necessary for handling large datasets with massive memory consumption.
"""

import numpy as np

# pylint: disable=too-few-public-methods,fixme


class CurvesMetadata:
    """
    A class representing the metadata for a dataset of curves, providing lazy loaded access to pixel metadata.

    This is a parent class that should be subclassed for specific file formats to implement
    the get_pixel_metadata method, which defines how the metadata is retrieved from the
    underlying data source.

    Parameters
    ----------
    toplevel : dict
        A dictionary containing the top-level metadata for the dataset.
    shape_x : int
        The number of columns in the image.
    shape_y : int
        The number of rows in the image.
    flip_image : bool, optional
        Whether to flip the image vertically. Default is True.
    """

    def __init__(self, toplevel: dict, shape_x: int, shape_y: int, flip_image: bool = True):
        """
        Initialise CurvesMetadata.

        Parameters
        ----------
        toplevel : dict
            A dictionary containing the top-level metadata for the dataset.
        shape_x : int
            The number of columns in the image.
        shape_y : int
            The number of rows in the image.
        flip_image : bool, optional
            Whether to flip the image vertically. Default is True.
        """
        self.toplevel = toplevel
        self.shape_x = shape_x
        self.shape_y = shape_y
        self.flip_image = flip_image

    def __getitem__(self, keys):
        """
        Fetch the metadata for a specific pixel or pixel direction.

        For example, metadata[y, x] would return the metadata for the pixel at row y and column x, while
        metadata[y, x, 0] would return the metadata for specifically the first direction of that pixel
        (usually the approach).

        Parameters
        ----------
        keys : tuple
            A tuple of (y, x) or (y, x, direction) representing the indices.

        Returns
        -------
        dict
            The metadata for the specified pixel or direction.
        """
        if isinstance(keys, tuple) and len(keys) == 2:
            y, x = keys
            return self.get_point_metadata(y, x)
        if isinstance(keys, tuple) and len(keys) == 3:
            y, x, direction = keys
            return self.get_point_metadata(y, x, direction)
        raise IndexError(
            f"Invalid indexing. Expected (y, x) or (y, x, direction) for point metadata indexing. Got {keys}."
        )

    # pylint: disable=unused-argument
    def get_point_metadata(self, y: int, x: int, direction: int | None = None):
        """
        Fetch the metadata for a specific pixel/ point, optionally for a specific direction.

        Should be implemented by subclasses if there exists per point metadata to define how the metadata is retrieved
        from the underlying data source. If there is no per point metadata, this can simply return an empty dict.

        Parameters
        ----------
        y : int
            Row index of the pixel.
        x : int
            Column index of the pixel.
        direction : int, optional
            The index of the direction to fetch metadata for. If None, returns metadata for the entire pixel.

        Returns
        -------
        dict
            The metadata for the specified pixel (or direction, if provided).
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
    shape_x : int
        The number of columns in the image.
    shape_y : int
        The number of rows in the image.
    channel_units : dict[str, str]
        A dictionary mapping channel names to their units.
    flip_image : bool, optional
        Whether to flip the image vertically. Default is True.
    """

    def __init__(self, name: str, shape_x: int, shape_y: int, channel_units: dict[str, str], flip_image: bool = True):
        """
        Initialise CurvesVolume.

        Parameters
        ----------
        name : str
            The name of the curve volume.
        shape_x : int
            The number of columns in the image.
        shape_y : int
            The number of rows in the image.
        channel_units : dict[str, str]
            A dictionary mapping channel names to their units.
        flip_image : bool, optional
            Whether to flip the image vertically. Default is True.
        """
        self.shape_x = shape_x
        self.shape_y = shape_y
        self.dims = (shape_y, shape_x)
        self.flip_image = flip_image
        self.name = name
        self.channel_units = channel_units

    def __len__(self):
        """
        Return the total number of pixels in the image.

        Returns
        -------
        int
            The total number of pixels in the image.
        """
        return self.shape_x * self.shape_y

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
            The QI curve data for the specified pixel.
        """
        if not isinstance(keys, tuple) or len(keys) != 2:
            raise IndexError(f"Invalid indexing. Expected (y, x) for pixel indexing. Got {keys}.")
        y, x = keys
        if y < 0 or y >= self.shape_y or x < 0 or x >= self.shape_x:
            raise IndexError(f"Pixel index ({y}, {x}) is out of bounds for image of shape {self.dims}.")
        return self.get_curve(y, x)

    def get_curve(self, y: int, x: int, flip_image: bool | None = None) -> dict:
        """
        Fetch the curve data for a specific pixel.

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
        for y in range(self.shape_y):
            for x in range(self.shape_x):
                yield self.get_curve(y, x, flip_image=flip_image)

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
    metadata : CurvesMetadata
        An instance of CurvesMetadata that provides access to the metadata
        for each curve.
    default_volume_name : str, optional
        The name of the default volume to use when accessing curve data.
        If None, the first volume in the dictionary is used.
    """

    def __init__(
        self, volumes: dict[str, CurvesVolume], metadata: CurvesMetadata, default_volume_name: str | None = None
    ):
        """
        Initialise CurvesDataset.

        Parameters
        ----------
        volumes : dict[str, CurvesVolume]
            A dictionary mapping curve names to CurvesVolume instances that
            provide access to the curve data for each pixel.
        metadata : CurvesMetadata
            An instance of CurvesMetadata that provides access to the metadata
            for each curve.
        default_volume_name : str | None, optional
            The name of the default volume to use when accessing curve data.
            If None, the first volume in the dictionary is used.
        """
        self.volumes = volumes
        self.metadata = metadata
        self.default_volume_name = default_volume_name or next(
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
