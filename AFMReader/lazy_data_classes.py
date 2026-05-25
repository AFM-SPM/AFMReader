"""Utility classes for lazy loading of QI curve data and metadata from JPK files."""

# pylint: disable=too-few-public-methods,fixme


class LazyQiData:
    """
    A proxy class that fetches QI curve data on demand.

    It behaves like a 2D array of shape (shape_y, shape_x) where each element is a dictionary
    containing the QI curve data for that pixel.

    Parameters
    ----------
    shape_x : int
        The number of columns in the image.
    shape_y : int
        The number of rows in the image.
    flip_image : bool, optional
        Whether to flip the image vertically. Default is True.
    """

    def __init__(self, shape_x: int, shape_y: int, flip_image: bool = True):
        """
        Initialize the LazyQiData instance.

        Parameters
        ----------
        shape_x : int
            The number of columns in the image.
        shape_y : int
            The number of rows in the image.
        flip_image : bool, optional
            Whether to flip the image vertically. Default is True.
        """
        self.shape_x = shape_x
        self.shape_y = shape_y
        self.dims = (shape_y, shape_x)
        self.flip_image = flip_image

    def __len__(self):
        """
        Return the total number of pixels in the image.

        Returns
        -------
        int
            The total number of pixels in the image.
        """
        return self.shape_x * self.shape_y

    def __getitem__(self, y: int):
        """
        Return a proxy object for the specified row that can be indexed to fetch curve data for each pixel in that row.

        This allows for lazy loading.

        Parameters
        ----------
        y : int
            The row index.

        Returns
        -------
        RowProxy
            A proxy object for the specified row.
        """

        class RowProxy:
            """
            A proxy class for a single row of the QI data that fetches curve data on demand.

            Parameters
            ----------
            parent : LazyQiData
                The parent LazyQiData instance.
            y : int
                The row index.
            """

            def __init__(self, parent, y):
                """
                Initialize RowProxy with parent LazyQiData and row index.

                Parameters
                ----------
                parent : LazyQiData
                    The parent LazyQiData instance.
                y : int
                    The row index.
                """
                self.parent = parent
                self.y = y

            def __getitem__(self, x: int):
                """
                Fetch curve data for column x in this row.

                Parameters
                ----------
                x : int
                    The column index.

                Returns
                -------
                dict
                    The QI curve data for the specified pixel.
                """
                return self.parent._fetch_curve(self.y, x)

        return RowProxy(self, y)

    def _fetch_curve(self, y: int, x: int):
        """
        Fetch the QI curve data for a specific pixel.

        Should be implemented by subclasses to define how the curve data is retrieved from the underlying data source.

        Parameters
        ----------
        y : int
            Row index of the pixel.
        x : int
            Column index of the pixel.

        Returns
        -------
        dict
            The QI curve data for the specified pixel.
        """
        raise NotImplementedError("This method should be implemented by subclasses to fetch curve data on demand.")


class LazyMetadata:
    """
    A proxy class that fetches metadata on demand. Superclass for metadata proxy classes.

    Parameters
    ----------
    top_level_meta : dict
        The top-level metadata dictionary.
    shape_x : int
        The number of columns in the image.
    shape_y : int
        The number of rows in the image.
    flip_image : bool, optional
        Whether to flip the image vertically. Default is True.
    """

    def __init__(self, top_level_meta: dict, shape_x: int, shape_y: int, flip_image: bool = True):
        """
        Initialize the LazyMetadata instance.

        Parameters
        ----------
        top_level_meta : dict
            The top-level metadata dictionary.
        shape_x : int
            The number of columns in the image.
        shape_y : int
            The number of rows in the image.
        flip_image : bool, optional
            Whether to flip the image vertically. Default is True.
        """
        self.top_level = top_level_meta
        self.shape_x = shape_x
        self.shape_y = shape_y
        self.flip_image = flip_image


class LazyMetaProxy:
    """
    A proxy class that fetches metadata on demand.

    It behaves like a 2D array of shape (shape_y, shape_x) where each element is a dictionary
    containing the metadata for that pixel.

    Parameters
    ----------
    meta_type : str
        The type of metadata to fetch ("curve" or "segment").
    shape_x : int
        The number of columns in the image.
    shape_y : int
        The number of rows in the image.
    flip_image : bool, optional
        Whether to flip the image vertically. Default is True.
    """

    def __init__(self, meta_type: str, shape_x: int, shape_y: int, flip_image: bool = True):
        """
        Initialize the LazyMetaProxy instance.

        Parameters
        ----------
        meta_type : str
            The type of metadata to fetch ("curve" or "segment").
        shape_x : int
            The number of columns in the image.
        shape_y : int
            The number of rows in the image.
        flip_image : bool, optional
            Whether to flip the image vertically. Default is True.
        """
        self.meta_type = meta_type
        self.shape_x = shape_x
        self.shape_y = shape_y
        self.flip_image = flip_image

    def __getitem__(self, y: int):
        """
        Return a proxy object for the specified row that can be indexed to fetch metadata for each pixel in that row.

        This allows for lazy loading.
        Nested proxy objects are used to allow for fetching segment metadata which requires both x and y indices
        as well as the direction of the segment (approach or retract).

        Parameters
        ----------
        y : int
            The row index.

        Returns
        -------
        RowProxy
            A proxy object for the specified row.
        """

        class RowProxy:
            """
            A proxy class for a single row of the metadata that fetches metadata on demand.

            Parameters
            ----------
            parent : LazyMetaProxy
                The parent LazyMetaProxy instance.
            y : int
                The row index.
            """

            def __init__(self, parent, y):
                """
                Initialize RowProxy with parent LazyMetaProxy and row index.

                Parameters
                ----------
                parent : LazyMetaProxy
                    The parent LazyMetaProxy instance.
                y : int
                    The row index.
                """
                self.parent = parent
                self.y = y

            def __getitem__(self, x):
                """
                Fetch metadata for column x in this row.

                Parameters
                ----------
                x : int
                    The column index.

                Returns
                -------
                dict or SegmentMetaProxy
                    The metadata for the specified column, or a proxy for segment metadata.
                """
                if self.parent.meta_type == "curve":
                    return self.parent._fetch_meta(self.y, x)
                if self.parent.meta_type == "segment":

                    class SegmentMetaProxy:
                        """A proxy class for a single pixel's segment metadata that fetches metadata on demand.

                        Parameters
                        ----------
                        parent : RowProxy
                            The parent RowProxy instance.
                        y : int
                            The row index.
                        x : int
                            The column index.
                        """

                        def __init__(self, parent, y, x):
                            self.parent = parent
                            self.y = y
                            self.x = x

                        def __getitem__(self, direction):
                            """
                            Fetch metadata for the specified segment direction.

                            Parameters
                            ----------
                            direction : int
                                The direction of the segment ("approach" or "retract").

                            Returns
                            -------
                            dict
                                The metadata for the specified segment direction.
                            """
                            return self.parent.parent._fetch_meta(self.y, self.x, direction)

                    return SegmentMetaProxy(self, self.y, x)
                raise IndexError(f"Unknown metadata type '{self.parent.meta_type}'. Expected 'curve' or 'segment'.")

        return RowProxy(self, y)

    def _fetch_meta(self, y: int, x: int, direction: int | None = None):
        """
        Fetch the metadata for a specific pixel.

        Should be implemented by subclasses to define how the metadata is retrieved
        from the underlying data source.

        Parameters
        ----------
        y : int
            The row index of the pixel.
        x : int
            The column index of the pixel.
        direction : int, optional
            The direction of the segment ("approach" or "retract"). Only used for segment metadata.

        Returns
        -------
        dict
            The metadata for the specified pixel.
        """
        raise NotImplementedError("This method should be implemented by subclasses to fetch metadata on demand.")
