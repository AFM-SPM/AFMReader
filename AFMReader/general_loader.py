"""Switchboard for input files."""

from pathlib import Path
from typing import Any


from AFMReader import ardf, asd, gwy, h5_jpk, ibw, jpk, raw_bin, spm, stp, top, topostats, jpk_qi
from AFMReader.data_classes import AFMLoad
from AFMReader.logging import logger

logger.enable(__package__)


# pylint: disable=too-few-public-methods,too-many-branches,too-many-statements,fixme
class LoadFile:
    """
    Class to handle the general loading of an AFM file.

    Parameters
    ----------
        filepath : Path
            Path to the AFM image.
        channel : str
            Channel to extract from the AFM image.
        kwargs : dict, optional
            Additional keyword arguments to pass to the specific loaders.
    """

    def __init__(self, filepath: str | Path, channel: str, kwargs: dict | None = None):
        """
        Initialise the general LoadFile class with a filepath and channel.

        Parameters
        ----------
        filepath : str | Path
            Path to the AFM image.
        channel : str
            Channel to extract from the AFM image.
        kwargs : dict, optional
            Additional keyword arguments to pass to the specific loaders.
        """
        self.filepath = Path(filepath)
        self.channel = channel
        self.suffix = self.filepath.suffix.lower()
        self.kwargs = kwargs if kwargs else {}

        # Store heavy loaded data in a dict to avoid having to reload it
        self.cached_data: dict[str, Any] = {}

    def load(self, channel: str | None = None, kwargs: dict | None = None) -> AFMLoad:  # noqa: C901
        """
        Generally loads a file type that can be handled by AFMReader.

        Parameters
        ----------
        channel : str, optional
            Overriding channel to extract from the AFM image.
        kwargs : dict, optional
            Additional keyword arguments to pass to the specific loaders.

        Returns
        -------
        AFMLoad
            An AFMLoad object containing the loaded AFM image data and metadata.

        Raises
        ------
        ValueError
            Where the channel is not found.
        """
        if channel:
            self.channel = channel
        if kwargs:
            self.kwargs = kwargs
        try:
            if self.suffix == ".ardf":
                afm_load = ardf.load_ardf(self.filepath, self.channel, self.cached_data)
            elif self.suffix == ".asd":
                afm_load = asd.load_asd(self.filepath, self.channel)
            elif self.suffix == ".gwy":
                afm_load = gwy.load_gwy(self.filepath, self.channel)
            elif self.suffix == ".ibw":
                afm_load = ibw.load_ibw(self.filepath, self.channel)
            elif self.suffix in [".jpk", ".jpk-qi-image"]:
                afm_load = jpk.load_jpk(self.filepath, self.channel)
            elif self.suffix == ".spm":
                afm_load = spm.load_spm(self.filepath, self.channel)
            elif self.suffix == ".h5-jpk":
                afm_load = h5_jpk.load_h5jpk(self.filepath, self.channel)
            elif self.suffix == ".jpk-qi-data":
                afm_load = jpk_qi.load_jpk_data(
                    filepath=self.filepath, channel=self.channel, cached_data=self.cached_data, **self.kwargs
                )
            elif self.suffix == ".stp":
                afm_load = stp.load_stp(self.filepath)
            elif self.suffix == ".top":
                afm_load = top.load_top(self.filepath)
            elif self.suffix == ".topostats":
                afm_load = topostats.load_topostats(self.filepath, self.channel)
            elif self.suffix == ".bin":
                afm_load = raw_bin.load_bin(self.filepath, **self.kwargs)
            else:
                raise ValueError(f"File type '{self.suffix}' is not currently handled by AFMReader.")

            return afm_load

        except ValueError as e:
            logger.error(f"{e}")
            raise e

    def get_available_channels(self, kwargs: dict | None = None):  # noqa: C901
        """
        Get the available channels for the file type.

        Parameters
        ----------
        kwargs : dict | None, optional
            Additional keyword arguments for channel extraction. Default is None.

        Returns
        -------
        list
            List of available channels.
        """
        self.kwargs = kwargs if kwargs else self.kwargs
        if self.suffix == ".ardf":
            available_channels = ardf.get_ardf_channels(self.filepath, self.cached_data)
        elif self.suffix == ".asd":
            available_channels = asd.get_asd_channels(self.filepath)
        elif self.suffix == ".gwy":
            available_channels = gwy.get_gwy_channels(self.filepath)
        elif self.suffix == ".ibw":
            available_channels = ibw.get_ibw_channels(self.filepath)
        elif self.suffix in [".jpk", ".jpk-qi-image"]:
            available_channels = jpk.get_jpk_channels(self.filepath)
        elif self.suffix == ".spm":
            available_channels = spm.get_spm_channels(self.filepath)
        elif self.suffix == ".h5-jpk":
            available_channels = h5_jpk.get_h5jpk_channels(self.filepath)
        elif self.suffix == ".jpk-qi-data":
            available_channels = jpk_qi.get_jpk_data_channels(self.filepath, self.cached_data)
        elif self.suffix == ".topostats":
            available_channels = topostats.get_topostats_channels()
        elif self.suffix in [".stp", ".top", ".bin"]:
            return []
        else:
            raise ValueError(f"File type '{self.suffix}' is not currently handled by AFMReader.")
        return available_channels

    def get_additional_params(self):
        """
        Get any additional parameters for the file type.

        Returns
        -------
        dict
            Dictionary of additional parameters.
        """
        if self.suffix == ".bin":
            return raw_bin.get_bin_params()
        if self.suffix == ".jpk-qi-data":
            return jpk_qi.get_jpk_data_params(self.filepath, self.cached_data)
        return {}

    def save_to_h5(self):
        """Save the loaded data to an h5 file."""
        if self.suffix == ".jpk-qi-data":
            h5_path = jpk_qi.save_jpk_data_to_h5(filepath=self.filepath, cached_data=self.cached_data)
        else:
            raise ValueError(f"Saving to h5 is not currently implemented for file type '{self.suffix}'.")
        # Once saved to h5, update the filepath and suffix to point to the new h5 file for future loading
        self.filepath = h5_path
        self.suffix = h5_path.suffix.lower()
