"""Switchboard for input files."""

from pathlib import Path
from typing import Any

import numpy.typing as npt

from AFMReader import asd, gwy, h5_jpk, ibw, jpk, raw_bin, spm, stp, top, topostats, jpk_qi
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
        self.suffix = self.filepath.suffix
        self.loaded_curves = False
        self.kwargs = kwargs if kwargs else {}

        # Store heavy loaded data in a dict to avoid having to reload it
        self.cached_data: dict[str, Any] = {}

    def load(  # noqa: C901
        self, channel: str | None = None, kwargs: dict | None = None
    ) -> tuple[npt.NDArray | str, float | None] | tuple[npt.NDArray | str, float | None, Any]:
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
        tuple
            The image data (stack if ''.asd'' or ''.h5-jpk'') and the pixel to nanometre scaling ratio.
            If curve data is found, also return the curve data (a large dict of all the curves).

        Raises
        ------
        ValueError
            Where the channel is not found, returned as a tuple of "error message" and "None" so that this can be
            propagated to Napari without outright failing.
        """
        if channel:
            self.channel = channel
        if kwargs:
            self.kwargs = kwargs
        try:
            if self.suffix == ".asd":
                image, pixel_to_nanometre_scaling_factor, _, _ = asd.load_asd(self.filepath, self.channel)
            elif self.suffix == ".gwy":
                image, pixel_to_nanometre_scaling_factor, _ = gwy.load_gwy(self.filepath, self.channel)
            elif self.suffix == ".ibw":
                image, pixel_to_nanometre_scaling_factor = ibw.load_ibw(self.filepath, self.channel)
            elif self.suffix in [".jpk", ".jpk-qi-image"]:
                image, pixel_to_nanometre_scaling_factor, _ = jpk.load_jpk(self.filepath, self.channel)
            elif self.suffix == ".spm":
                image, pixel_to_nanometre_scaling_factor, _ = spm.load_spm(self.filepath, self.channel)
            elif self.suffix == ".h5-jpk":
                h5_returned = h5_jpk.load_h5jpk(self.filepath, self.channel, load_curves=not self.loaded_curves)
                if len(h5_returned) == 4:
                    image, pixel_to_nanometre_scaling_factor, _, z_units = h5_returned  # type: ignore[misc]
                elif len(h5_returned) == 5:
                    image, pixel_to_nanometre_scaling_factor, _, z_units, curve_data = h5_returned  # type: ignore[misc]
                    self.loaded_curves = True
                    print(
                        f"Loaded image with shape {image.shape} and pixel to nanometre "
                        f"scaling factor {pixel_to_nanometre_scaling_factor}"
                    )
                    print(f"Image has max value {image.max()} and min value {image.min()}")
                    return image, pixel_to_nanometre_scaling_factor, curve_data
                else:
                    logger.error(f"Loading h5-jpk file returned unexpected number of items: {len(h5_returned)}")
                    raise ValueError(f"Loading h5-jpk file returned unexpected number of items: {len(h5_returned)}")
            elif self.suffix == ".jpk-qi-data":
                if "jpk_qi_loader" not in self.cached_data:
                    self.cached_data["jpk_qi_loader"] = jpk_qi.jpk_qi_loader(
                        filepath=self.filepath, channel=self.channel, **self.kwargs
                    )
                jpk_qi_returned = self.cached_data["jpk_qi_loader"].load(channel=self.channel, **self.kwargs)
                if len(jpk_qi_returned) == 2:
                    image, pixel_to_nanometre_scaling_factor = jpk_qi_returned
                elif len(jpk_qi_returned) == 3:
                    image, pixel_to_nanometre_scaling_factor, curve_data = jpk_qi_returned
                    self.loaded_curves = True
                    return image, pixel_to_nanometre_scaling_factor, curve_data
                else:
                    logger.error(f"Loading h5-jpk file returned unexpected number of items: {len(jpk_qi_returned)}")
            elif self.suffix == ".stp":
                image, pixel_to_nanometre_scaling_factor = stp.load_stp(self.filepath)
            elif self.suffix == ".top":
                image, pixel_to_nanometre_scaling_factor = top.load_top(self.filepath)
            elif self.suffix == ".topostats":
                ts_dict = topostats.load_topostats(self.filepath)
                try:
                    image = ts_dict[self.channel]
                    pixel_to_nanometre_scaling_factor = ts_dict["pixel_to_nm_scaling"]
                except KeyError as exc:
                    image_keys = ["image", "image_original"]
                    topostats_keys = list(ts_dict.keys())
                    raise ValueError(
                        f"'{self.channel}' not in available image keys: "
                        f"{[im for im in image_keys if im in topostats_keys]}"
                    ) from exc
            elif self.suffix == ".bin":
                image, pixel_to_nanometre_scaling_factor = raw_bin.load_bin(self.filepath, **self.kwargs)
            else:
                raise ValueError(f"File type '{self.suffix}' is not currently handled by AFMReader.")

            return image, pixel_to_nanometre_scaling_factor

        except ValueError as e:
            logger.error(f"{e}")
            raise e

    def get_available_channels(self):  # noqa: C901
        """
        Get the available channels for the file type.

        Returns
        -------
        list
            List of available channels.
        """
        if self.suffix == ".asd":
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
            if "jpk_qi_loader" not in self.cached_data:
                self.cached_data["jpk_qi_loader"] = jpk_qi.jpk_qi_loader(filepath=self.filepath, **self.kwargs)
            available_channels = self.cached_data["jpk_qi_loader"].get_available_channels()
        elif self.suffix == ".topostats":
            available_channels = ["image", "image_original"]
        elif self.suffix == ".bin":
            available_channels = raw_bin.get_bin_channels()
        elif self.suffix in [".stp", ".top"]:
            return []
        else:
            raise ValueError(f"File type '{self.suffix}' is not currently handled by AFMReader.")
        return available_channels
