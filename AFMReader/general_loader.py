"""Switchboard for input files."""

from pathlib import Path

import numpy.typing as npt

from AFMReader import asd, gwy, h5_jpk, ibw, jpk, spm, stp, top, topostats, jpk_qi
from AFMReader.logging import logger

logger.enable(__package__)


# pylint: disable=too-few-public-methods
class LoadFile:
    """
    Class to handle the general loading of an AFM file.

    Parameters
    ----------
        filepath : Path
            Path to the AFM image.
        channel : str
            Channel to extract from the AFM image.
    """

    def __init__(self, filepath: str | Path, channel: str):
        """
        Initialise the general LoadFile class with a filepath and channel.

        Parameters
        ----------
        filepath : str | Path
            Path to the AFM image.
        channel : str
            Channel to extract from the AFM image.
        """
        self.filepath = Path(filepath)
        self.channel = channel
        self.suffix = self.filepath.suffix

    def load(self) -> tuple[npt.NDArray | str, float | None]:  # noqa: C901
        """
        Generally loads a file type that can be handled by AFMReader.

        Returns
        -------
        tuple
            The image data (stack if ''.asd'' or ''.h5-jpk'') and the pixel to nanometre scaling ratio.

        Raises
        ------
        ValueError
            Where the channel is not found, returned as a tuple of "error message" and "None" so that this can be
            propagated to Napari without outright failing.
        """
        try:
            if self.suffix == ".asd":
                image, pixel_to_nanometre_scaling_factor, _ = asd.load_asd(self.filepath, self.channel)
            elif self.suffix == ".gwy":
                image, pixel_to_nanometre_scaling_factor = gwy.load_gwy(self.filepath, self.channel)
            elif self.suffix == ".ibw":
                image, pixel_to_nanometre_scaling_factor = ibw.load_ibw(self.filepath, self.channel)
            elif self.suffix in [".jpk", ".jpk-qi-image"]:
                image, pixel_to_nanometre_scaling_factor = jpk.load_jpk(self.filepath, self.channel)
            elif self.suffix == ".spm":
                image, pixel_to_nanometre_scaling_factor = spm.load_spm(self.filepath, self.channel)
            elif self.suffix == ".h5-jpk":
                image, pixel_to_nanometre_scaling_factor, _ = h5_jpk.load_h5jpk(self.filepath, self.channel)
            elif self.suffix == ".jpk-qi-data":
                image, pixel_to_nanometre_scaling_factor = jpk_qi.load_jpk_qi(self.filepath, self.channel)
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
            else:
                raise ValueError(f"File type '{self.suffix}' is not currently handled by AFMReader.")

            return image, pixel_to_nanometre_scaling_factor

        except ValueError as e:
            logger.error(f"{e}")
            return (e, None)  # cheeky return of an image, px2nm-like tuple object to propagate error message to Napari

    def get_available_channels(self):
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
            # Implement this
            available_channels = None
        elif self.suffix in [".stp", ".top"]:
            available_channels = stp.load_stp(self.filepath)
        elif self.suffix == ".topostats":
            available_channels = ["image", "image_original"]
        else:
            raise ValueError(f"File type '{self.suffix}' is not currently handled by AFMReader.")
        return available_channels
    # scope for a "check what channels are available" function similar to above.
