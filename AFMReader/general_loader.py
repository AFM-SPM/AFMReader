"""Switchboard for input files."""

from pathlib import Path

import numpy.typing as npt

from AFMReader import asd, gwy, ibw, jpk, spm, stp, top, topostats
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
            The image data (stack if ''.asd'') and the pixel to nanometre scaling ratio.

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
            elif self.suffix == ".jpk":
                image, pixel_to_nanometre_scaling_factor = jpk.load_jpk(self.filepath, self.channel)
            elif self.suffix == ".spm":
                image, pixel_to_nanometre_scaling_factor = spm.load_spm(self.filepath, self.channel)
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

    # scope for a "check what channels are available" function similar to above.
