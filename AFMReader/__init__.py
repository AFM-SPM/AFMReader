"""A module for loading AFM files of different formats."""

from importlib.metadata import version

release = version("topofileformats")
__version__ = ".".join(release.split("."[:2]))
