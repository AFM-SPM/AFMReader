"""A module for loading AFM files of different formats."""

from importlib.metadata import version

from loguru import logger

release = version("AFMReader")
__version__ = ".".join(release.split("."[:2]))

# Disable logging for this package by default but allow other loggers to remain
logger.disable("AFMReader")
