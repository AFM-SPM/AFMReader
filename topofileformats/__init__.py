"""topofileformats."""

from importlib.metadata import version

release = version("topofileformats")
__version__ = ".".join(release.split("."[:2]))
