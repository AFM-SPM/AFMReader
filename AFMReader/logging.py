"""Configure logging."""

import sys

import loguru
from loguru import logger


def configure_standalone_logging(level: str = "INFO") -> "loguru.Logger":
    """
    Configure logging for standalone usage of AFMReader.

    Note this will remove any existing loggers.

    Parameters
    ----------
    level : str
        Logging level to set for the logger. Options are include "DEBUG", "INFO", "WARNING", "ERROR", and "CRITICAL".

    Returns
    -------
    loguru.Logger
        Configured logger instance.
    """

    logger.remove()  # Remove any existing loggers

    # same format but properly invoked
    logger.add(
        sys.stderr,
        level=level,
        colorize=True,
        format="<blue>{time:HH:mm:ss}</blue> | <level>{level}</level> |"
        "<magenta>{file}</magenta>:<magenta>{module}</magenta>:<magenta>"
        "{function}</magenta>:<magenta>{line}</magenta> | <level>{message}</level>",
    )
    return logger
