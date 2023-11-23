"""Configure logging."""

import sys

from loguru import logger

logger.remove()
# Set the format to have blue time, green file, module, function and line, and white message
logger.add(
    sys.stderr,
    colorize=True,
    format="<blue>{time:HH:mm:ss}</blue> | <level>{level}</level> |"
    "<magenta>{file}</magenta>:<magenta>{module}</magenta>:<magenta>"
    "{function}</magenta>:<magenta>{line}</magenta> | <level>{message}</level>",
)
