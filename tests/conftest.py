"""Fixtures for testing."""

from pathlib import Path
import logging

import pySPM
import pytest
from _pytest.logging import caplog as _caplog

from AFMReader.logging import logger

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"


@pytest.fixture()
def caplog(_caplog):
    """Ensure the caplog works with loguru."""
    class PropogateHandler(logging.Handler):
        def emit(self, record):
            logging.getLogger(record.name).handle(record)

    handler_id = logger.add(PropogateHandler(), format="{message}")
    yield _caplog
    logger.remove(handler_id)


@pytest.fixture()
def spm_channel_data() -> pySPM.SPM.SPM_image:
    """Instantiate channel data from a LoadScans object."""
    scan = pySPM.Bruker(RESOURCES / "sample_0.spm")
    return scan.get_channel("Height")
