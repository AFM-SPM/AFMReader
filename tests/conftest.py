"""Fixtures for testing."""

from pathlib import Path

import pySPM
import pytest


BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"


@pytest.fixture()
def spm_channel_data() -> pySPM.SPM.SPM_image:
    """Instantiate channel data from a LoadScans object."""
    scan = pySPM.Bruker(RESOURCES / "sample_0.spm")
    return scan.get_channel("Height")
