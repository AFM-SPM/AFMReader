# topofileformats

<div align="center">

[![PyPI version](https://badge.fury.io/py/topofileformats.svg)](https://badge.fury.io/py/topofileformats)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/topofileformats)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Code style: flake8](https://img.shields.io/badge/code%20style-flake8-456789.svg)](https://github.com/psf/flake8)
<!-- [![codecov](https://codecov.io/gh/AFM-SPM/topofileformats/branch/dev/graph/badge.svg)]
(https://codecov.io/gh/AFM-SPM/topofileformats) -->
[![pre-commit.ci
status](https://results.pre-commit.ci/badge/github/AFM-SPM/topofileformats/main.svg)](https://results.pre-commit.ci/latest/github/AFM-SPM/topofileformats/main)

</div>

A library for loading various Atomic Force Microscopy (AFM) file formats into Python. This library is primarily intended
for use with [TopoStats](https://github.com/AFM-SPM/TopoStats).

Supported file formats

| File format | Description    |
|-------------|----------------|
| `.asd`      | High-speed AFM |

Support for the following additional formats is planned. Some of these are already supported in TopoStats and are
awaiting refactoring to move their functionality into topofileformats these are denoted in bold below.

| File format | Description                                             | Status                                     |
|-------------|---------------------------------------------------------|--------------------------------------------|
| `.spm`      | [Bruker](https://www.bruker.com/)                       | TopoStats supported, to be migrated (#16). |
| `.ibw`      | [WaveMetrics](https://www.wavemetrics.com/)             | TopoStats supported, to be migrated (#17). |
| `.gwy`      | [Gwyddion](http://gwyddion.net/)                        | TopoStats supported, to be migrated (#1).  |
| `.nhf`      | [Nanosurf](https://www.nanosurf.com/en/)                | To Be Implemented.                         |
| `.aris`     | [Imaris Oxford Instruments](https://imaris.oxinst.com/) | To Be Implemented.                         |
| `.tiff`     | [Park Systems](https://www.parksystems.com/)            | To Be Implemented.                         |

## Usage

If you wish to process AFM images supported by `topofileformats` it is recommend you use
[TopoStats](https://github.com/AFM-SPM/TopoStats) to do so, however the library can be used on its own.

### .asd

You can open `.asd` files using the `load_asd` function. Just pass in the path to the file and the channel name that you
want to use. (If in doubt use the `"TP"` topography channel).

Note: For `.asd` files, there seem to only ever be two channels in one file. `"TP"` (topography) is the main one you
will want to use unless you know you specifically want something else.

Other channels: `"ER"` - Error, `"PH"` - Phase

```python
from topofileformats import load_asd

frames, pixel_to_nanometre_scaling_factor, metadata = load_asd(file_path="./my_asd_file.asd")
```

## Contributing

Bug reports and feature requests are welcome. Please search for existing issues, if none relating to your bug/feature
are found then feel free to create a new [issue](https://github.com/AFM-SPM/topofileformats/issues/new) detailing what
went wrong or the feature you would like to see implemented.

Pull requests are also welcome, please note that we have a [Code of
Conduct](https://github.com/AFM-SPM/topofileformats/blob/main/CODE_OF_CONDUCT.md).

### Setup

We use [pre-commit](https://pre-commit.com) to apply linting via [ruff](https://github.com/astral-sh/ruff) and
[pylint](https://pylint.pycqa.org/en/latest/index.html) pre-commit hooks and use the
[Black](https://github.com/psf/black) and [Flake8](https://github.com/psf/flake8) code styles. To set yourself up for
contributing after cloning the package and creating a Python virtual environment you should install the development
dependencies and `pre-commit` as shown below.

``` bash
# Activate your virtual environment, this will depend on which system you use e.g. conda or virtualenvwrapper
# Clone the repository
git clone git@github.com:AFM-SPM/topofileformats.git
# Change directories into the newly cloned directory
cd topofileformats
# Install the package along with the optional development (dev) dependencies
pip install -e .[dev]
# Install pre-commit
pre-commit install
```

This will ensure that any commits and pull requests you make will pass the [Pre-commit Continuous
Integration](https://pre-commit.ci). Where possible `ruff` will correct the changes it can, but it may require you to
address some issues manually, before adding any changes and attempting to commit again.
