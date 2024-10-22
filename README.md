# AFMReader

<div align="center">

[![PyPI version](https://badge.fury.io/py/AFMReader.svg)](https://badge.fury.io/py/AFMReader)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/AFMReader)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Code style: flake8](https://img.shields.io/badge/code%20style-flake8-456789.svg)](https://github.com/psf/flake8)
<!-- [![codecov](https://codecov.io/gh/AFM-SPM/AFMReader/branch/dev/graph/badge.svg)]
(https://codecov.io/gh/AFM-SPM/AFMReader) -->
[![pre-commit.ci
status](https://results.pre-commit.ci/badge/github/AFM-SPM/AFMReader/main.svg)](https://results.pre-commit.ci/latest/github/AFM-SPM/AFMReader/main)
[![fair-software.eu](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B-yellow)](https://fair-software.eu)

</div>
<div align="center">

[![Downloads](https://static.pepy.tech/badge/afmreader)](https://pepy.tech/project/afmreader)
[![Downloads](https://static.pepy.tech/badge/afmreader/month)](https://pepy.tech/project/afmreader)
[![Downloads](https://static.pepy.tech/badge/afmreader/week)](https://pepy.tech/project/afmreader)

</div>

A library for loading various Atomic Force Microscopy (AFM) file formats into Python. This library is primarily intended
for use with [TopoStats](https://github.com/AFM-SPM/TopoStats).

Supported file formats

| File format | Description    |
|-------------|----------------|
| `.asd`      | High-speed AFM |
| `.ibw`      | [WaveMetrics](https://www.wavemetrics.com/)  |
| `.spm`      | [Bruker's Format](https://www.bruker.com/)  |
| `.jpk`      | [Bruker](https://www.bruker.com/) |
| `.topostats`| [TopoStats](https://github.com/AFM-SPM/TopoStats)  |
| `.gwy`      | [Gwydion](<http://gwyddion.net>) |

Support for the following additional formats is planned. Some of these are already supported in TopoStats and are
awaiting refactoring to move their functionality into AFMReader these are denoted in bold below.

| File format | Description                                             | Status                                     |
|-------------|---------------------------------------------------------|--------------------------------------------|
| `.nhf`      | [Nanosurf](https://www.nanosurf.com/en/)                | To Be Implemented.                         |
| `.aris`     | [Imaris Oxford Instruments](https://imaris.oxinst.com/) | To Be Implemented.                         |
| `.tiff`     | [Park Systems](https://www.parksystems.com/)            | To Be Implemented.                         |

## Usage

If you wish to process AFM images supported by `AFMReader` it is recommend you use
[TopoStats](https://github.com/AFM-SPM/TopoStats) to do so, however the library can be used on its own.

### .topostats

You can open `.topostats` files using the `load_topostats` function. Just pass in the path to the file.

```python
from AFMReader.topostats import load_topostats

image, pixel_to_nanometre_scaling_factor, metadata = load_topostats(file_path="./my_topostats_file.topostats")
```

### .spm

You can open `.spm` files using the `load_spm` function. Just pass in the path to the file and the
channel name that you want to use. (If in doubt use one of the following: "Height", "ZSensor",
"Height Sensor").

```python
from AFMReader.spm import load_spm

image, pixel_to_nanometre_scaling_factor = load_spm(file_path="./my_spm_file.spm", channel="Height")
```

### .gwy

You can open `.gwy` files using the `load_gwy` function. Just pass in the path to the file and the
channel name that you want to use. (If in doubt use one one of the following: "Height", "ZSensor",
"Height Sensor").

```python
from AFMReader.gwy import load_gwy

image, pixel_to_nanometre_scaling_factor = load_gwy(file_path="./my_gwy_file.gwy", channel="Height")
```

### .asd

You can open `.asd` files using the `load_asd` function. Just pass in the path to the file and the channel name that you
want to use. (If in doubt use the `"TP"` topography channel).

Note: For `.asd` files, there seem to only ever be two channels in one file. `"TP"` (topography) is the main one you
will want to use unless you know you specifically want something else.

Other channels: `"ER"` - Error, `"PH"` - Phase

```python
from AFMReader.asd import load_asd

frames, pixel_to_nanometre_scaling_factor, metadata = load_asd(file_path="./my_asd_file.asd", channel="TP")
```

### .ibw

You can open `.ibw` files using the `load_ibw` function. Just pass in the path to the file
and the channel name that you want to use. (If in doubt, use `HeightTracee` (yes, with the
extra 'e'), `ZSensorTrace`, or `ZSensor`).

```python
from AFMReader.ibw import load_ibw

image, pixel_to_nanometre_scaling_factor = load_ibw(file_path="./my_ibw_file.ibw", channel="HeightTracee")
```

### .jpk

You can open `.jpk` files using the `load_jpk` function. Just pass in the path
to the file and the channel name you want to use. (If in doubt, use `height_trace` or `measuredHeight_trace`).

```python
from AFMReader.jpk import load_jpk

image, pixel_to_nanometre_scaling_factor = load_jpk(file_path="./my_jpk_file.jpk", channel="height_trace")
```

## Contributing

Bug reports and feature requests are welcome. Please search for existing issues, if none relating to your bug/feature
are found then feel free to create a new [issue](https://github.com/AFM-SPM/AFMReader/issues/new) detailing what
went wrong or the feature you would like to see implemented.

Pull requests are also welcome, please note that we have a [Code of
Conduct](https://github.com/AFM-SPM/AFMReader/blob/main/CODE_OF_CONDUCT.md).

### Setup

We use [pre-commit](https://pre-commit.com) to apply linting via [ruff](https://github.com/astral-sh/ruff) and
[pylint](https://pylint.pycqa.org/en/latest/index.html) pre-commit hooks and use the
[Black](https://github.com/psf/black) and [Flake8](https://github.com/psf/flake8) code styles. To set yourself up for
contributing after cloning the package and creating a Python virtual environment you should install the development
dependencies and `pre-commit` as shown below.

``` bash
# Activate your virtual environment, this will depend on which system you use e.g. conda or virtualenvwrapper
# Clone the repository
git clone git@github.com:AFM-SPM/AFMReader.git
# Change directories into the newly cloned directory
cd AFMReader
# Install the package along with the optional development (dev) dependencies
pip install -e .[dev]
# Install pre-commit
pre-commit install
```

This will ensure that any commits and pull requests you make will pass the [Pre-commit Continuous
Integration](https://pre-commit.ci). Where possible `ruff` will correct the changes it can, but it may require you to
address some issues manually, before adding any changes and attempting to commit again.
