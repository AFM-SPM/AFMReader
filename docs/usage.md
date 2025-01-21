# Usage

If you wish to process AFM images supported by `AFMReader` it is recommend you use
[TopoStats](https://github.com/AFM-SPM/TopoStats) to do so, however the library can be used on its own if you wish to
integrate it into your workflow.

## Supported file formats

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

## .topostats

You can open `.topostats` files using the `load_topostats` function. Just pass in the path to the file.

```python
from AFMReader.topostats import load_topostats

image, pixel_to_nanometre_scaling_factor, metadata = load_topostats(file_path="./my_topostats_file.topostats")
```

## .spm

You can open `.spm` files using the `load_spm` function. Just pass in the path to the file and the
channel name that you want to use. (If in doubt use one of the following: "Height", "ZSensor",
"Height Sensor").

```python
from AFMReader.spm import load_spm

image, pixel_to_nanometre_scaling_factor = load_spm(file_path="./my_spm_file.spm", channel="Height")
```

## .gwy

You can open `.gwy` files using the `load_gwy` function. Just pass in the path to the file and the
channel name that you want to use. (If in doubt use one one of the following: "Height", "ZSensor",
"Height Sensor").

```python
from AFMReader.gwy import load_gwy

image, pixel_to_nanometre_scaling_factor = load_gwy(file_path="./my_gwy_file.gwy", channel="Height")
```

## .asd

You can open `.asd` files using the `load_asd` function. Just pass in the path to the file and the channel name that you
want to use. (If in doubt use the `"TP"` topography channel).

Note: For `.asd` files, there seem to only ever be two channels in one file. `"TP"` (topography) is the main one you
will want to use unless you know you specifically want something else.

Other channels: `"ER"` - Error, `"PH"` - Phase

```python
from AFMReader.asd import load_asd

frames, pixel_to_nanometre_scaling_factor, metadata = load_asd(file_path="./my_asd_file.asd", channel="TP")
```

## .ibw

You can open `.ibw` files using the `load_ibw` function. Just pass in the path to the file
and the channel name that you want to use. (If in doubt, use `HeightTracee` (yes, with the
extra 'e'), `ZSensorTrace`, or `ZSensor`).

```python
from AFMReader.ibw import load_ibw

image, pixel_to_nanometre_scaling_factor = load_ibw(file_path="./my_ibw_file.ibw", channel="HeightTrace")
```

## .jpk

You can open `.jpk` files using the `load_jpk` function. Just pass in the path
to the file and the channel name you want to use. (If in doubt, use `height_trace` or `measuredHeight_trace`).

```python
from AFMReader.jpk import load_jpk

image, pixel_to_nanometre_scaling_factor = load_jpk(file_path="./my_jpk_file.jpk", channel="height_trace")
```
