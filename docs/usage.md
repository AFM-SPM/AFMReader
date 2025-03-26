# Usage

AFMReader currently obtains the **raw image data** from your selected channel, and the
**pixel-to-nanometre scaling value** (pixels are assumed to be square) for the physical dimensions of the image for
all formats except the `.topostats` format. The `.topostats` metadata will contain that from the processing steps from
[TopoStats](https://afm-spm.github.io/TopoStats/main/index.html).

If you wish to process AFM images supported by `AFMReader` it is recommend you use
[TopoStats](https://github.com/AFM-SPM/TopoStats) to do so, however the library can be used on its own if you wish to
integrate it into your workflow.

## Supported file formats

| File format  | Description                                       |
|--------------|---------------------------------------------------|
| `.asd`       | High-speed AFM                                    |
| `.ibw`       | [WaveMetrics](https://www.wavemetrics.com/)       |
| `.spm`       | [Bruker's Format](https://www.bruker.com/)        |
| `.jpk`       | [Bruker](https://www.bruker.com/)                 |
| `.topostats` | [TopoStats](https://github.com/AFM-SPM/TopoStats) |
| `.gwy`       | [Gwydion](<http://gwyddion.net>)                  |

Support for the following additional formats is planned. Some of these are already supported in TopoStats and are
awaiting refactoring to move their functionality into AFMReader these are denoted in bold below.

| File format | Description                                             | Status             |
|-------------|---------------------------------------------------------|--------------------|
| `.nhf`      | [Nanosurf](https://www.nanosurf.com/en/)                | To Be Implemented. |
| `.aris`     | [Imaris Oxford Instruments](https://imaris.oxinst.com/) | To Be Implemented. |
| `.tiff`     | [Park Systems](https://www.parksystems.com/)            | To Be Implemented. |

## Configuration

A default configuration is included in the package which contains options for loading different file formats. This is a
[YAML](https://yaml.org) file with keys for each file type and the relevant key/value pair nested within. The table
below details the current configuration options. If loading a file format supports a custom configuration file then it
can be specified with the `config_path` parameter which should point to the location of the file.

| File format | Keys                 | Default value |
|-------------|----------------------|---------------|
| `jpk`       | `n_slots`            | `32896`       |
|             | `default_slot`       | `32897`       |
|             | `first_slot_tag`     | `32912`       |
|             | `first_scaling_type` | `32931`       |
|             | `first_scaling_name` | `32932`       |
|             | `first_offset_name`  | `32933`       |
|             | `channel_name`       | `32848`       |
|             | `trace_retrace`      | `32849`       |
|             | `grid_ulength`       | `32834`       |
|             | `grid_vlength`       | `32835`       |
|             | `grid_ilength`       | `32838`       |
|             | `grid_jlength`       | `32839`       |
|             | `slot_size`          | `48`          |

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
