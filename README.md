# topofileformats

A library for loading various AFM file formats.

File format support: `.asd`

## Usage:

## .asd

You can open `.asd` files using the `load_asd` function. Just pass in the path to the file and the channel name that you want to use. (If in doubt use the `"TP"` topography channel).

Note: For `.asd` files, there seem to only ever be two channels in one file. `"TP"` (topography) is the main one you will want to use unless you know you specifically want something else.
Other channels: `"ER"` - Error, `"PH"` - Phase
```python
from topofileformats import load_asd

frames, pixel_to_nanometre_scaling_factor, metadata = load_asd(file_path="./my_asd_file.asd")
```
