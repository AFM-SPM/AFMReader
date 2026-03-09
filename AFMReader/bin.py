import numpy as np
import math
from .logging import logger
from pathlib import Path

DTYPE_MAP = {
    'IEEE double': np.float64,
    'DBL': np.float64,
    'IEEE single': np.float32,
    'SGL': np.float32,
    'U32': np.uint32,
    'I32': np.int32,
    'U16': np.uint16,
    'I16': np.int16,
    'U8': np.uint8,
    'I8': np.int8,
    'float64': np.float64,
    'float32': np.float32,
    'int32': np.int32,
}

def load_bin(filepath: str | Path, data_type: str, offset_bytes: int, size_x: float = None, size_y: float = None, shape_x: int = None, shape_y: int = None, z_scaling: float = 1.0):
    filepath = Path(filepath)
    dt_key = str(data_type).strip()
    shape_x = None if shape_x == 0 else shape_x
    shape_y = None if shape_y == 0 else shape_y

    if dt_key in DTYPE_MAP:
        np_dtype = DTYPE_MAP[dt_key]
    else:
        logger.warning(f"Unknown data type '{dt_key}'. Defaulting to float64.")
        np_dtype = np.float64
    with open(filepath, 'rb') as f:
        f.seek(offset_bytes)
        flat_data = np.fromfile(f, dtype=np_dtype)
    if None in [shape_x, shape_y]:
        dimension = int(math.sqrt(len(flat_data)))
        shape_x, shape_y = dimension, dimension
    if shape_x * shape_y != len(flat_data):
        logger.error(f"Loading binary file {filepath.stem} did not receive a shape and is not square")
    image = flat_data.reshape((shape_x, shape_y))
    image *= z_scaling
    pixel_to_nm_scaling_factor_x = size_x / shape_x if shape_x > 0 else 1.0
    pixel_to_nm_scaling_factor_y = size_y / shape_y if shape_y > 0 else 1.0
    px2nm = (pixel_to_nm_scaling_factor_x + pixel_to_nm_scaling_factor_y) / 2
    return image, px2nm

def get_bin_channels():
    kwarg_types = {"data_type" : (str, DTYPE_MAP.keys()),
                   "offset_bytes": int,
                   "size_x": float,
                   "size_y": float,
                   "shape_x": int,
                   "shape_y": int,
                   "z_scaling": float}
    return [], kwarg_types

