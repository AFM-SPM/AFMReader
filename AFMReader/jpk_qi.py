from pathlib import Path
import numpy as np
from AFMReader.logging import logger
from AFMReader import jpk
import zipfile
import io
import javaproperties
import h5py


ADDITIONAL_CHANNELS = ["contact_point", "manual_trigger_point"]

def _get_channel_scaling(props, channel_index):
    """
    Parses the JPK properties dictionary to find the cumulative multiplier
    and offset for a specific channel index (e.g., '1' for vDeflection).
    """
    prefix = f"lcd-info.{channel_index}."

    current_slot = props.get(f"{prefix}conversion-set.conversions.default")

    if not current_slot:
        mult = float(props[f"{prefix}encoder.scaling.multiplier"])
        off = float(props[f"{prefix}encoder.scaling.offset"])
        return mult, off

    cumulative_multiplier = 1.0
    cumulative_offset = 0.0

    while current_slot:
        slot_prefix = f"{prefix}conversion-set.conversion.{current_slot}."

        if f"{slot_prefix}scaling.multiplier" in props:
            m = float(props[f"{slot_prefix}scaling.multiplier"])
            c = float(props[f"{slot_prefix}scaling.offset"])

            cumulative_offset = (cumulative_multiplier * c) + cumulative_offset
            cumulative_multiplier *= m

            current_slot = props.get(f"{slot_prefix}base-calibration-slot")

            if current_slot == props.get(f"{prefix}conversion-set.conversions.base"):
                break
        else:
            break

    enc_m = float(props.get(f"{prefix}encoder.scaling.multiplier", "1.0"))
    enc_c = float(props.get(f"{prefix}encoder.scaling.offset", "0.0"))

    final_multiplier = cumulative_multiplier * enc_m
    final_offset = (cumulative_multiplier * enc_c) + cumulative_offset

    return final_multiplier, final_offset


def load_jpk_qi(
    file_path: Path | str,
    channel: str,
    config_path: Path | str | None = None,
    flip_image: bool | None = True
) -> tuple[np.ndarray, float]:

    # Load the file path passed to the function
    file_path = Path(file_path)
    with zipfile.ZipFile(file_path, "r") as qi_archive:
        if channel not in ADDITIONAL_CHANNELS:
            path_to_image = None
            for file_name in qi_archive.namelist():
                if file_name.endswith(".jpk-qi-image"):
                    path_to_image = file_name
            if path_to_image not in qi_archive.namelist():
                raise FileNotFoundError(f"{path_to_image} not found in JPK archive")

            tif_bytes = qi_archive.read(path_to_image)

            virtual_file = io.BytesIO(tif_bytes)
            image, px2nm = jpk._load_jpk(virtual_file, path_to_image, channel=channel, file_suffix=".jpk-qi-data", config_path=config_path, flip_image=False)

        else:
            with qi_archive.open("header.properties") as archive_meta_file:
                props = javaproperties.load(archive_meta_file)
                size_x, size_y, shape_x, shape_y = None, None, None, None
                for key, value in props.items():
                    if key.endswith(".ulength"):
                        size_x = float(value)
                    if key.endswith(".vlength"):
                        size_y = float(value)
                    if key.endswith(".ilength"):
                        shape_x = int(value)
                    if key.endswith(".jlength"):
                        shape_y = int(value)

            if None in [size_x, size_y, shape_x, shape_y]:
                logger.error(f"Incomplete dimension data in {file_path}")

            image = np.zeros((shape_y, shape_x), dtype=np.float32)

            pixel_to_nm_scaling_factor_x = size_x / shape_x * 1e9 if shape_x > 0 else 1.0
            pixel_to_nm_scaling_factor_y = size_y / shape_y * 1e9 if shape_y > 0 else 1.0
            px2nm = (pixel_to_nm_scaling_factor_x + pixel_to_nm_scaling_factor_y) / 2

            segment_channels = []
            with qi_archive.open("shared-data/header.properties") as shared_data_file:
                shared_meta = javaproperties.load(shared_data_file)
                channel_i = 0
                while f"lcd-info.{channel_i}.channel.name" in shared_meta:
                    channel_dict = {}
                    channel_dict["name"] = shared_meta[f"lcd-info.{channel_i}.channel.name"]
                    multiplier, offset = _get_channel_scaling(shared_meta, channel_i)
                    channel_dict["offset"] = offset
                    channel_dict["multiplier"] = multiplier
                    segment_channels.append(channel_dict)
                    channel_i += 1

            if len(segment_channels) == 0:
                logger.error("Could not find channels for segments")

            with h5py.File(file_path.parent / f"{file_path.stem}.h5-jpk", "a") as h5file:
                vlen_type = h5py.vlen_dtype(np.float32)
                num_of_curves = shape_x * shape_y
                master_dataset = h5file.create_dataset("measured_height", shape=(num_of_curves,), dtype=vlen_type)
                for y in range(shape_y):
                    for x in range(shape_x):
                        curve_num = shape_x * y + x
                        print(f"Curve {curve_num}")
                        # with qi_archive.open(f"index/{i}/header.properties") as curve_meta_file:
                        #     curve_meta = javaproperties.load(curve_meta_file)
                        for direction in range(0, 2):
                            # with qi_archive.open(f"index/{i}/segments/{direction}/segment-header.properties") as segment_meta_file:
                            #     segments_meta = javaproperties.load(segment_meta_file)
                            curve_data = {}
                            for segment_channel in segment_channels:
                                try:
                                    with qi_archive.open(f"index/{curve_num}/segments/{direction}/channels/{segment_channel['name']}.dat") as segment_data:
                                        dtype_str = '>i4'
                                        raw_bytes = segment_data.read()
                                        raw_array = np.frombuffer(raw_bytes, dtype=dtype_str)
                                        metres_array = (raw_array * segment_channel["multiplier"]) + segment_channel["offset"]
                                        master_dataset[curve_num] = metres_array
                                        curve_data[segment_channel['name']] = metres_array
                                except KeyError:
                                    break
                            if channel == "contact_point":
                                if direction == 0:
                                    image[y, x] = _find_contact_point(curve_data)
                            elif channel == "manual_trigger_point":
                                if direction == 0:
                                    image[y, x] = _find_trigger_point(curve_data)



        # Need to include flip image as _load_jpk flip image is set to false
        if flip_image:
            image = np.flipud(image)

    return image, px2nm

def get_jpk_qi_channels(file_path: Path | str):
    jpk._get_jpk_channels()




def _fetch_qi_data(file_path: Path | str):
    qi_data = zipfile.ZipFile(file_path, "r")
    return qi_data


def _process_jpk_qi_data(curves_data: list, channel: str) -> tuple[np.ndarray, float]:
    """
    Process the curves data from a JPK QI file to extract the image and pixel to nanometre scaling factor.

    """
    # Calculate pixel to nanometre scaling factor
    metadata = curves_data[0].metadata
    shape_x = metadata.get("grid shape x", 0)
    shape_y = metadata.get("grid shape y", 0)

    size_x = metadata.get("grid size x", 0)
    size_y = metadata.get("grid size y", 0)

    pixel_to_nm_scaling_factor_x = size_x / shape_x if shape_x > 0 else 1.0
    pixel_to_nm_scaling_factor_y = size_y / shape_y if shape_y > 0 else 1.0
    avg_pixel_to_nm_scaling_factor = (pixel_to_nm_scaling_factor_x + pixel_to_nm_scaling_factor_y) / 2

    if channel == "Height (Trigger)":
        z_heights = _find_trigger_points(curves_data)
    elif channel == "Height (Contact)":
        z_heights = [_find_contact_point(curve) for curve in curves_data]
    else:
        raise ValueError(f"Channel '{channel}' not recognized. Only 'Height (Trigger)' and 'Height (Contact)' are currently supported.")

    print(f"Grid shape: {shape_x} x {shape_y}")
    print(f"Z heights: {z_heights}")
    image = np.array(z_heights).reshape(shape_y, shape_x)
    return image, avg_pixel_to_nm_scaling_factor

def _find_trigger_points(curves_data):
    max_points = _max_points_buffer(curves_data)
    n_curves = len(curves_data)

    logger.info(f"Allocating arrays: {n_curves} curves x {max_points} max points.")
    all_segments = np.full((n_curves, max_points), -1, dtype=np.int8)
    all_heights = np.full((n_curves, max_points), np.nan, dtype=np.float32)

    for i, curve in enumerate(curves_data):
        segment = curve["segment"]
        height = curve["height (measured)"]
        length = len(segment)

        all_segments[i, :length] = segment
        all_heights[i, :length] = height

    logger.info("Data stacked. Calculating trigger points...")
    is_approach = (all_segments == 0)

    transition_indices = np.sum(is_approach, axis=1) - 1

    transition_indices = np.maximum(transition_indices, 0)

    row_indices = np.arange(n_curves)
    trigger_values = all_heights[row_indices, transition_indices]

    return trigger_values

def _find_contact_point(curve):
    # find contact point in vertical deflection by peak in first derivative
    vdef = curve["vDeflection"]
    if len(vdef) < 2:
        return np.nan
    derivative_vert_deflection = np.diff(vdef)
    # Doesn't look like this line is needed: peak_derivative_value = np.max(derivative_vert_deflection)
    peak_derivative_index = np.argmax(derivative_vert_deflection)

    # find corresponding height value
    corresponding_height_at_peak = curve["measuredHeight"][peak_derivative_index]

    return corresponding_height_at_peak

def _find_trigger_point(curve):
    trigger_point = curve["measuredHeight"][-1]
    return trigger_point

def _max_points_buffer(curves_data, samples=20, points_buffer=1.2):

    step = len(curves_data) // samples
    max_points = np.max(len(curves_data[i]["segment"]) for i in range(0, len(curves_data), step))
    return max_points * points_buffer



