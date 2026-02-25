from pathlib import Path
from contextlib import nullcontext
import io
import zipfile

import numpy as np
import javaproperties
import h5py

from AFMReader.logging import logger
from AFMReader import jpk


ADDITIONAL_CHANNELS = ["contactPoint_trace", "manualTriggerPoint_trace"]
ADDITIONAL_CHANNELS_IN_M = ["contactPoint_trace", "manualTriggerPoint_trace"]

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
    flip_image: bool | None = True,
    save_as_h5: bool | None = False
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
            if save_as_h5:
                top_level_meta = {}
                changing_curve_keys = set()
                changing_segment_keys = set()
                all_curve_keys = set()
                all_segment_keys = set()

            with qi_archive.open("header.properties") as archive_meta_file:
                props = javaproperties.load(archive_meta_file)
                if save_as_h5:
                    for key, value in props.items():
                        top_level_meta[f"shared-data.{key}"] = value
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
                if save_as_h5:
                    for key, value in shared_meta.items():
                        top_level_meta[f"shared-data.{key}"] = value
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

            h5_context = h5py.File(file_path.parent / f"{file_path.stem}.h5-jpk", "a") if save_as_h5 else nullcontext()
            with h5_context as h5file:
                vlen_type = h5py.vlen_dtype(np.float32)
                num_of_curves = shape_x * shape_y

                if save_as_h5:
                    curve_meta = [{} for _ in range(num_of_curves)]
                    segment_meta = [{} for _ in range(num_of_curves * 2)]
                    qi_group = h5file.require_group("QI_Data")
                    global_meta_group = qi_group.require_group("Global_Metadata")
                    curves_meta_group = qi_group.require_group("Curve_Metadata")

                    curve_datasets = {}
                    for direction in range(2):
                        dir_group = qi_group.require_group(f"Segment_{direction}")
                        for seg_chan in segment_channels:
                            ds_name = seg_chan["name"]
                            if ds_name not in dir_group:
                                curve_datasets[f"{direction}_{ds_name}"] = dir_group.create_dataset(
                                    ds_name, shape=(num_of_curves,), dtype=vlen_type
                                )
                            else:
                                curve_datasets[f"{direction}_{ds_name}"] = dir_group[ds_name]

                for y in range(shape_y):
                    for x in range(shape_x):
                        curve_num = shape_x * y + x
                        if save_as_h5:
                            with qi_archive.open(f"index/{curve_num}/header.properties") as curve_meta_file:
                                curve_meta_raw = javaproperties.load(curve_meta_file)
                                for key, value in curve_meta_raw.items():
                                    key = ".".join(key.split(".")[1:])
                                    curve_meta[curve_num][key] = value
                                    all_curve_keys.add(key)
                                    if curve_num != 0 and (key not in curve_meta[0] or curve_meta[0][key] != value):
                                        changing_curve_keys.add(key)

                        for direction in range(2):
                            if save_as_h5:
                                with qi_archive.open(f"index/{curve_num}/segments/{direction}/segment-header.properties") as segment_meta_file:
                                    segment_meta_raw = javaproperties.load(segment_meta_file)
                                    for key, value in segment_meta_raw.items():
                                        key = ".".join(key.split(".")[1:])
                                        segment_meta[curve_num * 2 + direction][key] = value
                                        all_segment_keys.add(key)
                                        if curve_num != 0 and (key not in segment_meta[0] or segment_meta[0][key] != value):
                                            changing_segment_keys.add(key)
                            curve_data = {}
                            for segment_channel in segment_channels:
                                try:
                                    with qi_archive.open(f"index/{curve_num}/segments/{direction}/channels/{segment_channel['name']}.dat") as segment_data:
                                        dtype_str = '>i4'
                                        raw_bytes = segment_data.read()
                                        raw_array = np.frombuffer(raw_bytes, dtype=dtype_str)
                                        metres_array = (raw_array * segment_channel["multiplier"]) + segment_channel["offset"]
                                        curve_data[segment_channel['name']] = metres_array
                                        if save_as_h5:
                                            curve_datasets[f"{direction}_{segment_channel['name']}"][curve_num] = metres_array
                                except KeyError:
                                    break
                            if channel == "contactPoint_trace":
                                if direction == 0:
                                    image[y, x] = _find_contact_point(curve_data)
                            elif channel == "manualTriggerPoint_trace":
                                if direction == 0:
                                    image[y, x] = _find_trigger_point(curve_data)

                if save_as_h5:
                    # Move all the duplicated metadata to the top level metadata dict
                    for key in all_curve_keys - changing_curve_keys:
                        top_level_meta[f"curve.{key}"] = curve_meta[0][key]
                        for curve_metadata in curve_meta:
                            curve_metadata.pop(key)
                    for key in all_segment_keys - changing_segment_keys:
                        top_level_meta[f"segment.{key}"] = segment_meta[0][key]
                        for segment_metadata in segment_meta:
                            segment_metadata.pop(key)
                    for key, value in top_level_meta.items():
                        global_meta_group.attrs[key] = str(value).encode('utf-8')
                    for i, curve_metadata in enumerate(curve_meta):
                        curve_meta_group = curves_meta_group.require_group(f"{i}")
                        for key, value in curve_metadata.items():
                            curve_meta_group.attrs[key] = str(value).encode('utf-8')

                        for d in range(2):
                            segment_meta_group = curve_meta_group.require_group(f"{d}")
                            for key, value in segment_meta[i*2+d].items():
                                segment_meta_group.attrs[key] = str(value).encode('utf-8')


            if channel in ADDITIONAL_CHANNELS_IN_M:
                image = image * 1e9

            if save_as_h5:
                with h5py.File(file_path.parent / f"{file_path.stem}.h5-jpk", "a") as h5file:
                    meas_grp = h5file.require_group("Measurement_000")
                    meas_grp.attrs["position-pattern.grid.ulength"] = size_x
                    meas_grp.attrs["position-pattern.grid.ilength"] = shape_x
                    meas_grp.attrs["position-pattern.grid.vlength"] = size_y
                    meas_grp.attrs["position-pattern.grid.jlength"] = shape_y
                    meas_grp.attrs["timing-settings.scanRate"] = 1.0  # Dummy value to satisfy reader

                    h5_channels = [channel]
                    for file_name in qi_archive.namelist():
                        if file_name.endswith(".jpk-qi-image"):
                            path_to_image = file_name
                    with qi_archive.open(path_to_image, "r") as image_file:
                        h5_channels += jpk._get_jpk_channels(file=image_file, filename=file_path.stem, file_path=file_path / Path(path_to_image))
                    for i, h5_channel in enumerate(h5_channels):
                        chan_grp = meas_grp.require_group(f"Channel_{_make_num_min_characters(i)}")
                        chan_grp.attrs["channel.name"] = h5_channel.encode("utf-8")
                        chan_grp.attrs["retrace"] = "false".encode("utf-8")
                        chan_grp.attrs["net-encoder.scaling.multiplier"] = 1.0
                        chan_grp.attrs["net-encoder.scaling.offset"] = 0.0

                        # Format name and reshape image (flattened frame stack)
                        dataset_name = h5_channel.split("_")[0].capitalize()
                        frame_stack = image.flatten().reshape(-1, 1)

                        if dataset_name in chan_grp:
                            del chan_grp[dataset_name]
                        chan_grp.create_dataset(dataset_name, data=frame_stack)


        # Need to include flip image as _load_jpk flip image is set to false
        if flip_image:
            image = np.flipud(image)

    return image, px2nm

def load_fdcurves_from_h5(file_path: Path | str):
    file_path = Path(file_path)

    with h5py.File(file_path, "r") as h5file:
        meas_grp = h5file["Measurement_000"]
        shape_x = meas_grp.attrs["position-pattern.grid.ilength"]
        shape_y = meas_grp.attrs["position-pattern.grid.jlength"]
        size_x = meas_grp.attrs["position-pattern.grid.ulength"]
        size_y = meas_grp.attrs["position-pattern.grid.vlength"]

        pixel_to_nm_scaling_factor_x = size_x / shape_x * 1e9 if shape_x > 0 else 1.0
        pixel_to_nm_scaling_factor_y = size_y / shape_y * 1e9 if shape_y > 0 else 1.0
        px2nm = (pixel_to_nm_scaling_factor_x + pixel_to_nm_scaling_factor_y) / 2

        image = np.zeros((shape_y, shape_x), dtype=np.float32)
        segment_0_group = h5file["QI_Data"]["Segment_0"]
        channel_datasets = {name: segment_0_group[name] for name in segment_0_group.keys()}
        for y in range(shape_y):
            for x in range(shape_x):
                curve_num = (shape_x * y) + x
                curve_dict = {}
                for chan_name, dataset in channel_datasets.items():
                    curve_dict[chan_name] = dataset[curve_num]
                image[y, x] = _find_contact_point(curve=curve_dict)
        image = image * 1e9

        return image, px2nm


def _make_num_min_characters(num : int, min_chars: int = 3):
    string_num = str(num)
    if len(string_num) >= min_chars:
        return string_num
    string_num = "0" * (min_chars - len(string_num)) + string_num
    return string_num

def get_jpk_qi_channels(file_path: Path | str):
    file_path = Path(file_path)
    channels = []
    with zipfile.ZipFile(file_path, "r") as qi_archive:
        for file_name in qi_archive.namelist():
            if file_name.endswith(".jpk-qi-image"):
                path_to_image = file_name
        with qi_archive.open(path_to_image, "r") as image_file:
            channels += jpk._get_jpk_channels(file=image_file, filename=file_path.stem, file_path=file_path / Path(path_to_image))
    channels += ADDITIONAL_CHANNELS
    return channels

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



