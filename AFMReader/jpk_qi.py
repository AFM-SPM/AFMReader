from pathlib import Path
from contextlib import nullcontext
import io
import zipfile

import numpy as np
import javaproperties
import h5py

from AFMReader.logging import logger
from AFMReader import jpk


ADDITIONAL_CHANNELS = ["contactPoint", "manualTriggerPoint"]
ADDITIONAL_CHANNELS_IN_M = ["contactPoint", "manualTriggerPoint"]

def _get_channel_scaling(props, channel_index):
    """
    Parses the JPK properties dictionary to find the cumulative multiplier
    and offset for a specific channel index (e.g., '1' for vDeflection).
    """
    prefix = f"lcd-info.{channel_index}."

    current_slot = props.get(f"{prefix}conversion-set.conversions.default")

    if not current_slot:
        mult = float(props.get(f"{prefix}encoder.scaling.multiplier", "1.0"))
        off = float(props.get(f"{prefix}encoder.scaling.offset", "0.0"))
        unit = props.get(f"{prefix}encoder.scaling.unit.unit", "Unknown")
        return mult, off, unit

    cumulative_multiplier = 1.0
    cumulative_offset = 0.0
    unit = props.get(f"{prefix}conversion-set.conversion.{current_slot}.scaling.unit.unit")

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
    if not unit:
        unit = props.get(f"{prefix}encoder.scaling.unit.unit", "Unknown")

    return final_multiplier, final_offset, unit

def _load_preprocessed_image(qi_archive, channel, config_path=None):
    path_to_image = None
    for file_name in qi_archive.namelist():
        if file_name.endswith(".jpk-qi-image"):
            path_to_image = file_name
    if path_to_image not in qi_archive.namelist():
        raise FileNotFoundError(f"{path_to_image} not found in JPK archive")

    tif_bytes = qi_archive.read(path_to_image)

    virtual_file = io.BytesIO(tif_bytes)
    logger.info(f"Looking for channel {channel} in ")
    return jpk._load_jpk(virtual_file, path_to_image, channel=channel, file_suffix=".jpk-qi-data", config_path=config_path, flip_image=False)


def load_jpk_qi(
    file_path: Path | str,
    channel: str,
    config_path: Path | str | None = None,
    flip_image: bool | None = True,
    save_as_h5: bool | None = False
) -> tuple[np.ndarray, float]:

    # Load the file path passed to the function
    file_path = Path(file_path)
    all_curve_data = None
    with zipfile.ZipFile(file_path, "r") as qi_archive:
        if channel not in ADDITIONAL_CHANNELS and not save_as_h5:
            image, px2nm = _load_preprocessed_image(qi_archive=qi_archive, channel=channel, config_path=config_path)

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
                    multiplier, offset, unit = _get_channel_scaling(shared_meta, channel_i)
                    channel_dict["offset"] = offset
                    channel_dict["multiplier"] = multiplier
                    channel_dict["unit"] = unit
                    segment_channels.append(channel_dict)
                    channel_i += 1

            if len(segment_channels) == 0:
                logger.error("Could not find channels for segments")

            h5_context = h5py.File(file_path.parent / f"{file_path.stem}.h5-jpk", "a") if save_as_h5 else nullcontext()
            with h5_context as h5file:
                vlen_type = h5py.vlen_dtype(np.float32)
                num_of_curves = shape_x * shape_y
                all_curve_data = []

                if save_as_h5:
                    curve_meta = [{} for _ in range(num_of_curves)]
                    segment_meta = [{} for _ in range(num_of_curves * 2)]
                    qi_group = h5file.require_group("QI_Curve_Data")
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
                    row = []
                    for x in range(shape_x):
                        curve_num = shape_x * y + x
                        curve_data = {}
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
                                try:
                                    with qi_archive.open(f"index/{curve_num}/segments/{direction}/segment-header.properties") as segment_meta_file:
                                        segment_meta_raw = javaproperties.load(segment_meta_file)
                                        for key, value in segment_meta_raw.items():
                                            key = ".".join(key.split(".")[1:])
                                            segment_meta[curve_num * 2 + direction][key] = value
                                            all_segment_keys.add(key)
                                            if curve_num != 0 and (key not in segment_meta[0] or segment_meta[0][key] != value):
                                                changing_segment_keys.add(key)
                                except KeyError:
                                    pass
                            segment_dict = {}
                            for segment_channel in segment_channels:
                                try:
                                    with qi_archive.open(f"index/{curve_num}/segments/{direction}/channels/{segment_channel['name']}.dat") as segment_raw:
                                        dtype_str = '>i4'
                                        raw_bytes = segment_raw.read()
                                        raw_array = np.frombuffer(raw_bytes, dtype=dtype_str)
                                        segment_array = (raw_array * segment_channel["multiplier"]) + segment_channel["offset"]
                                        segment_dict[segment_channel['name']] = segment_array
                                        if save_as_h5:
                                            curve_datasets[f"{direction}_{segment_channel['name']}"][curve_num] = segment_array
                                        if segment_channel['name'] not in curve_data:
                                            curve_data[segment_channel['name']] = {}
                                        curve_data[segment_channel['name']][f"Segment_{direction}"] = segment_array

                                except KeyError:
                                    break
                            if channel == "contactPoint":
                                if direction == 0:
                                    image[y, x] = _find_contact_point(segment_dict)
                            elif channel == "manualTriggerPoint":
                                if direction == 0:
                                    image[y, x] = _find_trigger_point(segment_dict)
                        row.append(curve_data)
                    all_curve_data.append(row)
                if channel not in ADDITIONAL_CHANNELS:
                    image, px2nm = _load_preprocessed_image(qi_archive=qi_archive, channel=channel, config_path=config_path)
                channels_units = {}
                for segment_channel in segment_channels:
                    channels_units[segment_channel['name']] = segment_channel['unit']

                if save_as_h5:
                    # Move all the duplicated metadata to the top level metadata dict
                    for key in all_curve_keys - changing_curve_keys:
                        top_level_meta[f"curve.{key}"] = curve_meta[0][key]
                        for curve_metadata in curve_meta:
                            curve_metadata.pop(key)
                    for key in all_segment_keys - changing_segment_keys:
                        top_level_meta[f"segment.{key}"] = segment_meta[0][key]
                        for segment_metadata in segment_meta:
                            try:
                                segment_metadata.pop(key)
                            except KeyError:
                                pass
                    for segment_channel in segment_channels:
                        global_meta_group.attrs[f"channel.unit.{segment_channel['name']}"] = segment_channel['unit']
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

            # Convert to nanometers if in meters
            if channel in ADDITIONAL_CHANNELS_IN_M:
                image = image * 1e9

            if save_as_h5:
                with h5py.File(file_path.parent / f"{file_path.stem}.h5-jpk", "a") as h5file:
                    # Save data required for reading the h5 file as a normal image file
                    meas_grp = h5file.require_group("Measurement_000")
                    meas_grp.attrs["position-pattern.grid.ulength"] = size_x
                    meas_grp.attrs["position-pattern.grid.ilength"] = shape_x
                    meas_grp.attrs["position-pattern.grid.vlength"] = size_y
                    meas_grp.attrs["position-pattern.grid.jlength"] = shape_y
                    meas_grp.attrs["timing-settings.scanRate"] = 1.0  # Dummy value to satisfy reader

                    logger.info(f"Saving a hdf5 copy of the data {file_path.parent / f'{file_path.stem}.h5-jpk'}")

                    h5_channels = [channel]
                    for file_name in qi_archive.namelist():
                        if file_name.endswith(".jpk-qi-image"):
                            path_to_image = file_name
                    # Add the channels which exist in the jpk-qi-image file
                    with qi_archive.open(path_to_image, "r") as image_file:
                        h5_channels += jpk._get_jpk_channels(file=image_file, filename=file_path.stem, file_path=file_path / Path(path_to_image))
                    for i, h5_channel in enumerate(h5_channels):
                        # For each available channel, save the required data to the h5 file
                        # TODO make sure this metadata is accurate for the channels coming from the .jpk-qi-image file
                        chan_grp = meas_grp.require_group(f"Channel_{_make_num_min_characters(i)}")
                        if "_" in h5_channel:
                            base_name, trace_dir = h5_channel.rsplit("_", 1)
                            is_retrace = "true" if trace_dir.lower() == "retrace" else "false"
                        else:
                            base_name = h5_channel
                            is_retrace = "false"

                        chan_grp.attrs["channel.name"] = base_name.encode("utf-8")
                        chan_grp.attrs["retrace"] = is_retrace.encode("utf-8")
                        chan_grp.attrs["net-encoder.scaling.multiplier"] = 1.0
                        chan_grp.attrs["net-encoder.scaling.offset"] = 0.0

                        # Format name and reshape image (flattened frame stack)
                        dataset_name = h5_channel.split("_")[0].capitalize()
                        if h5_channel == channel:
                            channel_image = image
                        else:
                            channel_image, _ = _load_preprocessed_image(qi_archive=qi_archive, channel=h5_channel, config_path=config_path)
                        frame_stack = channel_image.flatten().reshape(-1, 1)

                        if dataset_name in chan_grp:
                            del chan_grp[dataset_name]
                        chan_grp.create_dataset(dataset_name, data=frame_stack)


        # Need to include flip image as _load_jpk flip image is set to false
        if flip_image:
            image = np.flipud(image)
    if all_curve_data:
        return (image, px2nm, (all_curve_data, channels_units))

    return image, px2nm


def load_fdcurves_from_h5(file_path: Path | str):
    file_path = Path(file_path)

    with h5py.File(file_path, "r") as h5file:
        meas_grp = h5file["Measurement_000"]
        shape_x = meas_grp.attrs["position-pattern.grid.ilength"]
        shape_y = meas_grp.attrs["position-pattern.grid.jlength"]
        num_of_curves = shape_x * shape_y
        qi_data_group = h5file["QI_Curve_Data"]
        all_curve_data = []
        for i in range(num_of_curves):
            curve_data = {}
            for direction, direction_group in qi_data_group.items():
                for channel, channel_group in direction_group.items():
                    if channel not in curve_data:
                        curve_data[channel] = {}
                    curve_data[channel][direction] = channel_group[str(i)]
            all_curve_data.append(curve_data)

        return all_curve_data


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
    return channels, {"save_as_h5": bool}

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



