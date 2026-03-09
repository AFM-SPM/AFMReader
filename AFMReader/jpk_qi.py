from pathlib import Path
from contextlib import nullcontext
import io
import re
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
    if path_to_image is None:
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
    channels_units = {}

    # Initialize metadata containers
    top_level_meta = {}
    full_metadata = {}

    with zipfile.ZipFile(file_path, "r") as qi_archive:
        if channel not in ADDITIONAL_CHANNELS and not save_as_h5:
            image, px2nm = _load_preprocessed_image(qi_archive=qi_archive, channel=channel, config_path=config_path)

        else:
            if "header.properties" in qi_archive.namelist():
                with qi_archive.open("header.properties") as archive_meta_file:
                    props = javaproperties.load(archive_meta_file)
                    for key, value in props.items():
                        top_level_meta[f"shared-data.{key}"] = value

            # Parse dimensions
            size_x = float(props.get("position-pattern.grid.ulength", 0)) if "position-pattern.grid.ulength" in props else None
            size_y = float(props.get("position-pattern.grid.vlength", 0)) if "position-pattern.grid.vlength" in props else None
            shape_x = int(props.get("position-pattern.grid.ilength", 0)) if "position-pattern.grid.ilength" in props else None
            shape_y = int(props.get("position-pattern.grid.jlength", 0)) if "position-pattern.grid.jlength" in props else None


            if None in [size_x, size_y, shape_x, shape_y]:
                logger.error(f"Incomplete dimension data in {file_path}")

            image = np.zeros((shape_y, shape_x), dtype=np.float32)

            pixel_to_nm_scaling_factor_x = size_x / shape_x * 1e9 if shape_x > 0 else 1.0
            pixel_to_nm_scaling_factor_y = size_y / shape_y * 1e9 if shape_y > 0 else 1.0
            px2nm = (pixel_to_nm_scaling_factor_x + pixel_to_nm_scaling_factor_y) / 2

            segment_channels = []
            if "shared-data/header.properties" in qi_archive.namelist():
                with qi_archive.open("shared-data/header.properties") as shared_data_file:
                    shared_meta = javaproperties.load(shared_data_file)
                    channel_i = 0
                    for key, value in shared_meta.items():
                        top_level_meta[f"shared-data.{key}"] = value

                    while f"lcd-info.{channel_i}.channel.name" in shared_meta:
                        channel_dict = {"name": shared_meta[f"lcd-info.{channel_i}.channel.name"]}
                        multiplier, offset, unit = _get_channel_scaling(shared_meta, channel_i)
                        channel_dict["offset"] = offset
                        channel_dict["multiplier"] = multiplier
                        channel_dict["unit"] = unit
                        segment_channels.append(channel_dict)
                        channel_i += 1

            if len(segment_channels) == 0:
                logger.error("Could not find channels for segments")

            channels_units = {seg_chan['name'] : seg_chan['unit'] for seg_chan in segment_channels}

            h5_context = h5py.File(file_path.parent / f"{file_path.stem}.h5-jpk", "a") if save_as_h5 else nullcontext()
            with h5_context as h5file:
                num_of_curves = shape_x * shape_y

                # Pre-allocate data structures
                curve_meta_dict = {}
                segment_meta_dict = {}
                flat_curve_data = [{} for _ in range(num_of_curves)]
                all_curve_keys = set()
                all_segment_keys = set()

                # Lookup map for binary scaling
                chan_scaling = {chan["name"]: chan for chan in segment_channels}

                # Compile Regexes
                dat_regex = re.compile(r"index/(\d+)/segments/(\d+)/channels/([^/]+)\.dat")
                curve_meta_regex = re.compile(r"index/(\d+)/header\.properties")
                segment_meta_regex = re.compile(r"index/(\d+)/segments/(\d+)/segment-header\.properties")

                # Setup H5 Data structures if needed
                curve_datasets = {}


                if save_as_h5:
                    vlen_type = h5py.vlen_dtype(np.float32)
                    qi_group = h5file.require_group("QI_Curve_Data")
                    global_meta_group = qi_group.require_group("Global_Metadata")
                    curves_meta_group = qi_group.require_group("Curve_Metadata")

                    # curve_datasets = {}
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

                for file_info in qi_archive.infolist():
                    filename = file_info.filename

                    # Check Binary Data
                    dat_match = dat_regex.match(filename)
                    if dat_match:
                        curve_num, direction, chan_name = int(dat_match.group(1)), int(dat_match.group(2)), dat_match.group(3)
                        if chan_name in chan_scaling:
                            scale = chan_scaling[chan_name]
                            with qi_archive.open(file_info) as f:
                                raw_array = np.frombuffer(f.read(), dtype='>i4')
                                segment_array = (raw_array * scale["multiplier"]) + scale["offset"]

                            if chan_name not in flat_curve_data[curve_num]:
                                flat_curve_data[curve_num][chan_name] = {}
                            flat_curve_data[curve_num][chan_name][f"Segment_{direction}"] = segment_array

                            if save_as_h5:
                                curve_datasets[f"{direction}_{chan_name}"][curve_num] = segment_array
                        continue

                    # Check Curve Metadata
                    c_match = curve_meta_regex.match(filename)
                    if c_match:
                        curve_num = int(c_match.group(1))
                        with qi_archive.open(file_info) as f:
                            cleaned_meta = {".".join(k.split(".")[1:]): v for k, v in javaproperties.load(f).items()}
                            curve_meta_dict[curve_num] = cleaned_meta
                            all_curve_keys.update(cleaned_meta.keys())
                        continue

                    # Check Segment Metadata
                    s_match = segment_meta_regex.match(filename)
                    if s_match:
                        curve_num, direction = int(s_match.group(1)), int(s_match.group(2))
                        idx = curve_num * 2 + direction
                        with qi_archive.open(file_info) as f:
                            cleaned_meta = {".".join(k.split(".")[1:]): v for k, v in javaproperties.load(f).items()}
                            segment_meta_dict[idx] = cleaned_meta
                            all_segment_keys.update(cleaned_meta.keys())

                curve_meta = [curve_meta_dict.get(i, {}) for i in range(num_of_curves)]
                segment_meta = [segment_meta_dict.get(i, {}) for i in range(num_of_curves * 2)]

                # Find keys that change across curves/segments
                changing_curve_keys = {k for k in all_curve_keys if any(curve_meta[i].get(k) != curve_meta[0].get(k) for i in range(1, num_of_curves))}
                changing_segment_keys = {k for k in all_segment_keys if any(segment_meta[i].get(k) != segment_meta[0].get(k) for i in range(1, len(segment_meta)))}

                all_curve_data = []
                for y in range(shape_y):
                    row = []
                    for x in range(shape_x):
                        curve_num = y * shape_x + x
                        curve_data = flat_curve_data[curve_num]
                        row.append(curve_data)

                        # Calculate on-the-fly image data if required
                        if channel in ADDITIONAL_CHANNELS:
                            seg_0_dict = {c: data["Segment_0"] for c, data in curve_data.items() if "Segment_0" in data}
                            if channel == "contactPoint":
                                image[y, x] = _find_contact_point(seg_0_dict)
                            elif channel == "manualTriggerPoint":
                                image[y, x] = _find_trigger_point(seg_0_dict)
                    all_curve_data.append(row)
                if channel not in ADDITIONAL_CHANNELS:
                    image, px2nm = _load_preprocessed_image(qi_archive=qi_archive, channel=channel, config_path=config_path)

                # Move duplicated meta to top level
                for key in all_curve_keys - changing_curve_keys:
                    if curve_meta and key in curve_meta[0]:
                        top_level_meta[f"curve.{key}"] = curve_meta[0][key]
                for key in all_segment_keys - changing_segment_keys:
                    if segment_meta and key in segment_meta[0]:
                        top_level_meta[f"segment.{key}"] = segment_meta[0][key]

                # Strip duplicated keys from individual curve/segment dicts
                for c_meta in curve_meta:
                    for k in all_curve_keys - changing_curve_keys: c_meta.pop(k, None)
                for s_meta in segment_meta:
                    for k in all_segment_keys - changing_segment_keys: s_meta.pop(k, None)

                full_metadata = {
                    "top_level": top_level_meta,
                    "curves": curve_meta,
                    "segments": segment_meta
                }

                if save_as_h5:
                    for seg_chan in segment_channels:
                        global_meta_group.attrs[f"channel.unit.{seg_chan['name']}"] = seg_chan['unit']
                    for key, value in top_level_meta.items():
                        global_meta_group.attrs[key] = str(value).encode('utf-8')
                    for i, c_meta in enumerate(curve_meta):
                        curve_meta_group = curves_meta_group.require_group(f"{i}")
                        for key, value in c_meta.items():
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
                            break
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
        return (image, px2nm, (all_curve_data, channels_units, full_metadata))

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



