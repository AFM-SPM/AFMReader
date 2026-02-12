import afmformats
from pathlib import Path
import numpy as np
from AFMReader.logging import logger



def load_jpk_qi(
    file_path: Path | str,
    channel: str,
    flip_image: bool | None = True
) -> tuple[np.ndarray, float]:

    # Load the file path passed to the function
    curves_data = _fetch_qi_data(file_path)

    if curves_data is None or len(curves_data) == 0:
        print("No data found in the file.")
        return

    image, px2nm = _process_jpk_qi_data(curves_data, channel)

    if flip_image:
        image = np.flipud(image)

    return image, px2nm



def _fetch_qi_data(file_path: Path | str):
    return afmformats.load_data(file_path)


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
    derivative_vert_deflection = np.diff(curve["force"])
    # Doesn't look like this line is needed: peak_derivative_value = np.max(derivative_vert_deflection)
    peak_derivative_index = np.argmax(derivative_vert_deflection)

    # find corresponding height value
    corresponding_height_at_peak = curve["height (measured)"][peak_derivative_index]

    return corresponding_height_at_peak

def _find_trigger_point(curve):
    segments = curve["segment"]
    approach_indices = np.where(segments == 0)[0]
    turn_index = approach_indices[-1]
    trigger_point = curve["height (measured)"][turn_index]
    return trigger_point

def _max_points_buffer(curves_data, samples=20, points_buffer=1.2):

    step = len(curves_data) // samples
    max_points = np.max(len(curves_data[i]["segment"]) for i in range(0, len(curves_data), step))
    return max_points * points_buffer



