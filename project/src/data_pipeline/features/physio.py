"""Preprocessing functions for physiological and pupil data.

This module processes PERCLOS (Percentage of Eye Closure), blood pressure,
skin conductance, respiration, oxygen saturation, and pupil diameter data
from raw MAT files. It computes window-based features for use in driver
drowsiness detection.

Functions:
    - perclos_process(): Save PERCLOS and physiological data as CSV.
    - pupil_process(): Save cleaned pupil size features as CSV.
"""

import pandas as pd
import numpy as np
import logging
from scipy.interpolate import interp1d

from src.utils.io.loaders import safe_load_mat, save_csv
from src.config import DATASET_PATH, MODEL_WINDOW_CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_physio_window_sec(model: str) -> float:
    """Retrieve the window size in seconds for a given model.

    Args:
        model (str): Model name (e.g., 'Lstm', 'SvmW', etc.).

    Returns:
        float: Window size in seconds.
    """
    return MODEL_WINDOW_CONFIG[model]["window_sec"]


def calculate_perclos(blinks: np.ndarray, window_size_sec: float) -> pd.DataFrame:
    """Calculate PERCLOS (Percentage of Eye Closure) per time window.

    Args:
        blinks (np.ndarray): Blink events array from MAT file.
        window_size_sec (float): Duration of each analysis window in seconds.

    Returns:
        pd.DataFrame: DataFrame with Timestamp and PERCLOS values per window.
    """
    blink_starts, blink_durations = blinks[:, 10], blinks[:, 2]
    total_duration = blink_starts[-1] - blink_starts[0]
    num_windows = int(total_duration // window_size_sec)

    perclos_values, timestamps = [], []
    start_time = blink_starts[0]

    for i in range(num_windows):
        window_start = start_time + i * window_size_sec
        mask = (blink_starts >= window_start) & (blink_starts < window_start + window_size_sec)
        perclos = blink_durations[mask].sum() / window_size_sec
        perclos_values.append(perclos)
        timestamps.append(window_start)

    return pd.DataFrame({"Timestamp": timestamps, "PERCLOS": perclos_values})


def process_physio_data(physio_content: np.ndarray, window_size_sec: float) -> pd.DataFrame:
    """Aggregate physiological signals over non-overlapping windows.

    Args:
        physio_content (np.ndarray): 2D array [5, N] of physio signals.
        window_size_sec (float): Duration of each analysis window in seconds.

    Returns:
        pd.DataFrame: Window-averaged physiological feature DataFrame.
    """
    physio_df = pd.DataFrame({
        "Timestamp": physio_content[0],
        "Blood_Pressure": physio_content[1],
        "Skin_Conductance": physio_content[2],
        "Respiration": physio_content[3],
        "Oxygen_Saturation": physio_content[4]
    })

    physio_df['Window'] = ((physio_df['Timestamp'] - physio_df['Timestamp'].min()) // window_size_sec).astype(int)
    return physio_df.groupby('Window').mean().reset_index()


def calculate_and_save_perclos_physio_combined(
    blink_data_path: str, physio_data_path: str, window_size_sec: float
) -> pd.DataFrame | None:
    """Combine and align PERCLOS and physiological data by time window.

    Args:
        blink_data_path (str): Path to blink MAT file.
        physio_data_path (str): Path to physiological data MAT file.
        window_size_sec (float): Duration of analysis window.

    Returns:
        pd.DataFrame or None: Combined features DataFrame or None if data missing.
    """
    blink_data = safe_load_mat(blink_data_path)
    physio_data = safe_load_mat(physio_data_path)

    if blink_data is None or physio_data is None:
        logging.error("Missing Blink or Physio data, skipping...")
        return None

    perclos_df = calculate_perclos(blink_data['Blinks'], window_size_sec)
    perclos_df['Window'] = ((perclos_df['Timestamp'] - perclos_df['Timestamp'].min()) // window_size_sec).astype(int)

    physio_resampled = process_physio_data(physio_data['PhysioData'], window_size_sec)

    return pd.merge(physio_resampled, perclos_df, on="Window", how="inner")


def perclos_process(subject: str) -> None:
    """Main function to extract and save combined perclos + physio features.

    Args:
        subject (str): Subject identifier in 'subjectID/version' format.

    Returns:
        None
    """
    subject_id, version = subject.split('/')[0], subject.split('/')[1].split('_')[-1]

    blink_data_file = f"{DATASET_PATH}/{subject_id}/Blinks_{subject_id}_{version}.mat"
    physio_data_file = f"{DATASET_PATH}/{subject_id}/Physio_{subject_id}_{version}.mat"

    df_combined = calculate_and_save_perclos_physio_combined(blink_data_file, physio_data_file, get_physio_window_sec(subject.split('/')[1].split('_')[0]))
    if df_combined is not None:
        save_csv(df_combined, subject_id, version, 'perclos')


# ----- Pupil Processing -----

def replace_outliers_with_interpolation(data: np.ndarray, threshold: float = 3) -> np.ndarray:
    """Replace outliers in a 1D signal with interpolated values.

    Args:
        data (np.ndarray): 1D time series.
        threshold (float): Number of std devs for outlier detection.

    Returns:
        np.ndarray: Signal with outliers interpolated.
    """
    mean, std_dev = np.nanmean(data), np.nanstd(data)
    outliers = (data < mean - threshold * std_dev) | (data > mean + threshold * std_dev)
    data[outliers] = np.nan
    valid_idx = np.where(~np.isnan(data))[0]
    interp_func = interp1d(valid_idx, data[valid_idx], bounds_error=False, fill_value="extrapolate")
    return interp_func(np.arange(len(data)))


def non_overlapping_average(data: np.ndarray, window_size: int) -> np.ndarray:
    """Compute non-overlapping moving average of 1D data.

    Args:
        data (np.ndarray): Input 1D signal.
        window_size (int): Number of samples per window.

    Returns:
        np.ndarray: Averaged signal.
    """
    return np.array([np.mean(data[i * window_size:(i + 1) * window_size]) for i in range(len(data) // window_size)])


def process_pupil_data(subject_id: str, version: str, threshold: float = 3) -> pd.DataFrame | None:
    """Load, clean, and average pupil data from MAT files.

    Args:
        subject_id (str): Subject ID.
        version (str): Recording version.
        threshold (float): Outlier threshold (default=3).

    Returns:
        pd.DataFrame or None: Cleaned and averaged pupil data, or None if file not found.
    """
    file_path = f'{DATASET_PATH}/{subject_id}/PupilData_{subject_id}_{version}.mat'
    data = safe_load_mat(file_path)

    if data is None:
        logging.error(f"Pupil data not found for {subject_id}_{version}")
        return None

    def process_dimension(pupil_data, timestamps):
        sampling_rate = len(timestamps) / (timestamps[-1] - timestamps[0])
        window_samples = int(3 * sampling_rate)  # Hardcoded window_sec = 3

        cleaned_data = replace_outliers_with_interpolation(pupil_data, threshold)
        avg_data = non_overlapping_average(cleaned_data, window_samples)
        avg_timestamps = timestamps[::window_samples][:len(avg_data)]

        return avg_timestamps, avg_data

    ts_2d, left_2d_avg = process_dimension(data['Pupil2D'][1], data['Pupil2D'][0])
    _, right_2d_avg = process_dimension(data['Pupil2D'][2], data['Pupil2D'][0])

    ts_3d, left_3d_avg = process_dimension(data['Pupil3D'][1], data['Pupil3D'][0])
    _, right_3d_avg = process_dimension(data['Pupil3D'][2], data['Pupil3D'][0])

    return pd.DataFrame({
        "Timestamp_2D": ts_2d,
        "Left_Pupil_2D_Avg": left_2d_avg,
        "Right_Pupil_2D_Avg": right_2d_avg,
        "Timestamp_3D": ts_3d,
        "Left_Pupil_3D_Avg": left_3d_avg,
        "Right_Pupil_3D_Avg": right_3d_avg
    })


def pupil_process(subject: str) -> None:
    """Main function to process and save pupil diameter data for a subject.

    Args:
        subject (str): Subject identifier in 'subjectID/version' format.

    Returns:
        None
    """
    subject_id, version = subject.split('/')[0], subject.split('/')[1].split('_')[-1]
    df_pupil = process_pupil_data(subject_id, version)

    if df_pupil is not None:
        save_csv(df_pupil, subject_id, version, 'pupil')
        logging.info(f"Processed and saved pupil data for {subject_id}_{version}")

