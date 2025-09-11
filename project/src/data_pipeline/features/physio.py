"""Physiological and Pupil Data Preprocessing for Driver Drowsiness Detection.

This module contains functions for processing raw physiological signals (blood pressure,
skin conductance, respiration, oxygen saturation) and pupil diameter data, typically
loaded from MAT files. It focuses on extracting meaningful features by calculating
PERCLOS (Percentage of Eye Closure) and aggregating physiological signals over
time windows. The processed data is then saved as CSV files for further use in
driver drowsiness detection models.

Functions:
    - `get_physio_window_sec()`: Retrieves window size for physiological data.
    - `calculate_perclos()`: Computes PERCLOS from blink event data.
    - `process_physio_data()`: Aggregates physiological signals over windows.
    - `calculate_and_save_perclos_physio_combined()`: Combines PERCLOS and physiological data.
    - `perclos_process()`: Main function to extract and save combined PERCLOS and physiological features.
    - `replace_outliers_with_interpolation()`: Handles outliers in time series data.
    - `non_overlapping_average()`: Computes non-overlapping moving averages.
    - `process_pupil_data()`: Loads, cleans, and averages pupil data.
    - `pupil_process()`: Main function to process and save pupil diameter data.
"""

import pandas as pd
import numpy as np
import logging
from scipy.interpolate import interp1d

from src.utils.io.loaders import safe_load_mat, save_csv
from src.config import DATASET_PATH, MODEL_WINDOW_CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_physio_window_sec(model: str) -> float:
    """Retrieve window size in seconds for physiological data.

    Parameters
    ----------
    model : str
        Model name (e.g., ``"Lstm"``, ``"SvmW"``).

    Returns
    -------
    float
        Window size in seconds.
    """
    return MODEL_WINDOW_CONFIG[model]["window_sec"]


def calculate_perclos(blinks: np.ndarray, window_size_sec: float) -> pd.DataFrame:
    """Calculate PERCLOS (Percentage of Eye Closure) for each window.

    Parameters
    ----------
    blinks : ndarray
        Blink event data (2D array with start times and durations).
    window_size_sec : float
        Duration of each analysis window in seconds.

    Returns
    -------
    DataFrame
        Table with columns:
        - ``Timestamp`` : start time of each window
        - ``PERCLOS`` : percentage of eye closure
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
    """Aggregate physiological signals into non-overlapping windows.

    Parameters
    ----------
    physio_content : ndarray
        Array where the first row is timestamps and subsequent rows are signals.
    window_size_sec : float
        Duration of each non-overlapping window.

    Returns
    -------
    DataFrame
        Window-averaged physiological features with a ``Window`` column.
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
    """Combine and align PERCLOS and physiological data by window.

    Parameters
    ----------
    blink_data_path : str
        Path to blink event MAT file.
    physio_data_path : str
        Path to physiological signal MAT file.
    window_size_sec : float
        Window size in seconds.

    Returns
    -------
    DataFrame or None
        Combined dataset if both files exist, else ``None``.
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
    """Extract and save combined PERCLOS and physiological features.

    Parameters
    ----------
    subject : str
        Subject identifier (format: ``"<id>_<version>"``).

    Returns
    -------
    None
        Processed features are saved to CSV.
    """
    parts = subject.split('_')
    if len(parts) != 2:
        logging.error(f"Unexpected subject format: {subject}")
        return

    subject_id, version = parts

    blink_data_file = f"{DATASET_PATH}/{subject_id}/Blinks_{subject_id}_{version}.mat"
    physio_data_file = f"{DATASET_PATH}/{subject_id}/Physio_{subject_id}_{version}.mat"

    df_combined = calculate_and_save_perclos_physio_combined(
        blink_data_file,
        physio_data_file,
        get_physio_window_sec(subject_id)
    )
    if df_combined is not None:
        save_csv(df_combined, subject_id, version, 'perclos')


# ----- Pupil Processing -----

def replace_outliers_with_interpolation(data: np.ndarray, threshold: float = 3) -> np.ndarray:
    """Replace outliers in a signal with interpolated values.

    Parameters
    ----------
    data : ndarray
        Input 1D time series.
    threshold : float, default=3
        Z-score threshold for outlier detection.

    Returns
    -------
    ndarray
        Signal with outliers replaced by interpolated values.
    """
    mean, std_dev = np.nanmean(data), np.nanstd(data)
    outliers = (data < mean - threshold * std_dev) | (data > mean + threshold * std_dev)
    data[outliers] = np.nan
    valid_idx = np.where(~np.isnan(data))[0]
    interp_func = interp1d(valid_idx, data[valid_idx], bounds_error=False, fill_value="extrapolate")
    return interp_func(np.arange(len(data)))


def non_overlapping_average(data: np.ndarray, window_size: int) -> np.ndarray:
    """Compute non-overlapping moving average of a signal.

    Parameters
    ----------
    data : ndarray
        Input 1D signal.
    window_size : int
        Window size in samples.

    Returns
    -------
    ndarray
        Averaged values for each window.
    """
    return np.array([np.mean(data[i * window_size:(i + 1) * window_size]) for i in range(len(data) // window_size)])


def process_pupil_data(subject_id: str, version: str, threshold: float = 3) -> pd.DataFrame | None:
    """Load, clean, and average pupil diameter data.

    Parameters
    ----------
    subject_id : str
        Subject identifier (e.g., ``"S0120"``).
    version : str
        Recording version.
    threshold : float, default=3
        Z-score threshold for outlier detection.

    Returns
    -------
    DataFrame or None
        Cleaned pupil data with left/right 2D and 3D signals,
        or ``None`` if file not found.
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
    """Process and save pupil diameter data.

    Parameters
    ----------
    subject : str
        Subject identifier (format: ``"<id>_<version>"``).

    Returns
    -------
    None
        Processed pupil features are saved to CSV.
    """
    parts = subject.split('_')
    if len(parts) != 2:
        logging.error(f"Unexpected subject format: {subject}")
        return

    subject_id, version = parts
    df_pupil = process_pupil_data(subject_id, version)

    if df_pupil is not None:
        save_csv(df_pupil, subject_id, version, 'pupil')
        logging.info(f"Processed and saved pupil data for {subject_id}_{version}")

