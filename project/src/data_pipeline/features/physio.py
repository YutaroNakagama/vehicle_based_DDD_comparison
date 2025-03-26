import pandas as pd
import numpy as np
import logging
from scipy.interpolate import interp1d

from src.utils.io.loaders import safe_load_mat, save_csv
from src.config import DATASET_PATH, MODEL_WINDOW_CONFIG 

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_physio_window_sec(model):
    return MODEL_WINDOW_CONFIG[model]["window_sec"]


def calculate_perclos(blinks, window_size_sec):
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


def process_physio_data(physio_content, window_size_sec):
    physio_df = pd.DataFrame({
        "Timestamp": physio_content[0],
        "Blood_Pressure": physio_content[1],
        "Skin_Conductance": physio_content[2],
        "Respiration": physio_content[3],
        "Oxygen_Saturation": physio_content[4]
    })

    physio_df['Window'] = ((physio_df['Timestamp'] - physio_df['Timestamp'].min()) // window_size_sec).astype(int)
    return physio_df.groupby('Window').mean().reset_index()


def calculate_and_save_perclos_physio_combined(blink_data_path, physio_data_path, window_size_sec):
    blink_data = safe_load_mat(blink_data_path)
    physio_data = safe_load_mat(physio_data_path)

    if blink_data is None or physio_data is None:
        logging.error("Missing Blink or Physio data, skipping...")
        return None

    perclos_df = calculate_perclos(blink_data['Blinks'], window_size_sec)
    perclos_df['Window'] = ((perclos_df['Timestamp'] - perclos_df['Timestamp'].min()) // window_size_sec).astype(int)

    physio_resampled = process_physio_data(physio_data['PhysioData'], window_size_sec)

    return pd.merge(physio_resampled, perclos_df, on="Window", how="inner")


def perclos_process(subject):
    subject_id, version = subject.split('/')[0], subject.split('/')[1].split('_')[-1]

    blink_data_file = f"{DATASET_PATH}/{subject_id}/Blinks_{subject_id}_{version}.mat"
    physio_data_file = f"{DATASET_PATH}/{subject_id}/Physio_{subject_id}_{version}.mat"

    df_combined = calculate_and_save_perclos_physio_combined(blink_data_file, physio_data_file)
    if df_combined is not None:
        save_csv(df_combined, subject_id, version, 'perclos')
        logging.info(f"Processed and saved PERCLOS & Physio data for {subject_id}_{version}")


# Pupil processing functions

def replace_outliers_with_interpolation(data, threshold=3):
    mean, std_dev = np.nanmean(data), np.nanstd(data)
    outliers = (data < mean - threshold * std_dev) | (data > mean + threshold * std_dev)
    data[outliers] = np.nan
    valid_idx = np.where(~np.isnan(data))[0]
    interp_func = interp1d(valid_idx, data[valid_idx], bounds_error=False, fill_value="extrapolate")
    return interp_func(np.arange(len(data)))


def non_overlapping_average(data, window_size):
    return np.array([np.mean(data[i * window_size:(i + 1) * window_size]) for i in range(len(data) // window_size)])


def process_pupil_data(subject_id, version, threshold=3):
    file_path = f'{DATASET_PATH}/{subject_id}/PupilData_{subject_id}_{version}.mat'
    data = safe_load_mat(file_path)

    if data is None:
        logging.error(f"Pupil data not found for {subject_id}_{version}")
        return None

    def process_dimension(pupil_data, timestamps):
        sampling_rate = len(timestamps) / (timestamps[-1] - timestamps[0])
        window_samples = int(WINDOW_SIZE_SEC * sampling_rate)

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


def pupil_process(subject):
    subject_id, version = subject.split('/')[0], subject.split('/')[1].split('_')[-1]
    df_pupil = process_pupil_data(subject_id, version)

    if df_pupil is not None:
        save_csv(df_pupil, subject_id, version, 'pupil')
        logging.info(f"Processed and saved pupil data for {subject_id}_{version}")


