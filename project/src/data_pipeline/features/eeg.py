from src.config import (
    WINDOW_SIZE_SAMPLE_EEG,
    STEP_SIZE_SAMPLE_EEG,
    SAMPLE_RATE_EEG,
    DATASET_PATH,
)
from src.utils.io.loaders import safe_load_mat, save_csv

import numpy as np
import pandas as pd
import logging
from scipy.signal import butter, filtfilt, welch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def bandpass_filter(data, lowcut, highcut, order=5):
    nyquist = 0.5 * SAMPLE_RATE_EEG
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


def calculate_band_power(signal, band):
    f, Pxx = welch(signal, SAMPLE_RATE_EEG, nperseg=1024)
    band_power = np.sum(Pxx[(f >= band[0]) & (f <= band[1])])
    return band_power


def load_eeg_data(subject):
    subject_name, subject_version = subject.split('/')
    mat_file_name = f'EEG_{subject_version}.mat'
    mat_data = safe_load_mat(f'{DATASET_PATH}/{subject_name}/{mat_file_name}')
    if mat_data is None:
        logging.error(f"EEG data not found for {subject}")
        return None, None

    eeg_data = mat_data.get('rawEEG')
    timestamps = eeg_data[0, :]
    return eeg_data, timestamps


def process_eeg_windows(eeg_data, timestamps, frequency_bands):
    channel_band_powers = {ch: {band: [] for band in frequency_bands} for ch in range(1, eeg_data.shape[0])}
    timestamp_windows = []

    for start in range(0, eeg_data.shape[1] - WINDOW_SIZE_SAMPLE_EEG + 1, STEP_SIZE_SAMPLE_EEG):
        window_data = eeg_data[:, start:start + WINDOW_SIZE_SAMPLE_EEG]
        timestamp_windows.append(timestamps[start])

        for ch in range(1, eeg_data.shape[0]):
            for band_name, (low, high) in frequency_bands.items():
                filtered_signal = bandpass_filter(window_data[ch, :], low, high)
                power = calculate_band_power(filtered_signal, (low, high))
                channel_band_powers[ch][band_name].append(power)

    return timestamp_windows, channel_band_powers


def eeg_process(subject, model):
    subject_id, version = subject.split('/')[0], subject.split('/')[1].split('_')[-1]

    frequency_bands = {
        "Delta (0.5-4 Hz)": (0.5, 4),
        "Theta (4-8 Hz)": (4, 8),
        "Alpha (8-13 Hz)": (8, 13),
        "Beta (13-30 Hz)": (13, 30),
        "Gamma (30-100 Hz)": (30, 100)
    }

    eeg_data, timestamps = load_eeg_data(subject)
    if eeg_data is None:
        return

    timestamp_windows, channel_band_powers = process_eeg_windows(eeg_data, timestamps, frequency_bands)

    data_for_csv = {'Timestamp': timestamp_windows}
    for ch in channel_band_powers:
        for band_name in frequency_bands:
            column_name = f"Channel_{ch}_{band_name}"
            data_for_csv[column_name] = channel_band_powers[ch][band_name]

    df_results = pd.DataFrame(data_for_csv)
    save_csv(df_results, subject_id, version, 'eeg', model)
    logging.info(f"EEG features saved for {subject_id}_{version} [{model}].")

