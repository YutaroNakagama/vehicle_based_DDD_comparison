"""EEG feature extraction pipeline for driver drowsiness detection.

This module includes functions for loading EEG data, applying bandpass filtering,
computing band power in standard frequency ranges, and saving the results.
"""

from src.config import (
    MODEL_WINDOW_CONFIG,
    SAMPLE_RATE_EEG,
    DATASET_PATH,
)
from src.utils.io.loaders import safe_load_mat, save_csv

import numpy as np
import pandas as pd
import logging
from scipy.signal import butter, filtfilt, welch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_eeg_window_params(model: str) -> tuple[int, int]:
    """Get the window and step size in samples for the given model.

    Args:
        model (str): The model name to lookup window config.

    Returns:
        tuple[int, int]: Window size and step size in samples.
    """
    config = MODEL_WINDOW_CONFIG[model]
    window_samples = int(config["window_sec"] * SAMPLE_RATE_EEG)
    step_samples = int(config["step_sec"] * SAMPLE_RATE_EEG)
    return window_samples, step_samples


def bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, order: int = 5) -> np.ndarray:
    """Apply a Butterworth bandpass filter to a 1D signal.

    Args:
        data (np.ndarray): Input signal.
        lowcut (float): Low frequency cutoff in Hz.
        highcut (float): High frequency cutoff in Hz.
        order (int): Filter order. Default is 5.

    Returns:
        np.ndarray: Filtered signal.
    """
    nyquist = 0.5 * SAMPLE_RATE_EEG
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


def calculate_band_power(signal: np.ndarray, band: tuple[float, float]) -> float:
    """Calculate power in a specific frequency band using Welch's method.

    Args:
        signal (np.ndarray): Input time-domain signal.
        band (tuple[float, float]): Frequency range (low, high) in Hz.

    Returns:
        float: Band power within the specified frequency range.
    """
    f, Pxx = welch(signal, SAMPLE_RATE_EEG, nperseg=1024)
    band_power = np.sum(Pxx[(f >= band[0]) & (f <= band[1])])
    return band_power


def load_eeg_data(subject: str) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Load EEG data from a .mat file for a given subject.

    Args:
        subject (str): Subject identifier in the format 'subjectID/version'.

    Returns:
        tuple: (EEG data as np.ndarray, timestamps as np.ndarray), or (None, None) if loading fails.
    """
    subject_name, subject_version = subject.split('/')
    mat_file_name = f'EEG_{subject_version}.mat'
    mat_data = safe_load_mat(f'{DATASET_PATH}/{subject_name}/{mat_file_name}')
    if mat_data is None:
        logging.error(f"EEG data not found for {subject}")
        return None, None

    eeg_data = mat_data.get('rawEEG')
    timestamps = eeg_data[0, :]
    return eeg_data, timestamps


def process_eeg_windows(eeg_data: np.ndarray, timestamps: np.ndarray,
                        frequency_bands: dict, model: str) -> tuple[list, dict]:
    """Segment EEG data and compute band power features for each window.

    Args:
        eeg_data (np.ndarray): EEG signals, shape (channels, time).
        timestamps (np.ndarray): Time values corresponding to EEG samples.
        frequency_bands (dict): Mapping of band name to (low, high) Hz.
        model (str): Model name used to determine window config.

    Returns:
        tuple: (List of timestamps per window, dict of band power features per channel).
    """
    window_size, step_size = get_eeg_window_params(model)
    channel_band_powers = {ch: {band: [] for band in frequency_bands} for ch in range(1, eeg_data.shape[0])}
    timestamp_windows = []

    for start in range(0, eeg_data.shape[1] - window_size + 1, step_size):
        window_data = eeg_data[:, start:start + window_size]
        timestamp_windows.append(timestamps[start])

        for ch in range(1, eeg_data.shape[0]):
            for band_name, (low, high) in frequency_bands.items():
                filtered_signal = bandpass_filter(window_data[ch, :], low, high)
                power = calculate_band_power(filtered_signal, (low, high))
                channel_band_powers[ch][band_name].append(power)

    return timestamp_windows, channel_band_powers


def eeg_process(subject: str, model: str) -> None:
    """Process EEG data for a single subject and save band power features.

    Args:
        subject (str): Subject identifier in the format 'subjectID/version'.
        model (str): Model name used to determine windowing strategy.

    Returns:
        None
    """
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

    timestamp_windows, channel_band_powers = process_eeg_windows(eeg_data, timestamps, frequency_bands, model)

    data_for_csv = {'Timestamp': timestamp_windows}
    for ch in channel_band_powers:
        for band_name in frequency_bands:
            column_name = f"Channel_{ch}_{band_name}"
            data_for_csv[column_name] = channel_band_powers[ch][band_name]

    df_results = pd.DataFrame(data_for_csv)
    save_csv(df_results, subject_id, version, 'eeg', model)

