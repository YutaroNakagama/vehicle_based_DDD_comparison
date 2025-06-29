"""EEG Feature Extraction Pipeline for Driver Drowsiness Detection.

This module provides a comprehensive pipeline for processing Electroencephalography (EEG) data.
It includes functionalities for loading raw EEG signals, applying bandpass filtering to isolate
specific frequency components, computing band power within standard EEG frequency ranges
(Delta, Theta, Alpha, Beta, Gamma), and saving the extracted features for use in
driver drowsiness detection models.

The module supports window-based feature extraction and parallel processing for efficiency.
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
from joblib import Parallel, delayed

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_eeg_window_params(model: str) -> tuple[int, int]:
    """Retrieves the window size and step size in samples for EEG data
    based on the specified model configuration.

    This function accesses the `MODEL_WINDOW_CONFIG` to find the appropriate
    window duration and step for processing EEG signals, ensuring consistency
    with the model's requirements and the EEG sampling rate.

    Args:
        model (str): The name of the model for which to retrieve window parameters.
                     This key is used to look up the configuration in `MODEL_WINDOW_CONFIG`.

    Returns:
        tuple[int, int]: A tuple containing two integers:
                         - The window size in samples.
                         - The step size in samples.
    """
    config = MODEL_WINDOW_CONFIG[model]
    window_samples = int(config["window_sec"] * SAMPLE_RATE_EEG)
    step_samples = int(config["step_sec"] * SAMPLE_RATE_EEG)
    return window_samples, step_samples


def bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, order: int = 5) -> np.ndarray:
    """Applies a Butterworth bandpass filter to a 1D signal.

    This function filters the input signal to retain frequencies within a specified
    range, which is crucial for isolating specific EEG rhythms (e.g., Alpha, Beta).

    Args:
        data (np.ndarray): The input 1D signal to be filtered.
        lowcut (float): The lower cutoff frequency of the bandpass filter in Hz.
        highcut (float): The upper cutoff frequency of the bandpass filter in Hz.
        order (int): The order of the Butterworth filter. A higher order results
                     in a steeper roll-off. Defaults to 5.

    Returns:
        np.ndarray: The filtered 1D signal.
    """
    nyquist = 0.5 * SAMPLE_RATE_EEG
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


def calculate_band_power(signal: np.ndarray, band: tuple[float, float]) -> float:
    """Calculates the power of a signal within a specific frequency band using Welch's method.

    Band power is a common feature in EEG analysis, representing the energy
    contributed by oscillations within a particular frequency range.

    Args:
        signal (np.ndarray): The input time-domain signal (e.g., a single EEG channel's data).
        band (tuple[float, float]): A tuple specifying the frequency range (low_frequency, high_frequency) in Hz.

    Returns:
        float: The calculated band power within the specified frequency range.
    """
    f, Pxx = welch(signal, SAMPLE_RATE_EEG, nperseg=1024)
    band_power = np.sum(Pxx[(f >= band[0]) & (f <= band[1])])
    return band_power


def load_eeg_data(subject: str) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Loads raw EEG data and corresponding timestamps from a .mat file for a given subject.

    This function handles the file path construction and safe loading of the .mat file.
    It extracts the raw EEG signals and their associated timestamps.

    Args:
        subject (str): The subject identifier in the format 'subjectID_version' (e.g., 'S0120_2').

    Returns:
        tuple[np.ndarray, np.ndarray] | tuple[None, None]: A tuple containing:
            - The raw EEG data as a NumPy array (channels x time).
            - The timestamps as a NumPy array.
            Returns (None, None) if the file cannot be loaded or the expected data is not found.
    """
    parts = subject.split('_')
    if len(parts) != 2:
        logging.error(f"Unexpected subject format: {subject}")
        return None, None

    subject_id, version = parts
    mat_file_name = f'EEG_{subject_id}_{version}.mat'
    mat_data = safe_load_mat(f'{DATASET_PATH}/{subject_id}/{mat_file_name}')
    if mat_data is None:
        logging.error(f"EEG data not found for {subject}")
        return None, None

    eeg_data = mat_data.get('rawEEG')
    timestamps = eeg_data[0, :] if eeg_data is not None else None
    return eeg_data, timestamps


def process_eeg_windows(eeg_data: np.ndarray, timestamps: np.ndarray,
                        frequency_bands: dict, model: str) -> tuple[list, dict]:
    """Segments EEG data into windows and computes band power features for each window.

    This function iterates through the EEG data using a sliding window approach,
    applies bandpass filtering for specified frequency bands, calculates the power
    within each band for every EEG channel, and collects timestamps for each window.
    Parallel processing is used to speed up feature extraction.

    Args:
        eeg_data (np.ndarray): The EEG signals, typically with shape (channels, time).
        timestamps (np.ndarray): Time values corresponding to the EEG samples.
        frequency_bands (dict): A dictionary mapping band names (str) to their
                                frequency ranges (tuple[float, float]) in Hz.
        model (str): The model name, used to determine the window size and step size.

    Returns:
        tuple[list, dict]: A tuple containing:
            - A list of timestamps, where each timestamp corresponds to the start of a processed window.
            - A dictionary where keys are channel indices and values are another dictionary
              mapping frequency band names to lists of band power values across all windows.
    """

    window_size, step_size = get_eeg_window_params(model)
    num_ch = eeg_data.shape[0] - 1

    window_starts = range(0, eeg_data.shape[1] - window_size + 1, step_size)

    def process_window(start):
        window_data = eeg_data[:, start:start + window_size]
        ts = timestamps[start]
        powers = []
        for ch in range(1, eeg_data.shape[0]):
            band_powers = []
            for band_name, (low, high) in frequency_bands.items():
                filtered_signal = bandpass_filter(window_data[ch, :], low, high)
                power = calculate_band_power(filtered_signal, (low, high))
                band_powers.append(power)
            powers.append(band_powers)
        return ts, powers

    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(process_window)(start) for start in window_starts
    )

    timestamp_windows = []
    channel_band_powers = {ch: {band: [] for band in frequency_bands} for ch in range(1, eeg_data.shape[0])}
    for ts, powers in results:
        timestamp_windows.append(ts)
        for ch_idx, band_powers in enumerate(powers):
            for b_idx, band_name in enumerate(frequency_bands):
                channel_band_powers[ch_idx + 1][band_name].append(band_powers[b_idx])

    return timestamp_windows, channel_band_powers

def eeg_process(subject: str, model: str) -> None:
    """Main function to process EEG data for a single subject and save band power features.

    This function orchestrates the entire EEG feature extraction pipeline:
    loading raw EEG data, defining frequency bands, processing EEG signals
    within sliding windows to compute band power for each channel, and finally
    saving the extracted features into a CSV file.

    Args:
        subject (str): The subject identifier in the format 'subjectID_version' (e.g., 'S0120_2').
        model (str): The model name, used to determine the windowing strategy for EEG processing.

    Returns:
        None: The function saves the processed features to a CSV file and does not return any value.
    """
    parts = subject.split('_')
    if len(parts) != 2:
        logging.error(f"Unexpected subject format: {subject}")
        return

    subject_id, version = parts

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

