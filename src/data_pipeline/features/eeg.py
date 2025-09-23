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
    """Retrieve window size and step size in samples for EEG data.

    This function accesses the ``MODEL_WINDOW_CONFIG`` to find the appropriate
    window duration and step for processing EEG signals, ensuring consistency
    with the model's requirements and the EEG sampling rate.

    Parameters
    ----------
    model : str
        Model name used to retrieve window parameters.

    Returns
    -------
    tuple of (int, int)
        - Window size in samples.
        - Step size in samples.
    """
    config = MODEL_WINDOW_CONFIG[model]
    window_samples = int(config["window_sec"] * SAMPLE_RATE_EEG)
    step_samples = int(config["step_sec"] * SAMPLE_RATE_EEG)
    return window_samples, step_samples


def bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, order: int = 5) -> np.ndarray:
    """Apply a Butterworth bandpass filter to a 1D signal.

    Parameters
    ----------
    data : ndarray
        Input 1D signal.
    lowcut : float
        Lower cutoff frequency in Hz.
    highcut : float
        Upper cutoff frequency in Hz.
    order : int, default=5
        Order of the Butterworth filter.

    Returns
    -------
    ndarray
        Filtered 1D signal.
    """
    nyquist = 0.5 * SAMPLE_RATE_EEG
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


def calculate_band_power(signal: np.ndarray, band: tuple[float, float]) -> float:
    """Calculate band power using Welch’s method.

    Parameters
    ----------
    signal : ndarray
        Input time-domain signal.
    band : tuple of (float, float)
        Frequency band (low, high) in Hz.

    Returns
    -------
    float
        Band power in the specified frequency range.
    """
    f, Pxx = welch(signal, SAMPLE_RATE_EEG, nperseg=1024)
    band_power = np.sum(Pxx[(f >= band[0]) & (f <= band[1])])
    return band_power


def load_eeg_data(subject: str) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Load raw EEG data and timestamps for a subject.

    Parameters
    ----------
    subject : str
        Subject identifier (format: ``"<id>_<version>"``).

    Returns
    -------
    tuple of (ndarray, ndarray) or (None, None)
        - Raw EEG data array (channels × time).
        - Corresponding timestamps array.
        Returns (None, None) if data is missing or invalid.
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
    """Segment EEG into windows and compute band power features.

    Parameters
    ----------
    eeg_data : ndarray
        EEG signals, shape (channels, time).
    timestamps : ndarray
        Time values corresponding to EEG samples.
    frequency_bands : dict
        Mapping from band name to (low, high) frequency range in Hz.
    model : str
        Model name used to determine window/step size.

    Returns
    -------
    tuple of (list, dict)
        - List of timestamps for each window.
        - Nested dictionary mapping channel → band → list of powers.
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
    """Run the full EEG feature extraction pipeline.

    This function loads raw EEG data, computes band powers per window,
    and saves the results as a CSV file.

    Parameters
    ----------
    subject : str
        Subject identifier (format: ``"<id>_<version>"``).
    model : str
        Model name to determine windowing strategy.

    Returns
    -------
    None
        Processed features are written to disk.
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

