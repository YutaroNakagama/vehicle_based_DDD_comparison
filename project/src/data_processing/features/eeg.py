from config import (
        WINDOW_SIZE_SEC, 
        STEP_SIZE_SEC,
        SAMPLE_RATE_EEG,
        WINDOW_SIZE_SAMPLE_EEG,
        STEP_SIZE_SAMPLE_EEG,
        DATASET_PATH,
        )
from src.utils.loaders import safe_load_mat, save_csv

import numpy as np
import pandas as pd
import logging
from scipy.signal import butter, filtfilt, welch
from sklearn.decomposition import FastICA

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def calculate_band_power(signal, band):
    """
    Calculates the power of a signal within a specific frequency band.
    
    Args:
        signal (ndarray): Input signal.
        band (tuple): Frequency range as (low, high).
        fs (float): Sampling frequency.

    Returns:
        float: Band power of the signal.
    """
    f, Pxx = welch(signal, SAMPLE_RATE_EEG, nperseg=1024)
    band_power = np.sum(Pxx[(f >= band[0]) & (f <= band[1])])
    return band_power

# Functions for bandpass filtering and power calculation
def bandpass_filter(data, lowcut, highcut, order=5):
    nyquist = 0.5 * SAMPLE_RATE_EEG
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def calculate_band_power(signal, band):
    f, Pxx = welch(signal, SAMPLE_RATE_EEG, nperseg=1024)
    band_power = np.sum(Pxx[(f >= band[0]) & (f <= band[1])])
    return band_power

# バンドパワーを計算
def bandpower(data, band):
    fmin, fmax = band
    freqs, psd = np.fft.rfftfreq(len(data), 1/SAMPLE_RATE_EEG), np.abs(np.fft.rfft(data))**2
    band_power = np.sum(psd[(freqs >= fmin) & (freqs <= fmax)])
    return band_power

# Theta/Beta比を計算
def theta_beta_ratio(signal, fs):
    theta_band = (4, 8)
    beta_band = (13, 30)
    theta_power = bandpower(signal, theta_band)
    beta_power = bandpower(signal, beta_band)
    return theta_power / beta_power if beta_power != 0 else 0

def eeg_process(subject):
    subject_id, version = subject.split('/')[0], subject.split('/')[1].split('_')[-1]
    frequency_bands = {
        "Delta (0.5-4 Hz)": (0.5, 4),
        "Theta (4-8 Hz)": (4, 8),
        "Alpha (8-13 Hz)": (8, 13),
        "Beta (13-30 Hz)": (13, 30),
        "Gamma (30-100 Hz)": (30, 100)
    }
    try:
        # Extract subject name and version from the path (e.g., S0210/S0210_2 -> S0210_2)
        subject_name_version = subject.split('/')[-1]
        subject_name = subject.split('/')[-2]
        mat_file_name = f'EEG_{subject_name_version}.mat'
        
        # Load EEG data file
        mat_data = safe_load_mat(f'{DATASET_PATH}/{subject_name}/{mat_file_name}')
        if mat_data is None:
            return  # エラー発生時にスキップ
        eeg_data = mat_data.get('rawEEG')
    
        # Extract timestamps (first row)
        timestamps = eeg_data[0, :]
    
        # Initialize storage for each channel
        channel_band_powers = {ch: {band: [] for band in frequency_bands.keys()} for ch in range(1, eeg_data.shape[0])}
        timestamp_windows = []
    
        # Calculate band powers for each 1-minute window
        num_windows = int(eeg_data.shape[1] / WINDOW_SIZE_SAMPLE_EEG)
        #for w in range(num_windows):
        for start in range(0, eeg_data.shape[1] - WINDOW_SIZE_SAMPLE_EEG + 1, STEP_SIZE_SAMPLE_EEG):
            window_data = eeg_data[:, start:start + WINDOW_SIZE_SAMPLE_EEG]
    
            # Record start timestamp for each window
            timestamp_windows.append(timestamps[start])
            for ch in range(1, eeg_data.shape[0]):  # For each EEG channel (excluding timestamps)
                for band_name, (low, high) in frequency_bands.items():
                    band_power = calculate_band_power(
                            bandpass_filter(window_data[ch, :], low, high), 
                            (low, high), 
                            )
                    channel_band_powers[ch][band_name].append(band_power)
    
        # Convert results to a DataFrame, including timestamps
        data_for_csv = {'Timestamp': timestamp_windows}
        for ch in range(1, eeg_data.shape[0]):
            for band_name in frequency_bands.keys():
                column_name = f"Channel_{ch}_{band_name}"
                data_for_csv[column_name] = channel_band_powers[ch][band_name]
        df_results = pd.DataFrame(data_for_csv)
        
        save_csv(df_results, subject_id, version, 'eeg')
        
    except Exception as e:
        #print(f"Failed to process {subject_name_version}: {e}")
        logging.error(f"Failed to process {subject_name_version}: {e}")
