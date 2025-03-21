# simlsl_features.py: SIM_lslデータの特徴量抽出
from src.config import (
    DATASET_PATH, 
    INTRIM_CSV_PATH, 
    SAMPLE_RATE_SIMLSL,
    WINDOW_SIZE_SEC, 
    STEP_SIZE_SEC,
    WINDOW_SIZE_SAMPLE_SIMLSL,
    STEP_SIZE_SAMPLE_SIMLSL,
)

from src.utils.io.loaders import safe_load_mat, save_csv

import numpy as np
import pandas as pd
import os
import logging
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis
from scipy.interpolate import interp1d
from scipy.signal import lfilter

def extract_aref_features(signal, prefix=""):
    features = {}
    features[f'{prefix}Range'] = np.max(signal) - np.min(signal)
    features[f'{prefix}Standard Deviation'] = np.std(signal)
    features[f'{prefix}Energy'] = np.sum(signal ** 2)
    features[f'{prefix}Zero Crossing Rate'] = ((signal[:-1] * signal[1:]) < 0).sum() / len(signal)
    features[f'{prefix}First Quartile'] = np.percentile(signal, 25)
    features[f'{prefix}Second Quartile'] = np.median(signal)
    features[f'{prefix}Third Quartile'] = np.percentile(signal, 75)
    L = np.sum(np.sqrt(1 + np.diff(signal)**2))
    d = np.max(np.abs(signal - np.mean(signal)))
    features[f'{prefix}Katz Fractal Dimension'] = np.log10(len(signal)) / (np.log10(len(signal)) + np.log10(d/L) if d != 0 else 1)
    features[f'{prefix}Skewness'] = skew(signal)
    features[f'{prefix}Kurtosis'] = kurtosis(signal)
    #hist, bin_edges = np.histogram(signal, bins=10, density=True)
    #prob = hist / np.sum(hist)
    #features[f'{prefix}Shannon Entropy'] = -np.sum(prob * np.log2(prob + np.finfo(float).eps))
    freqs = fftfreq(len(signal), 1/SAMPLE_RATE_SIMLSL)
    spectrum = np.abs(fft(signal))**2
    freq_band = (0.5, 30)
    band_indices = np.where((freqs >= freq_band[0]) & (freqs <= freq_band[1]))
    features[f'{prefix}Frequency Variability'] = np.var(freqs[band_indices])
    spectral_prob = spectrum[band_indices] / np.sum(spectrum[band_indices])
    features[f'{prefix}Spectral Entropy'] = -np.sum(spectral_prob * np.log2(spectral_prob + np.finfo(float).eps))
    features[f'{prefix}Spectral Flux'] = np.sqrt(np.sum((np.diff(spectrum) ** 2)))
    features[f'{prefix}Center of Gravity of Frequency'] = np.sum(freqs[band_indices] * spectrum[band_indices]) / np.sum(spectrum[band_indices])
    features[f'{prefix}Dominant Frequency'] = freqs[np.argmax(spectrum)]
    features[f'{prefix}Average Value of PSD'] = np.mean(spectrum[band_indices])
    features[f'{prefix}Sample Entropy'] = 0#sample_entropy(signal) # for time save
    return features

# SIMlslデータを処理
def process_simlsl_data(signal1, signal2, signal3, signal4):
    steering_features, lat_accel_features, lane_offset_features, long_accel_features = [], [], [], []

    for start in range(0, len(signal1) - WINDOW_SIZE_SAMPLE_SIMLSL + 1, STEP_SIZE_SAMPLE_SIMLSL):
        #start = i * STEP_SIZE_SAMPLE_SIMLSL
        #end = start + WINDOW_SIZE_SAMPLE_SIMLSL
        segment1 = signal1[start:start + WINDOW_SIZE_SAMPLE_SIMLSL]
        segment2 = signal2[start:start + WINDOW_SIZE_SAMPLE_SIMLSL]
        segment3 = signal3[start:start + WINDOW_SIZE_SAMPLE_SIMLSL]
        segment4 = signal4[start:start + WINDOW_SIZE_SAMPLE_SIMLSL]
        steering_features.append(extract_aref_features(segment1, prefix="Steering_"))
        lat_accel_features.append(extract_aref_features(segment2, prefix="Lateral_"))
        lane_offset_features.append(extract_aref_features(segment3, prefix="LaneOffset_"))
        long_accel_features.append(extract_aref_features(segment4, prefix="LongAcc_"))
    return (pd.DataFrame(steering_features), pd.DataFrame(lat_accel_features),
            pd.DataFrame(lane_offset_features), pd.DataFrame(long_accel_features))

def extract_simlsl_features(signal, window_size, step_size, prefix=""):
    """
    Extract statistical features from SIM_lsl signal data.
    
    Args:
        signal (ndarray): Input signal.
        window_size (int): Window size in samples.
        step_size (int): Step size in samples.
        prefix (str): Prefix for feature names.

    Returns:
        dict: Extracted features.
    """
    features = {f"{prefix}_mean": [], f"{prefix}_std_dev": []}
    for start in range(0, len(signal) - window_size + 1, step_size):
        window = signal[start:start + window_size]
        features[f"{prefix}_mean"].append(np.mean(window))
        features[f"{prefix}_std_dev"].append(np.std(window))
    return features

# Function to extract features
def extract_features(data):
    features = {'std_dev': [], 'pred_error': [], 'gaussian_smooth': []}
    
    for start in range(0, len(data) - WINDOW_SIZE_SAMPLE_SIMLSL + 1, STEP_SIZE_SAMPLE_SIMLSL):
        window = data[start:start + WINDOW_SIZE_SAMPLE_SIMLSL]

        # Standard Deviation
        std_dev = np.std(window)
        features['std_dev'].append(std_dev)

        # Predicted Error (using second-order Taylor approximation)
        if len(window) > 3:
            pred_value = window[-3] + 2 * (window[-2] - window[-3])
            pred_error = abs(pred_value - window[-1])
            features['pred_error'].append(pred_error)
        else:
            features['pred_error'].append(np.nan)

        # Gaussian Smoothed Value
        gaussian_weights = np.exp(-0.5 * (np.linspace(-1, 1, len(window)) ** 2))
        gaussian_weights /= gaussian_weights.sum()
        gaussian_smooth = np.sum(window * gaussian_weights)
        features['gaussian_smooth'].append(gaussian_smooth)
        
    return features

## Main function to process each MAT file and save results to CSV
def smooth_std_pe_process(subject, model):
    subject_id, version = subject.split('/')[0], subject.split('/')[1].split('_')[-1]
    file_path = f"{DATASET_PATH}/{subject_id}/SIMlsl_{subject_id}_{version}.mat"
    # Load the MAT file
    #data = scipy.io.loadmat(file_path)
    data = safe_load_mat(file_path)
    if data is None:
        return  # エラー発生時にスキップ

    # Extract the relevant data from the loaded MAT file
    sim_lsl_data = data.get('SIM_lsl', np.array([]))

    # Check if the data structure is valid and contains the required rows
    if sim_lsl_data.size > 0 and sim_lsl_data.shape[0] >= 31:
        # Extract the time data and other relevant series
        time_data = sim_lsl_data[0]             # LSL Time
        steering_position = sim_lsl_data[29]    # Steering Wheel Position (rad)
        lat_acceleration = sim_lsl_data[19]     # Lateral Acceleration (m/s²)
        long_acceleration = sim_lsl_data[18]    # Longitudinal Acceleration (m/s²)
        lane_offset = sim_lsl_data[27]          # Lane Offset (m)

        # Check if time data is valid to calculate sample rate
        if time_data[-1] - time_data[0] == 0:
            #print(f"Warning: Invalid time data in {file_path}. Skipping this file.")
            logging.warning(f"Invalid time data in {file_path}. Skipping this file.")
            return

        # Extract features for each data series
        steering_features = extract_features(steering_position)
        lat_acc_features = extract_features(lat_acceleration)
        long_acc_features = extract_features(long_acceleration)
        lane_offset_features = extract_features(lane_offset)

        # Combine all features into a DataFrame
        all_features = {
            'Timestamp': time_data[::STEP_SIZE_SAMPLE_SIMLSL][:len(steering_features['std_dev'])],  # Resampled timestamps
            'steering_std_dev': steering_features['std_dev'],
            'steering_pred_error': steering_features['pred_error'],
            'steering_gaussian_smooth': steering_features['gaussian_smooth'],
            'lat_acc_std_dev': lat_acc_features['std_dev'],
            'lat_acc_pred_error': lat_acc_features['pred_error'],
            'lat_acc_gaussian_smooth': lat_acc_features['gaussian_smooth'],
            'long_acc_std_dev': long_acc_features['std_dev'],
            'long_acc_pred_error': long_acc_features['pred_error'],
            'long_acc_gaussian_smooth': long_acc_features['gaussian_smooth'],
            'lane_offset_std_dev': lane_offset_features['std_dev'],
            'lane_offset_pred_error': lane_offset_features['pred_error'],
            'lane_offset_gaussian_smooth': lane_offset_features['gaussian_smooth']
        }

        # Convert to DataFrame
        all_features_df = pd.DataFrame(all_features)

        save_csv(all_features_df, subject_id, version, 'smooth_std_pe', model)

    else:
        #print(f"The data structure in {file_path} is invalid or does not contain the required rows.")
        logging.warning(f"The data structure in {file_path} is invalid or does not contain the required rows.")

def time_freq_domain_process(subject, model): 
    subject_id, version = subject.split('/')[0], subject.split('/')[1].split('_')[-1]
    eeg_file = f"{DATASET_PATH}/{subject_id}/EEG_{subject_id}_{version}.mat"
    simlsl_file = f"{DATASET_PATH}/{subject_id}/SIMlsl_{subject_id}_{version}.mat"
    
    eeg_data = safe_load_mat(eeg_file)['rawEEG'][1:, :]
    if eeg_data is None:
        return 

    simlsl_data = safe_load_mat(simlsl_file)['SIM_lsl']
    if simlsl_data is None:
        return  
    
    simlsl_timestamps = simlsl_data[0, :]
    
    # SIMlslデータから各信号を抽出し、特徴量を計算
    steering_wheel_position = simlsl_data[29, :]
    lateral_acceleration = simlsl_data[19, :]
    lane_offset = simlsl_data[27, :]
    long_acceleration = simlsl_data[18, :]
    steering_features_df, lat_accel_features_df, lane_offset_features_df, long_accel_features_df = process_simlsl_data(
        steering_wheel_position, lateral_acceleration, lane_offset, long_acceleration)
    
    # SIMlslのタイムスタンプをステップサイズにリサンプリング
    step_size = len(simlsl_timestamps) // len(steering_features_df)
    resampled_timestamps = simlsl_timestamps[::STEP_SIZE_SAMPLE_SIMLSL][:len(steering_features_df)]
    #steering_features_df["Timestamp"] = resampled_timestamps
    # サンプル数の一致を確認し、CSVファイルに出力
    if len(steering_features_df) == len(lat_accel_features_df) == len(lane_offset_features_df) == len(long_accel_features_df):
        # 全データを1つのデータフレームに結合
        combined_df = pd.concat([steering_features_df, lat_accel_features_df, lane_offset_features_df, long_accel_features_df], axis=1)

        combined_df.insert(0, 'Timestamp', resampled_timestamps)
        save_csv(combined_df, subject_id, version, 'time_freq_domain', model)
    else:
        logging.warning(f"Data for {subject_id}_{version} saved as separate CSV files due to sample count mismatch.")
