"""Feature extraction for vehicle dynamics (SIMlsl) signals.

This module includes functions to extract statistical and time-frequency
domain features from vehicle dynamics data such as steering angle,
acceleration, and lane offset.

Supports jittering-based data augmentation and multi-window sliding
feature extraction for machine learning models.
"""

import numpy as np
import pandas as pd
import logging
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis
from numba import njit
from joblib import Parallel, delayed

from src.utils.io.loaders import safe_load_mat, save_csv
from src.utils.domain_generalization.jitter import jittering
from src.config import (
    DATASET_PATH,
    SAMPLE_RATE_SIMLSL,
    MODEL_WINDOW_CONFIG,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@njit
def _count_similar_templates(signal, m, r):
    """
    Counts the number of similar template pairs for Sample Entropy calculation.
    This is a helper function optimized with Numba for performance.

    Args:
        signal (np.ndarray): The input signal array.
        m (int): The length of the template.
        r (float): The tolerance for similarity.

    Returns:
        int: The count of similar template pairs.
    """
    N = len(signal)
    count = 0
    for i in range(N - m):
        for j in range(i + 1, N - m + 1):
            if np.all(np.abs(signal[i:i + m] - signal[j:j + m]) <= r):
                count += 1
    return count

def sample_entropy(signal, m=2, r=None):
    """Calculates the Sample Entropy of a given signal.

    Sample Entropy is a measure of complexity and predictability of time-series data.
    A lower value indicates more regularity and predictability.

    Args:
        signal (np.ndarray): The input 1D signal.
        m (int): The length of the template vectors. Defaults to 2.
        r (float, optional): The tolerance for accepting matches. If None, it defaults
                             to 0.2 times the standard deviation of the signal.

    Returns:
        float: The Sample Entropy value, or np.nan if calculation is not possible
               (e.g., signal too short, or zero standard deviation).
    """
    signal = np.asarray(signal, dtype=np.float64)
    N = len(signal)
    if r is None:
        r = 0.2 * np.std(signal)
    if N <= m + 1 or r == 0:
        return np.nan

    try:
        A = _count_similar_templates(signal, m + 1, r)
        B = _count_similar_templates(signal, m, r)
        if B == 0 or A == 0:
            return np.nan
        return -np.log(A / B)
    except Exception:
        return np.nan

def extract_statistical_features(signal: np.ndarray, prefix: str = "") -> dict:
    """Extracts a comprehensive set of statistical and spectral features from a 1D signal.

    Features include range, standard deviation, energy, zero-crossing rate, quartiles,
    median, skewness, kurtosis, frequency variance, spectral entropy, spectral flux,
    frequency center of gravity (COG), dominant frequency, average Power Spectral Density (PSD),
    and Sample Entropy.

    Args:
        signal (np.ndarray): The input 1D signal from which to extract features.
        prefix (str): A string prefix to add to each feature name (e.g., 'Steering_').
                      Defaults to an empty string.

    Returns:
        dict: A dictionary where keys are feature names (prefixed) and values are the
              corresponding extracted feature values.
    """
    freqs = fftfreq(len(signal), 1 / SAMPLE_RATE_SIMLSL)
    spectrum = np.abs(fft(signal)) ** 2
    band = (freqs >= 0.5) & (freqs <= 30)
    band_sum = np.sum(spectrum[band]) + np.finfo(float).eps

    features = {
        f'{prefix}Range': np.ptp(signal),
        f'{prefix}StdDev': np.std(signal),
        f'{prefix}Energy': np.sum(signal ** 2),
        f'{prefix}ZeroCrossingRate': np.mean(np.diff(np.sign(signal)) != 0),
        f'{prefix}Quartile25': np.percentile(signal, 25),
        f'{prefix}Median': np.median(signal),
        f'{prefix}Quartile75': np.percentile(signal, 75),
        f'{prefix}Skewness': skew(signal) if np.std(signal) > 0 else 0,
        f'{prefix}Kurtosis': kurtosis(signal) if np.std(signal) > 0 else 0,
        f'{prefix}FreqVar': np.var(freqs[band]),
        f'{prefix}SpectralEntropy': -np.sum((spectrum[band] / band_sum) * np.log2((spectrum[band] / band_sum) + np.finfo(float).eps)),
        f'{prefix}SpectralFlux': np.sqrt(np.sum(np.diff(spectrum) ** 2)),
        f'{prefix}FreqCOG': np.sum(freqs[band] * spectrum[band]) / band_sum if band_sum > 0 else 0,
        f'{prefix}DominantFreq': freqs[np.argmax(spectrum)],
        f'{prefix}AvgPSD': np.mean(spectrum[band]),
        f'{prefix}SampleEntropy': sample_entropy(signal, m=2, r=0.2*np.std(signal)),
    }

    return features


def get_simlsl_window_params(model: str) -> tuple[int, int]:
    """Retrieves the window size and step size in samples for SIMlsl data
    based on the specified model configuration.

    Args:
        model (str): The name of the model for which to retrieve window parameters.
                     This key is used to look up the configuration in `MODEL_WINDOW_CONFIG`.

    Returns:
        tuple[int, int]: A tuple containing two integers:
                         - The window size in samples.
                         - The step size in samples.
    """
    config = MODEL_WINDOW_CONFIG[model]
    window_samples = int(config["window_sec"] * SAMPLE_RATE_SIMLSL)
    step_samples = int(config["step_sec"] * SAMPLE_RATE_SIMLSL)
    return window_samples, step_samples

def process_simlsl_data(signals: list[np.ndarray], prefixes: list[str], model: str) -> pd.DataFrame:
    """Processes a list of SIMlsl signals by applying a sliding window and extracting features.

    For each signal, statistical and spectral features are extracted from overlapping windows.
    The processing is parallelized for efficiency.

    Args:
        signals (list[np.ndarray]): A list of 1D NumPy arrays, where each array is a SIMlsl signal.
        prefixes (list[str]): A list of string prefixes corresponding to each signal,
                              used for naming the extracted features.
        model (str): The model name, used to determine window size and step size.

    Returns:
        pd.DataFrame: A DataFrame where each row represents a window and columns are the extracted features.
    """
    window_size, step_size = get_simlsl_window_params(model)
    starts = range(0, len(signals[0]) - window_size + 1, step_size)

    def process_one_window(start):
        window_features = {}
        for signal, prefix in zip(signals, prefixes):
            segment = signal[start:start + window_size]
            window_features.update(extract_statistical_features(segment, prefix))
        return window_features

    features_list = Parallel(n_jobs=-1, prefer="threads")(
        delayed(process_one_window)(start) for start in starts
    )

    return pd.DataFrame(features_list)

def smooth_std_pe_features(signal: np.ndarray, model: str) -> dict:
    """Computes smoothed standard deviation and prediction error features from a signal.

    This function applies a sliding window to the input signal and calculates
    the standard deviation, prediction error, and a Gaussian-smoothed value
    for each window. These features are indicative of signal variability and predictability.

    Args:
        signal (np.ndarray): The input 1D signal array.
        model (str): The model name, used to retrieve window parameters (size and step).

    Returns:
        dict: A dictionary containing lists of computed 'std_dev', 'pred_error', and
              'gaussian_smooth' features across all windows.
    """
    window_size, step_size = get_simlsl_window_params(model)
    features = {'std_dev': [], 'pred_error': [], 'gaussian_smooth': []}

    for start in range(0, len(signal) - window_size + 1, step_size):
        window = signal[start:start + window_size]
        features['std_dev'].append(np.std(window))

        if len(window) > 3:
            pred_val = window[-3] + 2 * (window[-2] - window[-3])
            features['pred_error'].append(abs(pred_val - window[-1]))
        else:
            features['pred_error'].append(np.nan)

        weights = np.exp(-0.5 * (np.linspace(-1, 1, len(window)) ** 2))
        weights /= weights.sum()
        features['gaussian_smooth'].append(np.sum(window * weights))

    return features


def smooth_std_pe_process(subject: str, model: str, use_jittering: bool = False) -> None:
    """Main function to extract smoothed standard deviation and prediction error features
    from SIMlsl (Simulated Lane-keeping System) data for a given subject.

    This function loads SIMlsl data, extracts relevant signals (steering, acceleration, lane offset),
    applies optional jittering for data augmentation, computes smoothed standard deviation
    and prediction error features using a sliding window, and saves the results to a CSV file.

    Args:
        subject (str): The subject identifier in 'subjectID_version' format (e.g., 'S0120_2').
        model (str): The model name, used to determine window settings for feature extraction.
        use_jittering (bool): If True, jittering-based data augmentation is applied to signals.
                              Defaults to False.

    Returns:
        None: The function saves the processed features to a CSV file and does not return any value.
    """
    parts = subject.split('_')
    if len(parts) != 2:
        logging.error(f"Unexpected subject format: {subject}")
        return

    subject_id, version = parts
    file_path = f"{DATASET_PATH}/{subject_id}/SIMlsl_{subject_id}_{version}.mat"
    data = safe_load_mat(file_path)

    if data is None:
        logging.error(f"File loading failed: {file_path}")
        return

    sim_data = data.get('SIM_lsl', np.array([]))
    if sim_data.size == 0 or sim_data.shape[0] < 31:
        logging.error(f"Invalid data structure in {file_path}")
        return

    steering = np.nan_to_num(sim_data[29])
    steering_speed = np.gradient(steering) * SAMPLE_RATE_SIMLSL
    lat_acc = np.nan_to_num(sim_data[19])
    long_acc = np.nan_to_num(sim_data[18])
    lane_offset = np.nan_to_num(sim_data[27])
    
    signals = [steering, steering_speed, lat_acc, long_acc, lane_offset]
    signal_names = ['steering', 'steering_speed', 'lat_acc', 'long_acc', 'lane_offset']

    if use_jittering:
        signals = [jittering(sig) for sig in signals]

    features = {}
    for signal, name in zip(signals, signal_names):
        result = smooth_std_pe_features(signal, model)
        for key in result:
            features[f'{name}_{key}'] = result[key]

    window_size, step_size = get_simlsl_window_params(model)
    timestamps = sim_data[0][::step_size][:len(features['steering_std_dev'])]

    df = pd.DataFrame(features)
    df.insert(0, 'Timestamp', timestamps)

    save_csv(df, subject_id, version, 'smooth_std_pe', model)


def time_freq_domain_process(subject: str, model: str, use_jittering: bool = False) -> None:
    """Main function to extract time-frequency domain features from SIMlsl signals for a given subject.

    This function loads SIMlsl data, extracts relevant vehicle dynamics signals,
    applies optional jittering for data augmentation, and then processes these
    signals using a sliding window to extract time-frequency domain features.
    The extracted features are then saved to a CSV file.

    Args:
        subject (str): The subject identifier in 'subjectID_version' format (e.g., 'S0120_2').
        model (str): The model name, used to determine window settings for feature extraction.
        use_jittering (bool): If True, jittering-based data augmentation is applied to signals.
                              Defaults to False.

    Returns:
        None: The function saves the processed features to a CSV file and does not return any value.
    """
    parts = subject.split('_')
    if len(parts) != 2:
        logging.error(f"Unexpected subject format: {subject}")
        return

    subject_id, version = parts
    simlsl_file = f"{DATASET_PATH}/{subject_id}/SIMlsl_{subject_id}_{version}.mat"

    simlsl_data = safe_load_mat(simlsl_file)
    if simlsl_data is None or 'SIM_lsl' not in simlsl_data:
        logging.error(f"SIMlsl data loading failed for {subject_id}_{version}")
        return

    sim_data = simlsl_data['SIM_lsl']

    steering = np.nan_to_num(sim_data[29, :])
    steering_speed = np.gradient(steering) * SAMPLE_RATE_SIMLSL
    
    signals = [steering, steering_speed, sim_data[19], sim_data[27], sim_data[18]]
    prefixes = ["Steering_", "SteeringSpeed_", "Lateral_", "LaneOffset_", "LongAcc_"]

    if use_jittering:
        signals = [jittering(sig, sigma=0.03) for sig in signals]

    window_size, step_size = get_simlsl_window_params(model)

    features_df = process_simlsl_data(signals, prefixes, model)
    timestamps = sim_data[0, ::step_size][:len(features_df)]

    features_df.insert(0, 'Timestamp', timestamps)
    save_csv(features_df, subject_id, version, 'time_freq_domain', model)


