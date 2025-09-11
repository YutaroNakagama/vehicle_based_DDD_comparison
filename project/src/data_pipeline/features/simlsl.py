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
    """Count similar template pairs for Sample Entropy.

    Parameters
    ----------
    signal : ndarray
        Input 1D signal array.
    m : int
        Length of the template.
    r : float
        Tolerance for similarity.

    Returns
    -------
    int
        Number of similar template pairs.
    """
    N = len(signal)
    count = 0
    for i in range(N - m):
        for j in range(i + 1, N - m + 1):
            if np.all(np.abs(signal[i:i + m] - signal[j:j + m]) <= r):
                count += 1
    return count

def sample_entropy(signal, m=2, r=None):
    """Calculate Sample Entropy of a signal.

    Sample Entropy is a measure of complexity and predictability of time-series data.
    A lower value indicates more regularity and predictability.

    Parameters
    ----------
    signal : ndarray
        Input 1D signal.
    m : int, default=2
        Length of the template vectors.
    r : float, optional
        Tolerance for accepting matches. If None, defaults to
        ``0.2 * std(signal)``.

    Returns
    -------
    float
        Sample Entropy value, or ``np.nan`` if calculation fails.
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
    """Extract statistical and spectral features from a 1D signal.

    Features include:
    - Range, standard deviation, energy
    - Zero-crossing rate, quartiles, median
    - Skewness, kurtosis
    - Frequency variance, spectral entropy, spectral flux
    - Frequency center of gravity (COG), dominant frequency, average PSD
    - Sample Entropy

    Parameters
    ----------
    signal : ndarray
        Input 1D signal from which to extract features.
    prefix : str, default=""
        Prefix added to each feature name (e.g., ``"Steering_"``).

    Returns
    -------
    dict
        Dictionary mapping feature names (with prefix) to their computed values.
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
    """Retrieve window and step size (in samples) for SIMlsl data.

    This function looks up the window configuration for the given model
    in ``MODEL_WINDOW_CONFIG`` and converts the window/step durations
    from seconds into samples.

    Parameters
    ----------
    model : str
        Model name used to retrieve configuration (e.g., ``"Lstm"``, ``"SvmW"``).

    Returns
    -------
    tuple of (int, int)
        - Window size in samples.
        - Step size in samples.
    """
    config = MODEL_WINDOW_CONFIG[model]
    window_samples = int(config["window_sec"] * SAMPLE_RATE_SIMLSL)
    step_samples = int(config["step_sec"] * SAMPLE_RATE_SIMLSL)
    return window_samples, step_samples

def process_simlsl_data(signals: list[np.ndarray], prefixes: list[str], model: str) -> pd.DataFrame:
    """Process SIMlsl signals with sliding windows and extract features.

    For each signal, statistical and spectral features are extracted
    from overlapping windows. Processing is parallelized for efficiency.

    Parameters
    ----------
    signals : list of ndarray
        List of 1D SIMlsl signals.
    prefixes : list of str
        Prefixes corresponding to each signal, used in feature names.
    model : str
        Model name used to determine window and step size.

    Returns
    -------
    DataFrame
        A DataFrame where rows represent windows and columns are extracted features.
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
    """Compute smoothed standard deviation and prediction error features.

    Applies a sliding window to compute the standard deviation,
    prediction error, and Gaussian-smoothed values for each segment.

    Parameters
    ----------
    signal : ndarray
        Input 1D signal.
    model : str
        Model name used to determine window and step size.

    Returns
    -------
    dict
        Dictionary containing lists of:
        - ``std_dev`` : standard deviation values.
        - ``pred_error`` : prediction error values.
        - ``gaussian_smooth`` : Gaussian-smoothed values.
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
    """Extract smoothed std-dev and prediction error features for a subject.

    Loads SIMlsl data, extracts steering/acceleration/lane offset signals,
    applies optional jittering, computes smoothed standard deviation and
    prediction error features, and saves results to CSV.

    Parameters
    ----------
    subject : str
        Subject identifier (format: ``"<id>_<version>"``).
    model : str
        Model name used to determine window settings.
    use_jittering : bool, default=False
        Whether to apply jittering-based data augmentation.

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
    """Extract time-frequency domain features for a subject.

    Loads SIMlsl data, extracts steering/acceleration/lane offset signals,
    applies optional jittering, and computes time-frequency domain features
    using sliding windows. Results are saved to CSV.

    Parameters
    ----------
    subject : str
        Subject identifier (format: ``"<id>_<version>"``).
    model : str
        Model name used to determine window settings.
    use_jittering : bool, default=False
        Whether to apply jittering-based data augmentation.

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


