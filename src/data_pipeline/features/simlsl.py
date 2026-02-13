"""Feature extraction for vehicle dynamics (SIMlsl) signals.

Extracts statistical, spectral, time-frequency, and prediction-error based
features from steering, acceleration, lane offset signals. Supports:
- Statistical & spectral window features (extract_statistical_features)
- Smooth/std/prediction-error features (smooth_std_pe_process)
- Time-frequency domain summary features (time_freq_domain_process)
- Optional jittering augmentation (Gaussian noise based)

Notes
-----
- Expects MAT key 'SIM_lsl' with time at row 0 and signals at fixed indices:
  steering=29, lat_acc=19, long_acc=18, lane_offset=27
- Window config derived from MODEL_WINDOW_CONFIG[model_name]
"""

import numpy as np
import pandas as pd
import logging
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis
from numba import njit
from joblib import Parallel, delayed

from src.utils.io.loaders import safe_load_mat, save_csv
from src.data_pipeline.augmentation.jitter import jittering
from src.config import (
    DATASET_PATH,
    SAMPLE_RATE_SIMLSL,
    MODEL_WINDOW_CONFIG,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


from typing import Optional


@njit
def _count_similar_templates(
    signal: np.ndarray, m: int, r: float
) -> int:
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


def sample_entropy(
    signal: np.ndarray, m: int = 2, r: Optional[float] = None
) -> float:
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
        Sample Entropy value. Returns 0.0 for constant signals (maximum regularity),
        or a large value (10.0) when templates cannot be matched (high irregularity).
    """
    signal = np.asarray(signal, dtype=np.float64)
    N = len(signal)
    if r is None:
        r = 0.2 * np.std(signal)
    
    # Constant signal or too short: maximum regularity = entropy 0
    if N <= m + 1 or r == 0:
        return 0.0

    try:
        A = _count_similar_templates(signal, m + 1, r)
        B = _count_similar_templates(signal, m, r)
        
        # No template matches found: high irregularity
        if B == 0:
            return 10.0  # Large finite value instead of inf/nan
        if A == 0:
            return 10.0  # -log(0) would be inf, use large value
            
        return -np.log(A / B)
    except Exception:
        return 0.0  # Fallback for any unexpected errors

def _katz_fractal_dimension(signal: np.ndarray) -> float:
    """Katz Fractal Dimension of a 1-D time series.

    KFD = log10(L) / log10(d)

    where *L* is the total path length (sum of absolute successive
    differences) and *d* is the maximum Euclidean distance between
    the first point and any other point (the *diameter* or *planar
    extent*).

    Reference
    ---------
    Katz, M. J. (1988). Fractals and the analysis of waveform
    complexity. *Computers and Biomedical Research*, 21(2), 150–166.
    """
    n = len(signal)
    if n < 2:
        return 0.0
    dists = np.abs(np.diff(signal))
    L = dists.sum()
    if L == 0:
        return 0.0
    # d = max distance from x[0] to any other point (time-indexed)
    indices = np.arange(n)
    d = np.max(np.sqrt((indices - 0) ** 2 + (signal - signal[0]) ** 2))
    if d == 0:
        return 0.0
    return np.log10(L) / np.log10(d)


def _shannon_entropy(signal: np.ndarray, n_bins: int = 50) -> float:
    """Shannon entropy of the signal amplitude distribution.

    The signal is discretised into *n_bins* equal-width bins and the
    entropy is computed as  H = −Σ p_i log₂(p_i).

    Reference
    ---------
    Shannon, C. E. (1948). A mathematical theory of communication.
    *Bell System Technical Journal*, 27(3), 379–423.
    """
    if len(signal) < 2 or np.std(signal) == 0:
        return 0.0
    counts, _ = np.histogram(signal, bins=n_bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


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
        f'{prefix}KatzFractalDim': _katz_fractal_dimension(signal),
        f'{prefix}ShannonEntropy': _shannon_entropy(signal),
        f'{prefix}FreqVar': np.var(freqs[band]),
        f'{prefix}SpectralEntropy': -np.sum((spectrum[band] / band_sum) * np.log2((spectrum[band] / band_sum) + np.finfo(float).eps)),
        f'{prefix}SpectralFlux': np.sqrt(np.sum(np.diff(spectrum) ** 2)),
        f'{prefix}FreqCOG': np.sum(freqs[band] * spectrum[band]) / band_sum if band_sum > 0 else 0,
        f'{prefix}DominantFreq': freqs[np.argmax(spectrum)],
        f'{prefix}AvgPSD': np.mean(spectrum[band]),
        f'{prefix}SampleEntropy': sample_entropy(signal, m=2, r=0.2*np.std(signal)),
    }

    return features


def get_simlsl_window_params(model_name: str) -> tuple[int, int]:
    """Return window and step size (samples) for SIMlsl data.

    Parameters
    ----------
    model_name : str
        Key for MODEL_WINDOW_CONFIG.
    """
    config = MODEL_WINDOW_CONFIG[model_name]
    window_samples = int(config["window_sec"] * SAMPLE_RATE_SIMLSL)
    step_samples = int(config["step_sec"] * SAMPLE_RATE_SIMLSL)
    return window_samples, step_samples

def process_simlsl_data(signals: list[np.ndarray], prefixes: list[str], model_name: str) -> pd.DataFrame:
    """Process SIMlsl signals with sliding windows and extract features.

    For each signal, statistical and spectral features are extracted
    from overlapping windows. Processing is parallelized for efficiency.

    Parameters
    ----------
    signals : list of ndarray
        List of 1D SIMlsl signals.
    prefixes : list of str
        Prefixes corresponding to each signal, used in feature names.
    model_name : str
        Model name used to determine window and step size.

    Returns
    -------
    DataFrame
        A DataFrame where rows represent windows and columns are extracted features.
    """
    window_size, step_size = get_simlsl_window_params(model_name)
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

def smooth_std_pe_features(signal: np.ndarray, model_name: str) -> dict:
    """Compute smoothed standard deviation and prediction error features.

    Applies a sliding window to compute the standard deviation,
    prediction error, and mean values for each segment.

    Parameters
    ----------
    signal : ndarray
        Input 1D signal.
    model_name : str
        Model name used to determine window and step size.

    Returns
    -------
    dict
        Dictionary containing lists of:
        - ``std_dev`` : standard deviation values.
        - ``pred_error`` : prediction error values.
        - ``mean`` : arithmetic mean values.
    """
    window_size, step_size = get_simlsl_window_params(model_name)
    features = {'std_dev': [], 'pred_error': [], 'mean': []}

    for start in range(0, len(signal) - window_size + 1, step_size):
        window = signal[start:start + window_size]
        features['std_dev'].append(np.std(window))

        if len(window) > 3:
            pred_val = window[-3] + 2 * (window[-2] - window[-3])
            features['pred_error'].append(abs(pred_val - window[-1]))
        else:
            features['pred_error'].append(np.nan)

        features['mean'].append(np.mean(window))

    return features


def smooth_std_pe_process(subject: str, model_name: str, use_jittering: bool = False) -> None:
    """Compute smooth/std/pred-error features for steering & related signals.

    Parameters
    ----------
    subject : str
        Subject identifier ("<id>_<version>").
    model_name : str
        Model name for window sizing.
    use_jittering : bool
        Apply jittering augmentation if True.
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
        result = smooth_std_pe_features(signal, model_name)
        for key in result:
            features[f'{name}_{key}'] = result[key]

    window_size, step_size = get_simlsl_window_params(model_name)
    timestamps = sim_data[0][::step_size][:len(features['steering_std_dev'])]

    df = pd.DataFrame(features)
    df.insert(0, 'Timestamp', timestamps)

    save_csv(df, subject_id, version, 'smooth_std_pe', model_name)


def time_freq_domain_process(subject: str, model_name: str, use_jittering: bool = False) -> None:
    """Extract time-frequency domain features (statistical + spectral) per window.

    Parameters
    ----------
    subject : str
        Subject identifier ("<id>_<version>").
    model_name : str
        Model name for window sizing.
    use_jittering : bool
        Apply jittering augmentation if True.
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

    window_size, step_size = get_simlsl_window_params(model_name)

    features_df = process_simlsl_data(signals, prefixes, model_name)
    timestamps = sim_data[0, ::step_size][:len(features_df)]

    features_df.insert(0, 'Timestamp', timestamps)
    save_csv(features_df, subject_id, version, 'time_freq_domain', model_name)


