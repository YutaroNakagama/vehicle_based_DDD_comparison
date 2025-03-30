import numpy as np
import pandas as pd
import logging
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis

from src.utils.io.loaders import safe_load_mat, save_csv
from src.utils.domain_generalization.data_aug import jittering
from src.config import (
    DATASET_PATH,
    SAMPLE_RATE_SIMLSL,
    MODEL_WINDOW_CONFIG,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def extract_statistical_features(signal, prefix=""):
    """Extract various statistical and spectral features from a signal."""
    freqs = fftfreq(len(signal), 1 / SAMPLE_RATE_SIMLSL)
    spectrum = np.abs(fft(signal)) ** 2
    band = (freqs >= 0.5) & (freqs <= 30)
    band_sum = np.sum(spectrum[band]) + np.finfo(float).eps  # avoid divide by zero

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
        #f'{prefix}SpectralEntropy': -np.sum((spectrum[band] / np.sum(spectrum[band])) * np.log2((spectrum[band] / np.sum(spectrum[band])) + np.finfo(float).eps)),
        f'{prefix}SpectralEntropy': -np.sum((spectrum[band] / band_sum) * np.log2((spectrum[band] / band_sum) + np.finfo(float).eps)),
        f'{prefix}SpectralFlux': np.sqrt(np.sum(np.diff(spectrum) ** 2)),
        #f'{prefix}FreqCOG': np.sum(freqs[band] * spectrum[band]) / np.sum(spectrum[band]),
        f'{prefix}FreqCOG': np.sum(freqs[band] * spectrum[band]) / band_sum if band_sum > 0 else 0,
        f'{prefix}DominantFreq': freqs[np.argmax(spectrum)],
        f'{prefix}AvgPSD': np.mean(spectrum[band]),
        f'{prefix}SampleEntropy': 0,  # Placeholder
    }

    return features


def get_simlsl_window_params(model):
    config = MODEL_WINDOW_CONFIG[model]
    window_samples = int(config["window_sec"] * SAMPLE_RATE_SIMLSL)
    step_samples = int(config["step_sec"] * SAMPLE_RATE_SIMLSL)
    return window_samples, step_samples


def process_simlsl_data(signals, prefixes, model):
    """Process multiple signals and return a DataFrame with extracted features."""
    window_size, step_size = get_simlsl_window_params(model)
    features_list = []

    for start in range(0, len(signals[0]) - window_size + 1, step_size):
        window_features = {}
        for signal, prefix in zip(signals, prefixes):
            segment = signal[start:start + window_size]
            window_features.update(extract_statistical_features(segment, prefix))
        features_list.append(window_features)

    return pd.DataFrame(features_list)


def smooth_std_pe_features(signal, model):
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

def smooth_std_pe_process(subject, model, use_jittering=False):
    subject_id, version = subject.split('/')[0], subject.split('/')[1].split('_')[-1]
    file_path = f"{DATASET_PATH}/{subject_id}/SIMlsl_{subject_id}_{version}.mat"
    data = safe_load_mat(file_path)

    if data is None:
        logging.error(f"File loading failed: {file_path}")
        return

    sim_data = data.get('SIM_lsl', np.array([]))
    if sim_data.size == 0 or sim_data.shape[0] < 31:
        logging.error(f"Invalid data structure in {file_path}")
        return

    signals = [sim_data[idx] for idx in [29, 19, 18, 27]]
    signal_names = ['steering', 'lat_acc', 'long_acc', 'lane_offset']

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


def time_freq_domain_process(subject, model, use_jittering=False):
    subject_id, version = subject.split('/')[0], subject.split('/')[1].split('_')[-1]
    simlsl_file = f"{DATASET_PATH}/{subject_id}/SIMlsl_{subject_id}_{version}.mat"

    simlsl_data = safe_load_mat(simlsl_file)
    if simlsl_data is None or 'SIM_lsl' not in simlsl_data:
        logging.error(f"SIMlsl data loading failed for {subject_id}_{version}")
        return

    sim_data = simlsl_data['SIM_lsl']
    signals = [sim_data[idx] for idx in [29, 19, 27, 18]]

    if use_jittering:
        signals = [jittering(sig, sigma=0.03) for sig in signals]

    prefixes = ["Steering_", "Lateral_", "LaneOffset_", "LongAcc_"]

    window_size, step_size = get_simlsl_window_params(model)
    features_df = process_simlsl_data(signals, prefixes, model)
    timestamps = sim_data[0, ::step_size][:len(features_df)]

    features_df.insert(0, 'Timestamp', timestamps)
    save_csv(features_df, subject_id, version, 'time_freq_domain', model)
