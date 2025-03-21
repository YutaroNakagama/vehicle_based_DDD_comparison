import numpy as np
import pandas as pd
import logging
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis

from src.utils.io.loaders import safe_load_mat, save_csv
from src.config import (
    DATASET_PATH,
    SAMPLE_RATE_SIMLSL,
    WINDOW_SIZE_SAMPLE_SIMLSL,
    STEP_SIZE_SAMPLE_SIMLSL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def extract_statistical_features(signal, prefix=""):
    """Extract various statistical and spectral features from a signal."""
    freqs = fftfreq(len(signal), 1 / SAMPLE_RATE_SIMLSL)
    spectrum = np.abs(fft(signal)) ** 2
    band = (freqs >= 0.5) & (freqs <= 30)

    features = {
        f'{prefix}Range': np.ptp(signal),
        f'{prefix}StdDev': np.std(signal),
        f'{prefix}Energy': np.sum(signal ** 2),
        f'{prefix}ZeroCrossingRate': np.mean(np.diff(np.sign(signal)) != 0),
        f'{prefix}Quartile25': np.percentile(signal, 25),
        f'{prefix}Median': np.median(signal),
        f'{prefix}Quartile75': np.percentile(signal, 75),
        f'{prefix}Skewness': skew(signal),
        f'{prefix}Kurtosis': kurtosis(signal),
        f'{prefix}FreqVar': np.var(freqs[band]),
        f'{prefix}SpectralEntropy': -np.sum((spectrum[band] / np.sum(spectrum[band])) * np.log2((spectrum[band] / np.sum(spectrum[band])) + np.finfo(float).eps)),
        f'{prefix}SpectralFlux': np.sqrt(np.sum(np.diff(spectrum) ** 2)),
        f'{prefix}FreqCOG': np.sum(freqs[band] * spectrum[band]) / np.sum(spectrum[band]),
        f'{prefix}DominantFreq': freqs[np.argmax(spectrum)],
        f'{prefix}AvgPSD': np.mean(spectrum[band]),
        f'{prefix}SampleEntropy': 0,  # Placeholder
    }

    return features


def process_simlsl_data(signals, prefixes):
    """Process multiple signals and return a DataFrame with extracted features."""
    features_list = []

    for start in range(0, len(signals[0]) - WINDOW_SIZE_SAMPLE_SIMLSL + 1, STEP_SIZE_SAMPLE_SIMLSL):
        window_features = {}
        for signal, prefix in zip(signals, prefixes):
            segment = signal[start:start + WINDOW_SIZE_SAMPLE_SIMLSL]
            window_features.update(extract_statistical_features(segment, prefix))
        features_list.append(window_features)

    return pd.DataFrame(features_list)


def smooth_std_pe_features(signal):
    features = {'std_dev': [], 'pred_error': [], 'gaussian_smooth': []}

    for start in range(0, len(signal) - WINDOW_SIZE_SAMPLE_SIMLSL + 1, STEP_SIZE_SAMPLE_SIMLSL):
        window = signal[start:start + WINDOW_SIZE_SAMPLE_SIMLSL]
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


def smooth_std_pe_process(subject, model):
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

    features = {}
    for signal, name in zip(signals, signal_names):
        result = smooth_std_pe_features(signal)
        for key in result:
            features[f'{name}_{key}'] = result[key]

    timestamps = sim_data[0][::STEP_SIZE_SAMPLE_SIMLSL][:len(features['steering_std_dev'])]
    df = pd.DataFrame(features)
    df.insert(0, 'Timestamp', timestamps)

    save_csv(df, subject_id, version, 'smooth_std_pe', model)
    logging.info(f"Saved smooth_std_pe features for {subject_id}_{version} [{model}]")


def time_freq_domain_process(subject, model):
    subject_id, version = subject.split('/')[0], subject.split('/')[1].split('_')[-1]
    simlsl_file = f"{DATASET_PATH}/{subject_id}/SIMlsl_{subject_id}_{version}.mat"

    simlsl_data = safe_load_mat(simlsl_file)
    if simlsl_data is None or 'SIM_lsl' not in simlsl_data:
        logging.error(f"SIMlsl data loading failed for {subject_id}_{version}")
        return

    sim_data = simlsl_data['SIM_lsl']
    signals = [sim_data[idx] for idx in [29, 19, 27, 18]]
    prefixes = ["Steering_", "Lateral_", "LaneOffset_", "LongAcc_"]

    features_df = process_simlsl_data(signals, prefixes)
    timestamps = sim_data[0, ::STEP_SIZE_SAMPLE_SIMLSL][:len(features_df)]

    features_df.insert(0, 'Timestamp', timestamps)
    save_csv(features_df, subject_id, version, 'time_freq_domain', model)
    logging.info(f"Saved time-frequency domain features for {subject_id}_{version} [{model}]")

