import numpy as np
import pandas as pd
import os
import logging
from scipy.signal import lfilter

from src.utils.io.loaders import safe_load_mat, save_csv
from src.config import (
    DATASET_PATH,
    WINDOW_SIZE_SAMPLE_SIMLSL,
    STEP_SIZE_SAMPLE_SIMLSL,
    SCALING_FILTER,
    WAVELET_FILTER,
    WAVELET_LEV,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def ghm_wavelet_transform(signal):
    coeffs = []
    approx = signal
    for _ in range(WAVELET_LEV):
        scaling_coeffs = lfilter(SCALING_FILTER, [1], approx)[::2]
        wavelet_coeffs = lfilter(WAVELET_FILTER, [1], approx)[::2]
        approx = scaling_coeffs
        coeffs.append((scaling_coeffs, wavelet_coeffs))
    return coeffs


def adjust_and_add(coeff1, coeff2):
    min_len = min(len(coeff1), len(coeff2))
    return coeff1[:min_len] + coeff2[:min_len]


def generate_decomposition_signals(coeffs):
    return [
        coeffs[0][1],
        adjust_and_add(coeffs[0][1], coeffs[1][0]),
        adjust_and_add(coeffs[0][1], coeffs[1][1]),
        adjust_and_add(coeffs[0][1], coeffs[2][0]),
        adjust_and_add(coeffs[1][1], coeffs[2][1]),
        adjust_and_add(coeffs[1][0], coeffs[2][1]),
        adjust_and_add(coeffs[1][0], coeffs[2][0]),
        coeffs[2][0]
    ]


def calculate_power(signal):
    return np.mean(signal ** 2)


def process_window(signal_window):
    coeffs = ghm_wavelet_transform(signal_window)
    decomposition_signals = generate_decomposition_signals(coeffs)
    return [calculate_power(signal) for signal in decomposition_signals]


def wavelet_process(subject, model):
    subject_id, version = subject.split('/')[0], subject.split('/')[1].split('_')[-1]
    mat_file_path = os.path.join(DATASET_PATH, subject_id, f"SIMlsl_{subject_id}_{version}.mat")

    mat_data = safe_load_mat(mat_file_path)
    if mat_data is None:
        logging.error(f"Failed to load data from {mat_file_path}")
        return

    sim_data = mat_data.get('SIM_lsl')
    if sim_data is None or sim_data.shape[0] < 30:
        logging.error(f"Invalid SIM_lsl data structure in {mat_file_path}")
        return

    signals = {
        'SteeringWheel': np.nan_to_num(sim_data[29, :]),
        'LongitudinalAccel': np.nan_to_num(sim_data[18, :]),
        'LateralAccel': np.nan_to_num(sim_data[19, :]),
        'LaneOffset': np.nan_to_num(sim_data[27, :]),
    }

    sim_time = sim_data[0, :]
    all_powers, all_timestamps = [], []

    for start in range(0, len(sim_time) - WINDOW_SIZE_SAMPLE_SIMLSL + 1, STEP_SIZE_SAMPLE_SIMLSL):
        window_powers = []
        for signal in signals.values():
            signal_window = signal[start:start + WINDOW_SIZE_SAMPLE_SIMLSL]
            window_powers.extend(process_window(signal_window))

        all_powers.append(window_powers)
        all_timestamps.append(sim_time[start])

    decomposition_labels = ['DDD', 'DDA', 'DAD', 'DAA', 'ADD', 'ADA', 'AAD', 'AAA']
    column_names = [f'{sig}_{label}' for sig in signals.keys() for label in decomposition_labels]

    df = pd.DataFrame(all_powers, columns=column_names)
    df.insert(0, 'Timestamp', all_timestamps)

    save_csv(df, subject_id, version, 'wavelet', model)
    logging.info(f"Wavelet features saved for {subject_id}_{version} [{model}].")



