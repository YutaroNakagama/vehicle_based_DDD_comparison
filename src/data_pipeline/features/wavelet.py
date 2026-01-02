"""Wavelet-based feature extraction for SIMlsl (vehicle dynamics) signals.

Performs multi-level GHM wavelet decomposition on steering, acceleration, and
lane offset signals; computes mean power per decomposition path; supports
optional jittering augmentation and sliding-window extraction.

Notes
-----
- Input MAT expected: SIM_lsl with time at row 0 and signals at fixed indices.
- Eight decomposition paths produced (DDD...AAA) via coefficient combinations.
"""

import numpy as np
import pandas as pd
import os
import logging
from scipy.signal import lfilter
from joblib import Parallel, delayed

from src.utils.io.loaders import safe_load_mat, save_csv
from src.data_pipeline.augmentation.jitter import jittering
from src.config import (
    DATASET_PATH,
    SAMPLE_RATE_SIMLSL,
    SCALING_FILTER,
    WAVELET_FILTER,
    WAVELET_LEV,
    MODEL_WINDOW_CONFIG,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


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


def ghm_wavelet_transform(signal: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Apply GHM wavelet transform to a 1D signal.

    Performs multi-level decomposition using predefined scaling and
    wavelet filters.

    Parameters
    ----------
    signal : ndarray
        Input 1D signal.

    Returns
    -------
    list of tuple of (ndarray, ndarray)
        List of (scaling_coeffs, wavelet_coeffs) for each level.
    """
    coeffs = []
    approx = signal
    for _ in range(WAVELET_LEV):
        scaling_coeffs = lfilter(SCALING_FILTER, [1], approx)[::2]
        wavelet_coeffs = lfilter(WAVELET_FILTER, [1], approx)[::2]
        approx = scaling_coeffs
        coeffs.append((scaling_coeffs, wavelet_coeffs))
    return coeffs


def adjust_and_add(coeff1: np.ndarray, coeff2: np.ndarray) -> np.ndarray:
    """Align and sum two coefficient arrays.

    Parameters
    ----------
    coeff1 : ndarray
        First coefficient array.
    coeff2 : ndarray
        Second coefficient array.

    Returns
    -------
    ndarray
        Element-wise sum of trimmed arrays.
    """
    min_len = min(len(coeff1), len(coeff2))
    return coeff1[:min_len] + coeff2[:min_len]


def generate_decomposition_signals(coeffs: list[tuple[np.ndarray, np.ndarray]]) -> list[np.ndarray]:
    """Generate 8 wavelet decomposition paths.

    Parameters
    ----------
    coeffs : list of tuple of (ndarray, ndarray)
        Wavelet coefficients per level.

    Returns
    -------
    list of ndarray
        Eight decomposition signals (DDD, DDA, ... AAA).
    """
    return [
        coeffs[0][1],  # D
        adjust_and_add(coeffs[0][1], coeffs[1][0]),  # D + A
        adjust_and_add(coeffs[0][1], coeffs[1][1]),  # D + D
        adjust_and_add(coeffs[0][1], coeffs[2][0]),  # D + A
        adjust_and_add(coeffs[1][1], coeffs[2][1]),  # D + D
        adjust_and_add(coeffs[1][0], coeffs[2][1]),  # A + D
        adjust_and_add(coeffs[1][0], coeffs[2][0]),  # A + A
        coeffs[2][0],  # A
    ]


def calculate_power(signal: np.ndarray) -> float:
    """Calculate mean power of a 1D signal.

    Parameters
    ----------
    signal : ndarray
        Input signal.

    Returns
    -------
    float
        Mean squared value of the signal.
    """
    return np.mean(signal ** 2)


def process_window(signal_window: np.ndarray) -> list[float]:
    """Apply wavelet transform to a signal window and compute powers.

    Parameters
    ----------
    signal_window : ndarray
        A 1D signal segment.

    Returns
    -------
    list of float
        Power values for 8 decomposition paths.
    """
    coeffs = ghm_wavelet_transform(signal_window)
    decomposition_signals = generate_decomposition_signals(coeffs)
    return [calculate_power(signal) for signal in decomposition_signals]


def wavelet_process(subject: str, model_name: str, use_jittering: bool = False) -> None:
    """Compute wavelet powers for steering/accel/lane offset signals and save CSV.

    Parameters
    ----------
    subject : str
        Subject identifier ("<id>_<version>").
    model_name : str
        Determines window size selection.
    use_jittering : bool
        Apply jitter augmentation if True.
    """
    parts = subject.split('_')
    if len(parts) != 2:
        logging.error(f"Unexpected subject format: {subject}")
        return

    subject_id, version = parts
    mat_file_path = os.path.join(DATASET_PATH, subject_id, f"SIMlsl_{subject_id}_{version}.mat")

    mat_data = safe_load_mat(mat_file_path)
    if mat_data is None:
        logging.error(f"Failed to load data from {mat_file_path}")
        return

    sim_data = mat_data.get('SIM_lsl')
    if sim_data is None or sim_data.shape[0] < 30:
        logging.error(f"Invalid SIM_lsl data structure in {mat_file_path}")
        return

    window_size, step_size = get_simlsl_window_params(model_name)

    steering = np.nan_to_num(sim_data[29, :])
    steering_speed = np.gradient(steering) * SAMPLE_RATE_SIMLSL

    signals = {
        'SteeringWheel': steering,
        'SteeringSpeed': steering_speed,
        'LongitudinalAccel': np.nan_to_num(sim_data[18, :]),
        'LateralAccel': np.nan_to_num(sim_data[19, :]),
        'LaneOffset': np.nan_to_num(sim_data[27, :]),
    }

    if use_jittering:
        signals = {key: jittering(sig) for key, sig in signals.items()}

    sim_time = sim_data[0, :]
    all_powers, all_timestamps = [], []

    def process_one_window(start):
        window_powers = []
        for signal in signals.values():
            signal_window = signal[start:start + window_size]
            window_powers.extend(process_window(signal_window))
        return window_powers, sim_time[start]

    window_starts = range(0, len(sim_time) - window_size + 1, step_size)
    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(process_one_window)(start) for start in window_starts
    )
    all_powers, all_timestamps = zip(*results)

    decomposition_labels = ['DDD', 'DDA', 'DAD', 'DAA', 'ADD', 'ADA', 'AAD', 'AAA']
    column_names = [f'{sig}_{label}' for sig in signals.keys() for label in decomposition_labels]

    df = pd.DataFrame(all_powers, columns=column_names)
    df.insert(0, 'Timestamp', all_timestamps)

    save_csv(df, subject_id, version, 'wavelet', model_name)

