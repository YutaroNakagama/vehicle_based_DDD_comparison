"""Wavelet-based feature extraction from vehicle dynamics signals (SIMlsl).

This module performs multi-level GHM wavelet decomposition on steering and
acceleration signals, computes energy per decomposition path, and outputs
features used in driver drowsiness detection.

Key output features include power values from decomposition paths like DDD, DDA, etc.
"""

import numpy as np
import pandas as pd
import os
import logging
from scipy.signal import lfilter

from src.utils.io.loaders import safe_load_mat, save_csv
from src.utils.domain_generalization.jitter import jittering
from src.config import (
    DATASET_PATH,
    SAMPLE_RATE_SIMLSL,
    SCALING_FILTER,
    WAVELET_FILTER,
    WAVELET_LEV,
    MODEL_WINDOW_CONFIG,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_simlsl_window_params(model: str) -> tuple[int, int]:
    """Get window and step size in samples based on model configuration.

    Args:
        model (str): Model name (e.g., 'SvmW').

    Returns:
        tuple[int, int]: (window size, step size) in samples.
    """
    config = MODEL_WINDOW_CONFIG[model]
    window_samples = int(config["window_sec"] * SAMPLE_RATE_SIMLSL)
    step_samples = int(config["step_sec"] * SAMPLE_RATE_SIMLSL)
    return window_samples, step_samples


def ghm_wavelet_transform(signal: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Apply GHM wavelet transform to a signal.

    Args:
        signal (np.ndarray): 1D input signal.

    Returns:
        list: List of tuples (scaling_coeffs, wavelet_coeffs) per level.
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
    """Align and sum two wavelet coefficient arrays to same length.

    Args:
        coeff1 (np.ndarray): First array.
        coeff2 (np.ndarray): Second array.

    Returns:
        np.ndarray: Element-wise sum of trimmed arrays.
    """
    min_len = min(len(coeff1), len(coeff2))
    return coeff1[:min_len] + coeff2[:min_len]


def generate_decomposition_signals(coeffs: list[tuple[np.ndarray, np.ndarray]]) -> list[np.ndarray]:
    """Generate 8 wavelet decomposition paths from coefficients.

    Args:
        coeffs (list): List of wavelet transform coefficient pairs.

    Returns:
        list[np.ndarray]: List of 8 decomposition signals (DDD, DDA, ... AAA).
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
    """Calculate signal power (mean squared value).

    Args:
        signal (np.ndarray): 1D input signal.

    Returns:
        float: Mean power of the signal.
    """
    return np.mean(signal ** 2)


def process_window(signal_window: np.ndarray) -> list[float]:
    """Apply wavelet transform and compute power for each decomposition.

    Args:
        signal_window (np.ndarray): 1D signal window.

    Returns:
        list[float]: Power values for 8 decomposition paths.
    """
    coeffs = ghm_wavelet_transform(signal_window)
    decomposition_signals = generate_decomposition_signals(coeffs)
    return [calculate_power(signal) for signal in decomposition_signals]


def wavelet_process(subject: str, model: str, use_jittering: bool = False) -> None:
    """Main function to extract wavelet features from vehicle signals.

    Applies multi-resolution wavelet decomposition to steering and acceleration
    signals, computes power for each path, and saves results as CSV.

    Args:
        subject (str): Subject identifier in 'subjectID_version' format, e.g., 'S0120_2'.
        model (str): Model name for window config.
        use_jittering (bool): Whether to apply jittering to raw signals.

    Returns:
        None
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

    window_size, step_size = get_simlsl_window_params(model)

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

    for start in range(0, len(sim_time) - window_size + 1, step_size):
        window_powers = []
        for signal in signals.values():
            signal_window = signal[start:start + window_size]
            window_powers.extend(process_window(signal_window))

        all_powers.append(window_powers)
        all_timestamps.append(sim_time[start])

    decomposition_labels = ['DDD', 'DDA', 'DAD', 'DAA', 'ADD', 'ADA', 'AAD', 'AAA']
    column_names = [f'{sig}_{label}' for sig in signals.keys() for label in decomposition_labels]

    df = pd.DataFrame(all_powers, columns=column_names)
    df.insert(0, 'Timestamp', all_timestamps)

    save_csv(df, subject_id, version, 'wavelet', model)

