"""Wavelet-based Feature Extraction for Driver Drowsiness Detection.

This module provides functionalities for extracting features from vehicle dynamics signals
(specifically SIMlsl data) using multi-level GHM (Generalized Haar Multiwavelet) wavelet decomposition.
It focuses on processing steering and acceleration signals to derive energy features
per decomposition path, which are then used in driver drowsiness detection models.

Key features generated include power values from various decomposition paths (e.g., DDD, DDA, etc.),
providing a multi-resolution analysis of the signals. The module also supports
optional jittering for data augmentation and integrates with a sliding window approach
for feature extraction across time.
"""

import numpy as np
import pandas as pd
import os
import logging
from scipy.signal import lfilter
from joblib import Parallel, delayed

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


def ghm_wavelet_transform(signal: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Applies the GHM (Generalized Haar Multiwavelet) wavelet transform to a 1D signal.

    This function performs a multi-level wavelet decomposition using predefined
    scaling and wavelet filters. It iteratively decomposes the approximation
    coefficients to generate detail and new approximation coefficients at each level.

    Args:
        signal (np.ndarray): The 1D input signal to be transformed.

    Returns:
        list[tuple[np.ndarray, np.ndarray]]: A list of tuples, where each tuple contains
                                             (scaling_coefficients, wavelet_coefficients)
                                             for each decomposition level.
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
    """Aligns two wavelet coefficient arrays to the same length and performs element-wise summation.

    This function is used to combine coefficients from different decomposition paths
    by truncating the longer array to match the length of the shorter one, ensuring
    compatible dimensions for summation.

    Args:
        coeff1 (np.ndarray): The first NumPy array of wavelet coefficients.
        coeff2 (np.ndarray): The second NumPy array of wavelet coefficients.

    Returns:
        np.ndarray: A new NumPy array representing the element-wise sum of the
                    trimmed input arrays.
    """
    min_len = min(len(coeff1), len(coeff2))
    return coeff1[:min_len] + coeff2[:min_len]


def generate_decomposition_signals(coeffs: list[tuple[np.ndarray, np.ndarray]]) -> list[np.ndarray]:
    """Generates 8 specific wavelet decomposition paths from a list of wavelet coefficients.

    These paths represent different combinations of detail (D) and approximation (A) coefficients
    across multiple decomposition levels, crucial for feature extraction in certain DDD models.
    The combinations are based on a predefined structure (e.g., DDD, DDA, ADA, AAA).

    Args:
        coeffs (list): A list of tuples, where each tuple contains (scaling_coefficients, wavelet_coefficients)
                       for each level of wavelet decomposition.

    Returns:
        list[np.ndarray]: A list of 8 NumPy arrays, each representing a specific wavelet
                          decomposition signal (e.g., DDD, DDA, ... AAA).
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
    """Calculates the mean power (mean squared value) of a 1D signal.

    This function is used to quantify the energy content within different
    wavelet decomposition paths, serving as a feature for drowsiness detection.

    Args:
        signal (np.ndarray): The 1D input signal.

    Returns:
        float: The mean power of the signal.
    """
    return np.mean(signal ** 2)


def process_window(signal_window: np.ndarray) -> list[float]:
    """Applies GHM wavelet transform to a single signal window and computes power for each decomposition path.

    This function serves as a core processing step within a sliding window approach,
    transforming raw signal segments into a set of power features based on their
    wavelet decomposition.

    Args:
        signal_window (np.ndarray): A 1D NumPy array representing a segment (window) of a signal.

    Returns:
        list[float]: A list of floating-point numbers, where each value is the
                     calculated power for one of the 8 wavelet decomposition paths.
    """
    coeffs = ghm_wavelet_transform(signal_window)
    decomposition_signals = generate_decomposition_signals(coeffs)
    return [calculate_power(signal) for signal in decomposition_signals]


def wavelet_process(subject: str, model: str, use_jittering: bool = False) -> None:
    """Main function to extract wavelet-based features from vehicle dynamics signals for a given subject.

    This function orchestrates the process of loading SIMlsl data, extracting relevant
    vehicle signals (steering, steering speed, longitudinal acceleration, lateral acceleration,
    and lane offset), applying optional jittering for data augmentation, and then performing
    multi-resolution GHM wavelet decomposition on these signals within sliding windows.
    The power of each decomposition path is computed and saved as features to a CSV file.

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

    save_csv(df, subject_id, version, 'wavelet', model)

