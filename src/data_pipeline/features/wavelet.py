"""GHM multiwavelet packet energy spectrum for vehicle dynamics signals.

Implements the feature extraction from:
    Zhao, S., Xu, G., & Tao, T. (2009). Detecting of Driver's Drowsiness
    Using Multiwavelet Packet Energy Spectrum. IEEE APCCAS.

GHM (Geronimo-Hardin-Massopust) multiwavelet with 2x2 matrix filter bank,
3-level packet decomposition -> 8 frequency bands, relative band energy.

Notes
-----
- GHM uses multiplicity r=2 (two scaling + two wavelet functions).
- Prefiltering converts scalar signal -> 2-component vector (even/odd split).
- 3-level packet decomposition produces 2^3 = 8 nodes.
- Band energy = mean squared value across both components per node.
- Relative normalization: each band energy as fraction of total (Eq. 12).
"""

import numpy as np
import pandas as pd
import os
import logging
from joblib import Parallel, delayed

from src.utils.io.loaders import safe_load_mat, save_csv
from src.data_pipeline.augmentation.jitter import jittering
from src.config import (
    DATASET_PATH,
    SAMPLE_RATE_SIMLSL,
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


# ============================================================================
# GHM Multiwavelet Filter Coefficients (2x2 matrix filter bank)
# Ref: Geronimo, Hardin & Massopust (1994), SIAM J. Math. Anal. 27, 1158-1192
# Each tap is a 2x2 matrix; 4 taps for lowpass (H) and highpass (G).
# ============================================================================
_S2 = np.sqrt(2)

GHM_H = np.array([
    [[ 3/(5*_S2),  4/(5*_S2)],
     [-1/(20*_S2), -3/(20*_S2)]],
    [[ 3/(5*_S2),  0],
     [ 9/(20*_S2), 1/_S2]],
    [[ 0,          0],
     [ 9/(20*_S2), -3/(20*_S2)]],
    [[ 0,          0],
     [-1/(20*_S2),  0]],
])  # shape (4, 2, 2)

GHM_G = np.array([
    [[-1/(20*_S2),  3/(20*_S2)],
     [ 3/(5*_S2),  -4/(5*_S2)]],
    [[ 9/(20*_S2), -1/_S2],
     [-3/(5*_S2),   0]],
    [[ 9/(20*_S2),  3/(20*_S2)],
     [ 0,           0]],
    [[-1/(20*_S2),  0],
     [ 0,           0]],
])  # shape (4, 2, 2)


def _prefilter(signal: np.ndarray) -> np.ndarray:
    """Convert scalar signal to 2-component vector via even/odd split.

    Parameters
    ----------
    signal : ndarray, shape (N,)
        Input scalar signal.

    Returns
    -------
    ndarray, shape (2, N//2)
        Two-component vector signal.
    """
    N = len(signal)
    if N % 2 != 0:
        signal = signal[:N - 1]
    return np.array([signal[0::2], signal[1::2]])


def _mw_decompose_one_level(v_signal: np.ndarray) -> tuple:
    """One level of GHM multiwavelet decomposition.

    Applies 2x2 matrix filter bank (lowpass H, highpass G)
    and downsamples by 2.

    Parameters
    ----------
    v_signal : ndarray, shape (2, L)
        Two-component vector signal.

    Returns
    -------
    tuple of (ndarray, ndarray)
        (low, high) each shape (2, L//2).
    """
    L = v_signal.shape[1]
    n_taps = GHM_H.shape[0]  # 4
    out_len = max(L // 2, 1)
    low = np.zeros((2, out_len))
    high = np.zeros((2, out_len))
    for n in range(out_len):
        for k in range(n_taps):
            idx = 2 * n + k
            if idx < L:
                low[:, n] += GHM_H[k] @ v_signal[:, idx]
                high[:, n] += GHM_G[k] @ v_signal[:, idx]
    return low, high


def ghm_wavelet_packet(signal: np.ndarray, n_levels: int = 3) -> list:
    """GHM multiwavelet packet decomposition.

    Recursively decomposes both low-pass and high-pass nodes at each
    level, producing 2^n_levels leaf nodes.

    Parameters
    ----------
    signal : ndarray, shape (N,)
        Input 1-D scalar signal.
    n_levels : int, default 3
        Number of decomposition levels.

    Returns
    -------
    list of ndarray
        2^n_levels coefficient arrays, each shape (2, M).
        Ordered from lowest to highest frequency (AAA ... DDD).
    """
    v_signal = _prefilter(signal)
    nodes = [v_signal]
    for _ in range(n_levels):
        new_nodes = []
        for node in nodes:
            if node.shape[1] < 4:
                new_nodes.extend([node, np.zeros_like(node)])
            else:
                low, high = _mw_decompose_one_level(node)
                new_nodes.append(low)
                new_nodes.append(high)
        nodes = new_nodes
    return nodes


def process_window(signal_window: np.ndarray) -> list[float]:
    """Compute relative band energies via GHM multiwavelet packet.

    Parameters
    ----------
    signal_window : ndarray
        A 1-D signal segment.

    Returns
    -------
    list of float
        Relative energy for 8 decomposition bands (DDD ... AAA order).
    """
    nodes = ghm_wavelet_packet(signal_window, n_levels=WAVELET_LEV)
    energies = [float(np.mean(node ** 2)) for node in nodes]
    total = sum(energies) + 1e-10
    # Reverse: natural order is AAA->DDD; paper convention is DDD->AAA
    return [e / total for e in reversed(energies)]


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

    # Zhao et al. 2009: decompose steering wheel angle only.
    # The paper decomposes entire driving sessions; we use sliding windows
    # (10 s, 50 % overlap) as an adaptation for continuous recordings.
    steering = np.nan_to_num(sim_data[29, :])

    signals = {
        'SteeringWheel': steering,
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

