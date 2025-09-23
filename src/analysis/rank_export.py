# -*- coding: utf-8 -*-
"""Rank exporter for subject-wise mean/std from distance matrices.

This module computes per-subject mean and std (excluding the diagonal) for each
distance type (MMD, Wasserstein, DTW), ranks subjects, and writes top/bottom-k
lists with neutral, research-friendly filenames:

- {metric}_mean_low.txt      : subjects with smallest mean distance (k)
- {metric}_mean_middle.txt   : subjects closest to the median mean  (k)
- {metric}_mean_high.txt     : subjects with largest mean distance  (k)
- {metric}_std_low.txt       : subjects with smallest std           (k)
- {metric}_std_middle.txt    : subjects closest to the median std   (k)
- {metric}_std_high.txt      : subjects with largest std            (k)


Notes
-----
- Distance matrices may be .npy or .csv (square numeric matrix).
- Subject list JSON may be a list or {"subjects": [...]}.
- Diagonal entries are excluded via NaN masking.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def _load_subject_names(path: Path) -> List[str]:
    """Load subject names from a JSON file.

    Supports both list format and dictionary format ``{"subjects": [...]}``.

    Parameters
    ----------
    path : Path
        Path to the JSON file containing subject names.

    Returns
    -------
    list of str
        List of subject names.

    Raises
    ------
    ValueError
        If the JSON format is not supported.
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "subjects" in data:
        return list(data["subjects"])
    if isinstance(data, list):
        return list(data)
    raise ValueError(f"Unsupported subjects json format: {path}")


def _load_matrix(path: Path) -> np.ndarray:
    """Load a square distance matrix from file.

    Parameters
    ----------
    path : Path
        Path to the matrix file. Supported formats are:
        - ``.npy``: NumPy binary format
        - ``.csv``: CSV file without header

    Returns
    -------
    ndarray of shape (n_subjects, n_subjects)
        Square distance matrix.

    Raises
    ------
    ValueError
        If the file type is unsupported or the array is not square.
    """
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path, header=None)
        arr = df.values
    else:
        raise ValueError(f"Unsupported matrix file type: {path}")
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Matrix must be square 2D array: {path}, shape={arr.shape}")
    return arr.astype(float)


def _compute_stats(m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-subject mean and standard deviation of distances.

    The diagonal entries are excluded from the calculation.

    Parameters
    ----------
    m : ndarray of shape (n_subjects, n_subjects)
        Square distance matrix.

    Returns
    -------
    tuple of (ndarray, ndarray)
        - Mean distances per subject.
        - Standard deviations per subject.
    """
    a = m.copy().astype(float)
    np.fill_diagonal(a, np.nan)
    mean = np.nanmean(a, axis=1)
    std = np.nanstd(a, axis=1)
    return mean, std

def _write_rank_with_middle(
    names: List[str],
    values: np.ndarray,
    k: int,
    out_low: Path,
    out_middle: Path,
    out_high: Path,
) -> None:
    """Write ranked subject lists (low, middle, high) to text files.

    Parameters
    ----------
    names : list of str
        Subject names.
    values : ndarray
        Values to rank (e.g., mean distances).
    k : int
        Number of subjects to include in each category.
    out_low : Path
        Output path for the lowest-k subjects.
    out_middle : Path
        Output path for the middle-k subjects closest to the median.
    out_high : Path
        Output path for the highest-k subjects.

    Returns
    -------
    None
    """
    order = np.argsort(values)  # ascending
    low_idx = order[:k]
    high_idx = order[-k:][::-1]

    median_val = np.median(values)
    middle_idx = np.argsort(np.abs(values - median_val))[:k]

    for path, idx_list in (
        (out_low, low_idx),
        (out_middle, middle_idx),
        (out_high, high_idx),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            # Join subject IDs in one line separated by spaces
            f.write(" ".join(names[i] for i in idx_list))
            f.write("\n")

def _write_uniform(names: List[str], k: int, out_path: Path) -> None:
    """Write uniformly sampled subject names to a text file.

    Parameters
    ----------
    names : list of str
        Ordered list of subject names.
    k : int
        Number of subjects to sample uniformly.
    out_path : Path
        Output file path.

    Returns
    -------
    None
    """
    N = len(names)
    if k >= N:
        indices = list(range(N))
    else:
        step = N // k
        indices = list(range(0, N, step))[:k]
    selected = [names[i] for i in indices]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(" ".join(selected))
        f.write("\n")

def _export_one(metric: str, mat_path: Optional[Path], subjects_path: Optional[Path], outdir: Path, k: int) -> None:
    """Export ranking files for one distance metric.

    Parameters
    ----------
    metric : str
        Name of the metric (e.g., ``mmd``, ``wasserstein``, ``dtw``).
    mat_path : Path, optional
        Path to the distance matrix file.
    subjects_path : Path, optional
        Path to the JSON file with subject names.
    outdir : Path
        Output directory for generated files.
    k : int
        Number of subjects per category (low/middle/high).

    Returns
    -------
    None
    """
    if mat_path is None or subjects_path is None:
        return
    if not mat_path.exists() or not subjects_path.exists():
        return

    names = _load_subject_names(subjects_path)
    mat = _load_matrix(mat_path)
    if len(names) != mat.shape[0]:
        raise ValueError(
            f"Subjects length ({len(names)}) != matrix size ({mat.shape[0]}) for {metric}"
        )

    mean, std = _compute_stats(mat)

    # mean-based (low / middle / high)
    _write_rank_with_middle(
        names,
        mean,
        k,
        outdir / f"{metric}_mean_low.txt",
        outdir / f"{metric}_mean_middle.txt",
        outdir / f"{metric}_mean_high.txt",
    )
    # std-based (low / middle / high)
    _write_rank_with_middle(
        names,
        std,
        k,
        outdir / f"{metric}_std_low.txt",
        outdir / f"{metric}_std_middle.txt",
        outdir / f"{metric}_std_high.txt",
    )

    order = np.argsort(mean)  # ascending
    ordered_names = [names[i] for i in order]
    _write_uniform(
        ordered_names,
        k,
        outdir / f"{metric}_mean_uniform.txt",
    )

def run_rank_export(
    *,
    outdir: Path,
    k: int = 10,
    # MMD
    mmd_matrix: Optional[Path] = None,
    mmd_subjects: Optional[Path] = None,
    # Wasserstein / DTW
    wasserstein_matrix: Optional[Path] = None,
    dtw_matrix: Optional[Path] = None,
    dist_subjects: Optional[Path] = None,
) -> int:
    """Export subject rankings for all distance metrics.

    Parameters
    ----------
    outdir : Path
        Directory to save the ranking files.
    k : int, default=10
        Number of subjects to export for each category.
    mmd_matrix, mmd_subjects : Path, optional
        Paths to the MMD distance matrix and subjects JSON.
    wasserstein_matrix, dtw_matrix, dist_subjects : Path, optional
        Paths to Wasserstein/DTW distance matrices and shared subjects JSON.

    Returns
    -------
    int
        0 on success.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # MMD (uses its own subjects json)
    _export_one("mmd", mmd_matrix, mmd_subjects, outdir, k)

    # Wasserstein / DTW (share subjects)
    _export_one("wasserstein", wasserstein_matrix, dist_subjects, outdir, k)
    _export_one("dtw", dtw_matrix, dist_subjects, outdir, k)

    return 0

