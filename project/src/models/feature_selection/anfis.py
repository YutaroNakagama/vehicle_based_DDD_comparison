"""Feature importance estimation using ANFIS-inspired Gaussian membership functions.

This module defines functions for computing fuzzy membership-based
importance degrees (ID) of features using Gaussian functions.
These IDs can be used to weight or filter features based on
statistical importance.
"""

import numpy as np


def gaussian_membership(x: np.ndarray, c: float, s: float) -> np.ndarray:
    """
    Compute Gaussian membership values.

    Parameters
    ----------
    x : numpy.ndarray
        Input array of feature values.
    c : float
        Center of the Gaussian function.
    s : float
        Spread (standard deviation) of the Gaussian function.

    Returns
    -------
    numpy.ndarray
        Membership values in the range [0, 1] with the same shape as ``x``.
    """
    return np.exp(-0.5 * ((x - c) / s) ** 2)


def calculate_id(indices: np.ndarray, params: list[float]) -> np.ndarray:
    """
    Calculate importance degrees (ID) using fuzzy Gaussian membership functions.

    Parameters
    ----------
    indices : pandas.DataFrame
        DataFrame where each column contains scores of one feature
        across different selection indices.
    params : list of float
        List of Gaussian parameters with even length:
        - First half : centers of Gaussians
        - Second half : spreads of Gaussians

    Returns
    -------
    numpy.ndarray
        Array of maximum membership values per feature,
        shape (n_features, n_samples).
    """
    c = params[:len(params) // 2]
    s = params[len(params) // 2:]
    ids = []

    for i, col in enumerate(indices.columns):
        # Compute membership for Low, Medium, High
        membership = [gaussian_membership(indices[col], c[j], s[j]) for j in range(3)]
        ids.append(np.max(membership, axis=0))  # take max across L, M, H

    return np.array(ids)

