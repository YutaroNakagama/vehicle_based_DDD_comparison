"""Feature importance estimation using ANFIS-inspired Gaussian membership functions.

This module defines functions for computing a fuzzy membership-based
importance score (ID) for features using Gaussian functions.

Used to weight or filter features based on statistical importance.
"""

import numpy as np


def gaussian_membership(x: np.ndarray, c: float, s: float) -> np.ndarray:
    """Compute Gaussian membership values.

    Args:
        x (np.ndarray): Input array (feature values).
        c (float): Center of the Gaussian.
        s (float): Spread (standard deviation) of the Gaussian.

    Returns:
        np.ndarray: Membership values in the range [0, 1].
    """
    return np.exp(-0.5 * ((x - c) / s) ** 2)


def calculate_id(indices: np.ndarray, params: list[float]) -> np.ndarray:
    """Calculate importance degrees (ID) using fuzzy Gaussian membership functions.

    Args:
        indices (pd.DataFrame): DataFrame where each column contains one feature's scores
                                across different selection indices.
        params (list[float]): List of Gaussian parameters. Should be of even length:
                              first half = centers, second half = spreads.

    Returns:
        np.ndarray: Array of maximum membership values per feature, shape (n_features, n_samples).
    """
    c = params[:len(params) // 2]
    s = params[len(params) // 2:]
    ids = []

    for i, col in enumerate(indices.columns):
        # Compute membership for Low, Medium, High
        membership = [gaussian_membership(indices[col], c[j], s[j]) for j in range(3)]
        ids.append(np.max(membership, axis=0))  # take max across L, M, H

    return np.array(ids)

