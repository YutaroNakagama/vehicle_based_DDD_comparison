"""Jittering augmentation for time-series data.

This module applies Gaussian noise to numeric features
in NumPy arrays or pandas DataFrames to improve robustness.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def jittering(data: pd.DataFrame | np.ndarray, sigma: float = 0.03) -> pd.DataFrame | np.ndarray:
    """
    Apply jittering (Gaussian noise) to time-series data.

    Jittering is a lightweight data augmentation technique
    that perturbs numeric values with small random noise
    to improve model generalization.

    Parameters
    ----------
    data : pandas.DataFrame or numpy.ndarray of shape (n_samples, n_features)
        Input dataset.  
        - If DataFrame: only numeric columns are perturbed.  
        - If ndarray: all values are perturbed.
    sigma : float, default=0.03
        Standard deviation of the Gaussian noise.

    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        Noisy version of the input data, same shape and type as input.

    Raises
    ------
    TypeError
        If the input is not a ``pandas.DataFrame`` or ``numpy.ndarray``.
    """
    if isinstance(data, pd.DataFrame):
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        noisy_data = data.copy()

        noise = np.random.normal(0, sigma, size=noisy_data[numeric_columns].shape)
        noisy_data[numeric_columns] += noise

        return noisy_data

    elif isinstance(data, np.ndarray):
        noise = np.random.normal(0, sigma, size=data.shape)
        return data + noise

    else:
        raise TypeError("Input data must be a pd.DataFrame or np.ndarray.")

