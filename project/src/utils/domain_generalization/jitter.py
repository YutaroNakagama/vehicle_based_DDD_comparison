"""Jittering augmentation for time-series data.

This module adds Gaussian noise to numeric features in either NumPy arrays or pandas DataFrames.
"""

import numpy as np
import pandas as pd


def jittering(data: pd.DataFrame | np.ndarray, sigma: float = 0.03) -> pd.DataFrame | np.ndarray:
    """Apply jittering (Gaussian noise) to time-series data.

    Jittering is a form of data augmentation that helps improve generalization
    by adding small random perturbations to numeric data.

    Args:
        data (pd.DataFrame or np.ndarray): Input data with shape (n_samples, n_features).
            - If DataFrame: only numeric columns are modified.
        sigma (float, optional): Standard deviation of the Gaussian noise. Defaults to 0.03.

    Returns:
        pd.DataFrame or np.ndarray: Noisy version of the input data, with the same shape and type.

    Raises:
        TypeError: If the input is not a DataFrame or ndarray.
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

