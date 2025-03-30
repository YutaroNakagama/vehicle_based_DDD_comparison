import numpy as np
import pandas as pd

def jittering(data, sigma=0.03):
    """
    Apply jittering (Gaussian noise) augmentation to time-series data.

    Parameters:
    ----------
    data : pd.DataFrame or np.ndarray
        Original time-series data (2D: samples × features).

    sigma : float, default=0.03
        Standard deviation of the Gaussian noise to be added.
        Higher values imply more noise.

    Returns:
    --------
    pd.DataFrame or np.ndarray
        Augmented data with Gaussian noise (same shape as input).
    """

    if isinstance(data, pd.DataFrame):
        # Select numeric columns only
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        noisy_data = data.copy()

        # Generate Gaussian noise and add it to numeric data
        noise = np.random.normal(0, sigma, size=noisy_data[numeric_columns].shape)
        noisy_data[numeric_columns] += noise

        return noisy_data

    elif isinstance(data, np.ndarray):
        # If data is numpy array, add noise directly
        noise = np.random.normal(0, sigma, size=data.shape)
        return data + noise

    else:
        raise TypeError("Input data must be a pd.DataFrame or np.ndarray.")
