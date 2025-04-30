"""CORrelation ALignment (CORAL) domain adaptation implementation.

This module provides a function to align source domain feature distributions
to match the target domain by whitening and recoloring using covariance matrices.
"""

import numpy as np
import scipy.linalg


def coral(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Apply CORAL to align source features to the target domain.

    This technique performs feature distribution alignment by
    adjusting the covariance of the source domain to match the target domain.

    Args:
        source (np.ndarray): Source domain data of shape (n_samples_source, n_features).
        target (np.ndarray): Target domain data of shape (n_samples_target, n_features).

    Returns:
        np.ndarray: CORAL-aligned source data with same shape as input.
    """
    # Compute covariance matrices with identity regularization
    cov_source = np.cov(source, rowvar=False) + np.eye(source.shape[1])
    cov_target = np.cov(target, rowvar=False) + np.eye(target.shape[1])

    # Compute transformation matrix A_coral = Cs^{-1/2} Ct^{1/2}
    A_coral = scipy.linalg.fractional_matrix_power(cov_source, -0.5) @ \
              scipy.linalg.fractional_matrix_power(cov_target, 0.5)

    # Center source, apply transformation, re-center to target mean
    source_aligned = (source - source.mean(axis=0)) @ A_coral + target.mean(axis=0)

    return source_aligned

