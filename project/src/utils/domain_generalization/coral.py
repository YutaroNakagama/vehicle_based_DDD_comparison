"""CORrelation ALignment (CORAL) domain adaptation.

This module implements the CORAL algorithm, which aligns the feature
distribution of a source domain to match that of a target domain by
adjusting their covariance structures.
"""

import numpy as np
import scipy.linalg


def coral(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Apply CORAL to align source features to the target domain.

    CORAL (CORrelation ALignment) performs second-order feature
    alignment by whitening the source distribution and re-coloring it
    with the target distribution's covariance.

    Parameters
    ----------
    source : np.ndarray of shape (n_samples_source, n_features)
        Source domain feature matrix.
    target : np.ndarray of shape (n_samples_target, n_features)
        Target domain feature matrix.

    Returns
    -------
    np.ndarray of shape (n_samples_source, n_features)
        Source domain features transformed to be aligned with the
        target domain distribution.
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

