import numpy as np
import scipy.linalg

def coral(source, target):
    """
    CORrelation ALignment (CORAL) implementation.

    Parameters:
    ----------
    source: np.ndarray, shape (n_samples_source, n_features)
        Source domain feature data.

    target: np.ndarray, shape (n_samples_target, n_features)
        Target domain feature data.

    Returns:
    --------
    source_aligned: np.ndarray, shape (n_samples_source, n_features)
        Source data aligned to the target domain.
    """
    # Compute covariance matrices
    cov_source = np.cov(source, rowvar=False) + np.eye(source.shape[1])
    cov_target = np.cov(target, rowvar=False) + np.eye(target.shape[1])

    # Calculate the CORAL transformation
    A_coral = scipy.linalg.fractional_matrix_power(cov_source, -0.5) @ \
              scipy.linalg.fractional_matrix_power(cov_target, 0.5)

    # Align the source domain data to the target domain
    source_aligned = (source - source.mean(axis=0)) @ A_coral + target.mean(axis=0)

    return source_aligned

