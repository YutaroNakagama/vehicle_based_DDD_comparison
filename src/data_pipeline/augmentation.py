"""Time-series Data Augmentation for Imbalanced Classification.

This module provides augmentation techniques for tabular feature data
derived from time-series signals. Unlike SMOTE (which creates synthetic
samples in feature space), these methods simulate realistic variations
that could occur in the original time-series data.

Supported Methods
-----------------
- Jittering: Add Gaussian noise to features
- Scaling: Multiply features by random scale factor
- Combined: Apply both jittering and scaling

References
----------
Um, T. T., et al. (2017). "Data Augmentation of Wearable Sensor Data for
Parkinson's Disease Monitoring using Convolutional Neural Networks." ICMI.

Note
----
These augmentations are applied to feature vectors (post-extraction),
not raw time-series. For raw time-series augmentation, use
``src/utils/domain_generalization/jitter.py``.
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, Union, Optional

logger = logging.getLogger(__name__)


def jitter_features(
    X: Union[np.ndarray, pd.DataFrame],
    sigma: float = 0.03,
    random_state: Optional[int] = None,
) -> Union[np.ndarray, pd.DataFrame]:
    """Apply Gaussian noise (jittering) to feature values.
    
    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Feature matrix of shape (n_samples, n_features).
    sigma : float, default=0.03
        Standard deviation of noise as a fraction of each feature's std.
    random_state : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    np.ndarray or pd.DataFrame
        Augmented feature matrix with same shape and type as input.
    
    Examples
    --------
    >>> X = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> X_aug = jitter_features(X, sigma=0.1, random_state=42)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    is_dataframe = isinstance(X, pd.DataFrame)
    if is_dataframe:
        X_arr = X.values.astype(float)
        columns = X.columns
        index = X.index
    else:
        X_arr = X.astype(float)
    
    # Scale noise by each feature's standard deviation
    feature_stds = np.std(X_arr, axis=0, keepdims=True)
    feature_stds = np.where(feature_stds == 0, 1.0, feature_stds)  # Avoid div by zero
    
    noise = np.random.normal(0, sigma, X_arr.shape) * feature_stds
    X_aug = X_arr + noise
    
    if is_dataframe:
        return pd.DataFrame(X_aug, columns=columns, index=index)
    return X_aug


def scale_features(
    X: Union[np.ndarray, pd.DataFrame],
    sigma: float = 0.1,
    random_state: Optional[int] = None,
) -> Union[np.ndarray, pd.DataFrame]:
    """Apply random scaling to feature values.
    
    Each sample is multiplied by a random scale factor drawn from
    N(1.0, sigma), simulating amplitude variations in the original signal.
    
    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Feature matrix of shape (n_samples, n_features).
    sigma : float, default=0.1
        Standard deviation of the scaling factor.
    random_state : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    np.ndarray or pd.DataFrame
        Scaled feature matrix with same shape and type as input.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    is_dataframe = isinstance(X, pd.DataFrame)
    if is_dataframe:
        X_arr = X.values.astype(float)
        columns = X.columns
        index = X.index
    else:
        X_arr = X.astype(float)
    
    # One scale factor per sample (not per feature)
    scale_factors = np.random.normal(1.0, sigma, (X_arr.shape[0], 1))
    X_aug = X_arr * scale_factors
    
    if is_dataframe:
        return pd.DataFrame(X_aug, columns=columns, index=index)
    return X_aug


def jitter_scale_features(
    X: Union[np.ndarray, pd.DataFrame],
    jitter_sigma: float = 0.03,
    scale_sigma: float = 0.1,
    random_state: Optional[int] = None,
) -> Union[np.ndarray, pd.DataFrame]:
    """Apply both jittering and scaling augmentation.
    
    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Feature matrix of shape (n_samples, n_features).
    jitter_sigma : float, default=0.03
        Jittering noise level.
    scale_sigma : float, default=0.1
        Scaling variation level.
    random_state : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    np.ndarray or pd.DataFrame
        Augmented feature matrix.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Apply scaling first, then jittering
    X_scaled = scale_features(X, sigma=scale_sigma, random_state=None)
    X_aug = jitter_features(X_scaled, sigma=jitter_sigma, random_state=None)
    
    return X_aug


def augment_minority_class(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    method: str = "jitter_scale",
    target_ratio: float = 0.33,
    jitter_sigma: float = 0.03,
    scale_sigma: float = 0.1,
    random_state: int = 42,
) -> Tuple[Union[np.ndarray, pd.DataFrame], Union[np.ndarray, pd.Series]]:
    """Augment minority class samples to reduce class imbalance.
    
    Creates augmented copies of minority class samples using specified
    augmentation method until the class ratio reaches target_ratio.
    
    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray or pd.Series
        Binary labels (0=majority, 1=minority).
    method : {'jitter', 'scale', 'jitter_scale'}, default='jitter_scale'
        Augmentation method to apply.
    target_ratio : float, default=0.33
        Target ratio of minority to majority class (e.g., 0.33 = 1:3).
    jitter_sigma : float, default=0.03
        Jittering noise level.
    scale_sigma : float, default=0.1
        Scaling variation level.
    random_state : int, default=42
        Random seed for reproducibility.
    
    Returns
    -------
    tuple of (X_augmented, y_augmented)
        Augmented feature matrix and labels.
    
    Examples
    --------
    >>> X, y = augment_minority_class(X_train, y_train, method='jitter_scale')
    >>> print(f"Before: {np.bincount(y_train)}, After: {np.bincount(y)}")
    """
    np.random.seed(random_state)
    
    is_dataframe = isinstance(X, pd.DataFrame)
    is_series = isinstance(y, pd.Series)
    
    if is_dataframe:
        X_arr = X.values.astype(float)
        columns = X.columns
    else:
        X_arr = X.astype(float)
    
    if is_series:
        y_arr = y.values.astype(int)
    else:
        y_arr = y.astype(int)
    
    # Identify minority and majority
    n_minority = (y_arr == 1).sum()
    n_majority = (y_arr == 0).sum()
    
    if n_minority == 0:
        logger.warning("No minority samples found. Returning original data.")
        return X, y
    
    # Calculate how many augmented samples needed
    target_minority = int(n_majority * target_ratio)
    n_augment = max(0, target_minority - n_minority)
    
    if n_augment == 0:
        logger.info("Minority class already at or above target ratio.")
        return X, y
    
    logger.info(f"Augmenting minority class: {n_minority} -> {n_minority + n_augment} "
                f"(target ratio: {target_ratio})")
    
    # Get minority samples
    minority_idx = np.where(y_arr == 1)[0]
    
    # Generate augmented samples
    augmented_X = []
    augmented_y = []
    
    for i in range(n_augment):
        # Randomly select a minority sample to augment
        idx = np.random.choice(minority_idx)
        x_sample = X_arr[idx:idx+1]  # Keep 2D shape
        
        # Apply augmentation
        if method == "jitter":
            x_aug = jitter_features(x_sample, sigma=jitter_sigma)
        elif method == "scale":
            x_aug = scale_features(x_sample, sigma=scale_sigma)
        elif method == "jitter_scale":
            x_aug = jitter_scale_features(
                x_sample, 
                jitter_sigma=jitter_sigma,
                scale_sigma=scale_sigma
            )
        else:
            raise ValueError(f"Unknown augmentation method: {method}")
        
        augmented_X.append(x_aug.flatten())
        augmented_y.append(1)
    
    # Combine original and augmented
    X_combined = np.vstack([X_arr, np.array(augmented_X)])
    y_combined = np.concatenate([y_arr, np.array(augmented_y)])
    
    # Shuffle to mix original and augmented
    shuffle_idx = np.random.permutation(len(y_combined))
    X_combined = X_combined[shuffle_idx]
    y_combined = y_combined[shuffle_idx]
    
    if is_dataframe:
        X_combined = pd.DataFrame(X_combined, columns=columns)
    if is_series:
        y_combined = pd.Series(y_combined, name=y.name)
    
    logger.info(f"Class distribution after augmentation: {np.bincount(y_combined.astype(int))}")
    
    return X_combined, y_combined
