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
- Adaptive: Automatically determine sigma based on data characteristics

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
from typing import Tuple, Union, Optional, Dict

logger = logging.getLogger(__name__)


def estimate_adaptive_sigma(
    X: Union[np.ndarray, pd.DataFrame],
    y: Optional[Union[np.ndarray, pd.Series]] = None,
    method: str = "snr",
) -> Dict[str, float]:
    """Estimate optimal sigma values based on data characteristics.
    
    This function analyzes the input data to determine appropriate
    noise levels for jittering and scaling augmentation.
    
    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray or pd.Series, optional
        Labels for class-aware estimation.
    method : {'snr', 'cv', 'interclass', 'conservative'}, default='snr'
        Method for sigma estimation:
        - 'snr': Based on signal-to-noise ratio of features
        - 'cv': Based on coefficient of variation
        - 'interclass': Based on inter-class distance (requires y)
        - 'conservative': Use conservative (small) values
    
    Returns
    -------
    dict
        Dictionary with 'jitter_sigma' and 'scale_sigma' values.
    
    Examples
    --------
    >>> sigmas = estimate_adaptive_sigma(X_train, y_train, method='interclass')
    >>> print(f"Jitter: {sigmas['jitter_sigma']:.4f}, Scale: {sigmas['scale_sigma']:.4f}")
    """
    if isinstance(X, pd.DataFrame):
        X_arr = X.values.astype(float)
    else:
        X_arr = X.astype(float)
    
    # Handle NaN values
    if np.any(np.isnan(X_arr)):
        logger.warning("NaN values detected in X. Filling with column median.")
        col_medians = np.nanmedian(X_arr, axis=0)
        nan_mask = np.isnan(X_arr)
        for j in range(X_arr.shape[1]):
            X_arr[nan_mask[:, j], j] = col_medians[j] if not np.isnan(col_medians[j]) else 0.0
    
    if y is not None:
        if isinstance(y, pd.Series):
            y_arr = y.values
        else:
            y_arr = y.copy()
        
        # Handle NaN in labels
        if np.any(pd.isna(y_arr)):
            valid_mask = ~pd.isna(y_arr)
            X_arr = X_arr[valid_mask]
            y_arr = y_arr[valid_mask]
            logger.warning(f"Removed {(~valid_mask).sum()} samples with NaN labels.")
        
        y_arr = y_arr.astype(int)
    else:
        y_arr = None
    
    if method == "snr":
        # Signal-to-Noise Ratio based estimation
        # Jitter sigma: based on feature variance ratio
        feature_means = np.abs(np.mean(X_arr, axis=0))
        feature_stds = np.std(X_arr, axis=0)
        
        # Avoid division by zero
        feature_means = np.where(feature_means == 0, 1e-10, feature_means)
        feature_stds = np.where(feature_stds == 0, 1e-10, feature_stds)
        
        # SNR = mean / std, higher SNR means less noise relative to signal
        snr = np.median(feature_means / feature_stds)
        
        # Jitter should be small relative to SNR
        # Low SNR (noisy data) -> smaller jitter to avoid adding more noise
        # High SNR (clean data) -> can tolerate more jitter
        jitter_sigma = np.clip(0.1 / (1 + snr), 0.005, 0.1)
        
        # Scale sigma based on coefficient of variation
        cv = np.median(feature_stds / feature_means)
        scale_sigma = np.clip(cv * 0.5, 0.02, 0.2)
        
    elif method == "cv":
        # Coefficient of Variation based estimation
        feature_means = np.abs(np.mean(X_arr, axis=0))
        feature_stds = np.std(X_arr, axis=0)
        
        feature_means = np.where(feature_means == 0, 1e-10, feature_means)
        
        cv = np.median(feature_stds / feature_means)
        
        # Use fraction of CV as sigma
        jitter_sigma = np.clip(cv * 0.1, 0.01, 0.1)
        scale_sigma = np.clip(cv * 0.3, 0.05, 0.2)
        
    elif method == "interclass":
        # Inter-class distance based estimation (requires labels)
        if y_arr is None:
            logger.warning("Labels required for 'interclass' method. Falling back to 'snr'.")
            return estimate_adaptive_sigma(X_arr, None, method="snr")
        
        # Calculate class centroids
        X_minority = X_arr[y_arr == 1]
        X_majority = X_arr[y_arr == 0]
        
        if len(X_minority) == 0 or len(X_majority) == 0:
            logger.warning("Empty class. Falling back to 'snr' method.")
            return estimate_adaptive_sigma(X_arr, None, method="snr")
        
        centroid_minority = np.mean(X_minority, axis=0)
        centroid_majority = np.mean(X_majority, axis=0)
        
        # Inter-class distance
        interclass_dist = np.linalg.norm(centroid_minority - centroid_majority)
        
        # Intra-class std of minority
        intraclass_std = np.mean(np.std(X_minority, axis=0))
        
        # Augmented samples should stay within minority distribution
        # but not overlap too much with majority
        # Sigma should be fraction of intra-class variation
        jitter_sigma = np.clip(intraclass_std / interclass_dist * 0.5, 0.005, 0.1)
        
        # Scale sigma based on minority class variation
        minority_cv = np.median(np.std(X_minority, axis=0) / (np.abs(np.mean(X_minority, axis=0)) + 1e-10))
        scale_sigma = np.clip(minority_cv * 0.3, 0.02, 0.15)
        
    elif method == "conservative":
        # Conservative values that are unlikely to cause issues
        jitter_sigma = 0.01
        scale_sigma = 0.05
        
    else:
        raise ValueError(f"Unknown method: {method}. Use 'snr', 'cv', 'interclass', or 'conservative'.")
    
    result = {
        "jitter_sigma": float(jitter_sigma),
        "scale_sigma": float(scale_sigma),
        "method": method,
    }
    
    logger.info(f"[Adaptive Sigma] method={method}, jitter_sigma={jitter_sigma:.4f}, scale_sigma={scale_sigma:.4f}")
    
    return result


def analyze_augmentation_quality(
    X_original: np.ndarray,
    X_augmented: np.ndarray,
    y_original: np.ndarray,
) -> Dict[str, float]:
    """Analyze the quality of augmented samples.
    
    Checks if augmented samples maintain the distribution characteristics
    of the original minority class.
    
    Parameters
    ----------
    X_original : np.ndarray
        Original feature matrix.
    X_augmented : np.ndarray
        Augmented feature matrix.
    y_original : np.ndarray
        Original labels.
    
    Returns
    -------
    dict
        Quality metrics including:
        - mean_shift: How much the minority centroid moved
        - std_change: Change in minority class standard deviation
        - overlap_score: Estimated overlap with majority class
    """
    # Original minority
    X_orig_minority = X_original[y_original == 1]
    
    # Find new augmented samples (those in augmented but not in original)
    n_orig = len(X_original)
    X_new = X_augmented[n_orig:]  # New samples are appended
    
    if len(X_new) == 0:
        return {"mean_shift": 0.0, "std_change": 0.0, "overlap_score": 0.0}
    
    # Mean shift
    orig_centroid = np.mean(X_orig_minority, axis=0)
    new_centroid = np.mean(X_new, axis=0)
    mean_shift = np.linalg.norm(new_centroid - orig_centroid) / (np.linalg.norm(orig_centroid) + 1e-10)
    
    # Std change
    orig_std = np.mean(np.std(X_orig_minority, axis=0))
    new_std = np.mean(np.std(X_new, axis=0))
    std_change = (new_std - orig_std) / (orig_std + 1e-10)
    
    # Overlap with majority (using simple distance heuristic)
    X_orig_majority = X_original[y_original == 0]
    majority_centroid = np.mean(X_orig_majority, axis=0)
    
    dist_to_minority = np.mean([np.linalg.norm(x - orig_centroid) for x in X_new])
    dist_to_majority = np.mean([np.linalg.norm(x - majority_centroid) for x in X_new])
    
    # Overlap score: closer to 1 means more overlap with majority (bad)
    overlap_score = dist_to_minority / (dist_to_majority + 1e-10)
    
    return {
        "mean_shift": float(mean_shift),
        "std_change": float(std_change),
        "overlap_score": float(overlap_score),
    }


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
    jitter_sigma: Optional[float] = None,
    scale_sigma: Optional[float] = None,
    adaptive_sigma: str = "none",
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
    jitter_sigma : float, optional
        Jittering noise level. If None and adaptive_sigma='none', uses 0.03.
    scale_sigma : float, optional
        Scaling variation level. If None and adaptive_sigma='none', uses 0.1.
    adaptive_sigma : {'none', 'snr', 'cv', 'interclass', 'conservative'}, default='none'
        Method for automatic sigma estimation:
        - 'none': Use provided or default sigma values
        - 'snr': Signal-to-noise ratio based
        - 'cv': Coefficient of variation based
        - 'interclass': Inter-class distance based (recommended)
        - 'conservative': Small, safe values
    random_state : int, default=42
        Random seed for reproducibility.
    
    Returns
    -------
    tuple of (X_augmented, y_augmented)
        Augmented feature matrix and labels.
    
    Examples
    --------
    >>> # Using adaptive sigma (recommended)
    >>> X, y = augment_minority_class(X_train, y_train, adaptive_sigma='interclass')
    
    >>> # Using fixed sigma
    >>> X, y = augment_minority_class(X_train, y_train, jitter_sigma=0.05, scale_sigma=0.1)
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
    
    # Determine sigma values
    if adaptive_sigma != "none":
        # Use adaptive sigma estimation
        sigma_dict = estimate_adaptive_sigma(X_arr, y_arr, method=adaptive_sigma)
        _jitter_sigma = sigma_dict["jitter_sigma"]
        _scale_sigma = sigma_dict["scale_sigma"]
        logger.info(f"[Adaptive] Using estimated sigmas: jitter={_jitter_sigma:.4f}, scale={_scale_sigma:.4f}")
    else:
        # Use provided or default values
        _jitter_sigma = jitter_sigma if jitter_sigma is not None else 0.03
        _scale_sigma = scale_sigma if scale_sigma is not None else 0.1
        logger.info(f"[Fixed] Using sigmas: jitter={_jitter_sigma:.4f}, scale={_scale_sigma:.4f}")
    
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
            x_aug = jitter_features(x_sample, sigma=_jitter_sigma)
        elif method == "scale":
            x_aug = scale_features(x_sample, sigma=_scale_sigma)
        elif method == "jitter_scale":
            x_aug = jitter_scale_features(
                x_sample, 
                jitter_sigma=_jitter_sigma,
                scale_sigma=_scale_sigma
            )
        else:
            raise ValueError(f"Unknown augmentation method: {method}")
        
        augmented_X.append(x_aug.flatten())
        augmented_y.append(1)
    
    # Combine original and augmented
    X_combined = np.vstack([X_arr, np.array(augmented_X)])
    y_combined = np.concatenate([y_arr, np.array(augmented_y)])
    
    # Analyze augmentation quality
    quality = analyze_augmentation_quality(X_arr, X_combined, y_arr, y_combined)
    logger.info(f"[Augmentation Quality] mean_shift={quality['mean_shift']:.4f}, "
                f"std_change={quality['std_change']:.4f}, overlap_score={quality['overlap_score']:.4f}")
    
    # Warn if quality is poor
    if quality['overlap_score'] > 0.8:
        logger.warning("Augmented samples may overlap significantly with majority class. "
                       "Consider using smaller sigma or 'conservative' adaptive method.")
    
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
