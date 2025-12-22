"""Oversampling and undersampling methods for class imbalance handling.

This module provides a unified interface for applying various sampling
strategies to address class imbalance in training data.
"""

import logging
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline

from src.data_pipeline.augmentation import augment_minority_class


def apply_oversampling(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str = "smote",
    target_ratio: float = 0.33,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Apply oversampling or undersampling to training data.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels.
    method : str, default="smote"
        Sampling method. Supported: "smote", "adasyn", "borderline",
        "smote_tomek", "smote_enn", "smote_rus", "undersample_rus",
        "undersample_tomek", "undersample_enn", "jitter", "scale", "jitter_scale".
    target_ratio : float, default=0.33
        Target minority/majority ratio after sampling.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Resampled (X_train, y_train).
    """
    logging.info(f"Applying oversampling method: {method}")
    logging.info(f"Class distribution before oversampling: {np.bincount(y_train)}")
    logging.info(f"Target ratio (minority/majority): {target_ratio}")

    minority_count = np.bincount(y_train).min()
    
    sampler = _get_sampler(method, minority_count, target_ratio, random_state)

    if sampler is None:
        # jitter/scale methods handled separately
        X_train, y_train = augment_minority_class(
            X_train, y_train,
            method=method,
            target_ratio=target_ratio,
            adaptive_sigma="interclass",
            random_state=random_state,
        )
    else:
        X_train, y_train = sampler.fit_resample(X_train, y_train)

    logging.info(f"Class distribution after oversampling: {np.bincount(y_train)}")
    logging.info(f"Oversampling ratio: minority increased from {minority_count} to {np.bincount(y_train).min()}")

    return X_train, y_train


def _get_sampler(
    method: str,
    minority_count: int,
    target_ratio: float,
    random_state: int,
) -> Optional[object]:
    """Get the appropriate sampler object based on method name.

    Parameters
    ----------
    method : str
        Sampling method name.
    minority_count : int
        Current minority class count.
    target_ratio : float
        Target minority/majority ratio.
    random_state : int
        Random seed.

    Returns
    -------
    Optional[object]
        Sampler object, or None for augmentation methods.

    Raises
    ------
    ValueError
        If method is unknown.
    """
    k_neighbors = min(5, minority_count - 1)

    if method == "smote":
        return SMOTE(
            sampling_strategy=target_ratio,
            random_state=random_state,
            k_neighbors=k_neighbors,
        )
    elif method == "adasyn":
        return ADASYN(
            sampling_strategy=target_ratio,
            random_state=random_state,
            n_neighbors=k_neighbors,
        )
    elif method == "borderline":
        return BorderlineSMOTE(
            sampling_strategy=target_ratio,
            random_state=random_state,
            k_neighbors=k_neighbors,
        )
    elif method == "smote_tomek":
        logging.info("Using SMOTE + Tomek Links (boundary cleaning)")
        return SMOTETomek(
            sampling_strategy=target_ratio,
            random_state=random_state,
            n_jobs=1,
            smote=SMOTE(random_state=random_state, k_neighbors=k_neighbors),
        )
    elif method == "smote_enn":
        logging.info("Using SMOTE + ENN (aggressive noise cleaning)")
        return SMOTEENN(
            sampling_strategy=target_ratio,
            random_state=random_state,
            n_jobs=1,
            smote=SMOTE(random_state=random_state, k_neighbors=k_neighbors),
        )
    elif method == "smote_rus":
        logging.info("Using SMOTE + RandomUnderSampler (hybrid sampling)")
        smote = SMOTE(
            sampling_strategy=0.5,
            random_state=random_state,
            k_neighbors=k_neighbors,
        )
        rus = RandomUnderSampler(
            sampling_strategy=0.8,
            random_state=random_state,
        )
        return ImbPipeline([('smote', smote), ('rus', rus)])
    elif method == "undersample_rus":
        logging.info(f"Using Random Under-Sampling only (target ratio: {target_ratio})")
        return RandomUnderSampler(
            sampling_strategy=target_ratio,
            random_state=random_state,
        )
    elif method == "undersample_tomek":
        logging.info("Using Tomek Links only (boundary cleaning)")
        return TomekLinks(sampling_strategy='majority', n_jobs=1)
    elif method == "undersample_enn":
        logging.info("Using Edited Nearest Neighbours (aggressive noise cleaning)")
        return EditedNearestNeighbours(
            sampling_strategy='majority',
            n_neighbors=3,
            kind_sel='all',
            n_jobs=1,
        )
    elif method in ["jitter", "scale", "jitter_scale"]:
        logging.info(f"Using time-series augmentation: {method}")
        return None
    else:
        raise ValueError(f"Unknown oversample_method: {method}")
