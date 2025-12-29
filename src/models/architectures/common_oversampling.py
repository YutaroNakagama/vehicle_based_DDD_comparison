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
    subject_wise: bool = False,
    subject_ids: Optional[pd.Series] = None,
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
    subject_wise : bool, default=False
        If True, apply oversampling separately for each subject to avoid
        generating synthetic samples across different subjects' data.
    subject_ids : pd.Series, optional
        Subject IDs for each sample. Required if subject_wise=True.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Resampled (X_train, y_train).
    """
    logging.info(f"Applying oversampling method: {method}")
    logging.info(f"Class distribution before oversampling: {np.bincount(y_train)}")
    logging.info(f"Target ratio (minority/majority): {target_ratio}")
    logging.info(f"Subject-wise oversampling: {subject_wise}")

    # Use subject-wise oversampling if requested
    if subject_wise:
        if subject_ids is None:
            raise ValueError("subject_ids must be provided when subject_wise=True")
        return _apply_subjectwise_oversampling(
            X_train, y_train, subject_ids,
            method=method,
            target_ratio=target_ratio,
            random_state=random_state,
        )

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


def _apply_subjectwise_oversampling(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    subject_ids: pd.Series,
    method: str = "smote",
    target_ratio: float = 0.33,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Apply oversampling separately for each subject.

    This prevents generating synthetic samples that mix characteristics
    from different subjects, which could introduce noise.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels.
    subject_ids : pd.Series
        Subject ID for each sample.
    method : str, default="smote"
        Sampling method to apply.
    target_ratio : float, default=0.33
        Target minority/majority ratio after sampling.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Resampled (X_train, y_train).
    """
    unique_subjects = subject_ids.unique()
    logging.info(f"[Subject-wise Oversampling] Processing {len(unique_subjects)} subjects")

    resampled_X_list = []
    resampled_y_list = []
    skipped_subjects = []

    for i, subject in enumerate(unique_subjects):
        mask = subject_ids == subject
        X_subj = X_train.loc[mask].copy()
        y_subj = y_train.loc[mask].copy()

        # Check class distribution for this subject
        class_counts = np.bincount(y_subj, minlength=2)
        minority_count = class_counts.min()
        majority_count = class_counts.max()

        # Skip if subject has no minority samples or insufficient samples for SMOTE
        if minority_count == 0:
            logging.debug(f"  Subject {subject}: No minority samples, keeping original data")
            resampled_X_list.append(X_subj)
            resampled_y_list.append(y_subj)
            skipped_subjects.append(subject)
            continue

        # For SMOTE-based methods, need at least k_neighbors + 1 minority samples
        min_samples_needed = 2  # At least 2 samples needed for k_neighbors=1
        if minority_count < min_samples_needed:
            logging.debug(f"  Subject {subject}: Only {minority_count} minority samples, keeping original")
            resampled_X_list.append(X_subj)
            resampled_y_list.append(y_subj)
            skipped_subjects.append(subject)
            continue

        # Get sampler for this subject (with adjusted k_neighbors)
        sampler = _get_sampler(method, minority_count, target_ratio, random_state + i)

        if sampler is None:
            # jitter/scale methods
            X_subj_resampled, y_subj_resampled = augment_minority_class(
                X_subj, y_subj,
                method=method,
                target_ratio=target_ratio,
                adaptive_sigma="interclass",
                random_state=random_state + i,
            )
        else:
            try:
                X_subj_resampled, y_subj_resampled = sampler.fit_resample(X_subj, y_subj)
            except ValueError as e:
                # Handle edge cases (e.g., not enough neighbors)
                logging.warning(f"  Subject {subject}: Oversampling failed ({e}), keeping original")
                resampled_X_list.append(X_subj)
                resampled_y_list.append(y_subj)
                skipped_subjects.append(subject)
                continue

        resampled_X_list.append(pd.DataFrame(X_subj_resampled, columns=X_subj.columns))
        resampled_y_list.append(pd.Series(y_subj_resampled))

        new_minority = np.bincount(y_subj_resampled, minlength=2).min()
        logging.debug(f"  Subject {subject}: {minority_count} -> {new_minority} minority samples")

    # Concatenate all resampled data
    X_resampled = pd.concat(resampled_X_list, ignore_index=True)
    y_resampled = pd.concat(resampled_y_list, ignore_index=True)

    final_counts = np.bincount(y_resampled)
    logging.info(f"[Subject-wise Oversampling] Complete:")
    logging.info(f"  Subjects processed: {len(unique_subjects) - len(skipped_subjects)}/{len(unique_subjects)}")
    logging.info(f"  Subjects skipped (insufficient samples): {len(skipped_subjects)}")
    logging.info(f"  Class distribution after: {final_counts}")
    logging.info(f"  Total samples: {len(X_train)} -> {len(X_resampled)}")

    return X_resampled, y_resampled
