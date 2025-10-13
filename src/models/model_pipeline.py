"""
Model Training Pipeline for Driver Drowsiness Detection (DDD).

This module exposes `train_pipeline`, the central entry point called from
`scripts/python/train.py`. The pipeline is intentionally thin and delegates
to focused helpers for each stage:

- Subject/temporal split selection
- Optional time-stratified splitting
- Feature selection (RF / MI / ANOVA)
- Model-specific training (RF, SVM, LSTM)
- Artifact saving (models, scalers, selected features, metrics)

Design goals
------------
- Keep public signature compatible with train.py (HPC jobs).
- Keep outputs compatible with existing directories.
- Keep internal logic easy to unit test by factoring major stages.
"""

from __future__ import annotations

import os
import json
import pickle
import logging
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import (
    SelectKBest, mutual_info_classif, f_classif
)

from src.config import TOP_K_FEATURES
from src.utils.io.loaders import read_subject_list, load_subject_csvs
from src.utils.io.split import _check_nonfinite
from src.utils.io.split_helpers import split_data, log_split_ratios
from src.utils.io.feature_utils import normalize_feature_names
from src.models.architectures.helpers import get_classifier
from src.models.architectures.SvmA import SvmA_train
from src.models.architectures.lstm import lstm_train
from src.models.architectures.common import common_train
from src.models.architectures.train_helpers import train_model
from src.utils.io.savers import save_artifacts
from src.models.feature_selection.feature_helpers import select_features_and_scale
from src.models.feature_selection.rf_importance import select_top_features_by_importance


logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# =========================
# Public API
# =========================
def train_pipeline(
    model_name: str,
    mode: Optional[str] = None,
    subject_split_strategy: str = "random",
    target_subjects: Optional[List[str]] = None,
    subject_wise_split: bool = False,  # Kept for compatibility; not used in this minimal pipeline
    seed: int = 42,
    tag: Optional[str] = None,
    time_stratify_labels: bool = False,
    time_stratify_tolerance: float = 0.02,
    time_stratify_window: float = 0.10,
    time_stratify_min_chunk: int = 100,
    *,
    feature_selection_method: str = "rf",
    data_leak: bool = False,
) -> None:
    """
    Train a model for driver drowsiness detection.

    Parameters
    ----------
    model_name : {"RF", "SvmA", "Lstm"}
        Model architecture to train.
    mode : {"pooled", "target_only", "source_only", "joint_train"}, optional
        Experimental mode (for suffixing and job bookkeeping).
    subject_split_strategy : {"random", "subject_time_split", "finetune_target_subjects"}, default="random"
        Strategy for how subjects are split into train/val/test.
    target_subjects : list of str, optional
        Target subject IDs (required for some strategies).
    subject_wise_split : bool, optional
        Kept for CLI compatibility; unused in this simplified pipeline.
    seed : int, default=42
        Random seed.
    tag : str, optional
        Optional suffix for saved artifacts.
    time_stratify_labels : bool, default=False
        Enable time-stratified splitting with class ratio tolerance.
    time_stratify_tolerance : float, default=0.02
        Allowed deviation in positive class ratio per split.
    time_stratify_window : float, default=0.10
        Window (fraction of N) for boundary search.
    time_stratify_min_chunk : int, default=100
        Minimum rows per split.

    Other Parameters
    ----------------
    feature_selection_method : {"rf", "mi", "anova"}, default="rf"
        Feature selection method.
    data_leak : bool, default=False
        If True, fit the scaler on train+val (kept for parity with older experiments).

    Returns
    -------
    None
    """
    # 1) Subject list and basic metadata
    subject_list = read_subject_list()
    # --- Resolve directory type using helper ---
    from src.utils.io.loaders import get_model_type
    model_type = get_model_type(model_name)

    suffix = f"_{mode}" if mode else ""
    if tag:
        suffix += f"_{tag}"

    logging.info(
        "[Start] model=%s | mode=%s | strategy=%s", 
        model_name, mode, subject_split_strategy
    )

    # 2) Split data according to the selected strategy
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        subject_split_strategy=subject_split_strategy,
        subject_list=subject_list,
        target_subjects=target_subjects or [],
        model_type=model_type,
        seed=seed,
        time_stratify_labels=time_stratify_labels,
        time_stratify_tolerance=time_stratify_tolerance,
        time_stratify_window=time_stratify_window,
        time_stratify_min_chunk=time_stratify_min_chunk,
    )
    log_split_ratios(
        y_train, y_val, y_test, 
        tag=f"{subject_split_strategy}|time_stratify={time_stratify_labels}"
    )

    # Sanity checks
    if y_train.nunique() < 2:
        logging.error(
            "Training labels are not binary. Stats: %s", 
            y_train.value_counts().to_dict()
        )
        return
    if min(len(X_train), len(X_val), len(X_test)) == 0:
        logging.error(
            "One of the splits is empty. Review subject/time filtering."
        )
        return

    # 3) Feature selection & scaling
    # Normalize feature names for consistency between training and evaluation
    X_train.columns = normalize_feature_names(X_train.columns)
    X_val.columns = normalize_feature_names(X_val.columns)
    if X_test is not None:
        X_test.columns = normalize_feature_names(X_test.columns)

    # Remove duplicated or non-numeric columns before feature selection
    X_train = X_train.loc[:, ~X_train.columns.duplicated()]
    X_val = X_val.loc[:, ~X_val.columns.duplicated()]
    if X_test is not None:
        X_test = X_test.loc[:, ~X_test.columns.duplicated()]

    # Drop unnecessary columns (e.g., subject_id) and keep only numeric columns
    X_train = X_train.drop(columns=["subject_id"], errors="ignore").select_dtypes(include=[np.number])
    X_val = X_val.drop(columns=["subject_id"], errors="ignore").select_dtypes(include=[np.number])
    if X_test is not None:
        X_test = X_test.drop(columns=["subject_id"], errors="ignore").select_dtypes(include=[np.number])

    # Align columns (train/val/test) to common subset to avoid misalignment
    common_cols = X_train.columns.intersection(X_val.columns)
    if X_test is not None:
        common_cols = common_cols.intersection(X_test.columns)
    X_train = X_train[common_cols]
    X_val = X_val[common_cols]
    if X_test is not None:
        X_test = X_test[common_cols]

    selected_features, scaler, X_train_fs, X_val_fs, X_test_fs = select_features_and_scale(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        feature_selection_method=feature_selection_method,
        top_k=TOP_K_FEATURES,
        data_leak=data_leak,
    )

    # --- Normalize selected feature names before saving ---
    selected_features = normalize_feature_names(selected_features)
    logging.info(f"[TRAIN] Normalized {len(selected_features)} feature names for consistency.")

    # 4) Train the model
    best_clf, scaler, best_threshold, feature_meta, results = train_model(
        model_name=model_name,
        model_type=model_type,
        X_train_fs=X_train_fs,
        X_val_fs=X_val_fs,
        X_test_fs=X_test_fs,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        selected_features=selected_features,
        scaler=scaler,
        suffix=suffix,
    )

    # 5) Save artifacts
    save_artifacts(
        model_name=model_name,
        model_type=model_type,
        suffix=suffix,
        best_clf=best_clf,
        scaler=scaler,
        selected_features=selected_features,
        feature_meta=feature_meta,
        results=results,
        best_threshold=best_threshold,
    )

    logging.info("[DONE] Training complete for %s%s", model_name, suffix)

