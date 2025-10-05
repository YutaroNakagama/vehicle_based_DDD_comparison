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
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.preprocessing import StandardScaler

from src.config import TOP_K_FEATURES
from src.utils.io.loaders import read_subject_list, load_subject_csvs
from src.utils.io.split import (
    data_split,
    data_split_by_subject,
    data_time_split_by_subject,
    time_stratified_three_way_split,
    _check_nonfinite,
)
from src.models.feature_selection.rf_importance import select_top_features_by_importance
from src.models.architectures.helpers import get_classifier
from src.models.architectures.SvmA import SvmA_train
from src.models.architectures.lstm import lstm_train
from src.models.architectures.common import common_train


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# =========================
# Helpers: data prep/split
# =========================
def _prepare_df_with_label_and_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Filter KSS labels and prepare feature columns.

    The function:
    - Keeps rows with valid KSS labels
    - Adds a binary `label` column using the configured mapping
    - Returns the feature column list from Steering_Range..LaneOffset_AAA plus subject_id if available

    Parameters
    ----------
    df : pandas.DataFrame
        Raw dataframe with KSS_Theta_Alpha_Beta column and features.

    Returns
    -------
    (df_filtered, feature_columns) : (pandas.DataFrame, list of str)
    """
    from src.config import KSS_BIN_LABELS, KSS_LABEL_MAP
    df = df[df["KSS_Theta_Alpha_Beta"].isin(KSS_BIN_LABELS)].copy()
    df["label"] = df["KSS_Theta_Alpha_Beta"].replace(KSS_LABEL_MAP)
    start_col = "Steering_Range"
    end_col = "LaneOffset_AAA"
    feature_columns = df.loc[:, start_col:end_col].columns.tolist()
    if "subject_id" in df.columns:
        feature_columns.append("subject_id")
    return df, feature_columns


def _log_split_ratios(y_tr: pd.Series, y_va: pd.Series, y_te: pd.Series, tag: str = "") -> None:
    """Log class distribution across splits."""
    def _summ(y):
        n = int(y.shape[0])
        p = int(y.sum()) if n else 0
        r = p / n if n else float("nan")
        return n, p, r

    n_tr, p_tr, r_tr = _summ(y_tr)
    n_va, p_va, r_va = _summ(y_va)
    n_te, p_te, r_te = _summ(y_te)
    n_all = n_tr + n_va + n_te
    p_all = p_tr + p_va + p_te
    r_all = p_all / n_all if n_all else float("nan")

    logging.info("[split:%s] train n=%d pos=%d (%.3f)", tag, n_tr, p_tr, r_tr)
    logging.info("[split:%s] valid n=%d pos=%d (%.3f)", tag, n_va, p_va, r_va)
    logging.info("[split:%s] test  n=%d pos=%d (%.3f)", tag, n_te, p_te, r_te)
    logging.info("[split:%s] total n=%d pos=%d (%.3f)", tag, n_all, p_all, r_all)


def _split_data(
    subject_split_strategy: str,
    subject_list: List[str],
    target_subjects: Optional[List[str]],
    model_type: str,
    seed: int,
    time_stratify_labels: bool,
    time_stratify_tolerance: float,
    time_stratify_window: float,
    time_stratify_min_chunk: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Create train/val/test splits according to the selected strategy.

    Supported strategies
    --------------------
    - "random": pooled random split across all available subjects
    - "subject_time_split": time-aware split within target subjects (or all if not provided)
    - "finetune_target_subjects": pretrain on general (non-target) + val/test on target

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    if subject_split_strategy == "subject_time_split":
        # Target-only (or all subjects if target list is None/empty)
        use_subjects = target_subjects if target_subjects else subject_list
        data, _ = load_subject_csvs(use_subjects, model_type, add_subject_id=True)

        if time_stratify_labels:
            df_lab, feature_columns = _prepare_df_with_label_and_features(data)
            sort_keys = ("subject_id", "Timestamp")
            idx_tr, idx_va, idx_te = time_stratified_three_way_split(
                df_lab,
                label_col="label",
                sort_keys=sort_keys,
                train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
                tolerance=time_stratify_tolerance,
                window_prop=time_stratify_window,
                min_chunk=time_stratify_min_chunk,
            )
            X_train = df_lab.loc[idx_tr, feature_columns].drop(columns=["subject_id"], errors="ignore")
            X_val   = df_lab.loc[idx_va, feature_columns].drop(columns=["subject_id"], errors="ignore")
            X_test  = df_lab.loc[idx_te, feature_columns].drop(columns=["subject_id"], errors="ignore")
            y_train = df_lab.loc[idx_tr, "label"]
            y_val   = df_lab.loc[idx_va, "label"]
            y_test  = df_lab.loc[idx_te, "label"]
        else:
            X_train, X_val, X_test, y_train, y_val, y_test = data_time_split_by_subject(
                data, subject_col="subject_id", time_col="Timestamp"
            )
        return X_train, X_val, X_test, y_train, y_val, y_test

    if subject_split_strategy == "finetune_target_subjects":
        if not target_subjects:
            raise ValueError("`finetune_target_subjects` requires non-empty target_subjects.")
        general_subjects = [s for s in subject_list if s not in target_subjects]
        use_subjects = list(set(general_subjects + target_subjects))
        data, _ = load_subject_csvs(use_subjects, model_type, add_subject_id=True)

        # Pretrain on general subjects, evaluate on target subjects
        train_subjects = general_subjects
        if len(target_subjects) == 1:
            # single target subject: create within-subject val/test
            single_df, _ = load_subject_csvs(target_subjects, model_type, add_subject_id=True)
            X_single_tr, X_single_va, X_single_te, y_single_tr, y_single_va, y_single_te = data_split(
                single_df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=seed
            )
            X_general_tr, _, _, y_general_tr, _, _ = data_split_by_subject(
                data, train_subjects, seed, val_subjects=[], test_subjects=[]
            )
            X_train = pd.concat([X_general_tr, X_single_tr], ignore_index=True)
            y_train = pd.concat([y_general_tr, y_single_tr], ignore_index=True)
            X_val, y_val = X_single_va, y_single_va
            X_test, y_test = X_single_te, y_single_te
        else:
            # multiple target subjects: split between val/test by subject
            val_subjects, test_subjects = train_test_split(target_subjects, test_size=0.5, random_state=seed)
            X_train, X_val, X_test, y_train, y_val, y_test = data_split_by_subject(
                data, train_subjects, seed, val_subjects=val_subjects, test_subjects=test_subjects
            )
        return X_train, X_val, X_test, y_train, y_val, y_test

    # Default pooled random split
    data, _ = load_subject_csvs(subject_list, model_type, add_subject_id=True)
    return data_split(data, random_state=seed)


# =========================
# Helpers: feature selection
# =========================
def _select_features_and_scale(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    feature_selection_method: str = "rf",
    top_k: int = TOP_K_FEATURES,
    data_leak: bool = False,
) -> Tuple[List[str], StandardScaler, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select features and fit a scaler.

    Notes
    -----
    - Non-finite values are checked and fixed using `_check_nonfinite`.
    - `data_leak=True` will fit the scaler on train+val (kept for parity with older experiments).

    Returns
    -------
    selected_features, scaler, X_train_fs, X_val_fs, X_test_fs
    """
    # Drop subject_id and ensure finites
    X_train = _check_nonfinite(X_train.drop(columns=["subject_id"], errors="ignore"), "X_train")
    X_val   = _check_nonfinite(X_val.drop(columns=["subject_id"], errors="ignore"), "X_val")
    X_test  = _check_nonfinite(X_test.drop(columns=["subject_id"], errors="ignore"), "X_test")

    # Feature selection
    if feature_selection_method == "mi":
        selector = SelectKBest(score_func=mutual_info_classif, k=top_k)
        selector.fit(X_train, y_train)
        mask = selector.get_support()
        selected_features = X_train.columns[mask].tolist()

    elif feature_selection_method == "anova":
        selector = SelectKBest(score_func=f_classif, k=top_k)
        selector.fit(X_train, y_train)
        mask = selector.get_support()
        selected_features = X_train.columns[mask].tolist()

    elif feature_selection_method == "rf":
        selected_features = select_top_features_by_importance(X_train, y_train, top_k=top_k)

        # Double-check non-finite after selection
        X_train = _check_nonfinite(X_train[selected_features], "X_train(post-rf)")
        X_val   = _check_nonfinite(X_val[selected_features], "X_val(post-rf)")
        X_test  = _check_nonfinite(X_test[selected_features], "X_test(post-rf)")

    else:
        raise ValueError(f"Unknown feature_selection_method: {feature_selection_method}")

    # Scaling
    scaler = StandardScaler()
    if data_leak:
        scaler.fit(pd.concat([X_train[selected_features], X_val[selected_features]], axis=0))
    else:
        scaler.fit(X_train[selected_features])

    logging.info("Selected %d features using '%s'.", len(selected_features), feature_selection_method)
    return selected_features, scaler, X_train, X_val, X_test


# =========================
# Helpers: model training & saving
# =========================
def _train_model(
    model_name: str,
    model_type: str,
    X_train_fs: pd.DataFrame,
    X_val_fs: pd.DataFrame,
    X_test_fs: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    selected_features: List[str],
    scaler: Optional[StandardScaler],
    suffix: str,
) -> Tuple[Optional[object], Optional[StandardScaler], Optional[float], Dict, Dict]:
    """Dispatch to the appropriate training routine and return artifacts.

    Returns
    -------
    best_clf, scaler, best_threshold, feature_meta, results
    """
    if model_name == "Lstm":
        # For LSTM we keep the existing behavior (train inside lstm_train)
        lstm_train(X_train_fs, y_train, model_name)
        logging.info("LSTM training completed.")
        return None, None, None, {}, {}

    if model_name == "SvmA":
        # SvmA has its own feature index logic inside SvmA_train
        SvmA_train(X_train_fs, X_val_fs, y_train, y_val, selected_features, model_name)
        logging.info("SvmA training completed.")
        return None, None, None, {}, {}

    # Tree-based / linear models route through common_train
    clf = get_classifier(model_name)
    best_clf, scaler, best_threshold, feature_meta, results = common_train(
        X_train_fs, X_val_fs, X_test_fs,
        y_train, y_val, y_test,
        selected_features,
        model_name, model_type, clf,
        scaler=scaler,
        suffix=suffix,
        data_leak=False,
    )
    return best_clf, scaler, best_threshold, feature_meta, results


def _save_artifacts(
    model_name: str,
    model_type: str,
    suffix: str,
    best_clf: Optional[object],
    scaler: Optional[StandardScaler],
    selected_features: Optional[List[str]],
    feature_meta: Optional[Dict],
    results: Optional[Dict],
    best_threshold: Optional[float],
) -> None:
    """Persist models, scalers, features, and training-time metrics to disk."""
    os.makedirs(f"models/{model_type}", exist_ok=True)
    os.makedirs(f"results/train/{model_name}", exist_ok=True)

    # Save trained model, scaler, features
    if best_clf is not None:
        with open(f"models/{model_type}/{model_name}{suffix}.pkl", "wb") as f:
            pickle.dump(best_clf, f)
    if scaler is not None:
        with open(f"models/{model_type}/scaler_{model_name}{suffix}.pkl", "wb") as f:
            pickle.dump(scaler, f)
    if selected_features is not None:
        with open(f"models/{model_type}/selected_features_{model_name}{suffix}.pkl", "wb") as f:
            pickle.dump(selected_features, f)
    if feature_meta:
        with open(f"models/{model_type}/feature_meta_{model_name}{suffix}.json", "w") as f:
            json.dump(feature_meta, f, indent=2)

    # Save threshold & metrics
    if best_threshold is not None:
        thr_meta = {"model": model_name, "threshold": float(best_threshold), "metric": "F1-optimal"}
        with open(f"results/train/{model_name}/threshold_{model_name}{suffix}.json", "w") as f:
            json.dump(thr_meta, f, indent=2)

    if results:
        rows = [{"phase": "training", "split": split, **metrics} for split, metrics in results.items()]
        df_results = pd.DataFrame(rows)
        df_results.to_csv(f"results/train/{model_name}/trainmetrics_{model_name}{suffix}.csv", index=False)
        with open(f"results/train/{model_name}/trainmetrics_{model_name}{suffix}.json", "w") as f:
            json.dump(rows, f, indent=2)

    logging.info("Artifacts saved under models/%s and results/train/%s", model_type, model_name)


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
    model_type = model_name  # preserved for legacy directory structure
    suffix = f"_{mode}" if mode else ""
    if tag:
        suffix += f"_{tag}"

    logging.info("[Start] model=%s | mode=%s | strategy=%s", model_name, mode, subject_split_strategy)

    # 2) Split data according to the selected strategy
    X_train, X_val, X_test, y_train, y_val, y_test = _split_data(
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
    _log_split_ratios(y_train, y_val, y_test, tag=f"{subject_split_strategy}|time_stratify={time_stratify_labels}")

    # Sanity checks
    if y_train.nunique() < 2:
        logging.error("Training labels are not binary. Stats: %s", y_train.value_counts().to_dict())
        return
    if min(len(X_train), len(X_val), len(X_test)) == 0:
        logging.error("One of the splits is empty. Review subject/time filtering.")
        return

    # 3) Feature selection & scaling
    selected_features, scaler, X_train_fs, X_val_fs, X_test_fs = _select_features_and_scale(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        feature_selection_method=feature_selection_method,
        top_k=TOP_K_FEATURES,
        data_leak=data_leak,
    )

    # 4) Train the model
    best_clf, scaler, best_threshold, feature_meta, results = _train_model(
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
    _save_artifacts(
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

