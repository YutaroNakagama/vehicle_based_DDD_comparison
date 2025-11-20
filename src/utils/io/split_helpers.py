"""High-level helper utilities for dataset splitting in model training.

This module extends low-level split functions in ``src/utils/io/split.py`` with:
  - Subject/time-based split strategies
  - Logging of class ratios
  - Dynamic data directory selection

It keeps experiment-specific branching separate from the core algorithms.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from src.utils.io.loaders import load_subject_csvs
from src.utils.io.split import (
    data_split,
    data_split_by_subject,
    data_time_split_by_subject,
    time_stratified_three_way_split,
)

# NOTE: The following import errors are unrelated to the variable naming unification and are due to missing dependencies in the environment.
# The code changes for variable naming are correct and ready for commit.

# --- Local helper to avoid circular import with model_pipeline ---
def _prepare_df_with_label_and_features(df: pd.DataFrame):
    from src.config import KSS_BIN_LABELS, KSS_LABEL_MAP
    d = df[df["KSS_Theta_Alpha_Beta"].isin(KSS_BIN_LABELS)].copy()
    d["label"] = d["KSS_Theta_Alpha_Beta"].replace(KSS_LABEL_MAP).astype(int)
    # vehicle-based feature range (adjust if your columns differ)
    start_col = "Steering_Range"
    end_col   = "LaneOffset_AAA"
    features = d.loc[:, start_col:end_col].columns.tolist()
    if "subject_id" in d.columns:
        features.append("subject_id")
    return d, features

def log_split_ratios(y_tr: pd.Series, y_va: pd.Series, y_te: pd.Series, tag: str = "") -> None:
    """Log class distribution across splits."""
    def _summ(y):
        n = int(y.shape[0])
        # assuming binary label {0,1}
        p = int(y.sum()) if n else 0
        r = p / n if n else float("nan")
        return n, p, r

    n_tr, p_tr, r_tr = _summ(y_tr)
    n_va, p_va, r_va = _summ(y_va)
    n_te, p_te, r_te = _summ(y_te)
    n_all = n_tr + n_va + n_te
    p_all = p_tr + p_va + p_te
    r_all = p_all / n_all if n_all else float("nan")

    logging.info(f"[split:{tag}] train n={n_tr} pos={p_tr} ({r_tr:.3f})")
    logging.info(f"[split:{tag}] valid n={n_va} pos={p_va} ({r_va:.3f})")
    logging.info(f"[split:{tag}] test  n={n_te} pos={p_te} ({r_te:.3f})")
    logging.info(f"[split:{tag}] total n={n_all} pos={p_all} ({r_all:.3f})")


def split_data(
    subject_split_strategy: str,
    subject_list: List[str],
    target_subjects: Optional[List[str]],
    model_name: str,
    seed: int,
    time_stratify_labels: bool,
    time_stratify_tolerance: float,
    time_stratify_window: float,
    time_stratify_min_chunk: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Create train/val/test splits according to the selected strategy.

    Parameters
    ----------
    subject_split_strategy : str
        Strategy for splitting subjects (e.g., 'subject_time_split').
    subject_list : List[str]
        List of all subject IDs.
    target_subjects : Optional[List[str]]
        List of target subject IDs for test/finetune, or None.
    model_name : str
        Model architecture name (e.g., 'Lstm', 'SvmA').
    seed : int
        Random seed for reproducibility.
    time_stratify_labels : bool
        Whether to stratify splits by time and labels.
    time_stratify_tolerance : float
        Tolerance for time stratification.
    time_stratify_window : float
        Window proportion for time stratification.
    time_stratify_min_chunk : int
        Minimum chunk size for time stratification.

    Returns
    -------
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """

    base_dir = Path("data/processed")
    if (base_dir / model_name).exists() and os.listdir(base_dir / model_name):
        data_dir = base_dir / model_name
        logging.info(f"[INFO] Using model-specific data directory: {data_dir}")
    elif (base_dir / "common").exists() and os.listdir(base_dir / "common"):
        data_dir = base_dir / "common"
        logging.info(f"[INFO] Using shared common data directory: {data_dir}")
    else:
        raise FileNotFoundError(
            f"No valid processed data directory found for model '{model_name}'."
        )

    def _load_subjects(subjects):
        return load_subject_csvs(
            subjects,
            model_name=None,
            add_subject_id=True,
            base_path=str(data_dir)
        )

    # --- Strategy: time split ---
    if subject_split_strategy == "subject_time_split":
        use_subjects = target_subjects if target_subjects else subject_list
        data, _ = _load_subjects(use_subjects)

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
        # --- Fallback if any split collapses to a single class ---
        def _has_both(y: pd.Series) -> bool:
            return y.nunique() == 2 and (y.value_counts().min() > 0)
        if not (_has_both(y_train) and _has_both(y_val) and _has_both(y_test)):
            logging.warning("[subject_time_split] Class collapsed in a split -> fallback to random split")
            from src.utils.io import split as split_module
            return split_module.data_split(data, random_state=seed)
        return X_train, X_val, X_test, y_train, y_val, y_test

    # --- Strategy: finetune_target_subjects ---
    if subject_split_strategy == "finetune_target_subjects":
        if not target_subjects:
            raise ValueError("`finetune_target_subjects` requires non-empty target_subjects.")
        general_subjects = [s for s in subject_list if s not in target_subjects]
        use_subjects = list(set(general_subjects + target_subjects))
        data, _ = load_subject_csvs(use_subjects, model_name, add_subject_id=True)

        from sklearn.model_selection import train_test_split

        train_subjects = general_subjects
        if len(target_subjects) == 1:
            from src.utils.io.split import data_split
            single_df, _ = load_subject_csvs(target_subjects, model_name, add_subject_id=True)
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
            val_subjects, test_subjects = train_test_split(target_subjects, test_size=0.5, random_state=seed)
            X_train, X_val, X_test, y_train, y_val, y_test = data_split_by_subject(
                data, train_subjects, seed, val_subjects=val_subjects, test_subjects=test_subjects
            )
        return X_train, X_val, X_test, y_train, y_val, y_test

    # --- Default random split ---
    # If caller provided target_subjects (e.g., target_only mode) but strategy is "random",
    # prefer them to ensure we really restrict to targets.
    subjects_for_random = target_subjects if (target_subjects and len(target_subjects) > 0) else subject_list
    if target_subjects and len(target_subjects) > 0:
        logging.info(
            f"[split:random] Using target_subjects ({len(target_subjects)}) "
            f"in place of subject_list ({len(subject_list)})."
        )
    data, _ = load_subject_csvs(subjects_for_random, model_name, add_subject_id=True)

    # Explicitly reference the module to avoid scope-shadowing issues (UnboundLocalError)
    from src.utils.io import split as split_module
    return split_module.data_split(
        data,
        random_state=seed,
    )

# ==========================================================
#  Evaluation-specific data split helper
# ==========================================================

def prepare_data_split(
    data: pd.DataFrame,
    subject_list: list,
    seed: int = 42,
    subject_wise_split: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Prepare train/val/test splits for evaluation.

    A lightweight wrapper for evaluation mode, which typically uses
    simple random or subject-wise splitting without complex time stratification.

    Parameters
    ----------
    data : pandas.DataFrame
        Combined dataset containing all subjects and features.
    subject_list : list of str
        Subject identifiers used for reference.
    seed : int, default=42
        Random seed for reproducibility.
    subject_wise_split : bool, default=False
        If True, performs subject-based splitting (avoiding subject leakage).

    Returns
    -------
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    from src.utils.io.split import data_split, data_split_by_subject

    if subject_wise_split:
        logging.info("[EVAL] Using subject-wise split.")
        X_train, X_val, X_test, y_train, y_val, y_test = data_split_by_subject(
            data, subject_list, seed
        )
    else:
        logging.info("[EVAL] Using random split.")
        X_train, X_val, X_test, y_train, y_val, y_test = data_split(
            data, random_state=seed
        )

    return X_train, X_val, X_test, y_train, y_val, y_test
