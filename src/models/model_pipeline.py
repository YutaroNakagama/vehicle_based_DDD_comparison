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

from src.config import TOP_K_FEATURES, KSS_BIN_LABELS, KSS_LABEL_MAP
from src.utils.io.loaders import read_subject_list, load_subject_csvs
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

# ------------------------------------------------------
# Helper used by time_stratified split
# ------------------------------------------------------
def _prepare_df_with_label_and_features(df: pd.DataFrame):
    """
    Build 'label' (0/1) and feature column list for time-stratified split.
    - Filters rows to KSS_BIN_LABELS
    - Maps to binary 'label' via KSS_LABEL_MAP
    - Returns (prepared_df, feature_columns)
    """
    d = df[df["KSS_Theta_Alpha_Beta"].isin(KSS_BIN_LABELS)].copy()
    d["label"] = d["KSS_Theta_Alpha_Beta"].replace(KSS_LABEL_MAP).astype(int)
    # vehicle-based features range
    start_col = "Steering_Range"
    end_col = "LaneOffset_AAA"
    features = d.loc[:, start_col:end_col].columns.tolist()
    if "subject_id" in d.columns:
        features.append("subject_id")
    return d, features

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


    # --- Append PBS job ID to suffix (for organized model output) ---
    jobid = os.environ.get("PBS_JOBID", "")
    if "." in jobid:
        jobid = jobid.split(".")[0]  # Remove hostname part like ".spcc-adm1"

    if jobid:
        # Prevent duplicate jobid in suffix if already included
        if jobid not in suffix:
            suffix = f"{suffix}_{jobid}"
        logging.info(f"[TRAIN] Appended jobid to suffix -> {suffix}")

    logging.info(
        "[Start] model=%s | mode=%s | strategy=%s", 
        model_name, mode, subject_split_strategy
    )

    # ------------------------------------------------------------------
    # Step 0: Load all subject CSVs from data/processed/common
    # ------------------------------------------------------------------

    # Initialize empty DataFrame for safety; actual data is loaded later
    import re
    data = pd.DataFrame()

    # ------------------------------------------------------------------
    # Step 1.5: Pre-filtering by target/source subjects (same logic as train_pipeline)
    # ------------------------------------------------------------------
    if mode in ["source_only", "target_only"]:
        # NOTE: We'll also filter `subject_list` so split_data() respects exclusions.
        default_rank_file = "results/domain_analysis/distance/rank_names.txt"
        # keep CLI arg if given; otherwise build from rank files
        target_subjects = target_subjects or []
        excluded_subjects: set[str] = set()

        # --- Load all subject CSVs before filtering (fix for empty data issue) ---
        data, _ = load_subject_csvs(
            subject_list=subject_list,     
            model_type=model_type,
            base_path="data/processed/common",  
            add_subject_id=True,   # ← ensure subject_id is available for row-level filtering
        )
        logging.info(f"[LOAD] Loaded {len(data)} rows from all subject CSVs before {mode} filtering.")

        # Extract base key from tag regardless of rank_names.txt existence
        tag_key = (tag or "").replace("rank_", "")

        # rank_names.txt is only needed for target_only; source_only does not use it.
        if mode == "target_only" and tag and os.path.exists(default_rank_file):
            with open(default_rank_file) as f:
                lines = [x.strip() for x in f.readlines() if x.strip()]

        # --- Ensure subject_id exists after loading actual data ---
        if "subject_id" not in data.columns:
            if "filename" in data.columns:
                fn_series = data["filename"]
            elif "FileName" in data.columns:
                fn_series = data["FileName"]
            else:
                fn_series = pd.Series(["unknown"] * len(data))
                logging.warning("[WARN] 'filename' column not found; subject_id set to 'unknown'")

            data["subject_id"] = fn_series.apply(
                lambda f: re.search(r"S\d{4}_\d", f).group(0)
                if isinstance(f, str) and re.search(r"S\d{4}_\d", f)
                else "unknown"
            )
            unique_ids = data["subject_id"].nunique()
            logging.info(f"[EVAL] Injected subject_id column (n={unique_ids} unique IDs).")

        # --- Branch cleanly by mode ---
        if mode == "target_only":
            # Use rank_names.txt mapping to find the target group file
            if 'lines' in locals():
                # match only exact metric-domain files (avoid substring overlaps)
                match = [x for x in lines if os.path.basename(x).startswith(tag_key)]
                if match:
                    group_file = os.path.normpath(match[0])
                    if os.path.exists(group_file):
                        with open(group_file) as g:
                            file_targets = [s.strip() for s in g.readlines() if s.strip()]
                            # If CLI also gave targets, take intersection to be safe; fallback to file if CLI empty
                            if target_subjects:
                                target_subjects = [s for s in target_subjects if s in file_targets]
                            else:
                                target_subjects = file_targets
            # Guard: if we couldn't resolve targets, do not drop everything
            if target_subjects:
                data = data[data["subject_id"].isin(target_subjects)].reset_index(drop=True)
            else:
                logging.warning("[target_only] target_subjects is empty (tag=%s). Skipping filtering.", tag_key)
            # === NEW: also restrict subject_list so split_data() truly uses only targets ===
            if target_subjects:
                before = len(subject_list)
                subject_list = list(dict.fromkeys(target_subjects))  # keep order, dedup
                logging.info("[target_only] Subject list overridden for split_data(): before=%d -> after=%d",
                             before, len(subject_list))
            else:
                logging.warning("[target_only] subject_list was NOT overridden because target_subjects is empty.")

        elif mode == "source_only":
            # ==========================================================
            # Exclude low/middle/high of the same metric (mmd/dtw/wasserstein)
            # ==========================================================
            from pathlib import Path
            ranks_dir = Path("results/domain_analysis/distance/ranks10")

            # Detect metric prefix (e.g., "dtw", "mmd", "wasserstein") from tag_key
            metric_prefix = None
            for prefix in ["dtw", "mmd", "wasserstein"]:
                if prefix in (tag_key or ""):
                    metric_prefix = prefix
                    break

            if metric_prefix:
                for domain in ["low", "middle", "high"]:
                    file_path = ranks_dir / f"{metric_prefix}_mean_{domain}.txt"
                    if file_path.exists():
                        with open(file_path) as f:
                            excluded_subjects.update(s.strip() for s in f if s.strip())
                logging.info(f"[source_only] Excluding {len(excluded_subjects)} subjects from {metric_prefix} domains")
            else:
                logging.warning("[source_only] Could not detect metric prefix from tag=%s. No subjects excluded.", tag_key)

            # Normalize & filter
            excl_norm = {s.strip().upper() for s in excluded_subjects}
            subj_norm = data["subject_id"].astype(str).str.strip().str.upper()
            mask_keep = ~subj_norm.isin(excl_norm)

            kept_subjects = data.loc[mask_keep, "subject_id"].unique().tolist()
            removed_subjects = sorted(excluded_subjects)

            logging.info(f"[source_only] Removed subjects: {removed_subjects}")
            logging.info(f"[source_only] Kept subjects ({len(kept_subjects)}): {sorted(kept_subjects)}")

            data = data[mask_keep].reset_index(drop=True)
            logging.info(f"[EVAL] source_only complete -> remaining samples: {len(data)}")

        else:
            # Fallback (should not normally reach here)
            data = data[~data["subject_id"].isin(target_subjects)].reset_index(drop=True)

        # === Also filter subject_list so split_data() reflects exclusions ===
        def _norm_sid(s: str) -> str:
            return str(s).strip().upper()
        if mode == "source_only" and excluded_subjects:
            excl_norm_set = {_norm_sid(s) for s in excluded_subjects}
            orig_cnt = len(subject_list)
            subject_list = [s for s in subject_list if _norm_sid(s) not in excl_norm_set]
            kept_cnt = len(subject_list)
            logging.info("[source_only] Subject list filtered for split_data(): original=%d, removed=%d, kept=%d",
                         orig_cnt, orig_cnt - kept_cnt, kept_cnt)
            try:
                removed_list = sorted(excluded_subjects)
                logging.info("[source_only] Removed subjects (sample): %s",
                             ", ".join(removed_list[:15]) + (" ..." if len(removed_list) > 15 else ""))
            except Exception:
                pass
        elif mode == "source_only" and not excluded_subjects:
            logging.warning("[source_only] excluded_subjects is empty (tag=%s). Check ranks10 files exist.", tag_key)

        # summary log for consistency
        logging.info(f"[EVAL] Data restricted to {len(data)} samples after {mode} filtering.")

    # For pooled or other modes (no filtering needed), load if not already done
    if mode not in ["source_only", "target_only"]:
        data, _ = load_subject_csvs(
            subject_list=subject_list,
            model_type=model_type,
            base_path="data/processed/common",
            add_subject_id=True,   # ← ensure subject_id is available for row-level filtering
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
    # Sanity: no excluded subject should appear in any split (source_only)
    if mode == "source_only" and excluded_subjects:
        def _get_sids(df):
            return set(df["subject_id"].astype(str).str.upper()) if "subject_id" in df.columns else set()
        leak = (_get_sids(X_train) | _get_sids(X_val) | _get_sids(X_test)) & {s.upper() for s in excluded_subjects}
        if leak:
            logging.error("[source_only] Found excluded subjects in splits: %s", sorted(list(leak))[:10])

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

    # --- Explicitly drop EEG-related columns (we use vehicle signals only) ---
    eeg_keywords = ["Channel_", "EEG", "Theta", "Alpha", "Beta", "Gamma", "Delta"]
    def drop_eeg(df):
        drop_cols = [c for c in df.columns if any(k in c for k in eeg_keywords)]
        if drop_cols:
            logging.info(f"[TRAIN] Dropping {len(drop_cols)} EEG-related columns (e.g., {drop_cols[:5]})")
            df = df.drop(columns=drop_cols)
        return df

    X_train = drop_eeg(X_train)
    X_val = drop_eeg(X_val)
    if X_test is not None:
        X_test = drop_eeg(X_test)

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

