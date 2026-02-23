"""Training pipeline stages for model training.

This module provides focused helper functions for each stage of the
training pipeline, keeping model_pipeline.py clean and orchestration-focused.
"""

import logging
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from src import config as cfg
from src.utils.io.loaders import read_subject_list, load_subject_csvs
from src.utils.io.target_resolution import (
    resolve_target_subjects_from_tag,
    resolve_source_group_subjects,
)


def prepare_suffix_with_jobid(mode: Optional[str], tag: Optional[str]) -> str:
    """Build artifact suffix including PBS job ID.

    Parameters
    ----------
    mode : str, optional
        Experiment mode (e.g., "target_only", "source_only").
    tag : str, optional
        Additional experiment tag.

    Returns
    -------
    str
        Suffix string including mode, tag, and job ID (e.g., "_target_only_rank_dtw_mean_out_domain_14209090[1]").
    """
    suffix = f"_{mode}" if mode else ""
    if tag:
        suffix += f"_{tag}"
    
    # Append PBS job ID to suffix
    jobid = os.environ.get("PBS_JOBID", "")
    if "." in jobid:
        jobid = jobid.split(".")[0]  # Remove hostname part
    
    if jobid and jobid not in suffix:
        # Check if jobid already has [n] format (from PBS_ARRAY_INDEX)
        array_idx = os.environ.get("PBS_ARRAY_INDEX", "")
        if array_idx:
            suffix = f"{suffix}_{jobid}[{array_idx}]"
        else:
            # For non-array jobs (like pooled mode), default to [1]
            suffix = f"{suffix}_{jobid}[1]"
        logging.info(f"[TRAIN] Appended jobid to suffix -> {suffix}")
    
    return suffix


def load_and_filter_data(
    model_name: str,
    mode: Optional[str],
    tag: Optional[str],
    target_subjects: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Load subject CSVs and apply mode-based filtering.

    Parameters
    ----------
    model_name : str
        Model name for determining data directory.
    mode : str, optional
        Experiment mode ("target_only", "source_only", "joint_train", "pooled").
    tag : str, optional
        Experiment tag for resolving target groups.
    target_subjects : list of str, optional
        CLI-provided target subjects.

    Returns
    -------
    tuple of (pd.DataFrame, list of str, list of str)
        - data : Filtered dataset
        - subject_list : Updated subject list
        - target_subjects_resolved : Resolved target subjects
    """
    subject_list = read_subject_list()
    target_subjects_resolved = []
    
    # Load all subject CSVs
    # Lstm uses model-specific processed data (Wang et al. 2022 features)
    if model_name == "Lstm":
        lstm_base = os.path.join(cfg.PROCESS_CSV_PATH, "Lstm")
        data, _ = load_subject_csvs(
            subject_list=subject_list,
            model_name=model_name,
            base_path=lstm_base,
            add_subject_id=True,
        )
    else:
        data, _ = load_subject_csvs(
            subject_list=subject_list,
            model_name=model_name,
            base_path=cfg.PROCESS_CSV_COMMON_PATH,
            add_subject_id=True,
        )
    logging.info(f"[LOAD] Loaded {len(data)} rows from all subject CSVs before filtering.")
    
    # Inject subject_id if missing
    if "subject_id" not in data.columns:
        if "filename" in data.columns:
            fn_series = data["filename"]
        elif "FileName" in data.columns:
            fn_series = data["FileName"]
        else:
            fn_series = pd.Series(["unknown"] * len(data))
            logging.warning("[LOAD] 'filename' column not found; subject_id set to 'unknown'")
        
        data["subject_id"] = fn_series.apply(
            lambda f: re.search(r"S\d{4}_\d", f).group(0)
            if isinstance(f, str) and re.search(r"S\d{4}_\d", f)
            else "unknown"
        )
        unique_ids = data["subject_id"].nunique()
        logging.info(f"[LOAD] Injected subject_id column (n={unique_ids} unique IDs).")
    
    # Apply mode-based filtering
    if mode == "target_only":
        # Resolve target subjects from tag or CLI
        target_subjects_resolved = resolve_target_subjects_from_tag(
            tag=tag,
            mode=mode,
            cli_target_subjects=target_subjects,
        )
        
        if target_subjects_resolved:
            data = data[data["subject_id"].isin(target_subjects_resolved)].reset_index(drop=True)
            subject_list = list(dict.fromkeys(target_subjects_resolved))
            logging.info(
                f"[LOAD] target_only: Restricted to {len(data)} samples from {len(subject_list)} subjects."
            )
        else:
            logging.warning("[LOAD] target_only: No target_subjects resolved. No filtering applied.")
    
    elif mode == "source_only":
        # Resolve target subjects for exclusion
        target_subjects_resolved = resolve_target_subjects_from_tag(
            tag=tag,
            mode=mode,
            cli_target_subjects=target_subjects,
        )
        
        if target_subjects_resolved:
            data = data[~data["subject_id"].isin(target_subjects_resolved)].reset_index(drop=True)
            subject_list = [s for s in subject_list if s not in target_subjects_resolved]
            logging.info(
                f"[LOAD] source_only: Excluded targets, kept {len(data)} samples from {len(subject_list)} subjects."
            )
        else:
            logging.warning("[LOAD] source_only: No target_subjects to exclude. Using all data.")
    
    elif mode == "mixed":
        # Multi-domain: resolve target subjects for evaluation, but keep ALL data for training
        target_subjects_resolved = resolve_target_subjects_from_tag(
            tag=tag,
            mode=mode,
            cli_target_subjects=target_subjects,
        )
        # No data filtering — prepare_mixed_splits handles the split logic
        logging.info(
            f"[LOAD] mixed: All {len(data)} samples retained for training. "
            f"Evaluation target: {len(target_subjects_resolved)} subjects."
        )
    
    elif mode == "domain_train":
        # Unified domain training: restrict to domain subjects (same as target_only)
        target_subjects_resolved = resolve_target_subjects_from_tag(
            tag=tag,
            mode=mode,
            cli_target_subjects=target_subjects,
        )
        
        if target_subjects_resolved:
            data = data[data["subject_id"].isin(target_subjects_resolved)].reset_index(drop=True)
            subject_list = list(dict.fromkeys(target_subjects_resolved))
            logging.info(
                f"[LOAD] domain_train: Restricted to {len(data)} samples from {len(subject_list)} subjects."
            )
        else:
            logging.warning("[LOAD] domain_train: No target_subjects resolved. No filtering applied.")
    
    elif mode == "joint_train":
        # Combine source and target subjects
        if target_subjects:
            target_subjects_resolved = target_subjects
            subject_list = list(dict.fromkeys(list(subject_list) + list(target_subjects)))
            logging.info(f"[LOAD] joint_train: Combined subject list size = {len(subject_list)}")
        else:
            logging.info("[LOAD] joint_train: No additional target_subjects to combine.")
    
    else:
        logging.info(f"[LOAD] Mode '{mode}': No filtering applied.")
    
    return data, subject_list, target_subjects_resolved


def prepare_source_only_splits(
    model_name: str,
    tag: str,
    seed: int,
    target_subjects: List[str],
    time_stratify_labels: bool,
    time_stratify_tolerance: float,
    time_stratify_window: float,
    time_stratify_min_chunk: int,
    keep_subject_id: bool = False,
) -> Tuple:
    """Prepare train/val/test splits for source_only mode.

    In source_only (cross-domain) mode:
    - Training: Use opposite domain from target (in_domain ↔ out_domain)
    - Evaluation: Use target group (out_domain/in_domain/mid_domain) specified by tag

    Examples:
    - Target=out_domain → train on in_domain, evaluate on out_domain
    - Target=in_domain → train on out_domain, evaluate on in_domain

    Parameters
    ----------
    model_name : str
        Model name.
    tag : str
        Experiment tag (e.g., "rank_dtw_mean_high").
    seed : int
        Random seed.
    time_stratify_labels : bool
        Whether to use time-stratified splitting.
    time_stratify_tolerance : float
        Class ratio tolerance for time stratification.
    time_stratify_window : float
        Search window for time stratification.
    time_stratify_min_chunk : int
        Minimum chunk size for time stratification.

    Returns
    -------
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    from src.utils.io.split_helpers import split_data
    
    # Resolve evaluation subjects (target domain: out_domain/in_domain/mid_domain based on tag)
    eval_subjects = resolve_target_subjects_from_tag(
        tag=tag,
        mode="source_only",
        cli_target_subjects=target_subjects,
    )
    
    if not eval_subjects:
        raise ValueError(
            "[SOURCE_ONLY] No evaluation subjects resolved. "
            "Cannot proceed without target group."
        )
    
    # Infer target domain from tag
    target_domain = None
    for domain in ["out_domain", "in_domain", "mid_domain"]:
        if domain in tag:
            target_domain = domain
            break
    
    if not target_domain:
        raise ValueError(f"[SOURCE_ONLY] Cannot infer target domain from tag: {tag}")
    
    # Resolve source group subjects for training (opposite domain from target)
    source_subjects = resolve_source_group_subjects(tag, target_domain=target_domain)
    
    # Check if training and evaluation groups are identical
    is_same_group_case = set(source_subjects) == set(eval_subjects)
    
    if is_same_group_case:
        # When source and target are the same group: use single split to ensure consistency with target_only
        # This should not happen with the new cross-domain logic, but kept for safety
        logging.warning(
            f"[SOURCE_ONLY] Same-group case detected (unexpected): training and evaluation use same {len(source_subjects)} subjects")
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            subject_split_strategy="subject_time_split",
            subject_list=source_subjects,
            target_subjects=source_subjects,
            model_name=model_name,
            seed=seed,
            time_stratify_labels=time_stratify_labels,
            time_stratify_tolerance=time_stratify_tolerance,
            time_stratify_window=time_stratify_window,
            time_stratify_min_chunk=time_stratify_min_chunk,
            keep_subject_id=keep_subject_id,
        )
        logging.info(
            f"[SOURCE_ONLY] Same-group case: train={len(y_train)}, val={len(y_val)}, test={len(y_test)} samples"
        )
    else:
        # Cross-domain case: train+val on source domain, test on target domain
        # This follows standard domain generalization protocol:
        #   - Train & Val from source (hyperparameter tuning uses source only)
        #   - Test from target (pure unseen-domain evaluation)
        source_domain = "out_domain" if target_domain == "in_domain" else "in_domain"
        logging.info(
            f"[SOURCE_ONLY] Cross-domain: target={target_domain} ({len(eval_subjects)} subjects), "
            f"source={source_domain} ({len(source_subjects)} subjects)"
        )
        
        # Split source group → train (80%) + val (10%) for training & tuning
        src_train, src_val, _, y_src_train, y_src_val, _ = split_data(
            subject_split_strategy="subject_time_split",
            subject_list=source_subjects,
            target_subjects=source_subjects,
            model_name=model_name,
            seed=seed,
            time_stratify_labels=time_stratify_labels,
            time_stratify_tolerance=time_stratify_tolerance,
            time_stratify_window=time_stratify_window,
            time_stratify_min_chunk=time_stratify_min_chunk,
            keep_subject_id=keep_subject_id,
        )
        
        X_train = src_train
        y_train = y_src_train.astype(int)
        X_val = src_val
        y_val = y_src_val.astype(int)
        logging.info(
            f"[SOURCE_ONLY] Source domain: train={len(X_train)}, val={len(X_val)} "
            f"samples from {source_domain} group"
        )
        
        # Target domain → ALL data used as test (no split)
        # Load target subjects' full data as test set
        from src.utils.io.split_helpers import _prepare_df_with_label_and_features
        from src.utils.io.loaders import load_subject_csvs
        from src.utils.io.split import _resolve_feature_columns
        from pathlib import Path
        import os

        base_dir = Path("data/processed")
        if (base_dir / model_name).exists() and os.listdir(base_dir / model_name):
            data_dir = base_dir / model_name
        elif (base_dir / "common").exists() and os.listdir(base_dir / "common"):
            data_dir = base_dir / "common"
        else:
            raise FileNotFoundError(
                f"No valid processed data directory for model '{model_name}'."
            )

        target_data, _ = load_subject_csvs(
            eval_subjects, model_name=None, add_subject_id=True,
            base_path=str(data_dir),
        )
        df_target, feature_columns = _prepare_df_with_label_and_features(
            target_data, model_name=model_name,
        )
        X_test = df_target[feature_columns].drop(
            columns=["subject_id"], errors="ignore",
        )
        y_test = df_target["label"].astype(int)
        logging.info(
            f"[SOURCE_ONLY] Target domain: test={len(y_test)} samples "
            f"(ALL data from {target_domain}, {len(eval_subjects)} subjects)"
        )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def prepare_mixed_splits(
    model_name: str,
    tag: str,
    seed: int,
    target_subjects: List[str],
    time_stratify_labels: bool,
    time_stratify_tolerance: float,
    time_stratify_window: float,
    time_stratify_min_chunk: int,
    keep_subject_id: bool = False,
) -> Tuple:
    """Prepare train/val/test splits for mixed (multi-domain) mode.

    In mixed mode:
    - Training: Use ALL subjects (87 pooled) for training
    - Evaluation: Use target domain group (in_domain/out_domain) for val/test

    This allows evaluating how a model trained on the full population
    performs on domain-specific subsets, compared to cross-domain
    (source_only) and within-domain (target_only) training.

    Parameters
    ----------
    model_name : str
        Model name.
    tag : str
        Experiment tag (must contain distance metric and domain info).
    seed : int
        Random seed.
    target_subjects : list of str
        Pre-resolved target subjects from CLI or tag.
    time_stratify_labels : bool
        Whether to use time-stratified splitting.
    time_stratify_tolerance : float
        Class ratio tolerance for time stratification.
    time_stratify_window : float
        Search window for time stratification.
    time_stratify_min_chunk : int
        Minimum chunk size for time stratification.
    keep_subject_id : bool, default=False
        Whether to keep subject_id column (needed for subject-wise oversampling).

    Returns
    -------
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    from src.utils.io.split_helpers import split_data

    # Resolve evaluation subjects (target domain based on tag)
    eval_subjects = resolve_target_subjects_from_tag(
        tag=tag,
        mode="mixed",
        cli_target_subjects=target_subjects,
    )

    if not eval_subjects:
        raise ValueError(
            "[MIXED] No evaluation subjects resolved. "
            "Cannot proceed without target group."
        )

    # Infer target domain from tag
    target_domain = None
    for domain in ["out_domain", "in_domain"]:
        if domain in tag:
            target_domain = domain
            break

    if not target_domain:
        raise ValueError(f"[MIXED] Cannot infer target domain from tag: {tag}")

    # For mixed mode: use ALL subjects for training
    all_subjects = read_subject_list()

    logging.info(
        f"[MIXED] Multi-domain: training on ALL {len(all_subjects)} subjects, "
        f"evaluating on {target_domain} ({len(eval_subjects)} subjects)"
    )

    # Split all subjects → use train partition only
    src_train, _, _, y_src_train, _, _ = split_data(
        subject_split_strategy="subject_time_split",
        subject_list=all_subjects,
        target_subjects=all_subjects,
        model_name=model_name,
        seed=seed,
        time_stratify_labels=time_stratify_labels,
        time_stratify_tolerance=time_stratify_tolerance,
        time_stratify_window=time_stratify_window,
        time_stratify_min_chunk=time_stratify_min_chunk,
        keep_subject_id=keep_subject_id,
    )

    X_train = src_train
    y_train = y_src_train.astype(int)
    logging.info(f"[MIXED] Training: {len(X_train)} samples from ALL {len(all_subjects)} subjects")

    # Split target domain subjects → use val/test partitions
    _, X_val, X_test, _, y_val, y_test = split_data(
        subject_split_strategy="subject_time_split",
        subject_list=eval_subjects,
        target_subjects=eval_subjects,
        model_name=model_name,
        seed=seed,
        time_stratify_labels=time_stratify_labels,
        time_stratify_tolerance=time_stratify_tolerance,
        time_stratify_window=time_stratify_window,
        time_stratify_min_chunk=time_stratify_min_chunk,
    )
    logging.info(
        f"[MIXED] Evaluation: val={len(y_val)}, test={len(y_test)} samples "
        f"from {target_domain} ({len(eval_subjects)} subjects)"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
