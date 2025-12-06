"""Target subject resolution utilities for domain analysis.

This module provides functions to resolve target subject groups
from rank files and apply mode-based filtering for domain generalization
experiments.
"""

import logging
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from src import config as cfg


def resolve_target_subjects_from_tag(
    tag: Optional[str],
    mode: Optional[str] = None,
    cli_target_subjects: Optional[List[str]] = None,
) -> List[str]:
    """Resolve target subjects from tag using rank_names.txt mapping.

    This function reads the rank_names.txt file to map a tag (e.g., "rank_dtw_mean_high")
    to a specific target group file, then loads the subject list from that file.

    Parameters
    ----------
    tag : str, optional
        Experiment tag containing distance metric and level (e.g., "rank_dtw_mean_high").
    mode : str, optional
        Experiment mode (e.g., "target_only", "source_only").
        Only resolves targets for relevant modes.
    cli_target_subjects : list of str, optional
        Target subjects provided via CLI. Used as fallback or for intersection.

    Returns
    -------
    list of str
        Resolved target subject IDs. Empty list if resolution fails.
    """
    target_subjects = []
    
    # Only resolve for modes that use target groups
    if mode not in ["source_only", "target_only"]:
        logging.info(f"[TARGET] Mode '{mode}' does not require target resolution.")
        return cli_target_subjects or []
    
    if not tag:
        logging.info("[TARGET] No tag provided; using CLI target_subjects if available.")
        return cli_target_subjects or []
    
    # Check if rank_names.txt exists (try both ranks29/mean_distance_legacy and root directory)
    rank_file_new = os.path.join(
        cfg.RESULTS_DOMAIN_ANALYSIS_PATH, "distance", "subject-wise", "ranks", "ranks29", "mean_distance_legacy", "ranks29_names.txt"
    )
    rank_file_old = os.path.join(
        cfg.RESULTS_DOMAIN_ANALYSIS_PATH, "distance", cfg.RANK_NAMES_FILENAME
    )
    
    rank_file = rank_file_new if os.path.exists(rank_file_new) else rank_file_old
    
    if not os.path.exists(rank_file):
        logging.warning(f"[TARGET] rank_names.txt not found at {rank_file}. Using CLI targets.")
        return cli_target_subjects or []
    
    # Read rank_names.txt
    with open(rank_file) as f:
        lines = [x.strip() for x in f.readlines() if x.strip()]
    
    # Extract base key from tag (remove "rank_" prefix)
    tag_key = tag.replace("rank_", "")
    
    # Find matching group file
    match = [x for x in lines if os.path.basename(x).startswith(tag_key)]
    
    if not match:
        logging.warning(f"[TARGET] No matching group found in rank_names.txt for tag='{tag_key}'.")
        return cli_target_subjects or []
    
    # Load subjects from group file
    group_file = os.path.normpath(match[0])
    if not os.path.exists(group_file):
        logging.warning(f"[TARGET] Group file not found: {group_file}")
        return cli_target_subjects or []
    
    with open(group_file) as g:
        file_targets = [s.strip() for s in g.readlines() if s.strip()]
    
    # If CLI also provided targets, take intersection
    if cli_target_subjects:
        target_subjects = [s for s in cli_target_subjects if s in file_targets]
    else:
        target_subjects = file_targets
    
    # Remove duplicates while preserving order
    target_subjects = list(dict.fromkeys(target_subjects))
    
    logging.info(
        f"[TARGET] Resolved {len(target_subjects)} target subjects from {group_file}"
    )
    
    return target_subjects


# Source group for source_only mode training
# Options: "in_domain" (LOW - typical subjects), "mid_domain" (MIDDLE)
SOURCE_ONLY_TRAIN_GROUP = "in_domain"  # Changed from "mid_domain" to use LOW group


def resolve_source_group_subjects(tag: Optional[str], source_group: Optional[str] = None) -> List[str]:
    """Resolve source group subjects for source_only mode training.

    This function extracts the ranking method and distance metric from the tag
    to locate the correct group file. For in_domain experiments with source_only,
    this ensures the same group is used as target_only mode.

    Parameters
    ----------
    tag : str, optional
        Experiment tag to extract ranking method and distance metric 
        (e.g., "rank_knn_mmd_in_domain", "full_mean_distance_dtw_in_domain").
    source_group : str, optional
        Which group to use for training. Options: "in_domain" (LOW), "mid_domain" (MIDDLE).
        If None, uses SOURCE_ONLY_TRAIN_GROUP constant.

    Returns
    -------
    list of str
        Source group subject IDs.

    Raises
    ------
    ValueError
        If metric prefix cannot be inferred from tag.
    FileNotFoundError
        If source group file does not exist.
    """
    if not tag:
        raise ValueError("[SOURCE] Tag is required to resolve source group.")
    
    # Determine which group to use
    group_level = source_group or SOURCE_ONLY_TRAIN_GROUP
    
    # Extract tag key (remove "rank_" or "full_" prefix)
    tag_key = tag.replace("rank_", "").replace("full_", "")
    
    # Extract distance metric (dtw, mmd, wasserstein)
    metric_prefix = None
    for prefix in ["dtw", "mmd", "wasserstein"]:
        if prefix in tag_key:
            metric_prefix = prefix
            break
    
    if metric_prefix is None:
        raise ValueError(
            f"[SOURCE] Cannot infer metric prefix from tag='{tag}'. "
            f"Expected one of ['dtw', 'mmd', 'wasserstein']."
        )
    
    # Extract ranking method from tag (knn, lof, mean_distance, median_distance, etc.)
    ranking_methods = ["mean_distance", "median_distance", "centroid_umap", "isolation_forest", "knn", "lof", "medoid", "centroid_mds"]
    ranking_method = None
    for method in ranking_methods:
        if method in tag_key:
            ranking_method = method
            break
    
    # Default to mean_distance if not found (for backward compatibility)
    if ranking_method is None:
        ranking_method = "mean_distance"
        logging.info(f"[SOURCE] No ranking method found in tag='{tag}', defaulting to 'mean_distance'")
    
    # Build path to source group file
    ranks_dir = Path(cfg.RESULTS_DOMAIN_ANALYSIS_PATH) / "distance" / "subject-wise" / "ranks" / "ranks29" / ranking_method
    source_file = ranks_dir / f"{metric_prefix}_{group_level}.txt"
    
    if not source_file.exists():
        # Fallback to mean_distance if ranking method directory doesn't exist
        fallback_dir = Path(cfg.RESULTS_DOMAIN_ANALYSIS_PATH) / "distance" / "subject-wise" / "ranks" / "ranks29" / "mean_distance"
        fallback_file = fallback_dir / f"{metric_prefix}_{group_level}.txt"
        
        if fallback_file.exists():
            logging.warning(
                f"[SOURCE] Group file not found at {source_file}, falling back to {fallback_file}"
            )
            source_file = fallback_file
        else:
            raise FileNotFoundError(
                f"[SOURCE] Source group file does not exist: {source_file}"
            )
    
    with open(source_file) as f:
        source_subjects = [s.strip() for s in f if s.strip()]
    
    group_name = "LOW" if group_level == "in_domain" else "MIDDLE"
    logging.info(f"[SOURCE] Loaded {len(source_subjects)} {group_name} group subjects from {source_file}")
    
    return source_subjects


def resolve_mid_domain_group_subjects(tag: Optional[str]) -> List[str]:
    """Resolve source group subjects for source_only mode (legacy wrapper).

    This function is kept for backward compatibility.
    It now uses resolve_source_group_subjects() with the configured source group.

    Parameters
    ----------
    tag : str, optional
        Experiment tag to extract distance metric prefix (e.g., "rank_dtw_mean_high").

    Returns
    -------
    list of str
        Source group subject IDs (LOW or MIDDLE based on SOURCE_ONLY_TRAIN_GROUP).
    """
    return resolve_source_group_subjects(tag)


def filter_data_by_mode(
    data: pd.DataFrame,
    mode: str,
    subject_list: List[str],
    target_subjects: Optional[List[str]] = None,
    tag: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Filter data and subject list based on experiment mode.

    This function applies mode-specific filtering for domain generalization:
    - target_only: Keep only target subjects
    - source_only: Exclude target subjects (keep source)
    - joint_train: Combine source and target subjects

    Parameters
    ----------
    data : pd.DataFrame
        Full dataset with subject_id column.
    mode : str
        Experiment mode ("target_only", "source_only", "joint_train", "pooled").
    subject_list : list of str
        Full subject list.
    target_subjects : list of str, optional
        Target subject IDs (required for target_only/source_only).
    tag : str, optional
        Experiment tag for resolving targets from rank files.

    Returns
    -------
    tuple of (pd.DataFrame, list of str, list of str)
        - Filtered data
        - Updated subject_list
        - Resolved target_subjects
    """
    # Ensure subject_id column exists
    if "subject_id" not in data.columns:
        if "filename" in data.columns:
            fn_series = data["filename"]
        elif "FileName" in data.columns:
            fn_series = data["FileName"]
        else:
            fn_series = pd.Series(["unknown"] * len(data))
            logging.warning("[FILTER] 'filename' column not found; subject_id set to 'unknown'")
        
        data["subject_id"] = fn_series.apply(
            lambda f: re.search(r"S\d{4}_\d", f).group(0)
            if isinstance(f, str) and re.search(r"S\d{4}_\d", f)
            else "unknown"
        )
        unique_ids = data["subject_id"].nunique()
        logging.info(f"[FILTER] Injected subject_id column (n={unique_ids} unique IDs).")
    
    # Resolve target subjects if not provided
    if mode in ["target_only", "source_only"] and not target_subjects:
        target_subjects = resolve_target_subjects_from_tag(tag, mode)
    
    # Apply mode-specific filtering
    if mode == "target_only":
        if target_subjects:
            data = data[data["subject_id"].isin(target_subjects)].reset_index(drop=True)
            subject_list = list(dict.fromkeys(target_subjects))
            logging.info(
                f"[FILTER] target_only: Restricted to {len(data)} samples from {len(subject_list)} subjects."
            )
        else:
            logging.warning("[FILTER] target_only: No target_subjects resolved. No filtering applied.")
    
    elif mode == "source_only":
        if target_subjects:
            data = data[~data["subject_id"].isin(target_subjects)].reset_index(drop=True)
            subject_list = [s for s in subject_list if s not in target_subjects]
            logging.info(
                f"[FILTER] source_only: Excluded targets, kept {len(data)} samples from {len(subject_list)} subjects."
            )
        else:
            logging.warning("[FILTER] source_only: No target_subjects to exclude. Using all data.")
    
    elif mode == "joint_train":
        if target_subjects:
            subject_list = list(dict.fromkeys(list(subject_list) + list(target_subjects)))
            logging.info(f"[FILTER] joint_train: Combined subject list size = {len(subject_list)}")
        else:
            logging.info("[FILTER] joint_train: No additional target_subjects to combine.")
    
    else:
        logging.info(f"[FILTER] Mode '{mode}': No filtering applied.")
    
    return data, subject_list, target_subjects or []
