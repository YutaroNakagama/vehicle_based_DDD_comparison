"""Evaluation pipeline stages for model evaluation.

This module provides focused helper functions for each stage of the
evaluation pipeline, keeping eval_pipeline.py clean and orchestration-focused.
"""

import glob
import logging
import os
from typing import Optional, Tuple

from src import config as cfg


def resolve_jobid_for_evaluation(
    model: str,
    mode: str,
    tag: Optional[str],
    jobid: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """Resolve job ID for loading trained model artifacts.

    Resolution priority:
    1. Provided jobid parameter
    2. Environment variable FIXED_JOBID
    3. Model file matching mode/tag pattern
    4. latest_job.txt file
    5. PBS_JOBID environment variable
    6. Default to "local"

    Parameters
    ----------
    model : str
        Model name (e.g., "RF", "SvmA").
    mode : str
        Experiment mode (e.g., "target_only").
    tag : str, optional
        Experiment tag (e.g., "rank_dtw_mean_high").
    jobid : str, optional
        Explicitly provided job ID.

    Returns
    -------
    tuple of (str, str or None)
        - jobid : Resolved job ID string
        - model_path : Path to model file if auto-detected, None otherwise
    """
    model_path = None
    
    if jobid is not None:
        logging.info(f"[JOBID] Using provided jobid: {jobid}")
        return jobid, model_path
    
    # Try environment variable FIXED_JOBID
    jobid = os.getenv("FIXED_JOBID")
    if jobid:
        logging.info(f"[JOBID] Using FIXED_JOBID from environment: {jobid}")
        return jobid, model_path
    
    # Try to find model file matching mode/tag
    model_root = f"models/{model}"
    if os.path.exists(model_root):
        tag_key = tag.replace("rank_", "") if tag else ""
        pattern = f"{model_root}/**/{model}_{mode}_rank_*{tag_key}*.pkl"
        matches = [
            m for m in glob.glob(pattern, recursive=True)
            if f"{model}_{mode}_" in os.path.basename(m)
        ]
        
        if matches:
            matches.sort(key=os.path.getmtime, reverse=True)
            model_path = matches[0]
            # Extract jobid from path: models/RF/14209090/...
            jobid = model_path.split("/")[2]
            logging.info(f"[JOBID] Auto-detected from model file: {jobid} ({model_path})")
            return jobid, model_path
    
    # Try latest_job.txt
    latest_path = f"models/{model}/{cfg.LATEST_JOB_FILENAME}"
    if os.path.exists(latest_path):
        with open(latest_path, "r") as f:
            jobid = f.readline().strip()
        logging.info(f"[JOBID] Loaded from {latest_path}: {jobid}")
        return jobid, model_path
    
    # Fallback to PBS_JOBID or "local"
    jobid = os.getenv("PBS_JOBID", "local")
    logging.warning(f"[JOBID] No specific jobid found; using fallback: {jobid}")
    
    return jobid, model_path


def extract_metadata_from_tag(tag: Optional[str]) -> Tuple[str, str]:
    """Extract distance metric and level from experiment tag.

    Parameters
    ----------
    tag : str, optional
        Experiment tag. Supports multiple formats:
        - Legacy: "rank_dtw_mean_high"
        - New full format: "full_mean_distance_mmd_out_domain"
        - New format: "rank_mean_distance_mmd_out_domain"

    Returns
    -------
    tuple of (str, str)
        - distance : Distance metric (e.g., "mmd", "dtw", "wasserstein")
        - level : Group level (e.g., "out_domain", "in_domain", "mid_domain")
    """
    import re
    
    if not tag:
        return "unknown", "unknown"
    
    # New full format: full_{method}_{metric}_{level}
    # Example: full_mean_distance_mmd_out_domain
    full_match = re.search(
        r'full_(mean_distance|centroid_umap|lof|knn|median_distance|isolation_forest)_(mmd|dtw|wasserstein)_(out_domain|mid_domain|in_domain)',
        tag
    )
    if full_match:
        method = full_match.group(1)
        metric = full_match.group(2)
        level = full_match.group(3)
        # Return distance as "{method}_{metric}" for file naming
        return f"{method}_{metric}", level
    
    # New rank format: rank_{method}_{metric}_{level}
    # Example: rank_mean_distance_mmd_out_domain
    rank_new_match = re.search(
        r'rank_(mean_distance|centroid_umap|lof|knn|median_distance|isolation_forest)_(mmd|dtw|wasserstein)_(out_domain|mid_domain|in_domain)',
        tag
    )
    if rank_new_match:
        method = rank_new_match.group(1)
        metric = rank_new_match.group(2)
        level = rank_new_match.group(3)
        return f"{method}_{metric}", level
    
    # Legacy format: rank_{metric}_mean_{level}
    # Example: rank_dtw_mean_high, rank_mmd_mean_out_domain
    if tag.startswith("rank_"):
        parts = tag.split("_")  # ["rank", "dtw", "mean", "out_domain"]
        if len(parts) >= 4:
            distance_key = "_".join(parts[1:-1])  # "dtw_mean"
            level = parts[-1]  # "out_domain" or "high"
            return distance_key, level
    
    return "unknown", "unknown"


def extract_full_metadata_from_tag(tag: Optional[str]) -> dict:
    """Extract all metadata components from experiment tag.

    Parameters
    ----------
    tag : str, optional
        Experiment tag. Supports multiple formats:
        - Legacy: "rank_dtw_mean_high"
        - New full format: "full_mean_distance_mmd_out_domain"
        - New format: "rank_mean_distance_mmd_out_domain"

    Returns
    -------
    dict
        Dictionary containing:
        - ranking_method : Ranking method (e.g., "mean_distance", "lof", "knn")
        - distance_metric : Distance metric (e.g., "mmd", "dtw", "wasserstein")
        - level : Group level (e.g., "out_domain", "in_domain", "mid_domain")
    """
    import re
    
    result = {
        "ranking_method": "unknown",
        "distance_metric": "unknown",
        "level": "unknown",
    }
    
    if not tag:
        return result
    
    # New full format: full_{method}_{metric}_{level}
    # Example: full_mean_distance_mmd_out_domain
    full_match = re.search(
        r'full_(mean_distance|centroid_umap|lof|knn|median_distance|isolation_forest)_(mmd|dtw|wasserstein)_(out_domain|mid_domain|in_domain)',
        tag
    )
    if full_match:
        result["ranking_method"] = full_match.group(1)
        result["distance_metric"] = full_match.group(2)
        result["level"] = full_match.group(3)
        return result
    
    # New rank format: rank_{method}_{metric}_{level}
    # Example: rank_mean_distance_mmd_out_domain
    rank_new_match = re.search(
        r'rank_(mean_distance|centroid_umap|lof|knn|median_distance|isolation_forest)_(mmd|dtw|wasserstein)_(out_domain|mid_domain|in_domain)',
        tag
    )
    if rank_new_match:
        result["ranking_method"] = rank_new_match.group(1)
        result["distance_metric"] = rank_new_match.group(2)
        result["level"] = rank_new_match.group(3)
        return result
    
    # Legacy format: rank_{metric}_mean_{level}
    # Example: rank_dtw_mean_high, rank_mmd_mean_out_domain
    if tag.startswith("rank_"):
        parts = tag.split("_")
        if len(parts) >= 4:
            result["ranking_method"] = "mean_distance"  # legacy default
            result["distance_metric"] = parts[1]  # "dtw", "mmd", etc.
            result["level"] = parts[-1]  # "out_domain" or "high"
    
    return result
