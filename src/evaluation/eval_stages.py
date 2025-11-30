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
        Experiment tag (e.g., "rank_dtw_mean_high").

    Returns
    -------
    tuple of (str, str)
        - distance : Distance metric (e.g., "dtw_mean")
        - level : Group level (e.g., "out_domain", "in_domain", "mid_domain")
    """
    if not tag or not tag.startswith("rank_"):
        return "unknown", "unknown"
    
    parts = tag.split("_")  # ["rank", "dtw", "mean", "out_domain"]
    if len(parts) < 3:
        return "unknown", "unknown"
    
    distance_key = "_".join(parts[1:-1])  # "dtw_mean"
    level = parts[-1]  # "out_domain"
    
    return distance_key, level
