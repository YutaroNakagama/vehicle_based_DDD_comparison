"""Threshold optimization utilities for classification models.

This module provides functions to optimize classification thresholds
for imbalanced datasets, with focus on F2 score (favoring recall).
"""

import glob
import json
import logging
import os
from typing import Optional, Tuple

import numpy as np
from sklearn.metrics import fbeta_score


def optimize_threshold_f2(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    num_thresholds: int = 1001,
) -> Tuple[float, float]:
    """Find optimal threshold by maximizing F2 score on validation set.

    F2 score gives more weight to recall than precision, which is appropriate
    for drowsiness detection where false negatives are more costly.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        True binary labels (0 or 1).
    y_proba : np.ndarray of shape (n_samples,)
        Predicted probabilities for positive class.
    num_thresholds : int, default=1001
        Number of threshold values to test (uniformly spaced in [0, 1]).

    Returns
    -------
    tuple of (float, float)
        - best_threshold : Optimal threshold value
        - best_f2 : Corresponding F2 score
    """
    thresholds = np.linspace(0, 1, num_thresholds)
    best_threshold = 0.5
    best_f2 = -1.0
    
    y_true = y_true.astype(int)
    
    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        
        if f2 > best_f2:
            best_f2 = f2
            best_threshold = thr
    
    logging.info(f"[THRESHOLD] Optimized F2: threshold={best_threshold:.3f}, F2={best_f2:.4f}")
    
    return float(best_threshold), float(best_f2)


def load_or_optimize_threshold(
    model: str,
    mode: str,
    tag: Optional[str],
    base_jobid: str,
    run_idx: str,
    fold_idx: int,
    clf,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Optional[float]:
    """Load threshold from file or optimize on validation set.

    This function first attempts to load a pre-computed threshold from a JSON file.
    If not found, it optimizes the threshold on the validation set using F2 score
    and saves the result for future use.

    Parameters
    ----------
    model : str
        Model name (e.g., "RF", "SvmA", "Lstm").
    mode : str
        Experiment mode (e.g., "target_only", "source_only").
    tag : str, optional
        Experiment tag containing distance metric and level.
    base_jobid : str
        Base job ID (without run index).
    run_idx : str
        Run index within the job.
    fold_idx : int
        Fold index (0 for single-fold experiments).
    clf : sklearn classifier
        Fitted classifier with predict_proba or decision_function.
    X_val : np.ndarray
        Validation features.
    y_val : np.ndarray
        Validation labels.

    Returns
    -------
    float or None
        Optimal threshold value, or None if optimization fails.
    """
    # Parse distance metric and level from tag
    if tag and tag.startswith("rank_"):
        parts = tag.split("_")  # ["rank", "dtw", "mean", "high"]
        distance_key = "_".join(parts[1:-1]) if len(parts) >= 3 else "unknown"
        level = parts[-1] if len(parts) >= 2 else "unknown"
    else:
        distance_key, level = "unknown", "unknown"
    
    jobid_idx = f"{base_jobid}_{run_idx}"
    
    # Try to load existing threshold file
    pattern = (
        f"models/{model}/{base_jobid}/{base_jobid}[{run_idx}]/"
        f"threshold_{model}_{mode}_rank_{distance_key}_{level}_{jobid_idx}_{fold_idx}.json"
    )
    candidates = glob.glob(pattern, recursive=True)
    
    if candidates:
        candidates.sort(key=os.path.getmtime, reverse=True)
        threshold_path = candidates[0]
        
        try:
            with open(threshold_path, "r") as f:
                meta = json.load(f)
            threshold = float(meta.get("threshold", 0.5))
            logging.info(f"[THRESHOLD] Loaded from {threshold_path}: {threshold:.3f}")
            return threshold
        except Exception as e:
            logging.warning(f"[THRESHOLD] Failed to load from {threshold_path}: {e}")
    
    # Threshold file not found; optimize on validation set
    logging.info(f"[THRESHOLD] No file found. Optimizing on validation set...")
    
    if X_val is None or y_val is None or len(y_val) == 0:
        logging.warning("[THRESHOLD] Validation set not available; cannot optimize.")
        return None
    
    # Get probabilities from classifier
    try:
        if hasattr(clf, "predict_proba"):
            y_proba = clf.predict_proba(X_val)[:, 1]
        elif hasattr(clf, "decision_function"):
            scores = clf.decision_function(X_val)
            # Normalize to [0, 1]
            score_min, score_max = float(np.min(scores)), float(np.max(scores))
            y_proba = (scores - score_min) / (score_max - score_min + 1e-12)
        else:
            logging.warning("[THRESHOLD] Classifier has no probability scores.")
            return None
    except Exception as e:
        logging.warning(f"[THRESHOLD] Failed to get probabilities: {e}")
        return None
    
    # Optimize threshold
    threshold, best_f2 = optimize_threshold_f2(y_val, y_proba)
    
    # Save threshold to file
    try:
        candidate_subdir = os.path.join("models", model, base_jobid, f"{base_jobid}[{run_idx}]")
        target_dir = candidate_subdir if os.path.isdir(candidate_subdir) else os.path.join("models", model, base_jobid)
        os.makedirs(target_dir, exist_ok=True)
        
        threshold_filename = (
            f"threshold_{model}_{mode}_rank_{distance_key}_{level}_{jobid_idx}_{fold_idx}.json"
        )
        threshold_path = os.path.join(target_dir, threshold_filename)
        
        with open(threshold_path, "w") as f:
            json.dump({"threshold": threshold, "f2_score": best_f2}, f, indent=2)
        
        logging.info(f"[THRESHOLD] Saved to {threshold_path}")
    except Exception as e:
        logging.warning(f"[THRESHOLD] Failed to save threshold file: {e}")
    
    return threshold


def extract_jobid_components(jobid: str, model_path: Optional[str] = None) -> Tuple[str, str]:
    """Extract base job ID and run index from job ID string or model path.

    Parameters
    ----------
    jobid : str
        Job ID string (may include run index like "14209090[1]").
    model_path : str, optional
        Path to model file (can be used to extract jobid components).

    Returns
    -------
    tuple of (str, str)
        - base_jobid : Job ID without run index
        - run_idx : Run index (default "1" if not found)
    """
    import re
    
    base_jobid = None
    run_idx = None
    
    # Try to extract from model_path first
    if model_path:
        # Example: .../models/RF/14209090/14209090[1]/RF_...
        parts = model_path.split("/")
        if len(parts) >= 4:
            base_jobid = parts[2]
            subdir = parts[3]  # "14209090[1]"
            match = re.match(r"^(\d+)\[(\d+)\]$", subdir)
            if match:
                base_jobid = match.group(1)
                run_idx = match.group(2)
    
    # Fallback to parsing jobid string
    if base_jobid is None:
        match = re.match(r"^(\d+)(?:\[(\d+)\])?$", str(jobid))
        if match:
            base_jobid = match.group(1)
            run_idx = match.group(2) or "1"
        else:
            # Clean up special characters
            base_jobid = str(jobid).replace("[", "").replace("]", "")
            run_idx = "1"
    
    if run_idx is None:
        run_idx = "1"
    
    return base_jobid, run_idx
