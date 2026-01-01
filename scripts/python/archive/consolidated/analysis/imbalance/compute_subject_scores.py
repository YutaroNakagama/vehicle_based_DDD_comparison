#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_subject_scores.py
=========================

Compute per-subject evaluation scores from saved evaluation results.

This script:
1. Loads evaluation JSON files with predictions (y_pred_proba)
2. Reloads test data with subject_id to match predictions to subjects
3. Computes per-subject metrics (Recall, Precision, F1, etc.)
4. Saves per-subject scores for statistical testing

Usage:
    python scripts/python/analysis/imbalance/compute_subject_scores.py

Output:
    results/imbalance_analysis/domain/subject_scores.csv
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src import config as cfg
from src.utils.io.loaders import load_subject_csvs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ============================================================
# Configuration
# ============================================================
EVAL_BASE_DIR = Path(cfg.RESULTS_EVALUATION_PATH) / "RF"
OUTPUT_DIR = Path("results/imbalance_analysis/domain")
OUTPUT_FILE = OUTPUT_DIR / "subject_scores.csv"

# Imbalance experiment job IDs (from collect_imbalance_metrics.py)
IMBALANCE_JOB_IDS = [
    "14572812", "14572963", "14572964", "14572965",
    "14573051", "14573104", "14573136", "14573166",
]

# Pattern to extract metadata from tag
IMBALANCE_TAG_PATTERN = r"imbalance_(knn|lof|median_distance)_(mmd|dtw|wasserstein)_(out_domain|mid_domain|in_domain)_(baseline|smote_tomek|smote_rus|smote)"


# ============================================================
# Helper Functions
# ============================================================
def parse_tag(tag: str) -> Optional[Dict[str, str]]:
    """Extract metadata from imbalance experiment tag."""
    import re
    pattern = IMBALANCE_TAG_PATTERN
    match = re.search(pattern, tag)
    if match:
        return {
            "ranking_method": match.group(1),
            "distance_metric": match.group(2),
            "level": match.group(3),
            "imbalance_method": match.group(4),
        }
    return None


def load_test_data_with_subject_id(
    subjects: List[str],
    mode: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load test data with subject_id column preserved.
    
    Returns
    -------
    X_test : pd.DataFrame
        Feature matrix
    y_test : pd.Series
        Labels
    subject_ids : pd.Series
        Subject ID for each sample
    """
    data, _ = load_subject_csvs(
        subjects,
        model_name=None,
        add_subject_id=True,
        base_path="data/processed/common"
    )
    
    # Extract features and labels
    label_col = "label"
    if label_col not in data.columns:
        # Try to create label from KSS
        if "KSS_Theta_Alpha_Beta_percent" in data.columns:
            data[label_col] = (data["KSS_Theta_Alpha_Beta_percent"] >= 8).astype(int)
    
    # Get subject_id before dropping
    subject_ids = data["subject_id"].copy()
    
    # Feature columns (exclude non-feature columns)
    exclude_cols = ["subject_id", "label", "Timestamp", "FileName", "filename"]
    feature_cols = [c for c in data.columns if c not in exclude_cols]
    
    X = data[feature_cols]
    y = data[label_col] if label_col in data.columns else pd.Series([0] * len(data))
    
    return X, y, subject_ids


def compute_subject_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subject_ids: np.ndarray,
) -> pd.DataFrame:
    """Compute per-subject metrics.
    
    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    subject_ids : array-like
        Subject ID for each sample
    
    Returns
    -------
    pd.DataFrame
        Per-subject metrics with columns: subject, recall, precision, f1, f2, accuracy, n_samples, n_pos
    """
    results = []
    
    unique_subjects = np.unique(subject_ids)
    
    for subj in unique_subjects:
        mask = subject_ids == subj
        y_t = y_true[mask]
        y_p = y_pred[mask]
        
        n_samples = len(y_t)
        n_pos = int(y_t.sum())
        
        # Skip subjects with no positive samples (can't compute recall)
        if n_pos == 0:
            logging.warning(f"Subject {subj} has no positive samples, skipping")
            continue
        
        results.append({
            "subject": subj,
            "recall": recall_score(y_t, y_p, zero_division=0),
            "precision": precision_score(y_t, y_p, zero_division=0),
            "f1": f1_score(y_t, y_p, zero_division=0),
            "f2": fbeta_score(y_t, y_p, beta=2, zero_division=0),
            "accuracy": accuracy_score(y_t, y_p),
            "n_samples": n_samples,
            "n_pos": n_pos,
        })
    
    return pd.DataFrame(results)


def process_eval_file(eval_json_path: Path) -> Optional[pd.DataFrame]:
    """Process a single evaluation JSON file.
    
    Returns per-subject scores if successful, None otherwise.
    """
    try:
        with open(eval_json_path) as f:
            data = json.load(f)
        
        tag = data.get("tag", "")
        mode = data.get("mode", "")
        subjects = data.get("subject_list", [])
        y_pred_proba = data.get("y_pred_proba", [])
        thr = data.get("thr", 0.5)
        
        # Parse tag to get experiment metadata
        meta = parse_tag(tag)
        if not meta:
            logging.debug(f"Skipping non-imbalance file: {eval_json_path.name}")
            return None
        
        if not subjects or not y_pred_proba:
            logging.warning(f"Missing subjects or predictions in {eval_json_path.name}")
            return None
        
        # Filter to target subjects based on level
        ranks_dir = Path("results/domain_analysis/distance/subject-wise/ranks/ranks29")
        rank_method = meta["ranking_method"]
        dist_metric = meta["distance_metric"]
        level = meta["level"]
        
        group_file = ranks_dir / rank_method / f"{dist_metric}_{level}.txt"
        if not group_file.exists():
            logging.warning(f"Group file not found: {group_file}")
            return None
        
        target_subjects = group_file.read_text().strip().split("\n")
        target_subjects = [s.strip() for s in target_subjects if s.strip()]
        
        # Load test data with subject_id
        X_test, y_test, subject_ids = load_test_data_with_subject_id(
            target_subjects, mode
        )
        
        # Check sample count matches
        if len(y_pred_proba) != len(y_test):
            logging.warning(
                f"Sample count mismatch: proba={len(y_pred_proba)}, test={len(y_test)} "
                f"in {eval_json_path.name}"
            )
            return None
        
        # Apply threshold to get predictions
        y_pred = (np.array(y_pred_proba) >= thr).astype(int)
        
        # Compute per-subject metrics
        subject_metrics = compute_subject_metrics(
            y_test.values if hasattr(y_test, 'values') else y_test,
            y_pred,
            subject_ids.values if hasattr(subject_ids, 'values') else subject_ids,
        )
        
        # Add experiment metadata
        subject_metrics["mode"] = mode
        subject_metrics["ranking_method"] = meta["ranking_method"]
        subject_metrics["distance_metric"] = meta["distance_metric"]
        subject_metrics["level"] = meta["level"]
        subject_metrics["imbalance_method"] = meta["imbalance_method"]
        subject_metrics["threshold"] = thr
        subject_metrics["source_file"] = eval_json_path.name
        
        return subject_metrics
        
    except Exception as e:
        logging.error(f"Error processing {eval_json_path}: {e}")
        return None


def collect_all_subject_scores() -> pd.DataFrame:
    """Collect per-subject scores from all imbalance evaluation files."""
    all_scores = []
    
    for jobid in IMBALANCE_JOB_IDS:
        job_dir = EVAL_BASE_DIR / jobid
        if not job_dir.exists():
            logging.warning(f"Job directory not found: {job_dir}")
            continue
        
        # Find all array task directories
        for array_dir in job_dir.iterdir():
            if not array_dir.is_dir():
                continue
            
            # Find JSON files
            for json_file in array_dir.glob("*.json"):
                if not json_file.name.startswith("eval_results"):
                    continue
                
                scores = process_eval_file(json_file)
                if scores is not None and len(scores) > 0:
                    scores["jobid"] = jobid
                    scores["array_index"] = array_dir.name
                    all_scores.append(scores)
    
    if not all_scores:
        logging.error("No valid subject scores collected!")
        return pd.DataFrame()
    
    return pd.concat(all_scores, ignore_index=True)


# ============================================================
# Main
# ============================================================
def main():
    """Main entry point."""
    print("=" * 60)
    print("[INFO] Computing per-subject scores for statistical testing")
    print("=" * 60)
    print()
    
    # Collect all subject scores
    df = collect_all_subject_scores()
    
    if df.empty:
        print("[ERROR] No subject scores collected!")
        return 1
    
    print(f"[INFO] Collected {len(df)} subject-level records")
    print(f"[INFO] Unique experiments: {df.groupby(['imbalance_method', 'ranking_method', 'distance_metric', 'level', 'mode']).ngroups}")
    print()
    
    # Summary statistics
    print("[INFO] Records by imbalance method:")
    for method in sorted(df["imbalance_method"].unique()):
        n = len(df[df["imbalance_method"] == method])
        print(f"  {method}: {n}")
    print()
    
    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"[INFO] Saved to: {OUTPUT_FILE}")
    
    # Quick summary
    print()
    print("[INFO] Mean Recall by imbalance method (target_only):")
    target_df = df[df["mode"] == "target_only"]
    if not target_df.empty:
        for method in sorted(target_df["imbalance_method"].unique()):
            mean_recall = target_df[target_df["imbalance_method"] == method]["recall"].mean()
            std_recall = target_df[target_df["imbalance_method"] == method]["recall"].std()
            print(f"  {method}: {mean_recall:.4f} ± {std_recall:.4f}")
    
    print()
    print("[DONE] Subject scores computed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
