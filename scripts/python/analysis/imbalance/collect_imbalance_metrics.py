#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_imbalance_metrics.py
============================
Collect evaluation metrics from all imbalance experiment job IDs.

Reads from multiple training job IDs and aggregates all evaluation JSON files
into a single CSV for visualization.

Usage:
    python scripts/python/analysis/imbalance/collect_imbalance_metrics.py
"""

import os
import re
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src import config as cfg
from src.utils.io.data_io import load_json_glob

# Training job IDs from the imbalance experiments
TRAIN_JOBIDS = [
    "14572812",  # OFFSET=0   (experiments 0-29)
    "14572963",  # OFFSET=30  (experiments 30-59)
    "14572964",  # OFFSET=60  (experiments 60-89)
    "14572965",  # OFFSET=90  (experiments 90-119)
    "14573051",  # OFFSET=120 (experiments 120-149)
    "14573104",  # OFFSET=150 (experiments 150-179)
    "14573136",  # OFFSET=180 (experiments 180-209)
    "14573166",  # OFFSET=210 (experiments 210-219)
]

# Output paths
EVAL_DIR = Path(cfg.RESULTS_EVALUATION_PATH) / "RF"
OUT_DIR = Path(cfg.RESULTS_PATH) / "imbalance_analysis" / "domain"


def extract_metadata_from_tag(tag: str) -> dict:
    """Extract metadata from imbalance tag.
    
    Tag formats:
    - Domain-specific: imbalance_{ranking}_{metric}_{level}_{imbalance_method}
      Example: imbalance_knn_mmd_in_domain_smote
    - Pooled: imbalance_pooled_{imbalance_method}
      Example: imbalance_pooled_smote_rus
    
    Parameters
    ----------
    tag : str
        Tag string from evaluation JSON
    
    Returns
    -------
    dict
        Dictionary with extracted metadata
    """
    result = {
        "ranking_method": "unknown",
        "distance_metric": "unknown", 
        "level": "unknown",
        "imbalance_method": "unknown"
    }
    
    # Match pooled pattern first: imbalance_pooled_{imbalance}
    pooled_pattern = r"imbalance_pooled_(baseline|smote_tomek|smote_rus|smote)"
    pooled_match = re.match(pooled_pattern, tag)
    
    if pooled_match:
        result["ranking_method"] = "pooled"
        result["distance_metric"] = "pooled"
        result["level"] = "pooled"
        result["imbalance_method"] = pooled_match.group(1)
        return result
    
    # Match domain-specific pattern: imbalance_{ranking}_{metric}_{level}_{imbalance}
    pattern = r"imbalance_(knn|lof|median_distance|mean_distance|centroid_umap|isolation_forest)_(mmd|dtw|wasserstein)_(out_domain|mid_domain|in_domain)_(baseline|smote_tomek|smote_rus|smote)"
    match = re.match(pattern, tag)
    
    if match:
        result["ranking_method"] = match.group(1)
        result["distance_metric"] = match.group(2)
        result["level"] = match.group(3)
        result["imbalance_method"] = match.group(4)
    
    return result


def extract_mode_from_filename(filename: str) -> str:
    """Extract training mode from evaluation filename."""
    mode_match = re.search(r"eval_results_RF_(pooled|source_only|target_only)", filename)
    if mode_match:
        return mode_match.group(1)
    return "unknown"


def extract_metrics_from_json(data: dict, filename: str) -> dict:
    """Extract metrics from a single evaluation JSON.
    
    Parameters
    ----------
    data : dict
        Parsed JSON data
    filename : str
        Filename of the JSON
    
    Returns
    -------
    dict
        Dictionary with extracted metrics
    """
    tag = data.get("tag", "")
    meta = extract_metadata_from_tag(tag)
    mode = extract_mode_from_filename(filename)
    
    # Extract classification report for positive class (pos_rate)
    cr = data.get("classification_report", {}) or {}
    pos_block = None
    for key in ["1", "1.0", "True", "pos", "positive"]:
        if key in cr and isinstance(cr[key], dict):
            pos_block = cr[key]
            break
    
    pos_rate = None
    if pos_block and "support" in pos_block:
        total_support = cr.get("weighted avg", {}).get("support", 0)
        if total_support > 0:
            pos_rate = pos_block["support"] / total_support
    
    # Get precision and recall
    precision = data.get("precision") or (pos_block.get("precision") if pos_block else None)
    recall = data.get("recall") or (pos_block.get("recall") if pos_block else None)
    
    # Calculate F2 score if not present: F2 = (1+2^2) * P * R / (2^2 * P + R) = 5*P*R / (4*P + R)
    f2 = data.get("f2")
    if f2 is None and precision is not None and recall is not None:
        if precision + recall > 0:  # Avoid division by zero
            f2 = (5 * precision * recall) / (4 * precision + recall)
        else:
            f2 = 0.0
    
    return {
        "file": filename,
        "tag": tag,
        "mode": mode,
        "imbalance_method": meta["imbalance_method"],
        "ranking_method": meta["ranking_method"],
        "distance_metric": meta["distance_metric"],
        "level": meta["level"],
        # Metrics
        "accuracy": data.get("accuracy"),
        "precision": precision,
        "recall": recall,
        "f1": data.get("f1"),
        "auc": data.get("roc_auc") or data.get("auc"),
        "auc_pr": data.get("auc_pr"),
        "pos_rate": pos_rate or data.get("pos_rate", 0.033),
        "f2": f2,
    }


def main():
    print("=" * 60)
    print("[INFO] Collecting Imbalance Experiment Metrics")
    print("=" * 60)
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    all_records = []
    
    for jobid in TRAIN_JOBIDS:
        job_eval_dir = EVAL_DIR / jobid
        
        if not job_eval_dir.exists():
            print(f"[WARN] Evaluation dir not found: {job_eval_dir}")
            continue
        
        print(f"\n[INFO] Processing job: {jobid}")
        
        # Find all JSON files
        pattern = str(job_eval_dir / "*" / "eval_results_*.json")
        json_files = load_json_glob(pattern, skip_errors=True)
        
        job_count = 0
        for path, data in json_files:
            metrics = extract_metrics_from_json(data, path.name)
            metrics["jobid"] = jobid
            metrics["array_index"] = path.parent.name
            all_records.append(metrics)
            job_count += 1
        
        print(f"  Loaded: {job_count} evaluation files")
    
    # Create DataFrame
    df = pd.DataFrame(all_records)
    
    print("\n" + "-" * 50)
    print("[INFO] Summary")
    print("-" * 50)
    print(f"Total records: {len(df)}")
    
    # Summary by imbalance method
    print("\nBy Imbalance Method:")
    for method, count in df["imbalance_method"].value_counts().items():
        print(f"  {method}: {count}")
    
    # Summary by ranking method
    print("\nBy Ranking Method:")
    for method, count in df["ranking_method"].value_counts().items():
        print(f"  {method}: {count}")
    
    # Save CSV
    out_path = OUT_DIR / "all_metrics.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[INFO] Saved: {out_path}")
    
    print("\n" + "=" * 60)
    print("[DONE]")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
