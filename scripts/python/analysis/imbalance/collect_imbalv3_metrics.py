#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_imbalv3_metrics.py
==========================
Collect evaluation metrics from imbalv3 (imbalance v3) experiments.

This version supports ratio-based experiments with tags like:
- imbalv3_{ranking}_{metric}_{level}_{imbalance_method}_ratio{X_Y}
- imbalv3_pooled_{imbalance_method}_ratio{X_Y}

Usage:
    python scripts/python/analysis/imbalance/collect_imbalv3_metrics.py
    
    # Specify job IDs
    python scripts/python/analysis/imbalance/collect_imbalv3_metrics.py --jobids 14600726 14600728
"""

import argparse
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

# Default imbalv3 training job IDs
DEFAULT_JOBIDS = [
    # Batch 1-3 (original imbalv3 experiments)
    "14600726",
    "14600728", 
    # Add more job IDs as needed
]

# Output paths
EVAL_DIR = Path(cfg.RESULTS_EVALUATION_PATH)
OUT_DIR = Path(cfg.RESULTS_PATH) / "imbalance_analysis" / "domain_v3"


def extract_metadata_from_tag(tag: str) -> dict:
    """Extract metadata from imbalv3 tag.
    
    Tag formats:
    - Domain-specific with ratio: imbalv3_{ranking}_{metric}_{level}_{imbalance_method}_ratio{X_Y}
      Example: imbalv3_knn_mmd_in_domain_smote_ratio0_5
    - Baseline (no ratio): imbalv3_{ranking}_{metric}_{level}_baseline
      Example: imbalv3_knn_mmd_in_domain_baseline
    - Pooled with ratio: imbalv3_pooled_{imbalance_method}_ratio{X_Y}
      Example: imbalv3_pooled_smote_ratio0_5
    
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
        "imbalance_method": "unknown",
        "ratio": "unknown"
    }
    
    if not tag or not tag.startswith("imbalv3"):
        return result
    
    # Match pooled pattern: imbalv3_pooled_{imbalance}_ratio{X_Y}
    pooled_pattern = r"imbalv3_pooled_(\w+?)(?:_ratio(\d+_\d+))?$"
    pooled_match = re.match(pooled_pattern, tag)
    
    if pooled_match:
        result["ranking_method"] = "pooled"
        result["distance_metric"] = "pooled"
        result["level"] = "pooled"
        result["imbalance_method"] = pooled_match.group(1)
        result["ratio"] = pooled_match.group(2).replace("_", ".") if pooled_match.group(2) else "none"
        return result
    
    # Match domain-specific pattern: imbalv3_{ranking}_{metric}_{level}_{imbalance}_ratio{X_Y}
    # ranking: knn, lof, median_distance
    # metric: mmd, wasserstein, dtw
    # level: in_domain, mid_domain, out_domain
    # imbalance: baseline, smote, smote_tomek, smote_rus, undersample_rus, etc.
    
    domain_pattern = r"imbalv3_(knn|lof|median_distance)_(mmd|wasserstein|dtw)_(in_domain|mid_domain|out_domain)_(.+?)(?:_ratio(\d+_\d+))?$"
    domain_match = re.match(domain_pattern, tag)
    
    if domain_match:
        result["ranking_method"] = domain_match.group(1)
        result["distance_metric"] = domain_match.group(2)
        result["level"] = domain_match.group(3)
        result["imbalance_method"] = domain_match.group(4)
        result["ratio"] = domain_match.group(5).replace("_", ".") if domain_match.group(5) else "none"
        return result
    
    # Fallback: try simpler pattern
    print(f"  [WARN] Could not parse tag: {tag}")
    return result


def extract_metrics_from_json(data: dict, filename: str) -> dict:
    """Extract evaluation metrics from JSON data.
    
    Parameters
    ----------
    data : dict
        JSON data from evaluation file
    filename : str
        Name of the JSON file
    
    Returns
    -------
    dict
        Dictionary with extracted metrics
    """
    tag = data.get("tag", "")
    mode = data.get("mode", "unknown")
    
    # Extract mode from filename if not in data
    if mode == "unknown":
        if "source_only" in filename:
            mode = "source_only"
        elif "target_only" in filename:
            mode = "target_only"
        elif "pooled" in filename:
            mode = "pooled"
    
    meta = extract_metadata_from_tag(tag)
    
    # Handle positive/negative class metrics
    precision = data.get("precision") or data.get("precision_pos", 0)
    recall = data.get("recall") or data.get("recall_pos", 0)
    
    # Calculate positive rate
    cm = data.get("confusion_matrix")
    if cm and len(cm) == 2:
        total = sum(sum(row) for row in cm)
        positives = cm[1][0] + cm[1][1] if len(cm[1]) >= 2 else 0
        pos_rate = positives / total if total > 0 else 0.033
    else:
        pos_rate = data.get("pos_rate", 0.033)
    
    # Calculate F2 if not present
    f2 = data.get("f2_thr") or data.get("f2")
    if f2 is None:
        if precision and recall and (4 * precision + recall) > 0:
            f2 = 5 * precision * recall / (4 * precision + recall)
        else:
            f2 = 0.0
    
    return {
        "file": filename,
        "tag": tag,
        "mode": mode,
        "imbalance_method": meta["imbalance_method"],
        "ratio": meta["ratio"],
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
        "pos_rate": pos_rate,
        "f2": f2,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Collect imbalv3 experiment metrics"
    )
    parser.add_argument(
        "--jobids",
        nargs="+",
        default=None,
        help="Job IDs to collect from. Default: predefined list"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path. Default: results/imbalance_analysis/domain_v3/all_metrics.csv"
    )
    args = parser.parse_args()
    
    jobids = args.jobids or DEFAULT_JOBIDS
    
    print("=" * 60)
    print("[INFO] Collecting imbalv3 Experiment Metrics")
    print("=" * 60)
    print(f"Job IDs: {jobids}")
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    all_records = []
    
    for jobid in jobids:
        # Check both RF and BalancedRF directories
        for model_type in ["RF", "BalancedRF", "EasyEnsemble"]:
            job_eval_dir = EVAL_DIR / model_type / jobid
            
            if not job_eval_dir.exists():
                continue
            
            print(f"\n[INFO] Processing: {model_type}/{jobid}")
            
            # Find all JSON files
            pattern = str(job_eval_dir / "*" / "eval_results_*.json")
            json_files = load_json_glob(pattern, skip_errors=True)
            
            job_count = 0
            for path, data in json_files:
                # Only process imbalv3 tags
                tag = data.get("tag", "")
                if not tag.startswith("imbalv3"):
                    continue
                
                metrics = extract_metrics_from_json(data, path.name)
                metrics["jobid"] = jobid
                metrics["array_index"] = path.parent.name
                metrics["model_type"] = model_type
                all_records.append(metrics)
                job_count += 1
            
            if job_count > 0:
                print(f"  Loaded: {job_count} evaluation files")
    
    if not all_records:
        print("\n[ERROR] No imbalv3 evaluation files found!")
        return 1
    
    # Create DataFrame
    df = pd.DataFrame(all_records)
    
    # Remove duplicates
    initial_count = len(df)
    dedup_keys = ["imbalance_method", "ratio", "ranking_method", "distance_metric", "level", "mode"]
    
    if all(k in df.columns for k in dedup_keys):
        df = df.sort_values("jobid", ascending=True)
        df = df.drop_duplicates(subset=dedup_keys, keep="first")
        removed_count = initial_count - len(df)
        if removed_count > 0:
            print(f"\n[INFO] Removed {removed_count} duplicate experiments")
    
    print("\n" + "-" * 50)
    print("[INFO] Summary")
    print("-" * 50)
    print(f"Total records: {len(df)}")
    
    # Summary by imbalance method
    print("\nBy Imbalance Method:")
    for method, count in df["imbalance_method"].value_counts().items():
        print(f"  {method}: {count}")
    
    # Summary by ratio
    print("\nBy Ratio:")
    for ratio, count in df["ratio"].value_counts().items():
        print(f"  {ratio}: {count}")
    
    # Summary by ranking method
    print("\nBy Ranking Method:")
    for method, count in df["ranking_method"].value_counts().items():
        print(f"  {method}: {count}")
    
    # Save CSV
    out_path = Path(args.output) if args.output else OUT_DIR
    if out_path.is_dir() or not str(out_path).endswith(".csv"):
        out_path = out_path / "all_metrics.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n[INFO] Saved: {out_path}")
    
    print("\n" + "=" * 60)
    print("[DONE]")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
