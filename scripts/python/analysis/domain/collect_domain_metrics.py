#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_domain_metrics.py
=========================
Unified script for collecting evaluation metrics from domain analysis experiments.

This script consolidates:
- collect_evaluation_metrics.py (pooled 40-cases analysis)
- collect_evaluation_metrics_ranked.py (ranked domain analysis with multiple methods)

Usage:
    python collect_domain_metrics.py                 # Default: pooled (40-cases)
    python collect_domain_metrics.py --ranked        # Ranked domain analysis
    python collect_domain_metrics.py --all           # Both pooled and ranked
"""

import argparse
import os
import re
import pandas as pd
from pathlib import Path
from glob import glob

from src import config as cfg
from src.utils.io.data_io import load_json_glob, save_csv
from src.evaluation.metrics import extract_metrics_from_eval_json

# Configuration
EVAL_DIR = Path(cfg.RESULTS_EVALUATION_PATH) / "RF"
OUT_DIR = Path(cfg.RESULTS_DOMAIN_ANALYSIS_PATH) / "summary" / "csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Ranking methods to collect (for ranked mode)
RANKING_METHODS = [
    "mean_distance",
    "centroid_umap",
    "lof",
    "knn",
    "median_distance",
    "isolation_forest",
]


# ============================================================
# Pooled Mode (40-cases)
# ============================================================
def collect_pooled_metrics():
    """Collect all evaluation metrics from pooled 40-cases analysis."""
    print("\n=== Collecting Pooled Metrics (40-cases) ===")
    
    # Find latest job or search all
    latest_job_file = EVAL_DIR / cfg.LATEST_JOB_FILENAME
    if latest_job_file.exists():
        with open(latest_job_file, "r") as f:
            latest_jobid = f.read().strip()
        search_pattern = str(EVAL_DIR / latest_jobid / "*" / "eval_results_*.json")
        print(f"[INFO] Using latest_jobid={latest_jobid}")
    else:
        search_pattern = str(EVAL_DIR / "*" / "*" / "eval_results_*.json")
        print(f"[WARN] latest_job.txt not found — scanning all jobs.")
    
    # Load all JSON files
    json_files = load_json_glob(search_pattern, skip_errors=True)
    
    if not json_files:
        raise FileNotFoundError(f"No eval_results_*.json found under {EVAL_DIR}")
    
    # Extract metrics
    records = []
    for path, data in json_files:
        metrics = extract_metrics_from_eval_json(data, filename=path.name)
        records.append(metrics)
    
    all_metrics = pd.DataFrame(records)
    print(f"[INFO] Loaded {len(all_metrics)} evaluation JSONs.")
    
    # Filter to test split
    test_df = all_metrics[all_metrics["split"] == "test"].copy()
    if "level" in test_df.columns:
        cat = pd.CategoricalDtype(categories=["out_domain", "mid_domain", "in_domain"], ordered=True)
        test_df["level"] = test_df["level"].astype(cat)
    
    # Save
    test_path = OUT_DIR / "summary_40cases_test.csv"
    save_csv(test_df, str(test_path))
    print(f"[INFO] Saved: {test_path}")
    
    return test_df


# ============================================================
# Ranked Mode
# ============================================================
def parse_filename_for_metadata(filename: str) -> dict:
    """Parse filename to extract distance metric and level.
    
    Format: eval_results_RF_{mode}_rank_{method}_{distance_metric}_{level}_mean_test.json
    """
    result = {"distance_metric": None, "level": None}
    
    pattern = r'rank_(mean_distance|centroid_umap|lof|knn|median_distance|isolation_forest)_(mmd|dtw|wasserstein)_(out_domain|mid_domain|in_domain)'
    match = re.search(pattern, filename)
    
    if match:
        result["distance_metric"] = match.group(2)
        result["level"] = match.group(3)
    
    return result


def collect_metrics_for_method(method: str, jobids: list = None) -> pd.DataFrame:
    """Collect evaluation metrics for a specific ranking method."""
    records = []
    
    if jobids:
        for jobid in jobids:
            pattern = str(EVAL_DIR / jobid / "*" / f"eval_results_*rank_{method}*.json")
            json_files = load_json_glob(pattern, skip_errors=True)
            for path, data in json_files:
                metrics = extract_metrics_from_eval_json(data, filename=path.name)
                metrics["ranking_method"] = method
                metrics["jobid"] = jobid
                
                parsed = parse_filename_for_metadata(path.name)
                if parsed["distance_metric"]:
                    metrics["distance"] = parsed["distance_metric"]
                if parsed["level"]:
                    metrics["level"] = parsed["level"]
                
                records.append(metrics)
    else:
        pattern = str(EVAL_DIR / "*" / "*" / f"eval_results_*rank_{method}*.json")
        json_files = load_json_glob(pattern, skip_errors=True)
        for path, data in json_files:
            metrics = extract_metrics_from_eval_json(data, filename=path.name)
            metrics["ranking_method"] = method
            
            parts = path.parts
            for i, part in enumerate(parts):
                if part == "RF" and i + 1 < len(parts):
                    metrics["jobid"] = parts[i + 1]
                    break
            
            parsed = parse_filename_for_metadata(path.name)
            if parsed["distance_metric"]:
                metrics["distance"] = parsed["distance_metric"]
            if parsed["level"]:
                metrics["level"] = parsed["level"]
            
            records.append(metrics)
    
    return pd.DataFrame(records)


def collect_ranked_metrics():
    """Collect evaluation metrics from ranked domain analysis."""
    print("\n=== Collecting Ranked Metrics ===")
    print(f"Ranking methods: {RANKING_METHODS}")
    print(f"Evaluation directory: {EVAL_DIR}")
    
    all_records = []
    
    for method in RANKING_METHODS:
        print(f"\n[{method}] Searching for evaluation results...")
        df = collect_metrics_for_method(method)
        
        if len(df) > 0:
            print(f"  Found {len(df)} evaluation results")
            all_records.append(df)
        else:
            print(f"  No results found for {method}")
    
    if not all_records:
        print("\n[ERROR] No evaluation results found for any ranking method.")
        return pd.DataFrame()
    
    # Combine all methods
    combined_df = pd.concat(all_records, ignore_index=True)
    print(f"\n[INFO] Total records: {len(combined_df)}")
    
    # Filter to test split
    if "split" in combined_df.columns:
        test_df = combined_df[combined_df["split"] == "test"].copy()
    else:
        test_df = combined_df.copy()
    
    # Ensure categorical level order
    if "level" in test_df.columns:
        cat = pd.CategoricalDtype(categories=["out_domain", "mid_domain", "in_domain"], ordered=True)
        test_df["level"] = test_df["level"].astype(cat)
    
    # Sort
    sort_cols = ["ranking_method", "distance", "level", "mode"]
    sort_cols = [c for c in sort_cols if c in test_df.columns]
    if sort_cols:
        test_df = test_df.sort_values(sort_cols)
    
    # Save
    out_path = OUT_DIR / "summary_ranked_test.csv"
    save_csv(test_df, str(out_path))
    print(f"\n[INFO] Saved: {out_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary by ranking method:")
    print("=" * 60)
    
    if "ranking_method" in test_df.columns:
        for method in test_df["ranking_method"].unique():
            method_df = test_df[test_df["ranking_method"] == method]
            print(f"\n{method}:")
            print(f"  Records: {len(method_df)}")
            if "mode" in method_df.columns:
                print(f"  Modes: {method_df['mode'].unique().tolist()}")
            if "level" in method_df.columns:
                print(f"  Levels: {method_df['level'].unique().tolist()}")
            if "f1" in method_df.columns:
                print(f"  F1 mean: {method_df['f1'].mean():.4f}")
    
    return test_df


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Collect evaluation metrics for domain analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--ranked",
        action="store_true",
        help="Collect ranked domain analysis metrics (multiple ranking methods)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Collect both pooled (40-cases) and ranked metrics"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("COLLECT DOMAIN METRICS")
    print("=" * 60)
    
    if args.all:
        collect_pooled_metrics()
        collect_ranked_metrics()
    elif args.ranked:
        collect_ranked_metrics()
    else:
        collect_pooled_metrics()
    
    print("\n" + "=" * 60)
    print("COLLECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
