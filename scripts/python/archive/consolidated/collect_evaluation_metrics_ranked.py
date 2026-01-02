#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_evaluation_metrics_ranked.py
====================================
Collect evaluation metrics from multiple ranking methods and combine into summary CSV.

This script supports the following ranking methods:
- mean_distance: baseline (average distance to all subjects)
- centroid_umap: UMAP centroid-based (cluster-aware)
- lof: Local Outlier Factor (density-based)
- knn: K-Nearest Neighbors (average distance to k closest subjects)
- median_distance: Median distance (robust to outliers)
- isolation_forest: Isolation Forest anomaly score (tree-based)

Output:
- summary_ranked_test.csv: Combined metrics for all ranking methods
"""

import os
import re
import pandas as pd
from pathlib import Path
from glob import glob

from src import config as cfg
from src.utils.io.data_io import load_json_glob, save_csv
from src.evaluation.metrics import extract_metrics_from_eval_json

EVAL_DIR = Path(cfg.RESULTS_EVALUATION_PATH) / "RF"
OUT_DIR = Path(cfg.RESULTS_DOMAIN_ANALYSIS_PATH) / "summary" / "csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Ranking methods to collect
RANKING_METHODS = [
    "mean_distance",
    "centroid_umap",
    "lof",
    "knn",
    "median_distance",
    "isolation_forest",
]


def parse_filename_for_metadata(filename: str) -> dict:
    """Parse filename to extract distance metric and level.
    
    Format: eval_results_RF_{mode}_rank_{method}_{distance_metric}_{level}_mean_test.json
    Example: eval_results_RF_source_only_rank_knn_mmd_out_domain_mean_test.json
    
    Returns
    -------
    dict
        Contains 'distance_metric' (mmd/dtw/wasserstein) and 'level' (out_domain/mid_domain/in_domain)
    """
    result = {"distance_metric": None, "level": None}
    
    # Pattern: rank_{method}_{distance}_{level}
    # Methods: mean_distance, centroid_umap, lof, knn, median_distance, isolation_forest
    # Distance: mmd, dtw, wasserstein
    # Level: out_domain, mid_domain, in_domain
    pattern = r'rank_(mean_distance|centroid_umap|lof|knn|median_distance|isolation_forest)_(mmd|dtw|wasserstein)_(out_domain|mid_domain|in_domain)'
    match = re.search(pattern, filename)
    
    if match:
        result["distance_metric"] = match.group(2)  # mmd, dtw, wasserstein
        result["level"] = match.group(3)  # out_domain, mid_domain, in_domain
    
    return result


def collect_metrics_for_method(method: str, jobids: list = None) -> pd.DataFrame:
    """Collect evaluation metrics for a specific ranking method.
    
    Parameters
    ----------
    method : str
        Ranking method name.
    jobids : list, optional
        List of job IDs to search. If None, search all.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with extracted metrics.
    """
    records = []
    
    # Search pattern
    if jobids:
        for jobid in jobids:
            pattern = str(EVAL_DIR / jobid / "*" / f"eval_results_*rank_{method}*.json")
            json_files = load_json_glob(pattern, skip_errors=True)
            for path, data in json_files:
                metrics = extract_metrics_from_eval_json(data, filename=path.name)
                metrics["ranking_method"] = method
                metrics["jobid"] = jobid
                
                # Override distance/level with correctly parsed values
                parsed = parse_filename_for_metadata(path.name)
                if parsed["distance_metric"]:
                    metrics["distance"] = parsed["distance_metric"]
                if parsed["level"]:
                    metrics["level"] = parsed["level"]
                
                records.append(metrics)
    else:
        # Search all directories
        pattern = str(EVAL_DIR / "*" / "*" / f"eval_results_*rank_{method}*.json")
        json_files = load_json_glob(pattern, skip_errors=True)
        for path, data in json_files:
            metrics = extract_metrics_from_eval_json(data, filename=path.name)
            metrics["ranking_method"] = method
            
            # Extract jobid from path
            parts = path.parts
            for i, part in enumerate(parts):
                if part == "RF" and i + 1 < len(parts):
                    metrics["jobid"] = parts[i + 1]
                    break
            
            # Override distance/level with correctly parsed values
            parsed = parse_filename_for_metadata(path.name)
            if parsed["distance_metric"]:
                metrics["distance"] = parsed["distance_metric"]
            if parsed["level"]:
                metrics["level"] = parsed["level"]
            
            records.append(metrics)
    
    return pd.DataFrame(records)


def main():
    print("=" * 60)
    print("Collecting evaluation metrics for ranked domain analysis")
    print("=" * 60)
    print(f"Ranking methods: {RANKING_METHODS}")
    print(f"Evaluation directory: {EVAL_DIR}")
    print("")
    
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
        print("Please run evaluation jobs first.")
        return
    
    # Combine all methods
    combined_df = pd.concat(all_records, ignore_index=True)
    print(f"\n[INFO] Total records: {len(combined_df)}")
    
    # Filter to test split only
    if "split" in combined_df.columns:
        test_df = combined_df[combined_df["split"] == "test"].copy()
    else:
        test_df = combined_df.copy()
    
    # Ensure categorical level order
    if "level" in test_df.columns:
        cat = pd.CategoricalDtype(categories=["out_domain", "mid_domain", "in_domain"], ordered=True)
        test_df["level"] = test_df["level"].astype(cat)
    
    # Sort by ranking_method, distance, level, mode
    sort_cols = ["ranking_method", "distance", "level", "mode"]
    sort_cols = [c for c in sort_cols if c in test_df.columns]
    if sort_cols:
        test_df = test_df.sort_values(sort_cols)
    
    # Save combined CSV
    out_path = OUT_DIR / "summary_ranked_test.csv"
    save_csv(test_df, str(out_path))
    print(f"\n[INFO] Saved: {out_path}")
    
    # Print summary statistics
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


if __name__ == "__main__":
    main()
