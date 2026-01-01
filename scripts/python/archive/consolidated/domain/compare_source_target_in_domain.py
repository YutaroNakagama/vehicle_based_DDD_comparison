#!/usr/bin/env python3
"""
Analyze differences between source_only and target_only in in_domain experiments.

This script:
1. Collects all in_domain evaluation results for source_only and target_only
2. Compares metrics between the two modes
3. Identifies patterns and potential causes of differences
"""

import os
import json
import glob
import re
from pathlib import Path
import pandas as pd
import numpy as np

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[4]
EVAL_DIR = PROJECT_ROOT / "results" / "evaluation" / "RF"

def extract_mode_and_details(filepath: str) -> dict:
    """Extract mode, ranking method, distance metric from filename."""
    basename = os.path.basename(filepath)
    
    result = {
        "filepath": filepath,
        "filename": basename,
        "mode": None,
        "ranking_method": None,
        "distance_metric": None,
        "level": None,
        "imbalance_method": None,
    }
    
    # Extract mode
    if "source_only" in basename:
        result["mode"] = "source_only"
    elif "target_only" in basename:
        result["mode"] = "target_only"
    elif "pooled" in basename:
        result["mode"] = "pooled"
    
    # Extract level
    for level in ["in_domain", "mid_domain", "out_domain"]:
        if level in basename:
            result["level"] = level
            break
    
    # Extract distance metric
    for metric in ["mmd", "dtw", "wasserstein"]:
        if metric in basename:
            result["distance_metric"] = metric
            break
    
    # Extract ranking method
    for method in ["knn", "lof", "mean_distance", "median_distance", "centroid_umap", "isolation_forest"]:
        if method in basename:
            result["ranking_method"] = method
            break
    
    # Extract imbalance method
    if "baseline" in basename:
        result["imbalance_method"] = "baseline"
    elif "smote_tomek" in basename or "smote_mean_tomek" in basename:
        result["imbalance_method"] = "smote_tomek"
    elif "smote_rus" in basename or "smote_mean_rus" in basename:
        result["imbalance_method"] = "smote_rus"
    elif "undersample_tomek" in basename or "undersample_mean_tomek" in basename:
        result["imbalance_method"] = "undersample_tomek"
    elif "undersample_rus" in basename or "undersample_mean_rus" in basename:
        result["imbalance_method"] = "undersample_rus"
    elif "smote" in basename:
        result["imbalance_method"] = "smote"
    else:
        result["imbalance_method"] = "unknown"
    
    return result


def load_metrics(filepath: str) -> dict:
    """Load metrics from JSON file."""
    try:
        with open(filepath) as f:
            data = json.load(f)
        return {
            "accuracy": data.get("accuracy"),
            "precision": data.get("precision"),
            "recall": data.get("recall"),
            "f1": data.get("f1"),
            "auc": data.get("auc"),
            "auc_pr": data.get("auc_pr"),
        }
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return {}


def collect_in_domain_results():
    """Collect all in_domain results."""
    pattern = str(EVAL_DIR / "**" / "*in_domain*.json")
    files = glob.glob(pattern, recursive=True)
    
    results = []
    for fp in files:
        info = extract_mode_and_details(fp)
        if info["level"] == "in_domain" and info["mode"] in ["source_only", "target_only"]:
            metrics = load_metrics(fp)
            info.update(metrics)
            results.append(info)
    
    return pd.DataFrame(results)


def compare_source_target(df: pd.DataFrame) -> pd.DataFrame:
    """Compare source_only vs target_only for matching experiments."""
    # Group by ranking method, distance metric, and imbalance method
    grouping_cols = ["ranking_method", "distance_metric", "imbalance_method"]
    
    comparison_rows = []
    
    for group_key, group_df in df.groupby(grouping_cols):
        source_df = group_df[group_df["mode"] == "source_only"]
        target_df = group_df[group_df["mode"] == "target_only"]
        
        if len(source_df) == 0 or len(target_df) == 0:
            continue
        
        # Take first row if multiple exist
        source_row = source_df.iloc[0]
        target_row = target_df.iloc[0]
        
        comparison = {
            "ranking_method": group_key[0],
            "distance_metric": group_key[1],
            "imbalance_method": group_key[2],
        }
        
        for metric in ["accuracy", "precision", "recall", "f1", "auc", "auc_pr"]:
            src_val = source_row.get(metric)
            tgt_val = target_row.get(metric)
            
            if src_val is not None and tgt_val is not None:
                comparison[f"source_{metric}"] = src_val
                comparison[f"target_{metric}"] = tgt_val
                comparison[f"diff_{metric}"] = src_val - tgt_val
            else:
                comparison[f"source_{metric}"] = src_val
                comparison[f"target_{metric}"] = tgt_val
                comparison[f"diff_{metric}"] = None
        
        comparison_rows.append(comparison)
    
    return pd.DataFrame(comparison_rows)


def main():
    print("=" * 80)
    print("In-Domain: Source-Only vs Target-Only Comparison")
    print("=" * 80)
    
    # Collect results
    df = collect_in_domain_results()
    print(f"\nFound {len(df)} in_domain evaluation files")
    print(f"  - source_only: {len(df[df['mode'] == 'source_only'])}")
    print(f"  - target_only: {len(df[df['mode'] == 'target_only'])}")
    
    if len(df) == 0:
        print("No results found!")
        return
    
    # Show all results
    print("\n" + "=" * 80)
    print("All In-Domain Results")
    print("=" * 80)
    
    display_cols = ["mode", "ranking_method", "distance_metric", "imbalance_method", 
                    "accuracy", "precision", "recall", "f1"]
    print(df[display_cols].to_string())
    
    # Compare source vs target
    print("\n" + "=" * 80)
    print("Source-Only vs Target-Only Comparison (Matching Experiments)")
    print("=" * 80)
    
    comparison_df = compare_source_target(df)
    if len(comparison_df) > 0:
        print(f"\nFound {len(comparison_df)} matched experiment pairs")
        
        # Show differences
        diff_cols = ["ranking_method", "distance_metric", "imbalance_method",
                     "diff_accuracy", "diff_precision", "diff_recall", "diff_f1"]
        print("\nDifferences (source - target):")
        print(comparison_df[diff_cols].to_string())
        
        # Statistics
        print("\n" + "-" * 60)
        print("Difference Statistics:")
        for metric in ["accuracy", "precision", "recall", "f1"]:
            diff_col = f"diff_{metric}"
            if diff_col in comparison_df.columns:
                diffs = comparison_df[diff_col].dropna()
                if len(diffs) > 0:
                    print(f"  {metric:12}: mean={diffs.mean():.4f}, std={diffs.std():.4f}, "
                          f"min={diffs.min():.4f}, max={diffs.max():.4f}")
        
        # Save comparison
        out_dir = PROJECT_ROOT / "results" / "domain_analysis" / "summary"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        out_path = out_dir / "in_domain_source_vs_target_comparison.csv"
        comparison_df.to_csv(out_path, index=False)
        print(f"\nSaved comparison to: {out_path}")
    else:
        print("No matching experiment pairs found for comparison")


if __name__ == "__main__":
    main()
