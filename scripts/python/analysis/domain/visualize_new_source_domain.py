#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_new_source_domain.py
==============================
Visualize domain analysis results with new source domain (in_domain for source_only).

This script:
1. Reads evaluation JSON files from a specified training job
2. Extracts metrics and generates summary CSV
3. Creates summary_metrics_bar_new_source_domain.png for each ranking method

Usage:
    python scripts/python/domain_analysis/visualize_new_source_domain.py --jobid 14552850
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib as mpl
mpl.use('Agg')  # Non-interactive backend
mpl.set_loglevel("warning")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src import config as cfg
from src.utils.visualization.visualization import save_figure, plot_grouped_bar_chart_raw

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# Metrics to plot
METRICS = ["accuracy", "recall", "precision", "f1", "f2", "auc", "auc_pr"]

# Ranking methods to process
RANKING_METHODS = [
    "mean_distance",
    "centroid_umap", 
    "lof",
    "knn",
    "median_distance",
    "isolation_forest"
]


def extract_metadata_from_filename(filename: str) -> dict:
    """Extract metadata from evaluation JSON filename.
    
    Parameters
    ----------
    filename : str
        Filename like 'eval_results_RF_pooled_rank_mean_distance_mmd_out_domain.json'
    
    Returns
    -------
    dict
        Dictionary with ranking_method, distance_metric, level, mode
    """
    result = {
        "ranking_method": "unknown",
        "distance_metric": "unknown",
        "level": "unknown",
        "mode": "unknown",
    }
    
    # Extract mode from filename
    mode_match = re.search(r'eval_results_RF_(pooled|source_only|target_only)', filename)
    if mode_match:
        result["mode"] = mode_match.group(1)
    
    # New format: rank_{method}_{metric}_{level}
    full_match = re.search(
        r'rank_(mean_distance|centroid_umap|lof|knn|median_distance|isolation_forest)_(mmd|dtw|wasserstein)_(out_domain|mid_domain|in_domain)',
        filename
    )
    if full_match:
        result["ranking_method"] = full_match.group(1)
        result["distance_metric"] = full_match.group(2)
        result["level"] = full_match.group(3)
    
    return result


def load_json_metrics(json_path: Path) -> dict:
    """Load metrics from evaluation JSON file.
    
    Parameters
    ----------
    json_path : Path
        Path to JSON file
    
    Returns
    -------
    dict
        Dictionary with metrics
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract filename metadata
    file_meta = extract_metadata_from_filename(json_path.name)
    
    # Try to get metadata from JSON content first, fallback to filename
    ranking_method = data.get("ranking_method") or file_meta["ranking_method"]
    distance_metric = data.get("distance_metric") or file_meta["distance_metric"]
    level = data.get("level") or file_meta["level"]
    mode = data.get("mode") or file_meta["mode"]
    
    # If still unknown, try to extract from tag
    tag = data.get("tag", "")
    if ranking_method == "unknown" or distance_metric == "unknown":
        tag_match = re.search(
            r'full_(mean_distance|centroid_umap|lof|knn|median_distance|isolation_forest)_(mmd|dtw|wasserstein)_(out_domain|mid_domain|in_domain)',
            tag
        )
        if tag_match:
            ranking_method = tag_match.group(1)
            distance_metric = tag_match.group(2)
            level = tag_match.group(3)
    
    # Extract classification report for positive class
    cr = data.get("classification_report", {}) or {}
    pos_block = None
    for key in ["1", "1.0", "True", "pos", "positive"]:
        if key in cr and isinstance(cr[key], dict):
            pos_block = cr[key]
            break
    
    # Calculate positive rate from support
    pos_rate = None
    if pos_block and "support" in pos_block:
        total_support = cr.get("weighted avg", {}).get("support", 0)
        if total_support > 0:
            pos_rate = pos_block["support"] / total_support
    
    metrics = {
        "file": json_path.name,
        "model": "RF",
        "mode": mode,
        "ranking_method": ranking_method,
        "distance_metric": distance_metric,
        "distance": f"{ranking_method}_{distance_metric}",  # Combined for compatibility
        "level": level,
        "pos_rate": pos_rate or data.get("pos_rate", 0.033),
        "auc": data.get("roc_auc") or data.get("auc"),
        "auc_pr": data.get("auc_pr"),
        "f1": data.get("f1"),
        "f2": data.get("f2") or (pos_block.get("f1-score", 0) if pos_block else None),
        "accuracy": data.get("accuracy"),
        "precision": data.get("precision") or (pos_block.get("precision", 0) if pos_block else None),
        "recall": data.get("recall") or (pos_block.get("recall", 0) if pos_block else None),
        "specificity": data.get("specificity"),
        "mse": data.get("mse"),
        # Threshold-optimized metrics
        "precision_thr": data.get("prec_thr"),
        "recall_thr": data.get("recall_thr"),
        "f1_thr": data.get("f1_thr"),
        "f2_thr": data.get("f2_thr"),
        "specificity_thr": data.get("specificity_thr"),
        "tag": tag,
    }
    
    return metrics


def collect_metrics_from_jobid(jobid: str, eval_dir: Path = None) -> pd.DataFrame:
    """Collect all evaluation metrics from a training job.
    
    Parameters
    ----------
    jobid : str
        Training job ID (e.g., "14552850")
    eval_dir : Path, optional
        Evaluation results directory
    
    Returns
    -------
    pd.DataFrame
        DataFrame with all metrics
    """
    if eval_dir is None:
        eval_dir = Path(cfg.RESULTS_EVALUATION_PATH) / "RF" / jobid
    
    if not eval_dir.exists():
        logger.error(f"Evaluation directory not found: {eval_dir}")
        return pd.DataFrame()
    
    all_metrics = []
    
    # Iterate through array job directories
    for subdir in sorted(eval_dir.iterdir()):
        if not subdir.is_dir():
            continue
        
        # Find JSON files
        for json_file in subdir.glob("eval_results_*.json"):
            try:
                metrics = load_json_metrics(json_file)
                metrics["array_index"] = subdir.name
                all_metrics.append(metrics)
                logger.info(f"  Loaded: {json_file.name} ({metrics['ranking_method']}/{metrics['level']}/{metrics['mode']})")
            except Exception as e:
                logger.warning(f"  Failed to load {json_file}: {e}")
    
    df = pd.DataFrame(all_metrics)
    logger.info(f"Collected {len(df)} evaluation results")
    
    return df


def plot_summary_bar_for_method(
    df: pd.DataFrame,
    ranking_method: str,
    output_path: Path,
    output_suffix: str = "_new_source_domain"
) -> bool:
    """Generate summary_metrics_bar plot for a single ranking method.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with metrics for this method
    ranking_method : str
        Ranking method name
    output_path : Path
        Output directory
    output_suffix : str
        Suffix for output filename
    
    Returns
    -------
    bool
        True if plot was generated successfully
    """
    method_df = df[df["ranking_method"] == ranking_method].copy()
    
    if len(method_df) == 0:
        logger.warning(f"No data for {ranking_method}")
        return False
    
    # Create output directory
    method_dir = output_path / ranking_method
    method_dir.mkdir(parents=True, exist_ok=True)
    
    # Use the same plotting function as the original script
    try:
        fig = plot_grouped_bar_chart_raw(
            data=method_df,
            metrics=METRICS,
            modes=["pooled", "source_only", "target_only"],
            distance_col="distance",
            level_col="level",
            baseline_rates={"auc_pr": method_df["pos_rate"].mean() if "pos_rate" in method_df.columns else 0.033}
        )
        
        if fig:
            out_file = method_dir / f"summary_metrics_bar{output_suffix}.png"
            save_figure(fig, str(out_file), dpi=200)
            logger.info(f"  Saved: {out_file}")
            plt.close(fig)
            return True
    except Exception as e:
        logger.error(f"Failed to generate plot for {ranking_method}: {e}")
    
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Visualize domain analysis results with new source domain"
    )
    parser.add_argument(
        "--jobid",
        type=str,
        required=True,
        help="Training job ID to process"
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="_new_source_domain",
        help="Suffix for output filenames"
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Save collected metrics as CSV"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"[INFO] Visualizing domain analysis results")
    print(f"[INFO] Training Job ID: {args.jobid}")
    print("=" * 60)
    
    # Collect metrics from evaluation JSONs
    print("\n[INFO] Collecting metrics from evaluation JSONs...")
    df = collect_metrics_from_jobid(args.jobid)
    
    if df.empty:
        print("[ERROR] No metrics collected. Exiting.")
        return 1
    
    # Filter out unknown entries
    df_valid = df[
        (df["ranking_method"] != "unknown") & 
        (df["level"] != "unknown")
    ].copy()
    
    print(f"\n[INFO] Valid metrics: {len(df_valid)}/{len(df)}")
    
    if df_valid.empty:
        print("[ERROR] No valid metrics found. Check if JSONs have proper metadata.")
        return 1
    
    # Save CSV if requested
    if args.save_csv:
        csv_dir = Path(cfg.RESULTS_DOMAIN_ANALYSIS_PATH) / "summary" / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_path = csv_dir / f"summary_{args.jobid}{args.output_suffix}.csv"
        df_valid.to_csv(csv_path, index=False)
        print(f"\n[INFO] Saved CSV: {csv_path}")
    
    # Summary statistics
    print("\n" + "-" * 40)
    print("[INFO] Summary Statistics")
    print("-" * 40)
    
    # By ranking method
    if "ranking_method" in df_valid.columns and "f1" in df_valid.columns:
        print("\nBy Ranking Method (mean F1):")
        method_stats = df_valid.groupby("ranking_method")["f1"].mean().sort_values(ascending=False)
        for method, f1 in method_stats.items():
            print(f"  {method}: {f1:.4f}")
    
    # By mode
    if "mode" in df_valid.columns and "f1" in df_valid.columns:
        print("\nBy Training Mode (mean F1):")
        mode_stats = df_valid.groupby("mode")["f1"].mean().sort_values(ascending=False)
        for mode, f1 in mode_stats.items():
            print(f"  {mode}: {f1:.4f}")
    
    # Generate plots for each ranking method
    print("\n" + "-" * 40)
    print("[INFO] Generating per-method plots...")
    print("-" * 40)
    
    output_dir = Path(cfg.RESULTS_DOMAIN_ANALYSIS_PATH) / "summary" / "png"
    
    methods_found = df_valid["ranking_method"].unique()
    success_count = 0
    
    for method in RANKING_METHODS:
        if method not in methods_found:
            print(f"\n[WARN] Skipping {method} (no data)")
            continue
        
        print(f"\n[INFO] Processing: {method}")
        if plot_summary_bar_for_method(df_valid, method, output_dir, args.output_suffix):
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"[DONE] Generated {success_count}/{len(RANKING_METHODS)} plots")
    print(f"  Output directory: {output_dir}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
