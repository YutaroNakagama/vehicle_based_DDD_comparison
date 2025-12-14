#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_imbalv3_domain.py
===========================
Visualize imbalv3 (imbalance v3) experiments for domain analysis with ratio support.

This script:
1. Reads the aggregated metrics CSV from imbalv3 experiments
2. Generates summary_metrics_bar plots for each ranking method × imbalance method × ratio
3. Output: results/domain_analysis/summary/png/{ranking_method}/summary_metrics_bar_{imbalance_method}_{ratio}.png

Usage:
    python scripts/python/analysis/imbalance/visualize_imbalv3_domain.py
    
    # Specify CSV file explicitly
    python scripts/python/analysis/imbalance/visualize_imbalv3_domain.py \\
        --csv results/imbalance_analysis/domain_v3/all_metrics.csv
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')  # Non-interactive backend
mpl.set_loglevel("warning")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src import config as cfg
from src.utils.visualization.visualization import save_figure, plot_grouped_bar_chart_raw

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# Metrics to plot
METRICS = ["accuracy", "recall", "precision", "f1", "f2", "auc_pr"]

# Ranking methods to process
RANKING_METHODS = ["knn", "lof", "median_distance"]

# Distance metrics and levels
DISTANCE_METRICS = ["mmd", "wasserstein", "dtw"]
LEVELS = ["in_domain", "mid_domain", "out_domain"]

# Imbalance methods (will be detected from data)
KNOWN_IMBALANCE_METHODS = [
    "baseline", "smote", "smote_tomek", "smote_rus", 
    "smote_balanced_rf", "undersample_rus", "undersample_enn", "undersample_tomek"
]

# Ratios
RATIOS = ["0.1", "0.5", "1.0"]


def load_metrics_csv(csv_path: Path) -> pd.DataFrame:
    """Load aggregated metrics CSV.
    
    Parameters
    ----------
    csv_path : Path
        Path to CSV file
    
    Returns
    -------
    pd.DataFrame
        DataFrame with metrics
    """
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows from {csv_path}")
    
    return df


def plot_summary_bar_for_imbalance_ratio(
    df: pd.DataFrame,
    df_pooled: pd.DataFrame,
    ranking_method: str,
    imbalance_method: str,
    ratio: str,
    output_path: Path
) -> bool:
    """Generate summary_metrics_bar plot for one ranking × imbalance × ratio combination.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with domain-specific metrics (knn, lof, median_distance)
    df_pooled : pd.DataFrame
        DataFrame with pooled metrics
    ranking_method : str
        Ranking method (knn, lof, median_distance)
    imbalance_method : str
        Imbalance handling method
    ratio : str
        Target ratio (0.1, 0.5, 1.0)
    output_path : Path
        Directory to save plots
        
    Returns
    -------
    bool
        True if successful
    """
    # Filter data for this ranking method, imbalance method, and ratio
    df_rank = df[
        (df["ranking_method"] == ranking_method) &
        (df["imbalance_method"] == imbalance_method) &
        (df["ratio"] == ratio)
    ].copy()
    
    # Get pooled data for this imbalance method and ratio
    df_pool = df_pooled[
        (df_pooled["imbalance_method"] == imbalance_method) &
        (df_pooled["ratio"] == ratio)
    ].copy()
    
    if df_rank.empty:
        logger.warning(f"No data for {ranking_method}/{imbalance_method}/{ratio}")
        return False
    
    # Create output filename with ratio
    ratio_str = str(ratio).replace(".", "_")
    out_file = output_path / ranking_method / f"summary_metrics_bar_{imbalance_method}_ratio{ratio_str}.png"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for grouped bar chart
    # X-axis: distance_metric × level (9 combinations)
    # Groups: source_only vs target_only
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    metrics_to_plot = ["recall", "precision", "f1", "f2", "auc_pr", "accuracy"]
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        # Prepare data matrix
        x_labels = []
        source_values = []
        target_values = []
        
        for dist_metric in DISTANCE_METRICS:
            for level in LEVELS:
                x_labels.append(f"{dist_metric}\n{level.replace('_domain', '')}")
                
                # Get source_only value
                src_row = df_rank[
                    (df_rank["distance_metric"] == dist_metric) &
                    (df_rank["level"] == level) &
                    (df_rank["mode"] == "source_only")
                ]
                source_values.append(src_row[metric].values[0] if len(src_row) > 0 else 0)
                
                # Get target_only value
                tgt_row = df_rank[
                    (df_rank["distance_metric"] == dist_metric) &
                    (df_rank["level"] == level) &
                    (df_rank["mode"] == "target_only")
                ]
                target_values.append(tgt_row[metric].values[0] if len(tgt_row) > 0 else 0)
        
        # Add pooled if available
        if not df_pool.empty:
            x_labels.append("Pooled")
            pool_row = df_pool[df_pool["mode"] == "pooled"]
            if not pool_row.empty:
                pool_val = pool_row[metric].values[0]
                source_values.append(pool_val)
                target_values.append(pool_val)
            else:
                source_values.append(0)
                target_values.append(0)
        
        # Plot grouped bars
        x = np.arange(len(x_labels))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, source_values, width, label='Source Only', color='#3498db', edgecolor='black')
        bars2 = ax.bar(x + width/2, target_values, width, label='Target Only', color='#e74c3c', edgecolor='black')
        
        ax.set_xlabel('Distance Metric / Level')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()}')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        
        # Set y-axis limits based on metric
        if metric in ["accuracy", "recall"]:
            ax.set_ylim(0, 1.0)
        elif metric in ["precision", "f1", "f2", "auc_pr"]:
            ax.set_ylim(0, max(max(source_values + target_values) * 1.2, 0.1))
    
    # Title
    ratio_display = ratio if ratio != "none" else "default"
    fig.suptitle(
        f"Domain Analysis: {ranking_method.upper()} | {imbalance_method.upper()} | Ratio={ratio_display}",
        fontsize=14, fontweight='bold', y=1.02
    )
    
    plt.tight_layout()
    
    # Save
    fig.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Saved: {out_file}")
    return True


def print_summary_stats(df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    
    print(f"\nTotal records: {len(df)}")
    
    print("\nImbalance Methods:")
    for method, count in df["imbalance_method"].value_counts().items():
        print(f"  {method}: {count}")
    
    print("\nRatios:")
    for ratio, count in df["ratio"].value_counts().items():
        print(f"  {ratio}: {count}")
    
    print("\nRanking Methods:")
    for method, count in df["ranking_method"].value_counts().items():
        print(f"  {method}: {count}")
    
    print("\nModes:")
    for mode, count in df["mode"].value_counts().items():
        print(f"  {mode}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize imbalv3 experiments for domain analysis"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to aggregated metrics CSV. Default: results/imbalance_analysis/domain_v3/all_metrics.csv"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory. Default: results/domain_analysis/summary/png"
    )
    args = parser.parse_args()
    
    # Paths
    csv_path = Path(args.csv) if args.csv else Path(cfg.RESULTS_PATH) / "imbalance_analysis" / "domain_v3" / "all_metrics.csv"
    output_dir = Path(args.output) if args.output else Path(cfg.RESULTS_PATH) / "domain_analysis" / "summary" / "png"
    
    print("=" * 60)
    print("[INFO] Visualizing imbalv3 Domain Experiments")
    print("=" * 60)
    print(f"CSV: {csv_path}")
    print(f"Output: {output_dir}")
    
    # Load data
    df = load_metrics_csv(csv_path)
    
    if df.empty:
        print("[ERROR] No data loaded.")
        return 1
    
    # Filter for valid domain data
    df_valid = df[
        (df["ranking_method"].isin(RANKING_METHODS)) &
        (df["imbalance_method"] != "unknown")
    ].copy()
    
    # Get pooled data separately
    df_pooled = df[
        (df["ranking_method"] == "pooled") &
        (df["imbalance_method"] != "unknown")
    ].copy()
    
    print(f"\n[INFO] Valid domain records: {len(df_valid)}/{len(df)}")
    print(f"[INFO] Pooled records: {len(df_pooled)}")
    
    if df_valid.empty:
        print("[ERROR] No valid metrics found.")
        return 1
    
    # Print summary
    print_summary_stats(df_valid)
    
    # Get unique combinations
    imbalance_methods = df_valid["imbalance_method"].unique()
    ratios = df_valid["ratio"].unique()
    
    print(f"\n[INFO] Found imbalance methods: {list(imbalance_methods)}")
    print(f"[INFO] Found ratios: {list(ratios)}")
    
    # Generate plots
    print("\n" + "=" * 60)
    print("[INFO] Generating plots...")
    print("=" * 60)
    
    success_count = 0
    total_expected = 0
    
    for ranking in RANKING_METHODS:
        print(f"\n[INFO] Ranking method: {ranking}")
        
        for imbalance in imbalance_methods:
            for ratio in ratios:
                total_expected += 1
                result = plot_summary_bar_for_imbalance_ratio(
                    df_valid, df_pooled, ranking, imbalance, ratio, output_dir
                )
                if result:
                    success_count += 1
                else:
                    print(f"  [WARN] No data: {ranking}/{imbalance}/ratio={ratio}")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"[DONE] Generated {success_count}/{total_expected} plots")
    print(f"  Output directory: {output_dir}")
    print("=" * 60)
    
    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
