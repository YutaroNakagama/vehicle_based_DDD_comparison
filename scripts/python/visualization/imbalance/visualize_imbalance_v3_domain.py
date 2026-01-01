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
    python scripts/python/visualization/imbalance/visualize_imbalv3_domain.py
    
    # Specify CSV file explicitly
    python scripts/python/visualization/imbalance/visualize_imbalv3_domain.py \\
        --csv results/imbalance_analysis/domain_v3/all_metrics.csv
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Setup matplotlib before importing pyplot
from src.utils.visualization.setup import setup_matplotlib_headless
setup_matplotlib_headless()

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
    
    Uses plot_grouped_bar_chart_raw for consistent 4-row × 6-column format.
    
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
    method_df = df[
        (df["ranking_method"] == ranking_method) &
        (df["imbalance_method"] == imbalance_method) &
        (df["ratio"] == ratio)
    ].copy()
    
    # Get pooled data for this imbalance method and ratio
    pooled_for_imbalance = df_pooled[
        (df_pooled["imbalance_method"] == imbalance_method) &
        (df_pooled["ratio"] == ratio)
    ].copy()
    
    if method_df.empty:
        logger.warning(f"No data for {ranking_method}/{imbalance_method}/{ratio}")
        return False
    
    # Rename distance_metric -> distance for plot_grouped_bar_chart_raw compatibility
    if "distance_metric" in method_df.columns:
        method_df = method_df.rename(columns={"distance_metric": "distance"})
    
    # Add pooled data (if available)
    if len(pooled_for_imbalance) > 0:
        # For pooled, set distance column to match the current ranking method pattern
        pooled_for_imbalance = pooled_for_imbalance.copy()
        pooled_for_imbalance["distance"] = f"{ranking_method}_pooled"
        pooled_for_imbalance["level"] = "pooled"
        method_df = pd.concat([method_df, pooled_for_imbalance], ignore_index=True)
        logger.info(f"    Added {len(pooled_for_imbalance)} pooled record(s)")
    
    # Create output directory
    method_dir = output_path / ranking_method
    method_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output filename with ratio
    ratio_str = str(ratio).replace(".", "_")
    out_file = method_dir / f"summary_metrics_bar_{imbalance_method}_ratio{ratio_str}.png"
    
    try:
        # Determine baseline pos_rate for auc_pr reference line
        baseline_rate = method_df["pos_rate"].mean() if "pos_rate" in method_df.columns else 0.033
        
        fig = plot_grouped_bar_chart_raw(
            data=method_df,
            metrics=METRICS,
            modes=["pooled", "source_only", "target_only"],
            distance_col="distance",
            level_col="level",
            baseline_rates={"auc_pr": baseline_rate}
        )
        
        if fig:
            save_figure(fig, str(out_file), dpi=200)
            logger.info(f"  Saved: {out_file}")
            plt.close(fig)
            return True
        else:
            logger.warning(f"  Failed to create figure for {ranking_method}/{imbalance_method}/{ratio}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to generate plot for {ranking_method}/{imbalance_method}/{ratio}: {e}")
        import traceback
        traceback.print_exc()
        return False


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
