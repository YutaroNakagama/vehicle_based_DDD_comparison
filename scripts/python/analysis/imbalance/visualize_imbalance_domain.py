#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_imbalance_domain.py
=============================
Visualize imbalance handling experiments for domain analysis.

This script:
1. Reads the aggregated metrics CSV from imbalance experiments
2. Generates 12 summary_metrics_bar plots: 3 ranking methods × 4 imbalance methods
3. Output: results/domain_analysis/summary/png/{ranking_method}/summary_metrics_bar_{imbalance_method}.png

Usage:
    python scripts/python/analysis/imbalance/visualize_imbalance_domain.py
    
    # Specify CSV file explicitly
    python scripts/python/analysis/imbalance/visualize_imbalance_domain.py \\
        --csv results/imbalance_analysis/domain/all_metrics.csv
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

# Ranking methods to process (only those used in imbalance experiments)
RANKING_METHODS = ["knn", "lof", "median_distance"]

# Imbalance methods
IMBALANCE_METHODS = ["baseline", "smote", "smote_tomek", "smote_rus"]


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
    
    # Create combined distance column for compatibility with plotting function
    if "distance" not in df.columns:
        df["distance"] = df["ranking_method"] + "_" + df["distance_metric"]
    
    return df


def plot_summary_bar_for_imbalance(
    df: pd.DataFrame,
    df_pooled: pd.DataFrame,
    ranking_method: str,
    imbalance_method: str,
    output_path: Path
) -> bool:
    """Generate summary_metrics_bar plot for one ranking × imbalance combination.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with domain-specific metrics (knn, lof, median_distance)
    df_pooled : pd.DataFrame
        DataFrame with pooled metrics
    ranking_method : str
        Ranking method (knn, lof, median_distance)
    imbalance_method : str
        Imbalance method (baseline, smote, smote_tomek, smote_rus)
    output_path : Path
        Base output directory
    
    Returns
    -------
    bool
        True if plot was generated successfully
    """
    # Filter data for this ranking method
    method_df = df[
        (df["ranking_method"] == ranking_method) &
        (df["imbalance_method"] == imbalance_method)
    ].copy()
    
    # Add pooled data for this imbalance method
    pooled_for_imbalance = df_pooled[
        df_pooled["imbalance_method"] == imbalance_method
    ].copy()
    
    if len(pooled_for_imbalance) > 0:
        # For pooled, set distance column to match the current ranking method pattern
        # This allows it to appear in the plot alongside source_only/target_only
        pooled_for_imbalance["distance"] = f"{ranking_method}_pooled"
        pooled_for_imbalance["level"] = "pooled"
        method_df = pd.concat([method_df, pooled_for_imbalance], ignore_index=True)
        logger.info(f"    Added {len(pooled_for_imbalance)} pooled record(s)")
    
    if len(method_df) == 0:
        logger.warning(f"No data for {ranking_method}/{imbalance_method}")
        return False
    
    # Create output directory
    method_dir = output_path / ranking_method
    method_dir.mkdir(parents=True, exist_ok=True)
    
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
            out_file = method_dir / f"summary_metrics_bar_{imbalance_method}.png"
            save_figure(fig, str(out_file), dpi=200)
            logger.info(f"  Saved: {out_file}")
            plt.close(fig)
            return True
        else:
            logger.warning(f"  Failed to create figure for {ranking_method}/{imbalance_method}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to generate plot for {ranking_method}/{imbalance_method}: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_summary_stats(df: pd.DataFrame):
    """Print summary statistics of the data."""
    print("\n" + "-" * 50)
    print("[INFO] Data Summary")
    print("-" * 50)
    
    # Count by imbalance method
    print("\nRecords by Imbalance Method:")
    for method in IMBALANCE_METHODS:
        count = len(df[df["imbalance_method"] == method])
        print(f"  {method:15s}: {count:3d}")
    
    # Count by ranking method  
    print("\nRecords by Ranking Method:")
    for method in RANKING_METHODS:
        count = len(df[df["ranking_method"] == method])
        print(f"  {method:15s}: {count:3d}")
    
    # Count by mode
    print("\nRecords by Training Mode:")
    for mode in ["pooled", "source_only", "target_only"]:
        count = len(df[df["mode"] == mode])
        print(f"  {mode:15s}: {count:3d}")
    
    # Mean F1 by imbalance method
    if "f1" in df.columns:
        print("\nMean F1 by Imbalance Method:")
        stats = df.groupby("imbalance_method")["f1"].agg(["mean", "std"])
        for method in IMBALANCE_METHODS:
            if method in stats.index:
                mean_f1 = stats.loc[method, "mean"]
                std_f1 = stats.loc[method, "std"]
                print(f"  {method:15s}: {mean_f1:.4f} ± {std_f1:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize imbalance handling experiments for domain analysis"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to aggregated metrics CSV. Default: results/imbalance_analysis/domain/all_metrics.csv"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots. Default: results/domain_analysis/summary/png"
    )
    
    args = parser.parse_args()
    
    # Set default paths
    if args.csv:
        csv_path = Path(args.csv)
    else:
        csv_path = Path(cfg.RESULTS_PATH) / "imbalance_analysis" / "domain" / "all_metrics.csv"
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(cfg.RESULTS_DOMAIN_ANALYSIS_PATH) / "summary" / "png"
    
    print("=" * 60)
    print("[INFO] Visualizing Imbalance Domain Analysis")
    print("=" * 60)
    print(f"  CSV file: {csv_path}")
    print(f"  Output dir: {output_dir}")
    
    # Load data
    print("\n[INFO] Loading metrics from CSV...")
    df = load_metrics_csv(csv_path)
    
    if df.empty:
        print("[ERROR] No data loaded. Exiting.")
        return 1
    
    # Filter valid entries for domain-specific methods (exclude pooled for now)
    df_valid = df[
        (df["ranking_method"].isin(RANKING_METHODS)) &
        (df["imbalance_method"].isin(IMBALANCE_METHODS))
    ].copy()
    
    # Get pooled data separately (will be added to each plot)
    df_pooled = df[
        (df["ranking_method"] == "pooled") &
        (df["imbalance_method"].isin(IMBALANCE_METHODS))
    ].copy()
    
    print(f"\n[INFO] Valid domain records: {len(df_valid)}/{len(df)}")
    print(f"[INFO] Pooled records: {len(df_pooled)}")
    
    if df_valid.empty:
        print("[ERROR] No valid metrics found.")
        return 1
    
    # Print summary
    print_summary_stats(df_valid)
    
    # Generate 12 plots: 3 ranking methods × 4 imbalance methods
    print("\n" + "=" * 60)
    print("[INFO] Generating plots...")
    print("=" * 60)
    
    success_count = 0
    total_expected = len(RANKING_METHODS) * len(IMBALANCE_METHODS)
    
    for ranking in RANKING_METHODS:
        print(f"\n[INFO] Ranking method: {ranking}")
        
        for imbalance in IMBALANCE_METHODS:
            result = plot_summary_bar_for_imbalance(df_valid, df_pooled, ranking, imbalance, output_dir)
            if result:
                success_count += 1
            else:
                print(f"  [WARN] Failed: {ranking}/{imbalance}")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"[DONE] Generated {success_count}/{total_expected} plots")
    print(f"  Output directory: {output_dir}")
    print("=" * 60)
    
    return 0 if success_count == total_expected else 1


if __name__ == "__main__":
    sys.exit(main())
