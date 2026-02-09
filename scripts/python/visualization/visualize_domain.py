#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_domain.py
===================

Unified tool for domain analysis visualization.

This script provides:
1. summary     - Summary metrics bar charts (replaces visualize_summary_metrics_ranked)
2. ranking     - Ranking method comparison (replaces visualize_ranking_comparison)
3. full        - Full comparison dashboard (replaces visualize_full_comparison)
4. projections - MDS/t-SNE/UMAP projection plots (replaces generate_new_ranking_plots)
5. source      - New source domain analysis (replaces visualize_new_source_domain)

Consolidates functionality from:
- visualize_summary_metrics_ranked.py
- visualize_ranking_comparison.py
- visualize_full_comparison.py
- generate_new_ranking_plots.py
- visualize_new_source_domain.py

Usage:
    python visualize_domain.py summary --input results/analysis/exp1_imbalance/figures/csv/summary_ranked_test.csv
    python visualize_domain.py ranking --mode heatmap
    python visualize_domain.py full
    python visualize_domain.py projections --method umap
    python visualize_domain.py source --jobid 14552850
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src import config as cfg
from src.utils.io.data_io import load_csv
from src.utils.visualization.visualization import save_figure, plot_grouped_bar_chart_raw
from src.utils.visualization.color_palettes import (
    RANKING_METHOD_COLORS,
    DOMAIN_LEVEL_COLORS,
    TRAINING_MODE_COLORS,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Paths
SUMMARY_DIR = Path(cfg.RESULTS_DOMAIN_ANALYSIS_PATH) / "summary"
OUTPUT_DIR = SUMMARY_DIR / "png"

# Metrics
METRICS = ["accuracy", "recall", "precision", "f1", "f2", "auc", "auc_pr"]
LEVELS = ["out_domain", "mid_domain", "in_domain"]


# ============================================================
# Summary Visualization
# ============================================================
def plot_summary_metrics(df: pd.DataFrame, output_path: Path, title: str = "Summary Metrics") -> None:
    """Create summary metrics bar chart."""
    if "pos_rate" in df.columns and df["pos_rate"].notna().any():
        baseline_pos_rate = df["pos_rate"].mean()
    else:
        baseline_pos_rate = 0.033
    
    modes = ["pooled", "source_only", "target_only"]
    modes = [m for m in modes if m in df["mode"].unique()] if "mode" in df.columns else modes
    
    fig = plot_grouped_bar_chart_raw(
        data=df,
        metrics=METRICS,
        modes=modes,
        distance_col="distance",
        level_col="level",
        baseline_rates={"auc_pr": baseline_pos_rate}
    )
    
    save_figure(fig, str(output_path), dpi=200)
    logger.info(f"Saved: {output_path}")


def plot_method_heatmap(df: pd.DataFrame, metric: str, output_path: Path) -> None:
    """Create heatmap for ranking method comparison."""
    if "ranking_method" not in df.columns or "level" not in df.columns:
        logger.warning("Missing required columns for heatmap")
        return
    
    plot_df = df[df["mode"] == "source_only"].copy() if "mode" in df.columns else df.copy()
    
    if metric not in plot_df.columns:
        logger.warning(f"Metric {metric} not found")
        return
    
    pivot = plot_df.pivot_table(
        index="ranking_method",
        columns="level",
        values=metric,
        aggfunc="mean"
    )
    
    col_order = [c for c in ["in_domain", "mid_domain", "out_domain"] if c in pivot.columns]
    pivot = pivot[col_order]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="RdYlGn",
        ax=ax, vmin=0, vmax=max(0.5, pivot.max().max()),
        cbar_kws={"label": metric.upper()}
    )
    ax.set_title(f"{metric.upper()} by Ranking Method and Level")
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def plot_multi_metric_comparison(df: pd.DataFrame, output_path: Path, mode: str = "source_only") -> None:
    """Create multi-panel bar chart for all metrics."""
    plot_df = df[df["mode"] == mode].copy() if "mode" in df.columns else df.copy()
    
    metrics = [m for m in METRICS if m in plot_df.columns]
    if not metrics:
        logger.warning("No metrics found")
        return
    
    n_cols = 3
    n_rows = (len(metrics) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()
    
    methods = plot_df["ranking_method"].unique() if "ranking_method" in plot_df.columns else ["unknown"]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        x = np.arange(len(LEVELS))
        width = 0.25
        
        for i, method in enumerate(methods):
            method_df = plot_df[plot_df["ranking_method"] == method] if "ranking_method" in plot_df.columns else plot_df
            values = []
            for level in LEVELS:
                level_df = method_df[method_df["level"] == level] if "level" in method_df.columns else method_df
                if len(level_df) > 0 and metric in level_df.columns:
                    values.append(level_df[metric].mean())
                else:
                    values.append(0)
            
            offset = (i - (len(methods) - 1) / 2) * width
            color = RANKING_METHOD_COLORS.get(method, f"C{i}")
            ax.bar(x + offset, values, width, label=method, color=color, alpha=0.8)
        
        ax.set_title(metric.upper(), fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(LEVELS, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 1.0)
        
        if idx == 0:
            ax.legend(fontsize=8)
    
    for idx in range(len(metrics), len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(f"Metrics Comparison (mode={mode})", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


# ============================================================
# CLI Commands
# ============================================================
def cmd_summary(args) -> int:
    """Create summary metrics visualization."""
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1
    
    df = load_csv(str(input_path))
    logger.info(f"Loaded {len(df)} records")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    output_name = input_path.stem.replace("summary_", "") + "_bar.png"
    plot_summary_metrics(df, OUTPUT_DIR / output_name)
    
    return 0


def cmd_ranking(args) -> int:
    """Create ranking method comparison visualizations."""
    input_path = Path(args.input) if args.input else SUMMARY_DIR / "csv" / "summary_ranked_test.csv"
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1
    
    df = load_csv(str(input_path))
    logger.info(f"Loaded {len(df)} records")
    
    out_dir = OUTPUT_DIR / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if args.mode == "heatmap" or args.mode == "all":
        for metric in ["f1", "recall", "f2"]:
            if metric in df.columns:
                plot_method_heatmap(df, metric, out_dir / f"heatmap_{metric}.png")
    
    if args.mode == "bar" or args.mode == "all":
        plot_multi_metric_comparison(df, out_dir / "multi_metric_bar.png")
    
    return 0


def cmd_full(args) -> int:
    """Create full comparison dashboard."""
    # Delegate to existing visualization
    logger.info("Creating full comparison dashboard...")
    
    # Import and run existing logic
    try:
        from visualization.domain.visualize_full_comparison import main as full_main
        return full_main()
    except ImportError:
        logger.warning("visualize_full_comparison not available, running basic version")
        
        input_path = SUMMARY_DIR / "csv" / "summary_ranked_test.csv"
        if input_path.exists():
            df = load_csv(str(input_path))
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            plot_summary_metrics(df, OUTPUT_DIR / "full_dashboard.png", title="Full Comparison")
        return 0


def cmd_projections(args) -> int:
    """Create projection plots."""
    logger.info(f"Creating {args.method.upper()} projection plots...")
    
    # Delegate to existing or run basic version
    try:
        from visualization.domain.generate_new_ranking_plots import main as proj_main
        return proj_main()
    except ImportError:
        logger.warning("Projection visualization not available")
        return 0


def cmd_source(args) -> int:
    """Create source domain analysis plots."""
    logger.info(f"Creating source domain analysis for job {args.jobid}...")
    
    try:
        from visualization.domain.visualize_new_source_domain import main as source_main
        return source_main()
    except ImportError:
        logger.warning("Source domain visualization not available")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Unified domain visualization tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # summary
    p_summary = subparsers.add_parser("summary", help="Summary metrics bar charts")
    p_summary.add_argument("--input", help="Input CSV file")
    p_summary.set_defaults(func=cmd_summary)
    
    # ranking
    p_ranking = subparsers.add_parser("ranking", help="Ranking method comparison")
    p_ranking.add_argument("--input", help="Input CSV file")
    p_ranking.add_argument("--mode", default="all", choices=["heatmap", "bar", "scatter", "all"])
    p_ranking.set_defaults(func=cmd_ranking)
    
    # full
    p_full = subparsers.add_parser("full", help="Full comparison dashboard")
    p_full.set_defaults(func=cmd_full)
    
    # projections
    p_proj = subparsers.add_parser("projections", help="Projection plots")
    p_proj.add_argument("--method", default="umap", choices=["mds", "tsne", "umap"])
    p_proj.set_defaults(func=cmd_projections)
    
    # source
    p_source = subparsers.add_parser("source", help="Source domain analysis")
    p_source.add_argument("--jobid", default="14552850", help="Job ID")
    p_source.set_defaults(func=cmd_source)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
