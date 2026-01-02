#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_imbalance.py
======================

Unified tool for imbalance experiment visualization.

This script provides:
1. boxplot   - Metric boxplots (replaces visualize_metric_boxplot)
2. tradeoff  - Recall vs Specificity scatter (replaces visualize_recall_vs_specificity)
3. baseline  - Baseline metrics visualization (replaces visualize_baseline_metrics)
4. domain    - Imbalance × domain analysis (replaces visualize_imbalance_v3_domain)
5. compare   - Method comparison (replaces visualize_imbalance_cli)

Consolidates functionality from:
- visualize_metric_boxplot.py
- visualize_recall_vs_specificity.py
- visualize_baseline_metrics.py
- visualize_imbalance_v3_domain.py
- visualize_imbalance_cli.py

Usage:
    python visualize_imbalance.py boxplot --metric f2 --input results/...
    python visualize_imbalance.py tradeoff --input results/...
    python visualize_imbalance.py baseline
    python visualize_imbalance.py domain --input results/...
    python visualize_imbalance.py compare --methods smote,smote_tomek
"""

import argparse
import json
import logging
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
from src.utils.visualization.color_palettes import IMBALANCE_METHOD_COLORS

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Default paths
RESULTS_DIR = Path(cfg.RESULTS_IMBALANCE_PATH)

# Metrics
METRICS = ["f2", "recall", "precision", "f1", "auc_pr", "specificity"]


# ============================================================
# Boxplot Visualization
# ============================================================
def plot_metric_boxplot(df: pd.DataFrame, metric: str, output_path: Path,
                         group_by: str = "method", title: Optional[str] = None) -> None:
    """Create boxplot for a specific metric."""
    if metric not in df.columns:
        logger.warning(f"Metric {metric} not found in data")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    groups = df[group_by].unique() if group_by in df.columns else ["all"]
    
    data_for_plot = [df[df[group_by] == g][metric].dropna().values for g in groups]
    
    bp = ax.boxplot(data_for_plot, labels=groups, patch_artist=True)
    
    for i, (patch, g) in enumerate(zip(bp["boxes"], groups)):
        color = IMBALANCE_METHOD_COLORS.get(g, f"C{i}")
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xticklabels(groups, rotation=45, ha="right")
    ax.set_ylabel(metric.upper())
    ax.set_title(title or f"{metric.upper()} Distribution by {group_by}")
    ax.grid(axis="y", alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def plot_multi_metric_boxplot(df: pd.DataFrame, output_path: Path, group_by: str = "method") -> None:
    """Create multi-panel boxplot for all metrics."""
    metrics = [m for m in METRICS if m in df.columns]
    
    n_cols = 3
    n_rows = (len(metrics) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()
    
    groups = df[group_by].unique() if group_by in df.columns else ["all"]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        data = [df[df[group_by] == g][metric].dropna().values for g in groups]
        bp = ax.boxplot(data, labels=groups, patch_artist=True)
        
        for i, (patch, g) in enumerate(zip(bp["boxes"], groups)):
            color = IMBALANCE_METHOD_COLORS.get(g, f"C{i}")
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title(metric.upper(), fontweight="bold")
        ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    
    for idx in range(len(metrics), len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle("Metric Distributions by Method", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


# ============================================================
# Tradeoff Visualization
# ============================================================
def plot_recall_vs_specificity(df: pd.DataFrame, output_path: Path,
                                 color_by: str = "method") -> None:
    """Create recall vs specificity scatter plot."""
    if "recall" not in df.columns or "specificity" not in df.columns:
        logger.warning("Missing recall or specificity columns")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    groups = df[color_by].unique() if color_by in df.columns else ["all"]
    
    for g in groups:
        subset = df[df[color_by] == g] if color_by in df.columns else df
        color = IMBALANCE_METHOD_COLORS.get(g, None)
        ax.scatter(subset["specificity"], subset["recall"], label=g, alpha=0.7, s=50, c=color)
    
    ax.set_xlabel("Specificity")
    ax.set_ylabel("Recall")
    ax.set_title("Recall vs Specificity Trade-off")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Add diagonal reference
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


# ============================================================
# Method Comparison
# ============================================================
def plot_method_comparison_bar(df: pd.DataFrame, output_path: Path,
                                 metrics: Optional[List[str]] = None) -> None:
    """Create bar chart comparing methods."""
    metrics = metrics or ["f2", "recall", "precision"]
    metrics = [m for m in metrics if m in df.columns]
    
    if not metrics:
        logger.warning("No metrics found for comparison")
        return
    
    methods = df["method"].unique() if "method" in df.columns else ["all"]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(methods))
    width = 0.8 / len(metrics)
    
    for i, metric in enumerate(metrics):
        values = [df[df["method"] == m][metric].mean() for m in methods]
        offset = (i - len(metrics) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=metric.upper(), alpha=0.8)
    
    ax.set_xlabel("Method")
    ax.set_ylabel("Score")
    ax.set_title("Method Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


# ============================================================
# CLI Commands
# ============================================================
def cmd_boxplot(args) -> int:
    """Create boxplot visualization."""
    if not args.input:
        logger.error("Input file required")
        return 1
    
    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df)} records")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.metric == "all":
        plot_multi_metric_boxplot(df, output_dir / "boxplot_all_metrics.png")
    else:
        plot_metric_boxplot(df, args.metric, output_dir / f"boxplot_{args.metric}.png")
    
    return 0


def cmd_tradeoff(args) -> int:
    """Create tradeoff visualization."""
    if not args.input:
        logger.error("Input file required")
        return 1
    
    df = pd.read_csv(args.input)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_recall_vs_specificity(df, output_dir / "recall_vs_specificity.png")
    
    return 0


def cmd_baseline(args) -> int:
    """Create baseline metrics visualization."""
    logger.info("Creating baseline metrics visualization...")
    
    # Import from archive location (preserved for backwards compatibility)
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "scripts/python/archive/consolidated/visualization/imbalance"))
        from visualize_baseline_metrics import main as baseline_main
        return baseline_main()
    except ImportError as e:
        logger.warning(f"Baseline visualization not available: {e}")
        return 0


def cmd_domain(args) -> int:
    """Create imbalance × domain visualization."""
    logger.info("Creating imbalance × domain visualization...")
    
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "scripts/python/archive/consolidated/visualization/imbalance"))
        from visualize_imbalance_v3_domain import main as domain_main
        return domain_main()
    except ImportError as e:
        logger.warning(f"Domain visualization not available: {e}")
        return 0


def cmd_compare(args) -> int:
    """Create method comparison visualization."""
    if not args.input:
        logger.error("Input file required")
        return 1
    
    df = pd.read_csv(args.input)
    
    if args.methods:
        methods = args.methods.split(",")
        df = df[df["method"].isin(methods)]
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_method_comparison_bar(df, output_dir / "method_comparison.png")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Unified imbalance visualization tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Common arguments
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--input", help="Input CSV file")
    common.add_argument("--output-dir", default="results/analysis/imbalance/plots", help="Output directory")
    
    # boxplot
    p_box = subparsers.add_parser("boxplot", parents=[common], help="Metric boxplots")
    p_box.add_argument("--metric", default="all", help="Metric to plot (or 'all')")
    p_box.set_defaults(func=cmd_boxplot)
    
    # tradeoff
    p_trade = subparsers.add_parser("tradeoff", parents=[common], help="Recall vs Specificity scatter")
    p_trade.set_defaults(func=cmd_tradeoff)
    
    # baseline
    p_base = subparsers.add_parser("baseline", help="Baseline metrics visualization")
    p_base.set_defaults(func=cmd_baseline)
    
    # domain
    p_domain = subparsers.add_parser("domain", parents=[common], help="Imbalance × domain analysis")
    p_domain.set_defaults(func=cmd_domain)
    
    # compare
    p_comp = subparsers.add_parser("compare", parents=[common], help="Method comparison")
    p_comp.add_argument("--methods", help="Comma-separated methods to compare")
    p_comp.set_defaults(func=cmd_compare)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
