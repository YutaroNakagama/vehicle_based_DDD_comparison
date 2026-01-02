#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_sampling.py - CLI Wrapper
====================================

Thin CLI wrapper for sampling distribution visualization.
All business logic is delegated to src/analysis/sampling.py

Subcommands:
    from-logs   - Extract and visualize sampling distribution from job logs
    theoretical - Calculate and visualize theoretical sampling distribution
    compare     - Compare actual vs theoretical distributions

Usage:
    python visualize_sampling.py from-logs --log-dir scripts/hpc/logs --prefix 14618
    python visualize_sampling.py theoretical --methods smote,smote_tomek --ratio 0.5
    python visualize_sampling.py compare --log-dir scripts/hpc/logs
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.sampling import (
    extract_sampling_distribution,
    calculate_sampling_distribution,
    calculate_batch_distributions,
    compare_actual_vs_theoretical,
    DEFAULT_TRAIN_ALERT,
    DEFAULT_TRAIN_DROWSY,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# Visualization Functions
# ============================================================
def plot_sampling_distribution(
    df: pd.DataFrame,
    output_path: Path,
    title: str = "Sampling Distribution"
) -> None:
    """Plot sampling distribution bar chart."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    methods = df["method"].unique()
    x = np.arange(len(methods))
    width = 0.35
    
    alert_vals = [df[df["method"] == m]["after_neg"].mean() for m in methods]
    drowsy_vals = [df[df["method"] == m]["after_pos"].mean() for m in methods]
    
    ax.bar(x - width/2, alert_vals, width, label="Alert", color="#3498db", alpha=0.8)
    ax.bar(x + width/2, drowsy_vals, width, label="Drowsy", color="#e74c3c", alpha=0.8)
    
    ax.set_xlabel("Sampling Method")
    ax.set_ylabel("Sample Count")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    # Add ratio labels
    for i, m in enumerate(methods):
        ratio = df[df["method"] == m]["after_ratio"].mean()
        ax.annotate(f"{ratio:.3f}", xy=(i, max(alert_vals[i], drowsy_vals[i])),
                    ha="center", va="bottom", fontsize=8)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def plot_theoretical_distribution(
    methods: list,
    ratio: float,
    output_path: Path
) -> None:
    """Plot theoretical sampling distribution."""
    df = calculate_batch_distributions(methods, ratio)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(methods))
    width = 0.35
    
    ax.bar(x - width/2, df["alert"], width, label="Alert", color="#3498db", alpha=0.8)
    ax.bar(x + width/2, df["drowsy"], width, label="Drowsy", color="#e74c3c", alpha=0.8)
    
    ax.axhline(y=DEFAULT_TRAIN_ALERT, color="#3498db", linestyle="--", alpha=0.5, label="Original Alert")
    ax.axhline(y=DEFAULT_TRAIN_DROWSY, color="#e74c3c", linestyle="--", alpha=0.5, label="Original Drowsy")
    
    ax.set_xlabel("Sampling Method")
    ax.set_ylabel("Sample Count")
    ax.set_title(f"Theoretical Sampling Distribution (ratio={ratio})")
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
def cmd_from_logs(args) -> int:
    """Extract and visualize from job logs."""
    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        logger.error(f"Log directory not found: {log_dir}")
        return 1
    
    df = extract_sampling_distribution(log_dir, args.prefix)
    
    if df.empty:
        logger.error("No data extracted from logs")
        return 1
    
    logger.info(f"Extracted {len(df)} records")
    print(df.groupby("method")[["after_neg", "after_pos", "after_ratio"]].mean().to_string())
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_sampling_distribution(df, output_dir / "sampling_distribution_from_logs.png")
    df.to_csv(output_dir / "sampling_distribution.csv", index=False)
    
    return 0


def cmd_theoretical(args) -> int:
    """Calculate and visualize theoretical distribution."""
    methods = args.methods.split(",")
    ratio = args.ratio
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_theoretical_distribution(methods, ratio, output_dir / f"sampling_theoretical_r{ratio}.png")
    
    # Print table
    df = calculate_batch_distributions(methods, ratio)
    print(df.to_string(index=False))
    
    return 0


def cmd_compare(args) -> int:
    """Compare actual vs theoretical distributions."""
    log_dir = Path(args.log_dir)
    df_actual = extract_sampling_distribution(log_dir, args.prefix)
    
    if df_actual.empty:
        logger.error("No actual data found")
        return 1
    
    comparison_df = compare_actual_vs_theoretical(df_actual)
    
    print("\n=== Comparison: Actual vs Theoretical ===")
    print(comparison_df.to_string(index=False))
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Unified sampling distribution visualization tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # from-logs
    p_logs = subparsers.add_parser("from-logs", help="Extract and visualize from job logs")
    p_logs.add_argument("--log-dir", default="scripts/hpc/logs", help="Log directory")
    p_logs.add_argument("--prefix", default="14618", help="Job ID prefix")
    p_logs.add_argument("--output-dir", default="results/imbalance/analysis/sampling", help="Output directory")
    p_logs.set_defaults(func=cmd_from_logs)
    
    # theoretical
    p_theo = subparsers.add_parser("theoretical", help="Calculate theoretical distribution")
    p_theo.add_argument("--methods", default="baseline,smote,smote_tomek,smote_rus,undersample_tomek",
                        help="Comma-separated methods")
    p_theo.add_argument("--ratio", type=float, default=0.5, help="Target ratio")
    p_theo.add_argument("--output-dir", default="results/imbalance/analysis/sampling", help="Output directory")
    p_theo.set_defaults(func=cmd_theoretical)
    
    # compare
    p_comp = subparsers.add_parser("compare", help="Compare actual vs theoretical")
    p_comp.add_argument("--log-dir", default="scripts/hpc/logs", help="Log directory")
    p_comp.add_argument("--prefix", default="14618", help="Job ID prefix")
    p_comp.set_defaults(func=cmd_compare)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
