#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_imbalance_sampling.py
===============================

Visualize sample distribution for imbalance handling experiments.
Shows before/after sampling class distributions for all methods.

Usage:
    python scripts/python/visualization/visualize_imbalance_sampling.py
    python scripts/python/visualization/visualize_imbalance_sampling.py --output results/figures/sampling
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src import config as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================
LOG_DIR = PROJECT_ROOT / "scripts/hpc/logs/imbalance"
OUTPUT_DIR = PROJECT_ROOT / "results/analysis/imbalance/sampling"

# Method display names and colors
METHOD_CONFIG = {
    "baseline": {"label": "Baseline", "color": "#7f8c8d", "order": 0},
    "smote_ratio0.1": {"label": "SMOTE (ratio=0.1)", "color": "#3498db", "order": 1},
    "smote_ratio0.5": {"label": "SMOTE (ratio=0.5)", "color": "#2980b9", "order": 2},
    "subjectwise_smote_ratio0.1": {"label": "Subject-wise SMOTE (ratio=0.1)", "color": "#e74c3c", "order": 3},
    "subjectwise_smote_ratio0.5": {"label": "Subject-wise SMOTE (ratio=0.5)", "color": "#c0392b", "order": 4},
    "undersample_rus_ratio0.1": {"label": "RUS (ratio=0.1)", "color": "#27ae60", "order": 5},
    "undersample_rus_ratio0.5": {"label": "RUS (ratio=0.5)", "color": "#1e8449", "order": 6},
    "balanced_rf": {"label": "Balanced RF", "color": "#9b59b6", "order": 7},
}


def extract_sampling_from_logs(log_dir: Path) -> pd.DataFrame:
    """Extract sampling distribution from HPC log files."""
    data = []
    
    # Find all imbalance experiment logs
    log_files = sorted(log_dir.glob("14658*.OU")) + sorted(log_dir.glob("14662*.OU"))
    
    for log_file in log_files:
        try:
            content = log_file.read_text()
            
            # Extract TAG
            tag_match = re.search(r"^TAG:\s*(\S+)", content, re.MULTILINE)
            if not tag_match:
                continue
            tag = tag_match.group(1)
            
            # Extract seed
            seed_match = re.search(r"_s(\d+)$", tag)
            seed = seed_match.group(1) if seed_match else "42"
            
            # Extract method from tag
            method = tag.replace(f"_s{seed}", "")
            
            # Skip non-imbalance experiments
            if method not in METHOD_CONFIG and not any(m in method for m in METHOD_CONFIG.keys()):
                continue
            
            # Extract class distribution
            before_neg, before_pos = 35522, 1445  # Default values
            after_neg, after_pos = before_neg, before_pos
            
            # Pattern: Class distribution before oversampling: [35522  1445]
            before_match = re.search(r"Class distribution before oversampling:\s*\[(\d+)\s+(\d+)\]", content)
            if before_match:
                before_neg = int(before_match.group(1))
                before_pos = int(before_match.group(2))
            
            # Pattern: Class distribution after oversampling: [35522 17761]
            after_match = re.search(r"Class distribution after oversampling:\s*\[(\d+)\s+(\d+)\]", content)
            if after_match:
                after_neg = int(after_match.group(1))
                after_pos = int(after_match.group(2))
            else:
                # Subject-wise pattern: Class distribution after: [35522 17742]
                sw_match = re.search(r"Class distribution after:\s*\[(\d+)\s+(\d+)\]", content)
                if sw_match:
                    after_neg = int(sw_match.group(1))
                    after_pos = int(sw_match.group(2))
            
            # Get config for this method
            config = METHOD_CONFIG.get(method, {"label": method, "color": "#999999", "order": 99})
            
            data.append({
                "method": method,
                "tag": tag,
                "seed": seed,
                "label": config["label"],
                "color": config["color"],
                "order": config["order"],
                "before_neg": before_neg,
                "before_pos": before_pos,
                "before_total": before_neg + before_pos,
                "before_ratio": before_pos / (before_neg + before_pos),
                "after_neg": after_neg,
                "after_pos": after_pos,
                "after_total": after_neg + after_pos,
                "after_ratio": after_pos / (after_neg + after_pos) if (after_neg + after_pos) > 0 else 0,
            })
            
        except Exception as e:
            logger.warning(f"Failed to parse {log_file}: {e}")
    
    df = pd.DataFrame(data)
    
    # Remove duplicates (keep first occurrence per method+seed)
    if not df.empty:
        df = df.drop_duplicates(subset=["method", "seed"], keep="first")
    
    return df


def plot_sampling_comparison(df: pd.DataFrame, output_path: Path, seed: str = "42") -> None:
    """Plot class distribution comparison (before vs after) as grouped bar chart."""
    if df.empty:
        logger.warning("No data to plot")
        return
    
    # Filter by seed
    df_seed = df[df["seed"] == seed].copy()
    if df_seed.empty:
        logger.warning(f"No data for seed {seed}")
        return
    
    # Sort by order
    df_seed = df_seed.sort_values("order").reset_index(drop=True)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    methods = df_seed["label"].tolist()
    colors = df_seed["color"].tolist()
    n = len(methods)
    y = np.arange(n)
    height = 0.35
    
    # === Left plot: Absolute counts ===
    # Before (lighter)
    ax1.barh(y - height/2, df_seed["before_neg"], height, 
             label="Alert (Before)", color="#3498db", alpha=0.4, edgecolor="#3498db")
    ax1.barh(y + height/2, df_seed["before_pos"], height,
             label="Drowsy (Before)", color="#e74c3c", alpha=0.4, edgecolor="#e74c3c")
    
    # After (solid)
    ax1.barh(y - height/2, df_seed["after_neg"], height,
             label="Alert (After)", color="#3498db", alpha=0.9, edgecolor="#2980b9", linewidth=2)
    ax1.barh(y + height/2, df_seed["after_pos"], height,
             label="Drowsy (After)", color="#e74c3c", alpha=0.9, edgecolor="#c0392b", linewidth=2)
    
    ax1.set_xlabel("Sample Count", fontsize=12)
    ax1.set_ylabel("Method", fontsize=12)
    ax1.set_title(f"Class Distribution (seed={seed})", fontsize=14, fontweight="bold")
    ax1.set_yticks(y)
    ax1.set_yticklabels(methods, fontsize=10)
    ax1.invert_yaxis()
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(axis="x", alpha=0.3)
    ax1.set_xscale("log")  # Log scale for better visibility
    
    # === Right plot: Ratio comparison ===
    bar_before = ax2.barh(y - height/2, df_seed["before_ratio"] * 100, height,
                          label="Before", color="#95a5a6", alpha=0.7)
    bar_after = ax2.barh(y + height/2, df_seed["after_ratio"] * 100, height,
                         label="After", color=colors, alpha=0.9)
    
    # Add percentage labels
    for i, row in df_seed.iterrows():
        ax2.annotate(f"{row['before_ratio']*100:.1f}%", 
                    xy=(row["before_ratio"]*100 + 0.5, list(df_seed.index).index(i) - height/2),
                    ha="left", va="center", fontsize=9, color="#666")
        ax2.annotate(f"{row['after_ratio']*100:.1f}%", 
                    xy=(row["after_ratio"]*100 + 0.5, list(df_seed.index).index(i) + height/2),
                    ha="left", va="center", fontsize=9, fontweight="bold")
    
    ax2.set_xlabel("Positive Class Ratio (%)", fontsize=12)
    ax2.set_title("Class Ratio (Drowsy / Total)", fontsize=14, fontweight="bold")
    ax2.set_yticks(y)
    ax2.set_yticklabels(methods, fontsize=10)
    ax2.invert_yaxis()
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(axis="x", alpha=0.3)
    ax2.set_xlim(0, 55)  # Max 50% + margin
    
    fig.suptitle("Imbalance Handling: Sample Distribution", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def plot_sampling_summary_table(df: pd.DataFrame, output_path: Path) -> None:
    """Create a summary table visualization."""
    if df.empty:
        return
    
    # Aggregate by method (average over seeds)
    summary = df.groupby("method").agg({
        "label": "first",
        "order": "first",
        "before_neg": "mean",
        "before_pos": "mean",
        "after_neg": "mean",
        "after_pos": "mean",
        "before_ratio": "mean",
        "after_ratio": "mean",
    }).reset_index()
    summary = summary.sort_values("order")
    
    # Create figure for table
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")
    
    # Prepare table data
    columns = ["Method", "Before\nAlert", "Before\nDrowsy", "Before\nRatio", 
               "After\nAlert", "After\nDrowsy", "After\nRatio", "Change"]
    
    table_data = []
    for _, row in summary.iterrows():
        change = row["after_ratio"] / row["before_ratio"] if row["before_ratio"] > 0 else 1
        table_data.append([
            row["label"],
            f"{int(row['before_neg']):,}",
            f"{int(row['before_pos']):,}",
            f"{row['before_ratio']*100:.2f}%",
            f"{int(row['after_neg']):,}",
            f"{int(row['after_pos']):,}",
            f"{row['after_ratio']*100:.2f}%",
            f"×{change:.1f}" if change != 1 else "-",
        ])
    
    table = ax.table(cellText=table_data, colLabels=columns, loc="center",
                     cellLoc="center", colColours=["#f0f0f0"]*len(columns))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_text_props(fontweight="bold")
    
    plt.title("Sample Distribution Summary (Averaged over Seeds)", fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def plot_method_comparison_stacked(df: pd.DataFrame, output_path: Path) -> None:
    """Create stacked bar chart comparing all methods."""
    if df.empty:
        return
    
    # Average over seeds
    summary = df.groupby("method").agg({
        "label": "first",
        "order": "first",
        "color": "first",
        "after_neg": "mean",
        "after_pos": "mean",
        "after_ratio": "mean",
    }).reset_index()
    summary = summary.sort_values("order")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    methods = summary["label"].tolist()
    x = np.arange(len(methods))
    width = 0.6
    
    # Stacked bar
    bars_neg = ax.bar(x, summary["after_neg"], width, label="Alert (Negative)", color="#3498db", alpha=0.8)
    bars_pos = ax.bar(x, summary["after_pos"], width, bottom=summary["after_neg"], 
                      label="Drowsy (Positive)", color="#e74c3c", alpha=0.8)
    
    # Add ratio labels on top
    for i, (neg, pos, ratio) in enumerate(zip(summary["after_neg"], summary["after_pos"], summary["after_ratio"])):
        ax.annotate(f"{ratio*100:.1f}%", xy=(i, neg + pos + 500), ha="center", va="bottom", 
                   fontsize=10, fontweight="bold")
    
    ax.set_ylabel("Sample Count", fontsize=12)
    ax.set_xlabel("Method", fontsize=12)
    ax.set_title("Sample Distribution by Method (After Sampling)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=10)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    
    # Add horizontal line for original ratio
    original_ratio_count = 35522 + 1445  # Original total
    ax.axhline(y=original_ratio_count, color="gray", linestyle="--", alpha=0.7, label="Original Total")
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize imbalance sampling distribution")
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR, help="Log directory")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--seed", default="42", help="Seed to visualize (42 or 123)")
    args = parser.parse_args()
    
    # Create output directory (use directly, no subdirectory)
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract data from logs
    logger.info(f"Extracting sampling info from {args.log_dir}")
    df = extract_sampling_from_logs(args.log_dir)
    
    if df.empty:
        logger.error("No sampling data found")
        return 1
    
    logger.info(f"Found {len(df)} experiments")
    print("\n" + "="*60)
    print("Sampling Distribution Summary")
    print("="*60)
    print(df[["method", "seed", "before_pos", "before_neg", "after_pos", "after_neg", "after_ratio"]].to_string(index=False))
    print("="*60 + "\n")
    
    # Generate visualizations
    for seed in ["42", "123"]:
        plot_sampling_comparison(df, output_dir / f"sampling_comparison_s{seed}.png", seed=seed)
    
    plot_sampling_summary_table(df, output_dir / "sampling_summary_table.png")
    plot_method_comparison_stacked(df, output_dir / "sampling_stacked_bar.png")
    
    # Save CSV
    csv_path = output_dir / "sampling_data.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV: {csv_path}")
    
    logger.info("Visualization complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
