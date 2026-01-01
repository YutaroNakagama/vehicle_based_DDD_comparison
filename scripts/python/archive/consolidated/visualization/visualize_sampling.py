#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_sampling.py
=====================

Unified tool for sampling distribution visualization.

This script provides:
1. from-logs   - Extract and visualize sampling distribution from job logs
2. theoretical - Calculate and visualize theoretical sampling distribution
3. compare     - Compare actual vs theoretical distributions

Consolidates functionality from:
- visualize_sampling_distribution.py
- visualize_training_data_sampling.py

Usage:
    python visualize_sampling.py from-logs --log-dir scripts/hpc/logs --prefix 14618
    python visualize_sampling.py theoretical --methods smote,smote_tomek --ratio 0.5
    python visualize_sampling.py compare --log-dir scripts/hpc/logs
"""

import argparse
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

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src import config as cfg

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Original training data distribution
ORIGINAL_TRAIN_ALERT = 35522
ORIGINAL_TRAIN_DROWSY = 1445


# ============================================================
# Log-based extraction
# ============================================================
def extract_sampling_distribution(log_dir: Path, job_prefix: str = "14618") -> pd.DataFrame:
    """Extract sampling distribution from job logs."""
    results = []
    
    for logfile in sorted(log_dir.glob(f"{job_prefix}*.OU")):
        jobid = logfile.stem.split(".")[0]
        try:
            content = logfile.read_text()
        except Exception:
            continue
        
        tag_match = re.search(r"tag=(\S+)", content)
        tag = tag_match.group(1) if tag_match else "unknown"
        
        orig_match = re.search(r"train n=(\d+) pos=(\d+)", content)
        if not orig_match:
            continue
        
        orig_total = int(orig_match.group(1))
        orig_pos = int(orig_match.group(2))
        orig_neg = orig_total - orig_pos
        
        dist_match = re.search(r"Class distribution after oversampling: \[(\d+)\s+(\d+)\]", content)
        if dist_match:
            after_neg = int(dist_match.group(1))
            after_pos = int(dist_match.group(2))
        else:
            after_neg = orig_neg
            after_pos = orig_pos
        
        method = tag.replace("imbal_v2_", "").split("_seed")[0]
        method = re.sub(r"_ratio\d+_\d+", "", method)
        
        ratio_match = re.search(r"ratio(\d+)_(\d+)", tag)
        ratio = float(f"{ratio_match.group(1)}.{ratio_match.group(2)}") if ratio_match else None
        
        seed_match = re.search(r"seed(\d+)", tag)
        seed = int(seed_match.group(1)) if seed_match else 42
        
        results.append({
            "jobid": jobid,
            "tag": tag,
            "method": method,
            "ratio": ratio,
            "seed": seed,
            "orig_neg": orig_neg,
            "orig_pos": orig_pos,
            "after_neg": after_neg,
            "after_pos": after_pos,
            "after_total": after_neg + after_pos,
            "after_ratio": after_pos / after_neg if after_neg > 0 else 0,
        })
    
    return pd.DataFrame(results)


# ============================================================
# Theoretical calculation
# ============================================================
def calculate_sampling_distribution(method: str, ratio: float) -> dict:
    """Calculate expected training data distribution after sampling."""
    alert = ORIGINAL_TRAIN_ALERT
    drowsy = ORIGINAL_TRAIN_DROWSY
    
    if method == "baseline":
        pass
    elif method.startswith("smote") or method.startswith("adasyn"):
        new_drowsy = int(alert * ratio)
        drowsy = max(drowsy, new_drowsy)
        
        if "tomek" in method:
            reduction = 0.01
            alert = int(alert * (1 - reduction))
            drowsy = int(drowsy * (1 - reduction * 0.5))
        elif "enn" in method:
            reduction = 0.02
            alert = int(alert * (1 - reduction))
            drowsy = int(drowsy * (1 - reduction * 0.3))
        elif "rus" in method:
            new_alert = int(drowsy / ratio) if ratio > 0 else alert
            alert = min(alert, new_alert)
    elif method.startswith("undersample"):
        new_alert = int(drowsy / ratio) if ratio > 0 else alert
        alert = min(alert, new_alert)
    
    total = alert + drowsy
    return {
        "method": method,
        "ratio": ratio,
        "alert": alert,
        "drowsy": drowsy,
        "total": total,
        "drowsy_pct": drowsy / total * 100 if total > 0 else 0,
    }


# ============================================================
# Visualization
# ============================================================
def plot_sampling_distribution(df: pd.DataFrame, output_path: Path, title: str = "Sampling Distribution") -> None:
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


def plot_theoretical_distribution(methods: List[str], ratio: float, output_path: Path) -> None:
    """Plot theoretical sampling distribution."""
    data = [calculate_sampling_distribution(m, ratio) for m in methods]
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(methods))
    width = 0.35
    
    ax.bar(x - width/2, df["alert"], width, label="Alert", color="#3498db", alpha=0.8)
    ax.bar(x + width/2, df["drowsy"], width, label="Drowsy", color="#e74c3c", alpha=0.8)
    
    ax.axhline(y=ORIGINAL_TRAIN_ALERT, color="#3498db", linestyle="--", alpha=0.5, label="Original Alert")
    ax.axhline(y=ORIGINAL_TRAIN_DROWSY, color="#e74c3c", linestyle="--", alpha=0.5, label="Original Drowsy")
    
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
    data = [calculate_sampling_distribution(m, ratio) for m in methods]
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    
    return 0


def cmd_compare(args) -> int:
    """Compare actual vs theoretical distributions."""
    log_dir = Path(args.log_dir)
    df_actual = extract_sampling_distribution(log_dir, args.prefix)
    
    if df_actual.empty:
        logger.error("No actual data found")
        return 1
    
    methods = df_actual["method"].unique().tolist()
    ratio = df_actual["ratio"].dropna().mode().iloc[0] if not df_actual["ratio"].dropna().empty else 0.5
    
    data_theoretical = [calculate_sampling_distribution(m, ratio) for m in methods]
    df_theoretical = pd.DataFrame(data_theoretical)
    
    print("\n=== Comparison: Actual vs Theoretical ===")
    for m in methods:
        actual = df_actual[df_actual["method"] == m]
        theoretical = df_theoretical[df_theoretical["method"] == m]
        
        if actual.empty or theoretical.empty:
            continue
        
        actual_ratio = actual["after_ratio"].mean()
        theo_ratio = theoretical["drowsy"].iloc[0] / theoretical["alert"].iloc[0] if theoretical["alert"].iloc[0] > 0 else 0
        
        print(f"{m:20s}: Actual={actual_ratio:.4f}, Theoretical={theo_ratio:.4f}")
    
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
    p_logs.add_argument("--output-dir", default="results/imbalance_analysis/sampling", help="Output directory")
    p_logs.set_defaults(func=cmd_from_logs)
    
    # theoretical
    p_theo = subparsers.add_parser("theoretical", help="Calculate theoretical distribution")
    p_theo.add_argument("--methods", default="baseline,smote,smote_tomek,smote_rus,undersample_tomek",
                        help="Comma-separated methods")
    p_theo.add_argument("--ratio", type=float, default=0.5, help="Target ratio")
    p_theo.add_argument("--output-dir", default="results/imbalance_analysis/sampling", help="Output directory")
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
