#!/usr/bin/env python3
"""
Prior Research Comparison Visualization
========================================

This script generates comparison plots between prior research methods
(SvmA, SvmW, Lstm) and the BalancedRF baseline using pooled training mode.

Output:
    results/analysis/prior_research/
    - metrics_comparison.png: Bar chart comparing F1, Precision, Recall
    - metrics_table.csv: Summary table of all metrics
"""

import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

# Paths
RESULTS_BASE = PROJECT_ROOT / "results" / "outputs" / "training"
OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis" / "prior_research"


def find_result_files() -> Dict[str, str]:
    """
    Find all prior research and baseline result files.
    
    Returns
    -------
    Dict[str, str]
        Dictionary mapping model_seed to file path
    """
    result_files = {}
    
    # Prior research models
    prior_research_patterns = {
        "SvmW": "**/SvmW/**/train_results_SvmW_pooled_prior_research_*.json",
        "SvmA": "**/SvmA/**/train_results_SvmA_pooled_prior_research_*.json",
        "Lstm": "**/Lstm/**/train_results_Lstm_pooled_prior_research_*.json",
    }
    
    for model, pattern in prior_research_patterns.items():
        for path in RESULTS_BASE.glob(pattern):
            # Extract seed from filename
            filename = path.stem
            if "_s42" in filename:
                seed = "s42"
            elif "_s123" in filename:
                seed = "s123"
            else:
                continue
            key = f"{model}_{seed}"
            result_files[key] = str(path)
    
    # Baseline BalancedRF (pooled mode)
    brf_pattern = "**/BalancedRF/**/train_results_BalancedRF_pooled_balanced_rf_*.json"
    for path in RESULTS_BASE.glob(brf_pattern):
        filename = path.stem
        if "_s42" in filename:
            seed = "s42"
        elif "_s123" in filename:
            seed = "s123"
        else:
            continue
        key = f"BalancedRF_{seed}"
        result_files[key] = str(path)
    
    return result_files


def load_results(result_files: Dict[str, str]) -> pd.DataFrame:
    """
    Load results from JSON files into a DataFrame.
    
    Parameters
    ----------
    result_files : Dict[str, str]
        Dictionary mapping model_seed to file path
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: model, seed, split, f1, precision, recall, accuracy, roc_auc, auc_pr
    """
    records = []
    
    for key, path in result_files.items():
        model, seed = key.rsplit("_", 1)
        print(f"  Loading {key}...", end=" ", flush=True)
        
        try:
            # Read file and extract only needed keys (avoid loading huge arrays)
            with open(path, "r") as f:
                content = f.read()
            
            # Quick extraction using string parsing for speed
            data = json.loads(content)
            
            for split in ["train", "val", "test"]:
                if split in data and isinstance(data[split], dict):
                    metrics = data[split]
                    records.append({
                        "model": model,
                        "seed": seed,
                        "split": split,
                        "f1": metrics.get("f1", None),
                        "precision": metrics.get("precision", None),
                        "recall": metrics.get("recall", None),
                        "accuracy": metrics.get("accuracy", None),
                        "roc_auc": metrics.get("roc_auc", None),
                        "auc_pr": metrics.get("auc_pr", None),
                    })
            print("OK")
        except Exception as e:
            print(f"ERROR: {e}")
            continue
    
    return pd.DataFrame(records)


def plot_metrics_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """
    Create a bar chart comparing F1, Precision, and Recall across models.
    
    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame
    output_path : Path
        Output path for the plot
    """
    # Focus on validation split (most comparable)
    val_df = df[df["split"] == "val"].copy()
    
    if val_df.empty:
        print("No validation data found, using train split")
        val_df = df[df["split"] == "train"].copy()
    
    # Average over seeds
    avg_df = val_df.groupby("model")[["f1", "precision", "recall"]].mean().reset_index()
    
    # Order models: prior research first, then baseline
    model_order = ["SvmA", "SvmW", "Lstm", "BalancedRF"]
    avg_df["model"] = pd.Categorical(avg_df["model"], categories=model_order, ordered=True)
    avg_df = avg_df.sort_values("model").dropna(subset=["model"])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(avg_df))
    width = 0.25
    
    metrics = ["f1", "precision", "recall"]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]
    labels = ["F1 Score", "Precision", "Recall"]
    
    for i, (metric, color, label) in enumerate(zip(metrics, colors, labels)):
        values = avg_df[metric].fillna(0).values
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width, label=label, color=color, alpha=0.8)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            if val > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Prior Research Methods vs BalancedRF Baseline\n(Validation Set Performance)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(avg_df["model"].values)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    
    # Add annotation about prior research
    ax.annotate("Prior Research Methods", xy=(1, -0.12), xycoords="axes fraction",
                ha="center", fontsize=10, style="italic")
    ax.annotate("Baseline", xy=(0.85, -0.12), xycoords="axes fraction",
                ha="center", fontsize=10, style="italic")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_train_val_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """
    Create a comparison plot showing train vs validation performance.
    
    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame
    output_path : Path
        Output path for the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ax, split, title in zip(axes, ["train", "val"], ["Training Set", "Validation Set"]):
        split_df = df[df["split"] == split].copy()
        avg_df = split_df.groupby("model")[["f1", "precision", "recall"]].mean().reset_index()
        
        model_order = ["SvmA", "SvmW", "Lstm", "BalancedRF"]
        avg_df["model"] = pd.Categorical(avg_df["model"], categories=model_order, ordered=True)
        avg_df = avg_df.sort_values("model").dropna(subset=["model"])
        
        x = np.arange(len(avg_df))
        width = 0.25
        
        for i, (metric, color) in enumerate(zip(["f1", "precision", "recall"], 
                                                  ["#1f77b4", "#2ca02c", "#ff7f0e"])):
            values = avg_df[metric].fillna(0).values
            offset = (i - 1) * width
            ax.bar(x + offset, values, width, label=metric.capitalize(), color=color, alpha=0.8)
        
        ax.set_xlabel("Model", fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(avg_df["model"].values)
        ax.legend(loc="upper right")
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=0.3)
    
    plt.suptitle("Prior Research Methods: Training vs Validation Performance", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def create_summary_table(df: pd.DataFrame, output_path: Path) -> None:
    """
    Create a summary CSV table of all results.
    
    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame
    output_path : Path
        Output path for the CSV
    """
    # Pivot to wide format
    summary = df.pivot_table(
        index=["model", "seed"],
        columns="split",
        values=["f1", "precision", "recall", "accuracy", "roc_auc", "auc_pr"]
    ).round(4)
    
    # Flatten column names
    summary.columns = [f"{split}_{metric}" for metric, split in summary.columns]
    summary = summary.reset_index()
    
    summary.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    
    # Also print to console
    print("\nSummary Table:")
    print(summary.to_string(index=False))


def main():
    """Main function."""
    print("=" * 60)
    print("Prior Research Comparison Visualization")
    print("=" * 60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find result files
    print("\nSearching for result files...")
    result_files = find_result_files()
    
    if not result_files:
        print("ERROR: No result files found!")
        return 1
    
    print(f"Found {len(result_files)} result files:")
    for key, path in sorted(result_files.items()):
        print(f"  - {key}: {Path(path).name}")
    
    # Load results
    print("\nLoading results...")
    df = load_results(result_files)
    
    if df.empty:
        print("ERROR: No data loaded!")
        return 1
    
    print(f"Loaded {len(df)} records")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    plot_metrics_comparison(df, OUTPUT_DIR / "metrics_comparison.png")
    plot_train_val_comparison(df, OUTPUT_DIR / "train_val_comparison.png")
    create_summary_table(df, OUTPUT_DIR / "metrics_table.csv")
    
    print("\n" + "=" * 60)
    print("Done! Output saved to:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
