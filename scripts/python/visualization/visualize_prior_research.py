#!/usr/bin/env python3
"""
Prior Research Comparison Visualization
========================================

This script generates comparison plots between prior research methods
(SvmA, SvmW, Lstm) and the RF baseline using pooled training mode.

Metrics visualized:
- Accuracy, Recall, Precision, Specificity
- F1, F2, AUROC, AUPRC

Output:
    results/analysis/exp3_prior_research/
    - train_metrics.png: Training set metrics
    - val_metrics.png: Validation set metrics  
    - test_metrics.png: Test set metrics
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

# Metrics to visualize
METRICS = ["accuracy", "recall", "precision", "specificity", "f1", "f2", "roc_auc", "auc_pr"]
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "recall": "Recall",
    "precision": "Precision", 
    "specificity": "Specificity",
    "f1": "F1 Score",
    "f2": "F2 Score",
    "roc_auc": "AUROC",
    "auc_pr": "AUPRC"
}


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
    
    # Baseline RF (pooled mode) - look for baseline results
    rf_pattern = "**/RF/**/train_results_RF_pooled_baseline_*.json"
    for path in RESULTS_BASE.glob(rf_pattern):
        filename = path.stem
        if "_s42" in filename:
            seed = "s42"
        elif "_s123" in filename:
            seed = "s123"
        else:
            continue
        key = f"RF_{seed}"
        result_files[key] = str(path)
    
    return result_files


def compute_derived_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute derived metrics (specificity, F2) from confusion matrix.
    
    Parameters
    ----------
    metrics : Dict[str, Any]
        Original metrics dictionary
    
    Returns
    -------
    Dict[str, Any]
        Metrics dictionary with derived values added
    """
    result = dict(metrics)
    
    # Extract confusion matrix if available
    cm = metrics.get("confusion_matrix")
    if cm and len(cm) == 2 and len(cm[0]) == 2:
        tn, fp = cm[0][0], cm[0][1]
        fn, tp = cm[1][0], cm[1][1]
        
        # Specificity = TN / (TN + FP)
        if (tn + fp) > 0:
            result["specificity"] = tn / (tn + fp)
        else:
            result["specificity"] = 0.0
        
        # F2 score: beta=2, weights recall higher
        # F_beta = (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall)
        precision = metrics.get("precision", 0)
        recall = metrics.get("recall", 0)
        beta = 2
        if (beta**2 * precision + recall) > 0:
            result["f2"] = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
        else:
            result["f2"] = 0.0
    else:
        result["specificity"] = None
        result["f2"] = None
    
    return result


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
        DataFrame with all metrics for each model/seed/split
    """
    records = []
    
    for key, path in result_files.items():
        model, seed = key.rsplit("_", 1)
        print(f"  Loading {key}...", end=" ", flush=True)
        
        try:
            with open(path, "r") as f:
                data = json.load(f)
            
            for split in ["train", "val", "test"]:
                if split in data and isinstance(data[split], dict):
                    metrics = compute_derived_metrics(data[split])
                    record = {
                        "model": model,
                        "seed": seed,
                        "split": split,
                    }
                    for metric in METRICS:
                        record[metric] = metrics.get(metric, None)
                    records.append(record)
            print("OK")
        except Exception as e:
            print(f"ERROR: {e}")
            continue
    
    return pd.DataFrame(records)


def plot_metrics_by_split(df: pd.DataFrame, split: str, output_path: Path) -> None:
    """
    Create a comprehensive metrics visualization for a specific split.
    
    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame
    split : str
        Split name ('train', 'val', 'test')
    output_path : Path
        Output path for the plot
    """
    split_df = df[df["split"] == split].copy()
    
    if split_df.empty:
        print(f"  No data for {split} split, skipping...")
        return
    
    # Average over seeds
    avg_df = split_df.groupby("model")[METRICS].mean().reset_index()
    std_df = split_df.groupby("model")[METRICS].std().reset_index()
    
    # Order models: prior research first, then baseline
    model_order = ["SvmA", "SvmW", "Lstm", "RF"]
    avg_df["model"] = pd.Categorical(avg_df["model"], categories=model_order, ordered=True)
    avg_df = avg_df.sort_values("model").dropna(subset=["model"])
    std_df["model"] = pd.Categorical(std_df["model"], categories=model_order, ordered=True)
    std_df = std_df.sort_values("model").dropna(subset=["model"])
    
    # Create figure with 2x4 subplots
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(avg_df)))
    
    split_titles = {
        "train": "Training Set",
        "val": "Validation Set",
        "test": "Test Set"
    }
    
    for idx, metric in enumerate(METRICS):
        ax = axes[idx]
        
        values = avg_df[metric].fillna(0).values
        errors = std_df[metric].fillna(0).values if len(std_df) == len(avg_df) else None
        
        x = np.arange(len(avg_df))
        bars = ax.bar(x, values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add error bars if we have multiple seeds
        if errors is not None and not np.all(np.isnan(errors)):
            ax.errorbar(x, values, yerr=errors, fmt='none', color='black', capsize=3)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            if val > 0.001:
                label = f'{val:.3f}' if val < 1 else f'{val:.2f}'
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       label, ha='center', va='bottom', fontsize=8, rotation=0)
        
        ax.set_xlabel("")
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=10)
        ax.set_title(METRIC_LABELS[metric], fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(avg_df["model"].values, rotation=45, ha='right', fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.suptitle(f"Prior Research Methods vs RF Baseline\n{split_titles[split]} Performance", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_all_splits_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """
    Create a comparison plot showing all splits for each model.
    
    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame
    output_path : Path
        Output path for the plot
    """
    # Average over seeds
    avg_df = df.groupby(["model", "split"])[METRICS].mean().reset_index()
    
    model_order = ["SvmA", "SvmW", "Lstm", "RF"]
    split_order = ["train", "val", "test"]
    
    # Create figure with 2x4 subplots for metrics
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    axes = axes.flatten()
    
    colors = {"train": "#1f77b4", "val": "#2ca02c", "test": "#ff7f0e"}
    
    for idx, metric in enumerate(METRICS):
        ax = axes[idx]
        
        x = np.arange(len(model_order))
        width = 0.25
        
        for i, split in enumerate(split_order):
            split_data = avg_df[avg_df["split"] == split]
            values = []
            for model in model_order:
                model_data = split_data[split_data["model"] == model]
                if not model_data.empty:
                    val = model_data[metric].values[0]
                    values.append(val if pd.notna(val) else 0)
                else:
                    values.append(0)
            
            offset = (i - 1) * width
            bars = ax.bar(x + offset, values, width, label=split.capitalize(), 
                         color=colors[split], alpha=0.8)
        
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=10)
        ax.set_title(METRIC_LABELS[metric], fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_order, rotation=45, ha='right', fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.suptitle("Prior Research Methods: Train vs Validation vs Test Performance", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


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
    cols_to_pivot = METRICS
    summary = df.pivot_table(
        index=["model", "seed"],
        columns="split",
        values=cols_to_pivot
    ).round(4)
    
    # Flatten column names
    summary.columns = [f"{split}_{metric}" for metric, split in summary.columns]
    summary = summary.reset_index()
    
    # Reorder columns
    col_order = ["model", "seed"]
    for split in ["train", "val", "test"]:
        for metric in METRICS:
            col_name = f"{split}_{metric}"
            if col_name in summary.columns:
                col_order.append(col_name)
    summary = summary[[c for c in col_order if c in summary.columns]]
    
    summary.to_csv(output_path, index=False)
    print(f"  Saved: {output_path.name}")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics (averaged over seeds)")
    print("=" * 80)
    
    avg_summary = df.groupby(["model", "split"])[METRICS].mean().round(4)
    for model in ["SvmA", "SvmW", "Lstm", "RF"]:
        if model in avg_summary.index.get_level_values(0):
            print(f"\n{model}:")
            for split in ["train", "val", "test"]:
                if (model, split) in avg_summary.index:
                    row = avg_summary.loc[(model, split)]
                    metrics_str = ", ".join([f"{METRIC_LABELS[m]}={row[m]:.4f}" 
                                            for m in METRICS if pd.notna(row[m])])
                    print(f"  {split:5s}: {metrics_str}")


def main():
    """Main function."""
    print("=" * 70)
    print("Prior Research Comparison Visualization")
    print("=" * 70)
    
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
    
    # Individual split plots
    for split in ["train", "val", "test"]:
        plot_metrics_by_split(df, split, OUTPUT_DIR / f"{split}_metrics.png")
    
    # All splits comparison
    plot_all_splits_comparison(df, OUTPUT_DIR / "all_splits_comparison.png")
    
    # Summary table
    create_summary_table(df, OUTPUT_DIR / "metrics_table.csv")
    
    print("\n" + "=" * 70)
    print("Done! Output saved to:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
