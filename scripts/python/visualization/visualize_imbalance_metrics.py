#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_imbalance_metrics.py
==============================

Visualize metrics comparison for imbalance handling experiments.
Extracts metrics from training result files and creates comparison plots.

Metrics: Accuracy, Recall, Precision, Specificity, F1, F2, AUROC, AUPRC
Splits: Train, Validation, Test

Usage:
    python scripts/python/visualization/visualize_imbalance_metrics.py
    python scripts/python/visualization/visualize_imbalance_metrics.py --split train
    python scripts/python/visualization/visualize_imbalance_metrics.py --split val
    python scripts/python/visualization/visualize_imbalance_metrics.py --split test
    python scripts/python/visualization/visualize_imbalance_metrics.py --split all
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

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src import config as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================
TRAIN_DIR = Path(cfg.RESULTS_OUTPUTS_PATH) / "training"
EVAL_DIR = Path(cfg.RESULTS_OUTPUTS_PATH) / "evaluation"
OUTPUT_DIR = PROJECT_ROOT / "results/analysis/imbalance/metrics"

SPLITS = ["train", "val", "test"]
SPLIT_LABELS = {"train": "Training Set", "val": "Validation Set", "test": "Test Set"}

# Method configuration
METHOD_CONFIG = {
    "baseline": {"label": "Baseline", "color": "#7f8c8d", "order": 0},
    "smote_ratio0.1": {"label": "SMOTE 0.1", "color": "#3498db", "order": 1},
    "smote_ratio0.5": {"label": "SMOTE 0.5", "color": "#2980b9", "order": 2},
    "subjectwise_smote_ratio0.1": {"label": "SW-SMOTE 0.1", "color": "#e74c3c", "order": 3},
    "subjectwise_smote_ratio0.5": {"label": "SW-SMOTE 0.5", "color": "#c0392b", "order": 4},
    "undersample_rus_ratio0.1": {"label": "RUS 0.1", "color": "#27ae60", "order": 5},
    "undersample_rus_ratio0.5": {"label": "RUS 0.5", "color": "#1e8449", "order": 6},
    "balanced_rf": {"label": "Balanced RF", "color": "#9b59b6", "order": 7},
}

SEED_STYLES = {
    "42": {"hatch": "", "alpha": 0.9},
    "123": {"hatch": "//", "alpha": 0.7},
}

METRICS = ["accuracy", "recall", "precision", "specificity", "f1", "f2", "roc_auc", "auc_pr"]
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "recall": "Recall",
    "precision": "Precision", 
    "specificity": "Specificity",
    "f1": "F1 Score",
    "f2": "F2 Score",
    "roc_auc": "AUROC",
    "auc_pr": "AUPRC",
}


def find_training_files(train_dir: Path) -> List[Path]:
    """Find all training result files for imbalance experiments (contain train/val/test)."""
    files = []
    
    for model_dir in ["RF", "BalancedRF"]:
        model_path = train_dir / model_dir
        if not model_path.exists():
            continue
        
        for job_dir in model_path.iterdir():
            if not job_dir.is_dir():
                continue
            
            for subdir in job_dir.iterdir():
                if not subdir.is_dir():
                    continue
                
                for f in subdir.glob("train_results_*_pooled_*.json"):
                    fname = f.name
                    if any(m in fname for m in ["baseline_s", "smote_ratio", "subjectwise_smote", 
                                                 "undersample_rus", "balanced_rf_s"]):
                        files.append(f)
    
    return files


def load_metrics_from_files(files: List[Path], split: str = "test") -> pd.DataFrame:
    """Load metrics from training JSON files for specified split (train/val/test)."""
    data = []
    
    for f in files:
        try:
            with open(f, "r") as fp:
                result = json.load(fp)
            
            # Extract tag from filename: train_results_RF_pooled_{tag}.json
            fname = f.stem
            tag_match = re.search(r"pooled_(.+)$", fname)
            if not tag_match:
                continue
            tag = tag_match.group(1)
            
            # Extract seed
            seed_match = re.search(r"_s(\d+)$", tag)
            seed = seed_match.group(1) if seed_match else "42"
            
            # Extract method
            method = tag.replace(f"_s{seed}", "")
            
            # Get config
            config = METHOD_CONFIG.get(method, {"label": method, "color": "#999999", "order": 99})
            
            row = {
                "method": method,
                "tag": tag,
                "seed": seed,
                "label": config["label"],
                "color": config["color"],
                "order": config["order"],
                "model": "BalancedRF" if "BalancedRF" in str(f) else "RF",
                "split": split,
            }
            
            # Get metrics from specified split
            if split not in result:
                logger.warning(f"Split '{split}' not found in {f}")
                continue
            
            metrics_source = result[split]
            
            # Map different key names to standard names
            metric_mappings = {
                "accuracy": ["accuracy", "acc_thr", "test_accuracy"],
                "recall": ["recall", "recall_thr", "test_recall"],
                "precision": ["precision", "prec_thr", "test_precision"],
                "f1": ["f1", "f1_thr", "test_f1"],
                "roc_auc": ["roc_auc", "auroc", "test_roc_auc"],
                "auc_pr": ["auc_pr", "auprc", "pr_auc", "test_auc_pr"],
            }
            
            for metric, keys in metric_mappings.items():
                for key in keys:
                    if key in metrics_source and metrics_source[key] is not None:
                        row[metric] = metrics_source[key]
                        break
            
            # Calculate specificity and F2 from confusion matrix if not present
            if "confusion_matrix" in metrics_source:
                cm = metrics_source["confusion_matrix"]
                # cm = [[TN, FP], [FN, TP]] or [[TP, FN], [FP, TN]] depending on format
                # Check based on typical drowsy detection setup (minority is positive)
                if len(cm) == 2 and len(cm[0]) == 2:
                    # Assume format: [[TN, FP], [FN, TP]]
                    tn, fp = cm[0][0], cm[0][1]
                    fn, tp = cm[1][0], cm[1][1]
                    
                    # Calculate specificity: TN / (TN + FP)
                    if (tn + fp) > 0:
                        row["specificity"] = tn / (tn + fp)
                    else:
                        row["specificity"] = 0.0
                    
                    # Calculate F2 score: (1 + 2^2) * (precision * recall) / (2^2 * precision + recall)
                    precision = row.get("precision", 0)
                    recall = row.get("recall", 0)
                    if (4 * precision + recall) > 0:
                        row["f2"] = (5 * precision * recall) / (4 * precision + recall)
                    else:
                        row["f2"] = 0.0
            
            data.append(row)
            
        except Exception as e:
            logger.warning(f"Failed to load {f}: {e}")
    
    df = pd.DataFrame(data)
    
    # Remove duplicates (keep latest per method+seed)
    if not df.empty:
        df = df.drop_duplicates(subset=["method", "seed"], keep="last")
    
    return df


def plot_metrics_grid(df: pd.DataFrame, output_path: Path, split: str = "test") -> None:
    """Create 2x4 grid of metric bar charts."""
    if df.empty:
        logger.warning("No data to plot")
        return
    
    # Sort by order and seed
    df = df.sort_values(["order", "seed"]).reset_index(drop=True)
    
    # Create unique experiment labels
    df["exp_label"] = df.apply(lambda r: f"{r['label']} (s{r['seed']})", axis=1)
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    axes = axes.flatten()
    
    experiments = df["exp_label"].tolist()
    x = np.arange(len(experiments))
    width = 0.7
    
    # Colors and hatches based on method and seed
    colors = df["color"].tolist()
    hatches = [SEED_STYLES.get(s, {}).get("hatch", "") for s in df["seed"]]
    
    available_metrics = [m for m in METRICS if m in df.columns and df[m].notna().any()]
    
    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        values = df[metric].fillna(0).values
        
        bars = ax.bar(x, values, width, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
        
        # Apply hatching for seed 123
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        
        ax.set_title(METRIC_LABELS.get(metric, metric.upper()), fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(experiments, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, min(1.0, max(values) * 1.15) if max(values) > 0 else 1)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        
        # Add value labels (including 0%)
        for bar, val in zip(bars, values):
            ax.annotate(f"{val:.3f}", xy=(bar.get_x() + bar.get_width()/2, max(bar.get_height(), 0.02)),
                       ha="center", va="bottom", fontsize=7, rotation=0)
    
    # Hide unused axes
    for idx in range(len(available_metrics), len(axes)):
        axes[idx].set_visible(False)
    
    split_label = SPLIT_LABELS.get(split, split.title())
    fig.suptitle(f"Imbalance Handling Methods: Metrics Comparison ({split_label})", 
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def plot_metrics_grouped_by_method(df: pd.DataFrame, output_path: Path, split: str = "test") -> None:
    """Create grouped bar chart (seed 42 vs 123) for each method."""
    if df.empty:
        return
    
    # Aggregate by method (average metrics)
    summary = df.groupby("method").agg({
        "label": "first",
        "order": "first",
        "color": "first",
        **{m: "mean" for m in METRICS if m in df.columns}
    }).reset_index()
    summary = summary.sort_values("order")
    
    available_metrics = [m for m in METRICS if m in summary.columns and summary[m].notna().any()]
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    axes = axes.flatten()
    
    methods = summary["label"].tolist()
    x = np.arange(len(methods))
    width = 0.35
    
    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        
        # Get values for each seed
        vals_42 = []
        vals_123 = []
        for method in summary["method"]:
            df_m = df[df["method"] == method]
            v42 = df_m[df_m["seed"] == "42"][metric].values
            v123 = df_m[df_m["seed"] == "123"][metric].values
            vals_42.append(v42[0] if len(v42) > 0 else 0)
            vals_123.append(v123[0] if len(v123) > 0 else 0)
        
        bars1 = ax.bar(x - width/2, vals_42, width, label="Seed 42", 
                       color=summary["color"].tolist(), alpha=0.9, edgecolor="black")
        bars2 = ax.bar(x + width/2, vals_123, width, label="Seed 123",
                       color=summary["color"].tolist(), alpha=0.6, edgecolor="black", hatch="//")
        
        ax.set_title(METRIC_LABELS.get(metric, metric.upper()), fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        
        if idx == 0:
            ax.legend(loc="upper right", fontsize=8)
    
    for idx in range(len(available_metrics), len(axes)):
        axes[idx].set_visible(False)
    
    split_label = SPLIT_LABELS.get(split, split.title())
    fig.suptitle(f"Metrics by Method - Seed 42 vs 123 ({split_label})", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def plot_recall_specificity_tradeoff(df: pd.DataFrame, output_path: Path, split: str = "test") -> None:
    """Plot recall vs specificity scatter to show tradeoff."""
    if df.empty or "recall" not in df.columns or "specificity" not in df.columns:
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for _, row in df.iterrows():
        marker = "o" if row["seed"] == "42" else "s"
        ax.scatter(row["specificity"], row["recall"], 
                  c=row["color"], s=150, marker=marker, alpha=0.8, edgecolors="black",
                  label=f"{row['label']} (s{row['seed']})")
    
    ax.set_xlabel("Specificity", fontsize=12)
    ax.set_ylabel("Recall", fontsize=12)
    split_label = SPLIT_LABELS.get(split, split.title())
    ax.set_title(f"Recall vs Specificity Tradeoff ({split_label})", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)
    
    # Add diagonal line
    ax.plot([0, 1], [1, 0], "k--", alpha=0.3, label="Tradeoff line")
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def create_summary_table(df: pd.DataFrame, output_path: Path, split: str = "test") -> None:
    """Create summary CSV with all metrics."""
    if df.empty:
        return
    
    # Select relevant columns
    cols = ["method", "label", "seed", "split", "model", "order"] + [m for m in METRICS if m in df.columns]
    cols = [c for c in cols if c in df.columns]
    summary = df[cols].copy()
    if "order" in summary.columns:
        summary = summary.sort_values(["order", "seed"]).drop(columns=["order"])
    else:
        summary = summary.sort_values(["method", "seed"])
    
    summary.to_csv(output_path, index=False, float_format="%.4f")
    logger.info(f"Saved: {output_path}")
    
    return summary


def plot_all_splits_comparison(all_dfs: Dict[str, pd.DataFrame], output_path: Path) -> None:
    """Create comparison plot showing Train/Val/Test for each metric."""
    if not all_dfs or all(df.empty for df in all_dfs.values()):
        return
    
    # Combine all splits
    combined = pd.concat([df.assign(split=split) for split, df in all_dfs.items()], ignore_index=True)
    
    # Get unique methods sorted by order
    methods = combined.drop_duplicates(subset=["method"]).sort_values("order")["method"].tolist()
    method_labels = {row["method"]: row["label"] for _, row in combined.drop_duplicates(subset=["method"]).iterrows()}
    method_colors = {row["method"]: row["color"] for _, row in combined.drop_duplicates(subset=["method"]).iterrows()}
    
    available_metrics = [m for m in METRICS if m in combined.columns and combined[m].notna().any()]
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    axes = axes.flatten()
    
    x = np.arange(len(methods))
    width = 0.25
    split_offsets = {"train": -width, "val": 0, "test": width}
    split_colors_alpha = {"train": 0.5, "val": 0.7, "test": 1.0}
    split_hatches = {"train": "//", "val": "..", "test": ""}
    
    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        
        for split_name, offset in split_offsets.items():
            df_split = all_dfs.get(split_name, pd.DataFrame())
            if df_split.empty:
                continue
            
            # Average over seeds for each method
            vals = []
            colors = []
            for method in methods:
                df_m = df_split[df_split["method"] == method]
                if not df_m.empty and metric in df_m.columns:
                    vals.append(df_m[metric].mean())
                else:
                    vals.append(0)
                colors.append(method_colors.get(method, "#999999"))
            
            bars = ax.bar(x + offset, vals, width, 
                         label=SPLIT_LABELS.get(split_name, split_name),
                         color=colors, alpha=split_colors_alpha[split_name],
                         edgecolor="black", hatch=split_hatches[split_name])
        
        ax.set_title(METRIC_LABELS.get(metric, metric.upper()), fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([method_labels[m] for m in methods], rotation=45, ha="right", fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        
        if idx == 0:
            ax.legend(loc="upper right", fontsize=8)
    
    for idx in range(len(available_metrics), len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle("Metrics Comparison: Train vs Validation vs Test", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize imbalance metrics comparison")
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR, help="Training results directory")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--split", type=str, default="all", choices=["train", "val", "test", "all"],
                        help="Which split to plot (train/val/test/all)")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find training files (contain train/val/test data)
    logger.info(f"Searching for training result files in {args.train_dir}")
    files = find_training_files(args.train_dir)
    logger.info(f"Found {len(files)} training result files")
    
    if not files:
        logger.error("No training result files found")
        return 1
    
    # Determine which splits to process
    splits_to_process = SPLITS if args.split == "all" else [args.split]
    all_dfs = {}
    
    for split in splits_to_process:
        logger.info(f"\nProcessing {split} split...")
        df = load_metrics_from_files(files, split=split)
        logger.info(f"Loaded metrics for {len(df)} experiments ({split})")
        
        if df.empty:
            logger.warning(f"No metrics data loaded for {split}")
            continue
        
        all_dfs[split] = df
        
        # Print summary
        split_label = SPLIT_LABELS.get(split, split.title())
        print("\n" + "="*80)
        print(f"Imbalance Experiment Metrics Summary - {split_label}")
        print("="*80)
        display_cols = ["label", "seed"] + [m for m in METRICS if m in df.columns]
        print(df.sort_values(["order", "seed"])[display_cols].to_string(index=False))
        print("="*80 + "\n")
        
        # Generate visualizations for this split
        plot_metrics_grid(df, output_dir / f"metrics_comparison_{split}.png", split=split)
        plot_metrics_grouped_by_method(df, output_dir / f"metrics_by_method_{split}.png", split=split)
        plot_recall_specificity_tradeoff(df, output_dir / f"recall_specificity_tradeoff_{split}.png", split=split)
        create_summary_table(df, output_dir / f"metrics_summary_{split}.csv", split=split)
    
    # If all splits processed, create comparison plot
    if args.split == "all" and len(all_dfs) > 1:
        plot_all_splits_comparison(all_dfs, output_dir / "metrics_train_val_test_comparison.png")
    
    logger.info("Visualization complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
