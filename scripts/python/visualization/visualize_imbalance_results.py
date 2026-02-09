#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_imbalance_results.py
==============================

Comprehensive visualization for imbalance experiment results.

Generates:
1. Sampling distribution - Class balance after SMOTE/oversampling
2. Confusion matrices - For train/val/test with metrics table
3. Metrics comparison - CSV and bar charts for all metrics
4. Optuna convergence - Hyperparameter optimization plots

Usage:
    python visualize_imbalance_results.py --experiments baseline,smote_0.1,smote_0.5,sw_smote_0.1,sw_smote_0.5
    python visualize_imbalance_results.py --job_id local --output results/figures/imbalance
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src import config as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Color palette for experiments
EXPERIMENT_COLORS = {
    "baseline": "#7f8c8d",
    "smote_0.1": "#3498db",
    "smote_0.5": "#2980b9",
    "sw_smote_0.1": "#e74c3c",
    "sw_smote_0.5": "#c0392b",
    "smote_ratio0.1": "#3498db",
    "smote_ratio0.5": "#2980b9",
    "subjectwise_smote_ratio0.1": "#e74c3c",
    "subjectwise_smote_ratio0.5": "#c0392b",
}

EXPERIMENT_LABELS = {
    # Seed 42
    "baseline_s42": "Baseline (s42)",
    "smote_ratio0.1_s42": "SMOTE 0.1 (s42)",
    "smote_ratio0.5_s42": "SMOTE 0.5 (s42)",
    "subjectwise_smote_ratio0.1_s42": "SW-SMOTE 0.1 (s42)",
    "subjectwise_smote_ratio0.5_s42": "SW-SMOTE 0.5 (s42)",
    # Seed 123
    "baseline_s123": "Baseline (s123)",
    "smote_ratio0.1_s123": "SMOTE 0.1 (s123)",
    "smote_ratio0.5_s123": "SMOTE 0.5 (s123)",
    "subjectwise_smote_ratio0.1_s123": "SW-SMOTE 0.1 (s123)",
    "subjectwise_smote_ratio0.5_s123": "SW-SMOTE 0.5 (s123)",
}


# ============================================================
# 1. Sampling Distribution Visualization
# ============================================================
def extract_sampling_info_from_logs(log_dir: Path, experiments: List[str]) -> pd.DataFrame:
    """Extract sampling distribution info from training logs."""
    data = []
    
    for exp in experiments:
        # Find log files (prefer recent ones with timestamp)
        pattern = f"*{exp}*.log"
        log_files = sorted(log_dir.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)
        
        for log_file in log_files:
            # Skip eval logs
            if "_eval_" in log_file.name:
                continue
            
            try:
                with open(log_file, "r") as f:
                    content = f.read()
                
                # Extract before/after sampling counts
                # Pattern: [split:random|time_stratify=False] train n=37052 pos=1454 (0.039)
                train_match = re.search(r"\[split.*\] train n=(\d+) pos=(\d+)", content)
                
                # New pattern: Class distribution before oversampling: [35598  1454]
                before_match = re.search(r"Class distribution before oversampling:\s*\[(\d+)\s+(\d+)\]", content)
                # New pattern: Class distribution after oversampling: [35598 17799]
                after_match = re.search(r"Class distribution after oversampling:\s*\[(\d+)\s+(\d+)\]", content)
                # Subject-wise pattern: Class distribution after: [35598 17777]
                sw_after_match = re.search(r"Class distribution after:\s*\[(\d+)\s+(\d+)\]", content)
                
                if train_match:
                    n_total = int(train_match.group(1))
                    n_pos = int(train_match.group(2))
                    n_neg = n_total - n_pos
                    
                    after_pos = n_pos
                    after_neg = n_neg
                    
                    # Use actual oversampling info if available
                    if before_match:
                        before_neg = int(before_match.group(1))
                        before_pos = int(before_match.group(2))
                        n_neg = before_neg
                        n_pos = before_pos
                        
                        if after_match:
                            after_neg = int(after_match.group(1))
                            after_pos = int(after_match.group(2))
                        elif sw_after_match:
                            # Subject-wise SMOTE uses different log format
                            after_neg = int(sw_after_match.group(1))
                            after_pos = int(sw_after_match.group(2))
                    
                    data.append({
                        "experiment": exp,
                        "label": EXPERIMENT_LABELS.get(exp, exp),
                        "before_pos": n_pos,
                        "before_neg": n_neg,
                        "before_total": n_pos + n_neg,
                        "before_ratio": n_pos / (n_pos + n_neg) if (n_pos + n_neg) > 0 else 0,
                        "after_pos": after_pos,
                        "after_neg": after_neg,
                        "after_total": after_pos + after_neg,
                        "after_ratio": after_pos / (after_pos + after_neg) if (after_pos + after_neg) > 0 else 0,
                    })
                    break  # Use first matching log
            except Exception as e:
                logger.warning(f"Failed to parse {log_file}: {e}")
    
    return pd.DataFrame(data)


def plot_sampling_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """Plot class distribution before/after sampling as horizontal bar chart."""
    if df.empty:
        logger.warning("No sampling data to plot")
        return
    
    # Sort: Baseline first, then others
    df = df.copy()
    df["is_baseline"] = df["experiment"].str.contains("baseline", case=False)
    df = df.sort_values(["is_baseline"], ascending=[False]).reset_index(drop=True)
    
    # Create single horizontal bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    experiments = df["label"].tolist()
    y = np.arange(len(experiments))
    height = 0.35
    
    # Horizontal bars: Negative (left side, positive values) and Positive (right side)
    bars_neg = ax.barh(y - height/2, df["after_neg"], height, label="Negative (Alert)", color="#3498db", alpha=0.8)
    bars_pos = ax.barh(y + height/2, df["after_pos"], height, label="Positive (Drowsy)", color="#e74c3c", alpha=0.8)
    
    ax.set_xlabel("Sample Count", fontsize=12)
    ax.set_ylabel("Experiment", fontsize=12)
    ax.set_title("Class Distribution After Sampling", fontsize=14, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(experiments, fontsize=10)
    ax.invert_yaxis()  # Baseline (y=0) at top
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    
    # Add ratio labels at the end of bars
    for i, row in df.iterrows():
        max_val = max(row["after_neg"], row["after_pos"])
        ax.annotate(f"ratio={row['after_ratio']:.3f}", 
                    xy=(max_val + max_val * 0.02, i),
                    ha="left", va="center", fontsize=9, fontweight="bold")
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def load_training_results(results_dir: Path, experiments: List[str], model: str = "RF", mode: str = "pooled") -> Dict[str, dict]:
    """Load training results (train/val/test) for all experiments."""
    results = {}
    
    for exp in experiments:
        # Try different filename patterns
        patterns = [
            f"train_results_{model}_{mode}_{exp}.json",
        ]
        
        for pattern in patterns:
            json_path = results_dir / pattern
            if json_path.exists():
                with open(json_path, "r") as f:
                    results[exp] = json.load(f)
                break
    
    return results


def extract_metrics_by_split(results: Dict[str, dict], split: str = "test") -> pd.DataFrame:
    """Extract metrics from training results for a specific split (train/val/test)."""
    metrics_list = []
    
    for exp, data in results.items():
        split_data = data.get(split, {})
        if not split_data:
            continue
        
        # Extract TP, TN, FP, FN from confusion matrix
        cm = split_data.get("confusion_matrix")
        if cm is not None and len(cm) == 2 and len(cm[0]) == 2:
            tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
            # Calculate specificity if not provided: TN / (TN + FP)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else None
            # Calculate F2 if not provided: (1 + 2^2) * precision * recall / (2^2 * precision + recall)
            precision = split_data.get("precision", 0)
            recall = split_data.get("recall", 0)
            if precision and recall and (4 * precision + recall) > 0:
                f2 = 5 * precision * recall / (4 * precision + recall)
            else:
                f2 = None
        else:
            tn, fp, fn, tp = None, None, None, None
            specificity = None
            f2 = None
        
        metrics = {
            "experiment": exp,
            "label": EXPERIMENT_LABELS.get(exp, exp),
            "accuracy": split_data.get("accuracy", split_data.get("acc_thr")),
            "precision": split_data.get("prec_thr", split_data.get("precision")),
            "recall": split_data.get("recall_thr", split_data.get("recall")),
            "specificity": split_data.get("specificity_thr", split_data.get("specificity", specificity)),
            "f1": split_data.get("f1_thr", split_data.get("f1")),
            "f2": split_data.get("f2_thr", split_data.get("f2", f2)),
            "roc_auc": split_data.get("roc_auc"),
            "auc_pr": split_data.get("auc_pr"),
            "threshold": split_data.get("thr"),
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
        }
        metrics_list.append(metrics)
    
    return pd.DataFrame(metrics_list)


def plot_metrics_comparison_by_split(df: pd.DataFrame, output_path: Path, split_name: str = "Test") -> None:
    """Plot bar chart comparing metrics across experiments for a specific split."""
    if df.empty:
        logger.warning(f"No metrics to plot for {split_name}")
        return
    
    metrics = ["accuracy", "precision", "recall", "specificity", "f1", "f2", "roc_auc", "auc_pr"]
    metrics = [m for m in metrics if m in df.columns and df[m].notna().any()]
    
    # Reorder experiments: group by method (baseline, smote0.1, ...), then by seed
    method_order = ["baseline", "smote_ratio0.1", "smote_ratio0.5", 
                    "subjectwise_smote_ratio0.1", "subjectwise_smote_ratio0.5"]
    seed_order = ["s42", "s123"]
    
    ordered_exps = []
    for method in method_order:
        for seed in seed_order:
            exp_name = f"{method}_{seed}"
            if exp_name in df["experiment"].values:
                ordered_exps.append(exp_name)
    
    # Add any remaining experiments not in the order
    for exp in df["experiment"].values:
        if exp not in ordered_exps:
            ordered_exps.append(exp)
    
    # Reorder dataframe
    df = df.set_index("experiment").loc[ordered_exps].reset_index()
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    experiments = df["label"].tolist()
    x = np.arange(len(experiments))
    
    # Color by method: baseline=red, smote=blue shades, sw-smote=green shades
    # Pair s42/s123 with similar colors (solid vs hatched)
    method_colors = {
        "baseline": "#e74c3c",           # red
        "smote_ratio0.1": "#3498db",     # blue
        "smote_ratio0.5": "#2980b9",     # dark blue
        "subjectwise_smote_ratio0.1": "#27ae60",  # green
        "subjectwise_smote_ratio0.5": "#1e8449",  # dark green
    }
    colors = []
    hatches = []
    for exp in df["experiment"].values:
        method = exp.replace("_s42", "").replace("_s123", "")
        colors.append(method_colors.get(method, "#7f8c8d"))
        hatches.append("" if "_s42" in exp else "//")
    
    for ax, metric in zip(axes, metrics):
        values = df[metric].fillna(0).values
        bars = ax.bar(x, values, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
        
        # Apply hatching for s123
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        
        ax.set_title(metric.upper().replace("_", " "), fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(experiments, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, max(values) * 1.15 if max(values) > 0 else 1)
        ax.grid(axis="y", alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, values):
            if val > 0:
                ax.annotate(f"{val:.3f}", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           ha="center", va="bottom", fontsize=8, rotation=0)
    
    # Hide unused axes
    for ax in axes[len(metrics):]:
        ax.set_visible(False)
    
    fig.suptitle(f"Metrics Comparison ({split_name} Set)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


# ============================================================
# 2. Confusion Matrix Visualization
# ============================================================
def load_evaluation_results(eval_dir: Path, experiments: List[str]) -> Dict[str, dict]:
    """Load evaluation results for all experiments."""
    results = {}
    
    for exp in experiments:
        # Try different filename patterns
        patterns = [
            f"eval_results_RF_pooled_{exp}.json",
            f"eval_results_RF_pooled_{exp}_s42.json",
        ]
        
        for pattern in patterns:
            json_path = eval_dir / pattern
            if json_path.exists():
                with open(json_path, "r") as f:
                    results[exp] = json.load(f)
                break
    
    return results


def plot_confusion_matrices(results: Dict[str, dict], output_path: Path) -> None:
    """Plot confusion matrices for all experiments.
    
    Layout: 5 columns (experiment types) x 2 rows (seeds)
    """
    n_experiments = len(results)
    if n_experiments == 0:
        logger.warning("No results to plot confusion matrices")
        return
    
    # Group experiments by method (without seed suffix)
    methods = ["baseline", "smote_ratio0.1", "smote_ratio0.5", 
               "subjectwise_smote_ratio0.1", "subjectwise_smote_ratio0.5"]
    seeds = ["s42", "s123"]
    
    method_labels = {
        "baseline": "Baseline",
        "smote_ratio0.1": "SMOTE 0.1",
        "smote_ratio0.5": "SMOTE 0.5",
        "subjectwise_smote_ratio0.1": "SW-SMOTE 0.1",
        "subjectwise_smote_ratio0.5": "SW-SMOTE 0.5",
    }
    
    n_cols = len(methods)
    n_rows = len(seeds)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    
    for row, seed in enumerate(seeds):
        for col, method in enumerate(methods):
            ax = axes[row, col]
            exp_key = f"{method}_{seed}"
            
            if exp_key in results:
                data = results[exp_key]
                cm = data.get("confusion_matrix")
                if cm is not None:
                    cm = np.array(cm)
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                                xticklabels=["Alert", "Drowsy"],
                                yticklabels=["Alert", "Drowsy"])
                else:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center")
            else:
                ax.text(0.5, 0.5, "Not available", ha="center", va="center", fontsize=10, color="gray")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
            
            # Set title for top row only
            if row == 0:
                ax.set_title(method_labels.get(method, method), fontsize=11, fontweight="bold")
            
            # Set ylabel for leftmost column only
            if col == 0:
                ax.set_ylabel(f"Seed {seed.replace('s', '')}\nActual", fontsize=10)
            else:
                ax.set_ylabel("")
            
            # Set xlabel for bottom row only
            if row == n_rows - 1:
                ax.set_xlabel("Predicted", fontsize=10)
            else:
                ax.set_xlabel("")
    
    fig.suptitle("Confusion Matrices (Test Set)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


# ============================================================
# 3. Metrics Comparison
# ============================================================
def extract_metrics(results: Dict[str, dict]) -> pd.DataFrame:
    """Extract metrics from evaluation results into DataFrame."""
    metrics_list = []
    
    for exp, data in results.items():
        # Extract TP, TN, FP, FN from confusion matrix
        cm = data.get("confusion_matrix")
        if cm is not None and len(cm) == 2 and len(cm[0]) == 2:
            tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        else:
            tn, fp, fn, tp = None, None, None, None
        
        metrics = {
            "experiment": exp,
            "label": EXPERIMENT_LABELS.get(exp, exp),
            "accuracy": data.get("accuracy", data.get("acc_thr")),
            "precision": data.get("prec_thr", data.get("precision")),
            "recall": data.get("recall_thr", data.get("recall")),
            "specificity": data.get("specificity_thr", data.get("specificity")),
            "f1": data.get("f1_thr", data.get("f1")),
            "f2": data.get("f2_thr"),
            "roc_auc": data.get("roc_auc"),
            "auc_pr": data.get("auc_pr"),
            "threshold": data.get("thr"),
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
        }
        metrics_list.append(metrics)
    
    return pd.DataFrame(metrics_list)


def save_metrics_csv(df: pd.DataFrame, output_path: Path) -> None:
    """Save metrics to CSV."""
    df.to_csv(output_path, index=False, float_format="%.4f")
    logger.info(f"Saved: {output_path}")


def plot_metrics_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """Plot bar chart comparing metrics across experiments."""
    if df.empty:
        logger.warning("No metrics to plot")
        return
    
    metrics = ["accuracy", "precision", "recall", "specificity", "f1", "f2", "roc_auc", "auc_pr"]
    metrics = [m for m in metrics if m in df.columns and df[m].notna().any()]
    
    # Reorder experiments: group by method (baseline, smote0.1, ...), then by seed
    method_order = ["baseline", "smote_ratio0.1", "smote_ratio0.5", 
                    "subjectwise_smote_ratio0.1", "subjectwise_smote_ratio0.5"]
    seed_order = ["s42", "s123"]
    
    ordered_exps = []
    for method in method_order:
        for seed in seed_order:
            exp_name = f"{method}_{seed}"
            if exp_name in df["experiment"].values:
                ordered_exps.append(exp_name)
    
    # Add any remaining experiments not in the order
    for exp in df["experiment"].values:
        if exp not in ordered_exps:
            ordered_exps.append(exp)
    
    # Reorder dataframe
    df = df.set_index("experiment").loc[ordered_exps].reset_index()
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    experiments = df["label"].tolist()
    x = np.arange(len(experiments))
    
    # Color by method: baseline=red, smote=blue shades, sw-smote=green shades
    # Pair s42/s123 with similar colors (solid vs hatched)
    method_colors = {
        "baseline": "#e74c3c",           # red
        "smote_ratio0.1": "#3498db",     # blue
        "smote_ratio0.5": "#2980b9",     # dark blue
        "subjectwise_smote_ratio0.1": "#27ae60",  # green
        "subjectwise_smote_ratio0.5": "#1e8449",  # dark green
    }
    colors = []
    hatches = []
    for exp in df["experiment"].values:
        method = exp.replace("_s42", "").replace("_s123", "")
        colors.append(method_colors.get(method, "#7f8c8d"))
        hatches.append("" if "_s42" in exp else "//")
    
    for ax, metric in zip(axes, metrics):
        values = df[metric].fillna(0).values
        bars = ax.bar(x, values, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
        
        # Apply hatching for s123
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        
        ax.set_title(metric.upper().replace("_", " "), fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(experiments, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, max(values) * 1.15 if max(values) > 0 else 1)
        ax.grid(axis="y", alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, values):
            if val > 0:
                ax.annotate(f"{val:.3f}", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           ha="center", va="bottom", fontsize=8, rotation=0)
    
    # Hide unused axes
    for ax in axes[len(metrics):]:
        ax.set_visible(False)
    
    fig.suptitle("Metrics Comparison (Test Set)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


# ============================================================
# 3b. ROC and PR Curves
# ============================================================
def plot_roc_curves(results: Dict[str, dict], output_path: Path) -> None:
    """Plot ROC curves for all experiments on single plot."""
    if not results:
        logger.warning("No results to plot ROC curves")
        return
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    for exp, data in results.items():
        roc = data.get("roc_curve")
        if roc is None:
            continue
        
        fpr = roc.get("fpr", [])
        tpr = roc.get("tpr", [])
        auc_val = roc.get("auc", data.get("roc_auc", 0))
        
        if not fpr or not tpr:
            continue
        
        label = EXPERIMENT_LABELS.get(exp, exp)
        color = "#e74c3c" if "baseline" in exp.lower() else "#7f8c8d"
        linewidth = 2.5 if "baseline" in exp.lower() else 1.5
        
        ax.plot(fpr, tpr, label=f"{label} (AUC={auc_val:.4f})", 
                color=color, linewidth=linewidth, alpha=0.8)
    
    # Diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label="Random")
    
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves (Test Set)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def plot_pr_curves(results: Dict[str, dict], output_path: Path) -> None:
    """Plot Precision-Recall curves for all experiments on single plot."""
    if not results:
        logger.warning("No results to plot PR curves")
        return
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    for exp, data in results.items():
        pr = data.get("pr_curve")
        if pr is None:
            continue
        
        precision = pr.get("precision", [])
        recall = pr.get("recall", [])
        auc_pr = pr.get("auc_pr", data.get("auc_pr", 0))
        
        if not precision or not recall:
            continue
        
        label = EXPERIMENT_LABELS.get(exp, exp)
        color = "#e74c3c" if "baseline" in exp.lower() else "#7f8c8d"
        linewidth = 2.5 if "baseline" in exp.lower() else 1.5
        
        ax.plot(recall, precision, label=f"{label} (AUC={auc_pr:.4f})", 
                color=color, linewidth=linewidth, alpha=0.8)
    
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves (Test Set)", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


# ============================================================
# 4. Optuna Convergence Visualization
# ============================================================
def load_optuna_results(models_dir: Path, experiments: List[str], model: str = "RF", mode: str = "pooled") -> Dict[str, Tuple[pd.DataFrame, dict]]:
    """Load Optuna trials and convergence data."""
    results = {}
    
    for exp in experiments:
        # Extract seed from experiment name (e.g., baseline_s42 -> s42)
        seed_suffix = "_s42" if "_s42" in exp else "_s123" if "_s123" in exp else ""
        
        # Find trials CSV - try multiple patterns
        # Actual filename: optuna_RF_pooled__pooled_baseline_s42_s42_trials.csv (seed duplicated)
        patterns = [
            f"optuna_{model}_{mode}__{mode}_{exp}{seed_suffix}_trials.csv",
            f"optuna_{model}_{mode}__{mode}_{exp}_trials.csv",
        ]
        
        trials_df = None
        convergence = None
        
        for pattern in patterns:
            csv_path = models_dir / pattern
            if csv_path.exists():
                trials_df = pd.read_csv(csv_path)
                break
        
        # Find convergence JSON
        for pattern in patterns:
            json_path = models_dir / pattern.replace("_trials.csv", "_convergence.json")
            if json_path.exists():
                with open(json_path, "r") as f:
                    convergence = json.load(f)
                break
        
        if trials_df is not None:
            results[exp] = (trials_df, convergence)
    
    return results


def plot_optuna_convergence(optuna_results: Dict[str, Tuple[pd.DataFrame, dict]], output_path: Path) -> None:
    """Plot Optuna optimization convergence, split by seed."""
    if not optuna_results:
        logger.warning("No Optuna results to plot")
        return
    
    # Split experiments by seed
    seed_42_exps = {k: v for k, v in optuna_results.items() if "_s42" in k}
    seed_123_exps = {k: v for k, v in optuna_results.items() if "_s123" in k}
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Define distinct colors and markers for each method (5 methods)
    method_colors = {
        "baseline": "#e74c3c",      # red
        "smote_ratio0.1": "#3498db", # blue
        "smote_ratio0.5": "#2ecc71", # green
        "subjectwise_smote_ratio0.1": "#9b59b6", # purple
        "subjectwise_smote_ratio0.5": "#f39c12", # orange
    }
    method_markers = {
        "baseline": "o",
        "smote_ratio0.1": "s",
        "smote_ratio0.5": "^",
        "subjectwise_smote_ratio0.1": "D",
        "subjectwise_smote_ratio0.5": "v",
    }
    
    def get_method(exp_name):
        """Extract method name from experiment name (remove seed suffix)."""
        return exp_name.replace("_s42", "").replace("_s123", "")
    
    # Plot seed=42
    ax1 = axes[0]
    for exp, (df, meta) in seed_42_exps.items():
        method = get_method(exp)
        label = EXPERIMENT_LABELS.get(exp, exp)
        color = method_colors.get(method, "#95a5a6")
        marker = method_markers.get(method, "o")
        
        trials = df["number"].values
        values = df["value"].values
        
        ax1.plot(trials, values, marker=marker, linestyle="-", label=label, color=color, markersize=6, alpha=0.8)
        best_values = np.maximum.accumulate(values)
        ax1.plot(trials, best_values, "--", color=color, alpha=0.5, linewidth=1)
    
    ax1.set_xlabel("Trial", fontsize=11)
    ax1.set_ylabel("Objective Value (F2)", fontsize=11)
    ax1.set_title("Optimization Convergence (Seed=42)", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Plot seed=123
    ax2 = axes[1]
    for exp, (df, meta) in seed_123_exps.items():
        method = get_method(exp)
        label = EXPERIMENT_LABELS.get(exp, exp)
        color = method_colors.get(method, "#95a5a6")
        marker = method_markers.get(method, "o")
        
        trials = df["number"].values
        values = df["value"].values
        
        ax2.plot(trials, values, marker=marker, linestyle="-", label=label, color=color, markersize=6, alpha=0.8)
        best_values = np.maximum.accumulate(values)
        ax2.plot(trials, best_values, "--", color=color, alpha=0.5, linewidth=1)
    
    ax2.set_xlabel("Trial", fontsize=11)
    ax2.set_ylabel("Objective Value (F2)", fontsize=11)
    ax2.set_title("Optimization Convergence (Seed=123)", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.suptitle("Optuna Optimization Convergence by Seed", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def plot_hyperparameter_trials(optuna_results: Dict[str, Tuple[pd.DataFrame, dict]], output_dir: Path) -> None:
    """Plot hyperparameter values over trials for all parameters.
    
    Creates one figure per experiment showing how each hyperparameter was explored.
    """
    if not optuna_results:
        return
    
    # All hyperparameters to visualize
    all_params = [
        "params_max_depth",
        "params_n_estimators", 
        "params_min_samples_split",
        "params_min_samples_leaf",
        "params_max_features",
        "params_max_samples",
        "params_min_weight_fraction_leaf",
        "params_class_weight",
    ]
    
    for exp, (df, meta) in optuna_results.items():
        label = EXPERIMENT_LABELS.get(exp, exp)
        
        # Filter to available params
        available_params = [p for p in all_params if p in df.columns]
        if not available_params:
            continue
        
        n_params = len(available_params)
        n_cols = 2
        n_rows = (n_params + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
        axes = axes.flatten() if n_params > 1 else [axes]
        
        trials = df["number"].values
        values = df["value"].values
        
        for ax, param in zip(axes[:n_params], available_params):
            param_name = param.replace("params_", "")
            param_values = df[param].values
            
            # Handle categorical/string parameters
            if df[param].dtype == object or param == "params_class_weight":
                # Map categories to numeric for plotting
                unique_vals = df[param].dropna().unique()
                val_map = {v: i for i, v in enumerate(unique_vals)}
                numeric_vals = [val_map.get(v, -1) for v in param_values]
                
                ax.scatter(trials, numeric_vals, c=values, cmap="viridis", s=80, edgecolors="black", linewidth=0.5)
                ax.set_yticks(list(val_map.values()))
                ax.set_yticklabels([str(v)[:15] for v in val_map.keys()], fontsize=8)
            else:
                # Numeric parameters
                numeric_vals = pd.to_numeric(param_values, errors="coerce")
                scatter = ax.scatter(trials, numeric_vals, c=values, cmap="viridis", s=80, edgecolors="black", linewidth=0.5)
            
            ax.set_xlabel("Trial")
            ax.set_ylabel(param_name)
            ax.set_title(param_name, fontsize=11, fontweight="bold")
            ax.grid(alpha=0.3)
        
        # Hide unused axes
        for ax in axes[n_params:]:
            ax.set_visible(False)
        
        # Add colorbar with manual positioning to avoid overlap
        if n_params > 0:
            # Create a new axis for colorbar on the far right
            cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
            cbar = fig.colorbar(scatter, cax=cbar_ax, label="Objective (F2)")
        
        fig.suptitle(f"Hyperparameter Exploration: {label}", fontsize=14, fontweight="bold", y=1.02)
        fig.tight_layout(rect=[0, 0, 0.90, 1])  # Leave more space on right for colorbar
        
        # Save per-experiment file
        safe_name = exp.replace("/", "_").replace(" ", "_")
        fig.savefig(output_dir / f"hp_trials_{safe_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {output_dir / f'hp_trials_{safe_name}.png'}")


# ============================================================
# Main Pipeline
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Visualize imbalance experiment results")
    parser.add_argument("--experiments", type=str, 
                        default="baseline_s42,smote_ratio0.1_s42,smote_ratio0.5_s42,subjectwise_smote_ratio0.1_s42,subjectwise_smote_ratio0.5_s42,baseline_s123,smote_ratio0.1_s123,smote_ratio0.5_s123,subjectwise_smote_ratio0.1_s123,subjectwise_smote_ratio0.5_s123",
                        help="Comma-separated experiment tags")
    parser.add_argument("--job_id", type=str, default="local",
                        help="Job ID (default: local)")
    parser.add_argument("--model", type=str, default="RF",
                        help="Model name (default: RF)")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Training log directory")
    args = parser.parse_args()
    
    experiments = [e.strip() for e in args.experiments.split(",")]
    job_id = args.job_id
    model = args.model
    
    # Setup paths
    # Output structure: results/analysis/exp1_imbalance/{job_id}/
    #   - sampling/       : sampling distribution plots and data
    #   - metrics/        : metrics comparison (CSV and plots)
    #   - confusion/      : confusion matrices
    #   - optuna/         : hyperparameter optimization plots
    base_output = Path(cfg.RESULTS_IMBALANCE_PATH) / job_id
    sampling_dir = base_output / "sampling"
    metrics_dir = base_output / "metrics"
    confusion_dir = base_output / "confusion"
    optuna_dir = base_output / "optuna"
    
    for d in [sampling_dir, metrics_dir, confusion_dir, optuna_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    eval_dir = Path(cfg.RESULTS_EVALUATION_PATH) / model / job_id / f"{job_id}[1]"
    models_dir = Path(cfg.MODEL_PKL_PATH) / model / job_id
    log_dir = Path(args.log_dir) if args.log_dir else Path("scripts/local/logs/imbalance")
    
    logger.info(f"Experiments: {experiments}")
    logger.info(f"Evaluation dir: {eval_dir}")
    logger.info(f"Models dir: {models_dir}")
    logger.info(f"Output base: {base_output}")
    
    # 1. Sampling Distribution
    logger.info("\n=== 1. Sampling Distribution ===")
    sampling_df = extract_sampling_info_from_logs(log_dir, experiments)
    if not sampling_df.empty:
        plot_sampling_distribution(sampling_df, sampling_dir / "sampling_distribution.png")
        sampling_df.to_csv(sampling_dir / "sampling_distribution.csv", index=False)
    else:
        logger.warning("Could not extract sampling information from logs")
    
    # 2. Load evaluation results
    logger.info("\n=== 2. Confusion Matrices ===")
    results = load_evaluation_results(eval_dir, experiments)
    if results:
        plot_confusion_matrices(results, confusion_dir / "confusion_matrices.png")
    else:
        logger.warning(f"No evaluation results found in {eval_dir}")
    
    # 3. Metrics comparison
    logger.info("\n=== 3. Metrics Comparison ===")
    if results:
        metrics_df = extract_metrics(results)
        save_metrics_csv(metrics_df, metrics_dir / "metrics_comparison.csv")
        plot_metrics_comparison(metrics_df, metrics_dir / "metrics_comparison.png")
    
    # 3a. Train/Val/Test Metrics comparison from training results
    logger.info("\n=== 3a. Train/Val/Test Metrics Comparison ===")
    train_results_dir = Path(cfg.RESULTS_OUTPUTS_TRAINING_PATH) / model / job_id / f"{job_id}[1]"
    train_results = load_training_results(train_results_dir, experiments, model=model, mode="pooled")
    if train_results:
        for split in ["train", "val", "test"]:
            split_df = extract_metrics_by_split(train_results, split=split)
            if not split_df.empty:
                split_name = split.capitalize()
                save_metrics_csv(split_df, metrics_dir / f"metrics_comparison_{split}.csv")
                plot_metrics_comparison_by_split(split_df, metrics_dir / f"metrics_comparison_{split}.png", split_name=split_name)
    else:
        logger.warning(f"No training results found in {train_results_dir}")
    
    # 3b. ROC and PR Curves
    logger.info("\n=== 3b. ROC and PR Curves ===")
    if results:
        plot_roc_curves(results, metrics_dir / "roc_curves.png")
        plot_pr_curves(results, metrics_dir / "pr_curves.png")
    
    # 4. Optuna convergence
    logger.info("\n=== 4. Optuna Convergence ===")
    optuna_results = load_optuna_results(models_dir, experiments, model=model, mode="pooled")
    if optuna_results:
        plot_optuna_convergence(optuna_results, optuna_dir / "optuna_convergence.png")
        plot_hyperparameter_trials(optuna_results, optuna_dir)
    else:
        logger.warning(f"No Optuna results found in {models_dir}")
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("VISUALIZATION COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Output directory: {base_output}")
    logger.info("Generated files:")
    for subdir in [sampling_dir, metrics_dir, confusion_dir, optuna_dir]:
        for f in sorted(subdir.glob("*")):
            logger.info(f"  - {f.relative_to(base_output)}")


if __name__ == "__main__":
    main()
