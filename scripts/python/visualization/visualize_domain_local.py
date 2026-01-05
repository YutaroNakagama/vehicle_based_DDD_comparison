#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_domain_local.py
=========================

Visualization tool for local domain analysis experiments with imbalance handling.

Generates:
1. 4x7 metrics comparison grid (DTW, MMD, Wasserstein, Pooled × 7 metrics including AUROC)
2. RF hyperparameter convergence plots per experiment
3. Optuna objective function convergence
4. Confusion matrices for each experiment

Usage:
    python visualize_domain_local.py --job_id local --patterns "smote_plain,baseline_domain,imbalv3"
    python visualize_domain_local.py --job_id local --patterns "smote_plain" --seed 42
"""

import argparse
import glob
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
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src import config as cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
DISTANCES = ["dtw", "mmd", "wasserstein"]
DOMAINS = ["out_domain", "mid_domain", "in_domain"]  # Order: Out -> Mid -> In
MODES = ["source_only", "target_only"]
METRICS = ["accuracy", "precision", "recall", "specificity", "f1", "f2", "roc_auc", "auc_pr"]
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision", 
    "recall": "Recall",
    "specificity": "Specificity",
    "f1": "F1",
    "f2": "F2",
    "roc_auc": "AUROC",
    "auc_pr": "AUPRC"
}

# Pooled experiment patterns mapping to methods (5 patterns)
POOLED_PATTERNS = {
    "Baseline": "pooled_baseline",
    "SW-SMOTE r=0.1": "pooled_subjectwise_smote_ratio0.1",
    "SW-SMOTE r=0.5": "pooled_subjectwise_smote_ratio0.5",
    "Plain SMOTE r=0.1": "pooled_smote_ratio0.1",
    "Plain SMOTE r=0.5": "pooled_smote_ratio0.5"
}


def load_evaluation_results(eval_dir: Path, pattern: str, seed: Optional[int] = None) -> Dict[str, dict]:
    """Load evaluation results matching pattern."""
    results = {}
    
    # Search pattern
    if seed:
        search_pattern = f"eval_results_*{pattern}*_s{seed}*.json"
    else:
        search_pattern = f"eval_results_*{pattern}*.json"
    
    for json_file in eval_dir.glob(f"**/{search_pattern}"):
        try:
            with open(json_file) as f:
                data = json.load(f)
            
            # Extract key from filename
            basename = json_file.stem.replace("eval_results_", "")
            results[basename] = data
            
        except Exception as e:
            logger.warning(f"Failed to load {json_file}: {e}")
    
    logger.info(f"Loaded {len(results)} evaluation results for pattern '{pattern}'")
    return results


def load_pooled_results(eval_dir: Path, method: str) -> Optional[dict]:
    """Load pooled experiment results for a specific method.
    
    Args:
        eval_dir: Evaluation results directory
        method: One of 'Baseline', 'SW-SMOTE', 'Plain SMOTE'
    
    Returns:
        Average metrics across seeds, or None if not found.
    """
    pattern = POOLED_PATTERNS.get(method)
    if not pattern:
        return None
    
    # Search for pooled results (excluding th05 threshold variants)
    search_pattern = f"eval_results_*{pattern}*.json"
    results = []
    
    for json_file in eval_dir.glob(f"**/{search_pattern}"):
        # Skip threshold variants
        if "_th05" in json_file.stem:
            continue
        try:
            with open(json_file) as f:
                data = json.load(f)
            results.append(data)
        except Exception as e:
            logger.warning(f"Failed to load pooled result {json_file}: {e}")
    
    if not results:
        logger.warning(f"No pooled results found for pattern '{pattern}'")
        return None
    
    # Average metrics across seeds
    avg_result = {}
    metrics_to_avg = ["accuracy", "precision", "recall", "f1", "roc_auc", "auc_pr"]
    
    for metric in metrics_to_avg:
        values = [r.get(metric) for r in results if r.get(metric) is not None]
        if values:
            avg_result[metric] = sum(values) / len(values)
    
    # Calculate F2 from confusion matrices
    cms = [r.get("confusion_matrix") for r in results if r.get("confusion_matrix")]
    if cms:
        # Sum confusion matrices
        total_cm = np.zeros((2, 2))
        for cm in cms:
            total_cm += np.array(cm)
        tn, fp, fn, tp = total_cm[0, 0], total_cm[0, 1], total_cm[1, 0], total_cm[1, 1]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        if precision + recall > 0:
            avg_result["f2"] = (1 + 4) * (precision * recall) / ((4 * precision) + recall)
        else:
            avg_result["f2"] = 0
        avg_result["confusion_matrix"] = total_cm.tolist()
    
    logger.info(f"Loaded {len(results)} pooled results for method '{method}'")
    return avg_result


def load_optuna_results(models_dir: Path, pattern: str, seed: Optional[int] = None) -> Dict[str, Tuple[pd.DataFrame, dict]]:
    """Load Optuna trial CSVs and convergence JSONs."""
    results = {}
    
    if seed:
        search_pattern = f"*{pattern}*_s{seed}*_trials.csv"
    else:
        search_pattern = f"*{pattern}*_trials.csv"
    
    for csv_file in models_dir.glob(f"**/{search_pattern}"):
        try:
            df = pd.read_csv(csv_file)
            
            # Look for corresponding convergence JSON
            convergence_file = str(csv_file).replace("_trials.csv", "_convergence.json")
            convergence_data = {}
            if os.path.exists(convergence_file):
                with open(convergence_file) as f:
                    convergence_data = json.load(f)
            
            key = csv_file.stem.replace("optuna_RF_", "").replace("_trials", "")
            results[key] = (df, convergence_data)
            
        except Exception as e:
            logger.warning(f"Failed to load {csv_file}: {e}")
    
    logger.info(f"Loaded {len(results)} Optuna results for pattern '{pattern}'")
    return results


def parse_experiment_key(key: str) -> dict:
    """Parse experiment key to extract distance, domain, mode, etc."""
    info = {
        "distance": None,
        "domain": None,
        "mode": None,
        "method": None,
        "seed": None
    }
    
    # Extract distance
    for dist in DISTANCES:
        if dist in key:
            info["distance"] = dist
            break
    
    # Extract domain
    for dom in DOMAINS:
        if dom in key:
            info["domain"] = dom
            break
    
    # Extract mode
    for mode in MODES:
        if mode in key:
            info["mode"] = mode
            break
    
    # Extract seed
    seed_match = re.search(r"_s(\d+)", key)
    if seed_match:
        info["seed"] = int(seed_match.group(1))
    
    # Extract method with ratio (5 patterns)
    ratio_match = re.search(r"ratio(0\.[0-9]+)", key)
    ratio = ratio_match.group(1) if ratio_match else None
    
    if "smote_plain" in key:
        if ratio:
            info["method"] = f"Plain SMOTE r={ratio}"
        else:
            info["method"] = "Plain SMOTE"
    elif "baseline_domain" in key:
        info["method"] = "Baseline"
    elif "imbalv3" in key:
        if ratio:
            info["method"] = f"SW-SMOTE r={ratio}"
        else:
            info["method"] = "SW-SMOTE"
    else:
        info["method"] = "Unknown"
    
    return info


def plot_4x7_metrics_grid(results: Dict[str, dict], output_path: Path, title: str = "Domain Analysis Metrics", pooled_result: Optional[dict] = None) -> None:
    """
    Create 4x7 grid: rows = DTW/MMD/Wasserstein/Pooled, cols = metrics.
    Each cell shows bar chart comparing source_only vs target_only for in/mid/out domains.
    For Pooled row, shows single bar from pooled experiment results.
    """
    metrics = ["accuracy", "precision", "recall", "f1", "f2", "roc_auc", "auc_pr"]
    distances = ["dtw", "mmd", "wasserstein"]
    
    fig, axes = plt.subplots(4, 7, figsize=(24, 14))
    
    # Color scheme for modes (source_only = blue, target_only = orange)
    mode_colors = {
        "source_only": "#3498db",  # Blue
        "target_only": "#e67e22"   # Orange
    }
    
    # Pre-calculate pooled values for each metric to use as reference lines
    pooled_values = {}
    if pooled_result is not None:
        for metric in metrics:
            value = pooled_result.get(metric)
            if value is None and metric == "roc_auc":
                value = pooled_result.get("roc_auc_score") or pooled_result.get("auc")
            pooled_values[metric] = value
    
    # Process results by distance
    for row_idx, distance in enumerate(distances + ["pooled"]):
        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            
            if distance == "pooled":
                # Show pooled experiment results
                if pooled_result is None:
                    ax.text(0.5, 0.5, "N/A\n(No Pooled Data)", ha="center", va="center", fontsize=10, color="gray")
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.axis("off")
                else:
                    # Get value for this metric
                    value = pooled_values.get(metric)
                    
                    if value is not None:
                        # Single thin bar for pooled (green)
                        bar = ax.bar([0], [value], width=0.25, color="#27ae60", alpha=0.9)  # Green, thinner
                        ax.text(0, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
                        ax.set_xticks([0])
                        ax.set_xticklabels(["Pooled"], fontsize=8)
                        ax.set_ylim(0, 1.0)
                        ax.grid(axis="y", alpha=0.3)
                    else:
                        ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=10, color="gray")
                        ax.set_xlim(0, 1)
                        ax.set_ylim(0, 1)
                
                # Labels for pooled row
                if col_idx == 0:
                    ax.set_ylabel("POOLED", fontweight="bold", fontsize=11)
                continue
            
            # Collect data for this distance/metric
            data = []
            for key, result in results.items():
                info = parse_experiment_key(key)
                if info["distance"] != distance:
                    continue
                
                value = result.get(metric) or result.get(metric.replace("_", ""))
                if value is None and metric == "roc_auc":
                    value = result.get("roc_auc_score") or result.get("auc")
                
                # Calculate F2 from confusion matrix if not present
                if value is None and metric == "f2":
                    cm = result.get("confusion_matrix")
                    if cm is not None and len(cm) == 2 and len(cm[0]) == 2:
                        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                        beta = 2
                        if precision + recall > 0:
                            value = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
                        else:
                            value = 0
                
                if value is not None:
                    data.append({
                        "domain": info["domain"],
                        "mode": info["mode"],
                        "value": value,
                        "method": info["method"]
                    })
            
            if not data:
                ax.text(0.5, 0.5, "No Data", ha="center", va="center", fontsize=10, color="gray")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                continue
            
            df = pd.DataFrame(data)
            
            # Create grouped bar chart
            x = np.arange(len(DOMAINS))
            width = 0.35
            
            for i, mode in enumerate(MODES):
                mode_df = df[df["mode"] == mode]
                values = []
                for domain in DOMAINS:
                    domain_df = mode_df[mode_df["domain"] == domain]
                    if len(domain_df) > 0:
                        values.append(domain_df["value"].mean())
                    else:
                        values.append(0)
                
                offset = (i - 0.5) * width
                bars = ax.bar(x + offset, values, width, 
                             label=mode.replace("_", " ").title(),
                             color=mode_colors.get(mode, "gray"),
                             alpha=0.8)
            
            # Add pooled reference line (green dashed)
            pooled_val = pooled_values.get(metric)
            if pooled_val is not None:
                ax.axhline(y=pooled_val, color="#27ae60", linestyle="--", linewidth=1.5, alpha=0.8, label="Pooled" if row_idx == 0 and col_idx == 6 else "")
            
            ax.set_xticks(x)
            ax.set_xticklabels(["Out", "Mid", "In"], fontsize=8)
            ax.set_ylim(0, 1.0)
            ax.grid(axis="y", alpha=0.3)
            
            # Labels
            if row_idx == 0:
                ax.set_title(METRIC_LABELS.get(metric, metric), fontweight="bold", fontsize=11)
            if col_idx == 0:
                ax.set_ylabel(distance.upper(), fontweight="bold", fontsize=11)
            if row_idx == 0 and col_idx == 6:
                ax.legend(fontsize=7, loc="upper right")
    
    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def plot_4x7_metrics_grid_by_method(all_results: Dict[str, dict], output_dir: Path, eval_dir: Path) -> None:
    """Plot 4x7 metrics grid separately for each imbalance method (5 patterns)."""
    # Group results by method (5 patterns)
    methods = {
        "Baseline": {},
        "SW-SMOTE r=0.1": {},
        "SW-SMOTE r=0.5": {},
        "Plain SMOTE r=0.1": {},
        "Plain SMOTE r=0.5": {}
    }
    
    for key, result in all_results.items():
        info = parse_experiment_key(key)
        method = info.get("method", "Unknown")
        if method in methods:
            methods[method][key] = result
        elif "baseline" in key.lower():
            methods["Baseline"][key] = result
    
    # Load pooled results for each method
    pooled_results = {}
    for method_name in methods.keys():
        pooled_results[method_name] = load_pooled_results(eval_dir, method_name)
    
    # Plot each method separately
    for method_name, method_results in methods.items():
        if method_results:
            safe_name = method_name.lower().replace(' ', '_').replace('-', '_')
            output_path = output_dir / f"metrics_grid_4x7_{safe_name}.png"
            plot_4x7_metrics_grid(
                method_results, 
                output_path, 
                title=f"Domain Analysis: {method_name}",
                pooled_result=pooled_results.get(method_name)
            )

def plot_confusion_matrices(results: Dict[str, dict], output_dir: Path, method_name: str = "all") -> None:
    """Plot confusion matrices grid for experiments of a specific method."""
    # Filter results that have confusion matrix
    cm_results = {k: v for k, v in results.items() if "confusion_matrix" in v}
    
    if not cm_results:
        logger.warning(f"No confusion matrices found for {method_name}")
        return
    
    # Group by distance and mode
    grouped = {}
    for key, result in cm_results.items():
        info = parse_experiment_key(key)
        group_key = f"{info['distance']}_{info['mode']}"
        if group_key not in grouped:
            grouped[group_key] = {}
        grouped[group_key][info["domain"]] = result["confusion_matrix"]
    
    n_groups = len(grouped)
    if n_groups == 0:
        return
    
    # Create grid: rows = distance_mode, cols = domains
    n_cols = 3  # in, mid, out domain
    n_rows = n_groups
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for row_idx, (group_key, domain_data) in enumerate(sorted(grouped.items())):
        for col_idx, domain in enumerate(DOMAINS):
            ax = axes[row_idx, col_idx]
            
            if domain in domain_data:
                cm = np.array(domain_data[domain])
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                           xticklabels=["Pred 0", "Pred 1"],
                           yticklabels=["True 0", "True 1"])
                ax.set_title(f"{domain.replace('_', ' ').title()}", fontsize=10)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center")
                ax.axis("off")
            
            if col_idx == 0:
                ax.set_ylabel(group_key.replace("_", "\n"), fontsize=9)
    
    fig.suptitle(f"Confusion Matrices: {method_name}", fontweight="bold", fontsize=14)
    fig.tight_layout()
    
    output_path = output_dir / f"confusion_matrices_{method_name.lower().replace(' ', '_').replace('-', '_')}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def plot_confusion_matrices_by_method(all_results: Dict[str, dict], output_dir: Path) -> None:
    """Plot confusion matrices separately for each imbalance method (5 patterns)."""
    # Group results by method (5 patterns)
    methods = {
        "Baseline": {},
        "SW-SMOTE r=0.1": {},
        "SW-SMOTE r=0.5": {},
        "Plain SMOTE r=0.1": {},
        "Plain SMOTE r=0.5": {}
    }
    
    for key, result in all_results.items():
        info = parse_experiment_key(key)
        method = info.get("method", "Unknown")
        if method in methods:
            methods[method][key] = result
        elif "baseline" in key.lower():
            methods["Baseline"][key] = result
    
    # Plot each method separately
    for method_name, method_results in methods.items():
        if method_results:
            plot_confusion_matrices(method_results, output_dir, method_name)



def plot_optuna_convergence(optuna_results: Dict[str, Tuple[pd.DataFrame, dict]], output_path: Path) -> None:
    """Plot Optuna objective function convergence for all experiments."""
    if not optuna_results:
        logger.warning("No Optuna results to plot")
        return
    
    # Group by distance
    by_distance = {}
    for key, (df, conv) in optuna_results.items():
        info = parse_experiment_key(key)
        dist = info["distance"] or "pooled"
        if dist not in by_distance:
            by_distance[dist] = []
        by_distance[dist].append((key, df, conv))
    
    n_distances = len(by_distance)
    fig, axes = plt.subplots(1, n_distances, figsize=(6 * n_distances, 5))
    if n_distances == 1:
        axes = [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for ax_idx, (distance, items) in enumerate(sorted(by_distance.items())):
        ax = axes[ax_idx]
        
        for i, (key, df, conv) in enumerate(items):
            if "value" in df.columns:
                values = df["value"].values
                best_so_far = np.maximum.accumulate(values)
                ax.plot(range(1, len(values) + 1), best_so_far, 
                       label=key[:30], color=colors[i % 10], alpha=0.8)
        
        ax.set_xlabel("Trial")
        ax.set_ylabel("Best Value (F2)")
        ax.set_title(f"{distance.upper()} Experiments", fontweight="bold")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")
    
    fig.suptitle("Optuna Convergence by Distance", fontweight="bold", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def plot_optuna_convergence_by_method(optuna_results: Dict[str, Tuple[pd.DataFrame, dict]], output_dir: Path) -> None:
    """Plot Optuna F2 values for each trial, separated by imbalance method (5 patterns)."""
    if not optuna_results:
        logger.warning("No Optuna results to plot")
        return
    
    # Group results by method (5 patterns)
    methods = {
        "Baseline": {},
        "SW-SMOTE r=0.1": {},
        "SW-SMOTE r=0.5": {},
        "Plain SMOTE r=0.1": {},
        "Plain SMOTE r=0.5": {}
    }
    
    for key, (df, conv) in optuna_results.items():
        info = parse_experiment_key(key)
        method = info.get("method", "Unknown")
        if method in methods:
            methods[method][key] = (df, conv, info)
        elif "baseline" in key.lower():
            methods["Baseline"][key] = (df, conv, info)
    
    # Define colors for distance metrics
    distance_colors = {
        "dtw": "#e74c3c",       # Red
        "mmd": "#3498db",       # Blue
        "wasserstein": "#2ecc71" # Green
    }
    
    # Define markers for modes
    mode_markers = {
        "source_only": "o",
        "target_only": "s"
    }
    
    # Plot each method separately
    for method_name, method_results in methods.items():
        if not method_results:
            continue
        
        # Group by distance for subplot layout
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        distances = ["dtw", "mmd", "wasserstein"]
        
        for ax_idx, distance in enumerate(distances):
            ax = axes[ax_idx]
            
            # Filter results for this distance
            dist_results = {k: v for k, v in method_results.items() if v[2]["distance"] == distance}
            
            if not dist_results:
                ax.text(0.5, 0.5, "No Data", ha="center", va="center", fontsize=12, color="gray")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_title(f"{distance.upper()}", fontweight="bold", fontsize=12)
                continue
            
            # Plot each experiment's trials
            for key, (df, conv, info) in dist_results.items():
                if "value" not in df.columns:
                    continue
                
                values = df["value"].values
                trials = range(1, len(values) + 1)
                
                domain = info.get("domain", "unknown")
                mode = info.get("mode", "unknown")
                
                # Create label with domain and mode
                domain_short = domain.replace("_domain", "").capitalize()
                mode_short = "Src" if mode == "source_only" else "Tgt"
                label = f"{domain_short} ({mode_short})"
                
                # Color by domain, marker by mode
                domain_colors = {
                    "out_domain": "#e74c3c",   # Red
                    "mid_domain": "#f39c12",   # Orange
                    "in_domain": "#27ae60"     # Green
                }
                color = domain_colors.get(domain, "gray")
                marker = mode_markers.get(mode, "o")
                
                # Plot individual trial values (scatter)
                ax.scatter(trials, values, c=color, marker=marker, alpha=0.5, s=20, label=label)
                
                # Add trend line (moving average)
                if len(values) >= 5:
                    window = min(5, len(values))
                    smoothed = pd.Series(values).rolling(window=window, min_periods=1).mean()
                    ax.plot(trials, smoothed, color=color, alpha=0.8, linewidth=1.5, 
                           linestyle="-" if mode == "source_only" else "--")
            
            ax.set_xlabel("Trial", fontsize=10)
            ax.set_ylabel("Objective Value", fontsize=10)
            ax.set_title(f"{distance.upper()}", fontweight="bold", fontsize=12)
            # Auto-scale Y axis based on data
            ax.set_ylim(0, 1.0)
            ax.grid(alpha=0.3)
            
            # Create legend with unique entries
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc="upper right")
        
        fig.suptitle(f"Optuna Trials F2 Values: {method_name}", fontweight="bold", fontsize=14)
        fig.tight_layout()
        
        safe_name = method_name.lower().replace(' ', '_').replace('-', '_').replace('=', '')
        output_path = output_dir / f"optuna_trials_{safe_name}.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {output_path}")


def plot_hyperparameter_trials(optuna_results: Dict[str, Tuple[pd.DataFrame, dict]], output_dir: Path) -> None:
    """Plot hyperparameter exploration for each experiment."""
    # Check for params_ prefix (Optuna format) or direct names
    hp_params_prefixed = ["params_n_estimators", "params_max_depth", "params_min_samples_split", "params_min_samples_leaf"]
    hp_params_direct = ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"]
    
    for key, (df, conv) in optuna_results.items():
        # Check which params exist (try prefixed first, then direct)
        if any(p in df.columns for p in hp_params_prefixed):
            available_params = [p for p in hp_params_prefixed if p in df.columns]
            param_labels = {p: p.replace("params_", "") for p in available_params}
        else:
            available_params = [p for p in hp_params_direct if p in df.columns]
            param_labels = {p: p for p in available_params}
        
        if not available_params:
            logger.debug(f"No HP params found for {key}")
            continue
            continue
        
        n_params = len(available_params)
        fig, axes = plt.subplots(2, (n_params + 1) // 2, figsize=(12, 8))
        axes = axes.flatten()
        
        # Color by trial value
        if "value" in df.columns:
            colors = df["value"].values
            norm = plt.Normalize(vmin=colors.min(), vmax=colors.max())
            cmap = plt.cm.viridis
        else:
            colors = range(len(df))
            norm = plt.Normalize(0, len(df))
            cmap = plt.cm.viridis
        
        for idx, param in enumerate(available_params):
            ax = axes[idx]
            scatter = ax.scatter(range(len(df)), df[param], c=colors, cmap=cmap, norm=norm, alpha=0.7)
            ax.set_xlabel("Trial")
            ax.set_ylabel(param_labels.get(param, param))
            ax.set_title(param_labels.get(param, param), fontweight="bold")
            ax.grid(alpha=0.3)
        
        # Hide unused axes
        for idx in range(len(available_params), len(axes)):
            axes[idx].set_visible(False)
        
        # Add colorbar
        cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax, label="Objective Value")
        
        fig.suptitle(f"Hyperparameter Exploration: {key[:50]}", fontweight="bold")
        fig.tight_layout(rect=[0, 0, 0.92, 0.95])
        
        # Sanitize filename
        safe_key = re.sub(r"[^\w\-]", "_", key)[:50]
        output_path = output_dir / f"hp_trials_{safe_key}.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {output_path}")


def create_metrics_csv(results: Dict[str, dict], output_path: Path) -> pd.DataFrame:
    """Create summary CSV with all metrics."""
    rows = []
    
    for key, result in results.items():
        info = parse_experiment_key(key)
        row = {
            "experiment": key,
            "distance": info["distance"],
            "domain": info["domain"],
            "mode": info["mode"],
            "method": info["method"],
            "seed": info["seed"]
        }
        
        for metric in METRICS:
            value = result.get(metric)
            if value is None and metric == "roc_auc":
                value = result.get("roc_auc_score") or result.get("auc")
            row[metric] = value
        
        # Add confusion matrix values
        if "confusion_matrix" in result:
            cm = result["confusion_matrix"]
            row["TN"] = cm[0][0]
            row["FP"] = cm[0][1]
            row["FN"] = cm[1][0]
            row["TP"] = cm[1][1]
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved: {output_path}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Visualize local domain analysis results")
    parser.add_argument("--job_id", default="local", help="Job ID (default: local)")
    parser.add_argument("--patterns", default="smote_plain,baseline_domain,imbalv3",
                       help="Comma-separated patterns to search (default: smote_plain,baseline_domain,imbalv3)")
    parser.add_argument("--seed", type=int, default=None, help="Filter by seed")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    args = parser.parse_args()
    
    # Paths
    eval_dir = Path(f"results/outputs/evaluation/RF/{args.job_id}")
    models_dir = Path(f"models/RF/{args.job_id}")
    
    if args.output_dir:
        output_base = Path(args.output_dir)
    else:
        output_base = Path(f"results/analysis/domain/imbalance")
    
    output_base.mkdir(parents=True, exist_ok=True)
    (output_base / "metrics").mkdir(exist_ok=True)
    (output_base / "optuna").mkdir(exist_ok=True)
    (output_base / "confusion").mkdir(exist_ok=True)
    
    patterns = [p.strip() for p in args.patterns.split(",")]
    
    # Load all results
    all_eval_results = {}
    all_optuna_results = {}
    
    for pattern in patterns:
        eval_results = load_evaluation_results(eval_dir, pattern, args.seed)
        all_eval_results.update(eval_results)
        
        optuna_results = load_optuna_results(models_dir, pattern, args.seed)
        all_optuna_results.update(optuna_results)
    
    if not all_eval_results:
        logger.error("No evaluation results found")
        return 1
    
    logger.info(f"Total: {len(all_eval_results)} eval results, {len(all_optuna_results)} Optuna results")
    
    # Generate visualizations
    logger.info("\n=== 1. 4x7 Metrics Grid (by method) ===")
    plot_4x7_metrics_grid_by_method(all_eval_results, output_base / "metrics", eval_dir)
    
    logger.info("\n=== 2. Metrics CSV ===")
    create_metrics_csv(all_eval_results, output_base / "metrics" / "metrics_summary.csv")
    
    logger.info("\n=== 3. Confusion Matrices (by method) ===")
    plot_confusion_matrices_by_method(all_eval_results, output_base / "confusion")
    
    logger.info("\n=== 4. Optuna Convergence (by method) ===")
    plot_optuna_convergence_by_method(all_optuna_results, output_base / "optuna")
    
    logger.info("\n=== 5. Hyperparameter Trials ===")
    plot_hyperparameter_trials(all_optuna_results, output_base / "optuna")
    
    logger.info("\n" + "=" * 50)
    logger.info("VISUALIZATION COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Output directory: {output_base}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
