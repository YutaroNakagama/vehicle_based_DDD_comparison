#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_optuna_convergence.py
===============================

Visualize Optuna hyperparameter optimization convergence for imbalance experiments.

Features:
- Objective value (F2) convergence over trials
- Best value progression
- Hyperparameter exploration patterns
- Per-method comparison

Usage:
    python scripts/python/visualization/visualize_optuna_convergence.py
    python scripts/python/visualization/visualize_optuna_convergence.py --output results/analysis/imbalance/optuna
"""

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
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
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "results/analysis/imbalance/optuna"

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

SEED_MARKERS = {"42": "o", "123": "s"}

HYPERPARAMS = [
    "params_n_estimators",
    "params_max_depth",
    "params_min_samples_split",
    "params_min_samples_leaf",
    "params_max_features",
    "params_max_samples",
    "params_class_weight",
]

HYPERPARAM_LABELS = {
    "params_n_estimators": "n_estimators",
    "params_max_depth": "max_depth",
    "params_min_samples_split": "min_samples_split",
    "params_min_samples_leaf": "min_samples_leaf",
    "params_max_features": "max_features",
    "params_max_samples": "max_samples",
    "params_class_weight": "class_weight",
    "params_min_weight_fraction_leaf": "min_weight_frac",
}


def find_optuna_trials_files(models_dir: Path) -> List[Path]:
    """Find all Optuna trials CSV files for imbalance experiments only.
    
    Excludes domain generalization experiments (source_only_*, target_only_*).
    """
    files = []
    
    # Patterns to exclude (domain generalization experiments)
    exclude_patterns = ["source_only_", "target_only_", "in_domain_", "out_domain_", "mid_domain_"]
    
    for model_dir in ["RF", "BalancedRF"]:
        model_path = models_dir / model_dir
        if not model_path.exists():
            continue
        
        for job_dir in model_path.iterdir():
            if not job_dir.is_dir():
                continue
            
            for f in job_dir.glob("optuna_*_trials.csv"):
                fname = f.name
                
                # Skip domain generalization experiments
                if any(p in fname for p in exclude_patterns):
                    continue
                
                # Include only imbalance single experiments
                if any(m in fname for m in ["baseline_s", "smote_ratio", "subjectwise_smote", 
                                             "undersample_rus", "balanced_rf_s"]):
                    files.append(f)
    
    return files


def load_optuna_trials(files: List[Path]) -> Dict[str, pd.DataFrame]:
    """Load Optuna trials from CSV files."""
    results = {}
    
    for f in files:
        try:
            df = pd.read_csv(f)
            
            # Extract method and seed from suffix column or filename
            if "suffix" in df.columns and len(df) > 0:
                suffix = df["suffix"].iloc[0]
                # Pattern: _pooled_baseline_s42_14658636[1]
                match = re.search(r"_pooled_(.+?)_(\d+)\[\d+\]", suffix)
                if match:
                    tag = match.group(1)
                else:
                    tag = suffix
            else:
                # Extract from filename
                fname = f.stem
                match = re.search(r"pooled__pooled_(.+?)_\d+\[", fname)
                if match:
                    tag = match.group(1)
                else:
                    continue
            
            # Extract seed
            seed_match = re.search(r"_s(\d+)$", tag)
            seed = seed_match.group(1) if seed_match else "42"
            
            # Extract method
            method = tag.replace(f"_s{seed}", "")
            
            # Get config
            config = METHOD_CONFIG.get(method, {"label": method, "color": "#999999", "order": 99})
            
            # Add metadata
            df["method"] = method
            df["seed"] = seed
            df["label"] = config["label"]
            df["color"] = config["color"]
            df["order"] = config["order"]
            
            key = f"{method}_s{seed}"
            
            # Keep only the most recent/longest trial set
            if key not in results or len(df) > len(results[key]):
                results[key] = df
            
        except Exception as e:
            logger.warning(f"Failed to load {f}: {e}")
    
    return results


def plot_convergence_all(trials: Dict[str, pd.DataFrame], output_path: Path) -> None:
    """Plot F2 convergence for all experiments on one figure."""
    if not trials:
        logger.warning("No trials data to plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Split by seed
    seed_42 = {k: v for k, v in trials.items() if "_s42" in k}
    seed_123 = {k: v for k, v in trials.items() if "_s123" in k}
    
    for ax, seed_data, seed_label in [(axes[0], seed_42, "Seed 42"), (axes[1], seed_123, "Seed 123")]:
        for key, df in sorted(seed_data.items(), key=lambda x: x[1]["order"].iloc[0]):
            if df.empty:
                continue
            
            color = df["color"].iloc[0]
            label = df["label"].iloc[0]
            
            trial_nums = df["number"].values
            values = df["value"].values
            
            # Plot trial values
            ax.plot(trial_nums, values, marker=".", linestyle="-", 
                   color=color, label=label, alpha=0.7, markersize=4)
            
            # Plot best-so-far line
            best_values = np.maximum.accumulate(values)
            ax.plot(trial_nums, best_values, linestyle="--", color=color, alpha=0.9, linewidth=2)
        
        ax.set_xlabel("Trial", fontsize=11)
        ax.set_ylabel("Objective Value (F2)", fontsize=11)
        ax.set_title(f"Optimization Convergence ({seed_label})", fontsize=12, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3, linestyle="--")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    fig.suptitle("Optuna Hyperparameter Optimization Convergence", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def plot_convergence_by_method(trials: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Plot convergence for each method separately (seed 42 vs 123)."""
    if not trials:
        return
    
    # Group by method
    methods = {}
    for key, df in trials.items():
        method = df["method"].iloc[0]
        if method not in methods:
            methods[method] = {}
        seed = df["seed"].iloc[0]
        methods[method][seed] = df
    
    for method, seed_dfs in sorted(methods.items(), key=lambda x: METHOD_CONFIG.get(x[0], {}).get("order", 99)):
        config = METHOD_CONFIG.get(method, {"label": method, "color": "#999999"})
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for seed, df in seed_dfs.items():
            marker = SEED_MARKERS.get(seed, "o")
            linestyle = "-" if seed == "42" else "--"
            alpha = 1.0 if seed == "42" else 0.7
            
            trial_nums = df["number"].values
            values = df["value"].values
            best_values = np.maximum.accumulate(values)
            
            ax.plot(trial_nums, values, marker=marker, linestyle="", 
                   color=config["color"], alpha=0.4, markersize=4, label=f"Trials (s{seed})")
            ax.plot(trial_nums, best_values, linestyle=linestyle, 
                   color=config["color"], alpha=alpha, linewidth=2, label=f"Best (s{seed})")
        
        ax.set_xlabel("Trial", fontsize=11)
        ax.set_ylabel("Objective Value (F2)", fontsize=11)
        ax.set_title(f"{config['label']}: Optimization Convergence", fontsize=12, fontweight="bold")
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(alpha=0.3, linestyle="--")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        fig.tight_layout()
        fig.savefig(output_dir / f"convergence_{method}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: convergence_{method}.png")


def plot_hyperparameter_exploration(trials: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Plot hyperparameter exploration for each experiment."""
    if not trials:
        return
    
    for key, df in trials.items():
        method = df["method"].iloc[0]
        seed = df["seed"].iloc[0]
        config = METHOD_CONFIG.get(method, {"label": method, "color": "#999999"})
        
        # Find available hyperparameters
        available_params = [p for p in HYPERPARAMS if p in df.columns and df[p].notna().any()]
        
        if not available_params:
            continue
        
        n_params = len(available_params)
        n_cols = min(4, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3.5*n_rows))
        if n_params == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        trial_nums = df["number"].values
        values = df["value"].values
        
        for idx, param in enumerate(available_params):
            ax = axes[idx]
            param_values = df[param].values
            
            # Handle categorical parameters
            if df[param].dtype == object:
                unique_vals = df[param].dropna().unique()
                val_to_num = {v: i for i, v in enumerate(unique_vals)}
                param_values = df[param].map(val_to_num).values
                ax.set_yticks(list(val_to_num.values()))
                ax.set_yticklabels(list(val_to_num.keys()), fontsize=8)
            
            # Color by objective value
            scatter = ax.scatter(trial_nums, param_values, c=values, 
                               cmap="RdYlGn", s=30, alpha=0.7, edgecolors="black", linewidth=0.5)
            
            ax.set_xlabel("Trial", fontsize=9)
            ax.set_ylabel(HYPERPARAM_LABELS.get(param, param.replace("params_", "")), fontsize=9)
            ax.set_title(HYPERPARAM_LABELS.get(param, param.replace("params_", "")), fontsize=10, fontweight="bold")
            ax.grid(alpha=0.3, linestyle="--")
        
        # Hide unused axes
        for idx in range(len(available_params), len(axes)):
            axes[idx].set_visible(False)
        
        # Add colorbar
        cbar = fig.colorbar(scatter, ax=axes[:len(available_params)], shrink=0.8, label="F2 Score")
        
        fig.suptitle(f"{config['label']} (s{seed}): Hyperparameter Exploration", 
                     fontsize=12, fontweight="bold", y=1.02)
        fig.tight_layout()
        fig.savefig(output_dir / f"hyperparam_{method}_s{seed}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: hyperparam_{method}_s{seed}.png")


def plot_best_params_comparison(trials: Dict[str, pd.DataFrame], output_path: Path) -> None:
    """Compare best hyperparameters across methods."""
    if not trials:
        return
    
    best_params_list = []
    
    for key, df in trials.items():
        method = df["method"].iloc[0]
        seed = df["seed"].iloc[0]
        config = METHOD_CONFIG.get(method, {"label": method, "color": "#999999", "order": 99})
        
        # Get best trial
        best_idx = df["value"].idxmax()
        best_row = df.loc[best_idx]
        
        row = {
            "method": method,
            "seed": seed,
            "label": config["label"],
            "order": config.get("order", 99),
            "best_f2": best_row["value"],
            "best_trial": best_row["number"],
        }
        
        for param in HYPERPARAMS:
            if param in df.columns:
                row[param] = best_row[param]
        
        best_params_list.append(row)
    
    best_df = pd.DataFrame(best_params_list)
    best_df = best_df.sort_values(["order", "seed"])
    
    # Save to CSV
    csv_path = output_path.with_suffix(".csv")
    best_df.to_csv(csv_path, index=False)
    logger.info(f"Saved: {csv_path}")
    
    # Create comparison plot for numeric params
    numeric_params = [p for p in HYPERPARAMS if p in best_df.columns and best_df[p].dtype in [np.float64, np.int64, float, int]]
    
    if not numeric_params:
        return
    
    n_params = len(numeric_params)
    fig, axes = plt.subplots(2, (n_params + 1) // 2, figsize=(14, 8))
    axes = axes.flatten()
    
    experiments = best_df.apply(lambda r: f"{r['label']} (s{r['seed']})", axis=1).tolist()
    x = np.arange(len(experiments))
    
    for idx, param in enumerate(numeric_params):
        ax = axes[idx]
        values = best_df[param].fillna(0).values
        colors = [METHOD_CONFIG.get(m, {}).get("color", "#999999") for m in best_df["method"]]
        
        ax.bar(x, values, color=colors, alpha=0.8, edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(experiments, rotation=45, ha="right", fontsize=8)
        ax.set_title(HYPERPARAM_LABELS.get(param, param.replace("params_", "")), fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
    
    for idx in range(len(numeric_params), len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle("Best Hyperparameters Comparison", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def create_summary_table(trials: Dict[str, pd.DataFrame], output_path: Path) -> None:
    """Create summary table of optimization results."""
    if not trials:
        return
    
    rows = []
    for key, df in trials.items():
        method = df["method"].iloc[0]
        seed = df["seed"].iloc[0]
        config = METHOD_CONFIG.get(method, {"label": method})
        
        best_idx = df["value"].idxmax()
        best_row = df.loc[best_idx]
        
        rows.append({
            "Method": config["label"],
            "Seed": seed,
            "Trials": len(df),
            "Best F2": best_row["value"],
            "Best Trial": int(best_row["number"]),
            "Final F2": df.iloc[-1]["value"],
            "Improvement": best_row["value"] - df.iloc[0]["value"],
        })
    
    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values(["Best F2"], ascending=False)
    
    summary_df.to_csv(output_path, index=False, float_format="%.4f")
    logger.info(f"Saved: {output_path}")
    
    print("\n" + "="*80)
    print("Optuna Optimization Summary")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Visualize Optuna convergence for imbalance experiments")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR, help="Models directory")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR, help="Output directory")
    args = parser.parse_args()
    
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find and load trials
    logger.info(f"Searching for Optuna trials in {args.models_dir}")
    files = find_optuna_trials_files(args.models_dir)
    logger.info(f"Found {len(files)} Optuna trial files")
    
    if not files:
        logger.error("No Optuna trial files found")
        return 1
    
    trials = load_optuna_trials(files)
    logger.info(f"Loaded trials for {len(trials)} experiments")
    
    if not trials:
        logger.error("No trials data loaded")
        return 1
    
    # Generate visualizations
    plot_convergence_all(trials, output_dir / "convergence_all.png")
    plot_convergence_by_method(trials, output_dir)
    plot_hyperparameter_exploration(trials, output_dir)
    plot_best_params_comparison(trials, output_dir / "best_params_comparison.png")
    create_summary_table(trials, output_dir / "optimization_summary.csv")
    
    logger.info("Visualization complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
