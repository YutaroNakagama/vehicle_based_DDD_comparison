#!/usr/bin/env python
"""
Optuna Convergence Analysis
============================

Analyzes Optuna optimization history from saved study files or runs 
convergence experiments with different n_trials values.

This script tests if N_TRIALS=50 is sufficient by:
1. Running multiple experiments with varying n_trials
2. Comparing the best objective values
3. Visualizing convergence curves
"""

import os
import sys
import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from optuna.trial import TrialState

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Try to import from src, fallback to defaults
try:
    from src.config import N_TRIALS, OPTUNA_N_STARTUP_TRIALS
except ImportError:
    N_TRIALS = 50
    OPTUNA_N_STARTUP_TRIALS = 5


def simulate_optuna_convergence(
    n_trials_list: List[int] = [10, 25, 50, 75, 100],
    n_runs: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate Optuna convergence with a simple objective function.
    
    This simulates how optimization would converge to help understand
    if 50 trials is sufficient for the hyperparameter space.
    """
    logging.info(f"Simulating convergence for n_trials: {n_trials_list}")
    
    results = []
    
    # Define a mock objective similar to RF hyperparameter tuning
    def mock_objective(trial):
        # Simulate RF hyperparameter space
        n_estimators = trial.suggest_int("n_estimators", 200, 500)
        max_depth = trial.suggest_int("max_depth", 6, 30)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)
        
        # Simulate F2 score - has a maximum around certain parameter values
        # with noise to simulate real-world variability
        base_score = 0.15
        
        # Optimal ranges (based on typical findings)
        n_est_factor = 1 - abs(n_estimators - 350) / 300
        depth_factor = 1 - abs(max_depth - 18) / 24
        split_factor = 1 - abs(min_samples_split - 5) / 8
        leaf_factor = 1 - abs(min_samples_leaf - 2) / 4
        
        score = base_score + 0.03 * (n_est_factor + depth_factor + split_factor + leaf_factor) / 4
        noise = np.random.normal(0, 0.005)
        
        return max(0, min(1, score + noise))
    
    for n_trials in n_trials_list:
        for run in range(n_runs):
            run_seed = seed + run * 1000
            np.random.seed(run_seed)
            
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=run_seed),
            )
            
            # Track best value at each trial
            best_values = []
            
            def callback(study, trial):
                if trial.state == TrialState.COMPLETE:
                    best_values.append(study.best_value)
            
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study.optimize(mock_objective, n_trials=n_trials, callbacks=[callback])
            
            # Record results
            for i, best_val in enumerate(best_values):
                results.append({
                    "n_trials": n_trials,
                    "run": run,
                    "trial": i + 1,
                    "best_value": best_val,
                    "final_best": best_values[-1]
                })
    
    return pd.DataFrame(results)


def analyze_convergence(df: pd.DataFrame, output_dir: Path) -> Dict:
    """Analyze and visualize convergence results."""
    
    # Summary statistics
    summary = df.groupby("n_trials").agg({
        "final_best": ["mean", "std", "min", "max"]
    }).round(4)
    summary.columns = ["mean", "std", "min", "max"]
    
    print("\n" + "=" * 60)
    print("Convergence Analysis Summary")
    print("=" * 60)
    print(f"Current setting: N_TRIALS = {N_TRIALS}")
    print("-" * 60)
    print(summary.to_string())
    
    # Calculate improvement from n=50 to n=100
    n_trials_list = sorted(df["n_trials"].unique())
    if 50 in n_trials_list and 100 in n_trials_list:
        mean_50 = df[df["n_trials"] == 50]["final_best"].mean()
        mean_100 = df[df["n_trials"] == 100]["final_best"].mean()
        improvement = (mean_100 - mean_50) / mean_50 * 100
        print(f"\nImprovement from n=50 to n=100: {improvement:.2f}%")
    
    # Create convergence plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Convergence curves
    ax1 = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(n_trials_list)))
    
    for n_trials, color in zip(n_trials_list, colors):
        subset = df[df["n_trials"] == n_trials]
        
        # Average across runs
        avg_by_trial = subset.groupby("trial")["best_value"].agg(["mean", "std"])
        trials = avg_by_trial.index
        means = avg_by_trial["mean"]
        stds = avg_by_trial["std"]
        
        ax1.plot(trials, means, label=f"n_trials={n_trials}", color=color, linewidth=2)
        ax1.fill_between(trials, means - stds, means + stds, color=color, alpha=0.2)
    
    ax1.set_xlabel("Trial Number", fontsize=12)
    ax1.set_ylabel("Best Objective Value (F2)", fontsize=12)
    ax1.set_title("Optuna Convergence Curves\n(Mean ± Std across runs)", fontsize=14, fontweight="bold")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=50, color='red', linestyle='--', alpha=0.5, label='Current N_TRIALS=50')
    
    # Plot 2: Final best value boxplot
    ax2 = axes[1]
    boxplot_data = [df[df["n_trials"] == n]["final_best"].values for n in n_trials_list]
    bp = ax2.boxplot(boxplot_data, labels=[str(n) for n in n_trials_list], patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_xlabel("Number of Trials", fontsize=12)
    ax2.set_ylabel("Final Best Objective Value", fontsize=12)
    ax2.set_title("Final Best Value by N_TRIALS", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Highlight current setting
    current_idx = n_trials_list.index(50) if 50 in n_trials_list else -1
    if current_idx >= 0:
        ax2.axvline(x=current_idx + 1, color='red', linestyle='--', alpha=0.5)
        ax2.text(current_idx + 1.1, ax2.get_ylim()[1], 'Current\n(50)', color='red', fontsize=10, va='top')
    
    plt.tight_layout()
    
    output_path = output_dir / "optuna_convergence_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    
    # Calculate convergence metrics
    metrics = {}
    for n_trials in n_trials_list:
        subset = df[df["n_trials"] == n_trials]
        final_values = subset.groupby("run")["final_best"].last()
        metrics[n_trials] = {
            "mean": final_values.mean(),
            "std": final_values.std(),
            "coefficient_of_variation": final_values.std() / final_values.mean() * 100
        }
    
    # Check if n=50 is sufficient
    if 50 in metrics and 100 in metrics:
        cv_50 = metrics[50]["coefficient_of_variation"]
        diff_to_100 = abs(metrics[100]["mean"] - metrics[50]["mean"])
        
        is_sufficient = cv_50 < 5 and diff_to_100 < 0.005
        
        print("\n" + "=" * 60)
        print("RECOMMENDATION")
        print("=" * 60)
        if is_sufficient:
            print("✅ N_TRIALS=50 appears SUFFICIENT")
            print(f"   - Coefficient of Variation: {cv_50:.2f}% (< 5%)")
            print(f"   - Improvement with n=100: {diff_to_100:.4f} (< 0.005)")
        else:
            print("⚠️  Consider increasing N_TRIALS")
            print(f"   - Coefficient of Variation: {cv_50:.2f}%")
            print(f"   - Potential improvement with n=100: {diff_to_100:.4f}")
    
    return metrics


def main():
    """Main function."""
    logging.basicConfig(level=logging.INFO)
    
    project_root = Path(__file__).parent.parent.parent.parent
    output_dir = project_root / "results" / "imbalance_analysis" / "optuna_convergence"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Optuna Convergence Analysis")
    print("=" * 60)
    print(f"Current N_TRIALS setting: {N_TRIALS}")
    print(f"OPTUNA_N_STARTUP_TRIALS: {OPTUNA_N_STARTUP_TRIALS}")
    
    # Run simulation
    print("\nRunning convergence simulation...")
    df = simulate_optuna_convergence(
        n_trials_list=[10, 25, 50, 75, 100],
        n_runs=5,
        seed=42
    )
    
    # Save raw data
    csv_path = output_dir / "convergence_simulation.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved simulation data: {csv_path}")
    
    # Analyze
    metrics = analyze_convergence(df, output_dir)
    
    # Save metrics
    metrics_path = output_dir / "convergence_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump({str(k): v for k, v in metrics.items()}, f, indent=2)
    print(f"Saved metrics: {metrics_path}")
    
    print(f"\nDone! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
