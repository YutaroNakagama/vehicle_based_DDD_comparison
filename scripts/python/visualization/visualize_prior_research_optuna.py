#!/usr/bin/env python3
"""
Prior Research Hyperparameter Optimization Visualization
=========================================================

This script generates visualization of hyperparameter optimization convergence
for prior research methods.

Optimization methods used:
- SvmW: Optuna (C parameter tuning)
- SvmA: PSO (Particle Swarm Optimization) - no Optuna
- Lstm: K-Fold Cross-Validation - no hyperparameter tuning

Output:
    results/analysis/exp3_prior_research/
    - optuna_convergence_SvmW.png: SvmW convergence plot
    - optuna_summary.csv: Summary of Optuna trials
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
MODELS_BASE = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis" / "prior_research"


def find_optuna_files() -> Dict[str, Dict[str, str]]:
    """
    Find all Optuna convergence files for prior research models.
    
    Returns
    -------
    Dict[str, Dict[str, str]]
        Dictionary mapping model_seed to {'convergence': path, 'trials': path}
    """
    result_files = {}
    
    # Search for convergence.json files
    for model in ["SvmA", "SvmW", "Lstm"]:
        model_dir = MODELS_BASE / model
        if not model_dir.exists():
            continue
        
        for conv_file in model_dir.glob("**/optuna*convergence.json"):
            # Extract seed from filename
            filename = conv_file.stem
            if "_s42_" in filename or filename.endswith("_s42"):
                seed = "s42"
            elif "_s123_" in filename or filename.endswith("_s123"):
                seed = "s123"
            else:
                # Try to extract from the file content
                try:
                    with open(conv_file) as f:
                        data = json.load(f)
                    seed = f"s{data['metadata'].get('seed', 'unknown')}"
                except:
                    continue
            
            key = f"{model}_{seed}"
            trials_file = conv_file.with_name(conv_file.name.replace("convergence.json", "trials.csv"))
            
            result_files[key] = {
                "convergence": str(conv_file),
                "trials": str(trials_file) if trials_file.exists() else None
            }
    
    return result_files


def load_convergence_data(filepath: str) -> Dict[str, Any]:
    """Load convergence data from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def plot_convergence(data: Dict[str, Any], model: str, seed: str, output_path: Path) -> None:
    """
    Create a convergence plot for Optuna optimization.
    
    Parameters
    ----------
    data : Dict[str, Any]
        Convergence data from JSON file
    model : str
        Model name
    seed : str
        Seed value
    output_path : Path
        Output path for the plot
    """
    trials = data["trials"]
    metadata = data["metadata"]
    
    trial_numbers = [t["trial_number"] for t in trials]
    values = [t["value"] for t in trials]
    best_so_far = [t["best_so_far"] for t in trials]
    
    # Extract parameter values
    param_names = list(trials[0]["params"].keys())
    
    # Create figure with subplots
    n_params = len(param_names)
    fig, axes = plt.subplots(2, max(n_params, 1), figsize=(6 * max(n_params, 1), 10))
    if n_params == 1:
        axes = axes.reshape(2, 1)
    
    # Top row: Objective value convergence
    ax1 = axes[0, 0]
    ax1.scatter(trial_numbers, values, alpha=0.6, label="Trial Value", color="blue", s=30)
    ax1.plot(trial_numbers, best_so_far, color="red", linewidth=2, label="Best So Far")
    ax1.axhline(y=metadata["best_value"], color="green", linestyle="--", 
                label=f"Best: {metadata['best_value']:.4f}")
    ax1.set_xlabel("Trial Number", fontsize=11)
    ax1.set_ylabel("Objective Value (CV Score)", fontsize=11)
    ax1.set_title(f"{model} Optuna Convergence ({seed})", fontsize=13, fontweight="bold")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Fill remaining top row if needed
    for i in range(1, max(n_params, 1)):
        axes[0, i].axis('off')
    
    # Bottom row: Parameter values over trials
    for i, param_name in enumerate(param_names):
        ax = axes[1, i]
        param_values = [t["params"][param_name] for t in trials]
        
        # Color by objective value
        colors = plt.cm.viridis(np.array(values) / max(values) if max(values) > 0 else [0.5] * len(values))
        scatter = ax.scatter(trial_numbers, param_values, c=values, cmap="viridis", 
                            alpha=0.7, s=40)
        
        # Mark best trial
        best_trial_idx = trial_numbers[np.argmax(best_so_far)]
        best_param_value = metadata["best_params"][param_name]
        ax.axhline(y=best_param_value, color="red", linestyle="--", 
                   label=f"Best: {best_param_value:.4f}")
        ax.scatter([best_trial_idx], [best_param_value], color="red", s=100, 
                   marker="*", zorder=5)
        
        ax.set_xlabel("Trial Number", fontsize=11)
        ax.set_ylabel(f"Parameter: {param_name}", fontsize=11)
        ax.set_title(f"Parameter {param_name} Search", fontsize=12)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, label="Objective Value")
        cbar.ax.tick_params(labelsize=8)
    
    # Fill remaining bottom row if needed
    for i in range(n_params, max(n_params, 1)):
        axes[1, i].axis('off')
    
    plt.suptitle(f"Hyperparameter Optimization: {model}\nn_trials={metadata['n_trials']}, "
                 f"best_value={metadata['best_value']:.4f}", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_all_models_comparison(all_data: Dict[str, Dict[str, Any]], output_path: Path) -> None:
    """
    Create a comparison plot of convergence across all models with Optuna.
    
    Parameters
    ----------
    all_data : Dict[str, Dict[str, Any]]
        Dictionary mapping model_seed to convergence data
    output_path : Path
        Output path for the plot
    """
    if not all_data:
        print("  No Optuna data available for comparison plot")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(all_data)))
    
    for (key, data), color in zip(all_data.items(), colors):
        trials = data["trials"]
        trial_numbers = [t["trial_number"] for t in trials]
        best_so_far = [t["best_so_far"] for t in trials]
        
        ax.plot(trial_numbers, best_so_far, linewidth=2, label=key, color=color)
        ax.scatter([trial_numbers[-1]], [best_so_far[-1]], color=color, s=100, 
                   marker="*", zorder=5)
    
    ax.set_xlabel("Trial Number", fontsize=12)
    ax.set_ylabel("Best Objective Value", fontsize=12)
    ax.set_title("Optuna Convergence Comparison\n(Prior Research Models)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")


def create_summary_table(all_data: Dict[str, Dict[str, Any]], output_path: Path) -> None:
    """
    Create a summary CSV of all Optuna trials.
    
    Parameters
    ----------
    all_data : Dict[str, Dict[str, Any]]
        Dictionary mapping model_seed to convergence data
    output_path : Path
        Output path for the CSV
    """
    records = []
    
    for key, data in all_data.items():
        model, seed = key.rsplit("_", 1)
        metadata = data["metadata"]
        trials = data["trials"]
        
        # Calculate statistics
        values = [t["value"] for t in trials]
        durations = [t["duration_seconds"] for t in trials]
        
        record = {
            "model": model,
            "seed": seed,
            "n_trials": metadata["n_trials"],
            "best_value": metadata["best_value"],
            "mean_value": np.mean(values),
            "std_value": np.std(values),
            "min_value": np.min(values),
            "max_value": np.max(values),
            "total_duration_min": np.sum(durations) / 60,
            "mean_trial_duration_sec": np.mean(durations),
        }
        
        # Add best params
        for param_name, param_value in metadata["best_params"].items():
            record[f"best_{param_name}"] = param_value
        
        records.append(record)
    
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path.name}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("Optuna Optimization Summary")
    print("=" * 70)
    for _, row in df.iterrows():
        print(f"\n{row['model']} ({row['seed']}):")
        print(f"  Trials: {row['n_trials']}")
        print(f"  Best Value: {row['best_value']:.4f}")
        print(f"  Mean ± Std: {row['mean_value']:.4f} ± {row['std_value']:.4f}")
        print(f"  Total Duration: {row['total_duration_min']:.1f} min")


def create_optimization_method_summary(output_path: Path) -> None:
    """
    Create a summary of optimization methods used by each model.
    """
    summary = """# Prior Research Hyperparameter Optimization Summary

## Optimization Methods

| Model | Method | Parameters Tuned | Notes |
|-------|--------|-----------------|-------|
| SvmW | Optuna (TPE) | C (regularization) | 50 trials, RBF kernel |
| SvmA | PSO (Particle Swarm) | ANFIS + SVM params | No Optuna trials available |
| Lstm | K-Fold CV | None (fixed architecture) | 5-fold, early stopping |

## SvmW Optuna Details
- Objective: Maximize CV F1 score
- Search space: C ∈ [0.001, 10] (log scale)
- Best C values found: ~0.006 (very low regularization)
- Convergence: Gradual improvement over 50 trials

## Notes
- SvmA uses PSO for joint ANFIS-SVM optimization (not Optuna)
- Lstm uses fixed architecture with early stopping (no hyperparameter search)
- Only SvmW convergence plots are available from Optuna
"""
    
    with open(output_path, "w") as f:
        f.write(summary)
    print(f"  Saved: {output_path.name}")


def main():
    """Main function."""
    print("=" * 70)
    print("Prior Research Hyperparameter Optimization Visualization")
    print("=" * 70)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find Optuna files
    print("\nSearching for Optuna files...")
    optuna_files = find_optuna_files()
    
    if not optuna_files:
        print("No Optuna files found.")
        print("Note: SvmA uses PSO, Lstm uses K-Fold CV (no Optuna).")
        create_optimization_method_summary(OUTPUT_DIR / "optimization_methods.md")
        return 0
    
    print(f"Found {len(optuna_files)} Optuna result files:")
    for key, files in sorted(optuna_files.items()):
        print(f"  - {key}: {Path(files['convergence']).name}")
    
    # Load all convergence data
    print("\nLoading convergence data...")
    all_data = {}
    for key, files in optuna_files.items():
        try:
            data = load_convergence_data(files["convergence"])
            all_data[key] = data
            model, seed = key.rsplit("_", 1)
            print(f"  {key}: {len(data['trials'])} trials, best={data['metadata']['best_value']:.4f}")
        except Exception as e:
            print(f"  {key}: ERROR - {e}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Individual convergence plots
    for key, data in all_data.items():
        model, seed = key.rsplit("_", 1)
        plot_convergence(data, model, seed, OUTPUT_DIR / f"optuna_convergence_{model}_{seed}.png")
    
    # Comparison plot
    plot_all_models_comparison(all_data, OUTPUT_DIR / "optuna_convergence_comparison.png")
    
    # Summary table
    create_summary_table(all_data, OUTPUT_DIR / "optuna_summary.csv")
    
    # Optimization method summary
    create_optimization_method_summary(OUTPUT_DIR / "optimization_methods.md")
    
    print("\n" + "=" * 70)
    print("Done! Output saved to:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
