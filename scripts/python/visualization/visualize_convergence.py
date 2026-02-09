#!/usr/bin/env python3
"""
Convergence Visualization for Prior Research Models

This script visualizes the training/optimization convergence for:
- Lstm: Epoch-by-epoch loss and accuracy per fold
- SvmA: PSO optimization history
- SvmW: Optuna trial history (already visualized separately)

Outputs are saved to: results/analysis/exp3_prior_research/convergence/
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import numpy as np
import pandas as pd
import optuna
from optuna.visualization import plot_optimization_history

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis" / "prior_research" / "convergence"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_lstm_history(job_id: str, array_idx: str = "1") -> list:
    """Load Lstm training history from JSON file."""
    pattern = f"training_history_*_{job_id}_{array_idx}.json"
    lstm_dir = MODELS_DIR / "Lstm" / job_id / f"{job_id}[{array_idx}]"
    
    if not lstm_dir.exists():
        print(f"[WARN] Lstm directory not found: {lstm_dir}")
        return []
    
    for filepath in lstm_dir.glob(pattern):
        print(f"[INFO] Loading Lstm history from: {filepath}")
        with open(filepath, 'r') as f:
            return json.load(f)
    
    # Try alternative pattern
    for filepath in lstm_dir.glob("training_history_*.json"):
        print(f"[INFO] Loading Lstm history from: {filepath}")
        with open(filepath, 'r') as f:
            return json.load(f)
    
    print(f"[WARN] No Lstm training history found in: {lstm_dir}")
    return []


def load_svma_pso_history(job_id: str, array_idx: str = "1") -> list:
    """Load SvmA PSO optimization history from JSON file."""
    svma_dir = MODELS_DIR / "SvmA" / job_id / f"{job_id}[{array_idx}]"
    
    if not svma_dir.exists():
        print(f"[WARN] SvmA directory not found: {svma_dir}")
        return []
    
    for filepath in svma_dir.glob("pso_history_*.json"):
        print(f"[INFO] Loading PSO history from: {filepath}")
        with open(filepath, 'r') as f:
            return json.load(f)
    
    print(f"[WARN] No PSO history found in: {svma_dir}")
    return []


def load_svmw_optuna_study(job_id: str, array_idx: str = "1"):
    """Load SvmW Optuna convergence from JSON file (or study.pkl)."""
    svmw_dir = MODELS_DIR / "SvmW" / job_id
    
    if not svmw_dir.exists():
        print(f"[WARN] SvmW directory not found: {svmw_dir}")
        return None
    
    # First try to find convergence JSON files
    convergence_files = list(svmw_dir.glob("*_convergence.json"))
    if convergence_files:
        filepath = convergence_files[0]
        print(f"[INFO] Loading SvmW convergence from: {filepath}")
        with open(filepath, 'r') as f:
            return json.load(f)
    
    # Fallback to subdirectory
    subdir = svmw_dir / f"{job_id}[{array_idx}]"
    if subdir.exists():
        convergence_files = list(subdir.glob("*_convergence.json"))
        if convergence_files:
            filepath = convergence_files[0]
            print(f"[INFO] Loading SvmW convergence from: {filepath}")
            with open(filepath, 'r') as f:
                return json.load(f)
    
    print(f"[WARN] No SvmW convergence data found in: {svmw_dir}")
    return None


def visualize_lstm_convergence(histories: dict, output_prefix: str = "lstm"):
    """
    Visualize Lstm training convergence.
    
    Parameters
    ----------
    histories : dict
        Dictionary with seed as key, list of fold histories as value
    output_prefix : str
        Prefix for output filename
    """
    if not histories:
        print("[WARN] No Lstm histories to visualize")
        return
    
    # Create figure with subplots for each seed
    n_seeds = len(histories)
    fig, axes = plt.subplots(n_seeds, 2, figsize=(14, 5 * n_seeds))
    if n_seeds == 1:
        axes = axes.reshape(1, -1)
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for seed_idx, (seed, fold_histories) in enumerate(histories.items()):
        ax_loss = axes[seed_idx, 0]
        ax_acc = axes[seed_idx, 1]
        
        for fold_data in fold_histories:
            fold_no = fold_data['fold']
            hist = fold_data['history']
            epochs = hist.get('epochs', list(range(1, len(hist.get('loss', [])) + 1)))
            
            # Plot loss
            if hist.get('loss'):
                ax_loss.plot(epochs, hist['loss'], 
                           label=f'Fold {fold_no} Train', 
                           color=colors[fold_no - 1], linestyle='-')
            if hist.get('val_loss'):
                ax_loss.plot(epochs, hist['val_loss'], 
                           label=f'Fold {fold_no} Val', 
                           color=colors[fold_no - 1], linestyle='--')
            
            # Plot accuracy
            if hist.get('accuracy'):
                ax_acc.plot(epochs, hist['accuracy'], 
                          label=f'Fold {fold_no} Train', 
                          color=colors[fold_no - 1], linestyle='-')
            if hist.get('val_accuracy'):
                ax_acc.plot(epochs, hist['val_accuracy'], 
                          label=f'Fold {fold_no} Val', 
                          color=colors[fold_no - 1], linestyle='--')
        
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.set_title(f'Lstm Training Loss (Seed {seed})')
        ax_loss.legend(loc='upper right', fontsize=8)
        ax_loss.grid(True, alpha=0.3)
        
        ax_acc.set_xlabel('Epoch')
        ax_acc.set_ylabel('Accuracy')
        ax_acc.set_title(f'Lstm Training Accuracy (Seed {seed})')
        ax_acc.legend(loc='lower right', fontsize=8)
        ax_acc.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / f"{output_prefix}_convergence.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved Lstm convergence plot to: {output_path}")


def visualize_svma_pso_convergence(histories: dict, output_prefix: str = "svma"):
    """
    Visualize SvmA PSO optimization convergence.
    
    Parameters
    ----------
    histories : dict
        Dictionary with seed as key, PSO history list as value
    output_prefix : str
        Prefix for output filename
    """
    if not histories:
        print("[WARN] No SvmA PSO histories to visualize")
        return
    
    n_seeds = len(histories)
    fig, axes = plt.subplots(n_seeds, 2, figsize=(14, 5 * n_seeds))
    if n_seeds == 1:
        axes = axes.reshape(1, -1)
    
    for seed_idx, (seed, pso_history) in enumerate(histories.items()):
        if not pso_history:
            continue
            
        evaluations = [h['evaluation'] for h in pso_history]
        accuracies = [h['accuracy'] for h in pso_history]
        fitness = [h['fitness'] for h in pso_history]
        
        # Extract parameter values
        C_values = [h['params']['C'] for h in pso_history]
        gamma_values = [h['params']['gamma'] for h in pso_history]
        
        ax_fitness = axes[seed_idx, 0]
        ax_params = axes[seed_idx, 1]
        
        # Plot fitness over evaluations (lower is better, 1.0 = penalty)
        # Convert to "score" for easier interpretation (higher is better)
        # fitness = -accuracy (when valid) or 1.0 (penalty)
        scores = [-f if f < 1.0 else 0.0 for f in fitness]
        penalty_mask = [f == 1.0 for f in fitness]
        valid_mask = [f < 1.0 for f in fitness]
        
        # Count valid vs penalty evaluations
        n_valid = sum(valid_mask)
        n_penalty = sum(penalty_mask)
        
        # Plot all evaluations
        ax_fitness.scatter([e for e, p in zip(evaluations, penalty_mask) if p], 
                          [0 for p in penalty_mask if p], 
                          c='red', alpha=0.3, s=20, label=f'Penalty (empty features): {n_penalty}')
        ax_fitness.scatter([e for e, v in zip(evaluations, valid_mask) if v], 
                          [s for s, v in zip(scores, valid_mask) if v], 
                          c='blue', alpha=0.6, s=30, label=f'Valid: {n_valid}')
        
        # Show best accuracy with horizontal line
        if n_valid > 0:
            best_acc = max(scores)
            best_eval = evaluations[scores.index(best_acc)]
            ax_fitness.axhline(y=best_acc, color='green', linestyle='--', alpha=0.7, 
                              label=f'Best Accuracy: {best_acc:.4f}')
        
        ax_fitness.set_xlabel('Evaluation')
        ax_fitness.set_ylabel('Accuracy (0 = penalty/empty features)')
        ax_fitness.set_title(f'SvmA PSO Optimization (Seed {seed})\n'
                            f'{n_valid}/{len(fitness)} valid evaluations')
        ax_fitness.legend(loc='upper right', fontsize=8)
        ax_fitness.grid(True, alpha=0.3)
        ax_fitness.set_ylim(-0.05, 1.05)
        
        # Plot parameter evolution
        ax_params.scatter(evaluations, C_values, c='blue', alpha=0.6, label='C', s=30)
        ax_params.scatter(evaluations, gamma_values, c='orange', alpha=0.6, label='gamma', s=30)
        ax_params.set_xlabel('Evaluation')
        ax_params.set_ylabel('Parameter Value')
        ax_params.set_title(f'SvmA PSO Parameter Search (Seed {seed})')
        ax_params.legend()
        ax_params.grid(True, alpha=0.3)
        ax_params.set_yscale('log')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / f"{output_prefix}_pso_convergence.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved SvmA PSO convergence plot to: {output_path}")


def visualize_svmw_optuna_convergence(studies: dict, output_prefix: str = "svmw"):
    """
    Visualize SvmW Optuna optimization convergence.
    
    Parameters
    ----------
    studies : dict
        Dictionary with seed as key, convergence data (dict) as value
    output_prefix : str
        Prefix for output filename
    """
    if not studies:
        print("[WARN] No SvmW Optuna studies to visualize")
        return
    
    n_seeds = len(studies)
    fig, axes = plt.subplots(n_seeds, 2, figsize=(14, 5 * n_seeds))
    if n_seeds == 1:
        axes = axes.reshape(1, -1)
    
    for seed_idx, (seed, data) in enumerate(studies.items()):
        if data is None:
            continue
        
        # Extract trials from JSON structure
        trials = data.get('trials', [])
        if not trials:
            continue
        
        ax_obj = axes[seed_idx, 0]
        ax_param = axes[seed_idx, 1]
        
        # Extract data
        trial_nums = [t['trial_number'] for t in trials]
        values = [t['value'] for t in trials]
        best_so_far = [t['best_so_far'] for t in trials]
        C_values = [t['params']['C'] for t in trials]
        
        # Plot objective value over trials
        ax_obj.plot(trial_nums, values, 'b-o', markersize=3, alpha=0.5, label='Trial Value (F1)')
        ax_obj.plot(trial_nums, best_so_far, 'r-', linewidth=2, label='Best So Far')
        ax_obj.set_xlabel('Trial')
        ax_obj.set_ylabel('Objective Value (F1 Score)')
        ax_obj.set_title(f'SvmW Optuna Optimization (Seed {seed})')
        ax_obj.legend()
        ax_obj.grid(True, alpha=0.3)
        
        # Plot C parameter over trials with color by objective
        scatter = ax_param.scatter(trial_nums, C_values, 
                       c=values, cmap='viridis', alpha=0.7, s=30)
        ax_param.set_xlabel('Trial')
        ax_param.set_ylabel('C Parameter')
        ax_param.set_title(f'SvmW C Parameter Search (Seed {seed})')
        ax_param.set_yscale('log')
        ax_param.grid(True, alpha=0.3)
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax_param, label='F1 Score')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / f"{output_prefix}_optuna_convergence.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved SvmW Optuna convergence plot to: {output_path}")


def create_combined_summary(lstm_histories: dict, svma_histories: dict, svmw_studies: dict):
    """Create a summary table of all optimization methods."""
    rows = []
    
    # Lstm summary
    for seed, fold_histories in lstm_histories.items():
        if fold_histories:
            total_epochs = sum(len(fh['history'].get('epochs', [])) for fh in fold_histories)
            avg_epochs = total_epochs / len(fold_histories) if fold_histories else 0
            rows.append({
                'Model': 'Lstm',
                'Seed': seed,
                'Method': 'K-Fold CV + Early Stopping',
                'Total Evaluations': total_epochs,
                'Avg per Fold': f"{avg_epochs:.1f}",
                'Best Value': '-'
            })
    
    # SvmA summary
    for seed, pso_history in svma_histories.items():
        if pso_history:
            best_acc = max(h['accuracy'] for h in pso_history)
            n_evals = len(pso_history)
            # Infer swarmsize and maxiter from total evaluations
            # swarmsize * (maxiter + 1) = n_evals
            rows.append({
                'Model': 'SvmA',
                'Seed': seed,
                'Method': f'PSO ({n_evals} evaluations)',
                'Total Evaluations': n_evals,
                'Avg per Fold': '-',
                'Best Value': f"{best_acc:.4f}"
            })
    
    # SvmW summary
    for seed, data in svmw_studies.items():
        if data:
            metadata = data.get('metadata', {})
            best_val = metadata.get('best_value', 0)
            n_trials = metadata.get('n_trials', len(data.get('trials', [])))
            rows.append({
                'Model': 'SvmW',
                'Seed': seed,
                'Method': f'Optuna TPE ({n_trials} trials)',
                'Total Evaluations': n_trials,
                'Avg per Fold': '-',
                'Best Value': f"{best_val:.4f}"
            })
    
    if rows:
        df = pd.DataFrame(rows)
        summary_path = OUTPUT_DIR / "convergence_summary.csv"
        df.to_csv(summary_path, index=False)
        print(f"[INFO] Saved convergence summary to: {summary_path}")
        print("\n" + df.to_string(index=False))
        return df
    return None


def main():
    """Main function to run convergence visualization."""
    print("=" * 60)
    print("Prior Research Convergence Visualization")
    print("=" * 60)
    
    # Define job IDs for each model and seed
    # These should be updated to match actual job IDs with convergence data
    LSTM_JOBS = {
        's42': '14674645',   # Lstm seed 42 with training history
        's123': '14674646',  # Lstm seed 123 with training history
    }
    
    SVMA_JOBS = {
        's42': '14674658',   # SvmA seed 42 with improved PSO (swarmsize=10, maxiter=20)
        's123': '14674659',  # SvmA seed 123 with improved PSO
    }
    
    SVMW_JOBS = {
        's42': '14662837',   # SvmW seed 42
        's123': '14662838',  # SvmW seed 123
    }
    
    # Load histories
    lstm_histories = {}
    for seed, job_id in LSTM_JOBS.items():
        history = load_lstm_history(job_id)
        if history:
            lstm_histories[seed] = history
    
    svma_histories = {}
    for seed, job_id in SVMA_JOBS.items():
        history = load_svma_pso_history(job_id)
        if history:
            svma_histories[seed] = history
    
    svmw_studies = {}
    for seed, job_id in SVMW_JOBS.items():
        study = load_svmw_optuna_study(job_id)
        if study:
            svmw_studies[seed] = study
    
    # Visualize
    if lstm_histories:
        visualize_lstm_convergence(lstm_histories)
    else:
        print("[INFO] No Lstm training histories available yet.")
        print("       Run Lstm training with updated code to generate history files.")
    
    if svma_histories:
        visualize_svma_pso_convergence(svma_histories)
    else:
        print("[INFO] No SvmA PSO histories available yet.")
        print("       Run SvmA training with updated code to generate history files.")
    
    if svmw_studies:
        visualize_svmw_optuna_convergence(svmw_studies)
    
    # Create summary
    create_combined_summary(lstm_histories, svma_histories, svmw_studies)
    
    print("\n" + "=" * 60)
    print("Convergence visualization complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
