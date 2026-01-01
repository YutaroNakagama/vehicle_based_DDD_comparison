#!/usr/bin/env python3
"""
optuna_analysis.py
==================
Unified Optuna convergence analysis and visualization tool.

This script consolidates:
- analyze_optuna_convergence.py (simulation)
- collect_optuna_convergence.py (log collection)
- visualize_optuna_convergence.py (study file visualization)
- visualize_real_optuna_convergence.py (log-based visualization)

Usage:
    python optuna_analysis.py simulate              # Run mock convergence simulation
    python optuna_analysis.py collect --job-log JOB_LOG_FILE  # Collect from HPC logs
    python optuna_analysis.py visualize --study-dir DIR      # Visualize from study files
    python optuna_analysis.py visualize-logs --log-dir DIR   # Visualize from log files
"""

import argparse
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
try:
    from src.config import N_TRIALS, OPTUNA_N_STARTUP_TRIALS
except ImportError:
    N_TRIALS = 50
    OPTUNA_N_STARTUP_TRIALS = 5

OUTPUT_DIR = PROJECT_ROOT / "results/imbalance_analysis/optuna_convergence"


# ============================================================
# Common Utilities
# ============================================================
def parse_trial_log(log_path: Path) -> Tuple[List[dict], Optional[dict]]:
    """Parse Optuna trial logs from HPC job output.
    
    Pattern: [Optuna] Trial   0: value=0.6284, best=0.6284
    """
    trials = []
    convergence = None
    
    trial_pattern = re.compile(
        r'\[Optuna\] Trial\s+(\d+):\s+value=([\d.]+),\s+best=([\d.]+)'
    )
    conv_pattern = re.compile(
        r'\[Optuna Convergence\] Total trials: (\d+), Best: ([\d.]+), '
        r'Last 10 best: ([\d.]+), Improvement in last 10: ([\d.]+)'
    )
    
    try:
        with open(log_path, 'r') as f:
            for line in f:
                trial_match = trial_pattern.search(line)
                if trial_match:
                    trials.append({
                        'trial_number': int(trial_match.group(1)),
                        'value': float(trial_match.group(2)),
                        'best_so_far': float(trial_match.group(3))
                    })
                
                conv_match = conv_pattern.search(line)
                if conv_match:
                    convergence = {
                        'total_trials': int(conv_match.group(1)),
                        'best_value': float(conv_match.group(2)),
                        'last_10_best': float(conv_match.group(3)),
                        'improvement_last_10': float(conv_match.group(4))
                    }
    except Exception as e:
        print(f"[WARN] Error reading {log_path}: {e}")
    
    return trials, convergence


def plot_convergence_curve(trials: List[dict], output_path: Path, title: str = "F2 Score Convergence"):
    """Plot convergence curve from trial data."""
    if not trials:
        print("[WARN] No trials to plot")
        return
    
    trial_nums = [t['trial_number'] for t in trials]
    values = [t['value'] for t in trials]
    best_so_far = [t.get('best_so_far', max(values[:i+1])) for i, v in enumerate(values)]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.scatter(trial_nums, values, alpha=0.5, s=30, label='Trial Value', color='#1f77b4')
    ax.plot(trial_nums, best_so_far, color='#d62728', linewidth=2, label='Best So Far')
    
    best_idx = np.argmax(values)
    ax.scatter([trial_nums[best_idx]], [values[best_idx]], 
               s=200, marker='*', color='gold', edgecolor='black', 
               zorder=5, label=f'Best (Trial {trial_nums[best_idx]})')
    
    ax.set_xlabel('Trial Number', fontsize=12)
    ax.set_ylabel('F2 Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# ============================================================
# Mode: simulate
# ============================================================
def mock_objective(trial):
    """Simulate RF hyperparameter tuning objective."""
    try:
        import optuna
    except ImportError:
        print("[ERROR] optuna not installed. Run: pip install optuna")
        return 0
    
    n_estimators = trial.suggest_int("n_estimators", 200, 500)
    max_depth = trial.suggest_int("max_depth", 6, 30)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)
    
    base_score = 0.15
    n_est_factor = 1 - abs(n_estimators - 350) / 300
    depth_factor = 1 - abs(max_depth - 18) / 24
    split_factor = 1 - abs(min_samples_split - 5) / 8
    leaf_factor = 1 - abs(min_samples_leaf - 2) / 4
    
    score = base_score + 0.03 * (n_est_factor + depth_factor + split_factor + leaf_factor) / 4
    noise = np.random.normal(0, 0.005)
    
    return max(0, min(1, score + noise))


def run_simulate(n_trials_list: List[int] = None, n_runs: int = 5, seed: int = 42):
    """Run convergence simulation with mock objective."""
    print("\n=== Optuna Convergence Simulation ===")
    
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("[ERROR] optuna not installed")
        return
    
    if n_trials_list is None:
        n_trials_list = [10, 25, 50, 75, 100]
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    
    for n_trials in n_trials_list:
        for run in range(n_runs):
            run_seed = seed + run * 1000
            sampler = optuna.samplers.TPESampler(seed=run_seed)
            study = optuna.create_study(direction="maximize", sampler=sampler)
            study.optimize(mock_objective, n_trials=n_trials, show_progress_bar=False)
            
            results.append({
                'n_trials': n_trials,
                'run': run,
                'best_value': study.best_value,
                'n_params': len(study.best_params),
            })
            print(f"  n_trials={n_trials}, run={run}, best={study.best_value:.4f}")
    
    df = pd.DataFrame(results)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='n_trials', y='best_value', ax=ax)
    ax.axhline(y=df['best_value'].max(), color='red', linestyle='--', label='Best Overall')
    ax.set_xlabel('Number of Trials')
    ax.set_ylabel('Best F2 Score')
    ax.set_title('Optuna Convergence Simulation')
    ax.legend()
    
    output_path = OUTPUT_DIR / 'simulation_convergence.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nSaved: {output_path}")
    
    # Summary
    summary = df.groupby('n_trials')['best_value'].agg(['mean', 'std', 'max'])
    print("\nSummary:")
    print(summary)
    summary.to_csv(OUTPUT_DIR / 'simulation_summary.csv')


# ============================================================
# Mode: collect
# ============================================================
def parse_job_log(job_log_path: Path) -> Dict[str, str]:
    """Parse job log file to get method -> job_id mapping."""
    job_map = {}
    with open(job_log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                parts = line.split('=')
                if len(parts) == 2:
                    method_key = parts[0].strip()
                    job_id = parts[1].strip().split('.')[0]
                    job_map[method_key] = job_id
    return job_map


def run_collect(job_log: Path, log_dir: Path = None, output_dir: Path = None):
    """Collect Optuna convergence data from HPC job logs."""
    print("\n=== Collecting Optuna Convergence Data ===")
    
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if log_dir is None:
        log_dir = PROJECT_ROOT / "scripts/hpc/logs"
    
    job_map = parse_job_log(job_log)
    print(f"Found {len(job_map)} experiments in job log")
    
    all_trials = []
    all_convergence = []
    
    for method_key, job_id in job_map.items():
        log_files = list(log_dir.glob(f"{job_id}*.OU"))
        
        for log_file in log_files:
            trials, conv = parse_trial_log(log_file)
            
            if trials:
                for t in trials:
                    t['method'] = method_key
                    t['job_id'] = job_id
                all_trials.extend(trials)
            
            if conv:
                conv['method'] = method_key
                conv['job_id'] = job_id
                all_convergence.append(conv)
    
    if all_trials:
        df_trials = pd.DataFrame(all_trials)
        df_trials.to_csv(output_dir / 'collected_trials.csv', index=False)
        print(f"Collected {len(all_trials)} trials")
    
    if all_convergence:
        df_conv = pd.DataFrame(all_convergence)
        df_conv.to_csv(output_dir / 'convergence_summary.csv', index=False)
        print(f"Collected {len(all_convergence)} convergence summaries")
    
    print(f"\nOutputs saved to: {output_dir}")


# ============================================================
# Mode: visualize (from study files)
# ============================================================
def load_study_from_pickle(pkl_path: Path):
    """Load Optuna study from pickle file."""
    try:
        with open(pkl_path, 'rb') as f:
            study = pickle.load(f)
        return study
    except Exception as e:
        print(f"[WARN] Failed to load {pkl_path}: {e}")
        return None


def load_convergence_from_json(json_path: Path) -> Optional[Dict]:
    """Load convergence data from JSON file."""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load {json_path}: {e}")
        return None


def run_visualize(study_dir: Path = None, json_pattern: str = None, output_dir: Path = None):
    """Visualize Optuna convergence from study files."""
    print("\n=== Visualizing Optuna Convergence (from study files) ===")
    
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load from JSON pattern
    if json_pattern:
        json_files = list(Path().glob(json_pattern))
        print(f"Found {len(json_files)} JSON files")
        
        for json_file in json_files:
            data = load_convergence_from_json(json_file)
            if data and 'trials' in data:
                trials = [{'trial_number': i, 'value': v, 'best_so_far': data.get('best_so_far', [v])[i] if i < len(data.get('best_so_far', [])) else v} 
                          for i, v in enumerate(data['trials'])]
                output_path = output_dir / f"{json_file.stem}_convergence.png"
                plot_convergence_curve(trials, output_path, json_file.stem)
    
    # Load from study directory
    elif study_dir:
        pkl_files = list(study_dir.glob("**/optuna_study*.pkl"))
        print(f"Found {len(pkl_files)} pickle files")
        
        for pkl_file in pkl_files:
            study = load_study_from_pickle(pkl_file)
            if study:
                trials = [{'trial_number': t.number, 'value': t.value, 
                           'best_so_far': max([tr.value for tr in study.trials[:t.number+1] if tr.value is not None])}
                          for t in study.trials if t.value is not None]
                output_path = output_dir / f"{pkl_file.stem}_convergence.png"
                plot_convergence_curve(trials, output_path, pkl_file.stem)
    
    print(f"\nOutputs saved to: {output_dir}")


# ============================================================
# Mode: visualize-logs
# ============================================================
def run_visualize_logs(log_dir: Path, output_dir: Path = None):
    """Visualize Optuna convergence from HPC log files."""
    print("\n=== Visualizing Optuna Convergence (from logs) ===")
    
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_records = []
    
    for log_file in log_dir.glob("*.OU"):
        try:
            with open(log_file, 'r') as f:
                first_lines = f.read(1000)
            
            # Extract method and ratio
            method_match = re.search(r'METHOD:\s*(\w+)', first_lines)
            ratio_match = re.search(r'TARGET_RATIO:\s*([\d.]+)', first_lines)
            
            method = method_match.group(1) if method_match else "unknown"
            ratio = ratio_match.group(1) if ratio_match else "unknown"
            
            trials, _ = parse_trial_log(log_file)
            
            if not trials:
                continue
            
            for trial in trials:
                trial['method'] = method
                trial['ratio'] = ratio
                trial['log_file'] = log_file.name
                all_records.append(trial)
            
            # Individual plot
            output_path = output_dir / f"{log_file.stem}_convergence.png"
            plot_convergence_curve(trials, output_path, f"{method} (ratio={ratio})")
            
        except Exception as e:
            print(f"[WARN] Error processing {log_file}: {e}")
    
    if not all_records:
        print("[WARN] No Optuna trials found in logs")
        return
    
    df = pd.DataFrame(all_records)
    df.to_csv(output_dir / 'all_trials_from_logs.csv', index=False)
    print(f"\nCollected {len(all_records)} trials from logs")
    
    # Aggregate plot
    if 'method' in df.columns and df['method'].nunique() > 1:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for method in df['method'].unique():
            method_df = df[df['method'] == method]
            ax.plot(method_df['trial_number'], method_df['best_so_far'], 
                    label=method, linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Trial Number')
        ax.set_ylabel('Best F2 Score')
        ax.set_title('Optuna Convergence Comparison by Method')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'convergence_comparison.png', dpi=150)
        plt.close()
    
    print(f"Outputs saved to: {output_dir}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Unified Optuna convergence analysis tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python optuna_analysis.py simulate
    python optuna_analysis.py collect --job-log scripts/hpc/imbalance/job_ids.txt
    python optuna_analysis.py visualize --study-dir models/RF/14621011
    python optuna_analysis.py visualize-logs --log-dir scripts/hpc/logs
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # simulate
    sim_parser = subparsers.add_parser('simulate', help='Run convergence simulation')
    sim_parser.add_argument('--n-trials', type=int, nargs='+', default=[10, 25, 50, 75, 100])
    sim_parser.add_argument('--n-runs', type=int, default=5)
    sim_parser.add_argument('--seed', type=int, default=42)
    
    # collect
    col_parser = subparsers.add_parser('collect', help='Collect from HPC logs')
    col_parser.add_argument('--job-log', type=Path, required=True)
    col_parser.add_argument('--log-dir', type=Path, default=None)
    col_parser.add_argument('--output', type=Path, default=None)
    
    # visualize
    viz_parser = subparsers.add_parser('visualize', help='Visualize from study files')
    viz_parser.add_argument('--study-dir', type=Path, default=None)
    viz_parser.add_argument('--json-pattern', type=str, default=None)
    viz_parser.add_argument('--output', type=Path, default=None)
    
    # visualize-logs
    log_parser = subparsers.add_parser('visualize-logs', help='Visualize from log files')
    log_parser.add_argument('--log-dir', type=Path, required=True)
    log_parser.add_argument('--output', type=Path, default=None)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    print("=" * 70)
    print(f"OPTUNA ANALYSIS (mode={args.command})")
    print("=" * 70)
    
    if args.command == 'simulate':
        run_simulate(args.n_trials, args.n_runs, args.seed)
    elif args.command == 'collect':
        run_collect(args.job_log, args.log_dir, args.output)
    elif args.command == 'visualize':
        run_visualize(args.study_dir, args.json_pattern, args.output)
    elif args.command == 'visualize-logs':
        run_visualize_logs(args.log_dir, args.output)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
