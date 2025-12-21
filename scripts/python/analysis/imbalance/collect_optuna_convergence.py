"""Collect and visualize Optuna convergence from job logs.

This script:
1. Parses [Optuna] Trial logs from HPC job outputs
2. Creates convergence plots for each method/ratio
3. Generates a summary comparison across all experiments

Usage:
    python scripts/python/analysis/imbalance/collect_optuna_convergence.py \
        --job_log scripts/hpc/imbalance/job_ids_convergence_YYYYMMDD_HHMMSS.txt \
        --output results/imbalance_analysis/optuna_convergence
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def parse_job_log(job_log_path: str) -> Dict[str, str]:
    """Parse job log file to get method -> job_id mapping.
    
    Parameters
    ----------
    job_log_path : str
        Path to job log file (e.g., job_ids_convergence_*.txt)
    
    Returns
    -------
    Dict[str, str]
        Mapping of method_ratioX_Y -> job_id
    """
    job_map = {}
    with open(job_log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                parts = line.split('=')
                if len(parts) == 2:
                    method_key = parts[0].strip()
                    job_id = parts[1].strip().split('.')[0]  # Remove .spcc-adm1
                    job_map[method_key] = job_id
    return job_map


def parse_optuna_log(log_path: str) -> Tuple[List[dict], Optional[dict]]:
    """Parse Optuna trial logs from HPC job output.
    
    Parameters
    ----------
    log_path : str
        Path to HPC job output file (*.OU)
    
    Returns
    -------
    Tuple[List[dict], Optional[dict]]
        (list of trial records, convergence summary)
    """
    trials = []
    convergence = None
    
    # Pattern: [Optuna] Trial   0: value=0.6284, best=0.6284
    trial_pattern = re.compile(
        r'\[Optuna\] Trial\s+(\d+):\s+value=([\d.]+),\s+best=([\d.]+)'
    )
    # Pattern: [Optuna Convergence] Total trials: 10, Best: 0.6464, ...
    conv_pattern = re.compile(
        r'\[Optuna Convergence\] Total trials: (\d+), Best: ([\d.]+), '
        r'Last 10 best: ([\d.]+), Improvement in last 10: ([\d.]+)'
    )
    
    try:
        with open(log_path, 'r') as f:
            for line in f:
                # Trial log
                trial_match = trial_pattern.search(line)
                if trial_match:
                    trials.append({
                        'trial': int(trial_match.group(1)),
                        'value': float(trial_match.group(2)),
                        'best': float(trial_match.group(3)),
                    })
                
                # Convergence summary
                conv_match = conv_pattern.search(line)
                if conv_match:
                    convergence = {
                        'total_trials': int(conv_match.group(1)),
                        'best': float(conv_match.group(2)),
                        'last_10_best': float(conv_match.group(3)),
                        'improvement_last_10': float(conv_match.group(4)),
                    }
    except FileNotFoundError:
        print(f"Warning: Log file not found: {log_path}")
    
    return trials, convergence


def collect_all_convergence(
    job_map: Dict[str, str],
    log_dir: str = "scripts/hpc/log"
) -> pd.DataFrame:
    """Collect convergence data from all job logs.
    
    Parameters
    ----------
    job_map : Dict[str, str]
        Method -> job_id mapping
    log_dir : str
        Directory containing HPC log files
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: method, ratio, trial, value, best, job_id
    """
    all_data = []
    
    for method_key, job_id in job_map.items():
        # Parse method and ratio from key (e.g., smote_ratio0_5 -> smote, 0.5)
        parts = method_key.rsplit('_ratio', 1)
        if len(parts) == 2:
            method = parts[0]
            ratio = parts[1].replace('_', '.')
        else:
            method = method_key
            ratio = '1.0'
        
        log_path = Path(log_dir) / f"{job_id}.spcc-adm1.OU"
        trials, convergence = parse_optuna_log(str(log_path))
        
        for trial in trials:
            all_data.append({
                'method': method,
                'ratio': ratio,
                'job_id': job_id,
                'trial': trial['trial'],
                'value': trial['value'],
                'best': trial['best'],
            })
        
        if convergence:
            print(f"{method_key}: Best={convergence['best']:.4f}, "
                  f"Last10Best={convergence['last_10_best']:.4f}, "
                  f"Improvement={convergence['improvement_last_10']:.4f}")
    
    return pd.DataFrame(all_data)


def plot_convergence_by_method(
    df: pd.DataFrame,
    output_dir: str,
    figsize: Tuple[int, int] = (14, 10)
):
    """Plot convergence curves grouped by method.
    
    Parameters
    ----------
    df : pd.DataFrame
        Convergence data
    output_dir : str
        Output directory for plots
    figsize : Tuple[int, int]
        Figure size
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    methods = df['method'].unique()
    n_methods = len(methods)
    
    # Calculate grid size
    n_cols = 3
    n_rows = (n_methods + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for idx, method in enumerate(sorted(methods)):
        ax = axes[idx]
        method_df = df[df['method'] == method]
        
        for ratio in sorted(method_df['ratio'].unique()):
            ratio_df = method_df[method_df['ratio'] == ratio]
            ax.plot(ratio_df['trial'], ratio_df['best'], 
                   label=f'ratio={ratio}', marker='o', markersize=3)
        
        ax.set_title(method.replace('_', ' ').title())
        ax.set_xlabel('Trial')
        ax.set_ylabel('Best F2 Score')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_methods, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Optuna Convergence by Method', fontsize=14, y=1.02)
    plt.tight_layout()
    
    save_path = output_path / 'convergence_by_method.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_convergence_summary(
    df: pd.DataFrame,
    output_dir: str,
    figsize: Tuple[int, int] = (12, 6)
):
    """Plot summary of convergence across all methods.
    
    Parameters
    ----------
    df : pd.DataFrame
        Convergence data
    output_dir : str
        Output directory for plots
    figsize : Tuple[int, int]
        Figure size
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate convergence metrics per method/ratio
    summary = []
    for (method, ratio), group in df.groupby(['method', 'ratio']):
        best_trial = group.loc[group['best'].idxmax(), 'trial']
        final_best = group['best'].max()
        
        # Check if converged (no improvement in last 20% of trials)
        n_trials = len(group)
        last_20pct = group[group['trial'] >= n_trials * 0.8]
        improvement_last_20pct = last_20pct['best'].max() - last_20pct['best'].min()
        
        summary.append({
            'method': method,
            'ratio': ratio,
            'best_trial': best_trial,
            'final_best': final_best,
            'n_trials': n_trials,
            'improvement_last_20pct': improvement_last_20pct,
            'converged': improvement_last_20pct < 0.005,  # Threshold
        })
    
    summary_df = pd.DataFrame(summary)
    
    # Save summary CSV
    csv_path = output_path / 'convergence_summary.csv'
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    
    # Plot: Best trial distribution
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Left: Best trial by method
    ax1 = axes[0]
    pivot_df = summary_df.pivot(index='method', columns='ratio', values='best_trial')
    pivot_df.plot(kind='barh', ax=ax1)
    ax1.set_xlabel('Trial # of Best Score')
    ax1.set_title('When Best Score Was Found')
    ax1.legend(title='Ratio')
    
    # Right: Final best score by method
    ax2 = axes[1]
    pivot_df2 = summary_df.pivot(index='method', columns='ratio', values='final_best')
    pivot_df2.plot(kind='barh', ax=ax2)
    ax2.set_xlabel('Best F2 Score')
    ax2.set_title('Final Best Score')
    ax2.legend(title='Ratio')
    
    plt.tight_layout()
    
    save_path = output_path / 'convergence_summary.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    
    return summary_df


def plot_all_convergence_overlay(
    df: pd.DataFrame,
    output_dir: str,
    figsize: Tuple[int, int] = (14, 8)
):
    """Plot all convergence curves overlaid for comparison.
    
    Parameters
    ----------
    df : pd.DataFrame
        Convergence data
    output_dir : str
        Output directory for plots
    figsize : Tuple[int, int]
        Figure size
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    ratios = ['0.1', '0.5', '1.0']
    
    # Color palette for methods
    methods = sorted(df['method'].unique())
    colors = plt.cm.tab20(np.linspace(0, 1, len(methods)))
    color_map = dict(zip(methods, colors))
    
    for idx, ratio in enumerate(ratios):
        ax = axes[idx]
        ratio_df = df[df['ratio'] == ratio]
        
        for method in methods:
            method_df = ratio_df[ratio_df['method'] == method]
            if len(method_df) > 0:
                ax.plot(method_df['trial'], method_df['best'],
                       label=method.replace('_', ' '),
                       color=color_map[method], alpha=0.8)
        
        ax.set_title(f'Ratio = {ratio}')
        ax.set_xlabel('Trial')
        ax.set_ylabel('Best F2 Score')
        ax.grid(True, alpha=0.3)
        
        if idx == 2:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.suptitle('Optuna Convergence Comparison', fontsize=14)
    plt.tight_layout()
    
    save_path = output_path / 'convergence_overlay.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect and visualize Optuna convergence from job logs."
    )
    parser.add_argument(
        '--job_log',
        type=str,
        required=True,
        help='Path to job log file (job_ids_convergence_*.txt)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/imbalance_analysis/optuna_convergence',
        help='Output directory for plots and CSV'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='scripts/hpc/log',
        help='Directory containing HPC log files'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Collecting Optuna Convergence Data")
    print("=" * 60)
    
    # Parse job log
    job_map = parse_job_log(args.job_log)
    print(f"Found {len(job_map)} jobs in {args.job_log}")
    
    # Collect convergence data
    df = collect_all_convergence(job_map, args.log_dir)
    
    if len(df) == 0:
        print("No convergence data found. Check if jobs have completed.")
        return
    
    print(f"\nCollected {len(df)} trial records")
    
    # Save raw data
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    raw_csv = output_path / 'convergence_raw.csv'
    df.to_csv(raw_csv, index=False)
    print(f"Saved: {raw_csv}")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_convergence_by_method(df, args.output)
    summary_df = plot_convergence_summary(df, args.output)
    plot_all_convergence_overlay(df, args.output)
    
    # Print convergence summary
    print("\n" + "=" * 60)
    print("Convergence Summary")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    
    # Check for non-converged experiments
    not_converged = summary_df[~summary_df['converged']]
    if len(not_converged) > 0:
        print(f"\n⚠️  {len(not_converged)} experiments may not have converged:")
        for _, row in not_converged.iterrows():
            print(f"  - {row['method']} (ratio={row['ratio']}): "
                  f"best at trial {row['best_trial']}, "
                  f"improvement in last 20%: {row['improvement_last_20pct']:.4f}")


if __name__ == '__main__':
    main()
