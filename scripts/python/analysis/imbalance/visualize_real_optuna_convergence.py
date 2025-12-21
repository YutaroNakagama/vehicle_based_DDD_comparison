#!/usr/bin/env python
"""
Visualize Optuna Convergence from Real Experiment Logs
========================================================

Parses [Optuna] Trial logs from HPC job outputs and creates convergence plots.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_optuna_log(log_path: Path) -> Tuple[List[dict], str, str]:
    """Parse Optuna trial logs from HPC job output.
    
    Returns
    -------
    trials : List[dict]
        List of trial records with trial number, value, and best value
    method : str
        The sampling method used
    ratio : str
        The target ratio used
    """
    trials = []
    method = "unknown"
    ratio = "unknown"
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Extract method and ratio from header
    method_match = re.search(r'METHOD:\s*(\w+)', content)
    ratio_match = re.search(r'TARGET_RATIO:\s*([\d.]+)', content)
    
    if method_match:
        method = method_match.group(1)
    if ratio_match:
        ratio = ratio_match.group(1)
    
    # Parse trial logs
    # Pattern: [Optuna] Trial   0: value=0.1699, best=0.1699
    trial_pattern = re.compile(
        r'\[Optuna\] Trial\s+(\d+):\s+value=([\d.]+),\s+best=([\d.]+)'
    )
    
    for match in trial_pattern.finditer(content):
        trials.append({
            'trial': int(match.group(1)),
            'value': float(match.group(2)),
            'best': float(match.group(3))
        })
    
    return trials, method, ratio


def collect_all_experiments(log_dir: Path) -> pd.DataFrame:
    """Collect all optuna_conv experiment data."""
    all_records = []
    
    for log_file in log_dir.glob("*.OU"):
        try:
            with open(log_file, 'r') as f:
                first_lines = f.read(1000)
            
            # Only process optuna_conv experiments
            if "optuna_conv" not in first_lines:
                continue
            
            trials, method, ratio = parse_optuna_log(log_file)
            
            if not trials:
                continue
            
            for trial in trials:
                all_records.append({
                    'job_id': log_file.stem,
                    'method': method,
                    'ratio': ratio,
                    'method_label': f"{method} (r={ratio})",
                    'trial': trial['trial'],
                    'value': trial['value'],
                    'best': trial['best']
                })
        except Exception as e:
            continue
    
    return pd.DataFrame(all_records)


def plot_convergence_curves(df: pd.DataFrame, output_dir: Path):
    """Create convergence curve plots."""
    
    # Get unique methods
    methods = df['method_label'].unique()
    n_methods = len(methods)
    
    # Color palette
    colors = plt.cm.tab20(np.linspace(0, 1, n_methods))
    color_map = dict(zip(methods, colors))
    
    # Figure 1: All methods overlay
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    
    for method in methods:
        subset = df[df['method_label'] == method].sort_values('trial')
        ax1.plot(subset['trial'], subset['best'], 
                 label=method, color=color_map[method], linewidth=1.5, alpha=0.8)
    
    ax1.set_xlabel("Trial Number", fontsize=12)
    ax1.set_ylabel("Best Objective Value (F2 Score)", fontsize=12)
    ax1.set_title("Optuna Convergence: All Methods\n(N_TRIALS=50)", fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 50)
    
    plt.tight_layout()
    path1 = output_dir / "optuna_convergence_all_methods.png"
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    print(f"Saved: {path1}")
    plt.close()
    
    # Figure 2: Grid by method category
    method_categories = {
        'baseline': df[df['method'] == 'baseline'],
        'smote': df[df['method'].str.startswith('smote')],
        'undersample': df[df['method'].str.startswith('undersample')],
    }
    
    fig2, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, (cat_name, cat_df) in zip(axes, method_categories.items()):
        if len(cat_df) == 0:
            ax.set_visible(False)
            continue
        
        methods_in_cat = cat_df['method_label'].unique()
        for method in methods_in_cat:
            subset = cat_df[cat_df['method_label'] == method].sort_values('trial')
            ax.plot(subset['trial'], subset['best'], 
                    label=method, linewidth=1.5, alpha=0.8)
        
        ax.set_xlabel("Trial", fontsize=11)
        ax.set_ylabel("Best F2", fontsize=11)
        ax.set_title(f"{cat_name.title()}", fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 50)
    
    plt.suptitle("Optuna Convergence by Method Category", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    path2 = output_dir / "optuna_convergence_by_category.png"
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    print(f"Saved: {path2}")
    plt.close()
    
    # Figure 3: Convergence analysis - when does best value stabilize?
    fig3, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Distribution of trial where best was found
    best_trial_per_job = []
    for job_id in df['job_id'].unique():
        job_data = df[df['job_id'] == job_id].sort_values('trial')
        if len(job_data) > 0:
            final_best = job_data['best'].iloc[-1]
            # Find first trial where this best was achieved
            first_best_trial = job_data[job_data['best'] == final_best]['trial'].iloc[0]
            best_trial_per_job.append({
                'job_id': job_id,
                'method': job_data['method'].iloc[0],
                'method_label': job_data['method_label'].iloc[0],
                'best_found_at': first_best_trial,
                'final_best': final_best
            })
    
    best_df = pd.DataFrame(best_trial_per_job)
    
    ax1 = axes[0]
    ax1.hist(best_df['best_found_at'], bins=range(0, 52, 2), 
             color='steelblue', edgecolor='white', alpha=0.8)
    ax1.axvline(x=best_df['best_found_at'].median(), color='red', 
                linestyle='--', linewidth=2, label=f"Median: {best_df['best_found_at'].median():.0f}")
    ax1.axvline(x=best_df['best_found_at'].quantile(0.9), color='orange', 
                linestyle='--', linewidth=2, label=f"90th %ile: {best_df['best_found_at'].quantile(0.9):.0f}")
    ax1.set_xlabel("Trial Number Where Best Was Found", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.set_title("Distribution of Best Trial Discovery", fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Right: Improvement over trials (normalized)
    ax2 = axes[1]
    
    # Calculate relative improvement at each trial
    improvements = []
    for method in df['method_label'].unique():
        method_data = df[df['method_label'] == method].sort_values('trial')
        if len(method_data) > 0:
            initial = method_data['best'].iloc[0]
            final = method_data['best'].iloc[-1]
            for _, row in method_data.iterrows():
                if final != initial:
                    rel_improvement = (row['best'] - initial) / (final - initial) * 100
                else:
                    rel_improvement = 100 if row['trial'] > 0 else 0
                improvements.append({
                    'trial': row['trial'],
                    'method': method,
                    'rel_improvement': rel_improvement
                })
    
    imp_df = pd.DataFrame(improvements)
    avg_imp = imp_df.groupby('trial')['rel_improvement'].mean()
    
    ax2.plot(avg_imp.index, avg_imp.values, color='steelblue', linewidth=2)
    ax2.fill_between(avg_imp.index, 0, avg_imp.values, alpha=0.3)
    ax2.axhline(y=90, color='green', linestyle='--', alpha=0.7, label='90% of improvement')
    ax2.axhline(y=99, color='red', linestyle='--', alpha=0.7, label='99% of improvement')
    
    # Find when 90% and 99% improvement is reached
    trial_90 = avg_imp[avg_imp >= 90].index[0] if any(avg_imp >= 90) else 50
    trial_99 = avg_imp[avg_imp >= 99].index[0] if any(avg_imp >= 99) else 50
    ax2.axvline(x=trial_90, color='green', linestyle=':', alpha=0.5)
    ax2.axvline(x=trial_99, color='red', linestyle=':', alpha=0.5)
    
    ax2.set_xlabel("Trial Number", fontsize=12)
    ax2.set_ylabel("% of Final Improvement Achieved", fontsize=12)
    ax2.set_title(f"Average Improvement Over Trials\n(90% at trial {trial_90}, 99% at trial {trial_99})", 
                  fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 50)
    ax2.set_ylim(0, 105)
    
    plt.tight_layout()
    path3 = output_dir / "optuna_convergence_analysis.png"
    plt.savefig(path3, dpi=150, bbox_inches='tight')
    print(f"Saved: {path3}")
    plt.close()
    
    return best_df


def main():
    """Main function."""
    project_root = Path(__file__).parent.parent.parent.parent
    log_dir = project_root / "scripts" / "hpc" / "log"
    
    # Fallback path if running from different location
    if not log_dir.exists():
        log_dir = Path("/home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/log")
    
    output_dir = project_root / "results" / "imbalance_analysis" / "optuna_convergence"
    if not output_dir.parent.exists():
        output_dir = Path("/home/s2240011/git/ddd/vehicle_based_DDD_comparison/results/imbalance_analysis/optuna_convergence")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Optuna Convergence Analysis from Real Experiments")
    print("=" * 60)
    
    print(f"\nCollecting experiments from: {log_dir}")
    df = collect_all_experiments(log_dir)
    
    if len(df) == 0:
        print("No optuna_conv experiments found!")
        return
    
    print(f"Found {len(df['job_id'].unique())} experiments")
    print(f"Methods: {df['method_label'].unique()}")
    print(f"Total trial records: {len(df)}")
    
    # Save raw data
    csv_path = output_dir / "optuna_trials_real.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved raw data: {csv_path}")
    
    # Create plots
    print("\nCreating convergence plots...")
    best_df = plot_convergence_curves(df, output_dir)
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("Convergence Summary")
    print("=" * 60)
    print(f"Median trial where best was found: {best_df['best_found_at'].median():.0f}")
    print(f"90th percentile: {best_df['best_found_at'].quantile(0.9):.0f}")
    print(f"Max trial where best was found: {best_df['best_found_at'].max():.0f}")
    
    # Per-method summary
    print("\nBy Method:")
    method_summary = best_df.groupby('method').agg({
        'best_found_at': ['median', 'max'],
        'final_best': ['mean', 'std']
    }).round(4)
    method_summary.columns = ['median_trial', 'max_trial', 'mean_f2', 'std_f2']
    print(method_summary.to_string())
    
    # Recommendation
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    q90 = best_df['best_found_at'].quantile(0.9)
    if q90 <= 40:
        print(f"✅ N_TRIALS=50 is SUFFICIENT")
        print(f"   90% of experiments found their best by trial {q90:.0f}")
    else:
        print(f"⚠️  Consider increasing N_TRIALS")
        print(f"   90% of experiments found their best by trial {q90:.0f}")
    
    print(f"\nDone! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
