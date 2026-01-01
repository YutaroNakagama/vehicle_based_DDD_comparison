#!/usr/bin/env python3
"""
Optuna Study Convergence Visualization

This script loads saved Optuna study data (pickle or JSON) and generates
convergence plots for hyperparameters and objective values.

Usage:
    python visualize_optuna_convergence.py --study-dir models/RF/14621011
    python visualize_optuna_convergence.py --json-pattern "models/RF/**/optuna_*_convergence.json"
    python visualize_optuna_convergence.py --aggregate-dir models/RF

Author: Vehicle-based DDD Comparison Project
"""

import argparse
import json
import glob
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_study_from_pickle(pkl_path: Path) -> Optional[Any]:
    """Load Optuna study from pickle file."""
    try:
        with open(pkl_path, 'rb') as f:
            study = pickle.load(f)
        return study
    except Exception as e:
        logger.warning(f"Failed to load {pkl_path}: {e}")
        return None


def load_convergence_from_json(json_path: Path) -> Optional[Dict]:
    """Load convergence data from JSON file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.warning(f"Failed to load {json_path}: {e}")
        return None


def plot_objective_convergence(trials: List[Dict], output_path: Path, title: str = "F2 Score Convergence"):
    """Plot objective value convergence over trials."""
    if not trials:
        logger.warning("No trials to plot")
        return
    
    trial_nums = [t['trial_number'] for t in trials]
    values = [t['value'] for t in trials]
    best_so_far = [t['best_so_far'] for t in trials]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Individual trial values
    ax.scatter(trial_nums, values, alpha=0.5, s=30, label='Trial Value', color='#1f77b4')
    
    # Best so far line
    ax.plot(trial_nums, best_so_far, color='#d62728', linewidth=2, label='Best So Far')
    
    # Highlight best trial
    best_idx = np.argmax(values)
    ax.scatter([trial_nums[best_idx]], [values[best_idx]], 
               s=200, marker='*', color='gold', edgecolor='black', 
               zorder=5, label=f'Best (Trial {trial_nums[best_idx]})')
    
    ax.set_xlabel('Trial Number', fontsize=12)
    ax.set_ylabel('F2 Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def plot_hyperparameter_convergence(trials: List[Dict], output_path: Path, title: str = "Hyperparameter Convergence"):
    """Plot hyperparameter values over trials with objective value coloring."""
    if not trials:
        logger.warning("No trials to plot")
        return
    
    # Extract all hyperparameters
    all_params = set()
    for t in trials:
        all_params.update(t['params'].keys())
    all_params = sorted(all_params)
    
    if not all_params:
        logger.warning("No hyperparameters found")
        return
    
    # Create subplots
    n_params = len(all_params)
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.atleast_2d(axes).flatten()
    
    trial_nums = [t['trial_number'] for t in trials]
    values = [t['value'] for t in trials]
    
    # Normalize values for color mapping
    vmin, vmax = min(values), max(values)
    norm_values = [(v - vmin) / (vmax - vmin + 1e-10) for v in values]
    
    cmap = plt.cm.RdYlGn  # Red (low) -> Yellow -> Green (high)
    
    for idx, param in enumerate(all_params):
        ax = axes[idx]
        
        param_values = [t['params'].get(param, np.nan) for t in trials]
        
        # Check if categorical
        if isinstance(param_values[0], str):
            # Encode categorical values
            unique_vals = sorted(set(v for v in param_values if v is not None))
            val_to_num = {v: i for i, v in enumerate(unique_vals)}
            numeric_vals = [val_to_num.get(v, np.nan) for v in param_values]
            
            scatter = ax.scatter(trial_nums, numeric_vals, c=norm_values, cmap=cmap, 
                                alpha=0.7, s=40, edgecolor='black', linewidth=0.5)
            ax.set_yticks(range(len(unique_vals)))
            ax.set_yticklabels(unique_vals)
        else:
            scatter = ax.scatter(trial_nums, param_values, c=norm_values, cmap=cmap,
                                alpha=0.7, s=40, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Trial')
        ax.set_ylabel(param.replace('_', ' ').title())
        ax.set_title(param.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_params, len(axes)):
        axes[idx].set_visible(False)
    
    # Add colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('F2 Score', fontsize=10)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def plot_hyperparameter_importance(study, output_path: Path, title: str = "Hyperparameter Importance"):
    """Plot hyperparameter importance using Optuna's built-in analysis."""
    try:
        import optuna
        from optuna.importance import get_param_importances
        
        importances = get_param_importances(study)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        params = list(importances.keys())
        values = list(importances.values())
        
        # Sort by importance
        sorted_indices = np.argsort(values)[::-1]
        params = [params[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(params)))
        
        bars = ax.barh(range(len(params)), values, color=colors)
        ax.set_yticks(range(len(params)))
        ax.set_yticklabels([p.replace('_', ' ').title() for p in params])
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {output_path}")
        
    except Exception as e:
        logger.warning(f"Failed to compute importance: {e}")


def plot_parallel_coordinate(trials: List[Dict], output_path: Path, title: str = "Parallel Coordinate Plot"):
    """Create parallel coordinate plot for hyperparameters."""
    if not trials:
        return
    
    # Build dataframe
    records = []
    for t in trials:
        record = {'trial': t['trial_number'], 'value': t['value']}
        record.update(t['params'])
        records.append(record)
    
    df = pd.DataFrame(records)
    
    # Select numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'trial' in numeric_cols:
        numeric_cols.remove('trial')
    
    if len(numeric_cols) < 2:
        logger.warning("Not enough numeric parameters for parallel coordinate plot")
        return
    
    try:
        from pandas.plotting import parallel_coordinates
        
        # Discretize value into bins for coloring
        df['score_bin'] = pd.qcut(df['value'], q=5, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4', 'Q5 (High)'])
        
        fig, ax = plt.subplots(figsize=(14, 6))
        parallel_coordinates(df[numeric_cols + ['score_bin']], 'score_bin', 
                           colormap=plt.cm.RdYlGn, alpha=0.5, ax=ax)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {output_path}")
        
    except Exception as e:
        logger.warning(f"Failed to create parallel coordinate plot: {e}")


def analyze_single_study(study_dir: Path, output_dir: Path = None):
    """Analyze a single study directory."""
    if output_dir is None:
        output_dir = study_dir / 'convergence_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find study files
    json_files = list(study_dir.glob("*_convergence.json"))
    pkl_files = list(study_dir.glob("*_study.pkl"))
    
    for json_file in json_files:
        data = load_convergence_from_json(json_file)
        if data is None:
            continue
        
        metadata = data.get('metadata', {})
        trials = data.get('trials', [])
        
        base_name = json_file.stem.replace('_convergence', '')
        title_prefix = f"{metadata.get('model', 'Unknown')} ({metadata.get('mode', '')}, seed={metadata.get('seed', '')})"
        
        # Plot objective convergence
        plot_objective_convergence(
            trials, 
            output_dir / f"{base_name}_f2_convergence.png",
            title=f"{title_prefix} - F2 Convergence"
        )
        
        # Plot hyperparameter convergence
        plot_hyperparameter_convergence(
            trials,
            output_dir / f"{base_name}_param_convergence.png",
            title=f"{title_prefix} - Hyperparameter Convergence"
        )
        
        # Plot parallel coordinate
        plot_parallel_coordinate(
            trials,
            output_dir / f"{base_name}_parallel_coord.png",
            title=f"{title_prefix} - Parallel Coordinates"
        )
    
    # Try to compute importance from pickle if available
    for pkl_file in pkl_files:
        study = load_study_from_pickle(pkl_file)
        if study is None:
            continue
        
        base_name = pkl_file.stem.replace('_study', '')
        
        plot_hyperparameter_importance(
            study,
            output_dir / f"{base_name}_importance.png",
            title=f"Hyperparameter Importance"
        )


def aggregate_studies(base_dir: Path, output_dir: Path):
    """Aggregate and compare multiple studies."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all convergence JSON files
    json_files = list(base_dir.rglob("*_convergence.json"))
    logger.info(f"Found {len(json_files)} convergence files")
    
    if not json_files:
        logger.warning("No convergence files found")
        return
    
    # Aggregate data
    all_trials = []
    all_metadata = []
    
    for json_file in json_files:
        data = load_convergence_from_json(json_file)
        if data is None:
            continue
        
        metadata = data.get('metadata', {})
        trials = data.get('trials', [])
        
        for t in trials:
            t['model'] = metadata.get('model', 'Unknown')
            t['mode'] = metadata.get('mode', 'Unknown')
            t['suffix'] = metadata.get('suffix', '')
            t['seed'] = metadata.get('seed', 0)
        
        all_trials.extend(trials)
        all_metadata.append(metadata)
    
    if not all_trials:
        logger.warning("No trial data found")
        return
    
    # Create aggregated plots
    df = pd.DataFrame([{
        'trial': t['trial_number'],
        'value': t['value'],
        'best_so_far': t['best_so_far'],
        'model': t['model'],
        'mode': t['mode'],
        'suffix': t['suffix'],
        'seed': t['seed'],
        'tag': f"{t['mode']}_{t['suffix']}_s{t['seed']}"
    } for t in all_trials])
    
    # Plot 1: Convergence by experiment
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for tag, group in df.groupby('tag'):
        group = group.sort_values('trial')
        ax.plot(group['trial'], group['best_so_far'], alpha=0.7, label=tag, linewidth=1.5)
    
    ax.set_xlabel('Trial Number', fontsize=12)
    ax.set_ylabel('Best F2 Score So Far', fontsize=12)
    ax.set_title('F2 Convergence Across All Experiments', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'aggregate_f2_convergence.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_dir / 'aggregate_f2_convergence.png'}")
    
    # Plot 2: Final best value comparison
    final_df = pd.DataFrame(all_metadata)
    
    if 'best_value' in final_df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        final_df['label'] = final_df.apply(
            lambda r: f"{r.get('mode', '')}_{r.get('suffix', '')}_s{r.get('seed', '')}", axis=1
        )
        final_df = final_df.sort_values('best_value', ascending=False)
        
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(final_df)))
        bars = ax.barh(range(len(final_df)), final_df['best_value'], color=colors)
        ax.set_yticks(range(len(final_df)))
        ax.set_yticklabels(final_df['label'], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Best F2 Score')
        ax.set_title('Final Best F2 Score Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        for bar, val in zip(bars, final_df['best_value']):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                   f'{val:.4f}', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'aggregate_final_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {output_dir / 'aggregate_final_comparison.png'}")
    
    # Save summary CSV
    summary_df = pd.DataFrame(all_metadata)
    summary_df.to_csv(output_dir / 'aggregate_summary.csv', index=False)
    logger.info(f"Saved: {output_dir / 'aggregate_summary.csv'}")


def main():
    parser = argparse.ArgumentParser(description='Visualize Optuna study convergence')
    parser.add_argument('--study-dir', type=str, help='Directory containing study files')
    parser.add_argument('--aggregate-dir', type=str, help='Base directory to aggregate all studies')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for plots')
    
    args = parser.parse_args()
    
    if args.study_dir:
        study_dir = Path(args.study_dir)
        output_dir = Path(args.output_dir) if args.output_dir else None
        analyze_single_study(study_dir, output_dir)
    
    elif args.aggregate_dir:
        base_dir = Path(args.aggregate_dir)
        output_dir = Path(args.output_dir) if args.output_dir else base_dir / 'convergence_aggregate'
        aggregate_studies(base_dir, output_dir)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
