#!/usr/bin/env python3
"""Visualize hyperparameter convergence across experiments.

This script extracts best hyperparameters from log files and visualizes
where they fall within the search ranges.
"""

import re
import json
import os
import sys
from pathlib import Path
from collections import defaultdict
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Search ranges (must match common.py definitions)
# ============================================================
SEARCH_RANGES = {
    "RF": {
        "n_estimators": {"type": "int", "low": 50, "high": 1000},
        "max_depth": {"type": "categorical", "choices": [None, 10, 20, 30, 50, 100]},
        "min_samples_split": {"type": "int", "low": 2, "high": 100},
        "min_samples_leaf": {"type": "int", "low": 1, "high": 50},
        "max_features": {"type": "categorical", "choices": ["sqrt", "log2", None]},
        "class_weight": {"type": "categorical", "choices": ["balanced", "balanced_subsample", None]},
        "max_samples": {"type": "categorical", "choices": [None, 0.5, 0.7, 0.9]},
    },
    "BalancedRF": {
        "n_estimators": {"type": "int", "low": 50, "high": 500},
        "max_depth": {"type": "categorical", "choices": [None, 10, 20, 30, 50]},
        "min_samples_split": {"type": "int", "low": 2, "high": 50},
        "min_samples_leaf": {"type": "int", "low": 1, "high": 20},
        "max_features": {"type": "categorical", "choices": ["sqrt", "log2", None]},
        "sampling_strategy": {"type": "categorical", "choices": ["not majority", "all", "auto"]},
        "replacement": {"type": "categorical", "choices": [True, False]},
    },
    "EasyEnsemble": {
        "n_estimators": {"type": "int", "low": 5, "high": 100},
        "sampling_strategy": {"type": "categorical", "choices": ["not majority", "all", "auto", "majority"]},
    },
}


def parse_hyperparams_from_log(log_file: Path) -> list:
    """Extract hyperparameters from a log file.
    
    Returns a list of (method, seed, ratio, params_dict) tuples.
    """
    results = []
    
    # Parse job info from filename
    # Format: 14609649[57].spcc-adm1.OU
    filename = log_file.name
    
    try:
        with open(log_file, 'r', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {log_file}: {e}")
        return results
    
    # Extract method, seed, ratio from log content
    method_match = re.search(r'--oversample[_-]method\s+(\S+)', content)
    seed_match = re.search(r'--seed\s+(\d+)', content)
    ratio_match = re.search(r'--target[_-]ratio\s+([\d.]+)', content)
    model_match = re.search(r'--model\s+(\S+)', content)
    
    method = method_match.group(1) if method_match else "baseline"
    seed = int(seed_match.group(1)) if seed_match else 42
    ratio = float(ratio_match.group(1)) if ratio_match else 1.0
    model = model_match.group(1) if model_match else "RF"
    
    # Extract all "Best hyperparameters" entries
    pattern = r'Best hyperparameters:\s*(\{[^}]+\})'
    matches = re.findall(pattern, content)
    
    for match in matches:
        try:
            # Convert Python dict string to proper JSON
            # Handle None -> null, True/False -> true/false
            json_str = match.replace("'", '"')
            json_str = re.sub(r'\bNone\b', 'null', json_str)
            json_str = re.sub(r'\bTrue\b', 'true', json_str)
            json_str = re.sub(r'\bFalse\b', 'false', json_str)
            
            params = json.loads(json_str)
            results.append({
                "method": method,
                "seed": seed,
                "ratio": ratio,
                "model": model,
                "params": params,
                "log_file": str(log_file),
            })
        except json.JSONDecodeError as e:
            print(f"JSON parse error in {log_file}: {e}")
            continue
    
    return results


def collect_all_hyperparams(log_dir: Path) -> pd.DataFrame:
    """Collect hyperparameters from all log files."""
    all_results = []
    
    log_files = list(log_dir.glob("*.OU"))
    print(f"Found {len(log_files)} log files")
    
    for log_file in log_files:
        results = parse_hyperparams_from_log(log_file)
        all_results.extend(results)
    
    if not all_results:
        print("No hyperparameters found in logs")
        return pd.DataFrame()
    
    # Flatten params into columns
    rows = []
    for r in all_results:
        row = {
            "method": r["method"],
            "seed": r["seed"],
            "ratio": r["ratio"],
            "model": r["model"],
        }
        for k, v in r["params"].items():
            row[f"param_{k}"] = v
        rows.append(row)
    
    df = pd.DataFrame(rows)
    print(f"Collected {len(df)} hyperparameter sets")
    return df


def normalize_param_position(value, param_spec: dict) -> float:
    """Normalize parameter value to [0, 1] range within search space."""
    if param_spec["type"] == "int":
        low, high = param_spec["low"], param_spec["high"]
        if value is None:
            return 0.5  # None treated as middle
        return (value - low) / (high - low)
    elif param_spec["type"] == "categorical":
        choices = param_spec["choices"]
        if value in choices:
            idx = choices.index(value)
            return idx / (len(choices) - 1) if len(choices) > 1 else 0.5
        return 0.5  # Unknown value
    return 0.5


def plot_param_distributions(df: pd.DataFrame, model: str, output_dir: Path):
    """Plot parameter distributions for a specific model."""
    if model not in SEARCH_RANGES:
        print(f"No search ranges defined for model: {model}")
        return
    
    model_df = df[df["model"] == model].copy()
    if len(model_df) == 0:
        print(f"No data for model: {model}")
        return
    
    search_ranges = SEARCH_RANGES[model]
    param_cols = [c for c in model_df.columns if c.startswith("param_")]
    
    # Filter to params that exist in search ranges
    valid_params = []
    for col in param_cols:
        param_name = col.replace("param_", "")
        if param_name in search_ranges:
            valid_params.append((col, param_name))
    
    if not valid_params:
        print(f"No matching parameters for model: {model}")
        return
    
    n_params = len(valid_params)
    n_cols = min(3, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_params == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (col, param_name) in enumerate(valid_params):
        ax = axes[idx]
        spec = search_ranges[param_name]
        
        values = model_df[col].dropna()
        
        if spec["type"] == "int":
            # Histogram for integer params
            low, high = spec["low"], spec["high"]
            bins = min(30, high - low + 1)
            ax.hist(values, bins=bins, edgecolor='black', alpha=0.7)
            ax.axvline(low, color='red', linestyle='--', label=f'Min: {low}')
            ax.axvline(high, color='red', linestyle='--', label=f'Max: {high}')
            ax.set_xlabel(param_name)
            ax.set_ylabel('Count')
            
            # Highlight if hitting bounds
            at_low = (values == low).sum()
            at_high = (values == high).sum()
            title = f"{param_name}"
            if at_low > 0 or at_high > 0:
                title += f" ⚠️ ({at_low} at min, {at_high} at max)"
            ax.set_title(title)
            
        elif spec["type"] == "categorical":
            # Bar chart for categorical params
            choices = spec["choices"]
            counts = values.value_counts()
            
            # Ensure all choices are represented
            all_counts = {str(c): counts.get(c, 0) for c in choices}
            
            bars = ax.bar(list(all_counts.keys()), list(all_counts.values()), 
                         edgecolor='black', alpha=0.7)
            ax.set_xlabel(param_name)
            ax.set_ylabel('Count')
            ax.set_title(f"{param_name}")
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Hide unused axes
    for idx in range(len(valid_params), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f"{model} Hyperparameter Distributions (n={len(model_df)})", fontsize=14)
    plt.tight_layout()
    
    output_file = output_dir / f"hyperparam_dist_{model}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def plot_param_by_method(df: pd.DataFrame, model: str, output_dir: Path):
    """Plot parameter distributions grouped by method."""
    if model not in SEARCH_RANGES:
        return
    
    model_df = df[df["model"] == model].copy()
    if len(model_df) == 0:
        return
    
    search_ranges = SEARCH_RANGES[model]
    
    # Focus on key numeric parameters
    key_params = ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"]
    
    for param_name in key_params:
        col = f"param_{param_name}"
        if col not in model_df.columns:
            continue
        if param_name not in search_ranges:
            continue
        
        spec = search_ranges[param_name]
        if spec["type"] != "int":
            continue
        
        # Group by method
        methods = model_df["method"].unique()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        data_for_plot = []
        labels = []
        
        for method in sorted(methods):
            method_data = model_df[model_df["method"] == method][col].dropna()
            if len(method_data) > 0:
                data_for_plot.append(method_data.values)
                labels.append(f"{method}\n(n={len(method_data)})")
        
        if data_for_plot:
            bp = ax.boxplot(data_for_plot, labels=labels, patch_artist=True)
            
            # Color boxes
            colors = plt.cm.tab10(np.linspace(0, 1, len(data_for_plot)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Add search range lines
            low, high = spec["low"], spec["high"]
            ax.axhline(low, color='red', linestyle='--', alpha=0.5, label=f'Search min: {low}')
            ax.axhline(high, color='red', linestyle='--', alpha=0.5, label=f'Search max: {high}')
            
            ax.set_ylabel(param_name)
            ax.set_title(f"{model}: {param_name} by Method")
            ax.legend(loc='upper right')
            plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            output_file = output_dir / f"hyperparam_by_method_{model}_{param_name}.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: {output_file}")


def plot_boundary_analysis(df: pd.DataFrame, model: str, output_dir: Path):
    """Analyze and visualize parameters hitting search boundaries."""
    if model not in SEARCH_RANGES:
        return
    
    model_df = df[df["model"] == model].copy()
    if len(model_df) == 0:
        return
    
    search_ranges = SEARCH_RANGES[model]
    
    boundary_stats = []
    
    for param_name, spec in search_ranges.items():
        col = f"param_{param_name}"
        if col not in model_df.columns:
            continue
        
        values = model_df[col].dropna()
        n_total = len(values)
        
        if spec["type"] == "int":
            low, high = spec["low"], spec["high"]
            at_low = (values == low).sum()
            at_high = (values == high).sum()
            pct_at_bounds = 100 * (at_low + at_high) / n_total if n_total > 0 else 0
            
            boundary_stats.append({
                "param": param_name,
                "type": "int",
                "at_low": at_low,
                "at_high": at_high,
                "pct_at_bounds": pct_at_bounds,
                "range": f"[{low}, {high}]",
            })
        elif spec["type"] == "categorical":
            most_common = values.mode().iloc[0] if len(values) > 0 else None
            pct_most_common = 100 * (values == most_common).sum() / n_total if n_total > 0 else 0
            
            boundary_stats.append({
                "param": param_name,
                "type": "categorical",
                "most_common": str(most_common),
                "pct_most_common": pct_most_common,
                "choices": str(spec["choices"]),
            })
    
    # Create summary table
    stats_df = pd.DataFrame(boundary_stats)
    
    if len(stats_df) > 0:
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        int_params = stats_df[stats_df["type"] == "int"]
        if len(int_params) > 0:
            x = range(len(int_params))
            width = 0.35
            
            ax.bar([i - width/2 for i in x], int_params["at_low"], width, 
                   label="At Lower Bound", color='blue', alpha=0.7)
            ax.bar([i + width/2 for i in x], int_params["at_high"], width,
                   label="At Upper Bound", color='orange', alpha=0.7)
            
            ax.set_xticks(x)
            ax.set_xticklabels(int_params["param"], rotation=45, ha='right')
            ax.set_ylabel("Count")
            ax.set_title(f"{model}: Parameters Hitting Search Boundaries")
            ax.legend()
            
            # Add annotations for ranges
            for i, row in enumerate(int_params.itertuples()):
                ax.annotate(row.range, (i, max(row.at_low, row.at_high) + 1),
                           ha='center', fontsize=8)
            
            plt.tight_layout()
            output_file = output_dir / f"boundary_analysis_{model}.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: {output_file}")
        
        # Save stats as CSV
        csv_file = output_dir / f"boundary_stats_{model}.csv"
        stats_df.to_csv(csv_file, index=False)
        print(f"Saved: {csv_file}")


def main():
    parser = argparse.ArgumentParser(description="Visualize hyperparameter convergence")
    parser.add_argument("--log-dir", type=str, default="logs/hpc",
                       help="Directory containing HPC log files")
    parser.add_argument("--output-dir", type=str, default="results/hyperparam_analysis",
                       help="Output directory for plots")
    parser.add_argument("--model", type=str, default=None,
                       help="Specific model to analyze (default: all)")
    args = parser.parse_args()
    
    log_dir = PROJECT_ROOT / args.log_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Collecting hyperparameters from: {log_dir}")
    df = collect_all_hyperparams(log_dir)
    
    if len(df) == 0:
        print("No data to visualize")
        return
    
    # Save raw data
    df.to_csv(output_dir / "hyperparams_raw.csv", index=False)
    print(f"Saved raw data: {output_dir / 'hyperparams_raw.csv'}")
    
    # Print summary
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    print(f"Total experiments: {len(df)}")
    print(f"Models: {df['model'].unique().tolist()}")
    print(f"Methods: {df['method'].unique().tolist()}")
    print(f"Seeds: {df['seed'].unique().tolist()}")
    print(f"Ratios: {df['ratio'].unique().tolist()}")
    
    # Generate plots for each model
    models = [args.model] if args.model else df["model"].unique()
    
    for model in models:
        print(f"\n{'='*60}")
        print(f"Analyzing: {model}")
        print("="*60)
        
        plot_param_distributions(df, model, output_dir)
        plot_param_by_method(df, model, output_dir)
        plot_boundary_analysis(df, model, output_dir)
    
    print(f"\n✅ Analysis complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
