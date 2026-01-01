#!/usr/bin/env python3
"""
Confusion Matrix Analysis Module
================================

Provides core functionality for confusion matrix analysis:
- Loading and parsing evaluation results
- Generating visualizations (heatmaps)
- Creating summary tables
- Multi-seed aggregation analysis

This module contains the business logic extracted from the CLI wrapper.
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set default style
sns.set_style("whitegrid")

# Constants
DISTANCES = ['dtw', 'mmd', 'wasserstein']
LEVELS = ['out_domain', 'mid_domain', 'in_domain']
MODES = ['pooled', 'source_only', 'target_only']

# Method ordering for aggregation
METHOD_ORDER = [
    "baseline", "balanced_rf", "easy_ensemble",
    "smote", "smote_tomek", "smote_enn", "smote_rus", "smote_balanced_rf",
    "undersample_rus", "undersample_tomek", "undersample_enn",
]


# ============================================================
# Data Loading Functions
# ============================================================
def load_confusion_matrix(json_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load confusion matrix from evaluation JSON file.
    
    Args:
        json_path: Path to the evaluation JSON file.
        
    Returns:
        Tuple of (confusion_matrix, metrics_dict)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    test_results = data.get('test', {})
    
    if 'confusion_matrix' in test_results:
        cm = np.array(test_results['confusion_matrix'])
        tn, fp = cm[0]
        fn, tp = cm[1]
    else:
        tn = test_results.get('tn', 0)
        fp = test_results.get('fp', 0)
        fn = test_results.get('fn', 0)
        tp = test_results.get('tp', 0)
        cm = np.array([[tn, fp], [fn, tp]])
    
    precision = test_results.get('precision', 0)
    recall = test_results.get('recall', 0)
    f1 = test_results.get('f1', 0)
    
    return cm, {
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'precision': precision, 'recall': recall, 'f1': f1
    }


def parse_filename(filename: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Parse evaluation result filename to extract metadata.
    
    Args:
        filename: Name of the evaluation file.
        
    Returns:
        Tuple of (mode, distance, level)
    """
    mode = None
    distance = None
    level = None
    
    if 'pooled' in filename:
        mode = 'pooled'
    elif 'source_only' in filename:
        mode = 'source_only'
    elif 'target_only' in filename:
        mode = 'target_only'
    
    for d in DISTANCES:
        if d in filename:
            distance = d
            break
    
    for l in LEVELS:
        if l in filename:
            level = l
            break
    
    return mode, distance, level


def collect_eval_data(eval_dir: Path) -> Dict[str, Dict[str, Dict[str, Tuple[np.ndarray, Dict]]]]:
    """Collect evaluation data organized by distance/level/mode.
    
    Args:
        eval_dir: Directory containing evaluation files.
        
    Returns:
        Nested dictionary: data_dict[distance][level][mode] = (cm, metrics)
        
    Raises:
        FileNotFoundError: If eval_dir does not exist.
    """
    if not eval_dir.exists():
        raise FileNotFoundError(f"Evaluation directory not found: {eval_dir}")
    
    eval_files = list(eval_dir.rglob('eval_results_*.json'))
    print(f"Found {len(eval_files)} evaluation files")
    
    data_dict = defaultdict(lambda: defaultdict(dict))
    
    for eval_file in eval_files:
        mode, distance, level = parse_filename(eval_file.name)
        
        if mode and distance and level:
            cm, metrics = load_confusion_matrix(eval_file)
            data_dict[distance][level][mode] = (cm, metrics)
    
    return data_dict


# ============================================================
# Visualization Functions
# ============================================================
def plot_confusion_matrix(
    cm: np.ndarray,
    title: str,
    ax: plt.Axes,
    metrics: Optional[Dict[str, Any]] = None,
    labels: Tuple[str, str] = ('Non-Drowsy', 'Drowsy')
) -> None:
    """Plot a single confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix array (2x2).
        title: Title for the subplot.
        ax: Matplotlib axes to plot on.
        metrics: Optional metrics dict with precision, recall, f1.
        labels: Labels for the classes.
    """
    cm_norm = cm.astype('float') / cm.sum() * 100
    
    annot = np.array([
        [f'{cm[i, j]}\n({cm_norm[i, j]:.1f}%)' for j in range(2)]
        for i in range(2)
    ])
    
    sns.heatmap(
        cm, annot=annot, fmt='', cmap='Blues',
        square=True, ax=ax, cbar=False,
        xticklabels=labels,
        yticklabels=labels
    )
    
    ax.set_xlabel('Predicted', fontsize=9)
    ax.set_ylabel('True', fontsize=9)
    
    if metrics:
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)
        f1 = metrics.get('f1', 0)
        title_with_metrics = f"{title}\nP={precision:.3f} R={recall:.3f} F1={f1:.3f}"
    else:
        title_with_metrics = title
    
    ax.set_title(title_with_metrics, fontsize=10, pad=10)


def generate_distance_plot(
    data_dict: Dict,
    distance: str,
    output_path: Path,
    dpi: int = 150
) -> None:
    """Generate confusion matrix heatmaps for a single distance metric.
    
    Args:
        data_dict: Data dictionary from collect_eval_data().
        distance: Distance metric name.
        output_path: Path to save the output PNG.
        dpi: Output resolution.
    """
    if distance not in data_dict:
        print(f"Warning: No data for {distance}")
        return
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(
        f'Confusion Matrices - {distance.upper()}',
        fontsize=16, fontweight='bold', y=0.995
    )
    
    for i, level in enumerate(LEVELS):
        for j, mode in enumerate(MODES):
            ax = axes[i, j]
            
            if level in data_dict[distance] and mode in data_dict[distance][level]:
                cm, metrics = data_dict[distance][level][mode]
                title = f"{level.upper()} - {mode.replace('_', ' ').title()}"
                plot_confusion_matrix(cm, title, ax, metrics)
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=12)
                ax.set_title(f"{level.upper()} - {mode.replace('_', ' ').title()}")
                ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def generate_overview_plot(
    data_dict: Dict,
    output_path: Path,
    mode: str = 'pooled',
    dpi: int = 150
) -> None:
    """Generate combined overview of all distances for a specific mode.
    
    Args:
        data_dict: Data dictionary from collect_eval_data().
        output_path: Path to save the output PNG.
        mode: Evaluation mode to visualize.
        dpi: Output resolution.
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(
        f'Confusion Matrices - All Distances ({mode.replace("_", " ").title()} Mode)',
        fontsize=16, fontweight='bold', y=0.995
    )
    
    for i, distance in enumerate(DISTANCES):
        for j, level in enumerate(LEVELS):
            ax = axes[i, j]
            
            if (distance in data_dict and
                level in data_dict[distance] and
                mode in data_dict[distance][level]):
                cm, metrics = data_dict[distance][level][mode]
                title = f"{distance.upper()} - {level.upper()}"
                plot_confusion_matrix(cm, title, ax, metrics)
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=12)
                ax.set_title(f"{distance.upper()} - {level.upper()}")
                ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


# ============================================================
# Table Generation Functions
# ============================================================
def generate_summary_table(data_dict: Dict) -> pd.DataFrame:
    """Generate summary table from collected evaluation data.
    
    Args:
        data_dict: Data dictionary from collect_eval_data().
        
    Returns:
        DataFrame with summary statistics.
    """
    summary_rows = []
    
    for distance in DISTANCES:
        if distance not in data_dict:
            continue
        for level in LEVELS:
            if level not in data_dict[distance]:
                continue
            for mode in MODES:
                if mode in data_dict[distance][level]:
                    _, m = data_dict[distance][level][mode]
                    summary_rows.append({
                        'Distance': distance,
                        'Level': level,
                        'Mode': mode,
                        'TN': m['tn'],
                        'FP': m['fp'],
                        'FN': m['fn'],
                        'TP': m['tp'],
                        'Precision': f"{m['precision']:.4f}",
                        'Recall': f"{m['recall']:.4f}",
                        'F1': f"{m['f1']:.4f}"
                    })
    
    return pd.DataFrame(summary_rows)


def print_detailed_tables(data_dict: Dict) -> None:
    """Print detailed tables to console for each distance/level combination.
    
    Args:
        data_dict: Data dictionary from collect_eval_data().
    """
    for distance in DISTANCES:
        if distance not in data_dict:
            continue
        
        print("=" * 100)
        print(f"Distance Metric: {distance.upper()}")
        print("=" * 100)
        
        for level in LEVELS:
            if level not in data_dict[distance]:
                continue
            
            print(f"\n--- Level: {level.upper()} ---")
            
            rows = []
            for mode in MODES:
                if mode in data_dict[distance][level]:
                    _, m = data_dict[distance][level][mode]
                    row = {
                        'Mode': mode,
                        'TN': m['tn'],
                        'FP': m['fp'],
                        'FN': m['fn'],
                        'TP': m['tp'],
                        'Precision': f"{m['precision']:.4f}",
                        'Recall': f"{m['recall']:.4f}",
                        'F1': f"{m['f1']:.4f}"
                    }
                    rows.append(row)
            
            if rows:
                df = pd.DataFrame(rows)
                print(df.to_string(index=False))
                
                total = rows[0]['TN'] + rows[0]['FP'] + rows[0]['FN'] + rows[0]['TP']
                drowsy = rows[0]['FN'] + rows[0]['TP']
                non_drowsy = rows[0]['TN'] + rows[0]['FP']
                print(f"\nTotal samples: {total} (Non-drowsy: {non_drowsy}, Drowsy: {drowsy})")
        
        print()


# ============================================================
# Multi-seed Aggregation Functions
# ============================================================
def extract_multiseed_results(eval_dir: Path) -> pd.DataFrame:
    """Extract confusion matrix and metrics from multi-seed evaluation files.
    
    Args:
        eval_dir: Root evaluation directory.
        
    Returns:
        DataFrame with extracted results.
    """
    records = []
    
    for model_type in ["RF", "BalancedRF", "EasyEnsemble"]:
        type_dir = eval_dir / model_type
        if not type_dir.exists():
            continue
        
        for job_dir in type_dir.iterdir():
            if not job_dir.is_dir() or job_dir.name.endswith('.txt'):
                continue
            
            for sub_dir in job_dir.iterdir():
                if not sub_dir.is_dir():
                    continue
                
                for json_file in sub_dir.glob("*.json"):
                    record = _parse_multiseed_file(json_file, model_type)
                    if record:
                        records.append(record)
    
    return pd.DataFrame(records)


def _parse_multiseed_file(
    json_file: Path,
    model_type: str
) -> Optional[Dict[str, Any]]:
    """Parse a single multi-seed evaluation file.
    
    Args:
        json_file: Path to the JSON file.
        model_type: Type of model (RF, BalancedRF, etc.).
        
    Returns:
        Record dict or None if parsing fails.
    """
    try:
        data = json.loads(json_file.read_text())
        tag = data.get("tag", "")
        
        if not tag.startswith("imbal_v2_"):
            return None
        
        tag_part = tag.replace("imbal_v2_", "")
        
        # Extract seed
        seed_match = re.search(r"seed(\d+)", tag_part)
        if seed_match:
            seed = seed_match.group(1)
            method_part = re.sub(r"_seed\d+", "", tag_part)
        else:
            base_method = tag_part.split("_ratio")[0]
            if base_method in ["easy_ensemble", "balanced_rf", "baseline"]:
                seed = "42"
                method_part = tag_part
            else:
                return None
        
        # Skip fixed variants
        if "_fixed" in method_part or "_fix" in method_part:
            return None
        
        # Extract ratio
        ratio_match = re.search(r"ratio(\d+)_(\d+)", method_part)
        
        if not ratio_match and method_part not in ["baseline", "easy_ensemble", "balanced_rf"]:
            return None
        
        if ratio_match:
            ratio = float(f"{ratio_match.group(1)}.{ratio_match.group(2)}")
            method = method_part.replace(f"_ratio{ratio_match.group(1)}_{ratio_match.group(2)}", "")
        else:
            ratio = None
            method = method_part
        
        # Create label
        if method == "baseline":
            method_label = "Baseline"
        elif method == "easy_ensemble":
            method_label = "EasyEnsemble"
        elif method == "balanced_rf":
            method_label = "BalancedRF"
        elif ratio is not None:
            method_label = f"{method} (r={ratio})"
        else:
            method_label = method
        
        # Extract confusion matrix
        cm = data.get("confusion_matrix", [[0, 0], [0, 0]])
        tn, fp = cm[0]
        fn, tp = cm[1]
        
        return {
            "method": method,
            "ratio": ratio,
            "seed": seed,
            "method_label": method_label,
            "model_type": model_type,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "total": tn + fp + fn + tp,
            "recall": data.get("recall_thr", 0),
            "specificity": data.get("specificity_thr", 0),
            "precision": data.get("prec_thr", 0),
            "f2": data.get("f2_thr", 0),
            "tag": tag,
        }
    except Exception:
        return None


def aggregate_multiseed_results(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate multi-seed results by method.
    
    Args:
        df: DataFrame from extract_multiseed_results().
        
    Returns:
        Aggregated DataFrame.
    """
    if df.empty:
        return df
    
    def get_sort_key(row):
        method = row["method"]
        ratio = row["ratio"] if row["ratio"] is not None else 0
        try:
            method_idx = METHOD_ORDER.index(method)
        except ValueError:
            method_idx = len(METHOD_ORDER)
        return (method_idx, -float(ratio) if ratio else 0)
    
    df["sort_key"] = df.apply(get_sort_key, axis=1)
    
    agg_df = df.groupby(["method_label", "method", "ratio"]).agg({
        "tn": ["sum", "mean", "std"],
        "fp": ["sum", "mean", "std"],
        "fn": ["sum", "mean", "std"],
        "tp": ["sum", "mean", "std"],
        "total": "sum",
        "recall": "mean",
        "specificity": "mean",
        "precision": "mean",
        "f2": "mean",
        "seed": "count",
        "sort_key": "first"
    }).reset_index()
    
    agg_df.columns = [
        "method_label", "method", "ratio",
        "tn_sum", "tn_mean", "tn_std",
        "fp_sum", "fp_mean", "fp_std",
        "fn_sum", "fn_mean", "fn_std",
        "tp_sum", "tp_mean", "tp_std",
        "total", "recall", "specificity", "precision", "f2", "n_seeds", "sort_key"
    ]
    
    return agg_df.sort_values("sort_key")


def generate_rates_visualization(
    agg_df: pd.DataFrame,
    output_path: Path,
    dpi: int = 150
) -> None:
    """Generate confusion matrix rates visualization.
    
    Args:
        agg_df: Aggregated DataFrame from aggregate_multiseed_results().
        output_path: Path to save the output PNG.
        dpi: Output resolution.
    """
    n_methods = len(agg_df)
    fig, axes = plt.subplots(1, 2, figsize=(12, max(10, n_methods * 0.5)))
    
    methods = agg_df["method_label"].values
    recall_vals = agg_df["recall"].values * 100
    fnr_vals = (1 - agg_df["recall"].values) * 100
    specificity_vals = agg_df["specificity"].values * 100
    fpr_vals = (1 - agg_df["specificity"].values) * 100
    
    y_pos = np.arange(n_methods)
    
    # Left panel: Drowsy class
    ax1 = axes[0]
    ax1.barh(y_pos, recall_vals, color='#27ae60', label='TP Rate (Recall)', alpha=0.8)
    ax1.barh(y_pos, -fnr_vals, color='#e74c3c', label='FN Rate (Missed)', alpha=0.8)
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(methods)
    ax1.set_xlabel("Rate (%)")
    ax1.set_title("Drowsy Class (Positive)\nRecall vs Miss Rate", fontsize=12, fontweight='bold')
    ax1.axvline(x=0, color='black', linewidth=0.5)
    ax1.set_xlim(-100, 100)
    ax1.legend(loc='lower right', fontsize=9)
    
    for i, (r, f) in enumerate(zip(recall_vals, fnr_vals)):
        if r > 5:
            ax1.text(r / 2, i, f"{r:.0f}%", ha='center', va='center',
                     fontsize=8, color='white', fontweight='bold')
        if f > 5:
            ax1.text(-f / 2, i, f"{f:.0f}%", ha='center', va='center',
                     fontsize=8, color='white', fontweight='bold')
    
    # Right panel: Alert class
    ax2 = axes[1]
    ax2.barh(y_pos, specificity_vals, color='#3498db', label='TN Rate (Specificity)', alpha=0.8)
    ax2.barh(y_pos, -fpr_vals, color='#e67e22', label='FP Rate (False Alarm)', alpha=0.8)
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])
    ax2.set_xlabel("Rate (%)")
    ax2.set_title("Alert Class (Negative)\nSpecificity vs False Alarm Rate",
                  fontsize=12, fontweight='bold')
    ax2.axvline(x=0, color='black', linewidth=0.5)
    ax2.set_xlim(-100, 100)
    ax2.legend(loc='lower right', fontsize=9)
    
    for i, (s, f) in enumerate(zip(specificity_vals, fpr_vals)):
        if s > 5:
            ax2.text(s / 2, i, f"{s:.0f}%", ha='center', va='center',
                     fontsize=8, color='white', fontweight='bold')
        if f > 5:
            ax2.text(-f / 2, i, f"{f:.0f}%", ha='center', va='center',
                     fontsize=8, color='white', fontweight='bold')
    
    plt.suptitle(
        "Confusion Matrix Rates Summary\nMulti-seed Comparison",
        fontsize=14, fontweight='bold', y=1.02
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_aggregate_summary(agg_df: pd.DataFrame) -> pd.DataFrame:
    """Create summary DataFrame for aggregated results.
    
    Args:
        agg_df: Aggregated DataFrame from aggregate_multiseed_results().
        
    Returns:
        Summary DataFrame.
    """
    summary_data = []
    
    for _, row in agg_df.iterrows():
        tn, fp = row["tn_mean"], row["fp_mean"]
        fn, tp = row["fn_mean"], row["tp_mean"]
        tn_std, fp_std = row["tn_std"], row["fp_std"]
        fn_std, tp_std = row["fn_std"], row["tp_std"]
        
        total_neg = tn + fp
        total_pos = fn + tp
        fpr = fp / total_neg * 100 if total_neg > 0 else 0
        fnr = fn / total_pos * 100 if total_pos > 0 else 0
        
        summary_data.append({
            "Method": row["method_label"],
            "Seeds": int(row["n_seeds"]),
            "TN": f"{tn:.0f} ± {tn_std:.0f}" if not np.isnan(tn_std) else f"{tn:.0f}",
            "FP": f"{fp:.0f} ± {fp_std:.0f}" if not np.isnan(fp_std) else f"{fp:.0f}",
            "FN": f"{fn:.0f} ± {fn_std:.0f}" if not np.isnan(fn_std) else f"{fn:.0f}",
            "TP": f"{tp:.0f} ± {tp_std:.0f}" if not np.isnan(tp_std) else f"{tp:.0f}",
            "Recall (%)": f"{row['recall'] * 100:.1f}",
            "Spec (%)": f"{row['specificity'] * 100:.1f}",
            "Prec (%)": f"{row['precision'] * 100:.1f}",
            "F2": f"{row['f2']:.3f}",
            "FPR (%)": f"{fpr:.1f}",
            "FNR (%)": f"{fnr:.1f}",
        })
    
    return pd.DataFrame(summary_data)
