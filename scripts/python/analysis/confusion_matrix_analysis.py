#!/usr/bin/env python3
"""
confusion_matrix_analysis.py
============================
Unified script for confusion matrix analysis and visualization.

This script consolidates:
- plot_confusion_matrices.py (PNG heatmap generation)
- show_confusion_matrices.py (table/CSV output)
- summarize_confusion_matrices.py (multi-seed aggregation)

Usage:
    python confusion_matrix_analysis.py plot       # Generate PNG heatmaps
    python confusion_matrix_analysis.py table      # Generate console output + CSV
    python confusion_matrix_analysis.py aggregate  # Multi-seed aggregation analysis
    python confusion_matrix_analysis.py all        # Run all modes
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EVAL_DIR = PROJECT_ROOT / "results/evaluation/RF/14357179"
OUTPUT_DIR_PNG = PROJECT_ROOT / "results/domain_analysis/summary/png/confusion_matrices"
OUTPUT_DIR_CSV = PROJECT_ROOT / "results/domain_analysis/summary/csv"
OUTPUT_DIR_MULTISEED = PROJECT_ROOT / "results/imbalance_analysis/multiseed"

DISTANCES = ['dtw', 'mmd', 'wasserstein']
LEVELS = ['out_domain', 'mid_domain', 'in_domain']
MODES = ['pooled', 'source_only', 'target_only']


# ============================================================
# Common Functions
# ============================================================
def load_confusion_matrix(json_path):
    """Load confusion matrix from evaluation JSON file."""
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
    
    return cm, {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 
                'precision': precision, 'recall': recall, 'f1': f1}


def parse_filename(filename):
    """Parse evaluation result filename to extract metadata."""
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


def collect_eval_data(eval_dir):
    """Collect evaluation data organized by distance/level/mode."""
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
# Mode: plot (PNG heatmaps)
# ============================================================
def plot_confusion_matrix(cm, title, ax, metrics=None):
    """Plot a single confusion matrix."""
    cm_norm = cm.astype('float') / cm.sum() * 100
    
    annot = np.array([[f'{cm[i,j]}\n({cm_norm[i,j]:.1f}%)' 
                       for j in range(2)] for i in range(2)])
    
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', 
                square=True, ax=ax, cbar=False,
                xticklabels=['Non-Drowsy', 'Drowsy'],
                yticklabels=['Non-Drowsy', 'Drowsy'])
    
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


def run_plot_mode(eval_dir):
    """Generate PNG heatmap visualizations."""
    print("\n=== Mode: plot (PNG heatmaps) ===")
    
    data_dict = collect_eval_data(eval_dir)
    OUTPUT_DIR_PNG.mkdir(parents=True, exist_ok=True)
    
    # Plot for each distance metric
    for distance in DISTANCES:
        if distance not in data_dict:
            print(f"Warning: No data for {distance}")
            continue
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        fig.suptitle(f'Confusion Matrices - {distance.upper()}', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        for i, level in enumerate(LEVELS):
            for j, mode in enumerate(MODES):
                ax = axes[i, j]
                
                if level in data_dict[distance] and mode in data_dict[distance][level]:
                    cm, metrics = data_dict[distance][level][mode]
                    title = f"{level.upper()} - {mode.replace('_', ' ').title()}"
                    plot_confusion_matrix(cm, title, ax, metrics)
                else:
                    ax.text(0.5, 0.5, 'No Data', 
                           ha='center', va='center', fontsize=12)
                    ax.set_title(f"{level.upper()} - {mode.replace('_', ' ').title()}")
                    ax.axis('off')
        
        plt.tight_layout()
        output_file = OUTPUT_DIR_PNG / f'confusion_matrices_{distance}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()
    
    # Combined overview (all distances for pooled only)
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('Confusion Matrices - All Distances (Pooled Mode)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    for i, distance in enumerate(DISTANCES):
        for j, level in enumerate(LEVELS):
            ax = axes[i, j]
            mode = 'pooled'
            
            if (distance in data_dict and 
                level in data_dict[distance] and 
                mode in data_dict[distance][level]):
                cm, metrics = data_dict[distance][level][mode]
                title = f"{distance.upper()} - {level.upper()}"
                plot_confusion_matrix(cm, title, ax, metrics)
            else:
                ax.text(0.5, 0.5, 'No Data', 
                       ha='center', va='center', fontsize=12)
                ax.set_title(f"{distance.upper()} - {level.upper()}")
                ax.axis('off')
    
    plt.tight_layout()
    output_file = OUTPUT_DIR_PNG / 'confusion_matrices_pooled_overview.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()
    
    print(f"\n✓ All confusion matrices saved to {OUTPUT_DIR_PNG}")


# ============================================================
# Mode: table (console + CSV)
# ============================================================
def run_table_mode(eval_dir):
    """Generate console output and CSV."""
    print("\n=== Mode: table (console + CSV) ===")
    
    data_dict = collect_eval_data(eval_dir)
    OUTPUT_DIR_CSV.mkdir(parents=True, exist_ok=True)
    
    # Display tables
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
    
    # Summary table across all cases
    print("\n" + "=" * 120)
    print("SUMMARY: All Cases")
    print("=" * 120)
    
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
    
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        print(summary_df.to_string(index=False))
        
        output_file = OUTPUT_DIR_CSV / 'confusion_matrices_all_cases.csv'
        summary_df.to_csv(output_file, index=False)
        print(f"\n✓ Saved to: {output_file}")


# ============================================================
# Mode: aggregate (multi-seed analysis)
# ============================================================
def extract_multiseed_results(eval_dir):
    """Extract confusion matrix and metrics from multi-seed evaluation JSON files."""
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
                    try:
                        data = json.loads(json_file.read_text())
                        tag = data.get("tag", "")
                        
                        if not tag.startswith("imbal_v2_"):
                            continue
                        
                        tag_part = tag.replace("imbal_v2_", "")
                        
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
                                continue
                        
                        if "_fixed" in method_part or "_fix" in method_part:
                            continue
                        
                        ratio_match = re.search(r"ratio(\d+)_(\d+)", method_part)
                        
                        if not ratio_match and method_part not in ["baseline", "easy_ensemble", "balanced_rf"]:
                            continue
                        if ratio_match:
                            ratio = float(f"{ratio_match.group(1)}.{ratio_match.group(2)}")
                            method = method_part.replace(f"_ratio{ratio_match.group(1)}_{ratio_match.group(2)}", "")
                        else:
                            ratio = None
                            method = method_part
                        
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
                        
                        cm = data.get("confusion_matrix", [[0, 0], [0, 0]])
                        tn, fp = cm[0]
                        fn, tp = cm[1]
                        
                        records.append({
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
                        })
                    except Exception:
                        continue
    
    return pd.DataFrame(records)


def run_aggregate_mode(eval_dir):
    """Run multi-seed aggregation analysis."""
    print("\n=== Mode: aggregate (multi-seed analysis) ===")
    
    OUTPUT_DIR_MULTISEED.mkdir(parents=True, exist_ok=True)
    
    print("Extracting evaluation results...")
    df = extract_multiseed_results(eval_dir)
    
    if len(df) == 0:
        print("No multi-seed results found.")
        return
    
    print(f"Found {len(df)} evaluation records")
    print(f"Seeds: {sorted(df['seed'].unique())}")
    print(f"Methods: {df['method_label'].nunique()} unique")
    
    # Method ordering
    method_order = [
        "baseline", "balanced_rf", "easy_ensemble",
        "smote", "smote_tomek", "smote_enn", "smote_rus", "smote_balanced_rf",
        "undersample_rus", "undersample_tomek", "undersample_enn",
    ]
    
    def get_sort_key(row):
        method = row["method"]
        ratio = row["ratio"] if row["ratio"] is not None else 0
        try:
            method_idx = method_order.index(method)
        except ValueError:
            method_idx = len(method_order)
        return (method_idx, -float(ratio) if ratio else 0)
    
    df["sort_key"] = df.apply(get_sort_key, axis=1)
    
    # Aggregate by method
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
    
    agg_df = agg_df.sort_values("sort_key")
    
    # Create summary
    summary_data = []
    for _, row in agg_df.iterrows():
        tn, fp, fn, tp = row["tn_mean"], row["fp_mean"], row["fn_mean"], row["tp_mean"]
        tn_std, fp_std, fn_std, tp_std = row["tn_std"], row["fp_std"], row["fn_std"], row["tp_std"]
        
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
            "Recall (%)": f"{row['recall']*100:.1f}",
            "Spec (%)": f"{row['specificity']*100:.1f}",
            "Prec (%)": f"{row['precision']*100:.1f}",
            "F2": f"{row['f2']:.3f}",
            "FPR (%)": f"{fpr:.1f}",
            "FNR (%)": f"{fnr:.1f}",
        })
    
    summary_df = pd.DataFrame(summary_data)
    csv_path = OUTPUT_DIR_MULTISEED / "confusion_matrix_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")
    
    # Print summary
    print("\n" + "=" * 120)
    print("Confusion Matrix Summary (Mean ± Std across seeds)")
    print("=" * 120)
    print(f"{'Method':<35} | {'TN':>12} | {'FP':>12} | {'FN':>10} | {'TP':>10} | Recall | Spec | FPR | FNR")
    print("-" * 120)
    
    for _, row in agg_df.iterrows():
        tn, fp, fn, tp = row["tn_mean"], row["fp_mean"], row["fn_mean"], row["tp_mean"]
        total_neg = tn + fp
        total_pos = fn + tp
        fpr = fp / total_neg * 100 if total_neg > 0 else 0
        fnr = fn / total_pos * 100 if total_pos > 0 else 0
        
        print(f"{row['method_label']:<35} | {tn:>12.0f} | {fp:>12.0f} | {fn:>10.0f} | {tp:>10.0f} | "
              f"{row['recall']*100:>5.1f}% | {row['specificity']*100:>4.1f}% | {fpr:>4.1f}% | {fnr:>4.1f}%")
    
    # Create rates visualization
    print("\nCreating confusion matrix rates visualization...")
    
    n_methods = len(agg_df)
    fig, axes = plt.subplots(1, 2, figsize=(12, max(10, n_methods * 0.5)))
    
    methods = agg_df["method_label"].values
    recall_vals = agg_df["recall"].values * 100
    fnr_vals = (1 - agg_df["recall"].values) * 100
    specificity_vals = agg_df["specificity"].values * 100
    fpr_vals = (1 - agg_df["specificity"].values) * 100
    
    ax1 = axes[0]
    y_pos = np.arange(n_methods)
    
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
            ax1.text(r/2, i, f"{r:.0f}%", ha='center', va='center', fontsize=8, color='white', fontweight='bold')
        if f > 5:
            ax1.text(-f/2, i, f"{f:.0f}%", ha='center', va='center', fontsize=8, color='white', fontweight='bold')
    
    ax2 = axes[1]
    ax2.barh(y_pos, specificity_vals, color='#3498db', label='TN Rate (Specificity)', alpha=0.8)
    ax2.barh(y_pos, -fpr_vals, color='#e67e22', label='FP Rate (False Alarm)', alpha=0.8)
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])
    ax2.set_xlabel("Rate (%)")
    ax2.set_title("Alert Class (Negative)\nSpecificity vs False Alarm Rate", fontsize=12, fontweight='bold')
    ax2.axvline(x=0, color='black', linewidth=0.5)
    ax2.set_xlim(-100, 100)
    ax2.legend(loc='lower right', fontsize=9)
    
    for i, (s, f) in enumerate(zip(specificity_vals, fpr_vals)):
        if s > 5:
            ax2.text(s/2, i, f"{s:.0f}%", ha='center', va='center', fontsize=8, color='white', fontweight='bold')
        if f > 5:
            ax2.text(-f/2, i, f"{f:.0f}%", ha='center', va='center', fontsize=8, color='white', fontweight='bold')
    
    plt.suptitle("Confusion Matrix Rates Summary\nMulti-seed Comparison", 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR_MULTISEED / "confusion_matrix_rates.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    print(f"\n✓ All outputs saved to {OUTPUT_DIR_MULTISEED}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Unified confusion matrix analysis and visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python confusion_matrix_analysis.py plot
    python confusion_matrix_analysis.py table --eval-dir results/evaluation/RF/14357179
    python confusion_matrix_analysis.py aggregate
    python confusion_matrix_analysis.py all
        """
    )
    parser.add_argument(
        "mode",
        choices=["plot", "table", "aggregate", "all"],
        help="Analysis mode: plot (PNG), table (CSV), aggregate (multi-seed), or all"
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=None,
        help=f"Evaluation directory (default: {DEFAULT_EVAL_DIR})"
    )
    args = parser.parse_args()
    
    # Determine eval directory
    if args.eval_dir:
        eval_dir = args.eval_dir
    elif args.mode == "aggregate":
        eval_dir = PROJECT_ROOT / "results/evaluation"
    else:
        eval_dir = DEFAULT_EVAL_DIR
    
    print("=" * 80)
    print(f"CONFUSION MATRIX ANALYSIS (mode={args.mode})")
    print("=" * 80)
    print(f"Evaluation directory: {eval_dir}")
    
    if args.mode == "plot":
        run_plot_mode(eval_dir)
    elif args.mode == "table":
        run_table_mode(eval_dir)
    elif args.mode == "aggregate":
        run_aggregate_mode(eval_dir)
    elif args.mode == "all":
        run_plot_mode(eval_dir)
        run_table_mode(eval_dir)
        run_aggregate_mode(PROJECT_ROOT / "results/evaluation")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
