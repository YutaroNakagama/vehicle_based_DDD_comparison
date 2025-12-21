#!/usr/bin/env python
"""
Summarize Confusion Matrices - Multi-seed comparison
=====================================================

Creates a summary table and heatmap of confusion matrices for each method.
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def extract_results_from_evaluation(eval_dir: Path) -> pd.DataFrame:
    """Extract confusion matrix and metrics from evaluation JSON files."""
    records = []
    
    for model_type in ["RF", "BalancedRF", "EasyEnsemble"]:
        type_dir = eval_dir / model_type
        if not type_dir.exists():
            continue
        
        for job_dir in type_dir.iterdir():
            if not job_dir.is_dir() or job_dir.name.endswith('.txt') or job_dir.name.endswith('.png'):
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
                            # Include baseline, EasyEnsemble and BalancedRF without seed 
                            # (older experiments before multi-seed)
                            base_method = tag_part.split("_ratio")[0]  # Remove ratio part if exists
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
                        
                        # Create method label
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
                    except Exception as e:
                        continue
    
    return pd.DataFrame(records)


def create_confusion_matrix_summary(df: pd.DataFrame, output_dir: Path):
    """Create summary table and visualization of confusion matrices."""
    
    # Define method order
    method_order = [
        "baseline",
        "balanced_rf",
        "easy_ensemble",
        "smote",
        "smote_tomek",
        "smote_enn",
        "smote_rus",
        "smote_balanced_rf",
        "undersample_rus",
        "undersample_tomek",
        "undersample_enn",
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
    
    # Aggregate by method (sum confusion matrix across seeds)
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
    
    # Flatten column names
    agg_df.columns = [
        "method_label", "method", "ratio",
        "tn_sum", "tn_mean", "tn_std",
        "fp_sum", "fp_mean", "fp_std",
        "fn_sum", "fn_mean", "fn_std",
        "tp_sum", "tp_mean", "tp_std",
        "total", "recall", "specificity", "precision", "f2", "n_seeds", "sort_key"
    ]
    
    agg_df = agg_df.sort_values("sort_key")
    
    # Create summary CSV
    summary_data = []
    for _, row in agg_df.iterrows():
        tn, fp, fn, tp = row["tn_mean"], row["fp_mean"], row["fn_mean"], row["tp_mean"]
        tn_std, fp_std, fn_std, tp_std = row["tn_std"], row["fp_std"], row["fn_std"], row["tp_std"]
        
        # Calculate rates
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
    
    # Save CSV
    csv_path = output_dir / "confusion_matrix_summary.csv"
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
    
    return agg_df, summary_df


def plot_confusion_matrix_heatmap(df: pd.DataFrame, output_dir: Path):
    """Create heatmap visualization showing normalized confusion matrices."""
    
    # Define method order
    method_order = [
        "baseline",
        "balanced_rf",
        "easy_ensemble",
        "smote",
        "smote_tomek",
        "smote_enn",
        "smote_rus",
        "smote_balanced_rf",
        "undersample_rus",
        "undersample_tomek",
        "undersample_enn",
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
    agg_df = df.groupby(["method_label"]).agg({
        "tn": "mean",
        "fp": "mean",
        "fn": "mean",
        "tp": "mean",
        "recall": "mean",
        "specificity": "mean",
        "sort_key": "first"
    }).reset_index()
    
    agg_df = agg_df.sort_values("sort_key")
    
    # Create figure with subplots - 2 columns showing rates
    n_methods = len(agg_df)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, max(10, n_methods * 0.5)))
    
    # Prepare data for heatmap
    methods = agg_df["method_label"].values
    
    # Left: Raw counts (normalized by row)
    recall_vals = agg_df["recall"].values * 100
    fnr_vals = (1 - agg_df["recall"].values) * 100
    specificity_vals = agg_df["specificity"].values * 100
    fpr_vals = (1 - agg_df["specificity"].values) * 100
    
    # Create horizontal bar chart for Recall/FNR
    ax1 = axes[0]
    y_pos = np.arange(n_methods)
    
    bars1 = ax1.barh(y_pos, recall_vals, color='#27ae60', label='TP Rate (Recall)', alpha=0.8)
    bars2 = ax1.barh(y_pos, -fnr_vals, color='#e74c3c', label='FN Rate (Missed)', alpha=0.8)
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(methods)
    ax1.set_xlabel("Rate (%)")
    ax1.set_title("Drowsy Class (Positive)\nRecall vs Miss Rate", fontsize=12, fontweight='bold')
    ax1.axvline(x=0, color='black', linewidth=0.5)
    ax1.set_xlim(-100, 100)
    ax1.legend(loc='lower right', fontsize=9)
    
    # Add value labels
    for i, (r, f) in enumerate(zip(recall_vals, fnr_vals)):
        if r > 5:
            ax1.text(r/2, i, f"{r:.0f}%", ha='center', va='center', fontsize=8, color='white', fontweight='bold')
        if f > 5:
            ax1.text(-f/2, i, f"{f:.0f}%", ha='center', va='center', fontsize=8, color='white', fontweight='bold')
    
    # Create horizontal bar chart for Specificity/FPR
    ax2 = axes[1]
    
    bars3 = ax2.barh(y_pos, specificity_vals, color='#3498db', label='TN Rate (Specificity)', alpha=0.8)
    bars4 = ax2.barh(y_pos, -fpr_vals, color='#e67e22', label='FP Rate (False Alarm)', alpha=0.8)
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])  # Hide labels on right plot
    ax2.set_xlabel("Rate (%)")
    ax2.set_title("Alert Class (Negative)\nSpecificity vs False Alarm Rate", fontsize=12, fontweight='bold')
    ax2.axvline(x=0, color='black', linewidth=0.5)
    ax2.set_xlim(-100, 100)
    ax2.legend(loc='lower right', fontsize=9)
    
    # Add value labels
    for i, (s, f) in enumerate(zip(specificity_vals, fpr_vals)):
        if s > 5:
            ax2.text(s/2, i, f"{s:.0f}%", ha='center', va='center', fontsize=8, color='white', fontweight='bold')
        if f > 5:
            ax2.text(-f/2, i, f"{f:.0f}%", ha='center', va='center', fontsize=8, color='white', fontweight='bold')
    
    plt.suptitle("Confusion Matrix Rates Summary\nMulti-seed Comparison", 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    output_path = output_dir / "confusion_matrix_rates.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    return fig


def plot_individual_confusion_matrices(df: pd.DataFrame, output_dir: Path):
    """Create grid of individual confusion matrices for key methods."""
    
    # Select key methods to display
    key_methods = [
        "Baseline",
        "BalancedRF", 
        "EasyEnsemble",
        "smote (r=1.0)",
        "smote_enn (r=1.0)",
        "undersample_rus (r=1.0)",
    ]
    
    # Aggregate by method
    agg_df = df.groupby(["method_label"]).agg({
        "tn": "mean",
        "fp": "mean",
        "fn": "mean",
        "tp": "mean",
    }).reset_index()
    
    # Filter to key methods
    agg_df = agg_df[agg_df["method_label"].isin(key_methods)]
    
    if len(agg_df) == 0:
        print("No key methods found, skipping individual confusion matrices")
        return None
    
    n_methods = len(agg_df)
    n_cols = min(3, n_methods)
    n_rows = (n_methods + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_methods == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (_, row) in enumerate(agg_df.iterrows()):
        ax = axes[idx]
        
        # Create confusion matrix
        cm = np.array([[row["tn"], row["fp"]], 
                       [row["fn"], row["tp"]]])
        
        # Normalize by row
        cm_normalized = cm / cm.sum(axis=1, keepdims=True) * 100
        
        # Plot heatmap
        im = ax.imshow(cm_normalized, cmap='Blues', vmin=0, vmax=100)
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                value = cm[i, j]
                pct = cm_normalized[i, j]
                color = 'white' if pct > 50 else 'black'
                ax.text(j, i, f"{value:.0f}\n({pct:.1f}%)", 
                        ha='center', va='center', fontsize=10, color=color)
        
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Alert', 'Drowsy'])
        ax.set_yticklabels(['Alert', 'Drowsy'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(row["method_label"], fontsize=11, fontweight='bold')
    
    # Hide unused axes
    for idx in range(n_methods, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle("Confusion Matrices (Mean across seeds)\n[Row-normalized percentages]", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / "confusion_matrices_grid.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    return fig


def main():
    """Main function."""
    project_root = Path(__file__).parent.parent.parent
    eval_dir = project_root / "results" / "evaluation"
    output_dir = project_root / "results" / "imbalance_analysis" / "multiseed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Extracting evaluation results...")
    df = extract_results_from_evaluation(eval_dir)
    print(f"Found {len(df)} evaluation records")
    print(f"Seeds: {sorted(df['seed'].unique())}")
    print(f"Methods: {df['method_label'].nunique()} unique")
    
    # Create summary
    print("\nCreating confusion matrix summary...")
    agg_df, summary_df = create_confusion_matrix_summary(df, output_dir)
    
    # Create visualizations
    print("\nCreating confusion matrix rates visualization...")
    plot_confusion_matrix_heatmap(df, output_dir)
    
    print("\nCreating individual confusion matrices grid...")
    plot_individual_confusion_matrices(df, output_dir)
    
    print(f"\nDone! Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
