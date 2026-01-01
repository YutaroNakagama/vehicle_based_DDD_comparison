#!/usr/bin/env python
"""
Plot Specificity Boxplot - Multi-seed comparison
=================================================

Creates a boxplot of Specificity (True Negative Rate) across multiple seeds for each method.
Specificity = TN / (TN + FP) = 1 - False Alarm Rate
"""

import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def extract_results_from_evaluation(eval_dir: Path) -> pd.DataFrame:
    """Extract Specificity and other metrics from evaluation JSON files."""
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
                        
                        # Filter: only imbal_v2_ tags (not imbalv3_ or imbalv2_ variants)
                        if not tag.startswith("imbal_v2_"):
                            continue
                        
                        tag_part = tag.replace("imbal_v2_", "")
                        
                        # Find seed - REQUIRE seed in tag for consistency
                        seed_match = re.search(r"seed(\d+)", tag_part)
                        if seed_match:
                            seed = seed_match.group(1)
                            # Remove seed part
                            method_part = re.sub(r"_seed\d+", "", tag_part)
                        else:
                            # Include EasyEnsemble and BalancedRF without seed 
                            # (older experiments before multi-seed)
                            if tag_part in ["easy_ensemble", "balanced_rf"]:
                                seed = "42"
                                method_part = tag_part
                            else:
                                continue  # Skip other methods without seed
                        
                        # Skip "fixed" variants (old experiments)
                        if "_fixed" in method_part or "_fix" in method_part:
                            continue
                        
                        # Parse ratio if exists
                        ratio_match = re.search(r"ratio(\d+)_(\d+)", method_part)
                        
                        # Skip methods without explicit ratio (used default 0.33, causes confusion)
                        # Exception: baseline, easy_ensemble, balanced_rf don't need ratio
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
                            method_label = "Baseline (class_weight)"
                        elif method == "easy_ensemble":
                            method_label = "EasyEnsemble"
                        elif method == "balanced_rf":
                            method_label = "BalancedRF"
                        elif ratio is not None:
                            method_label = f"{method} (ratio={ratio})"
                        else:
                            method_label = method
                        
                        records.append({
                            "method": method,
                            "ratio": ratio,
                            "seed": seed,
                            "method_label": method_label,
                            "model_type": model_type,
                            "f2_thr": data.get("f2_thr", 0),
                            "recall_thr": data.get("recall_thr", 0),
                            "precision_thr": data.get("prec_thr", 0),
                            "specificity_thr": data.get("specificity_thr", 0),
                            "auprc": data.get("auc_pr", 0),
                            "auroc": data.get("auroc", data.get("auc_roc", 0)),
                            "tag": tag,
                        })
                    except Exception as e:
                        continue
    
    return pd.DataFrame(records)


def plot_specificity_boxplot(
    df: pd.DataFrame,
    figsize: tuple = (14, 12),
    output_path: Path = None,
) -> plt.Figure:
    """
    Create boxplot of Specificity scores across seeds for each method.
    """
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
    
    # Create sort key
    def get_sort_key(row):
        method = row["method"]
        ratio = row["ratio"] if row["ratio"] is not None else 0
        try:
            method_idx = method_order.index(method)
        except ValueError:
            method_idx = len(method_order)
        return (method_idx, -float(ratio) if ratio else 0)
    
    df["sort_key"] = df.apply(get_sort_key, axis=1)
    
    # Get unique method labels in order
    df_sorted = df.sort_values("sort_key")
    method_labels = df_sorted["method_label"].unique()
    
    # Prepare data for boxplot (convert to percentage)
    data_for_plot = [df[df["method_label"] == m]["specificity_thr"].values * 100 for m in method_labels]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create horizontal boxplot
    bp = ax.boxplot(data_for_plot, vert=False, patch_artist=True,
                    tick_labels=method_labels,
                    widths=0.6)
    
    # Color boxes: Baseline = red, others = blue
    for i, (patch, label) in enumerate(zip(bp['boxes'], method_labels)):
        if "Baseline" in label:
            patch.set_facecolor('#f1948a')  # Light red
            patch.set_edgecolor('#c0392b')
        else:
            patch.set_facecolor('#85c1e9')  # Light blue
            patch.set_edgecolor('#2874a6')
    
    # Style whiskers and caps
    for whisker in bp['whiskers']:
        whisker.set(color='#7f8c8d', linewidth=1.5, linestyle='--')
    for cap in bp['caps']:
        cap.set(color='#7f8c8d', linewidth=1.5)
    for median in bp['medians']:
        median.set(color='#2c3e50', linewidth=2)
    for flier in bp['fliers']:
        flier.set(marker='o', color='#e74c3c', alpha=0.5)
    
    # Add median value labels
    for i, (data, label) in enumerate(zip(data_for_plot, method_labels)):
        if len(data) > 0:
            med = np.median(data)
            ax.text(med + 0.5, i + 1, f"{med:.1f}%", 
                    va="center", fontsize=9, fontweight="bold")
    
    ax.set_xlabel("Specificity (%)", fontsize=14)
    ax.set_title("Specificity (True Negative Rate)\nMulti-seed Comparison", 
                 fontsize=16, fontweight="bold", pad=15)
    ax.tick_params(axis='y', labelsize=11)
    ax.tick_params(axis='x', labelsize=11)
    
    # Set x-axis range (will adjust based on data)
    ax.set_xlim(0, 100)
    
    # Add grid
    ax.xaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Add legend for seed count
    n_seeds = df["seed"].nunique()
    ax.text(0.98, 0.02, f"Seeds: {n_seeds} | n per method varies",
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=10, style='italic', color='gray')
    
    plt.tight_layout()
    
    if output_path:
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
    
    # Summary statistics
    print("\nSpecificity Summary (mean ± std):")
    print("-" * 60)
    summary = df.groupby("method_label").agg({
        "specificity_thr": ["mean", "std", "count"]
    }).round(4)
    summary.columns = ["mean", "std", "n"]
    summary = summary.sort_values("mean", ascending=False)
    for idx, row in summary.iterrows():
        print(f"{idx:40s} | Spec: {row['mean']*100:.1f}% ± {row['std']*100:.1f}% (n={int(row['n'])})")
    
    print("\nCreating Specificity boxplot...")
    output_path = output_dir / "specificity_boxplot.png"
    plot_specificity_boxplot(df, output_path=output_path)
    
    print(f"\nDone! Output saved to: {output_path}")


if __name__ == "__main__":
    main()
