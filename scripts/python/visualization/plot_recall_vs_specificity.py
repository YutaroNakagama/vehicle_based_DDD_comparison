#!/usr/bin/env python
"""
Plot Recall vs Specificity Scatter - Multi-seed comparison
============================================================

Creates a scatter plot showing the trade-off between Recall and Specificity.
Helps identify Pareto-optimal methods.
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def extract_results_from_evaluation(eval_dir: Path) -> pd.DataFrame:
    """Extract metrics from evaluation JSON files."""
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
                            if tag_part in ["easy_ensemble", "balanced_rf"]:
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
                        
                        # Create method category for coloring
                        if method == "baseline":
                            category = "Baseline"
                        elif method in ["easy_ensemble", "balanced_rf"]:
                            category = "Ensemble"
                        elif method.startswith("smote"):
                            category = "SMOTE-based"
                        elif method.startswith("undersample"):
                            category = "Undersample"
                        else:
                            category = "Other"
                        
                        records.append({
                            "method": method,
                            "ratio": ratio,
                            "seed": seed,
                            "method_label": method_label,
                            "category": category,
                            "model_type": model_type,
                            "recall": data.get("recall_thr", 0) * 100,
                            "specificity": data.get("specificity_thr", 0) * 100,
                            "f2": data.get("f2_thr", 0),
                            "precision": data.get("prec_thr", 0) * 100,
                            "tag": tag,
                        })
                    except Exception as e:
                        continue
    
    return pd.DataFrame(records)


def is_pareto_optimal(costs):
    """
    Find Pareto-optimal points (maximize both dimensions).
    costs: array of shape (n_points, 2)
    Returns: boolean mask of Pareto-optimal points
    """
    is_optimal = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        # Check if any other point dominates this one
        is_optimal[i] = not np.any(np.all(costs >= c, axis=1) & np.any(costs > c, axis=1))
    return is_optimal


def plot_recall_vs_specificity(
    df: pd.DataFrame,
    figsize: tuple = (14, 10),
    output_path: Path = None,
) -> plt.Figure:
    """
    Create scatter plot of Recall vs Specificity.
    """
    # Aggregate by method (mean across seeds)
    agg_df = df.groupby(["method_label", "category"]).agg({
        "recall": ["mean", "std"],
        "specificity": ["mean", "std"],
        "f2": "mean",
        "seed": "count"
    }).reset_index()
    agg_df.columns = ["method_label", "category", "recall_mean", "recall_std", 
                      "spec_mean", "spec_std", "f2_mean", "n_seeds"]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color map for categories
    colors = {
        "Baseline": "#e74c3c",      # Red
        "Ensemble": "#9b59b6",      # Purple
        "SMOTE-based": "#3498db",   # Blue
        "Undersample": "#27ae60",   # Green
        "Other": "#95a5a6",         # Gray
    }
    
    markers = {
        "Baseline": "s",      # Square
        "Ensemble": "D",      # Diamond
        "SMOTE-based": "o",   # Circle
        "Undersample": "^",   # Triangle
        "Other": "x",         # X
    }
    
    # Plot each category
    for category in colors.keys():
        cat_df = agg_df[agg_df["category"] == category]
        if len(cat_df) == 0:
            continue
        
        ax.scatter(
            cat_df["spec_mean"], cat_df["recall_mean"],
            c=colors[category],
            marker=markers[category],
            s=150,
            label=category,
            alpha=0.8,
            edgecolors='white',
            linewidths=1
        )
        
        # Add error bars
        ax.errorbar(
            cat_df["spec_mean"], cat_df["recall_mean"],
            xerr=cat_df["spec_std"].fillna(0),
            yerr=cat_df["recall_std"].fillna(0),
            fmt='none',
            c=colors[category],
            alpha=0.3,
            capsize=3
        )
    
    # Find and highlight Pareto-optimal points
    costs = agg_df[["spec_mean", "recall_mean"]].values
    pareto_mask = is_pareto_optimal(costs)
    pareto_df = agg_df[pareto_mask].sort_values("spec_mean")
    
    # Draw Pareto frontier line
    if len(pareto_df) > 1:
        ax.plot(
            pareto_df["spec_mean"], pareto_df["recall_mean"],
            'k--', linewidth=2, alpha=0.5,
            label="Pareto Frontier"
        )
    
    # Annotate points
    for _, row in agg_df.iterrows():
        # Offset annotation to avoid overlap
        x_offset = 1.5
        y_offset = 1.5
        
        # Adjust for crowded areas
        if row["spec_mean"] < 10:
            x_offset = 3
        if row["recall_mean"] > 95:
            y_offset = -3
        
        ax.annotate(
            row["method_label"],
            (row["spec_mean"], row["recall_mean"]),
            xytext=(x_offset, y_offset),
            textcoords='offset points',
            fontsize=8,
            alpha=0.8
        )
    
    ax.set_xlabel("Specificity (%) - True Negative Rate", fontsize=14)
    ax.set_ylabel("Recall (%) - Drowsy Detection Rate", fontsize=14)
    ax.set_title("Recall vs Specificity Trade-off\nMulti-seed Comparison", 
                 fontsize=16, fontweight="bold", pad=15)
    
    ax.set_xlim(-5, 105)
    ax.set_ylim(40, 105)
    
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='lower left', fontsize=10)
    
    # Add interpretation note
    note = "↑ High Recall = Few missed drowsy events\n→ High Specificity = Few false alarms"
    ax.text(0.98, 0.02, note,
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9, style='italic', color='gray',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
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
    
    # Summary
    print("\nRecall vs Specificity Summary (sorted by F2):")
    print("-" * 80)
    agg = df.groupby("method_label").agg({
        "recall": "mean",
        "specificity": "mean",
        "f2": "mean"
    }).round(2)
    agg = agg.sort_values("f2", ascending=False)
    for idx, row in agg.iterrows():
        status = "✓" if row["recall"] >= 80 and row["specificity"] >= 50 else " "
        print(f"{status} {idx:35s} | Recall: {row['recall']:5.1f}% | Spec: {row['specificity']:5.1f}% | F2: {row['f2']:.3f}")
    
    # Find Pareto-optimal
    print("\n" + "=" * 80)
    print("Pareto-optimal methods (not dominated by any other):")
    print("=" * 80)
    agg_df = df.groupby("method_label").agg({
        "recall": "mean",
        "specificity": "mean",
    }).reset_index()
    costs = agg_df[["specificity", "recall"]].values
    pareto_mask = is_pareto_optimal(costs)
    pareto_methods = agg_df[pareto_mask].sort_values("specificity", ascending=False)
    for _, row in pareto_methods.iterrows():
        print(f"  • {row['method_label']:35s} | Recall: {row['recall']:5.1f}% | Spec: {row['specificity']:5.1f}%")
    
    print("\nCreating Recall vs Specificity scatter plot...")
    output_path = output_dir / "recall_vs_specificity_scatter.png"
    plot_recall_vs_specificity(df, output_path=output_path)
    
    print(f"\nDone! Output saved to: {output_path}")


if __name__ == "__main__":
    main()
