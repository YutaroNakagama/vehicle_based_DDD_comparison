#!/usr/bin/env python3
"""
Visualize comparison of imbalance handling methods on KNN-ranked groups.

Compares baseline (no imbalance handling) vs various imbalance methods
across different domain levels (out_domain, mid_domain, in_domain).
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

def load_results():
    """Load all available KNN imbalance results."""
    results_dir = PROJECT_ROOT / "results" / "domain_analysis" / "knn_imbalance"
    
    all_results = []
    for csv_file in results_dir.glob("summary_*.csv"):
        df = pd.read_csv(csv_file)
        # Parse filename: summary_{jobid}_{mode}_{distance}.csv
        # e.g., summary_14471570_source_only_mmd.csv
        filename = csv_file.stem  # summary_14471570_source_only_mmd
        parts = filename.replace("summary_", "").split("_")
        # parts = ['14471570', 'source', 'only', 'mmd']
        if len(parts) >= 4:
            df["job_id"] = parts[0]
            df["mode"] = f"{parts[1]}_{parts[2]}"  # source_only or target_only
            df["distance"] = parts[3]  # mmd, wasserstein, dtw
        all_results.append(df)
    
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()


def plot_f2_comparison(df, output_dir):
    """Plot F2 score comparison: baseline vs imbalance methods."""
    
    # Filter successful results only
    df = df[df["status"] == "success"].copy()
    
    if df.empty:
        print("No successful results to plot")
        return
    
    # Get unique combinations
    distances = df["distance"].unique()
    modes = df["mode"].unique()
    levels = ["out_domain", "mid_domain", "in_domain"]
    methods = ["baseline", "undersample_rus", "undersample_tomek", "smote_rus", "smote_tomek"]
    
    # Create figure for each distance
    for distance in distances:
        fig, axes = plt.subplots(1, len(modes), figsize=(6*len(modes), 5), squeeze=False)
        fig.suptitle(f"F2 Score Comparison - {distance.upper()} Distance + KNN Ranking", fontsize=14, fontweight='bold')
        
        for mode_idx, mode in enumerate(modes):
            ax = axes[0, mode_idx]
            subset = df[(df["distance"] == distance) & (df["mode"] == mode)]
            
            if subset.empty:
                ax.set_title(f"{mode}\n(No data)")
                continue
            
            x = np.arange(len(levels))
            width = 0.15
            
            colors = {
                "baseline": "#888888",
                "undersample_rus": "#2ecc71",
                "undersample_tomek": "#27ae60",
                "smote_rus": "#3498db",
                "smote_tomek": "#2980b9"
            }
            
            for i, method in enumerate(methods):
                method_data = subset[subset["method"] == method]
                values = []
                for level in levels:
                    level_data = method_data[method_data["level"] == level]
                    if not level_data.empty:
                        values.append(level_data["f2_thr"].values[0])
                    else:
                        values.append(0)
                
                offset = (i - len(methods)/2 + 0.5) * width
                bars = ax.bar(x + offset, values, width, label=method, color=colors.get(method, "#888"))
            
            ax.set_xlabel("Domain Level")
            ax.set_ylabel("F2 Score")
            ax.set_title(f"{mode}")
            ax.set_xticks(x)
            ax.set_xticklabels(["Out\n(Far)", "Mid", "In\n(Close)"])
            ax.set_ylim(0, 0.25)
            ax.legend(loc="upper left", fontsize=8)
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = output_dir / f"f2_comparison_{distance}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")


def plot_recall_comparison(df, output_dir):
    """Plot Recall comparison: baseline vs imbalance methods."""
    
    df = df[df["status"] == "success"].copy()
    
    if df.empty:
        return
    
    distances = df["distance"].unique()
    modes = df["mode"].unique()
    levels = ["out_domain", "mid_domain", "in_domain"]
    methods = ["baseline", "undersample_rus", "undersample_tomek", "smote_rus", "smote_tomek"]
    
    for distance in distances:
        fig, axes = plt.subplots(1, len(modes), figsize=(6*len(modes), 5), squeeze=False)
        fig.suptitle(f"Recall Comparison - {distance.upper()} Distance + KNN Ranking", fontsize=14, fontweight='bold')
        
        for mode_idx, mode in enumerate(modes):
            ax = axes[0, mode_idx]
            subset = df[(df["distance"] == distance) & (df["mode"] == mode)]
            
            if subset.empty:
                ax.set_title(f"{mode}\n(No data)")
                continue
            
            x = np.arange(len(levels))
            width = 0.15
            
            colors = {
                "baseline": "#888888",
                "undersample_rus": "#2ecc71",
                "undersample_tomek": "#27ae60",
                "smote_rus": "#3498db",
                "smote_tomek": "#2980b9"
            }
            
            for i, method in enumerate(methods):
                method_data = subset[subset["method"] == method]
                values = []
                for level in levels:
                    level_data = method_data[method_data["level"] == level]
                    if not level_data.empty:
                        values.append(level_data["recall_thr"].values[0])
                    else:
                        values.append(0)
                
                offset = (i - len(methods)/2 + 0.5) * width
                bars = ax.bar(x + offset, values, width, label=method, color=colors.get(method, "#888"))
            
            ax.set_xlabel("Domain Level")
            ax.set_ylabel("Recall")
            ax.set_title(f"{mode}")
            ax.set_xticks(x)
            ax.set_xticklabels(["Out\n(Far)", "Mid", "In\n(Close)"])
            ax.set_ylim(0, 1.1)
            ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.3, label='Perfect Recall')
            ax.legend(loc="upper left", fontsize=8)
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = output_dir / f"recall_comparison_{distance}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")


def plot_baseline_vs_best(df, output_dir):
    """Plot baseline vs best imbalance method improvement."""
    
    df = df[df["status"] == "success"].copy()
    
    if df.empty:
        return
    
    # Calculate improvement over baseline
    summary_data = []
    
    for distance in df["distance"].unique():
        for mode in df["mode"].unique():
            for level in df["level"].unique():
                subset = df[(df["distance"] == distance) & (df["mode"] == mode) & (df["level"] == level)]
                
                baseline = subset[subset["method"] == "baseline"]
                non_baseline = subset[subset["method"] != "baseline"]
                
                if baseline.empty or non_baseline.empty:
                    continue
                
                baseline_f2 = baseline["f2_thr"].values[0]
                baseline_recall = baseline["recall_thr"].values[0]
                
                best_f2_row = non_baseline.loc[non_baseline["f2_thr"].idxmax()]
                best_recall_row = non_baseline.loc[non_baseline["recall_thr"].idxmax()]
                
                summary_data.append({
                    "distance": distance,
                    "mode": mode,
                    "level": level,
                    "baseline_f2": baseline_f2,
                    "best_f2": best_f2_row["f2_thr"],
                    "best_f2_method": best_f2_row["method"],
                    "f2_improvement": best_f2_row["f2_thr"] - baseline_f2,
                    "baseline_recall": baseline_recall,
                    "best_recall": best_recall_row["recall_thr"],
                    "best_recall_method": best_recall_row["method"],
                    "recall_improvement": best_recall_row["recall_thr"] - baseline_recall,
                })
    
    if not summary_data:
        return
    
    summary_df = pd.DataFrame(summary_data)
    
    # Plot improvement
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # F2 improvement
    ax = axes[0]
    modes = summary_df["mode"].unique()
    levels = ["out_domain", "mid_domain", "in_domain"]
    x = np.arange(len(levels))
    width = 0.35
    
    for i, mode in enumerate(modes):
        mode_data = summary_df[summary_df["mode"] == mode]
        baseline_vals = [mode_data[mode_data["level"] == l]["baseline_f2"].values[0] if len(mode_data[mode_data["level"] == l]) > 0 else 0 for l in levels]
        best_vals = [mode_data[mode_data["level"] == l]["best_f2"].values[0] if len(mode_data[mode_data["level"] == l]) > 0 else 0 for l in levels]
        
        offset = (i - len(modes)/2 + 0.5) * width
        ax.bar(x + offset - width/4, baseline_vals, width/2, label=f"{mode} baseline", alpha=0.5, color=f"C{i}")
        ax.bar(x + offset + width/4, best_vals, width/2, label=f"{mode} best", color=f"C{i}")
    
    ax.set_xlabel("Domain Level")
    ax.set_ylabel("F2 Score")
    ax.set_title("F2: Baseline vs Best Imbalance Method")
    ax.set_xticks(x)
    ax.set_xticklabels(["Out (Far)", "Mid", "In (Close)"])
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    
    # Recall improvement
    ax = axes[1]
    for i, mode in enumerate(modes):
        mode_data = summary_df[summary_df["mode"] == mode]
        baseline_vals = [mode_data[mode_data["level"] == l]["baseline_recall"].values[0] if len(mode_data[mode_data["level"] == l]) > 0 else 0 for l in levels]
        best_vals = [mode_data[mode_data["level"] == l]["best_recall"].values[0] if len(mode_data[mode_data["level"] == l]) > 0 else 0 for l in levels]
        
        offset = (i - len(modes)/2 + 0.5) * width
        ax.bar(x + offset - width/4, baseline_vals, width/2, label=f"{mode} baseline", alpha=0.5, color=f"C{i}")
        ax.bar(x + offset + width/4, best_vals, width/2, label=f"{mode} best", color=f"C{i}")
    
    ax.set_xlabel("Domain Level")
    ax.set_ylabel("Recall")
    ax.set_title("Recall: Baseline vs Best Imbalance Method")
    ax.set_xticks(x)
    ax.set_xticklabels(["Out (Far)", "Mid", "In (Close)"])
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.3)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / "baseline_vs_best_improvement.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    
    # Print summary table
    print("\n=== Best Imbalance Method Summary ===")
    print(summary_df[["distance", "mode", "level", "baseline_f2", "best_f2", "best_f2_method", "f2_improvement"]].to_string(index=False))


def main():
    output_dir = PROJECT_ROOT / "results" / "domain_analysis" / "knn_imbalance" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading results...")
    df = load_results()
    
    if df.empty:
        print("No results found!")
        return
    
    print(f"Found {len(df)} results")
    print(f"Distances: {df['distance'].unique()}")
    print(f"Modes: {df['mode'].unique()}")
    
    print("\nGenerating visualizations...")
    plot_f2_comparison(df, output_dir)
    plot_recall_comparison(df, output_dir)
    plot_baseline_vs_best(df, output_dir)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
