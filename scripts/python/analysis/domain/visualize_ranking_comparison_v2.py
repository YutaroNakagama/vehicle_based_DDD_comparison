#!/usr/bin/env python3
"""Visualize ranking method comparison results with distance metric analysis.

Enhanced version that compares:
1. Ranking methods (6 types): mean_distance, median_distance, knn, lof, isolation_forest, centroid_umap
2. Distance metrics (3 types): DTW, MMD, Wasserstein
3. Domain levels (2 types): out_domain, mid_domain
4. Modes (2 types): pooled, target_only

Creates comprehensive visualizations for the ranking method comparison experiment.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))


# Define constants for expected experiment combinations
RANKING_METHODS = ["mean_distance", "median_distance", "knn", "lof", "isolation_forest", "centroid_umap"]
DISTANCE_METRICS = ["dtw", "mmd", "wasserstein"]
DOMAIN_LEVELS = ["out_domain", "mid_domain"]
MODES = ["pooled", "target_only"]

# Display name mappings
RANKING_DISPLAY = {
    "mean_distance": "Mean Distance",
    "median_distance": "Median Distance",
    "knn": "KNN",
    "lof": "LOF",
    "isolation_forest": "Isolation Forest",
    "centroid_umap": "Centroid UMAP"
}

DISTANCE_DISPLAY = {
    "dtw": "DTW",
    "mmd": "MMD",
    "wasserstein": "Wasserstein"
}

LEVEL_DISPLAY = {
    "out_domain": "Out-Domain",
    "mid_domain": "Mid-Domain"
}


def find_evaluation_files(eval_dir: Path) -> list:
    """Find all evaluation result files for ranking experiments."""
    files = []
    
    # Look in subdirectories (job output folders)
    for job_dir in eval_dir.iterdir():
        if job_dir.is_dir():
            for f in job_dir.glob("eval_results_RF_*rank_*.json"):
                files.append(f)
    
    # Also look directly in the directory
    for f in eval_dir.glob("eval_results_RF_*rank_*.json"):
        files.append(f)
    
    return list(set(files))  # Remove duplicates


def parse_eval_filename(filepath: Path) -> dict:
    """Parse evaluation filename to extract experiment metadata.
    
    Expected filename format:
    eval_results_RF_{mode}_rank_{method}_{distance}_{level}_{job_id}.json
    Example: eval_results_RF_pooled_rank_knn_dtw_out_domain_14620001.json
    """
    name = filepath.stem
    parts = name.split("_")
    
    info = {
        "file": str(filepath),
        "model": "RF",
        "mode": None,
        "ranking_method": None,
        "distance": None,
        "level": None,
    }
    
    # Find mode
    if "pooled" in parts:
        info["mode"] = "pooled"
    elif "source_only" in parts:
        info["mode"] = "source_only"
    elif "target_only" in parts:
        info["mode"] = "target_only"
    
    # Find ranking method (after "rank_")
    try:
        rank_idx = parts.index("rank")
        if rank_idx + 1 < len(parts):
            # Handle multi-word ranking methods (e.g., "mean_distance", "isolation_forest")
            method_parts = []
            for i in range(rank_idx + 1, len(parts)):
                part = parts[i]
                if part in DISTANCE_METRICS:
                    info["distance"] = part
                    break
                method_parts.append(part)
            info["ranking_method"] = "_".join(method_parts)
    except ValueError:
        pass
    
    # Find domain level
    if "out_domain" in name:
        info["level"] = "out_domain"
    elif "mid_domain" in name:
        info["level"] = "mid_domain"
    elif "in_domain" in name:
        info["level"] = "in_domain"
    
    return info


def load_eval_results(filepath: Path) -> dict:
    """Load evaluation results from JSON file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return {}


def collect_all_results(eval_dir: Path, verbose: bool = True) -> pd.DataFrame:
    """Collect all evaluation results into a DataFrame."""
    files = find_evaluation_files(eval_dir)
    
    if verbose:
        print(f"Found {len(files)} evaluation files")
    
    rows = []
    for filepath in files:
        info = parse_eval_filename(filepath)
        data = load_eval_results(filepath)
        
        if not data:
            continue
        
        # Extract test metrics (prefer test over val)
        metrics = data.get("test", data.get("val", {}))
        
        row = {
            **info,
            "f2": metrics.get("f2", metrics.get("f2_score")),
            "recall": metrics.get("recall"),
            "precision": metrics.get("precision"),
            "f1": metrics.get("f1", metrics.get("f1_score")),
            "auc": metrics.get("auc", metrics.get("roc_auc")),
            "auc_pr": metrics.get("auc_pr", metrics.get("pr_auc")),
            "accuracy": metrics.get("accuracy"),
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Filter to ranking experiments only
    df = df[df["ranking_method"].notna() & (df["ranking_method"] != "")]
    
    if verbose:
        print(f"Collected {len(df)} ranking experiment results")
    
    return df


def plot_ranking_by_distance(df: pd.DataFrame, output_dir: Path, mode: str = "target_only"):
    """Plot F2 score comparison: ranking method × distance metric."""
    mode_df = df[df["mode"] == mode]
    
    if len(mode_df) == 0:
        print(f"No data for mode: {mode}")
        return
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Pivot for grouped bar plot
    pivot = mode_df.groupby(["ranking_method", "distance"])["f2"].mean().unstack()
    
    # Reorder index and columns
    pivot = pivot.reindex(index=[m for m in RANKING_METHODS if m in pivot.index])
    pivot = pivot.reindex(columns=[d for d in DISTANCE_METRICS if d in pivot.columns])
    
    x = np.arange(len(pivot.index))
    width = 0.25
    
    colors = {"dtw": "#2E86AB", "mmd": "#A23B72", "wasserstein": "#F18F01"}
    
    for i, dist in enumerate(pivot.columns):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, pivot[dist], width, 
                     label=DISTANCE_DISPLAY.get(dist, dist),
                     color=colors.get(dist, f"C{i}"))
        
        # Add value labels
        for bar, val in zip(bars, pivot[dist]):
            if not np.isnan(val):
                ax.annotate(f'{val:.3f}', 
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           ha='center', va='bottom', fontsize=8, rotation=90)
    
    ax.set_xlabel("Ranking Method", fontsize=12)
    ax.set_ylabel("F2 Score", fontsize=12)
    ax.set_title(f"Ranking Method × Distance Metric Comparison\n(mode: {mode}, N_TRIALS=75)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([RANKING_DISPLAY.get(m, m) for m in pivot.index], rotation=45, ha="right")
    ax.legend(title="Distance Metric", loc="upper right")
    ax.set_ylim(0, min(1.0, pivot.max().max() * 1.15))
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"ranking_x_distance_{mode}.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / f'ranking_x_distance_{mode}.png'}")


def plot_distance_comparison_heatmap(df: pd.DataFrame, output_dir: Path, mode: str = "target_only"):
    """Create heatmap showing F2 scores: ranking method × distance metric."""
    mode_df = df[df["mode"] == mode]
    
    if len(mode_df) == 0:
        return
    
    # Aggregate by ranking method and distance
    pivot = mode_df.groupby(["ranking_method", "distance"])["f2"].mean().unstack()
    
    # Reorder
    pivot = pivot.reindex(index=[m for m in RANKING_METHODS if m in pivot.index])
    pivot = pivot.reindex(columns=[d for d in DISTANCE_METRICS if d in pivot.columns])
    
    # Rename for display
    pivot.index = [RANKING_DISPLAY.get(m, m) for m in pivot.index]
    pivot.columns = [DISTANCE_DISPLAY.get(d, d) for d in pivot.columns]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        ax=ax,
        cbar_kws={"label": "F2 Score"},
        linewidths=0.5,
        vmin=pivot.min().min() * 0.95,
        vmax=pivot.max().max() * 1.02
    )
    
    ax.set_title(f"F2 Score Heatmap: Ranking × Distance\n(mode: {mode})", fontsize=12)
    ax.set_xlabel("Distance Metric", fontsize=11)
    ax.set_ylabel("Ranking Method", fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"ranking_distance_heatmap_{mode}.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / f'ranking_distance_heatmap_{mode}.png'}")


def plot_domain_level_comparison(df: pd.DataFrame, output_dir: Path, mode: str = "target_only"):
    """Compare performance across domain levels."""
    mode_df = df[df["mode"] == mode]
    
    if len(mode_df) == 0:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, level in enumerate(DOMAIN_LEVELS):
        level_df = mode_df[mode_df["level"] == level]
        
        if len(level_df) == 0:
            continue
        
        # Aggregate
        pivot = level_df.groupby(["ranking_method", "distance"])["f2"].mean().unstack()
        pivot = pivot.reindex(index=[m for m in RANKING_METHODS if m in pivot.index])
        pivot = pivot.reindex(columns=[d for d in DISTANCE_METRICS if d in pivot.columns])
        pivot.index = [RANKING_DISPLAY.get(m, m) for m in pivot.index]
        pivot.columns = [DISTANCE_DISPLAY.get(d, d) for d in pivot.columns]
        
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="YlGnBu",
            ax=axes[idx],
            cbar_kws={"label": "F2 Score"},
            linewidths=0.5
        )
        
        axes[idx].set_title(f"{LEVEL_DISPLAY.get(level, level)}", fontsize=12)
        axes[idx].set_xlabel("Distance Metric", fontsize=10)
        axes[idx].set_ylabel("Ranking Method", fontsize=10)
    
    plt.suptitle(f"Domain Level Comparison (mode: {mode})", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / f"domain_level_comparison_{mode}.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / f'domain_level_comparison_{mode}.png'}")


def plot_best_combinations(df: pd.DataFrame, output_dir: Path):
    """Rank and visualize the best method+distance combinations."""
    # Group by method, distance, mode, level
    summary = df.groupby(["ranking_method", "distance", "mode", "level"]).agg({
        "f2": ["mean", "std", "count"],
        "recall": "mean",
        "precision": "mean"
    }).reset_index()
    summary.columns = ["ranking_method", "distance", "mode", "level", 
                       "f2_mean", "f2_std", "n", "recall_mean", "precision_mean"]
    
    # Focus on target_only mode
    target_summary = summary[summary["mode"] == "target_only"].copy()
    
    if len(target_summary) == 0:
        print("No target_only data for best combinations plot")
        return
    
    # Create combination label
    target_summary["combination"] = (
        target_summary["ranking_method"].map(lambda x: RANKING_DISPLAY.get(x, x)) + 
        " + " + 
        target_summary["distance"].map(lambda x: DISTANCE_DISPLAY.get(x, x))
    )
    
    # Sort by F2 mean
    target_summary = target_summary.sort_values("f2_mean", ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot top 15 combinations
    top_n = min(15, len(target_summary))
    top_df = target_summary.head(top_n)
    
    colors = []
    for _, row in top_df.iterrows():
        if row["distance"] == "dtw":
            colors.append("#2E86AB")
        elif row["distance"] == "mmd":
            colors.append("#A23B72")
        else:
            colors.append("#F18F01")
    
    bars = ax.barh(range(top_n), top_df["f2_mean"], color=colors, edgecolor='black')
    
    # Add error bars
    ax.errorbar(top_df["f2_mean"], range(top_n), 
                xerr=top_df["f2_std"], fmt='none', color='black', capsize=3)
    
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_df["combination"])
    ax.set_xlabel("F2 Score", fontsize=12)
    ax.set_title("Top 15 Ranking+Distance Combinations\n(target_only mode)", fontsize=14)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    
    # Add legend for distance colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2E86AB", label="DTW"),
        Patch(facecolor="#A23B72", label="MMD"),
        Patch(facecolor="#F18F01", label="Wasserstein")
    ]
    ax.legend(handles=legend_elements, loc="lower right", title="Distance")
    
    # Add value labels
    for i, (_, row) in enumerate(top_df.iterrows()):
        ax.annotate(f'{row["f2_mean"]:.3f} ± {row["f2_std"]:.3f}',
                   xy=(row["f2_mean"] + row["f2_std"] + 0.005, i),
                   va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / "best_combinations_ranking.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'best_combinations_ranking.png'}")


def plot_mode_comparison(df: pd.DataFrame, output_dir: Path):
    """Compare pooled vs target_only modes."""
    # Aggregate by mode and ranking method
    summary = df.groupby(["mode", "ranking_method"])["f2"].mean().unstack()
    
    if len(summary) < 2:
        print("Not enough modes for comparison")
        return
    
    summary = summary.T
    summary = summary.reindex([m for m in RANKING_METHODS if m in summary.index])
    summary.index = [RANKING_DISPLAY.get(m, m) for m in summary.index]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(summary.index))
    width = 0.35
    
    if "pooled" in summary.columns:
        ax.bar(x - width/2, summary["pooled"], width, label="Pooled", color="#5B8FB9")
    if "target_only" in summary.columns:
        ax.bar(x + width/2, summary["target_only"], width, label="Target-Only", color="#B97D5B")
    
    ax.set_xlabel("Ranking Method", fontsize=12)
    ax.set_ylabel("F2 Score (mean)", fontsize=12)
    ax.set_title("Mode Comparison: Pooled vs Target-Only", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(summary.index, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "mode_comparison.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'mode_comparison.png'}")


def print_detailed_summary(df: pd.DataFrame):
    """Print detailed summary statistics."""
    print("\n" + "=" * 80)
    print("RANKING METHOD × DISTANCE METRIC COMPARISON RESULTS")
    print("=" * 80)
    
    for mode in ["target_only", "pooled"]:
        mode_df = df[df["mode"] == mode]
        if len(mode_df) == 0:
            continue
        
        print(f"\n{'━' * 40}")
        print(f"【{mode.upper()} MODE】")
        print(f"{'━' * 40}")
        
        # Summary by ranking method
        print("\n📊 By Ranking Method:")
        method_summary = mode_df.groupby("ranking_method").agg({
            "f2": ["mean", "std"],
            "recall": "mean",
            "precision": "mean"
        }).round(4)
        method_summary.columns = ["F2_mean", "F2_std", "Recall", "Precision"]
        method_summary = method_summary.sort_values("F2_mean", ascending=False)
        print(method_summary.to_string())
        
        # Summary by distance metric
        print("\n📊 By Distance Metric:")
        dist_summary = mode_df.groupby("distance").agg({
            "f2": ["mean", "std"],
            "recall": "mean",
            "precision": "mean"
        }).round(4)
        dist_summary.columns = ["F2_mean", "F2_std", "Recall", "Precision"]
        dist_summary = dist_summary.sort_values("F2_mean", ascending=False)
        print(dist_summary.to_string())
        
        # Best combination
        combo_summary = mode_df.groupby(["ranking_method", "distance"])["f2"].mean()
        best_combo = combo_summary.idxmax()
        best_f2 = combo_summary.max()
        print(f"\n🏆 Best Combination: {best_combo[0]} + {best_combo[1]} (F2={best_f2:.4f})")


def generate_latex_table(df: pd.DataFrame, output_dir: Path, mode: str = "target_only"):
    """Generate LaTeX table for paper."""
    mode_df = df[df["mode"] == mode]
    
    if len(mode_df) == 0:
        return
    
    pivot = mode_df.groupby(["ranking_method", "distance"])["f2"].mean().unstack()
    pivot = pivot.reindex(index=[m for m in RANKING_METHODS if m in pivot.index])
    pivot = pivot.reindex(columns=[d for d in DISTANCE_METRICS if d in pivot.columns])
    
    # Find best in each row and column
    row_best = pivot.idxmax(axis=1)
    col_best = pivot.idxmax(axis=0)
    
    latex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{F2 Scores: Ranking Method × Distance Metric}",
        "\\label{tab:ranking_comparison}",
        "\\begin{tabular}{l" + "c" * len(pivot.columns) + "}",
        "\\toprule",
        "Ranking Method & " + " & ".join([DISTANCE_DISPLAY.get(c, c) for c in pivot.columns]) + " \\\\",
        "\\midrule"
    ]
    
    for method in pivot.index:
        row = [RANKING_DISPLAY.get(method, method)]
        for dist in pivot.columns:
            val = pivot.loc[method, dist]
            if pd.isna(val):
                row.append("-")
            else:
                # Bold if best in row or column
                if row_best[method] == dist or col_best[dist] == method:
                    row.append(f"\\textbf{{{val:.3f}}}")
                else:
                    row.append(f"{val:.3f}")
        latex_lines.append(" & ".join(row) + " \\\\")
    
    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    latex_content = "\n".join(latex_lines)
    
    with open(output_dir / f"ranking_comparison_table_{mode}.tex", "w") as f:
        f.write(latex_content)
    
    print(f"Saved: {output_dir / f'ranking_comparison_table_{mode}.tex'}")


def main():
    parser = argparse.ArgumentParser(description="Visualize ranking method comparison results")
    parser.add_argument("--eval-dir", type=str, default="results/evaluation/RF",
                       help="Directory containing evaluation results")
    parser.add_argument("--output-dir", type=str, default="results/domain_analysis/ranking_comparison",
                       help="Output directory for plots and summaries")
    parser.add_argument("--verbose", action="store_true", help="Print detailed output")
    args = parser.parse_args()
    
    eval_dir = PROJECT_ROOT / args.eval_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Collecting results from: {eval_dir}")
    df = collect_all_results(eval_dir, verbose=True)
    
    if len(df) == 0:
        print("No results found.")
        return
    
    # Save raw data
    df.to_csv(output_dir / "ranking_comparison_v2_raw.csv", index=False)
    print(f"\nSaved: {output_dir / 'ranking_comparison_v2_raw.csv'}")
    
    # Check experiment coverage
    print("\n📋 Experiment Coverage:")
    print(f"  Ranking methods: {df['ranking_method'].unique().tolist()}")
    print(f"  Distance metrics: {df['distance'].unique().tolist()}")
    print(f"  Domain levels: {df['level'].unique().tolist()}")
    print(f"  Modes: {df['mode'].unique().tolist()}")
    
    # Print detailed summary
    print_detailed_summary(df)
    
    # Generate visualizations
    print("\n📊 Generating visualizations...")
    
    for mode in ["target_only", "pooled"]:
        if mode in df["mode"].values:
            plot_ranking_by_distance(df, output_dir, mode)
            plot_distance_comparison_heatmap(df, output_dir, mode)
            plot_domain_level_comparison(df, output_dir, mode)
            generate_latex_table(df, output_dir, mode)
    
    plot_best_combinations(df, output_dir)
    plot_mode_comparison(df, output_dir)
    
    print(f"\n✅ Analysis complete. All results saved to: {output_dir}")


if __name__ == "__main__":
    main()
