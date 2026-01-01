#!/usr/bin/env python3
"""Collect and analyze ranking method comparison results.

This script collects evaluation results from the ranking method comparison
experiment and generates summary statistics and visualizations.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


def find_evaluation_files(eval_dir: Path, pattern: str = "rank_*") -> list:
    """Find all evaluation result files matching the pattern."""
    files = []
    for job_dir in eval_dir.iterdir():
        if job_dir.is_dir():
            for f in job_dir.glob(f"eval_results_RF_*{pattern}*.json"):
                files.append(f)
    return files


def parse_eval_filename(filepath: Path) -> dict:
    """Parse evaluation filename to extract experiment metadata."""
    # Example: eval_results_RF_source_only_rank_knn_dtw_out_domain_14620001.json
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
            # Handle multi-word ranking methods
            method_parts = []
            for i in range(rank_idx + 1, len(parts)):
                if parts[i] in ["dtw", "mmd", "wasserstein"]:
                    info["distance"] = parts[i]
                    break
                method_parts.append(parts[i])
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


def collect_all_results(eval_dir: Path) -> pd.DataFrame:
    """Collect all evaluation results into a DataFrame."""
    files = find_evaluation_files(eval_dir, pattern="rank_")
    
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
    print(f"Collected {len(df)} evaluation results")
    return df


def summarize_by_ranking_method(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize results by ranking method."""
    summary = df.groupby(["ranking_method", "mode", "level"]).agg({
        "f2": ["mean", "std", "count"],
        "recall": ["mean", "std"],
        "precision": ["mean", "std"],
        "auc": ["mean", "std"],
    }).round(4)
    
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    return summary


def plot_ranking_comparison(df: pd.DataFrame, output_dir: Path):
    """Create comparison plots for ranking methods."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # === Plot 1: F2 Score by Ranking Method (target_only mode) ===
    target_only = df[df["mode"] == "target_only"]
    
    if len(target_only) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        method_order = target_only.groupby("ranking_method")["f2"].mean().sort_values(ascending=False).index.tolist()
        
        sns.boxplot(
            data=target_only,
            x="ranking_method",
            y="f2",
            order=method_order,
            palette="viridis",
            ax=ax
        )
        
        ax.set_xlabel("Ranking Method")
        ax.set_ylabel("F2 Score")
        ax.set_title("Ranking Method Comparison (target_only mode)\nN_TRIALS=75, 3 seeds")
        plt.xticks(rotation=45, ha="right")
        
        # Add mean values as text
        for i, method in enumerate(method_order):
            mean_val = target_only[target_only["ranking_method"] == method]["f2"].mean()
            ax.annotate(f"{mean_val:.3f}", (i, mean_val + 0.01), ha="center", fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_dir / "ranking_f2_comparison_target_only.png", dpi=150)
        plt.close()
        print(f"Saved: {output_dir / 'ranking_f2_comparison_target_only.png'}")
    
    # === Plot 2: Heatmap of F2 by method and level ===
    if len(target_only) > 0:
        pivot = target_only.pivot_table(
            values="f2",
            index="ranking_method",
            columns="level",
            aggfunc="mean"
        )
        
        # Reorder columns
        col_order = ["out_domain", "mid_domain", "in_domain"]
        pivot = pivot.reindex(columns=[c for c in col_order if c in pivot.columns])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="YlGnBu",
            ax=ax,
            cbar_kws={"label": "F2 Score"}
        )
        ax.set_title("F2 Score: Ranking Method × Domain Level\n(target_only mode)")
        ax.set_xlabel("Domain Level")
        ax.set_ylabel("Ranking Method")
        
        plt.tight_layout()
        plt.savefig(output_dir / "ranking_f2_heatmap.png", dpi=150)
        plt.close()
        print(f"Saved: {output_dir / 'ranking_f2_heatmap.png'}")
    
    # === Plot 3: All modes comparison ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, mode in enumerate(["pooled", "source_only", "target_only"]):
        mode_df = df[df["mode"] == mode]
        if len(mode_df) == 0:
            continue
        
        method_order = mode_df.groupby("ranking_method")["f2"].mean().sort_values(ascending=False).index.tolist()
        
        sns.boxplot(
            data=mode_df,
            x="ranking_method",
            y="f2",
            order=method_order,
            palette="viridis",
            ax=axes[idx]
        )
        
        axes[idx].set_xlabel("Ranking Method")
        axes[idx].set_ylabel("F2 Score")
        axes[idx].set_title(f"{mode}")
        axes[idx].tick_params(axis="x", rotation=45)
    
    plt.suptitle("Ranking Method Comparison by Mode (N_TRIALS=75)")
    plt.tight_layout()
    plt.savefig(output_dir / "ranking_f2_all_modes.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'ranking_f2_all_modes.png'}")


def print_ranking_summary(df: pd.DataFrame):
    """Print a summary of ranking method performance."""
    print("\n" + "=" * 70)
    print("RANKING METHOD COMPARISON RESULTS (N_TRIALS=75)")
    print("=" * 70)
    
    # Best ranking method by mode
    for mode in ["target_only", "source_only", "pooled"]:
        mode_df = df[df["mode"] == mode]
        if len(mode_df) == 0:
            continue
        
        print(f"\n【{mode} mode】")
        summary = mode_df.groupby("ranking_method").agg({
            "f2": ["mean", "std", "count"],
            "recall": "mean",
            "auc": "mean"
        }).round(4)
        summary.columns = ["f2_mean", "f2_std", "n", "recall_mean", "auc_mean"]
        summary = summary.sort_values("f2_mean", ascending=False)
        
        print(summary.to_string())
        
        best = summary.index[0]
        print(f"\n  → Best: {best} (F2={summary.loc[best, 'f2_mean']:.4f})")


def main():
    parser = argparse.ArgumentParser(description="Collect ranking method comparison results")
    parser.add_argument("--eval-dir", type=str, default="results/evaluation/RF",
                       help="Directory containing evaluation results")
    parser.add_argument("--output-dir", type=str, default="results/domain_analysis/ranking_comparison",
                       help="Output directory for summary and plots")
    args = parser.parse_args()
    
    eval_dir = PROJECT_ROOT / args.eval_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Collecting results from: {eval_dir}")
    df = collect_all_results(eval_dir)
    
    if len(df) == 0:
        print("No results found.")
        return
    
    # Filter to ranking experiments only
    df = df[df["ranking_method"].notna()]
    
    if len(df) == 0:
        print("No ranking experiment results found.")
        return
    
    # Save raw data
    df.to_csv(output_dir / "ranking_comparison_raw.csv", index=False)
    print(f"Saved: {output_dir / 'ranking_comparison_raw.csv'}")
    
    # Generate summary
    summary = summarize_by_ranking_method(df)
    summary.to_csv(output_dir / "ranking_comparison_summary.csv")
    print(f"Saved: {output_dir / 'ranking_comparison_summary.csv'}")
    
    # Print summary
    print_ranking_summary(df)
    
    # Generate plots
    plot_ranking_comparison(df, output_dir)
    
    print(f"\n✅ Analysis complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
