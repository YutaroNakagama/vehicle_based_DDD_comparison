#!/usr/bin/env python3
"""Visualize Multi-seed Imbalance Method Comparison Results.

This script collects evaluation results from multi-seed experiments
and creates comprehensive visualizations.
"""

import json
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def collect_evaluation_results(project_root: Path, job_prefix: str = "14618") -> pd.DataFrame:
    """Collect evaluation results from all model types.
    
    Parameters
    ----------
    project_root : Path
        Project root directory
    job_prefix : str
        Job ID prefix to filter (e.g., "14618")
    
    Returns
    -------
    pd.DataFrame
        DataFrame with all evaluation results
    """
    results = []
    
    for model_type in ["RF", "BalancedRF", "EasyEnsemble"]:
        eval_base = project_root / f"results/evaluation/{model_type}"
        if not eval_base.exists():
            continue
        
        for jobdir in sorted(eval_base.glob(f"{job_prefix}*")):
            jobid = jobdir.name
            subdir = jobdir / f"{jobid}[1]"
            if subdir.exists():
                for jsonfile in subdir.glob("eval_results*.json"):
                    try:
                        with open(jsonfile) as f:
                            data = json.load(f)
                            tag = data.get("tag", "unknown")
                            
                            # Extract seed
                            seed_match = re.search(r'seed(\d+)', tag)
                            seed = int(seed_match.group(1)) if seed_match else 42
                            
                            # Extract ratio
                            ratio_match = re.search(r'ratio(\d+)_(\d+)', tag)
                            if ratio_match:
                                ratio = float(f"{ratio_match.group(1)}.{ratio_match.group(2)}")
                            else:
                                ratio = None
                            
                            # Extract method
                            method = tag.replace("imbal_v2_", "").split("_seed")[0]
                            if ratio:
                                method = re.sub(r'_ratio\d+_\d+', '', method)
                            
                            results.append({
                                "jobid": jobid,
                                "model_type": model_type,
                                "tag": tag,
                                "method": method,
                                "ratio": ratio,
                                "seed": seed,
                                "f2": data.get("f2_thr", 0),
                                "recall": data.get("recall_thr", 0),
                                "precision": data.get("prec_thr", 0),
                                "accuracy": data.get("acc_thr", 0),
                                "f1": data.get("f1_thr", 0),
                                "specificity": data.get("specificity_thr", 0),
                            })
                    except Exception as e:
                        logger.warning(f"Error reading {jsonfile}: {e}")
    
    return pd.DataFrame(results)


def plot_method_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """Create method comparison visualization.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with evaluation results
    output_path : Path
        Output path for the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Method order by mean F2
    method_order = df.groupby("method")["f2"].mean().sort_values(ascending=False).index.tolist()
    
    # Panel 1: F2 Score by Method (box plot)
    ax1 = axes[0, 0]
    data_for_box = [df[df["method"] == m]["f2"].values for m in method_order]
    bp = ax1.boxplot(data_for_box, labels=method_order, patch_artist=True)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(method_order)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_xticklabels(method_order, rotation=45, ha="right", fontsize=9)
    ax1.set_ylabel("F2 Score")
    ax1.set_title("F2 Score by Method (All Seeds & Ratios)", fontweight="bold")
    ax1.axhline(df[df["method"] == "baseline"]["f2"].mean(), color="red", linestyle="--", 
                alpha=0.7, label="Baseline mean")
    ax1.legend(loc="upper right")
    ax1.grid(axis="y", alpha=0.3)
    
    # Panel 2: Recall vs Precision scatter
    ax2 = axes[0, 1]
    for method in method_order:
        subset = df[df["method"] == method]
        ax2.scatter(subset["precision"], subset["recall"], label=method, alpha=0.7, s=50)
    
    ax2.set_xlabel("Precision")
    ax2.set_ylabel("Recall")
    ax2.set_title("Precision-Recall Trade-off by Method", fontweight="bold")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax2.grid(alpha=0.3)
    
    # Panel 3: F2 by Ratio (for ratio-variable methods)
    ax3 = axes[1, 0]
    ratio_methods = df[df["ratio"].notna()]["method"].unique()
    
    x = np.arange(len([0.1, 0.5, 1.0]))
    width = 0.12
    
    for i, method in enumerate(ratio_methods):
        subset = df[df["method"] == method]
        means = []
        stds = []
        for ratio in [0.1, 0.5, 1.0]:
            ratio_data = subset[subset["ratio"] == ratio]["f2"]
            means.append(ratio_data.mean() if len(ratio_data) > 0 else 0)
            stds.append(ratio_data.std() if len(ratio_data) > 1 else 0)
        
        ax3.bar(x + i * width, means, width, label=method, yerr=stds, capsize=2, alpha=0.8)
    
    ax3.set_xticks(x + width * (len(ratio_methods) - 1) / 2)
    ax3.set_xticklabels(["0.1", "0.5", "1.0"])
    ax3.set_xlabel("Sampling Ratio (minority/majority)")
    ax3.set_ylabel("F2 Score")
    ax3.set_title("F2 Score by Sampling Ratio", fontweight="bold")
    ax3.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax3.grid(axis="y", alpha=0.3)
    
    # Panel 4: Summary table
    ax4 = axes[1, 1]
    ax4.axis("off")
    
    summary = df.groupby("method").agg({
        "f2": ["mean", "std", "count"],
        "recall": "mean",
        "precision": "mean"
    }).round(4)
    summary.columns = ["F2 Mean", "F2 Std", "N", "Recall", "Precision"]
    summary = summary.sort_values("F2 Mean", ascending=False).reset_index()
    
    table_data = []
    for _, row in summary.iterrows():
        table_data.append([
            row["method"],
            f"{row['F2 Mean']:.4f}",
            f"{row['F2 Std']:.4f}" if pd.notna(row['F2 Std']) else "-",
            f"{int(row['N'])}",
            f"{row['Recall']:.3f}",
            f"{row['Precision']:.4f}",
        ])
    
    table = ax4.table(
        cellText=table_data,
        colLabels=["Method", "F2 Mean", "F2 Std", "N", "Recall", "Precision"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Color header
    for j in range(6):
        table[(0, j)].set_facecolor("#34495e")
        table[(0, j)].set_text_props(color="white", fontweight="bold")
    
    # Highlight best result
    table[(1, 0)].set_facecolor("#d5f5e3")
    table[(1, 1)].set_facecolor("#d5f5e3")
    
    ax4.set_title("Summary Statistics (Sorted by F2)", fontweight="bold", y=0.95)
    
    fig.suptitle("Multi-seed Imbalance Method Comparison\n(3 seeds × multiple ratios)", 
                 fontsize=14, fontweight="bold", y=0.98)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {output_path}")
    plt.close(fig)


def plot_seed_stability(df: pd.DataFrame, output_path: Path) -> None:
    """Create seed stability visualization.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with evaluation results
    output_path : Path
        Output path for the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel 1: F2 by seed for each method
    ax1 = axes[0]
    methods = df.groupby("method")["f2"].mean().sort_values(ascending=False).index.tolist()
    
    x = np.arange(len(methods))
    width = 0.25
    
    for i, seed in enumerate([42, 123, 456]):
        means = []
        for method in methods:
            seed_data = df[(df["method"] == method) & (df["seed"] == seed)]["f2"]
            means.append(seed_data.mean() if len(seed_data) > 0 else 0)
        ax1.bar(x + i * width, means, width, label=f"Seed {seed}", alpha=0.8)
    
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(methods, rotation=45, ha="right", fontsize=9)
    ax1.set_ylabel("F2 Score")
    ax1.set_title("F2 Score Stability Across Seeds", fontweight="bold")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    
    # Panel 2: Coefficient of variation (CV) by method
    ax2 = axes[1]
    cv_data = df.groupby("method")["f2"].agg(["mean", "std"])
    cv_data["cv"] = (cv_data["std"] / cv_data["mean"] * 100).fillna(0)
    cv_data = cv_data.sort_values("cv")
    
    colors = ["#2ecc71" if cv < 10 else "#f39c12" if cv < 20 else "#e74c3c" 
              for cv in cv_data["cv"]]
    
    ax2.barh(cv_data.index, cv_data["cv"], color=colors, edgecolor="black", alpha=0.8)
    ax2.set_xlabel("Coefficient of Variation (%)")
    ax2.set_title("Seed Stability (Lower = More Stable)", fontweight="bold")
    ax2.axvline(10, color="green", linestyle="--", alpha=0.5, label="Good (<10%)")
    ax2.axvline(20, color="orange", linestyle="--", alpha=0.5, label="Moderate (<20%)")
    ax2.legend(loc="lower right")
    ax2.grid(axis="x", alpha=0.3)
    
    fig.suptitle("Seed Stability Analysis", fontsize=14, fontweight="bold", y=1.02)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {output_path}")
    plt.close(fig)


def main():
    """Main function."""
    project_root = Path("/home/s2240011/git/ddd/vehicle_based_DDD_comparison")
    output_dir = project_root / "results/imbalance_analysis/multiseed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Collecting evaluation results...")
    df = collect_evaluation_results(project_root, job_prefix="14618")
    logger.info(f"Collected {len(df)} results")
    
    # Also include recent re-run jobs
    df2 = collect_evaluation_results(project_root, job_prefix="14619")
    if len(df2) > 0:
        logger.info(f"Also collected {len(df2)} results from re-run jobs")
        df = pd.concat([df, df2], ignore_index=True)
    
    # Remove duplicates (keep latest)
    df = df.drop_duplicates(subset=["method", "ratio", "seed"], keep="last")
    logger.info(f"After deduplication: {len(df)} unique results")
    
    # Print summary
    print("\n" + "="*60)
    print("MULTI-SEED EXPERIMENT RESULTS SUMMARY")
    print("="*60)
    print(f"\nTotal results: {len(df)}")
    print(f"\nBy method:")
    print(df.groupby("method")["f2"].agg(["mean", "std", "count"]).sort_values("mean", ascending=False).to_string())
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    
    plot_method_comparison(df, output_dir / "method_comparison.png")
    plot_seed_stability(df, output_dir / "seed_stability.png")
    
    # Save raw data
    df.to_csv(output_dir / "multiseed_results.csv", index=False)
    logger.info(f"Saved raw data to {output_dir / 'multiseed_results.csv'}")
    
    print(f"\n✅ Visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
