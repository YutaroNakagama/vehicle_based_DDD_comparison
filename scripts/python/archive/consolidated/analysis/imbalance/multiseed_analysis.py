#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
multiseed_analysis.py
=====================

Unified tool for multi-seed experiment analysis.

This script provides:
1. aggregate - Collect and aggregate metrics from multi-seed experiments
2. visualize - Create visualizations for multi-seed results
3. report    - Generate statistical comparison report

Consolidates functionality from:
- aggregate_multiseed_results.py
- visualize_multiseed_results.py

Usage:
    python scripts/python/analysis/imbalance/multiseed_analysis.py aggregate
    python scripts/python/analysis/imbalance/multiseed_analysis.py visualize
    python scripts/python/analysis/imbalance/multiseed_analysis.py report
    python scripts/python/analysis/imbalance/multiseed_analysis.py all
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

# Setup matplotlib before importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# Configuration
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[4]
RESULTS_DIR = PROJECT_ROOT / "results" / "evaluation"
OUTPUT_DIR = PROJECT_ROOT / "results" / "imbalance_analysis" / "multiseed"

# Pattern to extract method and seed from filenames
PATTERN = re.compile(
    r"eval_(?P<model>\w+)_(?P<mode>\w+)_imbal_v2_(?P<method>\w+)_seed(?P<seed>\d+)_\d+\[\d+\]\.json"
)

# Metrics to aggregate
METRICS = ["recall", "f1", "precision", "f2", "auc_pr", "auc_roc"]

# Setup logger
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# Data Collection
# ============================================================
def collect_from_flat_structure() -> pd.DataFrame:
    """Collect results from flat evaluation directory structure."""
    records = []
    
    for json_file in RESULTS_DIR.glob("eval_*.json"):
        match = PATTERN.match(json_file.name)
        if not match:
            continue
        
        try:
            with open(json_file) as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load {json_file}: {e}")
            continue
        
        record = {
            "model": match.group("model"),
            "mode": match.group("mode"),
            "method": match.group("method"),
            "seed": int(match.group("seed")),
            "file": json_file.name,
        }
        
        for metric in METRICS:
            record[metric] = data.get(metric, np.nan)
        
        records.append(record)
    
    return pd.DataFrame(records)


def collect_from_job_structure(job_prefix: str = "14618") -> pd.DataFrame:
    """Collect results from model/job directory structure."""
    records = []
    
    for model_type in ["RF", "BalancedRF", "EasyEnsemble"]:
        eval_base = RESULTS_DIR / model_type
        if not eval_base.exists():
            continue
        
        for jobdir in sorted(eval_base.glob(f"{job_prefix}*")):
            jobid = jobdir.name
            subdir = jobdir / f"{jobid}[1]"
            if not subdir.exists():
                continue
                
            for jsonfile in subdir.glob("eval_results*.json"):
                try:
                    with open(jsonfile) as f:
                        data = json.load(f)
                    
                    tag = data.get("tag", "unknown")
                    
                    # Extract seed
                    seed_match = re.search(r"seed(\d+)", tag)
                    seed = int(seed_match.group(1)) if seed_match else 42
                    
                    # Extract ratio
                    ratio_match = re.search(r"ratio(\d+)_(\d+)", tag)
                    ratio = float(f"{ratio_match.group(1)}.{ratio_match.group(2)}") if ratio_match else None
                    
                    # Extract method
                    method = tag.replace("imbal_v2_", "").split("_seed")[0]
                    if ratio:
                        method = re.sub(r"_ratio\d+_\d+", "", method)
                    
                    records.append({
                        "jobid": jobid,
                        "model": model_type,
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
                        "auc_pr": data.get("auc_pr", np.nan),
                        "auc_roc": data.get("auc_roc", np.nan),
                    })
                except Exception as e:
                    logger.warning(f"Error reading {jsonfile}: {e}")
    
    return pd.DataFrame(records)


def collect_results(job_prefixes: Optional[List[str]] = None) -> pd.DataFrame:
    """Collect all multi-seed evaluation results."""
    # First try flat structure
    df = collect_from_flat_structure()
    
    # If not found, try job structure
    if df.empty and job_prefixes:
        dfs = []
        for prefix in job_prefixes:
            df_prefix = collect_from_job_structure(prefix)
            if not df_prefix.empty:
                dfs.append(df_prefix)
        
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            df = df.drop_duplicates(subset=["method", "ratio", "seed"], keep="last")
    
    return df


# ============================================================
# Statistical Functions
# ============================================================
def cohens_d(g1: np.ndarray, g2: np.ndarray) -> float:
    """Calculate Cohen's d effect size."""
    n1, n2 = len(g1), len(g2)
    var1, var2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(g1) - np.mean(g2)) / pooled_std


def interpret_effect(d: float) -> str:
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    elif abs_d < 1.0:
        return "large"
    else:
        return "very large"


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean ± std for each method across seeds."""
    summary = []
    
    for method, grp in df.groupby("method"):
        record = {
            "method": method,
            "n_seeds": grp["seed"].nunique(),
            "seeds": sorted(grp["seed"].unique().tolist()),
        }
        
        for metric in METRICS:
            if metric not in grp.columns:
                continue
            values = grp[metric].dropna().values
            if len(values) > 0:
                record[f"{metric}_mean"] = np.mean(values)
                record[f"{metric}_std"] = np.std(values, ddof=1) if len(values) > 1 else 0.0
            else:
                record[f"{metric}_mean"] = np.nan
                record[f"{metric}_std"] = np.nan
        
        summary.append(record)
    
    return pd.DataFrame(summary)


def compare_methods(df: pd.DataFrame, baseline: str = "baseline") -> pd.DataFrame:
    """Compare each method to baseline across seeds."""
    results = []
    
    if baseline not in df["method"].values:
        logger.warning(f"Baseline method '{baseline}' not found")
        return pd.DataFrame()
    
    baseline_df = df[df["method"] == baseline]
    baseline_seeds = set(baseline_df["seed"].values)
    
    for method in df["method"].unique():
        if method == baseline:
            continue
        
        method_df = df[df["method"] == method]
        common_seeds = baseline_seeds & set(method_df["seed"].values)
        
        if len(common_seeds) < 2:
            logger.warning(f"Not enough common seeds for {method} vs {baseline}")
            continue
        
        for metric in METRICS:
            if metric not in df.columns:
                continue
            
            b_vals = baseline_df[baseline_df["seed"].isin(common_seeds)].sort_values("seed")[metric].values
            m_vals = method_df[method_df["seed"].isin(common_seeds)].sort_values("seed")[metric].values
            
            if len(b_vals) != len(m_vals) or len(b_vals) == 0:
                continue
            
            # Paired t-test
            t_stat, t_p = stats.ttest_rel(m_vals, b_vals)
            
            # Wilcoxon if enough samples
            w_stat, w_p = np.nan, np.nan
            if len(b_vals) >= 5:
                try:
                    w_stat, w_p = stats.wilcoxon(m_vals, b_vals)
                except:
                    pass
            
            d = cohens_d(m_vals, b_vals)
            
            results.append({
                "method": method,
                "metric": metric,
                "n_seeds": len(common_seeds),
                "baseline_mean": np.mean(b_vals),
                "baseline_std": np.std(b_vals, ddof=1),
                "method_mean": np.mean(m_vals),
                "method_std": np.std(m_vals, ddof=1),
                "mean_diff": np.mean(m_vals - b_vals),
                "ttest_p": t_p,
                "wilcoxon_p": w_p if not np.isnan(w_p) else None,
                "cohens_d": d,
                "effect_size": interpret_effect(d),
            })
    
    return pd.DataFrame(results)


# ============================================================
# Visualization Functions
# ============================================================
def plot_method_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """Create method comparison visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Method order by mean F2
    method_order = df.groupby("method")["f2"].mean().sort_values(ascending=False).index.tolist()
    
    # Panel 1: F2 Score by Method (box plot)
    ax1 = axes[0, 0]
    data_for_box = [df[df["method"] == m]["f2"].values for m in method_order]
    bp = ax1.boxplot(data_for_box, labels=method_order, patch_artist=True)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(method_order)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_xticklabels(method_order, rotation=45, ha="right", fontsize=9)
    ax1.set_ylabel("F2 Score")
    ax1.set_title("F2 Score by Method (All Seeds & Ratios)", fontweight="bold")
    
    if "baseline" in method_order:
        ax1.axhline(
            df[df["method"] == "baseline"]["f2"].mean(),
            color="red", linestyle="--", alpha=0.7, label="Baseline mean"
        )
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
    if "ratio" in df.columns:
        ratio_methods = df[df["ratio"].notna()]["method"].unique()
        ratios = [0.1, 0.5, 1.0]
        
        if len(ratio_methods) > 0:
            x = np.arange(len(ratios))
            width = 0.12
            
            for i, method in enumerate(ratio_methods):
                subset = df[df["method"] == method]
                means = []
                stds = []
                for ratio in ratios:
                    ratio_data = subset[subset["ratio"] == ratio]["f2"]
                    means.append(ratio_data.mean() if len(ratio_data) > 0 else 0)
                    stds.append(ratio_data.std() if len(ratio_data) > 1 else 0)
                
                ax3.bar(x + i * width, means, width, label=method, yerr=stds, capsize=2, alpha=0.8)
            
            ax3.set_xticks(x + width * (len(ratio_methods) - 1) / 2)
            ax3.set_xticklabels(["0.1", "0.5", "1.0"])
            ax3.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    
    ax3.set_xlabel("Sampling Ratio (minority/majority)")
    ax3.set_ylabel("F2 Score")
    ax3.set_title("F2 Score by Sampling Ratio", fontweight="bold")
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
            f"{row['F2 Std']:.4f}" if pd.notna(row["F2 Std"]) else "-",
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
    
    for j in range(6):
        table[(0, j)].set_facecolor("#34495e")
        table[(0, j)].set_text_props(color="white", fontweight="bold")
    
    if len(table_data) > 0:
        table[(1, 0)].set_facecolor("#d5f5e3")
        table[(1, 1)].set_facecolor("#d5f5e3")
    
    ax4.set_title("Summary Statistics (Sorted by F2)", fontweight="bold", y=0.95)
    
    fig.suptitle(
        "Multi-seed Imbalance Method Comparison\n(multiple seeds × multiple ratios)",
        fontsize=14, fontweight="bold", y=0.98
    )
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {output_path}")
    plt.close(fig)


def plot_seed_stability(df: pd.DataFrame, output_path: Path) -> None:
    """Create seed stability visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    methods = df.groupby("method")["f2"].mean().sort_values(ascending=False).index.tolist()
    seeds = sorted(df["seed"].unique())
    
    # Panel 1: F2 by seed for each method
    ax1 = axes[0]
    x = np.arange(len(methods))
    width = 0.25
    
    for i, seed in enumerate(seeds[:3]):  # Max 3 seeds for readability
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
    
    colors = [
        "#2ecc71" if cv < 10 else "#f39c12" if cv < 20 else "#e74c3c"
        for cv in cv_data["cv"]
    ]
    
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


# ============================================================
# Report Generation
# ============================================================
def generate_report(summary_df: pd.DataFrame, comparison_df: pd.DataFrame) -> str:
    """Generate text report for publication."""
    lines = []
    lines.append("=" * 70)
    lines.append("Multi-Seed Experiment Results Summary")
    lines.append("=" * 70)
    lines.append("")
    
    lines.append("1. Performance Summary (Mean ± Std across seeds)")
    lines.append("-" * 50)
    
    for _, row in summary_df.iterrows():
        seeds_str = str(row["seeds"]) if "seeds" in row else "N/A"
        lines.append(f"\n  {row['method'].upper()} (n={row['n_seeds']} seeds: {seeds_str})")
        for metric in ["recall", "f1", "precision", "auc_pr"]:
            mean_key = f"{metric}_mean"
            std_key = f"{metric}_std"
            if mean_key in row and pd.notna(row.get(mean_key)):
                lines.append(f"    {metric:12}: {row[mean_key]:.4f} ± {row[std_key]:.4f}")
    
    if not comparison_df.empty:
        lines.append("\n" + "=" * 70)
        lines.append("2. Statistical Comparison vs Baseline")
        lines.append("-" * 50)
        
        for method in comparison_df["method"].unique():
            method_results = comparison_df[comparison_df["method"] == method]
            lines.append(f"\n  {method.upper()} vs BASELINE:")
            
            for _, row in method_results.iterrows():
                sig = (
                    "***" if row["ttest_p"] < 0.001 else
                    "**" if row["ttest_p"] < 0.01 else
                    "*" if row["ttest_p"] < 0.05 else ""
                )
                lines.append(
                    f"    {row['metric']:12}: {row['mean_diff']:+.4f} "
                    f"(p={row['ttest_p']:.4f}{sig}, d={row['cohens_d']:.2f} [{row['effect_size']}])"
                )
    
    lines.append("\n" + "=" * 70)
    lines.append("End of Report")
    lines.append("=" * 70)
    
    return "\n".join(lines)


# ============================================================
# CLI Commands
# ============================================================
def cmd_aggregate(args) -> int:
    """Aggregate multi-seed results."""
    print("=" * 60)
    print("[INFO] Multi-Seed Results Aggregation")
    print("=" * 60)
    
    df = collect_results(job_prefixes=args.job_prefixes.split(",") if args.job_prefixes else None)
    
    if df.empty:
        print("[ERROR] No multi-seed results found")
        return 1
    
    print(f"[INFO] Found {len(df)} evaluation files")
    print(f"[INFO] Methods: {sorted(df['method'].unique())}")
    print(f"[INFO] Seeds: {sorted(df['seed'].unique())}")
    
    summary_df = compute_summary(df)
    print(f"\n[INFO] Summary computed for {len(summary_df)} methods")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    summary_file = OUTPUT_DIR / "multiseed_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"[INFO] Saved summary to: {summary_file}")
    
    raw_file = OUTPUT_DIR / "multiseed_results.csv"
    df.to_csv(raw_file, index=False)
    print(f"[INFO] Saved raw data to: {raw_file}")
    
    return 0


def cmd_visualize(args) -> int:
    """Create visualizations for multi-seed results."""
    print("=" * 60)
    print("[INFO] Multi-Seed Visualization")
    print("=" * 60)
    
    df = collect_results(job_prefixes=args.job_prefixes.split(",") if args.job_prefixes else None)
    
    if df.empty:
        print("[ERROR] No results to visualize")
        return 1
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    plot_method_comparison(df, OUTPUT_DIR / "method_comparison.png")
    plot_seed_stability(df, OUTPUT_DIR / "seed_stability.png")
    
    print(f"\n✅ Visualizations saved to: {OUTPUT_DIR}")
    return 0


def cmd_report(args) -> int:
    """Generate statistical comparison report."""
    print("=" * 60)
    print("[INFO] Multi-Seed Report Generation")
    print("=" * 60)
    
    df = collect_results(job_prefixes=args.job_prefixes.split(",") if args.job_prefixes else None)
    
    if df.empty:
        print("[ERROR] No results found")
        return 1
    
    summary_df = compute_summary(df)
    comparison_df = compare_methods(df, baseline=args.baseline)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not comparison_df.empty:
        comparison_file = OUTPUT_DIR / "multiseed_comparison.csv"
        comparison_df.to_csv(comparison_file, index=False)
        print(f"[INFO] Saved comparisons to: {comparison_file}")
    
    report = generate_report(summary_df, comparison_df)
    report_file = OUTPUT_DIR / "multiseed_report.txt"
    report_file.write_text(report)
    print(f"[INFO] Saved report to: {report_file}")
    
    print()
    print(report)
    
    return 0


def cmd_all(args) -> int:
    """Run all analysis steps."""
    for cmd in [cmd_aggregate, cmd_visualize, cmd_report]:
        result = cmd(args)
        if result != 0:
            return result
        print()
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Multi-seed experiment analysis tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python multiseed_analysis.py aggregate
    python multiseed_analysis.py visualize --job-prefixes 14618,14619
    python multiseed_analysis.py report --baseline baseline
    python multiseed_analysis.py all
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Common arguments
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--job-prefixes",
        dest="job_prefixes",
        help="Comma-separated job ID prefixes to search (e.g., 14618,14619)"
    )
    
    # aggregate
    p_agg = subparsers.add_parser(
        "aggregate", parents=[common],
        help="Collect and aggregate metrics from multi-seed experiments"
    )
    p_agg.set_defaults(func=cmd_aggregate)
    
    # visualize
    p_vis = subparsers.add_parser(
        "visualize", parents=[common],
        help="Create visualizations for multi-seed results"
    )
    p_vis.set_defaults(func=cmd_visualize)
    
    # report
    p_rep = subparsers.add_parser(
        "report", parents=[common],
        help="Generate statistical comparison report"
    )
    p_rep.add_argument(
        "--baseline", default="baseline",
        help="Baseline method for comparison (default: baseline)"
    )
    p_rep.set_defaults(func=cmd_report)
    
    # all
    p_all = subparsers.add_parser(
        "all", parents=[common],
        help="Run aggregate, visualize, and report"
    )
    p_all.add_argument("--baseline", default="baseline")
    p_all.set_defaults(func=cmd_all)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
