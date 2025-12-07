#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggregate_multiseed_results.py
==============================

Aggregate evaluation results from multiple seed experiments.

This script collects metrics from experiments run with different random seeds
and computes:
1. Mean ± standard deviation across seeds
2. Statistical tests comparing methods
3. Summary tables for publication

Usage:
    python scripts/python/analysis/imbalance/aggregate_multiseed_results.py

Input:
    results/evaluation/*.json (with seed in tag)

Output:
    results/imbalance_analysis/domain/multiseed_summary.csv
    results/imbalance_analysis/domain/multiseed_report.txt
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# ============================================================
# Configuration
# ============================================================
RESULTS_DIR = Path("results/evaluation")
OUTPUT_DIR = Path("results/imbalance_analysis/domain")

# Pattern to extract method and seed from filenames
# Example: eval_RF_pooled_imbal_v2_smote_seed42_14574661[1].json
PATTERN = re.compile(
    r"eval_(?P<model>\w+)_(?P<mode>\w+)_imbal_v2_(?P<method>\w+)_seed(?P<seed>\d+)_\d+\[\d+\]\.json"
)

# Metrics to aggregate
METRICS = ["recall", "f1", "precision", "f2", "auc_pr", "auc_roc"]


# ============================================================
# Data Collection
# ============================================================
def collect_results() -> pd.DataFrame:
    """Collect all multi-seed evaluation results."""
    records = []
    
    for json_file in RESULTS_DIR.glob("eval_*.json"):
        match = PATTERN.match(json_file.name)
        if not match:
            continue
        
        try:
            with open(json_file) as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load {json_file}: {e}")
            continue
        
        record = {
            "model": match.group("model"),
            "mode": match.group("mode"),
            "method": match.group("method"),
            "seed": int(match.group("seed")),
            "file": json_file.name,
        }
        
        # Extract metrics
        for metric in METRICS:
            record[metric] = data.get(metric, np.nan)
        
        records.append(record)
    
    return pd.DataFrame(records)


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


# ============================================================
# Analysis Functions
# ============================================================
def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean ± std for each method across seeds."""
    summary = []
    
    for method, grp in df.groupby("method"):
        record = {
            "method": method,
            "n_seeds": len(grp),
            "seeds": sorted(grp["seed"].unique().tolist()),
        }
        
        for metric in METRICS:
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
        print(f"[WARN] Baseline method '{baseline}' not found")
        return pd.DataFrame()
    
    baseline_df = df[df["method"] == baseline]
    baseline_seeds = set(baseline_df["seed"].values)
    
    for method in df["method"].unique():
        if method == baseline:
            continue
        
        method_df = df[df["method"] == method]
        common_seeds = baseline_seeds & set(method_df["seed"].values)
        
        if len(common_seeds) < 2:
            print(f"[WARN] Not enough common seeds for {method} vs {baseline}")
            continue
        
        for metric in METRICS:
            b_vals = baseline_df[baseline_df["seed"].isin(common_seeds)].sort_values("seed")[metric].values
            m_vals = method_df[method_df["seed"].isin(common_seeds)].sort_values("seed")[metric].values
            
            if len(b_vals) != len(m_vals):
                continue
            
            # Paired t-test
            t_stat, t_p = stats.ttest_rel(m_vals, b_vals)
            
            # Wilcoxon if enough samples
            if len(b_vals) >= 5:
                try:
                    w_stat, w_p = stats.wilcoxon(m_vals, b_vals)
                except:
                    w_stat, w_p = np.nan, np.nan
            else:
                w_stat, w_p = np.nan, np.nan
            
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


def generate_report(summary_df: pd.DataFrame, comparison_df: pd.DataFrame) -> str:
    """Generate text report for publication."""
    lines = []
    lines.append("=" * 70)
    lines.append("Multi-Seed Experiment Results Summary")
    lines.append("=" * 70)
    lines.append("")
    
    # Summary table
    lines.append("1. Performance Summary (Mean ± Std across seeds)")
    lines.append("-" * 50)
    
    for _, row in summary_df.iterrows():
        lines.append(f"\n  {row['method'].upper()} (n={row['n_seeds']} seeds: {row['seeds']})")
        for metric in ["recall", "f1", "precision", "auc_pr"]:
            mean_key = f"{metric}_mean"
            std_key = f"{metric}_std"
            if pd.notna(row.get(mean_key)):
                lines.append(f"    {metric:12}: {row[mean_key]:.4f} ± {row[std_key]:.4f}")
    
    # Comparison table
    if not comparison_df.empty:
        lines.append("\n" + "=" * 70)
        lines.append("2. Statistical Comparison vs Baseline")
        lines.append("-" * 50)
        
        for method in comparison_df["method"].unique():
            method_results = comparison_df[comparison_df["method"] == method]
            lines.append(f"\n  {method.upper()} vs BASELINE:")
            
            for _, row in method_results.iterrows():
                sig = "***" if row["ttest_p"] < 0.001 else "**" if row["ttest_p"] < 0.01 else "*" if row["ttest_p"] < 0.05 else ""
                lines.append(
                    f"    {row['metric']:12}: {row['mean_diff']:+.4f} "
                    f"(p={row['ttest_p']:.4f}{sig}, d={row['cohens_d']:.2f} [{row['effect_size']}])"
                )
    
    lines.append("\n" + "=" * 70)
    lines.append("End of Report")
    lines.append("=" * 70)
    
    return "\n".join(lines)


# ============================================================
# Main
# ============================================================
def main():
    """Main entry point."""
    print("=" * 60)
    print("[INFO] Multi-Seed Results Aggregation")
    print("=" * 60)
    
    # Collect results
    df = collect_results()
    
    if df.empty:
        print("[ERROR] No multi-seed results found")
        print("        Expected pattern: eval_*_imbal_v2_*_seed*_*.json")
        return 1
    
    print(f"[INFO] Found {len(df)} evaluation files")
    print(f"[INFO] Methods: {sorted(df['method'].unique())}")
    print(f"[INFO] Seeds: {sorted(df['seed'].unique())}")
    
    # Compute summary
    summary_df = compute_summary(df)
    print(f"\n[INFO] Summary computed for {len(summary_df)} methods")
    
    # Compare methods
    comparison_df = compare_methods(df)
    if comparison_df.empty:
        print("[WARN] Could not compute method comparisons")
    else:
        print(f"[INFO] Generated {len(comparison_df)} comparisons")
    
    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    summary_file = OUTPUT_DIR / "multiseed_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"[INFO] Saved summary to: {summary_file}")
    
    if not comparison_df.empty:
        comparison_file = OUTPUT_DIR / "multiseed_comparison.csv"
        comparison_df.to_csv(comparison_file, index=False)
        print(f"[INFO] Saved comparisons to: {comparison_file}")
    
    # Generate report
    report = generate_report(summary_df, comparison_df)
    report_file = OUTPUT_DIR / "multiseed_report.txt"
    report_file.write_text(report)
    print(f"[INFO] Saved report to: {report_file}")
    
    # Print report
    print()
    print(report)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
