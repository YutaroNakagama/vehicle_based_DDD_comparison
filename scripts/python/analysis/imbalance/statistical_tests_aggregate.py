#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
statistical_tests_aggregate.py
==============================

Perform statistical tests on aggregated evaluation metrics.

Since per-subject scores require complex data reconstruction,
this script uses the aggregated metrics across different experimental
configurations as independent samples for statistical testing.

Approach:
- Each (ranking_method × distance_metric × level × mode) combination
  provides one data point per imbalance method
- Compare baseline vs SMOTE methods using paired tests

Usage:
    python scripts/python/analysis/imbalance/statistical_tests_aggregate.py

Input:
    results/imbalance_analysis/domain/all_metrics.csv

Output:
    results/imbalance_analysis/domain/statistical_tests.csv
    results/imbalance_analysis/domain/statistical_tests_summary.txt
"""

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# ============================================================
# Configuration
# ============================================================
INPUT_FILE = Path("results/imbalance_analysis/domain/all_metrics.csv")
OUTPUT_DIR = Path("results/imbalance_analysis/domain")
OUTPUT_CSV = OUTPUT_DIR / "statistical_tests.csv"
OUTPUT_TXT = OUTPUT_DIR / "statistical_tests_summary.txt"

# Metrics to test
METRICS = ["recall", "f1", "precision", "f2", "auc_pr"]

# Method pairs to compare
BASELINE_METHOD = "baseline"
COMPARISON_METHODS = ["smote", "smote_tomek", "smote_rus"]


# ============================================================
# Statistical Functions
# ============================================================
def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size for paired samples."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


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


def wilcoxon_test(g1: np.ndarray, g2: np.ndarray) -> Tuple[float, float]:
    """Perform Wilcoxon signed-rank test."""
    if np.allclose(g1, g2):
        return np.nan, 1.0
    try:
        stat, p = stats.wilcoxon(g1, g2, alternative="two-sided")
        return float(stat), float(p)
    except Exception:
        return np.nan, np.nan


def paired_ttest(g1: np.ndarray, g2: np.ndarray) -> Tuple[float, float]:
    """Perform paired t-test."""
    try:
        stat, p = stats.ttest_rel(g1, g2)
        return float(stat), float(p)
    except Exception:
        return np.nan, np.nan


def bootstrap_ci(data: np.ndarray, confidence: float = 0.95, n_boot: int = 10000) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for the mean."""
    if len(data) == 0:
        return np.nan, np.nan
    np.random.seed(42)
    means = [np.mean(np.random.choice(data, len(data), replace=True)) for _ in range(n_boot)]
    alpha = 1 - confidence
    return np.percentile(means, 100 * alpha / 2), np.percentile(means, 100 * (1 - alpha / 2))


def format_p(p: float) -> str:
    """Format p-value for display."""
    if np.isnan(p):
        return "N/A"
    elif p < 0.001:
        return "< 0.001***"
    elif p < 0.01:
        return f"{p:.4f}**"
    elif p < 0.05:
        return f"{p:.4f}*"
    else:
        return f"{p:.4f}"


# ============================================================
# Analysis Functions
# ============================================================
def run_pairwise_tests(df: pd.DataFrame, metric: str, mode: str, 
                       level_filter: str = None) -> pd.DataFrame:
    """Run pairwise tests between baseline and each comparison method.
    
    Uses each (ranking_method × distance_metric × level) as a paired sample.
    If level_filter is specified, only that level is used.
    """
    results = []
    
    # Filter by mode
    mode_df = df[df["mode"] == mode].copy()
    if mode_df.empty:
        return pd.DataFrame()
    
    # Filter by level if specified
    if level_filter is not None:
        mode_df = mode_df[mode_df["level"] == level_filter]
        if mode_df.empty:
            return pd.DataFrame()
        config_cols = ["ranking_method", "distance_metric"]
    else:
        config_cols = ["ranking_method", "distance_metric", "level"]
    
    for comp_method in COMPARISON_METHODS:
        baseline_scores = []
        method_scores = []
        configs = []
        
        for config_key, grp in mode_df.groupby(config_cols):
            base = grp[grp["imbalance_method"] == BASELINE_METHOD][metric].values
            comp = grp[grp["imbalance_method"] == comp_method][metric].values
            
            if len(base) == 1 and len(comp) == 1:
                baseline_scores.append(base[0])
                method_scores.append(comp[0])
                configs.append(str(config_key))
        
        if len(baseline_scores) < 3:
            continue
        
        b = np.array(baseline_scores)
        m = np.array(method_scores)
        
        # Statistical tests
        w_stat, w_p = wilcoxon_test(b, m)
        t_stat, t_p = paired_ttest(b, m)
        d = cohens_d(m, b)  # positive means method > baseline
        
        # Confidence intervals
        diff = m - b
        diff_ci = bootstrap_ci(diff)
        
        results.append({
            "mode": mode,
            "metric": metric,
            "baseline": BASELINE_METHOD,
            "comparison": comp_method,
            "n_pairs": len(b),
            "baseline_mean": np.mean(b),
            "baseline_std": np.std(b),
            "method_mean": np.mean(m),
            "method_std": np.std(m),
            "mean_diff": np.mean(diff),
            "diff_ci_lower": diff_ci[0],
            "diff_ci_upper": diff_ci[1],
            "wilcoxon_stat": w_stat,
            "wilcoxon_p": w_p,
            "ttest_stat": t_stat,
            "ttest_p": t_p,
            "cohens_d": d,
            "effect_size": interpret_effect(d),
            "significant_05": w_p < 0.05 if not np.isnan(w_p) else False,
            "significant_01": w_p < 0.01 if not np.isnan(w_p) else False,
            "level_filter": level_filter if level_filter else "all",
        })
    
    return pd.DataFrame(results)


def generate_report(df: pd.DataFrame) -> str:
    """Generate summary report."""
    lines = []
    lines.append("=" * 70)
    lines.append("Statistical Analysis Report: Imbalance Methods Comparison")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Methodology:")
    lines.append("  - Paired comparison across experimental configurations")
    lines.append("  - Each (ranking × distance) or (ranking × distance × level) = 1 paired sample")
    lines.append("  - Tests: Wilcoxon signed-rank (non-parametric), paired t-test")
    lines.append("  - Effect size: Cohen's d")
    lines.append("")
    
    for mode in ["target_only", "source_only"]:
        lines.append("=" * 70)
        lines.append(f"Mode: {mode.upper()}")
        lines.append("=" * 70)
        
        mode_df = df[df["mode"] == mode]
        
        # All levels combined
        lines.append("\n  [ALL LEVELS COMBINED]")
        for metric in METRICS:
            metric_df = mode_df[(mode_df["metric"] == metric) & (mode_df["level_filter"] == "all")]
            if metric_df.empty:
                continue
            
            lines.append(f"\n  Metric: {metric.upper()}")
            lines.append("  " + "-" * 50)
            
            for _, row in metric_df.iterrows():
                lines.append(f"\n    {BASELINE_METHOD} vs {row['comparison']}:")
                lines.append(f"      N pairs: {row['n_pairs']}")
                lines.append(f"      Baseline: {row['baseline_mean']:.4f} ± {row['baseline_std']:.4f}")
                lines.append(f"      {row['comparison']}: {row['method_mean']:.4f} ± {row['method_std']:.4f}")
                lines.append(f"      Difference: {row['mean_diff']:+.4f} [{row['diff_ci_lower']:.4f}, {row['diff_ci_upper']:.4f}]")
                lines.append(f"      Wilcoxon p: {format_p(row['wilcoxon_p'])}")
                lines.append(f"      Cohen's d: {row['cohens_d']:.3f} ({row['effect_size']})")
        
        # By domain level
        for level in ["in_domain", "mid_domain", "out_domain"]:
            lines.append(f"\n\n  [{level.upper()}]")
            for metric in ["recall", "f1"]:  # Key metrics only for brevity
                metric_df = mode_df[(mode_df["metric"] == metric) & (mode_df["level_filter"] == level)]
                if metric_df.empty:
                    continue
                
                lines.append(f"\n  Metric: {metric.upper()}")
                lines.append("  " + "-" * 50)
                
                for _, row in metric_df.iterrows():
                    lines.append(f"\n    {BASELINE_METHOD} vs {row['comparison']}:")
                    lines.append(f"      N pairs: {row['n_pairs']}")
                    lines.append(f"      Baseline: {row['baseline_mean']:.4f} ± {row['baseline_std']:.4f}")
                    lines.append(f"      {row['comparison']}: {row['method_mean']:.4f} ± {row['method_std']:.4f}")
                    lines.append(f"      Difference: {row['mean_diff']:+.4f} [{row['diff_ci_lower']:.4f}, {row['diff_ci_upper']:.4f}]")
                    lines.append(f"      Wilcoxon p: {format_p(row['wilcoxon_p'])}")
                    lines.append(f"      Cohen's d: {row['cohens_d']:.3f} ({row['effect_size']})")
    
    # Citation-ready summary
    lines.append("\n" + "=" * 70)
    lines.append("Citation-Ready Statements (target_only, recall)")
    lines.append("=" * 70)
    
    target_recall = df[(df["mode"] == "target_only") & (df["metric"] == "recall")]
    for _, row in target_recall.iterrows():
        sig = "significantly" if row["significant_05"] else "not significantly"
        lines.append(f"\n  {row['comparison'].upper()}:")
        lines.append(
            f'    "{row["comparison"].upper()} {sig} improved recall compared to baseline '
            f'(mean difference: {row["mean_diff"]:+.3f}, 95% CI [{row["diff_ci_lower"]:.3f}, {row["diff_ci_upper"]:.3f}], '
            f'Wilcoxon p = {row["wilcoxon_p"]:.4f}, Cohen\'s d = {row["cohens_d"]:.2f})."'
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
    print("[INFO] Statistical Tests on Aggregate Metrics")
    print("=" * 60)
    print()
    
    if not INPUT_FILE.exists():
        print(f"[ERROR] Input file not found: {INPUT_FILE}")
        return 1
    
    df = pd.read_csv(INPUT_FILE)
    print(f"[INFO] Loaded {len(df)} records from {INPUT_FILE}")
    
    # Exclude pooled records
    df = df[df["level"] != "pooled"]
    print(f"[INFO] After excluding pooled: {len(df)} records")
    print()
    
    # Run tests for each mode and metric
    all_results = []
    
    # Test levels
    levels = [None, "in_domain", "mid_domain", "out_domain"]
    
    for mode in ["target_only", "source_only"]:
        print(f"[INFO] Processing mode: {mode}")
        for level in levels:
            level_name = level if level else "all"
            print(f"  - Level: {level_name}")
            for metric in METRICS:
                result = run_pairwise_tests(df, metric, mode, level_filter=level)
                if not result.empty:
                    all_results.append(result)
    
    if not all_results:
        print("[ERROR] No test results generated!")
        return 1
    
    results_df = pd.concat(all_results, ignore_index=True)
    print(f"\n[INFO] Generated {len(results_df)} test results")
    
    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"[INFO] Saved to: {OUTPUT_CSV}")
    
    # Generate and save report
    report = generate_report(results_df)
    OUTPUT_TXT.write_text(report)
    print(f"[INFO] Saved report to: {OUTPUT_TXT}")
    
    # Print report
    print()
    print(report)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
