#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
statistical_tests.py
====================

Perform statistical tests on per-subject scores to validate differences
between imbalance handling methods.

Tests performed:
1. Wilcoxon signed-rank test (paired comparison between methods)
2. Effect size (Cohen's d)
3. Confidence intervals

Usage:
    python scripts/python/analysis/imbalance/statistical_tests.py

Input:
    results/imbalance_analysis/domain/subject_scores.csv

Output:
    results/imbalance_analysis/domain/statistical_tests.csv
    results/imbalance_analysis/domain/statistical_tests_summary.txt
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

# Import shared statistical utilities
from src.utils.analysis.statistical_utils import (
    cohens_d,
    interpret_cohens_d,
    wilcoxon_test,
    bootstrap_ci,
    format_p_value,
)

# ============================================================
# Configuration
# ============================================================
INPUT_FILE = Path("results/imbalance_analysis/domain/subject_scores.csv")
OUTPUT_DIR = Path("results/imbalance_analysis/domain")
OUTPUT_CSV = OUTPUT_DIR / "statistical_tests.csv"
OUTPUT_TXT = OUTPUT_DIR / "statistical_tests_summary.txt"

# Metrics to test
METRICS = ["recall", "f1", "precision", "f2"]

# Method pairs to compare
BASELINE_METHOD = "baseline"
COMPARISON_METHODS = ["smote", "smote_tomek", "smote_rus"]


# ============================================================
# Analysis Functions
# ============================================================
def compare_methods(
    df: pd.DataFrame,
    metric: str,
    group_cols: List[str],
) -> pd.DataFrame:
    """Compare baseline vs each method for a given metric.
    
    Parameters
    ----------
    df : pd.DataFrame
        Subject scores with columns: subject, metric, imbalance_method, ...
    metric : str
        Metric to compare (e.g., 'recall')
    group_cols : list
        Columns to group by (e.g., ['ranking_method', 'distance_metric', 'level', 'mode'])
    
    Returns
    -------
    pd.DataFrame
        Test results for each comparison
    """
    results = []
    
    # Get unique groups
    groups = df.groupby(group_cols)
    
    for group_key, group_df in groups:
        group_dict = dict(zip(group_cols, group_key if isinstance(group_key, tuple) else [group_key]))
        
        # Get baseline scores
        baseline_df = group_df[group_df["imbalance_method"] == BASELINE_METHOD]
        if baseline_df.empty:
            continue
        
        baseline_scores = baseline_df.set_index("subject")[metric]
        
        for method in COMPARISON_METHODS:
            method_df = group_df[group_df["imbalance_method"] == method]
            if method_df.empty:
                continue
            
            method_scores = method_df.set_index("subject")[metric]
            
            # Align subjects (only compare subjects present in both)
            common_subjects = baseline_scores.index.intersection(method_scores.index)
            if len(common_subjects) < 5:  # Need at least 5 subjects for meaningful test
                continue
            
            b_vals = baseline_scores.loc[common_subjects].values
            m_vals = method_scores.loc[common_subjects].values
            
            # Perform tests
            stat, p_val = wilcoxon_test(b_vals, m_vals)
            d = cohens_d(m_vals, b_vals)  # Positive d means method > baseline
            
            # Confidence intervals
            b_ci = bootstrap_ci(b_vals)
            m_ci = bootstrap_ci(m_vals)
            
            results.append({
                **group_dict,
                "metric": metric,
                "baseline_method": BASELINE_METHOD,
                "comparison_method": method,
                "n_subjects": len(common_subjects),
                "baseline_mean": np.mean(b_vals),
                "baseline_std": np.std(b_vals),
                "baseline_ci_lower": b_ci[0],
                "baseline_ci_upper": b_ci[1],
                "method_mean": np.mean(m_vals),
                "method_std": np.std(m_vals),
                "method_ci_lower": m_ci[0],
                "method_ci_upper": m_ci[1],
                "mean_diff": np.mean(m_vals) - np.mean(b_vals),
                "wilcoxon_stat": stat,
                "p_value": p_val,
                "cohens_d": d,
                "effect_size": interpret_cohens_d(d),
                "significant_05": p_val < 0.05 if not np.isnan(p_val) else False,
                "significant_01": p_val < 0.01 if not np.isnan(p_val) else False,
            })
    
    return pd.DataFrame(results)


def generate_summary_report(df: pd.DataFrame) -> str:
    """Generate human-readable summary report."""
    lines = []
    lines.append("=" * 70)
    lines.append("Statistical Analysis Summary")
    lines.append("=" * 70)
    lines.append("")
    
    # Overall summary
    lines.append("1. Overall Comparison (across all configurations)")
    lines.append("-" * 50)
    
    for metric in METRICS:
        lines.append(f"\n  Metric: {metric.upper()}")
        metric_df = df[df["metric"] == metric]
        
        for method in COMPARISON_METHODS:
            method_df = metric_df[metric_df["comparison_method"] == method]
            if method_df.empty:
                continue
            
            n_sig_05 = method_df["significant_05"].sum()
            n_sig_01 = method_df["significant_01"].sum()
            n_total = len(method_df)
            mean_d = method_df["cohens_d"].mean()
            mean_diff = method_df["mean_diff"].mean()
            
            lines.append(f"    {BASELINE_METHOD} vs {method}:")
            lines.append(f"      Significant (p<0.05): {n_sig_05}/{n_total} ({100*n_sig_05/n_total:.1f}%)")
            lines.append(f"      Significant (p<0.01): {n_sig_01}/{n_total} ({100*n_sig_01/n_total:.1f}%)")
            lines.append(f"      Mean difference: {mean_diff:+.4f}")
            lines.append(f"      Mean Cohen's d: {mean_d:.3f} ({interpret_cohens_d(mean_d)})")
    
    # Best configuration
    lines.append("\n" + "=" * 70)
    lines.append("2. Best Configurations (largest significant improvements)")
    lines.append("-" * 50)
    
    recall_df = df[(df["metric"] == "recall") & (df["significant_05"] == True)]
    if not recall_df.empty:
        best = recall_df.nlargest(5, "cohens_d")
        lines.append("\n  Top 5 by Cohen's d (Recall):")
        for _, row in best.iterrows():
            lines.append(
                f"    {row['comparison_method']}/{row.get('ranking_method', 'N/A')}/{row.get('distance_metric', 'N/A')}/{row.get('level', 'N/A')}:"
            )
            lines.append(
                f"      d={row['cohens_d']:.3f}, p={format_p_value(row['p_value'])}, "
                f"diff={row['mean_diff']:+.4f}"
            )
    
    # Citation-ready summary
    lines.append("\n" + "=" * 70)
    lines.append("3. Citation-Ready Statements")
    lines.append("-" * 50)
    
    for method in COMPARISON_METHODS:
        recall_method = df[(df["metric"] == "recall") & (df["comparison_method"] == method)]
        if recall_method.empty:
            continue
        
        # Aggregate test (use mean across configurations)
        all_diffs = recall_method["mean_diff"].values
        all_d = recall_method["cohens_d"].values
        n_sig = recall_method["significant_05"].sum()
        n_total = len(recall_method)
        
        lines.append(f"\n  {method.upper()} vs baseline:")
        lines.append(
            f"    \"{method.upper()} significantly improved Recall compared to baseline "
            f"in {n_sig}/{n_total} ({100*n_sig/n_total:.0f}%) experimental configurations "
            f"(mean improvement: {np.mean(all_diffs):+.3f}, "
            f"mean Cohen's d = {np.mean(all_d):.2f}, {interpret_cohens_d(np.mean(all_d))}).\""
        )
    
    lines.append("")
    lines.append("=" * 70)
    lines.append("End of Report")
    lines.append("=" * 70)
    
    return "\n".join(lines)


# ============================================================
# Main
# ============================================================
def main():
    """Main entry point."""
    print("=" * 60)
    print("[INFO] Running Statistical Tests on Subject Scores")
    print("=" * 60)
    print()
    
    # Load subject scores
    if not INPUT_FILE.exists():
        print(f"[ERROR] Input file not found: {INPUT_FILE}")
        print("[INFO] Run compute_subject_scores.py first!")
        return 1
    
    df = pd.read_csv(INPUT_FILE)
    print(f"[INFO] Loaded {len(df)} subject-level records")
    print()
    
    # Run comparisons for each metric
    all_results = []
    group_cols = ["ranking_method", "distance_metric", "level", "mode"]
    
    for metric in METRICS:
        print(f"[INFO] Testing metric: {metric}")
        result = compare_methods(df, metric, group_cols)
        all_results.append(result)
    
    # Combine results
    results_df = pd.concat(all_results, ignore_index=True)
    print(f"\n[INFO] Completed {len(results_df)} statistical comparisons")
    
    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"[INFO] Saved detailed results to: {OUTPUT_CSV}")
    
    # Generate and save summary
    summary = generate_summary_report(results_df)
    OUTPUT_TXT.write_text(summary)
    print(f"[INFO] Saved summary report to: {OUTPUT_TXT}")
    
    # Print summary
    print()
    print(summary)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
