#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
statistical_tests_unified.py
============================
Unified script for statistical tests on imbalance method comparisons.

This script consolidates:
- statistical_tests.py (per-subject analysis)
- statistical_tests_aggregate.py (aggregated metrics analysis)

Usage:
    python statistical_tests_unified.py per-subject   # Per-subject analysis
    python statistical_tests_unified.py aggregate     # Aggregated metrics analysis
    python statistical_tests_unified.py all           # Both analyses
"""

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

# Import shared statistical utilities
from src.analysis.domain.statistical_utils import (
    cohens_d,
    interpret_cohens_d,
    wilcoxon_test,
    paired_ttest,
    bootstrap_ci,
    format_p_value,
)

# ============================================================
# Configuration
# ============================================================
INPUT_FILE_SUBJECTS = Path("results/imbalance_analysis/domain/subject_scores.csv")
INPUT_FILE_AGGREGATE = Path("results/imbalance_analysis/domain/all_metrics.csv")
OUTPUT_DIR = Path("results/imbalance_analysis/domain")

# Metrics to test
METRICS_SUBJECT = ["recall", "f1", "precision", "f2"]
METRICS_AGGREGATE = ["recall", "f1", "precision", "f2", "auc_pr"]

# Methods
BASELINE_METHOD = "baseline"
COMPARISON_METHODS = ["smote", "smote_tomek", "smote_rus"]


# ============================================================
# Per-Subject Analysis
# ============================================================
def compare_methods_per_subject(
    df: pd.DataFrame,
    metric: str,
    group_cols: List[str],
) -> pd.DataFrame:
    """Compare baseline vs each method for a given metric using per-subject scores."""
    results = []
    groups = df.groupby(group_cols)
    
    for group_key, group_df in groups:
        group_dict = dict(zip(group_cols, group_key if isinstance(group_key, tuple) else [group_key]))
        
        baseline_df = group_df[group_df["imbalance_method"] == BASELINE_METHOD]
        if baseline_df.empty:
            continue
        
        baseline_scores = baseline_df.set_index("subject")[metric]
        
        for method in COMPARISON_METHODS:
            method_df = group_df[group_df["imbalance_method"] == method]
            if method_df.empty:
                continue
            
            method_scores = method_df.set_index("subject")[metric]
            
            common_subjects = baseline_scores.index.intersection(method_scores.index)
            if len(common_subjects) < 5:
                continue
            
            b_vals = baseline_scores.loc[common_subjects].values
            m_vals = method_scores.loc[common_subjects].values
            
            stat, p_val = wilcoxon_test(b_vals, m_vals)
            d = cohens_d(m_vals, b_vals)
            
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


def run_per_subject_analysis():
    """Run per-subject statistical analysis."""
    print("\n=== Per-Subject Statistical Analysis ===")
    
    if not INPUT_FILE_SUBJECTS.exists():
        print(f"[ERROR] Input file not found: {INPUT_FILE_SUBJECTS}")
        print("[INFO] Run compute_subject_scores.py first!")
        return None
    
    df = pd.read_csv(INPUT_FILE_SUBJECTS)
    print(f"[INFO] Loaded {len(df)} subject-level records")
    
    all_results = []
    group_cols = ["ranking_method", "distance_metric", "level", "mode"]
    
    for metric in METRICS_SUBJECT:
        print(f"[INFO] Testing metric: {metric}")
        result = compare_methods_per_subject(df, metric, group_cols)
        all_results.append(result)
    
    results_df = pd.concat(all_results, ignore_index=True)
    print(f"\n[INFO] Completed {len(results_df)} statistical comparisons")
    
    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_csv = OUTPUT_DIR / "statistical_tests_per_subject.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"[INFO] Saved to: {output_csv}")
    
    # Generate report
    report = generate_per_subject_report(results_df)
    output_txt = OUTPUT_DIR / "statistical_tests_per_subject_summary.txt"
    output_txt.write_text(report)
    print(f"[INFO] Saved report to: {output_txt}")
    
    return results_df


def generate_per_subject_report(df: pd.DataFrame) -> str:
    """Generate per-subject analysis report."""
    lines = []
    lines.append("=" * 70)
    lines.append("Per-Subject Statistical Analysis Summary")
    lines.append("=" * 70)
    lines.append("")
    
    lines.append("Overall Comparison (across all configurations)")
    lines.append("-" * 50)
    
    for metric in METRICS_SUBJECT:
        lines.append(f"\n  Metric: {metric.upper()}")
        metric_df = df[df["metric"] == metric]
        
        for method in COMPARISON_METHODS:
            method_df = metric_df[metric_df["comparison_method"] == method]
            if method_df.empty:
                continue
            
            n_sig_05 = method_df["significant_05"].sum()
            n_total = len(method_df)
            mean_d = method_df["cohens_d"].mean()
            mean_diff = method_df["mean_diff"].mean()
            
            lines.append(f"    {BASELINE_METHOD} vs {method}:")
            lines.append(f"      Significant (p<0.05): {n_sig_05}/{n_total} ({100*n_sig_05/max(n_total, 1):.1f}%)")
            lines.append(f"      Mean difference: {mean_diff:+.4f}")
            lines.append(f"      Mean Cohen's d: {mean_d:.3f} ({interpret_cohens_d(mean_d)})")
    
    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


# ============================================================
# Aggregate Analysis
# ============================================================
def run_pairwise_tests_aggregate(
    df: pd.DataFrame, metric: str, mode: str, level_filter: str = None
) -> pd.DataFrame:
    """Run pairwise tests on aggregated metrics."""
    results = []
    
    mode_df = df[df["mode"] == mode].copy()
    if mode_df.empty:
        return pd.DataFrame()
    
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
        
        for config_key, grp in mode_df.groupby(config_cols):
            base = grp[grp["imbalance_method"] == BASELINE_METHOD][metric].values
            comp = grp[grp["imbalance_method"] == comp_method][metric].values
            
            if len(base) == 1 and len(comp) == 1:
                baseline_scores.append(base[0])
                method_scores.append(comp[0])
        
        if len(baseline_scores) < 3:
            continue
        
        b = np.array(baseline_scores)
        m = np.array(method_scores)
        
        w_stat, w_p = wilcoxon_test(b, m)
        t_stat, t_p = paired_ttest(b, m)
        d = cohens_d(m, b)
        
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
            "effect_size": interpret_cohens_d(d),
            "significant_05": w_p < 0.05 if not np.isnan(w_p) else False,
            "significant_01": w_p < 0.01 if not np.isnan(w_p) else False,
            "level_filter": level_filter if level_filter else "all",
        })
    
    return pd.DataFrame(results)


def run_aggregate_analysis():
    """Run aggregate metrics statistical analysis."""
    print("\n=== Aggregate Metrics Statistical Analysis ===")
    
    if not INPUT_FILE_AGGREGATE.exists():
        print(f"[ERROR] Input file not found: {INPUT_FILE_AGGREGATE}")
        return None
    
    df = pd.read_csv(INPUT_FILE_AGGREGATE)
    print(f"[INFO] Loaded {len(df)} records")
    
    df = df[df["level"] != "pooled"]
    print(f"[INFO] After excluding pooled: {len(df)} records")
    
    all_results = []
    levels = [None, "in_domain", "mid_domain", "out_domain"]
    
    for mode in ["target_only", "source_only"]:
        print(f"[INFO] Processing mode: {mode}")
        for level in levels:
            level_name = level if level else "all"
            for metric in METRICS_AGGREGATE:
                result = run_pairwise_tests_aggregate(df, metric, mode, level_filter=level)
                if not result.empty:
                    all_results.append(result)
    
    if not all_results:
        print("[ERROR] No test results generated!")
        return None
    
    results_df = pd.concat(all_results, ignore_index=True)
    print(f"\n[INFO] Generated {len(results_df)} test results")
    
    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_csv = OUTPUT_DIR / "statistical_tests_aggregate.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"[INFO] Saved to: {output_csv}")
    
    # Generate report
    report = generate_aggregate_report(results_df)
    output_txt = OUTPUT_DIR / "statistical_tests_aggregate_summary.txt"
    output_txt.write_text(report)
    print(f"[INFO] Saved report to: {output_txt}")
    
    return results_df


def generate_aggregate_report(df: pd.DataFrame) -> str:
    """Generate aggregate analysis report."""
    lines = []
    lines.append("=" * 70)
    lines.append("Aggregate Metrics Statistical Analysis Report")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Methodology:")
    lines.append("  - Paired comparison across experimental configurations")
    lines.append("  - Each (ranking × distance) or (ranking × distance × level) = 1 paired sample")
    lines.append("  - Tests: Wilcoxon signed-rank, paired t-test")
    lines.append("  - Effect size: Cohen's d")
    lines.append("")
    
    for mode in ["target_only", "source_only"]:
        lines.append("=" * 70)
        lines.append(f"Mode: {mode.upper()}")
        lines.append("=" * 70)
        
        mode_df = df[df["mode"] == mode]
        
        # All levels combined
        lines.append("\n  [ALL LEVELS COMBINED]")
        for metric in ["recall", "f1"]:  # Key metrics
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
                lines.append(f"      Wilcoxon p: {format_p_value(row['wilcoxon_p'])}")
                lines.append(f"      Cohen's d: {row['cohens_d']:.3f} ({row['effect_size']})")
    
    # Citation-ready summary
    lines.append("\n" + "=" * 70)
    lines.append("Citation-Ready Statements (target_only, recall)")
    lines.append("=" * 70)
    
    target_recall = df[(df["mode"] == "target_only") & (df["metric"] == "recall") & (df["level_filter"] == "all")]
    for _, row in target_recall.iterrows():
        sig = "significantly" if row["significant_05"] else "not significantly"
        lines.append(f"\n  {row['comparison'].upper()}:")
        lines.append(
            f'    "{row["comparison"].upper()} {sig} improved recall compared to baseline '
            f'(mean difference: {row["mean_diff"]:+.3f}, 95% CI [{row["diff_ci_lower"]:.3f}, {row["diff_ci_upper"]:.3f}], '
            f'Wilcoxon p = {row["wilcoxon_p"]:.4f}, Cohen\'s d = {row["cohens_d"]:.2f})."'
        )
    
    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Unified statistical tests for imbalance method comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "mode",
        choices=["per-subject", "aggregate", "all"],
        help="Analysis mode: per-subject, aggregate, or all"
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"STATISTICAL TESTS (mode={args.mode})")
    print("=" * 70)
    
    if args.mode == "per-subject":
        run_per_subject_analysis()
    elif args.mode == "aggregate":
        run_aggregate_analysis()
    elif args.mode == "all":
        run_per_subject_analysis()
        run_aggregate_analysis()
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
