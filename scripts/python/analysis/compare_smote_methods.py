#!/usr/bin/env python3
"""
Compare SMOTE Methods Performance

Generates comparison tables and statistical tests for SMOTE experiments:
1. Pooled mode: Subject-wise SMOTE vs Simple SMOTE vs SMOTE+BalancedRF
2. Ranking mode: KNN vs LOF rankings × SMOTE methods

Usage:
    python scripts/python/analysis/compare_smote_methods.py \
        --input results/analysis/exp1_imbalance/smote_comparison/aggregated_results.csv \
        --output-imbalance results/analysis/exp1_imbalance/smote_comparison/ \
        --output-domain results/analysis/exp1_imbalance/smote_comparison/
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from scipy import stats

# Setup path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# Key metrics for comparison
KEY_METRICS = [
    "test_f1",
    "test_accuracy",
    "test_precision",
    "test_recall",
    "test_auc",
    "test_balanced_accuracy",
]


def load_results(input_path: str) -> pd.DataFrame:
    """Load aggregated results CSV."""
    df = pd.read_csv(input_path)
    logging.info(f"Loaded {len(df)} records from {input_path}")
    return df


def create_pooled_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create comparison table for pooled mode experiments.
    
    Compares:
    - Subject-wise SMOTE (RF)
    - Simple SMOTE (RF)
    - SMOTE + BalancedRF
    """
    pooled = df[df["mode"] == "pooled"].copy()
    
    if pooled.empty:
        logging.warning("No pooled mode experiments found")
        return pd.DataFrame()
    
    # Group by SMOTE method and aggregate metrics
    agg_funcs = {metric: ["mean", "std", "count"] for metric in KEY_METRICS if metric in pooled.columns}
    
    summary = pooled.groupby("smote_method").agg(agg_funcs)
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    
    # Flatten and format
    result_rows = []
    for method in summary.index:
        row = {"smote_method": method}
        for metric in KEY_METRICS:
            if f"{metric}_mean" in summary.columns:
                mean_val = summary.loc[method, f"{metric}_mean"]
                std_val = summary.loc[method, f"{metric}_std"]
                count = summary.loc[method, f"{metric}_count"]
                row[metric] = f"{mean_val:.4f} ± {std_val:.4f}"
                row[f"{metric}_mean"] = mean_val
                row[f"{metric}_std"] = std_val
                row[f"{metric}_n"] = count
        result_rows.append(row)
    
    return pd.DataFrame(result_rows)


def create_ranking_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create comparison table for ranking-based experiments.
    
    Dimensions:
    - Ranking method: knn, lof
    - Domain level: out_domain, in_domain
    - Mode: source_only, target_only
    - SMOTE method: subject_wise_smote, simple_smote, smote_balanced_rf
    """
    ranking_df = df[df["ranking"].isin(["knn", "lof"])].copy()
    
    if ranking_df.empty:
        logging.warning("No ranking-based experiments found")
        return pd.DataFrame()
    
    # Create pivot table for each metric
    result_rows = []
    
    for (ranking, domain, mode), group in ranking_df.groupby(["ranking", "domain_level", "mode"]):
        row = {
            "ranking": ranking,
            "domain_level": domain,
            "mode": mode,
        }
        
        for smote_method in group["smote_method"].unique():
            subset = group[group["smote_method"] == smote_method]
            for metric in KEY_METRICS:
                if metric in subset.columns:
                    mean_val = subset[metric].mean()
                    std_val = subset[metric].std()
                    key = f"{smote_method}_{metric}"
                    row[key] = mean_val
                    row[f"{key}_std"] = std_val
        
        result_rows.append(row)
    
    return pd.DataFrame(result_rows)


def perform_statistical_tests(df: pd.DataFrame, metric: str = "test_f1") -> pd.DataFrame:
    """Perform statistical tests comparing SMOTE methods.
    
    Uses:
    - Wilcoxon signed-rank test for pairwise comparisons
    - Friedman test for overall comparison
    """
    results = []
    
    # Get unique SMOTE methods
    methods = df["smote_method"].unique()
    
    if len(methods) < 2:
        logging.warning("Need at least 2 methods for statistical comparison")
        return pd.DataFrame()
    
    # Pairwise Wilcoxon tests
    for i, method1 in enumerate(methods):
        for method2 in methods[i+1:]:
            data1 = df[df["smote_method"] == method1][metric].dropna().values
            data2 = df[df["smote_method"] == method2][metric].dropna().values
            
            # Match sample sizes (use minimum)
            n = min(len(data1), len(data2))
            if n < 3:
                logging.warning(f"Insufficient samples for {method1} vs {method2}")
                continue
            
            data1 = data1[:n]
            data2 = data2[:n]
            
            try:
                stat, pvalue = stats.wilcoxon(data1, data2, alternative="two-sided")
                results.append({
                    "test": "wilcoxon",
                    "method1": method1,
                    "method2": method2,
                    "metric": metric,
                    "statistic": stat,
                    "p_value": pvalue,
                    "significant_005": pvalue < 0.05,
                    "significant_001": pvalue < 0.01,
                    "n_samples": n,
                    "method1_mean": data1.mean(),
                    "method2_mean": data2.mean(),
                    "effect_direction": "method1 > method2" if data1.mean() > data2.mean() else "method2 > method1",
                })
            except Exception as e:
                logging.warning(f"Wilcoxon test failed for {method1} vs {method2}: {e}")
    
    # Friedman test (if 3+ methods)
    if len(methods) >= 3:
        try:
            # Align data by seed/experiment
            pivot = df.pivot_table(
                index=["mode", "ranking", "domain_level", "seed"],
                columns="smote_method",
                values=metric,
                aggfunc="first",
            ).dropna()
            
            if len(pivot) >= 3:
                stat, pvalue = stats.friedmanchisquare(
                    *[pivot[m].values for m in pivot.columns]
                )
                results.append({
                    "test": "friedman",
                    "method1": "all",
                    "method2": "all",
                    "metric": metric,
                    "statistic": stat,
                    "p_value": pvalue,
                    "significant_005": pvalue < 0.05,
                    "significant_001": pvalue < 0.01,
                    "n_samples": len(pivot),
                    "method1_mean": None,
                    "method2_mean": None,
                    "effect_direction": "see pairwise",
                })
        except Exception as e:
            logging.warning(f"Friedman test failed: {e}")
    
    return pd.DataFrame(results)


def generate_latex_table(comparison_df: pd.DataFrame, caption: str = "") -> str:
    """Generate LaTeX table from comparison DataFrame."""
    if comparison_df.empty:
        return ""
    
    # Select display columns
    display_cols = ["smote_method"] + [m for m in KEY_METRICS if m in comparison_df.columns]
    
    latex = comparison_df[display_cols].to_latex(
        index=False,
        caption=caption,
        label="tab:smote_comparison",
        escape=False,
        column_format="l" + "c" * (len(display_cols) - 1),
    )
    
    return latex


def main():
    parser = argparse.ArgumentParser(description="Compare SMOTE methods performance")
    parser.add_argument(
        "--input", "-i",
        default="results/analysis/exp1_imbalance/smote_comparison/aggregated_results.csv",
        help="Input CSV path (from aggregate_smote_results.py)",
    )
    parser.add_argument(
        "--output-imbalance",
        default="results/analysis/exp1_imbalance/smote_comparison/",
        help="Output directory for imbalance-only (pooled) results",
    )
    parser.add_argument(
        "--output-domain",
        default="results/analysis/exp1_imbalance/smote_comparison/",
        help="Output directory for domain (ranking-based) results",
    )
    parser.add_argument(
        "--metric", "-m",
        default="test_f1",
        help="Primary metric for statistical tests",
    )
    args = parser.parse_args()
    
    # Load data
    input_path = PROJECT_ROOT / args.input
    if not input_path.exists():
        logging.error(f"Input file not found: {input_path}")
        logging.info("Run aggregate_smote_results.py first to collect results.")
        sys.exit(1)
    
    df = load_results(input_path)
    
    # Create output directories
    output_imbalance = PROJECT_ROOT / args.output_imbalance
    output_domain = PROJECT_ROOT / args.output_domain
    output_imbalance.mkdir(parents=True, exist_ok=True)
    output_domain.mkdir(parents=True, exist_ok=True)
    
    # Generate comparison tables
    print("\n" + "=" * 70)
    print("SMOTE Methods Comparison Analysis")
    print("=" * 70)
    
    # 1. Pooled mode comparison -> imbalance directory
    print("\n[1] Pooled Mode Comparison (Imbalance-only)")
    print("-" * 50)
    pooled_table = create_pooled_comparison_table(df)
    if not pooled_table.empty:
        print(pooled_table.to_string(index=False))
        pooled_table.to_csv(output_imbalance / "pooled_comparison.csv", index=False)
        logging.info(f"Saved: {output_imbalance / 'pooled_comparison.csv'}")
    
    # 2. Ranking mode comparison -> domain directory
    print("\n[2] Ranking-based Comparison (Domain Analysis)")
    print("-" * 50)
    ranking_table = create_ranking_comparison_table(df)
    if not ranking_table.empty:
        print(ranking_table.to_string(index=False))
        ranking_table.to_csv(output_domain / "ranking_comparison.csv", index=False)
        logging.info(f"Saved: {output_domain / 'ranking_comparison.csv'}")
    
    # 3. Statistical tests -> both directories
    print(f"\n[3] Statistical Tests (metric: {args.metric})")
    print("-" * 50)
    stats_table = perform_statistical_tests(df, metric=args.metric)
    if not stats_table.empty:
        print(stats_table.to_string(index=False))
        stats_table.to_csv(output_imbalance / "statistical_tests.csv", index=False)
        stats_table.to_csv(output_domain / "statistical_tests.csv", index=False)
        logging.info(f"Saved: statistical_tests.csv to both directories")
    
    # 4. Summary by all dimensions
    print("\n[4] Overall Summary")
    print("-" * 50)
    summary_cols = ["smote_method", "mode", "ranking", "domain_level"]
    available_cols = [c for c in summary_cols if c in df.columns]
    if args.metric in df.columns:
        summary = df.groupby(available_cols)[args.metric].agg(["mean", "std", "count"])
        print(summary.to_string())
        # Save pooled summary to imbalance, ranking summary to domain
        pooled_summary = summary.loc[summary.index.get_level_values("mode") == "pooled"] if "mode" in available_cols else summary
        domain_summary = summary.loc[summary.index.get_level_values("mode") != "pooled"] if "mode" in available_cols else pd.DataFrame()
        if not pooled_summary.empty:
            pooled_summary.to_csv(output_imbalance / "overall_summary.csv")
        if not domain_summary.empty:
            domain_summary.to_csv(output_domain / "overall_summary.csv")
        logging.info(f"Saved: overall_summary.csv to respective directories")
    
    # 5. Generate LaTeX table
    if not pooled_table.empty:
        latex = generate_latex_table(pooled_table, caption="SMOTE Methods Comparison (Pooled Mode)")
        with open(output_imbalance / "pooled_comparison.tex", "w") as f:
            f.write(latex)
        logging.info(f"Saved: {output_imbalance / 'pooled_comparison.tex'}")
    
    print("\n" + "=" * 70)
    print(f"Imbalance results saved to: {output_imbalance}")
    print(f"Domain results saved to:    {output_domain}")
    print("=" * 70)


if __name__ == "__main__":
    main()
