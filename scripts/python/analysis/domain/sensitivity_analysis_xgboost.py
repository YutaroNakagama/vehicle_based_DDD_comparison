#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sensitivity_analysis_xgboost.py
================================
Sobol-Hoeffding variance decomposition for XGBoost validation experiment.
Computes the same indices as sensitivity_analysis_exp2.py and generates
a comparison table with RF results.

This validates that the Sobol hierarchy (S_TM > S_TR >> S_TD ≈ S_TG)
is classifier-independent (bagging vs boosting).

Usage:
    python scripts/python/analysis/domain/sensitivity_analysis_xgboost.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the RF analysis functions
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from sensitivity_analysis_exp2 import (
    _encode_factors,
    compute_ss_decomposition,
    compute_sobol_indices,
    bootstrap_sobol,
    FACTORS,
    FACTOR_NAMES,
    PRIMARY_METRICS,
    OFFICIAL_SEEDS,
)

REPORT_DIR = PROJECT_ROOT / "results" / "analysis" / "exp2_domain_shift"
CSV_OUT = REPORT_DIR / "figures" / "csv" / "split2" / "sensitivity"
CSV_OUT.mkdir(parents=True, exist_ok=True)

XGB_CSV_BASE = REPORT_DIR / "figures" / "csv" / "split2" / "xgboost"
RF_CSV_BASE = REPORT_DIR / "figures" / "csv" / "split2"

CONDITIONS_7 = [
    "baseline", "rus_r01", "rus_r05",
    "smote_r01", "smote_r05", "sw_smote_r01", "sw_smote_r05",
]


def load_xgboost_data() -> pd.DataFrame:
    """Load XGBoost evaluation CSVs."""
    files = {
        "baseline": XGB_CSV_BASE / "baseline" / "xgb_baseline_domain_split2_metrics_v2.csv",
        "smote": XGB_CSV_BASE / "smote_plain" / "xgb_smote_plain_split2_metrics_v2.csv",
        "rus": XGB_CSV_BASE / "undersample_rus" / "xgb_undersample_rus_split2_metrics_v2.csv",
        "sw_smote": XGB_CSV_BASE / "sw_smote" / "xgb_sw_smote_split2_metrics_v2.csv",
    }
    dfs = []
    for method, path in files.items():
        if not path.exists():
            print(f"  [WARN] Missing: {path.relative_to(PROJECT_ROOT)}")
            continue
        df = pd.read_csv(path)
        if method == "baseline":
            df["condition"] = "baseline"
        else:
            df["condition"] = df["ratio"].apply(
                lambda r: f"{method}_r{str(r).replace('.', '')}"
                if pd.notna(r) else method
            )
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError("No XGBoost CSV files found. Run collect_split2_xgboost_metrics.py first.")

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged[merged["condition"].isin(CONDITIONS_7)].copy()
    merged = merged[merged["seed"].isin(OFFICIAL_SEEDS)].copy()
    print(f"Loaded {len(merged)} XGBoost records")
    return merged


def load_rf_sobol() -> pd.DataFrame | None:
    """Load existing RF Sobol indices for comparison."""
    path = CSV_OUT / "sobol_indices.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def generate_comparison_report(xgb_results: dict, rf_sobol: pd.DataFrame | None) -> str:
    """Generate markdown report comparing XGBoost and RF Sobol indices."""
    lines = []
    w = lines.append

    w("# Classifier-Independence Validation — Sobol Analysis\n")
    w("## XGBoost vs Random Forest\n")
    w("This report validates that the Sobol-Hoeffding variance decomposition")
    w("results are consistent across two architecturally different classifiers:\n")
    w("- **Random Forest (RF)**: Bagging + random feature subsets (variance reduction)")
    w("- **XGBoost**: Gradient boosting + sequential residual correction (bias reduction)\n")

    for metric, mlabel in PRIMARY_METRICS:
        w(f"\n### {mlabel}\n")
        res = xgb_results[metric]

        # Build comparison table
        header = "| Factor | XGBoost $S_i$ | XGBoost $S_{Ti}$ |"

        if rf_sobol is not None:
            header += " RF $S_i$ | RF $S_{Ti}$ | $\\Delta S_{Ti}$ |"
            w(header)
            w("|--------|-------------|-----------------|---------|-------------|-----------------|")
        else:
            w(header)
            w("|--------|-------------|-----------------|")

        for f in FACTORS:
            s1_val, s1_lo, s1_hi = res[f"S1_{f}"]
            st_val, st_lo, st_hi = res[f"ST_{f}"]

            row = f"| {FACTOR_NAMES[f]} | {s1_val:.4f} [{s1_lo:.4f},{s1_hi:.4f}] | {st_val:.4f} [{st_lo:.4f},{st_hi:.4f}] |"

            if rf_sobol is not None:
                rf_row = rf_sobol[(rf_sobol["metric"] == mlabel) &
                                  (rf_sobol["factor_key"] == f)]
                if not rf_row.empty:
                    rf_s1 = rf_row["S1"].values[0]
                    rf_st = rf_row["ST"].values[0]
                    delta = st_val - rf_st
                    row += f" {rf_s1:.4f} | {rf_st:.4f} | {delta:+.4f} |"
                else:
                    row += " — | — | — |"

            w(row)

        s_res, _, _ = res["S_residual"]
        w(f"\nResidual (seed variation): {s_res:.4f} ({s_res*100:.1f}%)\n")

        # Hierarchy check
        st_values = {f: res[f"ST_{f}"][0] for f in FACTORS}
        sorted_factors = sorted(st_values.items(), key=lambda x: -x[1])
        hierarchy = " > ".join(
            f"{FACTOR_NAMES[f]} ({v:.3f})" for f, v in sorted_factors
        )
        w(f"**XGBoost hierarchy:** {hierarchy}\n")

    # Summary
    w("\n## Conclusion\n")
    w("If the factor ranking (Mode > Rebalancing >> Distance ≈ Membership)")
    w("is consistent between RF and XGBoost, the Sobol sensitivity analysis")
    w("results reflect the inherent data/problem structure rather than")
    w("classifier-specific artifacts.\n")

    w("\n---\n")
    w("*Generated by `scripts/python/analysis/domain/sensitivity_analysis_xgboost.py`*\n")
    return "\n".join(lines)


def export_xgboost_sobol(all_results: dict):
    """Export XGBoost Sobol indices to CSV."""
    rows = []
    for metric, mlabel in PRIMARY_METRICS:
        res = all_results[metric]
        for f in FACTORS:
            s1_val, s1_lo, s1_hi = res[f"S1_{f}"]
            st_val, st_lo, st_hi = res[f"ST_{f}"]
            rows.append({
                "model": "XGBoost",
                "metric": mlabel,
                "factor": FACTOR_NAMES[f],
                "factor_key": f,
                "S1": s1_val, "S1_lo": s1_lo, "S1_hi": s1_hi,
                "ST": st_val, "ST_lo": st_lo, "ST_hi": st_hi,
            })

    df_out = pd.DataFrame(rows)
    out_path = CSV_OUT / "sobol_indices_xgboost.csv"
    df_out.to_csv(out_path, index=False)
    print(f"Saved: {out_path.relative_to(PROJECT_ROOT)}")

    # Combined RF + XGBoost CSV
    rf_path = CSV_OUT / "sobol_indices.csv"
    if rf_path.exists():
        df_rf = pd.read_csv(rf_path)
        df_rf["model"] = "RF"
        df_combined = pd.concat([df_rf, df_out], ignore_index=True)
        combined_path = CSV_OUT / "sobol_indices_combined.csv"
        df_combined.to_csv(combined_path, index=False)
        print(f"Saved: {combined_path.relative_to(PROJECT_ROOT)}")


def main():
    print("=" * 60)
    print("Sensitivity Analysis — XGBoost Validation")
    print("=" * 60)

    df = load_xgboost_data()

    # Run Sobol analysis
    all_results = {}
    for metric, mlabel in PRIMARY_METRICS:
        print(f"\n--- {mlabel} ---")
        result = bootstrap_sobol(df, metric, B=2000)
        all_results[metric] = result

        print(f"  First-order indices:")
        for f in FACTORS:
            s1, _, _ = result[f"S1_{f}"]
            st, _, _ = result[f"ST_{f}"]
            print(f"    {FACTOR_NAMES[f]:25s}  S1={s1:.4f}  ST={st:.4f}")
        s_res, _, _ = result["S_residual"]
        print(f"    {'Residual (seed)':25s}  {s_res:.4f}")

    # Load RF results for comparison
    rf_sobol = load_rf_sobol()

    # Generate report
    report = generate_comparison_report(all_results, rf_sobol)
    report_path = REPORT_DIR / "sensitivity_analysis_xgboost_validation.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\nReport: {report_path.relative_to(PROJECT_ROOT)}")

    # Export CSVs
    export_xgboost_sobol(all_results)

    print("\nDone.")


if __name__ == "__main__":
    main()
