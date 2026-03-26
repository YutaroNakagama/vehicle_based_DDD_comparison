#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sensitivity_analysis_exp2.py
============================
Variance-based sensitivity analysis (functional ANOVA decomposition) for
Experiment 2's four-factor factorial design.

Computes first-order and total-order Sobol indices from the balanced
factorial design (7 R × 3 D × 2 G × 3 M × 12 seeds = 1,512 observations).

For a balanced factorial design, the Sobol indices can be computed
analytically via ANOVA-style sum-of-squares decomposition:

    S_i       = SS_i / SS_total           (first-order index)
    S_{ij}    = SS_{ij} / SS_total        (second-order interaction index)
    S_{Ti}    = S_i + Σ S_{ij} + ...      (total-order index)

Output:
    results/analysis/exp2_domain_shift/sensitivity_analysis_report.md
    CSV tables for plotting
"""

from __future__ import annotations

import sys
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

REPORT_DIR = PROJECT_ROOT / "results" / "analysis" / "exp2_domain_shift"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CSV_OUT = REPORT_DIR / "figures" / "csv" / "split2" / "sensitivity"
CSV_OUT.mkdir(parents=True, exist_ok=True)
CSV_BASE = REPORT_DIR / "figures" / "csv" / "split2"

# ---------------------------------------------------------------------------
OFFICIAL_SEEDS = {0, 1, 3, 7, 13, 42, 123, 256, 512, 999, 1337, 2024}
CONDITIONS_7 = [
    "baseline", "rus_r01", "rus_r05",
    "smote_r01", "smote_r05", "sw_smote_r01", "sw_smote_r05",
]
MODES = ["source_only", "target_only", "mixed"]
DISTANCES = ["mmd", "dtw", "wasserstein"]
LEVELS = ["in_domain", "out_domain"]
PRIMARY_METRICS = [("f2", "F2-score"), ("auc", "AUROC"), ("auc_pr", "AUPRC")]

FACTOR_NAMES = {
    "condition": "Rebalancing ($R$)",
    "distance": "Distance ($D$)",
    "level": "Membership ($G$)",
    "mode": "Mode ($M$)",
}
FACTORS = ["condition", "distance", "level", "mode"]


# ---------------------------------------------------------------------------
# Data loading (same as other scripts)
# ---------------------------------------------------------------------------
def load_all_data() -> pd.DataFrame:
    files = {
        "baseline": CSV_BASE / "baseline" / "baseline_domain_split2_metrics_v2.csv",
        "smote": CSV_BASE / "smote_plain" / "smote_plain_split2_metrics_v2.csv",
        "rus": CSV_BASE / "undersample_rus" / "undersample_rus_split2_metrics_v2.csv",
        "sw_smote": CSV_BASE / "sw_smote" / "sw_smote_split2_metrics_v2.csv",
    }
    dfs = []
    for method, path in files.items():
        df = pd.read_csv(path)
        if method == "baseline":
            df["condition"] = "baseline"
        else:
            df["condition"] = df["ratio"].apply(
                lambda r: f"{method}_r{str(r).replace('.', '')}"
                if pd.notna(r) else method
            )
        dfs.append(df)
    merged = pd.concat(dfs, ignore_index=True)
    merged = merged[merged["condition"].isin(CONDITIONS_7)].copy()
    merged = merged[merged["seed"].isin(OFFICIAL_SEEDS)].copy()
    print(f"Loaded {len(merged)} records")
    return merged


# ---------------------------------------------------------------------------
# Vectorised ANOVA-based variance decomposition
# ---------------------------------------------------------------------------
def _encode_factors(df: pd.DataFrame) -> tuple:
    """Encode factor columns as integer codes for fast numpy indexing.
    Returns (y_array_dict, code_arrays, level_counts).
    """
    codes = {}
    level_counts = {}
    for f in FACTORS:
        cat = pd.Categorical(df[f])
        codes[f] = np.asarray(cat.codes)
        level_counts[f] = len(cat.categories)
    return codes, level_counts


def compute_ss_decomposition(y: np.ndarray, codes: dict,
                             level_counts: dict) -> dict:
    """Compute SS decomposition using numpy groupby via bincount.

    Parameters
    ----------
    y : 1-D array of metric values
    codes : dict mapping factor name -> integer code array
    level_counts : dict mapping factor name -> number of levels
    """
    N = len(y)
    grand_mean = y.mean()
    ss_total = np.sum((y - grand_mean) ** 2)
    results = {"SS_total": ss_total}

    # Precompute all marginal means (1-factor)
    marginal_1 = {}
    for f in FACTORS:
        k = level_counts[f]
        sums = np.bincount(codes[f], weights=y, minlength=k)
        counts = np.bincount(codes[f], minlength=k).astype(float)
        counts[counts == 0] = 1  # avoid div/0
        marginal_1[f] = sums / counts  # mean per level

    # Main effects
    for f in FACTORS:
        k = level_counts[f]
        counts = np.bincount(codes[f], minlength=k).astype(float)
        means = marginal_1[f]
        results[f"SS_{f}"] = np.sum(counts * (means - grand_mean) ** 2)

    # Precompute 2-factor cell means
    marginal_2 = {}
    for f1, f2 in combinations(FACTORS, 2):
        k1, k2 = level_counts[f1], level_counts[f2]
        combined = codes[f1] * k2 + codes[f2]
        total_cells = k1 * k2
        sums = np.bincount(combined, weights=y, minlength=total_cells)
        counts = np.bincount(combined, minlength=total_cells).astype(float)
        counts[counts == 0] = 1
        means_2d = (sums / counts).reshape(k1, k2)
        counts_2d = np.bincount(combined, minlength=total_cells).astype(float).reshape(k1, k2)
        marginal_2[(f1, f2)] = (means_2d, counts_2d)

    # Two-way interactions
    for f1, f2 in combinations(FACTORS, 2):
        means_2d, counts_2d = marginal_2[(f1, f2)]
        k1, k2 = level_counts[f1], level_counts[f2]
        m1 = marginal_1[f1]  # shape (k1,)
        m2 = marginal_1[f2]  # shape (k2,)
        # interaction_effect[i,j] = cell_mean[i,j] - m1[i] - m2[j] + grand_mean
        interaction = means_2d - m1[:, None] - m2[None, :] + grand_mean
        results[f"SS_{f1}x{f2}"] = np.sum(counts_2d * interaction ** 2)

    # Precompute 3-factor cell means
    marginal_3 = {}
    for f1, f2, f3 in combinations(FACTORS, 3):
        k1, k2, k3 = level_counts[f1], level_counts[f2], level_counts[f3]
        combined = (codes[f1] * k2 + codes[f2]) * k3 + codes[f3]
        total_cells = k1 * k2 * k3
        sums = np.bincount(combined, weights=y, minlength=total_cells)
        counts = np.bincount(combined, minlength=total_cells).astype(float)
        counts[counts == 0] = 1
        means_3d = (sums / counts).reshape(k1, k2, k3)
        counts_3d = np.bincount(combined, minlength=total_cells).astype(float).reshape(k1, k2, k3)
        marginal_3[(f1, f2, f3)] = (means_3d, counts_3d)

    # Three-way interactions
    for f1, f2, f3 in combinations(FACTORS, 3):
        means_3d, counts_3d = marginal_3[(f1, f2, f3)]
        k1, k2, k3 = level_counts[f1], level_counts[f2], level_counts[f3]
        m1 = marginal_1[f1]
        m2 = marginal_1[f2]
        m3 = marginal_1[f3]
        m12, _ = marginal_2[(f1, f2)]
        # Need to get other 2-factor combos in the right order
        if (f1, f3) in marginal_2:
            m13, _ = marginal_2[(f1, f3)]
        else:
            m13, _ = marginal_2[(f3, f1)]
            m13 = m13.T
        if (f2, f3) in marginal_2:
            m23, _ = marginal_2[(f2, f3)]
        else:
            m23, _ = marginal_2[(f3, f2)]
            m23 = m23.T
        # 3-way interaction = cell - all 2-way - all main + 2*grand_mean
        effect = (means_3d
                  - m12[:, :, None] - m13[:, None, :] - m23[None, :, :]
                  + m1[:, None, None] + m2[None, :, None] + m3[None, None, :]
                  - grand_mean)
        results[f"SS_{f1}x{f2}x{f3}"] = np.sum(counts_3d * effect ** 2)

    # Four-way interaction
    f1, f2, f3, f4 = FACTORS
    k1, k2, k3, k4 = [level_counts[f] for f in FACTORS]
    combined_4 = ((codes[f1] * k2 + codes[f2]) * k3 + codes[f3]) * k4 + codes[f4]
    total_cells_4 = k1 * k2 * k3 * k4
    sums_4 = np.bincount(combined_4, weights=y, minlength=total_cells_4)
    counts_4 = np.bincount(combined_4, minlength=total_cells_4).astype(float)
    counts_4[counts_4 == 0] = 1
    means_4d = (sums_4 / counts_4).reshape(k1, k2, k3, k4)
    counts_4d = np.bincount(combined_4, minlength=total_cells_4).astype(float).reshape(k1, k2, k3, k4)

    # Get all marginals needed
    m_1 = marginal_1[f1]
    m_2 = marginal_1[f2]
    m_3 = marginal_1[f3]
    m_4 = marginal_1[f4]

    def get_m2(a, b):
        if (a, b) in marginal_2:
            return marginal_2[(a, b)][0]
        return marginal_2[(b, a)][0].T

    m_12 = get_m2(f1, f2)
    m_13 = get_m2(f1, f3)
    m_14 = get_m2(f1, f4)
    m_23 = get_m2(f2, f3)
    m_24 = get_m2(f2, f4)
    m_34 = get_m2(f3, f4)

    def get_m3(a, b, c):
        if (a, b, c) in marginal_3:
            return marginal_3[(a, b, c)][0]
        # All factors are in order, so this should always match
        raise KeyError(f"Missing 3-way marginal for {a}, {b}, {c}")

    m_123 = get_m3(f1, f2, f3)
    m_124 = get_m3(f1, f2, f4)
    m_134 = get_m3(f1, f3, f4)
    m_234 = get_m3(f2, f3, f4)

    effect_4 = (means_4d
                - m_123[:, :, :, None] - m_124[:, :, None, :]
                - m_134[:, None, :, :] - m_234[None, :, :, :]
                + m_12[:, :, None, None] + m_13[:, None, :, None]
                + m_14[:, None, None, :] + m_23[None, :, :, None]
                + m_24[None, :, None, :] + m_34[None, None, :, :]
                - m_1[:, None, None, None] - m_2[None, :, None, None]
                - m_3[None, None, :, None] - m_4[None, None, None, :]
                + grand_mean)
    results[f"SS_{f1}x{f2}x{f3}x{f4}"] = np.sum(counts_4d * effect_4 ** 2)

    # Residual
    ss_model = sum(v for k, v in results.items()
                   if k.startswith("SS_") and k != "SS_total")
    results["SS_residual"] = ss_total - ss_model

    return results


def compute_sobol_indices(ss: dict) -> dict:
    """Convert SS decomposition to first-order and total-order Sobol indices."""
    ss_total = ss["SS_total"]
    if ss_total == 0:
        return {}

    indices = {}

    # First-order indices
    for f in FACTORS:
        indices[f"S1_{f}"] = ss[f"SS_{f}"] / ss_total

    # Second-order interaction indices
    for f1, f2 in combinations(FACTORS, 2):
        indices[f"S2_{f1}x{f2}"] = ss[f"SS_{f1}x{f2}"] / ss_total

    # Third-order
    for f1, f2, f3 in combinations(FACTORS, 3):
        indices[f"S3_{f1}x{f2}x{f3}"] = ss[f"SS_{f1}x{f2}x{f3}"] / ss_total

    # Fourth-order
    f_all = "x".join(FACTORS)
    indices[f"S4_{f_all}"] = ss[f"SS_{f_all}"] / ss_total

    # Residual
    indices["S_residual"] = ss["SS_residual"] / ss_total

    # Total-order indices: S_Ti = S_i + all interactions involving i
    for f in FACTORS:
        st = indices[f"S1_{f}"]
        for f1, f2 in combinations(FACTORS, 2):
            if f in (f1, f2):
                st += indices[f"S2_{f1}x{f2}"]
        for f1, f2, f3 in combinations(FACTORS, 3):
            if f in (f1, f2, f3):
                st += indices[f"S3_{f1}x{f2}x{f3}"]
        st += indices[f"S4_{f_all}"]
        indices[f"ST_{f}"] = st

    return indices


def bootstrap_sobol(df: pd.DataFrame, metric: str,
                    B: int = 2000, rng_seed: int = 42) -> dict:
    """Bootstrap confidence intervals for Sobol indices.

    Resamples seeds (the blocking factor) to preserve the factorial structure.
    Uses vectorised SS decomposition for speed.
    Returns {index_name: (point_est, ci_lo, ci_hi)}.
    """
    rng = np.random.RandomState(rng_seed)
    seeds = sorted(df["seed"].unique())
    n_seeds = len(seeds)

    # Encode factors once
    codes_orig, level_counts = _encode_factors(df)

    # Point estimates
    y_point = df[metric].values
    ss_point = compute_ss_decomposition(y_point, codes_orig, level_counts)
    indices_point = compute_sobol_indices(ss_point)

    # Pre-group rows by seed for fast bootstrap assembly
    seed_groups = {}
    for s in seeds:
        mask = df["seed"] == s
        idxs = np.where(mask)[0]
        seed_groups[s] = idxs

    # Pre-extract all arrays
    y_all = df[metric].values
    code_arrays = {f: codes_orig[f] for f in FACTORS}

    # Bootstrap
    boot_indices = {k: np.empty(B) for k in indices_point}
    for b in range(B):
        # Resample seeds with replacement
        boot_seeds = rng.choice(seeds, size=n_seeds, replace=True)
        # Gather row indices
        row_idxs = np.concatenate([seed_groups[s] for s in boot_seeds])
        y_boot = y_all[row_idxs]
        codes_boot = {f: code_arrays[f][row_idxs] for f in FACTORS}

        ss_boot = compute_ss_decomposition(y_boot, codes_boot, level_counts)
        idx_boot = compute_sobol_indices(ss_boot)
        for k in boot_indices:
            boot_indices[k][b] = idx_boot.get(k, 0.0)

    # Percentile CIs
    result = {}
    for k, point_val in indices_point.items():
        boots = boot_indices[k]
        ci_lo = np.percentile(boots, 2.5)
        ci_hi = np.percentile(boots, 97.5)
        result[k] = (point_val, ci_lo, ci_hi)

    return result


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_report(all_results: dict) -> str:
    """Generate markdown report from sensitivity analysis results."""
    lines = []
    w = lines.append

    w("# Sensitivity Analysis Report — Experiment 2\n")
    w("## Variance-Based Sensitivity Analysis (Functional ANOVA Decomposition)\n")
    w("This analysis decomposes the total variance of each performance metric into")
    w("contributions from the four experimental factors and their interactions.\n")
    w("- **First-order index ($S_i$)**: fraction of variance explained by factor $i$ alone")
    w("- **Total-order index ($S_{Ti}$)**: fraction explained by $i$ including all interactions")
    w("- **$S_{Ti} - S_i$**: variance due to interactions involving factor $i$")
    w(f"- Bootstrap CIs: 95% percentile, $B = 2000$, resampling over seeds\n")

    for metric, mlabel in PRIMARY_METRICS:
        w(f"\n## {mlabel}\n")
        res = all_results[metric]

        # Main effects table
        w("### First-Order and Total-Order Indices\n")
        w("| Factor | $S_i$ | 95% CI | $S_{Ti}$ | 95% CI | Interaction ($S_{Ti}-S_i$) |")
        w("|--------|-------|--------|----------|--------|---------------------------|")
        for f in FACTORS:
            s1_val, s1_lo, s1_hi = res[f"S1_{f}"]
            st_val, st_lo, st_hi = res[f"ST_{f}"]
            inter = st_val - s1_val
            w(f"| {FACTOR_NAMES[f]} | {s1_val:.4f} | [{s1_lo:.4f}, {s1_hi:.4f}] "
              f"| {st_val:.4f} | [{st_lo:.4f}, {st_hi:.4f}] | {inter:.4f} |")

        # Residual
        s_res, _, _ = res["S_residual"]
        w(f"\nResidual (seed variation): {s_res:.4f} ({s_res*100:.1f}%)\n")

        # Full decomposition table
        w("### Complete Variance Decomposition\n")
        w("| Effect | $S$ | % of total |")
        w("|--------|-----|-----------|")

        # Collect all effects sorted by magnitude
        effects = []
        for k, (val, _, _) in res.items():
            if k.startswith("S1_") or k.startswith("S2_") or k.startswith("S3_") or k.startswith("S4_"):
                # Pretty name
                if k.startswith("S1_"):
                    name = FACTOR_NAMES[k[3:]]
                elif k.startswith("S2_"):
                    parts = k[3:].split("x")
                    name = " × ".join(FACTOR_NAMES[p] for p in parts)
                elif k.startswith("S3_"):
                    parts = k[3:].split("x")
                    name = " × ".join(FACTOR_NAMES[p] for p in parts)
                else:
                    parts = k[3:].split("x")
                    name = " × ".join(FACTOR_NAMES[p] for p in parts)
                effects.append((name, val))

        effects.sort(key=lambda x: -x[1])
        for name, val in effects:
            w(f"| {name} | {val:.4f} | {val*100:.1f}% |")
        w(f"| Residual (seed) | {s_res:.4f} | {s_res*100:.1f}% |")
        total_check = sum(v for _, v in effects) + s_res
        w(f"| **Total** | **{total_check:.4f}** | **{total_check*100:.1f}%** |")

    w("\n---\n")
    w("*Generated by `scripts/python/analysis/domain/sensitivity_analysis_exp2.py`*\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV export for plotting
# ---------------------------------------------------------------------------
def export_plot_data(all_results: dict):
    """Export sensitivity indices to CSV for the plot script."""
    rows = []
    for metric, mlabel in PRIMARY_METRICS:
        res = all_results[metric]
        for f in FACTORS:
            s1_val, s1_lo, s1_hi = res[f"S1_{f}"]
            st_val, st_lo, st_hi = res[f"ST_{f}"]
            rows.append({
                "metric": mlabel,
                "factor": FACTOR_NAMES[f],
                "factor_key": f,
                "S1": s1_val, "S1_lo": s1_lo, "S1_hi": s1_hi,
                "ST": st_val, "ST_lo": st_lo, "ST_hi": st_hi,
            })

    df_out = pd.DataFrame(rows)
    out_path = CSV_OUT / "sobol_indices.csv"
    df_out.to_csv(out_path, index=False)
    print(f"Saved: {out_path.relative_to(PROJECT_ROOT)}")

    # Also export the full decomposition
    rows_full = []
    for metric, mlabel in PRIMARY_METRICS:
        res = all_results[metric]
        for k, (val, lo, hi) in res.items():
            if k.startswith("S_"):
                effect_type = "residual"
                order = 99
            elif k.startswith("S1_"):
                effect_type = "main"
                order = 1
            elif k.startswith("S2_"):
                effect_type = "interaction_2way"
                order = 2
            elif k.startswith("S3_"):
                effect_type = "interaction_3way"
                order = 3
            elif k.startswith("S4_"):
                effect_type = "interaction_4way"
                order = 4
            elif k.startswith("ST_"):
                continue  # total-order exported separately
            else:
                continue
            rows_full.append({
                "metric": mlabel,
                "effect": k,
                "type": effect_type,
                "order": order,
                "S": val, "S_lo": lo, "S_hi": hi,
            })

    df_full = pd.DataFrame(rows_full)
    out_path2 = CSV_OUT / "variance_decomposition.csv"
    df_full.to_csv(out_path2, index=False)
    print(f"Saved: {out_path2.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Sensitivity Analysis — Experiment 2")
    print("=" * 60)

    df = load_all_data()

    all_results = {}
    for metric, mlabel in PRIMARY_METRICS:
        print(f"\n--- {mlabel} ---")
        result = bootstrap_sobol(df, metric, B=2000)
        all_results[metric] = result

        # Print summary
        print(f"  First-order indices:")
        for f in FACTORS:
            s1, _, _ = result[f"S1_{f}"]
            st, _, _ = result[f"ST_{f}"]
            print(f"    {FACTOR_NAMES[f]:25s}  S1={s1:.4f}  ST={st:.4f}")
        s_res, _, _ = result["S_residual"]
        print(f"    {'Residual (seed)':25s}  {s_res:.4f}")

    # Generate report
    report = generate_report(all_results)
    report_path = REPORT_DIR / "sensitivity_analysis_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\nReport: {report_path.relative_to(PROJECT_ROOT)}")

    # Export CSV for plotting
    export_plot_data(all_results)

    print("\nDone.")


if __name__ == "__main__":
    main()
