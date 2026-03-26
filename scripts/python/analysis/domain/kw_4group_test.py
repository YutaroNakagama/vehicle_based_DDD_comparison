#!/usr/bin/env python3
"""
Compute 4-group Kruskal-Wallis test (Method: baseline, RUS, SMOTE, SW-SMOTE)
and SMOTE vs SW-SMOTE post-hoc Mann-Whitney U test for the H1 restructuring.

Pools r=0.1 and r=0.5 within each method.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

CSV_BASE = PROJECT_ROOT / "results" / "analysis" / "exp2_domain_shift" / "figures" / "csv" / "split2"

MODES = ["source_only", "target_only", "mixed"]
DISTANCES = ["mmd", "dtw", "wasserstein"]
LEVELS = ["in_domain", "out_domain"]
METRICS = [("f2", "F2-score"), ("auc", "AUROC"), ("auc_pr", "AUPRC")]
OFFICIAL_SEEDS = {0, 1, 3, 7, 13, 42, 123, 256, 512, 999, 1337, 2024}

METHOD_GROUPS = {
    "Baseline": ["baseline"],
    "RUS": ["rus_r01", "rus_r05"],
    "SMOTE": ["smote_r01", "smote_r05"],
    "SW-SMOTE": ["sw_smote_r01", "sw_smote_r05"],
}

MODE_LABEL = {"source_only": "Cross", "target_only": "Within", "mixed": "Mixed"}
LEVEL_LABEL = {"in_domain": "In", "out_domain": "Out"}


def load_data() -> pd.DataFrame:
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
                lambda r: f"{method}_r{str(r).replace('.', '')}" if pd.notna(r) else method
            )
        dfs.append(df)
    merged = pd.concat(dfs, ignore_index=True)
    merged = merged[merged["seed"].isin(OFFICIAL_SEEDS)]
    return merged


def eta_squared_from_H(H: float, n: int, k: int) -> float:
    denom = n - k
    if denom <= 0:
        return np.nan
    return max(0.0, (H - k + 1) / denom)


def cliffs_delta(x, y):
    nx, ny = len(x), len(y)
    more = sum(1 for xi in x for yj in y if xi > yj)
    less = sum(1 for xi in x for yj in y if xi < yj)
    return (more - less) / (nx * ny)


def main():
    df = load_data()
    alpha = 0.05
    n_cells = len(MODES) * len(LEVELS) * len(DISTANCES)  # 18
    alpha_c = alpha / n_cells

    print("=" * 80)
    print("4-GROUP KRUSKAL-WALLIS TEST (Method: Baseline, RUS, SMOTE, SW-SMOTE)")
    print(f"Bonferroni α' = {alpha}/{n_cells} = {alpha_c:.4f}")
    print("=" * 80)

    for metric_col, metric_name in METRICS:
        print(f"\n### {metric_name}\n")
        rows = []
        for mode in MODES:
            for level in LEVELS:
                for dist in DISTANCES:
                    groups = []
                    group_ns = []
                    for method_name, conds in METHOD_GROUPS.items():
                        mask = (
                            (df["condition"].isin(conds))
                            & (df["mode"] == mode)
                            & (df["level"] == level)
                            & (df["distance"] == dist)
                        )
                        v = df.loc[mask, metric_col].dropna().values
                        groups.append(v)
                        group_ns.append(len(v))
                    n_total = sum(group_ns)
                    H, p = stats.kruskal(*groups)
                    eta2 = eta_squared_from_H(H, n_total, 4)
                    sig = "✓" if p < alpha_c else ""
                    rows.append({
                        "mode": mode, "level": level, "dist": dist,
                        "H": H, "p": p, "eta2": eta2, "sig": sig,
                        "n": n_total, "ns": group_ns,
                    })

        sig_count = sum(1 for r in rows if r["sig"] == "✓")
        eta2_vals = [r["eta2"] for r in rows]
        H_vals = [r["H"] for r in rows]
        print(f"Significant: {sig_count}/{len(rows)}")
        print(f"Mean η²: {np.mean(eta2_vals):.3f}")
        print(f"H range: {min(H_vals):.2f}–{max(H_vals):.2f}")
        print(f"η² range: {min(eta2_vals):.3f}–{max(eta2_vals):.3f}")
        print()
        print("| Mode | Level | Dist | n (BL/RUS/SM/SW) | H | p | η² | Sig |")
        print("|------|-------|------|------------------|--:|--:|---:|:---:|")
        for r in rows:
            ns_str = "/".join(str(n) for n in r["ns"])
            print(f"| {MODE_LABEL[r['mode']]} | {LEVEL_LABEL[r['level']]} | "
                  f"{r['dist'].upper()} | {ns_str} | "
                  f"{r['H']:.2f} | {r['p']:.6f} | {r['eta2']:.3f} | {r['sig']} |")

    # -----------------------------------------------------------------------
    # SMOTE vs SW-SMOTE post-hoc (absorbed from H7)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SMOTE vs SW-SMOTE POST-HOC (Mann-Whitney U)")
    print("=" * 80)

    n_smote_tests = len(MODES) * len(LEVELS)  # 6 cells (pool distances)
    alpha_smote = alpha / (3 * n_smote_tests)  # 3 contrasts × 6 cells = 18
    print(f"Bonferroni α' = {alpha_smote:.5f} (18 comparisons)")

    for metric_col, metric_name in METRICS:
        print(f"\n### {metric_name} — SMOTE vs SW-SMOTE\n")
        smote_wins = 0
        sw_wins = 0
        sig_count = 0
        deltas = []
        for mode in MODES:
            for level in LEVELS:
                mask_sm = (
                    (df["condition"].isin(["smote_r01", "smote_r05"]))
                    & (df["mode"] == mode) & (df["level"] == level)
                )
                mask_sw = (
                    (df["condition"].isin(["sw_smote_r01", "sw_smote_r05"]))
                    & (df["mode"] == mode) & (df["level"] == level)
                )
                x_sm = df.loc[mask_sm, metric_col].dropna().values
                x_sw = df.loc[mask_sw, metric_col].dropna().values
                U, p = stats.mannwhitneyu(x_sm, x_sw, alternative="two-sided")
                d = cliffs_delta(x_sm, x_sw)
                deltas.append(d)
                winner = "SMOTE" if d > 0 else "SW-SMOTE"
                if d > 0:
                    smote_wins += 1
                else:
                    sw_wins += 1
                sig = "✓" if p < alpha_smote else ""
                if sig:
                    sig_count += 1
                print(f"  {MODE_LABEL[mode]:6s} × {LEVEL_LABEL[level]:3s}: "
                      f"δ={d:+.3f} ({winner}), p={p:.6f} {sig}")
        print(f"  Summary: SMOTE wins {smote_wins}/6, SW-SMOTE wins {sw_wins}/6, "
              f"Significant: {sig_count}/6, Mean |δ|={np.mean(np.abs(deltas)):.3f}")

    # -----------------------------------------------------------------------
    # Pooled method means for paper text
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("POOLED METHOD MEANS (across all cells)")
    print("=" * 80)

    for metric_col, metric_name in METRICS:
        print(f"\n### {metric_name}\n")
        print("| Method | Mean ± SD | n |")
        print("|--------|-----------|---|")
        for method_name, conds in METHOD_GROUPS.items():
            vals = df.loc[df["condition"].isin(conds), metric_col].dropna()
            print(f"| {method_name} | {vals.mean():.3f} ± {vals.std():.3f} | {len(vals)} |")


if __name__ == "__main__":
    main()
