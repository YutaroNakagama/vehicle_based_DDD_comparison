#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_xgboost_validation.py
===========================
Generate publication-quality plots for the XGBoost validation experiment.

Produces:
  1. fig9_sobol_comparison.pdf — Side-by-side RF vs XGBoost Sobol bars
  2. fig10_xgb_ranking_reversal.pdf — XGBoost ranking reversal bump chart
  3. Console output with detailed comparison statistics

Usage:
    python scripts/python/analysis/domain/plot_xgboost_validation.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

LATEX_DIR = PROJECT_ROOT / "docs" / "experiments" / "results" / "exp2-analysis" / "latex"
EVAL_DIR = PROJECT_ROOT / "results" / "outputs" / "evaluation" / "XGBoost"
RF_CSV_BASE = PROJECT_ROOT / "results" / "analysis" / "exp2_domain_shift" / "figures" / "csv" / "split2"

SEEDS = [0, 1, 3, 7, 13, 42, 123, 256, 512, 999, 1337, 2024]

# ---------- Data loading ----------

XGB_BASELINE_PAT = re.compile(
    r"eval_results_XGBoost_(?P<mode>source_only|target_only|mixed)_"
    r"xgb_baseline_domain_knn_(?P<distance>mmd|dtw|wasserstein)_"
    r"(?P<domain>in_domain|out_domain)_(?:source_only|target_only|mixed)"
    r"(?:_split2)_s(?P<seed>\d+)\.json$"
)
XGB_SMOTE_PLAIN_PAT = re.compile(
    r"eval_results_XGBoost_(?P<mode>source_only|target_only|mixed)_"
    r"xgb_smote_plain_knn_(?P<distance>mmd|dtw|wasserstein)_"
    r"(?P<domain>in_domain|out_domain)_(?:source_only|target_only|mixed)"
    r"(?:_split2)_ratio(?P<ratio>[0-9.]+)_s(?P<seed>\d+)\.json$"
)
XGB_UNDERSAMPLE_PAT = re.compile(
    r"eval_results_XGBoost_(?P<mode>source_only|target_only|mixed)_"
    r"xgb_undersample_rus_knn_(?P<distance>mmd|dtw|wasserstein)_"
    r"(?P<domain>in_domain|out_domain)_(?:source_only|target_only|mixed)"
    r"(?:_split2)_ratio(?P<ratio>[0-9.]+)_s(?P<seed>\d+)\.json$"
)
XGB_SW_SMOTE_PAT = re.compile(
    r"eval_results_XGBoost_(?P<mode>source_only|target_only|mixed)_"
    r"xgb_imbalv3_knn_(?P<distance>mmd|dtw|wasserstein)_"
    r"(?P<domain>in_domain|out_domain)_(?:source_only|target_only|mixed)"
    r"(?:_split2)(?:_subjectwise)_ratio(?P<ratio>[0-9.]+)_s(?P<seed>\d+)\.json$"
)


def load_xgboost_data() -> pd.DataFrame:
    """Load all XGBoost eval JSONs into a DataFrame."""
    patterns = [
        ("baseline", XGB_BASELINE_PAT),
        ("smote_plain", XGB_SMOTE_PLAIN_PAT),
        ("undersample_rus", XGB_UNDERSAMPLE_PAT),
        ("sw_smote", XGB_SW_SMOTE_PAT),
    ]

    records = []
    for json_path in sorted(EVAL_DIR.rglob("eval_results_XGBoost_*.json")):
        meta = None
        for cond, pat in patterns:
            m = pat.match(json_path.name)
            if m:
                meta = m.groupdict()
                meta["condition"] = cond
                break
        if meta is None:
            continue

        with open(json_path) as f:
            d = json.load(f)

        prec = d.get("precision", 0.0)
        rec = d.get("recall", 0.0)
        f2 = (5 * prec * rec) / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0

        cond = meta["condition"]
        ratio = meta.get("ratio", "")
        if cond == "baseline":
            rebal = "Baseline"
        elif cond == "smote_plain":
            rebal = f"SMOTE r={ratio}"
        elif cond == "undersample_rus":
            rebal = f"RUS r={ratio}"
        elif cond == "sw_smote":
            rebal = f"SW-SMOTE r={ratio}"

        records.append({
            "rebal": rebal,
            "mode": meta["mode"],
            "distance": meta["distance"],
            "domain": meta["domain"],
            "seed": int(meta["seed"]),
            "f2": f2,
            "auc": d.get("roc_auc", d.get("auc", 0.0)),
            "auc_pr": d.get("auc_pr", d.get("average_precision", 0.0)),
            "f1": d.get("f1", 0.0),
            "jobid": json_path.parent.parent.name,
        })

    df = pd.DataFrame(records)
    # De-duplicate
    key_cols = ["rebal", "mode", "distance", "domain", "seed"]
    df = df.sort_values("jobid").groupby(key_cols).last().reset_index()
    return df


def sobol_indices_bootstrap(data: pd.DataFrame, metric: str, factors: list[str],
                            B: int = 2000):
    """Compute Sobol indices with bootstrap CIs."""
    Y = data[metric].values
    V_total = np.var(Y, ddof=0)

    if V_total < 1e-12:
        return {}

    results = {}

    # First-order indices
    for factor in factors:
        levels = sorted(data[factor].unique())
        cond_means = np.array([data.loc[data[factor] == lv, metric].mean()
                               for lv in levels])
        weights = np.array([len(data[data[factor] == lv]) for lv in levels],
                           dtype=float)
        weights /= weights.sum()
        grand = np.sum(cond_means * weights)
        V_i = np.sum(weights * (cond_means - grand) ** 2)
        results[f"S1_{factor}"] = V_i / V_total

    # Second-order: all pairs
    for i, f1 in enumerate(factors):
        for f2 in factors[i + 1:]:
            lv1 = sorted(data[f1].unique())
            lv2 = sorted(data[f2].unique())
            cell_means = []
            cell_weights = []
            for l1, l2 in product(lv1, lv2):
                mask = (data[f1] == l1) & (data[f2] == l2)
                subset = data.loc[mask, metric]
                if len(subset) > 0:
                    cell_means.append(subset.mean())
                    cell_weights.append(len(subset))
            cell_means = np.array(cell_means)
            cell_weights = np.array(cell_weights, dtype=float)
            cell_weights /= cell_weights.sum()
            grand = np.sum(cell_means * cell_weights)
            V_ij_total = np.sum(cell_weights * (cell_means - grand) ** 2)
            V_ij = max(0, V_ij_total - results.get(f"S1_{f1}", 0) * V_total
                       - results.get(f"S1_{f2}", 0) * V_total) / V_total
            results[f"S_{f1}x{f2}"] = V_ij

    # Residual
    explained = sum(v for v in results.values())
    results["S_residual"] = max(0, 1 - explained)

    # Bootstrap CIs
    rng = np.random.default_rng(42)
    boot_results = {k: [] for k in results}
    n = len(data)
    for _ in range(B):
        idx = rng.choice(n, n, replace=True)
        boot_df = data.iloc[idx].reset_index(drop=True)
        boot_Y = boot_df[metric].values
        boot_V = np.var(boot_Y, ddof=0)
        if boot_V < 1e-12:
            continue

        for factor in factors:
            levels = sorted(boot_df[factor].unique())
            cm = np.array([boot_df.loc[boot_df[factor] == lv, metric].mean()
                           for lv in levels])
            w = np.array([len(boot_df[boot_df[factor] == lv]) for lv in levels],
                         dtype=float)
            w /= w.sum()
            g = np.sum(cm * w)
            vi = np.sum(w * (cm - g) ** 2)
            boot_results[f"S1_{factor}"].append(vi / boot_V)

        for i, f1 in enumerate(factors):
            for f2 in factors[i + 1:]:
                lv1 = sorted(boot_df[f1].unique())
                lv2 = sorted(boot_df[f2].unique())
                cms, cws = [], []
                for l1, l2 in product(lv1, lv2):
                    mask = (boot_df[f1] == l1) & (boot_df[f2] == l2)
                    s = boot_df.loc[mask, metric]
                    if len(s) > 0:
                        cms.append(s.mean())
                        cws.append(len(s))
                cms = np.array(cms)
                cws = np.array(cws, dtype=float)
                cws /= cws.sum()
                g = np.sum(cms * cws)
                vij_t = np.sum(cws * (cms - g) ** 2)
                s1_f1 = boot_results[f"S1_{f1}"][-1] if boot_results[f"S1_{f1}"] else 0
                s1_f2 = boot_results[f"S1_{f2}"][-1] if boot_results[f"S1_{f2}"] else 0
                vij = max(0, vij_t - s1_f1 * boot_V - s1_f2 * boot_V) / boot_V
                boot_results[f"S_{f1}x{f2}"].append(vij)

        boot_expl = sum(boot_results[k][-1] for k in results if k != "S_residual"
                        and boot_results[k])
        boot_results["S_residual"].append(max(0, 1 - boot_expl))

    ci = {}
    for k, v in results.items():
        boots = boot_results[k]
        if boots:
            lo = np.percentile(boots, 2.5)
            hi = np.percentile(boots, 97.5)
        else:
            lo = hi = v
        ci[k] = (v, lo, hi)

    return ci


# ---------- Plot 1: Sobol comparison ----------

def plot_sobol_comparison(xgb_sobol: dict, ax_title_prefix: str = ""):
    """Side-by-side RF vs XGBoost Sobol bar chart."""
    # RF values from the paper (F2-score)
    rf_sobol_f2 = {
        "S1_mode": 0.368, "S1_rebal": 0.243,
        "S1_distance": 0.005, "S1_domain": 0.010,
        "ST_mode": 0.577, "ST_rebal": 0.458,
        "ST_distance": 0.011, "ST_domain": 0.028,
    }
    rf_sobol_auc = {
        "S1_mode": 0.504, "S1_rebal": 0.246,
        "S1_distance": 0.004, "S1_domain": 0.003,
        "ST_mode": 0.657, "ST_rebal": 0.401,
        "ST_distance": 0.012, "ST_domain": 0.020,
    }

    factors = ["mode", "rebal", "distance", "domain"]
    factor_labels = ["Mode ($M$)", "Rebalancing ($R$)", "Distance ($D$)", "Membership ($G$)"]

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2), sharey=True)

    for ax_i, (metric, mlabel, rf_vals) in enumerate([
        ("f2", "F2-score", rf_sobol_f2),
        ("auc", "AUROC", rf_sobol_auc),
    ]):
        ax = axes[ax_i]
        xgb_s = xgb_sobol[metric]

        x = np.arange(len(factors))
        width = 0.35

        # RF bars
        rf_s1 = [rf_vals[f"S1_{f}"] for f in factors]
        rf_int = [rf_vals[f"ST_{f}"] - rf_vals[f"S1_{f}"] for f in factors]

        # XGBoost bars
        xgb_s1 = [xgb_s[f"S1_{f}"][0] for f in factors]
        xgb_st = []
        for f in factors:
            st = xgb_s[f"S1_{f}"][0]
            for k, (v, _, _) in xgb_s.items():
                if k.startswith("S_") and f in k and k != "S_residual":
                    st += v
            xgb_st.append(st)
        xgb_int = [st - s1 for st, s1 in zip(xgb_st, xgb_s1)]

        # Plot RF
        bars1 = ax.bar(x - width / 2, rf_s1, width, label="RF $S_i$",
                       color="#2196F3", alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.bar(x - width / 2, rf_int, width, bottom=rf_s1,
               label="RF interact.", color="#2196F3", alpha=0.35,
               hatch="//", edgecolor="#2196F3", linewidth=0.5)

        # Plot XGBoost
        bars2 = ax.bar(x + width / 2, xgb_s1, width, label="XGB $S_i$",
                       color="#FF9800", alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.bar(x + width / 2, xgb_int, width, bottom=xgb_s1,
               label="XGB interact.", color="#FF9800", alpha=0.35,
               hatch="\\\\", edgecolor="#FF9800", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(factor_labels, fontsize=7.5, rotation=15, ha="right")
        ax.set_title(mlabel, fontsize=10, fontweight="bold")
        ax.set_ylim(0, 0.75)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if ax_i == 0:
            ax.set_ylabel("Sobol Index", fontsize=9)

    axes[0].legend(fontsize=6.5, loc="upper right", ncol=1,
                   framealpha=0.8, handlelength=1.5)

    fig.suptitle("Classifier-Independence Validation: RF vs XGBoost",
                 fontsize=10.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ---------- Plot 2: XGBoost ranking reversal ----------

def plot_ranking_reversal(df: pd.DataFrame):
    """Bump chart of strategy rankings across modes for XGBoost."""
    strategies = sorted(df["rebal"].unique())
    modes = ["source_only", "mixed", "target_only"]
    mode_labels = ["Source-only\n(cross-domain)", "Mixed", "Target-only\n(within-domain)"]

    # Compute mean F2 per strategy × mode
    rankings = {}
    for mode in modes:
        mode_df = df[df["mode"] == mode]
        means = mode_df.groupby("rebal")["f2"].mean()
        ranked = means.rank(ascending=False)
        for strat in strategies:
            if strat in ranked.index:
                rankings.setdefault(strat, []).append(ranked[strat])
            else:
                rankings.setdefault(strat, []).append(np.nan)

    fig, ax = plt.subplots(figsize=(5, 3.5))

    colors = {
        "Baseline": "#607D8B",
        "RUS r=0.1": "#F44336", "RUS r=0.5": "#E91E63",
        "SMOTE r=0.1": "#4CAF50", "SMOTE r=0.5": "#8BC34A",
        "SW-SMOTE r=0.1": "#2196F3", "SW-SMOTE r=0.5": "#03A9F4",
    }
    markers = {
        "Baseline": "D",
        "RUS r=0.1": "v", "RUS r=0.5": "v",
        "SMOTE r=0.1": "^", "SMOTE r=0.5": "^",
        "SW-SMOTE r=0.1": "s", "SW-SMOTE r=0.5": "s",
    }

    x_pos = [0, 1, 2]
    for strat, ranks in rankings.items():
        ax.plot(x_pos, ranks, "-o", color=colors.get(strat, "gray"),
                marker=markers.get(strat, "o"), markersize=6,
                linewidth=1.8, label=strat, alpha=0.85)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(mode_labels, fontsize=8)
    ax.set_ylabel("Rank (1 = best)", fontsize=9)
    ax.set_ylim(7.8, 0.2)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=6.5, loc="center left", bbox_to_anchor=(1.01, 0.5),
              framealpha=0.8)
    ax.set_title("XGBoost — Strategy Ranking Reversal", fontsize=10, fontweight="bold")

    fig.tight_layout()
    return fig


# ---------- Plot 3: XGBoost per-mode performance table data ----------

def print_permode_table(df: pd.DataFrame):
    """Print XGBoost per-mode performance for paper Table insertion."""
    strategies_order = [
        "Baseline", "RUS r=0.1", "RUS r=0.5",
        "SMOTE r=0.1", "SMOTE r=0.5",
        "SW-SMOTE r=0.1", "SW-SMOTE r=0.5",
    ]
    modes = ["source_only", "mixed", "target_only"]

    print("\n" + "=" * 80)
    print("XGBoost Per-Mode Performance (for Table)")
    print("=" * 80)
    header = f"{'Strategy':20s} | {'F2(Cross)':>10s} | {'F2(Within)':>10s} | {'F2(Mixed)':>10s} | {'AUC(Within)':>11s} | {'AUPRC(Within)':>13s}"
    print(header)
    print("-" * len(header))

    for strat in strategies_order:
        vals = {}
        for mode in modes:
            subset = df[(df["rebal"] == strat) & (df["mode"] == mode)]
            if len(subset) > 0:
                vals[f"f2_{mode}"] = subset["f2"].mean()
                vals[f"auc_{mode}"] = subset["auc"].mean()
                vals[f"auc_pr_{mode}"] = subset["auc_pr"].mean()

        f2_s = vals.get("f2_source_only", 0)
        f2_t = vals.get("f2_target_only", 0)
        f2_m = vals.get("f2_mixed", 0)
        auc_t = vals.get("auc_target_only", 0)
        auprc_t = vals.get("auc_pr_target_only", 0)

        print(f"{strat:20s} | {f2_s:10.3f} | {f2_t:10.3f} | {f2_m:10.3f} | {auc_t:11.3f} | {auprc_t:13.3f}")

    # Spearman correlation of rankings across modes
    print("\n--- Ranking Correlations (Spearman ρ) ---")
    for m1, m2 in [("source_only", "target_only"), ("source_only", "mixed"),
                    ("target_only", "mixed")]:
        r1 = df[df["mode"] == m1].groupby("rebal")["f2"].mean()
        r2 = df[df["mode"] == m2].groupby("rebal")["f2"].mean()
        common = r1.index.intersection(r2.index)
        if len(common) >= 3:
            rho, pval = stats.spearmanr(r1[common].values, r2[common].values)
            print(f"  {m1} vs {m2}: ρ = {rho:.3f} (p = {pval:.4f})")


# ---------- Main ----------

def main():
    print("Loading XGBoost data...")
    df = load_xgboost_data()
    print(f"Loaded {len(df)} records")
    print(f"Modes: {sorted(df['mode'].unique())}")
    print(f"Rebalancing: {sorted(df['rebal'].unique())}")

    # Sobol indices
    factors = ["rebal", "mode", "distance", "domain"]
    xgb_sobol = {}
    for metric in ["f2", "auc"]:
        print(f"\nComputing Sobol indices for {metric}...")
        xgb_sobol[metric] = sobol_indices_bootstrap(df, metric, factors, B=2000)

    # Print comparison statistics
    print("\n" + "=" * 70)
    print("SOBOL INDICES COMPARISON")
    print("=" * 70)

    rf_f2 = {"S1_mode": 0.368, "S1_rebal": 0.243, "S1_distance": 0.005,
             "S1_domain": 0.010, "S_RxM": 0.212, "S_residual": 0.157}

    print(f"\n{'Factor':25s} | {'RF S1':>8s} | {'XGB S1':>8s} | {'Δ':>8s}")
    print("-" * 60)
    for f in factors:
        rf_v = rf_f2.get(f"S1_{f}", rf_f2.get(f"S1_{f}", 0))
        xgb_v = xgb_sobol["f2"][f"S1_{f}"][0]
        delta = xgb_v - rf_v
        print(f"{f:25s} | {rf_v:8.4f} | {xgb_v:8.4f} | {delta:+8.4f}")

    rf_rxm = rf_f2.get("S_RxM", 0)
    xgb_rxm = xgb_sobol["f2"].get("S_rebalxmode", (0,))[0]
    print(f"{'R×M interaction':25s} | {rf_rxm:8.4f} | {xgb_rxm:8.4f} | {xgb_rxm - rf_rxm:+8.4f}")

    rf_res = rf_f2.get("S_residual", 0)
    xgb_res = xgb_sobol["f2"].get("S_residual", (0,))[0]
    print(f"{'Residual':25s} | {rf_res:8.4f} | {xgb_res:8.4f} | {xgb_res - rf_res:+8.4f}")

    # Systematic variance fraction
    xgb_sys = (xgb_sobol["f2"]["S1_mode"][0] + xgb_sobol["f2"]["S1_rebal"][0]
               + xgb_sobol["f2"].get("S_rebalxmode", (0,))[0])
    xgb_res_val = xgb_sobol["f2"].get("S_residual", (0,))[0]
    xgb_sys_frac = xgb_sys / (1 - xgb_res_val) if (1 - xgb_res_val) > 0 else 0
    print(f"\nM+R+R×M systematic fraction: {xgb_sys_frac:.1%}")

    # Plot 1: Sobol comparison
    print("\nGenerating fig9_sobol_comparison.pdf ...")
    fig1 = plot_sobol_comparison(xgb_sobol)
    fig1.savefig(LATEX_DIR / "fig9_sobol_comparison.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig1)
    print(f"  Saved: {LATEX_DIR / 'fig9_sobol_comparison.pdf'}")

    # Plot 2: Ranking reversal
    print("\nGenerating fig10_xgb_ranking_reversal.pdf ...")
    fig2 = plot_ranking_reversal(df)
    fig2.savefig(LATEX_DIR / "fig10_xgb_ranking_reversal.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig2)
    print(f"  Saved: {LATEX_DIR / 'fig10_xgb_ranking_reversal.pdf'}")

    # Per-mode table
    print_permode_table(df)

    print("\nDone.")


if __name__ == "__main__":
    main()
