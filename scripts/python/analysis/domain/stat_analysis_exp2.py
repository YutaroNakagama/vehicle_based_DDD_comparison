#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stat_analysis_exp2.py
=====================
Comprehensive statistical analysis of Experiment 2 domain shift results.

Factors:
  - condition:  baseline, rus (r0.1), smote (r0.1), sw_smote (r0.1)
  - distance:   mmd, dtw, wasserstein
  - mode:       source_only (cross-domain), target_only (within-domain), mixed
  - level:      in_domain, out_domain

Metrics analysed:  F2-score, AUROC

Analysis performed:
  1. Descriptive statistics (mean ± SD per cell)
  2. Kruskal-Wallis H-test across conditions (per mode × level × distance)
  3. Mann-Whitney U pairwise tests (baseline vs each method)
  4. Cliff's delta effect size for each pairwise comparison
  5. Domain gap Δ = out_domain − in_domain analysis
  6. Friedman test for repeated-measures (seeds) across conditions
  7. Bonferroni-corrected significance assessment
  8. Interaction analysis: does the best condition depend on distance metric?

Output:  results/analysis/exp2_domain_shift/statistical_report.md
"""

from __future__ import annotations

import sys
import textwrap
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

REPORT_DIR = PROJECT_ROOT / "results" / "analysis" / "exp2_domain_shift"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CSV_BASE = REPORT_DIR / "figures" / "csv" / "split2"

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
def load_data(ratio: float = 0.1) -> pd.DataFrame:
    """Load & merge all condition CSVs, keeping only the specified ratio for
    methods that have ratios (to ensure fair comparison with baseline)."""
    files = {
        "baseline": CSV_BASE / "baseline" / "baseline_domain_split2_metrics_v2.csv",
        "smote":    CSV_BASE / "smote_plain" / "smote_plain_split2_metrics_v2.csv",
        "rus":      CSV_BASE / "undersample_rus" / "undersample_rus_split2_metrics_v2.csv",
        "sw_smote": CSV_BASE / "sw_smote" / "sw_smote_split2_metrics_v2.csv",
    }
    dfs = []
    for cond, path in files.items():
        df = pd.read_csv(path)
        df["condition"] = cond
        # baseline has no ratio; for others keep the specified ratio
        if cond != "baseline":
            df = df[df["ratio"] == ratio].copy()
        dfs.append(df)
    merged = pd.concat(dfs, ignore_index=True)
    return merged


# ---------------------------------------------------------------------------
# 2. Descriptive statistics
# ---------------------------------------------------------------------------
def descriptive_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return mean ± sd table grouped by condition × mode × level."""
    grp = df.groupby(["condition", "mode", "level"])[metric]
    desc = grp.agg(["mean", "std", "count"]).reset_index()
    desc.columns = ["condition", "mode", "level", "mean", "std", "n"]
    return desc


# ---------------------------------------------------------------------------
# 3. Kruskal-Wallis across conditions
# ---------------------------------------------------------------------------
def kruskal_wallis_per_cell(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Kruskal-Wallis H-test for each (mode × level × distance) cell."""
    rows = []
    conditions = sorted(df["condition"].unique())
    for mode in ["source_only", "target_only", "mixed"]:
        for level in ["in_domain", "out_domain"]:
            for dist in ["mmd", "dtw", "wasserstein"]:
                groups = []
                for c in conditions:
                    vals = df[(df["condition"] == c) & (df["mode"] == mode)
                              & (df["level"] == level) & (df["distance"] == dist)][metric].dropna()
                    groups.append(vals.values)
                if all(len(g) >= 2 for g in groups):
                    H, p = stats.kruskal(*groups)
                else:
                    H, p = np.nan, np.nan
                rows.append({
                    "mode": mode, "level": level, "distance": dist,
                    "H": H, "p": p, "n_groups": len(groups),
                    "n_per_group": [len(g) for g in groups],
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Mann-Whitney U pairwise (baseline vs each)
# ---------------------------------------------------------------------------
def cliff_delta(x: np.ndarray, y: np.ndarray) -> float:
    r"""Cliff's delta (non-parametric effect size).

    $$\delta = \frac{\#(x_i > y_j) - \#(x_i < y_j)}{n_x \cdot n_y}$$

    Interpretation: |δ| < 0.147 negligible, < 0.33 small, < 0.474 medium, else large.
    """
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return np.nan
    more = sum(1 for xi in x for yj in y if xi > yj)
    less = sum(1 for xi in x for yj in y if xi < yj)
    return (more - less) / (nx * ny)


def cliff_delta_label(d: float) -> str:
    ad = abs(d)
    if ad < 0.147:
        return "negligible"
    elif ad < 0.33:
        return "small"
    elif ad < 0.474:
        return "medium"
    else:
        return "large"


def pairwise_vs_baseline(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mann-Whitney U tests: baseline vs each method, per (mode × level × distance)."""
    rows = []
    methods = [c for c in sorted(df["condition"].unique()) if c != "baseline"]
    for mode in ["source_only", "target_only", "mixed"]:
        for level in ["in_domain", "out_domain"]:
            for dist in ["mmd", "dtw", "wasserstein"]:
                base = df[(df["condition"] == "baseline") & (df["mode"] == mode)
                          & (df["level"] == level) & (df["distance"] == dist)][metric].dropna().values
                for meth in methods:
                    treat = df[(df["condition"] == meth) & (df["mode"] == mode)
                              & (df["level"] == level) & (df["distance"] == dist)][metric].dropna().values
                    if len(base) >= 2 and len(treat) >= 2:
                        U, p = stats.mannwhitneyu(base, treat, alternative="two-sided")
                        d = cliff_delta(treat, base)
                    else:
                        U, p, d = np.nan, np.nan, np.nan
                    rows.append({
                        "mode": mode, "level": level, "distance": dist,
                        "comparison": f"{meth} vs baseline",
                        "U": U, "p": p,
                        "cliff_d": d,
                        "effect": cliff_delta_label(d) if not np.isnan(d) else "",
                        "n_base": len(base), "n_treat": len(treat),
                        "mean_base": np.mean(base) if len(base) else np.nan,
                        "mean_treat": np.mean(treat) if len(treat) else np.nan,
                    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Domain gap analysis
# ---------------------------------------------------------------------------
def domain_gap_analysis(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Compute domain gap Δ = metric(out_domain) − metric(in_domain) per seed,
    then test whether Δ differs across conditions."""
    rows = []
    for mode in ["source_only", "target_only", "mixed"]:
        for dist in ["mmd", "dtw", "wasserstein"]:
            gap_by_cond = {}
            for cond in sorted(df["condition"].unique()):
                sub = df[(df["condition"] == cond) & (df["mode"] == mode)
                         & (df["distance"] == dist)]
                in_vals = sub[sub["level"] == "in_domain"].set_index("seed")[metric]
                out_vals = sub[sub["level"] == "out_domain"].set_index("seed")[metric]
                common_seeds = in_vals.index.intersection(out_vals.index)
                gaps = out_vals.loc[common_seeds].values - in_vals.loc[common_seeds].values
                gap_by_cond[cond] = gaps

            # Kruskal-Wallis on gaps
            valid = {k: v for k, v in gap_by_cond.items() if len(v) >= 2}
            if len(valid) >= 2:
                H, p = stats.kruskal(*valid.values())
            else:
                H, p = np.nan, np.nan

            for cond, gaps in gap_by_cond.items():
                rows.append({
                    "mode": mode, "distance": dist, "condition": cond,
                    "mean_gap": np.mean(gaps) if len(gaps) else np.nan,
                    "std_gap": np.std(gaps, ddof=1) if len(gaps) > 1 else np.nan,
                    "n_pairs": len(gaps),
                    "KW_H": H, "KW_p": p,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Friedman test (repeated-measures across seeds)
# ---------------------------------------------------------------------------
def friedman_across_conditions(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Friedman test treating seeds as blocks (repeated measures).
    For each (mode × level × distance), compare conditions across shared seeds."""
    rows = []
    conditions = sorted(df["condition"].unique())
    for mode in ["source_only", "target_only", "mixed"]:
        for level in ["in_domain", "out_domain"]:
            for dist in ["mmd", "dtw", "wasserstein"]:
                seed_vals = {}
                for cond in conditions:
                    sub = df[(df["condition"] == cond) & (df["mode"] == mode)
                             & (df["level"] == level) & (df["distance"] == dist)]
                    seed_vals[cond] = sub.set_index("seed")[metric]
                # Find common seeds
                idx = None
                for sv in seed_vals.values():
                    idx = sv.index if idx is None else idx.intersection(sv.index)
                if idx is not None and len(idx) >= 3:
                    arrays = [seed_vals[c].loc[idx].values for c in conditions]
                    chi2, p = stats.friedmanchisquare(*arrays)
                    n = len(idx)
                else:
                    chi2, p, n = np.nan, np.nan, 0
                rows.append({
                    "mode": mode, "level": level, "distance": dist,
                    "chi2": chi2, "p": p, "n_seeds": n,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 7. Wilcoxon signed-rank (paired: baseline vs method, per seed)
# ---------------------------------------------------------------------------
def wilcoxon_paired(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Wilcoxon signed-rank test (paired by seed): baseline vs each method."""
    rows = []
    methods = [c for c in sorted(df["condition"].unique()) if c != "baseline"]
    for mode in ["source_only", "target_only", "mixed"]:
        for level in ["in_domain", "out_domain"]:
            for dist in ["mmd", "dtw", "wasserstein"]:
                base = df[(df["condition"] == "baseline") & (df["mode"] == mode)
                          & (df["level"] == level) & (df["distance"] == dist)]
                base_s = base.set_index("seed")[metric]
                for meth in methods:
                    treat = df[(df["condition"] == meth) & (df["mode"] == mode)
                              & (df["level"] == level) & (df["distance"] == dist)]
                    treat_s = treat.set_index("seed")[metric]
                    common = base_s.index.intersection(treat_s.index)
                    if len(common) >= 6:
                        b = base_s.loc[common].values
                        t = treat_s.loc[common].values
                        diff = t - b
                        if np.all(diff == 0):
                            W, p = 0.0, 1.0
                        else:
                            W, p = stats.wilcoxon(b, t, alternative="two-sided")
                        d = cliff_delta(t, b)
                    else:
                        W, p, d = np.nan, np.nan, np.nan
                    rows.append({
                        "mode": mode, "level": level, "distance": dist,
                        "comparison": f"{meth} vs baseline",
                        "W": W, "p": p, "cliff_d": d,
                        "effect": cliff_delta_label(d) if not np.isnan(d) else "",
                        "n_pairs": len(common) if isinstance(common, pd.Index) else 0,
                    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 8. Interaction: best condition depends on distance?
# ---------------------------------------------------------------------------
def interaction_analysis(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """For each (mode × level), find best condition per distance metric."""
    rows = []
    for mode in ["source_only", "target_only", "mixed"]:
        for level in ["in_domain", "out_domain"]:
            best_by_dist = {}
            for dist in ["mmd", "dtw", "wasserstein"]:
                means = df[(df["mode"] == mode) & (df["level"] == level)
                           & (df["distance"] == dist)].groupby("condition")[metric].mean()
                best_cond = means.idxmax()
                best_val = means.max()
                best_by_dist[dist] = (best_cond, best_val)
                rows.append({
                    "mode": mode, "level": level, "distance": dist,
                    "best_condition": best_cond,
                    f"best_{metric}": best_val,
                })
            # Check consistency
            conds = [v[0] for v in best_by_dist.values()]
            consistent = len(set(conds)) == 1
            rows[-1]["consistent_across_dists"] = consistent
            rows[-2]["consistent_across_dists"] = consistent
            rows[-3]["consistent_across_dists"] = consistent
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 9. Bonferroni correction helper
# ---------------------------------------------------------------------------
def bonferroni_summary(p_values: pd.Series, alpha: float = 0.05) -> dict:
    """Return Bonferroni-corrected results."""
    m = len(p_values)
    alpha_corr = alpha / m
    n_sig = (p_values < alpha_corr).sum()
    return {
        "n_tests": m,
        "alpha": alpha,
        "alpha_bonferroni": alpha_corr,
        "n_significant": n_sig,
        "pct_significant": 100 * n_sig / m if m > 0 else 0,
    }


# ---------------------------------------------------------------------------
# 10. Aggregate cross-condition rankings
# ---------------------------------------------------------------------------
def rank_conditions(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean rank of each condition across all (mode × level × distance) cells."""
    cells = []
    conditions = sorted(df["condition"].unique())
    for mode in ["source_only", "target_only", "mixed"]:
        for level in ["in_domain", "out_domain"]:
            for dist in ["mmd", "dtw", "wasserstein"]:
                means = {}
                for c in conditions:
                    vals = df[(df["condition"] == c) & (df["mode"] == mode)
                              & (df["level"] == level) & (df["distance"] == dist)][metric]
                    means[c] = vals.mean() if len(vals) else np.nan
                # Rank (higher is better for F2/AUC → rank 1 = best)
                sorted_conds = sorted(means, key=lambda k: means[k], reverse=True)
                ranks = {c: r + 1 for r, c in enumerate(sorted_conds)}
                cells.append(ranks)
    rank_df = pd.DataFrame(cells)
    summary = rank_df.mean().reset_index()
    summary.columns = ["condition", "mean_rank"]
    summary = summary.sort_values("mean_rank")
    return summary


# ===========================================================================
# Report generation
# ===========================================================================
MODE_LABEL = {
    "source_only": "Cross-domain",
    "target_only": "Within-domain",
    "mixed": "Mixed",
}
LEVEL_LABEL = {
    "in_domain": "In-domain",
    "out_domain": "Out-domain",
}


def generate_report(df: pd.DataFrame) -> str:
    """Generate a comprehensive markdown report."""
    lines = []
    w = lines.append

    w("# Experiment 2 — Domain Shift Statistical Analysis Report\n")
    w(f"**Generated**: auto  ")
    w(f"**Records**: {len(df)}  ")
    seeds_list = sorted(int(s) for s in df['seed'].unique())
    w(f"**Seeds**: {seeds_list} (n={len(seeds_list)})  ")
    w(f"**Conditions**: {sorted(df['condition'].unique())}  ")
    ratio_val = df['ratio'].dropna().mode().values[0] if not df['ratio'].dropna().empty else 'N/A'
    w(f"**Ratio**: {ratio_val} (for rus/smote/sw_smote; n/a for baseline)  ")
    w("")

    # ---- Experimental design description --------------------------------
    w("## 1. Experimental Design\n")
    w("### 1.1 Factor Structure\n")
    w("This experiment uses a **4-factor factorial design**:\n")
    w("| Factor | Levels | Description |")
    w("|--------|--------|-------------|")
    w("| Condition $C$ | baseline, rus, smote, sw_smote | Imbalance handling method |")
    w("| Mode $M$ | cross-domain, within-domain, mixed | Training data composition |")
    w("| Distance $D$ | MMD, DTW, Wasserstein | Domain distance metric for grouping |")
    w("| Level $L$ | in-domain, out-domain | Target domain proximity |")
    w("")
    w("Each observation is the evaluation metric (F2 or AUROC) for a specific")
    w("combination $(C, M, D, L, s)$ where $s$ is the random seed.\n")
    w("### 1.2 Statistical Model\n")
    w("The observed metric for configuration $(c, m, d, l, s)$:")
    w("")
    w("$$Y_{cmdls} = \\mu + \\alpha_c + \\beta_m + \\gamma_d + \\delta_l + "
      "(\\alpha\\beta)_{cm} + (\\alpha\\gamma)_{cd} + (\\alpha\\delta)_{cl} + \\varepsilon_{cmdls}$$")
    w("")
    w("where:")
    w("- $\\mu$: grand mean")
    w("- $\\alpha_c$: main effect of condition")
    w("- $\\beta_m$: main effect of mode")
    w("- $\\gamma_d$: main effect of distance metric")
    w("- $\\delta_l$: main effect of domain level")
    w("- $(\\alpha\\beta)_{cm}$, etc.: two-way interaction terms")
    w("- $\\varepsilon_{cmdls} \\sim \\mathcal{N}(0, \\sigma^2)$: residual error\n")
    w("Due to non-normality of classification metrics, we employ **non-parametric** tests.\n")

    n_seeds = df["seed"].nunique()

    # ==== F2 Analysis ====
    for metric, metric_label in [("f2", "F2-score"), ("auc", "AUROC")]:
        w(f"---\n## {'2' if metric == 'f2' else '3'}. Analysis: {metric_label}\n")

        # 2.1 Descriptive
        w(f"### {'2' if metric == 'f2' else '3'}.1 Descriptive Statistics\n")
        desc = descriptive_table(df, metric)

        # Pivot for readable table
        for mode in ["source_only", "target_only", "mixed"]:
            w(f"#### Mode: {MODE_LABEL[mode]}\n")
            w("| Condition | In-domain (mean±SD) | Out-domain (mean±SD) | Δ (out−in) |")
            w("|-----------|--------------------:|---------------------:|-----------:|")
            for cond in ["baseline", "rus", "smote", "sw_smote"]:
                in_row = desc[(desc["condition"] == cond) & (desc["mode"] == mode) & (desc["level"] == "in_domain")]
                out_row = desc[(desc["condition"] == cond) & (desc["mode"] == mode) & (desc["level"] == "out_domain")]
                if len(in_row) and len(out_row):
                    in_m, in_s = in_row["mean"].values[0], in_row["std"].values[0]
                    out_m, out_s = out_row["mean"].values[0], out_row["std"].values[0]
                    delta = out_m - in_m
                    w(f"| {cond} | {in_m:.4f}±{in_s:.4f} | {out_m:.4f}±{out_s:.4f} | {delta:+.4f} |")
            w("")

        # 2.2 Kruskal-Wallis
        sec = '2' if metric == 'f2' else '3'
        w(f"### {sec}.2 Kruskal-Wallis H-test (Condition effect)\n")
        w("Tests whether the distribution of " + metric_label + " differs across the 4 conditions.\n")
        w("$$H = \\frac{12}{N(N+1)} \\sum_{i=1}^{k} \\frac{R_i^2}{n_i} - 3(N+1)$$\n")
        w("where $R_i$ is the sum of ranks in group $i$, $n_i$ the group size, $N = \\sum n_i$.\n")
        kw = kruskal_wallis_per_cell(df, metric)
        w("| Mode | Level | Distance | H | p-value | Sig (α=0.05) |")
        w("|------|-------|----------|--:|--------:|:------------:|")
        for _, r in kw.iterrows():
            sig = "✓" if r["p"] < 0.05 else ""
            w(f"| {MODE_LABEL.get(r['mode'], r['mode'])} | {LEVEL_LABEL.get(r['level'], r['level'])} "
              f"| {r['distance'].upper()} | {r['H']:.3f} | {r['p']:.4f} | {sig} |")
        n_sig_kw = (kw["p"] < 0.05).sum()
        bon = bonferroni_summary(kw["p"])
        w(f"\n**Summary**: {n_sig_kw}/{len(kw)} cells significant at α=0.05; "
          f"{bon['n_significant']}/{bon['n_tests']} after Bonferroni correction "
          f"(α'={bon['alpha_bonferroni']:.4f}).\n")

        # 2.3 Pairwise Mann-Whitney U
        w(f"### {sec}.3 Pairwise Comparisons (Mann-Whitney U)\n")
        w("Baseline vs each method, testing:\n")
        w("$$H_0: F_{\\text{baseline}}(x) = F_{\\text{method}}(x)$$")
        w("$$H_1: F_{\\text{baseline}}(x) \\neq F_{\\text{method}}(x)$$\n")
        pw = pairwise_vs_baseline(df, metric)
        # Count total tests for Bonferroni
        bon_pw = bonferroni_summary(pw["p"].dropna())

        w("| Comparison | Mode | Level | Distance | U | p | Cliff's δ | Effect | Mean(method) | Mean(baseline) |")
        w("|------------|------|-------|----------|--:|--:|----------:|:------:|-------------:|---------------:|")
        for _, r in pw.iterrows():
            sig_mark = " *" if r["p"] < bon_pw["alpha_bonferroni"] else ""
            w(f"| {r['comparison']} | {MODE_LABEL.get(r['mode'], r['mode'])} "
              f"| {LEVEL_LABEL.get(r['level'], r['level'])} "
              f"| {r['distance'].upper()} | {r['U']:.1f} | {r['p']:.4f}{sig_mark} "
              f"| {r['cliff_d']:+.3f} | {r['effect']} "
              f"| {r['mean_treat']:.4f} | {r['mean_base']:.4f} |")
        w(f"\n**Bonferroni threshold**: α'={bon_pw['alpha_bonferroni']:.5f} "
          f"(m={bon_pw['n_tests']}). "
          f"**{bon_pw['n_significant']}** comparisons significant after correction.\n")

        # 2.4 Wilcoxon signed-rank (paired)
        w(f"### {sec}.4 Paired Comparison (Wilcoxon Signed-Rank)\n")
        w("Paired by seed — more powerful than Mann-Whitney when seeds are shared.\n")
        w("$$W = \\sum_{i=1}^{n} \\text{sign}(d_i) \\cdot R_i, \\quad d_i = Y_{\\text{method},i} - Y_{\\text{baseline},i}$$\n")
        wsr = wilcoxon_paired(df, metric)
        bon_wsr = bonferroni_summary(wsr["p"].dropna())

        w("| Comparison | Mode | Level | Distance | W | p | Cliff's δ | Effect | n |")
        w("|------------|------|-------|----------|--:|--:|----------:|:------:|--:|")
        for _, r in wsr.iterrows():
            if np.isnan(r["p"]):
                continue
            sig_mark = " *" if r["p"] < bon_wsr["alpha_bonferroni"] else ""
            w(f"| {r['comparison']} | {MODE_LABEL.get(r['mode'], r['mode'])} "
              f"| {LEVEL_LABEL.get(r['level'], r['level'])} "
              f"| {r['distance'].upper()} "
              f"| {r['W']:.1f} | {r['p']:.4f}{sig_mark} "
              f"| {r['cliff_d']:+.3f} | {r['effect']} | {r['n_pairs']} |")
        w(f"\n**Bonferroni threshold**: α'={bon_wsr['alpha_bonferroni']:.5f} "
          f"(m={bon_wsr['n_tests']}). "
          f"**{bon_wsr['n_significant']}** comparisons significant after correction.\n")
        min_p_wilcoxon = 1 / (2 ** (n_seeds - 1))
        w(f"**Note**: With n={n_seeds} paired observations, the minimum achievable "
          f"p-value for Wilcoxon signed-rank is $p_{{\\min}} = 1/2^{{{n_seeds}-1}} "
          f"= {min_p_wilcoxon:.6f}$. "
          f"When $p_{{\\min}} > \\alpha'$, no comparison can reach Bonferroni significance "
          f"regardless of effect magnitude. This is a **floor effect** of small sample size, "
          f"not evidence of no difference.\n")

        # 2.5 Friedman test
        w(f"### {sec}.5 Friedman Test (Repeated-Measures)\n")
        w("Seeds serve as blocks (subjects). Tests whether at least one condition differs.\n")
        w("$$\\chi_F^2 = \\frac{12}{bk(k+1)} \\sum_{j=1}^{k} R_j^2 - 3b(k+1)$$\n")
        w("where $b$ = number of blocks (seeds), $k$ = number of conditions, $R_j$ = rank sum.\n")
        fri = friedman_across_conditions(df, metric)
        w("| Mode | Level | Distance | χ² | p-value | n_seeds | Sig |")
        w("|------|-------|----------|----|--------:|--------:|:---:|")
        for _, r in fri.iterrows():
            sig = "✓" if r["p"] < 0.05 else ""
            if np.isnan(r["chi2"]):
                w(f"| {MODE_LABEL.get(r['mode'], r['mode'])} | {LEVEL_LABEL.get(r['level'], r['level'])} "
                  f"| {r['distance'].upper()} | — | — | {r['n_seeds']} | |")
            else:
                w(f"| {MODE_LABEL.get(r['mode'], r['mode'])} | {LEVEL_LABEL.get(r['level'], r['level'])} "
                  f"| {r['distance'].upper()} | {r['chi2']:.3f} | {r['p']:.4f} | {int(r['n_seeds'])} | {sig} |")
        w("")

        # 2.6 Domain gap
        w(f"### {sec}.6 Domain Gap Analysis\n")
        w("Domain gap: $\\Delta = Y_{\\text{out-domain}} - Y_{\\text{in-domain}}$")
        w(" (paired by seed).\n")
        w("Negative Δ indicates performance degrades in out-domain (expected for domain shift).\n")
        dg = domain_gap_analysis(df, metric)
        w("| Mode | Distance | Condition | Mean Δ | SD Δ | n | KW p (across conds) |")
        w("|------|----------|-----------|-------:|-----:|--:|--------------------:|")
        for _, r in dg.iterrows():
            w(f"| {MODE_LABEL.get(r['mode'], r['mode'])} | {r['distance'].upper()} "
              f"| {r['condition']} | {r['mean_gap']:+.4f} | {r['std_gap']:.4f} "
              f"| {r['n_pairs']} | {r['KW_p']:.4f} |")
        w("")

        # 2.7 Interaction
        w(f"### {sec}.7 Condition × Distance Interaction\n")
        w("Does the best-performing condition depend on which distance metric is used?\n")
        inter = interaction_analysis(df, metric)
        w(f"| Mode | Level | Distance | Best Condition | Mean {metric_label} | Consistent? |")
        w("|------|-------|----------|----------------|---:|:-----------:|")
        for _, r in inter.iterrows():
            consist = "✓" if r.get("consistent_across_dists", True) else "✗"
            w(f"| {MODE_LABEL.get(r['mode'], r['mode'])} | {LEVEL_LABEL.get(r['level'], r['level'])} "
              f"| {r['distance'].upper()} | {r['best_condition']} "
              f"| {r[f'best_{metric}']:.4f} | {consist} |")
        w("")

        # 2.8 Condition ranking
        w(f"### {sec}.8 Overall Condition Ranking\n")
        w("Mean rank across all 18 cells (3 modes × 2 levels × 3 distances). Rank 1 = best.\n")
        rank = rank_conditions(df, metric)
        w("| Rank | Condition | Mean Rank |")
        w("|:----:|-----------|----------:|")
        for i, (_, r) in enumerate(rank.iterrows()):
            w(f"| {i+1} | {r['condition']} | {r['mean_rank']:.2f} |")
        w("")

    # ==== Section 4: Cross-metric synthesis ====
    w("---\n## 4. Cross-Metric Synthesis\n")

    # F2 ranking
    rank_f2 = rank_conditions(df, "f2")
    rank_auc = rank_conditions(df, "auc")

    w("### 4.1 Overall Rankings Comparison\n")
    w("| Condition | Mean Rank (F2) | Mean Rank (AUROC) | Average |")
    w("|-----------|:--------------:|:-----------------:|:-------:|")
    for cond in ["baseline", "rus", "smote", "sw_smote"]:
        r_f2 = rank_f2[rank_f2["condition"] == cond]["mean_rank"].values[0]
        r_auc = rank_auc[rank_auc["condition"] == cond]["mean_rank"].values[0]
        avg = (r_f2 + r_auc) / 2
        w(f"| {cond} | {r_f2:.2f} | {r_auc:.2f} | {avg:.2f} |")
    w("")

    # 4.2 Key findings
    w("### 4.2 Summary of Significant Findings\n")

    kw_f2 = kruskal_wallis_per_cell(df, "f2")
    kw_auc = kruskal_wallis_per_cell(df, "auc")
    pw_f2 = pairwise_vs_baseline(df, "f2")
    pw_auc = pairwise_vs_baseline(df, "auc")
    wsr_f2 = wilcoxon_paired(df, "f2")
    wsr_auc = wilcoxon_paired(df, "auc")

    w("| Test | Metric | Sig/Total (raw α=0.05) | Sig/Total (Bonferroni) |")
    w("|------|--------|:----------------------:|:----------------------:|")
    for test_name, test_df, metric_name in [
        ("Kruskal-Wallis", kw_f2, "F2"),
        ("Kruskal-Wallis", kw_auc, "AUROC"),
        ("Mann-Whitney U", pw_f2, "F2"),
        ("Mann-Whitney U", pw_auc, "AUROC"),
        ("Wilcoxon SR", wsr_f2, "F2"),
        ("Wilcoxon SR", wsr_auc, "AUROC"),
    ]:
        p = test_df["p"].dropna()
        raw = (p < 0.05).sum()
        bon = bonferroni_summary(p)
        w(f"| {test_name} | {metric_name} | {raw}/{len(p)} | {bon['n_significant']}/{bon['n_tests']} |")
    w("")

    # 4.3 Effect size summary
    w("### 4.3 Effect Size Summary (Cliff's δ)\n")
    w("Distribution of effect sizes across all pairwise comparisons:\n")
    for metric_name, pw_df in [("F2", pw_f2), ("AUROC", pw_auc)]:
        w(f"**{metric_name}**:")
        for eff in ["negligible", "small", "medium", "large"]:
            n = (pw_df["effect"] == eff).sum()
            pct = 100 * n / len(pw_df)
            w(f"  - {eff}: {n}/{len(pw_df)} ({pct:.0f}%)")
        w("")

    # ==== Section 5: Power analysis ====
    w("---\n## 5. Statistical Power Analysis\n")
    w("### 5.1 Current Design Power\n")
    n_seeds = df["seed"].nunique()
    w(f"- Current number of seeds: **n={n_seeds}**")
    w(f"- Unique seeds: {seeds_list}")
    w(f"- Cells per condition: 3 modes × 2 levels × 3 distances = 18")
    w(f"- Observations per cell per condition: ~{n_seeds} seeds")
    w("")
    w("### 5.2 Power Considerations\n")
    w("For Wilcoxon signed-rank test with n paired observations:\n")
    w("$$\\text{Power} = P(\\text{reject } H_0 \\mid H_1 \\text{ true})$$\n")
    w("With $n = " + str(n_seeds) + "$ seeds, the minimum detectable Cliff's δ at α=0.05 is approximately:\n")
    w("$$|\\delta_{\\min}| \\approx \\frac{z_{\\alpha/2}}{\\sqrt{n}} \\approx "
      f"\\frac{{1.96}}{{\\sqrt{{{n_seeds}}}}} \\approx {1.96/np.sqrt(n_seeds):.3f}$$\n")
    w("This means only **large** effects (|δ| > 0.474) are reliably detectable.")
    w("Medium effects require n ≥ 20-30 seeds; small effects n ≥ 50+.\n")

    # ==== Section 6: Recommendations ====
    w("---\n## 6. Proposed Additional Experiments\n")

    # Determine what we've found
    pw_all = pd.concat([pw_f2, pw_auc])
    large_effects = pw_all[pw_all["effect"] == "large"]
    medium_effects = pw_all[pw_all["effect"] == "medium"]

    w("### 6.1 Experiment A: Increased Seed Count\n")
    w("**Rationale**: Current $n=" + str(n_seeds) + "$ seeds may lack power for medium/small effects.\n")
    w("**Proposal**: Increase to $n \\geq 20$ seeds to detect medium effects (|δ| ≈ 0.33).\n")
    w("Required sample size for Wilcoxon signed-rank (approximation):")
    w("$$n \\geq \\left(\\frac{z_{\\alpha'/2} + z_\\beta}{\\delta}\\right)^2$$")
    w("")
    w("For Bonferroni-adjusted α' and power 0.80:\n")
    n_tests_total = len(pw_f2) + len(pw_auc)
    alpha_bon = 0.05 / n_tests_total
    z_alpha = stats.norm.ppf(1 - alpha_bon / 2)
    z_beta = 0.842  # power = 0.80
    for delta_target in [0.33, 0.474]:
        n_needed = int(np.ceil(((z_alpha + z_beta) / delta_target) ** 2))
        w(f"- To detect |δ| ≥ {delta_target}: $n \\geq {n_needed}$ seeds")
    w("")

    w("### 6.2 Experiment B: Ratio Sensitivity Analysis\n")
    w("**Rationale**: Current primary analysis fixes ratio=0.1. Different ratios may yield different rankings.\n")
    w("**Proposal**: Repeat full analysis with ratio=0.5 and compare rankings.\n")
    w("- If rankings differ: ratio is a significant moderator → report both\n")
    w("- If rankings agree: robust finding, ratio is secondary\n")

    w("### 6.3 Experiment C: Cross-Validation of Distance Metric Groupings\n")
    w("**Rationale**: Domain groups are defined by distance metric thresholds. ")
    w("Different threshold choices could change which subjects are in/out-domain.\n")
    w("**Proposal**: Vary the distance threshold (e.g., percentile-based: 25th, 50th, 75th) ")
    w("and check robustness of findings.\n")

    w("### 6.4 Experiment D: Permutation Test for Global Null\n")
    w("**Rationale**: Multiple testing correction is conservative. A permutation-based ")
    w("global test can provide an exact p-value.\n")
    w("**Proposal**:")
    w("1. For each seed, permute condition labels across all cells")
    w("2. Compute a global test statistic (e.g., sum of |Δ| across cells)")
    w("3. Repeat 10,000+ times")
    w("4. Compare observed statistic to null distribution\n")
    w("$$p_{\\text{perm}} = \\frac{1}{B}\\sum_{b=1}^{B} \\mathbb{1}[T^{(b)} \\geq T_{\\text{obs}}]$$\n")

    w("### 6.5 Experiment E: Bootstrap Confidence Intervals\n")
    w("**Rationale**: Provide interval estimates rather than just point estimates / p-values.\n")
    w("**Proposal**: For each (condition × mode × level) cell:\n")
    w("1. Bootstrap resample seeds B=10,000 times")
    w("2. Compute 95% BCa confidence interval for mean F2 and AUROC")
    w("3. Compare overlap of CIs between conditions\n")
    w("$$\\text{CI}_{95\\%} = \\left[\\hat{\\theta}^*_{(\\alpha/2)}, \\hat{\\theta}^*_{(1-\\alpha/2)}\\right]$$\n")

    w("### 6.6 Experiment F: Cross-Domain Degradation Ratio\n")
    w("**Rationale**: Absolute domain gap Δ depends on baseline performance level. ")
    w("A relative measure normalizes this.\n")
    w("**Proposal**: Compute degradation ratio:\n")
    w("$$\\rho = \\frac{Y_{\\text{out-domain}}}{Y_{\\text{in-domain}}}$$\n")
    w("$\\rho < 1$ indicates degradation; $\\rho = 1$ means no domain shift effect; $\\rho > 1$ means improvement.\n")
    w("Compare $\\rho$ across conditions to identify which method best preserves performance under domain shift.\n")

    return "\n".join(lines)


# ===========================================================================
# Ratio sensitivity comparison
# ===========================================================================
def ratio_sensitivity_report(df_r01: pd.DataFrame, df_r05: pd.DataFrame) -> str:
    """Compare rankings between ratio=0.1 and ratio=0.5."""
    lines = []
    w = lines.append

    w("# Experiment B — Ratio Sensitivity Analysis\n")
    w("Compare condition rankings between ratio=0.1 and ratio=0.5.\n")
    w("If rankings are consistent, the findings are robust to ratio choice.\n")

    # 1. Overall rankings comparison
    w("## 1. Overall Condition Rankings by Ratio\n")
    for metric, mlabel in [("f2", "F2-score"), ("auc", "AUROC")]:
        rank_01 = rank_conditions(df_r01, metric)
        rank_05 = rank_conditions(df_r05, metric)
        w(f"### {mlabel}\n")
        w(f"| Condition | Mean Rank (r=0.1) | Mean Rank (r=0.5) | Δ Rank |")
        w(f"|-----------|:-----------------:|:-----------------:|:------:|")
        for cond in ["baseline", "rus", "smote", "sw_smote"]:
            r01 = rank_01[rank_01["condition"] == cond]["mean_rank"].values[0]
            r05 = rank_05[rank_05["condition"] == cond]["mean_rank"].values[0]
            delta_r = r05 - r01
            w(f"| {cond} | {r01:.2f} | {r05:.2f} | {delta_r:+.2f} |")
        # Spearman correlation of rankings
        r01_ranks = rank_01.set_index("condition")["mean_rank"]
        r05_ranks = rank_05.set_index("condition")["mean_rank"]
        common = r01_ranks.index.intersection(r05_ranks.index)
        rho, p = stats.spearmanr(r01_ranks.loc[common], r05_ranks.loc[common])
        w(f"\nSpearman rank correlation: $\\rho_s = {rho:.3f}$ (p={p:.4f})\n")

    # 2. Cell-level comparison: best condition per cell
    w("## 2. Best Condition per Cell\n")
    w("Does the best-performing condition change when ratio changes?\n")
    for metric, mlabel in [("f2", "F2-score"), ("auc", "AUROC")]:
        w(f"### {mlabel}\n")
        w("| Mode | Level | Distance | Best (r=0.1) | Best (r=0.5) | Match? |")
        w("|------|-------|----------|:------------:|:------------:|:------:|")
        n_match = 0
        n_total = 0
        for mode in ["source_only", "target_only", "mixed"]:
            for level in ["in_domain", "out_domain"]:
                for dist in ["mmd", "dtw", "wasserstein"]:
                    m01 = df_r01[(df_r01["mode"]==mode)&(df_r01["level"]==level)
                                 &(df_r01["distance"]==dist)].groupby("condition")[metric].mean()
                    m05 = df_r05[(df_r05["mode"]==mode)&(df_r05["level"]==level)
                                 &(df_r05["distance"]==dist)].groupby("condition")[metric].mean()
                    best01 = m01.idxmax() if len(m01) else "—"
                    best05 = m05.idxmax() if len(m05) else "—"
                    match = "✓" if best01 == best05 else "✗"
                    if best01 == best05:
                        n_match += 1
                    n_total += 1
                    w(f"| {MODE_LABEL.get(mode,mode)} | {LEVEL_LABEL.get(level,level)} "
                      f"| {dist.upper()} | {best01} | {best05} | {match} |")
        w(f"\n**Agreement**: {n_match}/{n_total} cells ({100*n_match/n_total:.0f}%)\n")

    # 3. Pairwise effect size comparison
    w("## 3. Effect Size Stability\n")
    w("Cliff's δ (baseline vs method) comparison between ratios.\n")
    for metric, mlabel in [("f2", "F2-score"), ("auc", "AUROC")]:
        pw01 = pairwise_vs_baseline(df_r01, metric)
        pw05 = pairwise_vs_baseline(df_r05, metric)
        w(f"### {mlabel}\n")
        w("| Comparison | Mode | Level | Dist | δ (r=0.1) | δ (r=0.5) | Δδ | Same direction? |")
        w("|------------|------|-------|------|----------:|----------:|---:|:---------------:|")
        n_same_dir = 0
        n_compared = 0
        for i in range(len(pw01)):
            r01 = pw01.iloc[i]
            r05 = pw05.iloc[i]
            d01 = r01["cliff_d"]
            d05 = r05["cliff_d"]
            if np.isnan(d01) or np.isnan(d05):
                continue
            dd = d05 - d01
            same = "✓" if (d01 * d05 > 0) or (d01 == 0 and d05 == 0) else "✗"
            if same == "✓":
                n_same_dir += 1
            n_compared += 1
            w(f"| {r01['comparison']} | {MODE_LABEL.get(r01['mode'],r01['mode'])} "
              f"| {LEVEL_LABEL.get(r01['level'],r01['level'])} "
              f"| {r01['distance'].upper()} "
              f"| {d01:+.3f} | {d05:+.3f} | {dd:+.3f} | {same} |")
        w(f"\n**Directional agreement**: {n_same_dir}/{n_compared} "
          f"({100*n_same_dir/n_compared:.0f}%)\n")

    # 4. Descriptive comparison
    w("## 4. Mean Performance by Ratio\n")
    for metric, mlabel in [("f2", "F2-score"), ("auc", "AUROC")]:
        w(f"### {mlabel}\n")
        w("| Condition | Mode | Mean (r=0.1) | Mean (r=0.5) | Δ |")
        w("|-----------|------|-------------:|-------------:|--:|")
        for cond in ["baseline", "rus", "smote", "sw_smote"]:
            for mode in ["source_only", "target_only", "mixed"]:
                v01 = df_r01[(df_r01["condition"]==cond)&(df_r01["mode"]==mode)][metric].mean()
                v05 = df_r05[(df_r05["condition"]==cond)&(df_r05["mode"]==mode)][metric].mean()
                delta = v05 - v01
                w(f"| {cond} | {MODE_LABEL.get(mode,mode)} "
                  f"| {v01:.4f} | {v05:.4f} | {delta:+.4f} |")
        w("")

    # 5. Conclusion
    w("## 5. Conclusion\n")
    r01_f2 = rank_conditions(df_r01, "f2").set_index("condition")["mean_rank"]
    r05_f2 = rank_conditions(df_r05, "f2").set_index("condition")["mean_rank"]
    r01_auc = rank_conditions(df_r01, "auc").set_index("condition")["mean_rank"]
    r05_auc = rank_conditions(df_r05, "auc").set_index("condition")["mean_rank"]

    rho_f2, _ = stats.spearmanr(r01_f2, r05_f2)
    rho_auc, _ = stats.spearmanr(r01_auc, r05_auc)

    w(f"- Spearman rank correlation between r=0.1 and r=0.5: "
      f"F2 $\\rho_s={rho_f2:.3f}$, AUROC $\\rho_s={rho_auc:.3f}$")
    if rho_f2 > 0.8 and rho_auc > 0.8:
        w("- **Conclusion**: Rankings are **highly consistent** across ratios. "
          "The findings are robust to ratio choice.")
    elif rho_f2 > 0.6 or rho_auc > 0.6:
        w("- **Conclusion**: Rankings show **moderate consistency**. "
          "Ratio is a partial moderator; report both ratios.")
    else:
        w("- **Conclusion**: Rankings **differ substantially** between ratios. "
          "Ratio is a significant moderator.")
    w("")

    return "\n".join(lines)


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("=" * 60)
    print("Experiment 2 — Comprehensive Statistical Analysis")
    print("=" * 60)

    # --- Primary analysis (ratio=0.1) ---
    df_r01 = load_data(ratio=0.1)
    print(f"Loaded {len(df_r01)} records (ratio=0.1)")
    print(f"  Conditions: {sorted(df_r01['condition'].unique())}")
    seeds = sorted(int(s) for s in df_r01['seed'].unique())
    print(f"  Seeds: {seeds} (n={len(seeds)})")

    report = generate_report(df_r01)
    out_path = REPORT_DIR / "statistical_report.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"  Report saved to: {out_path}")

    # --- Experiment B: Ratio sensitivity (ratio=0.5) ---
    df_r05 = load_data(ratio=0.5)
    print(f"\nLoaded {len(df_r05)} records (ratio=0.5)")

    report_r05 = generate_report(df_r05)
    out_r05 = REPORT_DIR / "statistical_report_ratio05.md"
    out_r05.write_text(report_r05, encoding="utf-8")
    print(f"  Report (r=0.5) saved to: {out_r05}")

    # --- Ratio sensitivity comparison ---
    sensitivity = ratio_sensitivity_report(df_r01, df_r05)
    out_sens = REPORT_DIR / "ratio_sensitivity_report.md"
    out_sens.write_text(sensitivity, encoding="utf-8")
    print(f"  Sensitivity report saved to: {out_sens}")

    print("=" * 60)


if __name__ == "__main__":
    main()
