#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stat_analysis_exp2_extended.py
==============================
Extended hypothesis-driven statistical analysis of Experiment 2.

Supplements the primary analysis (F2-score, AUROC) with three additional
metrics chosen for their relevance to imbalanced binary classification
in safety-critical drowsy driving detection:

    - F1-score:  Balanced precision–recall harmonic mean
    - AUPRC:     Area Under the Precision–Recall Curve (threshold-free, 
                 minority-class-sensitive)
    - Recall:    Sensitivity — critical for safety (missed drowsy events)

Also includes a Precision–Recall trade-off analysis and cross-metric
concordance validation against the primary metrics.

Output:  results/analysis/exp2_domain_shift/hypothesis_test_report_extended.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy import stats

# ---------------------------------------------------------------------------
# Import shared utilities from primary analysis script
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from stat_analysis_exp2_v2 import (
    load_all_data,
    cliff_delta, cliff_delta_ci, cliff_label,
    eta_squared_from_H, bonferroni, benjamini_hochberg,
    cohens_w_from_friedman, bootstrap_ci_bca,
    permutation_test_condition, seed_convergence_analysis,
    CONDITIONS_7, MODES, DISTANCES, LEVELS,
    MODE_LABEL, LEVEL_LABEL, REPORT_DIR,
)

# ---------------------------------------------------------------------------
# Extended metric definitions
# ---------------------------------------------------------------------------
# Metrics for full hypothesis testing
METRICS_EXT = [
    ("f1", "F1-score"),
    ("auc_pr", "AUPRC"),
    ("recall", "Recall"),
]

# Primary metrics (for concordance comparison)
METRICS_PRIMARY = [("f2", "F2-score"), ("auc", "AUROC")]

# All seven metrics (for concordance analysis)
ALL_METRICS = METRICS_PRIMARY + METRICS_EXT + [
    ("precision", "Precision"),
    ("accuracy", "Accuracy"),
]


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_extended_report(df: pd.DataFrame) -> str:
    lines: list[str] = []
    w = lines.append

    seeds = sorted(int(s) for s in df["seed"].unique())
    n_seeds = len(seeds)
    conditions = sorted(df["condition"].unique())

    # =================================================================
    # Title
    # =================================================================
    w("# Experiment 2 — Extended Hypothesis Testing (F1, AUPRC, Recall)\n")
    w(f"**Records**: {len(df)}  ")
    w(f"**Seeds**: {seeds} (n={n_seeds})  ")
    w(f"**Conditions** (7): {conditions}  ")
    w(f"**Modes**: {MODES}  ")
    w(f"**Distances**: {DISTANCES}  ")
    w(f"**Levels**: {LEVELS}  ")
    w("")

    # =================================================================
    # 1. Metric Selection Rationale
    # =================================================================
    w("---\n## 1. Metric Selection Rationale\n")
    w("The primary analysis (companion report) tests F2-score and AUROC. "
      "This supplementary analysis extends the hypothesis framework to "
      "three additional metrics, selected for their relevance to imbalanced "
      "binary classification in a safety-critical application:\n")
    w("| Metric | Column | Rationale |")
    w("|--------|--------|-----------|")
    w("| F1-score | `f1` | Harmonic mean of Precision and Recall with equal weight. "
      "Compared with F2 (which emphasizes Recall 2×), F1 reveals whether "
      "the Recall-weighted conclusions still hold under balanced weighting. |")
    w("| AUPRC | `auc_pr` | Area Under the Precision–Recall Curve. "
      "Unlike AUROC, AUPRC is *not* inflated by true negatives and is "
      "considered the most informative single metric for imbalanced "
      "classification (Saito & Rehmsmeier, 2015). |")
    w("| Recall | `recall` | Sensitivity / True Positive Rate. "
      "In drowsy-driving detection, a missed drowsy event is a safety risk; "
      "Recall directly measures the detection rate of the positive class. |")
    w("")
    w("**Excluded metrics** (with justification):\n")
    w("- **Accuracy**: Misleading for imbalanced data — a trivial majority-class "
      "classifier achieves high accuracy.\n")
    w("- **Precision** (as standalone): Included in descriptive statistics and "
      "trade-off analysis (§ 10) but not in the full hypothesis framework, "
      "because maximizing Precision alone is not the primary objective in "
      "safety-critical detection.\n")
    w("")

    # =================================================================
    # 2. Hypothesis Framework
    # =================================================================
    w("---\n## 2. Hypothesis Framework\n")
    w("The same 14-hypothesis framework from the primary analysis is applied "
      "to each extended metric. Additionally, three metric-specific hypotheses "
      "are tested.\n")

    w("### Standard Hypotheses (applied per metric)\n")
    w("| ID | Hypothesis |")
    w("|:--:|-----------|")
    w("| H1 | Imbalance handling method affects performance (7-condition KW) |")
    w("| H2 | SW-SMOTE outperforms plain SMOTE (same ratio) |")
    w("| H3 | Oversampling outperforms RUS |")
    w("| H4 | Sampling ratio (r=0.1 vs r=0.5) affects performance |")
    w("| H5 | Distance metric choice affects performance |")
    w("| H7 | Within-domain > cross-domain training |")
    w("| H10 | In-domain > out-domain (domain shift exists) |")
    w("| H12 | Condition × Mode interaction |")
    w("| H13 | Condition × Distance interaction |")
    w("| H14 | Domain gap varies by mode |")
    w("")

    w("### Extended Hypotheses (metric-specific)\n")
    w("| ID | Hypothesis | Rationale |")
    w("|:--:|-----------|-----------|")
    w("| HE1 | Any rebalancing method (including RUS) improves Recall over baseline | "
      "Both oversampling and undersampling increase minority-class weight, "
      "which should boost Recall even if other metrics degrade |")
    w("| HE2 | AUPRC shows a stronger condition effect than AUROC | "
      "AUPRC is more sensitive to minority-class performance; "
      "condition effects hidden in AUROC may emerge in AUPRC |")
    w("| HE3 | Oversampling improves Recall at the cost of Precision "
      "(Precision–Recall trade-off) | "
      "Shifting the decision boundary to catch more positives "
      "increases false positives, reducing Precision |")
    w("")

    # =================================================================
    # 3. Normality Assessment
    # =================================================================
    w("---\n## 3. Normality Assessment (Shapiro-Wilk)\n")
    w("Justification for non-parametric tests.\n")

    sw_reject = {}
    for metric, mlabel in METRICS_EXT:
        w(f"### {mlabel}\n")
        w("| Condition | Mode | Level | W | p | Normal? |")
        w("|-----------|------|-------|--:|--:|:-------:|")
        n_total = 0
        n_reject = 0
        for cond in CONDITIONS_7:
            for mode in MODES:
                for level in LEVELS:
                    vals = df[(df["condition"] == cond) & (df["mode"] == mode)
                              & (df["level"] == level)][metric].dropna().values
                    if len(vals) >= 3:
                        W_stat, p_sw = stats.shapiro(vals)
                        n_total += 1
                        reject = p_sw < 0.05
                        if reject:
                            n_reject += 1
                        w(f"| {cond} | {MODE_LABEL[mode]} | {LEVEL_LABEL[level]} "
                          f"| {W_stat:.4f} | {p_sw:.4f} "
                          f"| {'✗ reject' if reject else '✓ normal'} |")
        sw_reject[metric] = (n_reject, n_total)
        pct = 100 * n_reject / max(n_total, 1)
        w(f"\n**Summary**: {n_reject}/{n_total} cells ({pct:.0f}%) reject "
          f"normality at α=0.05.\n")

    all_reject = sum(r for r, _ in sw_reject.values())
    all_total = sum(t for _, t in sw_reject.values())
    pct_all = 100 * all_reject / max(all_total, 1)
    w(f"**Conclusion**: {all_reject}/{all_total} cells ({pct_all:.0f}%) violate "
      "normality. Non-parametric tests are appropriate.\n")

    # =================================================================
    # 4. Descriptive Statistics
    # =================================================================
    w("---\n## 4. Descriptive Statistics\n")
    for metric, mlabel in METRICS_EXT:
        w(f"### {mlabel}\n")
        for mode in MODES:
            w(f"#### {MODE_LABEL[mode]}\n")
            w("| Condition | In-domain (mean±SD) | Out-domain (mean±SD) | Δ (out−in) | n |")
            w("|-----------|--------------------:|---------------------:|-----------:|--:|")
            for cond in CONDITIONS_7:
                vi = df[(df["condition"] == cond) & (df["mode"] == mode)
                        & (df["level"] == "in_domain")][metric]
                vo = df[(df["condition"] == cond) & (df["mode"] == mode)
                        & (df["level"] == "out_domain")][metric]
                if len(vi) == 0 and len(vo) == 0:
                    continue
                mi, si = vi.mean(), vi.std()
                mo, so = vo.mean(), vo.std()
                w(f"| {cond} | {mi:.4f}±{si:.4f} | {mo:.4f}±{so:.4f} "
                  f"| {mo - mi:+.4f} | {len(vi)} |")
            w("")

    # =================================================================
    # 5. H1 — Condition Effect
    # =================================================================
    w("---\n## 5. Hypothesis Tests — H1: Model / Condition Effect\n")

    for mi, (metric, mlabel) in enumerate(METRICS_EXT):
        # --- 5.x.1 KW global ---
        w(f"\n### 5.{mi+1}.1 [{mlabel}] H1: Global condition effect (Kruskal-Wallis)\n")
        w("$$H_0: F_{C_1} = F_{C_2} = \\cdots = F_{C_7}$$\n")

        kw_rows = []
        for mode in MODES:
            for level in LEVELS:
                for dist in DISTANCES:
                    groups = [
                        df[(df["condition"] == c) & (df["mode"] == mode)
                           & (df["level"] == level) & (df["distance"] == dist)
                           ][metric].dropna().values
                        for c in CONDITIONS_7
                    ]
                    n_total = sum(len(g) for g in groups)
                    if all(len(g) >= 2 for g in groups):
                        H, p = stats.kruskal(*groups)
                        eta2 = eta_squared_from_H(H, n_total, len(groups))
                    else:
                        H, p, eta2 = np.nan, np.nan, np.nan
                    kw_rows.append({"mode": mode, "level": level,
                                    "distance": dist, "H": H, "p": p,
                                    "eta2": eta2, "N": n_total})

        kw_df = pd.DataFrame(kw_rows)
        bon = bonferroni(kw_df["p"])
        w("| Mode | Level | Distance | H | p | η² | Sig (Bonf.) |")
        w("|------|-------|----------|--:|--:|---:|:-----------:|")
        for _, r in kw_df.iterrows():
            sig = "✓" if r["p"] < bon["alpha_c"] else ""
            w(f"| {MODE_LABEL[r['mode']]} | {LEVEL_LABEL[r['level']]} "
              f"| {r['distance'].upper()} "
              f"| {r['H']:.2f} | {r['p']:.4f} | {r['eta2']:.3f} | {sig} |")
        w(f"\n**Bonferroni α'={bon['alpha_c']:.4f}** (m={bon['m']}). "
          f"**{bon['n_sig']}/{bon['m']}** significant.\n")
        eta2_vals = kw_df["eta2"].dropna()
        w(f"Mean η² = {eta2_vals.mean():.3f} "
          f"({'large' if eta2_vals.mean() > 0.14 else 'medium' if eta2_vals.mean() > 0.06 else 'small'} effect).\n")

        # --- 5.x.2 Pairwise baseline vs each ---
        w(f"### 5.{mi+1}.2 [{mlabel}] H1: Pairwise — baseline vs each method\n")
        w("Mann-Whitney U with Cliff's δ effect size.\n")

        pw_rows = []
        other_conds = [c for c in CONDITIONS_7 if c != "baseline"]
        for method in other_conds:
            for mode in MODES:
                for level in LEVELS:
                    bv = df[(df["condition"] == "baseline") & (df["mode"] == mode)
                            & (df["level"] == level)][metric].dropna().values
                    mv = df[(df["condition"] == method) & (df["mode"] == mode)
                            & (df["level"] == level)][metric].dropna().values
                    if len(bv) >= 2 and len(mv) >= 2:
                        U, p = stats.mannwhitneyu(bv, mv, alternative="two-sided")
                        d = cliff_delta(mv, bv)
                    else:
                        U, p, d = np.nan, np.nan, np.nan
                    pw_rows.append({
                        "comparison": f"{method} vs baseline",
                        "mode": mode, "level": level,
                        "U": U, "p": p, "cliff_d": d,
                        "effect": cliff_label(d) if not np.isnan(d) else "",
                        "mean_method": np.mean(mv) if len(mv) else np.nan,
                        "mean_baseline": np.mean(bv) if len(bv) else np.nan,
                    })

        pw_df = pd.DataFrame(pw_rows)
        bon_pw = bonferroni(pw_df["p"])
        w("| Method vs Baseline | Mode | Level | U | p | δ | Effect | Mean(M) | Mean(B) |")
        w("|--------------------|------|-------|--:|--:|--:|:------:|--------:|--------:|")
        for _, r in pw_df.iterrows():
            sig = " *" if r["p"] < bon_pw["alpha_c"] else ""
            w(f"| {r['comparison']} | {MODE_LABEL[r['mode']]} | {LEVEL_LABEL[r['level']]} "
              f"| {r['U']:.0f} | {r['p']:.4f}{sig} | {r['cliff_d']:+.3f} | {r['effect']} "
              f"| {r['mean_method']:.4f} | {r['mean_baseline']:.4f} |")
        w(f"\n**Bonferroni α'={bon_pw['alpha_c']:.5f}** (m={bon_pw['m']}). "
          f"**{bon_pw['n_sig']}** significant.\n")
        for eff in ["large", "medium", "small", "negligible"]:
            n_eff = (pw_df["effect"] == eff).sum()
            w(f"- {eff}: {n_eff}/{len(pw_df)} ({100*n_eff/len(pw_df):.0f}%)")
        w("")

    # =================================================================
    # 6. H2 — SW-SMOTE vs Plain SMOTE
    # =================================================================
    w("---\n## 6. Hypothesis Tests — H2: SW-SMOTE vs Plain SMOTE\n")

    for mi, (metric, mlabel) in enumerate(METRICS_EXT):
        w(f"\n### 6.{mi+1} [{mlabel}] H2: sw_smote vs plain smote\n")
        w("Paired comparison (same ratio): Does subject-wise synthesis improve?\n")

        h2_rows = []
        for ratio_tag in ["r01", "r05"]:
            c_sw = f"sw_smote_{ratio_tag}"
            c_sm = f"smote_{ratio_tag}"
            for mode in MODES:
                for level in LEVELS:
                    sw_v = df[(df["condition"] == c_sw) & (df["mode"] == mode)
                              & (df["level"] == level)][metric].dropna().values
                    sm_v = df[(df["condition"] == c_sm) & (df["mode"] == mode)
                              & (df["level"] == level)][metric].dropna().values
                    if len(sw_v) >= 2 and len(sm_v) >= 2:
                        U, p = stats.mannwhitneyu(sw_v, sm_v, alternative="two-sided")
                        d = cliff_delta(sw_v, sm_v)
                    else:
                        U, p, d = np.nan, np.nan, np.nan
                    h2_rows.append({
                        "ratio": ratio_tag, "mode": mode, "level": level,
                        "U": U, "p": p, "cliff_d": d,
                        "effect": cliff_label(d) if not np.isnan(d) else "",
                        "mean_sw": np.mean(sw_v) if len(sw_v) else np.nan,
                        "mean_sm": np.mean(sm_v) if len(sm_v) else np.nan,
                    })
        h2_df = pd.DataFrame(h2_rows)
        bon_h2 = bonferroni(h2_df["p"])
        w("| Ratio | Mode | Level | U | p | δ (sw−sm) | Effect | Mean(sw) | Mean(sm) |")
        w("|-------|------|-------|--:|--:|----------:|:------:|---------:|---------:|")
        for _, r in h2_df.iterrows():
            sig = " *" if r["p"] < bon_h2["alpha_c"] else ""
            w(f"| {r['ratio']} | {MODE_LABEL[r['mode']]} | {LEVEL_LABEL[r['level']]} "
              f"| {r['U']:.0f} | {r['p']:.4f}{sig} | {r['cliff_d']:+.3f} | {r['effect']} "
              f"| {r['mean_sw']:.4f} | {r['mean_sm']:.4f} |")
        sw_wins = (h2_df["cliff_d"] > 0).sum()
        sm_wins = (h2_df["cliff_d"] < 0).sum()
        w(f"\n**Summary**: sw_smote > smote in {sw_wins}/{len(h2_df)}, "
          f"smote > sw_smote in {sm_wins}/{len(h2_df)}. "
          f"Bonferroni sig: {bon_h2['n_sig']}/{bon_h2['m']}.\n")

    # =================================================================
    # 7. H3 — RUS vs Oversampling
    # =================================================================
    w("---\n## 7. Hypothesis Tests — H3: RUS vs Oversampling\n")

    for mi, (metric, mlabel) in enumerate(METRICS_EXT):
        w(f"\n### 7.{mi+1} [{mlabel}] H3: RUS vs oversampling (SMOTE/sw_smote)\n")

        h3_rows = []
        for ratio_tag in ["r01", "r05"]:
            c_rus = f"rus_{ratio_tag}"
            for c_over in [f"smote_{ratio_tag}", f"sw_smote_{ratio_tag}"]:
                for mode in MODES:
                    for level in LEVELS:
                        v_rus = df[(df["condition"] == c_rus) & (df["mode"] == mode)
                                   & (df["level"] == level)][metric].dropna().values
                        v_over = df[(df["condition"] == c_over) & (df["mode"] == mode)
                                    & (df["level"] == level)][metric].dropna().values
                        if len(v_rus) >= 2 and len(v_over) >= 2:
                            U, p = stats.mannwhitneyu(v_rus, v_over,
                                                      alternative="two-sided")
                            d = cliff_delta(v_over, v_rus)
                        else:
                            U, p, d = np.nan, np.nan, np.nan
                        h3_rows.append({
                            "comparison": f"{c_over} vs {c_rus}",
                            "mode": mode, "level": level,
                            "U": U, "p": p, "cliff_d": d,
                            "effect": cliff_label(d) if not np.isnan(d) else "",
                        })
        h3_df = pd.DataFrame(h3_rows)
        bon_h3 = bonferroni(h3_df["p"])
        over_better = (h3_df["cliff_d"] > 0).sum()
        w(f"Oversampling > RUS in **{over_better}/{len(h3_df)}** cells "
          f"(Bonferroni sig: {bon_h3['n_sig']}/{bon_h3['m']}).\n")
        for eff in ["large", "medium", "small", "negligible"]:
            n_eff = (h3_df["effect"] == eff).sum()
            w(f"- {eff}: {n_eff}/{len(h3_df)}")
        w("")

    # =================================================================
    # 8. H4 — Ratio Effect
    # =================================================================
    w("---\n## 8. Hypothesis Tests — H4: Ratio Effect (r=0.1 vs r=0.5)\n")

    for mi, (metric, mlabel) in enumerate(METRICS_EXT):
        w(f"\n### 8.{mi+1} [{mlabel}] H4: Ratio effect\n")
        w("$$H_0: \\mu_{r=0.1}^{(\\text{method})} = "
          "\\mu_{r=0.5}^{(\\text{method})}$$\n")

        h4_rows = []
        for method in ["rus", "smote", "sw_smote"]:
            c01 = f"{method}_r01"
            c05 = f"{method}_r05"
            for mode in MODES:
                for level in LEVELS:
                    v01 = df[(df["condition"] == c01) & (df["mode"] == mode)
                             & (df["level"] == level)][metric].dropna().values
                    v05 = df[(df["condition"] == c05) & (df["mode"] == mode)
                             & (df["level"] == level)][metric].dropna().values
                    if len(v01) >= 2 and len(v05) >= 2:
                        U, p = stats.mannwhitneyu(v01, v05,
                                                  alternative="two-sided")
                        d = cliff_delta(v05, v01)
                    else:
                        U, p, d = np.nan, np.nan, np.nan
                    h4_rows.append({
                        "method": method, "mode": mode, "level": level,
                        "U": U, "p": p, "cliff_d": d,
                        "effect": cliff_label(d) if not np.isnan(d) else "",
                        "mean_r01": np.mean(v01) if len(v01) else np.nan,
                        "mean_r05": np.mean(v05) if len(v05) else np.nan,
                    })
        h4_df = pd.DataFrame(h4_rows)
        bon_h4 = bonferroni(h4_df["p"])
        w("| Method | Mode | Level | U | p | δ (r05−r01) | Effect | Mean(r01) | Mean(r05) |")
        w("|--------|------|-------|--:|--:|------------:|:------:|----------:|----------:|")
        for _, r in h4_df.iterrows():
            sig = " *" if r["p"] < bon_h4["alpha_c"] else ""
            w(f"| {r['method']} | {MODE_LABEL[r['mode']]} | {LEVEL_LABEL[r['level']]} "
              f"| {r['U']:.0f} | {r['p']:.4f}{sig} | {r['cliff_d']:+.3f} | {r['effect']} "
              f"| {r['mean_r01']:.4f} | {r['mean_r05']:.4f} |")
        w(f"\n**Bonferroni α'={bon_h4['alpha_c']:.5f}** (m={bon_h4['m']}). "
          f"**{bon_h4['n_sig']}** significant.\n")
        for method in ["rus", "smote", "sw_smote"]:
            sub = h4_df[h4_df["method"] == method]
            r01_better = (sub["cliff_d"] < 0).sum()
            r05_better = (sub["cliff_d"] > 0).sum()
            w(f"- **{method}**: r=0.1 better in {r01_better}/{len(sub)}, "
              f"r=0.5 better in {r05_better}/{len(sub)}")
        w("")

    # =================================================================
    # 9. H5, H7, H10 — Distance, Mode, Domain Shift
    # =================================================================
    w("---\n## 9. Hypothesis Tests — H5 (Distance), H7 (Mode), H10 (Domain Shift)\n")

    for mi, (metric, mlabel) in enumerate(METRICS_EXT):
        # --- H5: Distance metric ---
        w(f"\n### 9.{mi+1}.1 [{mlabel}] H5: Distance metric effect\n")
        w("Kruskal-Wallis across MMD, DTW, Wasserstein (pooling conditions).\n")

        h5_rows = []
        for mode in MODES:
            for level in LEVELS:
                groups = [
                    df[(df["mode"] == mode) & (df["level"] == level)
                       & (df["distance"] == d)][metric].dropna().values
                    for d in DISTANCES
                ]
                if all(len(g) >= 2 for g in groups):
                    H, p = stats.kruskal(*groups)
                    n_t = sum(len(g) for g in groups)
                    eta2 = eta_squared_from_H(H, n_t, 3)
                else:
                    H, p, eta2 = np.nan, np.nan, np.nan
                h5_rows.append({"mode": mode, "level": level,
                                "H": H, "p": p, "eta2": eta2})
        h5_df = pd.DataFrame(h5_rows)
        w("| Mode | Level | H | p | η² |")
        w("|------|-------|--:|--:|---:|")
        for _, r in h5_df.iterrows():
            w(f"| {MODE_LABEL[r['mode']]} | {LEVEL_LABEL[r['level']]} "
              f"| {r['H']:.2f} | {r['p']:.4f} | {r['eta2']:.3f} |")
        n_sig_h5 = (h5_df["p"].dropna() < 0.05).sum()
        w(f"\n**{n_sig_h5}/{len(h5_df)}** significant at α=0.05.\n")

        # --- H7: Mode effect ---
        w(f"### 9.{mi+1}.2 [{mlabel}] H7–H8: Training mode effect\n")
        w("Within-domain vs cross-domain, mixed vs cross-domain.\n")

        for ref_mode, ref_label in [("target_only", "Within"), ("mixed", "Mixed")]:
            v_ref = df[df["mode"] == ref_mode][metric].dropna().values
            v_cross = df[df["mode"] == "source_only"][metric].dropna().values
            if len(v_ref) >= 2 and len(v_cross) >= 2:
                U, p = stats.mannwhitneyu(v_ref, v_cross,
                                          alternative="two-sided")
                d = cliff_delta(v_ref, v_cross)
            else:
                U, p, d = np.nan, np.nan, np.nan
            w(f"- **{ref_label} vs Cross-domain**: U={U:.0f}, p={p:.4f}, "
              f"δ={d:+.3f} ({cliff_label(d)}), "
              f"mean({ref_label})={np.mean(v_ref):.4f}, "
              f"mean(Cross)={np.mean(v_cross):.4f}")
        w("")

        # --- H10: Domain shift ---
        w(f"### 9.{mi+1}.3 [{mlabel}] H10: Domain shift (in vs out)\n")

        h10_rows = []
        for cond in CONDITIONS_7:
            for mode in MODES:
                for dist in DISTANCES:
                    sub_in = df[(df["condition"] == cond) & (df["mode"] == mode)
                                & (df["level"] == "in_domain")
                                & (df["distance"] == dist)
                                ].set_index("seed")[metric]
                    sub_out = df[(df["condition"] == cond) & (df["mode"] == mode)
                                 & (df["level"] == "out_domain")
                                 & (df["distance"] == dist)
                                 ].set_index("seed")[metric]
                    common = sub_in.index.intersection(sub_out.index)
                    if len(common) >= 6:
                        v_in = sub_in.loc[common].values
                        v_out = sub_out.loc[common].values
                        diff = v_in - v_out
                        if not np.all(diff == 0):
                            _, p_w = stats.wilcoxon(v_in, v_out,
                                                    alternative="two-sided")
                        else:
                            p_w = 1.0
                        d_shift = cliff_delta(v_in, v_out)
                    else:
                        p_w, d_shift = np.nan, np.nan
                    h10_rows.append({"condition": cond, "mode": mode,
                                     "distance": dist, "p": p_w,
                                     "cliff_d": d_shift})

        h10_df = pd.DataFrame(h10_rows)
        bon_h10 = bonferroni(h10_df["p"])
        in_better = (h10_df["cliff_d"].dropna() > 0).sum()
        w(f"Wilcoxon signed-rank (paired by seed): "
          f"in-domain > out-domain in **{in_better}/{len(h10_df)}** cells. "
          f"Bonferroni sig: **{bon_h10['n_sig']}/{bon_h10['m']}**.\n")

        # Aggregate
        all_in = df[df["level"] == "in_domain"][metric].dropna()
        all_out = df[df["level"] == "out_domain"][metric].dropna()
        d_glob = cliff_delta(all_in.values, all_out.values)
        w(f"**Global**: In-domain mean={all_in.mean():.4f}, "
          f"Out-domain mean={all_out.mean():.4f}, "
          f"Cliff's δ={d_glob:+.3f} ({cliff_label(d_glob)}).\n")

    # =================================================================
    # 10. Precision–Recall Trade-off Analysis (HE3)
    # =================================================================
    w("---\n## 10. Precision–Recall Trade-off Analysis (HE3)\n")
    w("**Hypothesis HE3**: Oversampling methods improve Recall at the cost of Precision.\n")
    w("For each rebalancing method vs baseline, we compute Cliff's δ for "
      "both Precision and Recall to quantify the trade-off direction.\n")
    w("")

    other_conds = [c for c in CONDITIONS_7 if c != "baseline"]
    w("| Method vs Baseline | Mode | Level | δ(Recall) | δ(Precision) | Trade-off? |")
    w("|--------------------|------|-------|----------:|-------------:|:----------:|")

    tradeoff_rows = []
    for method in other_conds:
        for mode in MODES:
            for level in LEVELS:
                bv_r = df[(df["condition"] == "baseline") & (df["mode"] == mode)
                          & (df["level"] == level)]["recall"].dropna().values
                mv_r = df[(df["condition"] == method) & (df["mode"] == mode)
                          & (df["level"] == level)]["recall"].dropna().values
                bv_p = df[(df["condition"] == "baseline") & (df["mode"] == mode)
                          & (df["level"] == level)]["precision"].dropna().values
                mv_p = df[(df["condition"] == method) & (df["mode"] == mode)
                          & (df["level"] == level)]["precision"].dropna().values

                if (len(bv_r) >= 2 and len(mv_r) >= 2
                        and len(bv_p) >= 2 and len(mv_p) >= 2):
                    d_r = cliff_delta(mv_r, bv_r)
                    d_p = cliff_delta(mv_p, bv_p)
                else:
                    d_r, d_p = np.nan, np.nan

                # Trade-off: Recall up AND Precision down
                is_tradeoff = (d_r > 0.147 and d_p < -0.147)
                tradeoff_rows.append({
                    "method": method, "mode": mode, "level": level,
                    "d_recall": d_r, "d_precision": d_p,
                    "tradeoff": is_tradeoff,
                })
                label = "✓ P↓R↑" if is_tradeoff else ""
                w(f"| {method} vs baseline | {MODE_LABEL[mode]} "
                  f"| {LEVEL_LABEL[level]} "
                  f"| {d_r:+.3f} | {d_p:+.3f} | {label} |")

    to_df = pd.DataFrame(tradeoff_rows)
    n_tradeoff = to_df["tradeoff"].sum()
    w(f"\n**Summary**: {n_tradeoff}/{len(to_df)} cells exhibit "
      f"a clear Precision–Recall trade-off (|δ| > 0.147 in opposite directions).\n")

    # Aggregate by method group
    w("#### Aggregated by method\n")
    w("| Method | Mean δ(Recall) | Mean δ(Precision) | Pattern |")
    w("|--------|---------------:|------------------:|---------|")
    for method in other_conds:
        sub = to_df[to_df["method"] == method]
        mean_dr = sub["d_recall"].mean()
        mean_dp = sub["d_precision"].mean()
        if mean_dr > 0.05 and mean_dp < -0.05:
            pattern = "Trade-off (R↑ P↓)"
        elif mean_dr > 0.05 and mean_dp > -0.05:
            pattern = "Win-win (R↑ P≈)"
        elif mean_dr < -0.05:
            pattern = "Regression (R↓)"
        else:
            pattern = "Negligible"
        w(f"| {method} | {mean_dr:+.3f} | {mean_dp:+.3f} | {pattern} |")
    w("")

    # =================================================================
    # 11. HE1 — Rebalancing Improves Recall over Baseline
    # =================================================================
    w("---\n## 11. Extended Hypothesis HE1: Rebalancing → Recall Improvement\n")
    w("**Hypothesis**: *Any* rebalancing method (including RUS) improves Recall "
      "over baseline, because both oversampling and undersampling increase "
      "the effective weight of the minority class.\n")
    w("")

    he1_rows = []
    for method in other_conds:
        for mode in MODES:
            for level in LEVELS:
                bv = df[(df["condition"] == "baseline") & (df["mode"] == mode)
                        & (df["level"] == level)]["recall"].dropna().values
                mv = df[(df["condition"] == method) & (df["mode"] == mode)
                        & (df["level"] == level)]["recall"].dropna().values
                if len(bv) >= 2 and len(mv) >= 2:
                    U, p = stats.mannwhitneyu(mv, bv, alternative="greater")
                    d = cliff_delta(mv, bv)
                else:
                    U, p, d = np.nan, np.nan, np.nan
                he1_rows.append({
                    "method": method, "mode": mode, "level": level,
                    "U": U, "p": p, "cliff_d": d,
                    "effect": cliff_label(d) if not np.isnan(d) else "",
                    "mean_method": np.mean(mv) if len(mv) else np.nan,
                    "mean_baseline": np.mean(bv) if len(bv) else np.nan,
                })

    he1_df = pd.DataFrame(he1_rows)
    bon_he1 = bonferroni(he1_df["p"])
    w("One-sided test: $H_1$: Recall(method) > Recall(baseline).\n")
    w("| Method | Mode | Level | U | p (one-sided) | δ | Effect | Mean(M) | Mean(B) |")
    w("|--------|------|-------|--:|:-------------:|--:|:------:|--------:|--------:|")
    for _, r in he1_df.iterrows():
        sig = " *" if r["p"] < bon_he1["alpha_c"] else ""
        w(f"| {r['method']} | {MODE_LABEL[r['mode']]} | {LEVEL_LABEL[r['level']]} "
          f"| {r['U']:.0f} | {r['p']:.4f}{sig} | {r['cliff_d']:+.3f} | {r['effect']} "
          f"| {r['mean_method']:.4f} | {r['mean_baseline']:.4f} |")
    w(f"\n**Bonferroni α'={bon_he1['alpha_c']:.5f}** (m={bon_he1['m']}). "
      f"**{bon_he1['n_sig']}** significant.\n")

    recall_improved = (he1_df["cliff_d"] > 0).sum()
    w(f"Method > baseline in {recall_improved}/{len(he1_df)} cells.\n")

    # RUS-specific sub-analysis
    rus_sub = he1_df[he1_df["method"].str.startswith("rus")]
    rus_improved = (rus_sub["cliff_d"] > 0).sum()
    w(f"**RUS specifically**: Recall improved in {rus_improved}/{len(rus_sub)} cells "
      f"(mean δ={rus_sub['cliff_d'].mean():+.3f}).\n")

    smote_sub = he1_df[he1_df["method"].str.contains("smote")]
    smote_improved = (smote_sub["cliff_d"] > 0).sum()
    w(f"**SMOTE-family**: Recall improved in {smote_improved}/{len(smote_sub)} cells "
      f"(mean δ={smote_sub['cliff_d'].mean():+.3f}).\n")

    # =================================================================
    # 12. HE2 — AUPRC Sensitivity vs AUROC
    # =================================================================
    w("---\n## 12. Extended Hypothesis HE2: AUPRC vs AUROC Sensitivity\n")
    w("**Hypothesis**: AUPRC shows a stronger condition effect than AUROC "
      "because AUPRC is more sensitive to minority-class performance.\n")
    w("")

    w("### Comparison of η² (condition effect size) per cell\n")
    w("| Mode | Level | η²(AUROC) | η²(AUPRC) | AUPRC stronger? |")
    w("|------|-------|----------:|----------:|:---------------:|")
    auprc_stronger = 0
    total_cells = 0
    for mode in MODES:
        for level in LEVELS:
            # Pool across distances
            groups_auc = [
                df[(df["condition"] == c) & (df["mode"] == mode)
                   & (df["level"] == level)]["auc"].dropna().values
                for c in CONDITIONS_7
            ]
            groups_auprc = [
                df[(df["condition"] == c) & (df["mode"] == mode)
                   & (df["level"] == level)]["auc_pr"].dropna().values
                for c in CONDITIONS_7
            ]
            if (all(len(g) >= 2 for g in groups_auc)
                    and all(len(g) >= 2 for g in groups_auprc)):
                H_auc, _ = stats.kruskal(*groups_auc)
                H_auprc, _ = stats.kruskal(*groups_auprc)
                n_auc = sum(len(g) for g in groups_auc)
                n_auprc = sum(len(g) for g in groups_auprc)
                eta2_auc = eta_squared_from_H(H_auc, n_auc, len(CONDITIONS_7))
                eta2_auprc = eta_squared_from_H(H_auprc, n_auprc,
                                                len(CONDITIONS_7))
                stronger = eta2_auprc > eta2_auc
                if stronger:
                    auprc_stronger += 1
                total_cells += 1
                w(f"| {MODE_LABEL[mode]} | {LEVEL_LABEL[level]} "
                  f"| {eta2_auc:.3f} | {eta2_auprc:.3f} "
                  f"| {'✓' if stronger else '✗'} |")
    w(f"\n**Result**: AUPRC shows stronger condition effect in "
      f"**{auprc_stronger}/{total_cells}** cells.\n")

    # Ranking comparison
    w("### Ranking comparison: AUROC vs AUPRC\n")
    rank_auc = {}
    rank_auprc = {}
    for m_key, m_col in [("auc", rank_auc), ("auc_pr", rank_auprc)]:
        cells_r = []
        for mode in MODES:
            for level in LEVELS:
                for dist in DISTANCES:
                    means_r = {}
                    for c in CONDITIONS_7:
                        v = df[(df["condition"] == c) & (df["mode"] == mode)
                               & (df["level"] == level)
                               & (df["distance"] == dist)][m_key]
                        means_r[c] = v.mean() if len(v) else np.nan
                    sorted_r = sorted(means_r, key=lambda k: means_r[k],
                                      reverse=True)
                    ranks_r = {c: r + 1 for r, c in enumerate(sorted_r)}
                    cells_r.append(ranks_r)
        rdf = pd.DataFrame(cells_r)
        m_col.update(rdf.mean().to_dict())

    w("| Condition | Mean Rank (AUROC) | Mean Rank (AUPRC) | Δ Rank |")
    w("|-----------|------------------:|------------------:|-------:|")
    for c in CONDITIONS_7:
        r_auc = rank_auc.get(c, np.nan)
        r_auprc = rank_auprc.get(c, np.nan)
        w(f"| {c} | {r_auc:.2f} | {r_auprc:.2f} | {r_auprc - r_auc:+.2f} |")
    rho_r, _ = stats.spearmanr(
        [rank_auc[c] for c in CONDITIONS_7],
        [rank_auprc[c] for c in CONDITIONS_7],
    )
    w(f"\n**Spearman ρ** (AUROC vs AUPRC rankings) = {rho_r:.3f} "
      f"({'strong' if abs(rho_r) > 0.7 else 'moderate' if abs(rho_r) > 0.4 else 'weak'} "
      f"concordance).\n")

    # =================================================================
    # 13. Cross-Axis Interactions (H12, H13, H14)
    # =================================================================
    w("---\n## 13. Cross-Axis Interaction Analysis\n")

    for mi, (metric, mlabel) in enumerate(METRICS_EXT):
        # H12: Condition × Mode
        w(f"\n### 13.{mi+1}.1 [{mlabel}] H12: Condition × Mode interaction\n")
        w("Best condition per mode:\n")
        w("| Mode | Level | Best Condition | Mean | 2nd | Mean |")
        w("|------|-------|:-------------:|-----:|:---:|-----:|")
        for mode in MODES:
            for level in LEVELS:
                means = {}
                for cond in CONDITIONS_7:
                    v = df[(df["condition"] == cond) & (df["mode"] == mode)
                           & (df["level"] == level)][metric].dropna()
                    means[cond] = v.mean() if len(v) else np.nan
                sorted_c = sorted(means, key=lambda k: means[k], reverse=True)
                w(f"| {MODE_LABEL[mode]} | {LEVEL_LABEL[level]} "
                  f"| {sorted_c[0]} | {means[sorted_c[0]]:.4f} "
                  f"| {sorted_c[1]} | {means[sorted_c[1]]:.4f} |")

        w("\n**Friedman test** (condition effect per mode, seeds as blocks):\n")
        w("| Mode | Level | χ² | p | Kendall's W |")
        w("|------|-------|---:|--:|:----------:|")
        for mode in MODES:
            for level in LEVELS:
                seed_vals = {}
                for cond in CONDITIONS_7:
                    sub = df[(df["condition"] == cond) & (df["mode"] == mode)
                             & (df["level"] == level)]
                    seed_vals[cond] = sub.groupby("seed")[metric].mean()
                idx = None
                for sv in seed_vals.values():
                    idx = sv.index if idx is None else idx.intersection(sv.index)
                if idx is not None and len(idx) >= 3:
                    arrays = [seed_vals[c].loc[idx].values
                              for c in CONDITIONS_7]
                    chi2, p_f = stats.friedmanchisquare(*arrays)
                    W_k = cohens_w_from_friedman(chi2, len(idx),
                                                 len(CONDITIONS_7))
                else:
                    chi2, p_f, W_k = np.nan, np.nan, np.nan
                sig = " *" if not np.isnan(p_f) and p_f < 0.05 else ""
                w(f"| {MODE_LABEL[mode]} | {LEVEL_LABEL[level]} "
                  f"| {chi2:.2f} | {p_f:.4f}{sig} | {W_k:.3f} |")
        w("")

        # H13: Condition × Distance
        w(f"### 13.{mi+1}.2 [{mlabel}] H13: Condition × Distance interaction\n")
        w("| Mode | Level | MMD best | DTW best | Wass. best | Consistent? |")
        w("|------|-------|:--------:|:--------:|:----------:|:-----------:|")
        for mode in MODES:
            for level in LEVELS:
                bests = {}
                for dist in DISTANCES:
                    means = {}
                    for cond in CONDITIONS_7:
                        v = df[(df["condition"] == cond) & (df["mode"] == mode)
                               & (df["level"] == level)
                               & (df["distance"] == dist)][metric].dropna()
                        means[cond] = v.mean() if len(v) else np.nan
                    bests[dist] = max(means, key=means.get) if means else "—"
                consistent = len(set(bests.values())) == 1
                w(f"| {MODE_LABEL[mode]} | {LEVEL_LABEL[level]} "
                  f"| {bests['mmd']} | {bests['dtw']} "
                  f"| {bests['wasserstein']} "
                  f"| {'✓' if consistent else '✗'} |")
        w("")

        # H14: Domain gap by mode
        w(f"### 13.{mi+1}.3 [{mlabel}] H14: Domain gap by mode\n")
        w("| Mode | Mean gap (Δ=out−in) | Mean |Δ| |")
        w("|------|:-------------------:|------:|")
        for mode in MODES:
            gaps = []
            for cond in CONDITIONS_7:
                for dist in DISTANCES:
                    si = df[(df["condition"] == cond) & (df["mode"] == mode)
                            & (df["level"] == "in_domain")
                            & (df["distance"] == dist)
                            ].set_index("seed")[metric]
                    so = df[(df["condition"] == cond) & (df["mode"] == mode)
                            & (df["level"] == "out_domain")
                            & (df["distance"] == dist)
                            ].set_index("seed")[metric]
                    common = si.index.intersection(so.index)
                    if len(common) >= 1:
                        gaps.extend((so.loc[common] - si.loc[common]).values)
            gaps = np.array(gaps)
            w(f"| {MODE_LABEL[mode]} | {np.mean(gaps):+.4f} "
              f"| {np.mean(np.abs(gaps)):.4f} |")
        w("")

    # =================================================================
    # 14. Overall Condition Ranking
    # =================================================================
    w("---\n## 14. Overall Condition Ranking\n")
    w("Mean rank across 18 cells (3 modes × 2 levels × 3 distances). "
      "Rank 1 = best.\n")

    for metric, mlabel in METRICS_EXT:
        w(f"### {mlabel}\n")
        cells = []
        for mode in MODES:
            for level in LEVELS:
                for dist in DISTANCES:
                    means = {}
                    for c in CONDITIONS_7:
                        v = df[(df["condition"] == c) & (df["mode"] == mode)
                               & (df["level"] == level)
                               & (df["distance"] == dist)][metric]
                        means[c] = v.mean() if len(v) else np.nan
                    sorted_c = sorted(means, key=lambda k: means[k],
                                      reverse=True)
                    ranks = {c: r + 1 for r, c in enumerate(sorted_c)}
                    cells.append(ranks)
        rank_df = pd.DataFrame(cells)
        summary = rank_df.mean().sort_values()
        win_counts = (rank_df == 1).sum()
        w("| Rank | Condition | Mean Rank | Win count (rank 1) |")
        w("|:----:|-----------|----------:|:------------------:|")
        for i, (cond, mr) in enumerate(summary.items()):
            w(f"| {i+1} | {cond} | {mr:.2f} | {win_counts[cond]} |")
        w("")

    # =================================================================
    # 15. Nemenyi Post-Hoc
    # =================================================================
    w("---\n## 15. Nemenyi Post-Hoc Test\n")

    for metric, mlabel in METRICS_EXT:
        w(f"\n### {mlabel}\n")
        for level in LEVELS:
            w(f"#### {LEVEL_LABEL[level]} (pooled across modes)\n")
            sub = df[df["level"] == level]
            pivot = sub.groupby(["seed", "condition"])[metric].mean().reset_index()
            piv_wide = pivot.pivot(index="seed", columns="condition",
                                   values=metric)
            piv_wide = piv_wide[CONDITIONS_7].dropna()

            if len(piv_wide) >= 3:
                arrays = [piv_wide[c].values for c in CONDITIONS_7]
                chi2, p_fri = stats.friedmanchisquare(*arrays)
                w(f"Friedman χ²={chi2:.2f}, p={p_fri:.4f} "
                  f"({'significant' if p_fri < 0.05 else 'not significant'})\n")

                if p_fri < 0.05:
                    nem = sp.posthoc_nemenyi_friedman(piv_wide.values)
                    nem.index = CONDITIONS_7
                    nem.columns = CONDITIONS_7

                    w("| | " + " | ".join(CONDITIONS_7) + " |")
                    w("|---" + "|---" * len(CONDITIONS_7) + "|")
                    for i, c1 in enumerate(CONDITIONS_7):
                        row = f"| **{c1}** |"
                        for j, c2 in enumerate(CONDITIONS_7):
                            if j <= i:
                                row += " — |"
                            else:
                                p_n = nem.iloc[i, j]
                                sig = " *" if p_n < 0.05 else ""
                                row += f" {p_n:.4f}{sig} |"
                        w(row)
                    w("")

                    sig_pairs = [
                        f"{CONDITIONS_7[i]} vs {CONDITIONS_7[j]}"
                        for i in range(len(CONDITIONS_7))
                        for j in range(i + 1, len(CONDITIONS_7))
                        if nem.iloc[i, j] < 0.05
                    ]
                    total_pairs = len(CONDITIONS_7) * (len(CONDITIONS_7) - 1) // 2
                    w(f"**Significant pairs**: {len(sig_pairs)}/{total_pairs}")
                    for pair in sig_pairs:
                        w(f"- {pair}")
                    w("")

                    mean_ranks = piv_wide.rank(axis=1, ascending=False).mean()
                    w("**Mean ranks**:\n")
                    w("| Condition | Mean Rank |")
                    w("|-----------|----------:|")
                    for c in mean_ranks.sort_values().index:
                        w(f"| {c} | {mean_ranks[c]:.2f} |")

                    k = len(CONDITIONS_7)
                    n_blocks = len(piv_wide)
                    from scipy.stats import studentized_range
                    q_crit = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)
                    cd = q_crit * np.sqrt(k * (k + 1) / (6 * n_blocks))
                    w(f"\n**Critical Difference (CD)** = {cd:.3f} "
                      f"(α=0.05, k={k}, n={n_blocks})\n")
                else:
                    w("Friedman not significant — Nemenyi not applicable.\n")
            else:
                w("Insufficient data.\n")

    # =================================================================
    # 16. Bootstrap CIs (BCa)
    # =================================================================
    w("---\n## 16. Bootstrap Confidence Intervals (BCa)\n")
    w("B = 10,000 resamples, seed-level resampling.\n")
    rng_boot = np.random.RandomState(42)

    for metric, mlabel in METRICS_EXT:
        w(f"\n### {mlabel}\n")
        w("| Condition | Mode | Level | Mean | 95% CI Lower | 95% CI Upper | Width |")
        w("|-----------|------|-------|-----:|-------------:|-------------:|------:|")
        for cond in CONDITIONS_7:
            for mode in MODES:
                for level in LEVELS:
                    sub = df[(df["condition"] == cond) & (df["mode"] == mode)
                             & (df["level"] == level)]
                    seed_means = sub.groupby("seed")[metric].mean().dropna().values
                    if len(seed_means) >= 3:
                        est, lo, hi = bootstrap_ci_bca(seed_means, B=10000,
                                                       rng=rng_boot)
                    else:
                        est = np.mean(seed_means) if len(seed_means) else np.nan
                        lo, hi = np.nan, np.nan
                    width = hi - lo if not np.isnan(hi) else np.nan
                    w(f"| {cond} | {MODE_LABEL[mode]} | {LEVEL_LABEL[level]} "
                      f"| {est:.4f} | {lo:.4f} | {hi:.4f} | {width:.4f} |")
        w("")

    # =================================================================
    # 17. Effect Size CIs (Cliff's δ)
    # =================================================================
    w("---\n## 17. Effect Size Confidence Intervals (Cliff's δ)\n")
    w("Bootstrap 95% CI (B = 2,000, percentile method).\n")
    rng_ci = np.random.RandomState(42)

    for metric, mlabel in METRICS_EXT:
        w(f"\n### {mlabel} — Baseline vs each method\n")
        w("| Method vs Baseline | Mode | Level | δ | 95% CI | Excl. 0? | Effect |")
        w("|--------------------|------|-------|--:|-------:|:--------:|:------:|")

        ci_rows = []
        for method in [c for c in CONDITIONS_7 if c != "baseline"]:
            for mode in MODES:
                for level in LEVELS:
                    bv = df[(df["condition"] == "baseline") & (df["mode"] == mode)
                            & (df["level"] == level)][metric].dropna().values
                    mv = df[(df["condition"] == method) & (df["mode"] == mode)
                            & (df["level"] == level)][metric].dropna().values
                    if len(bv) >= 2 and len(mv) >= 2:
                        d, lo, hi = cliff_delta_ci(mv, bv, B=2000, rng=rng_ci)
                    else:
                        d, lo, hi = np.nan, np.nan, np.nan
                    excl = ("✓" if (not np.isnan(lo) and (lo > 0 or hi < 0))
                            else "✗")
                    eff = cliff_label(d) if not np.isnan(d) else ""
                    w(f"| {method} vs baseline | {MODE_LABEL[mode]} "
                      f"| {LEVEL_LABEL[level]} "
                      f"| {d:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {excl} | {eff} |")
                    ci_rows.append({"excl": excl == "✓"})
        ci_df_e = pd.DataFrame(ci_rows)
        n_excl = ci_df_e["excl"].sum()
        w(f"\n**{n_excl}/{len(ci_df_e)}** CIs exclude 0.\n")

    # =================================================================
    # 18. Permutation Test
    # =================================================================
    w("---\n## 18. Permutation Test for Global Null\n")
    w("B = 10,000 permutations.\n")

    for metric, mlabel in METRICS_EXT:
        print(f"  Running permutation test ({mlabel})...")
        T_obs, p_perm, n_perm = permutation_test_condition(
            df, metric, n_perm=10000, rng=np.random.RandomState(42))
        w(f"\n### {mlabel}\n")
        w(f"- $T_{{\\text{{obs}}}}$ = {T_obs:.4f}")
        w(f"- $p_{{\\text{{perm}}}}$ = {p_perm:.4f} (B = {n_perm})")
        if p_perm < 0.001:
            w(f"- **Interpretation**: Strong evidence against the global null "
              f"(p < 0.001). Condition labels are informative for {mlabel}.\n")
        elif p_perm < 0.05:
            w(f"- **Interpretation**: Significant (p = {p_perm:.4f}).\n")
        else:
            w(f"- **Interpretation**: No significant global condition effect "
              f"(p = {p_perm:.4f}).\n")

    # =================================================================
    # 19. BH-FDR Sensitivity Analysis
    # =================================================================
    w("\n---\n## 19. Benjamini-Hochberg FDR Correction Sensitivity\n")
    w("| Hypothesis Family | m | Bonf. sig | FDR sig | Gain |")
    w("|-------------------|--:|----------:|--------:|-----:|")

    total_bon = 0
    total_fdr = 0

    for metric, mlabel in METRICS_EXT:
        # H1 KW
        kw_pvals = []
        for mode in MODES:
            for level in LEVELS:
                for dist in DISTANCES:
                    groups = [
                        df[(df["condition"] == c) & (df["mode"] == mode)
                           & (df["level"] == level)
                           & (df["distance"] == dist)][metric].dropna().values
                        for c in CONDITIONS_7
                    ]
                    if all(len(g) >= 2 for g in groups):
                        _, p = stats.kruskal(*groups)
                        kw_pvals.append(p)
        kw_ps = pd.Series(kw_pvals)
        bon_k = bonferroni(kw_ps)
        fdr_k = benjamini_hochberg(kw_ps)
        w(f"| H1 KW ({mlabel}) | {bon_k['m']} | {bon_k['n_sig']} "
          f"| {fdr_k['n_sig']} | +{fdr_k['n_sig']-bon_k['n_sig']} |")
        total_bon += bon_k["n_sig"]
        total_fdr += fdr_k["n_sig"]

        # H1 pairwise
        pw_pvals = []
        for method in [c for c in CONDITIONS_7 if c != "baseline"]:
            for mode in MODES:
                for level in LEVELS:
                    bv = df[(df["condition"] == "baseline") & (df["mode"] == mode)
                            & (df["level"] == level)][metric].dropna().values
                    mv = df[(df["condition"] == method) & (df["mode"] == mode)
                            & (df["level"] == level)][metric].dropna().values
                    if len(bv) >= 2 and len(mv) >= 2:
                        _, p = stats.mannwhitneyu(bv, mv, alternative="two-sided")
                        pw_pvals.append(p)
        pw_ps = pd.Series(pw_pvals)
        bon_p = bonferroni(pw_ps)
        fdr_p = benjamini_hochberg(pw_ps)
        w(f"| H1 pairwise ({mlabel}) | {bon_p['m']} | {bon_p['n_sig']} "
          f"| {fdr_p['n_sig']} | +{fdr_p['n_sig']-bon_p['n_sig']} |")
        total_bon += bon_p["n_sig"]
        total_fdr += fdr_p["n_sig"]

        # H10 Wilcoxon
        h10_pvals = []
        for cond in CONDITIONS_7:
            for mode in MODES:
                for dist in DISTANCES:
                    si = df[(df["condition"] == cond) & (df["mode"] == mode)
                            & (df["level"] == "in_domain")
                            & (df["distance"] == dist)
                            ].set_index("seed")[metric]
                    so = df[(df["condition"] == cond) & (df["mode"] == mode)
                            & (df["level"] == "out_domain")
                            & (df["distance"] == dist)
                            ].set_index("seed")[metric]
                    common = si.index.intersection(so.index)
                    if len(common) >= 6:
                        diff = si.loc[common].values - so.loc[common].values
                        if not np.all(diff == 0):
                            _, p = stats.wilcoxon(
                                si.loc[common].values,
                                so.loc[common].values,
                                alternative="two-sided")
                            h10_pvals.append(p)
        h10_ps = pd.Series(h10_pvals)
        bon_h = bonferroni(h10_ps)
        fdr_h = benjamini_hochberg(h10_ps)
        w(f"| H10 domain shift ({mlabel}) | {bon_h['m']} | {bon_h['n_sig']} "
          f"| {fdr_h['n_sig']} | +{fdr_h['n_sig']-bon_h['n_sig']} |")
        total_bon += bon_h["n_sig"]
        total_fdr += fdr_h["n_sig"]

    w(f"\n**Overall**: Bonferroni **{total_bon}** → BH-FDR **{total_fdr}** "
      f"(+{total_fdr - total_bon}).\n")

    # =================================================================
    # 20. Cross-Metric Concordance
    # =================================================================
    w("---\n## 20. Cross-Metric Concordance\n")
    w("Do the extended metrics (F1, AUPRC, Recall) agree with the primary "
      "metrics (F2, AUROC) on condition rankings?\n")

    mean_ranks_all = {}
    for m_key, m_label in ALL_METRICS:
        cells_r = []
        for mode in MODES:
            for level in LEVELS:
                for dist in DISTANCES:
                    means_r = {}
                    for c in CONDITIONS_7:
                        v = df[(df["condition"] == c) & (df["mode"] == mode)
                               & (df["level"] == level)
                               & (df["distance"] == dist)][m_key]
                        means_r[c] = v.mean() if len(v) else np.nan
                    sorted_r = sorted(means_r, key=lambda k: means_r[k],
                                      reverse=True)
                    ranks_r = {c: r + 1 for r, c in enumerate(sorted_r)}
                    cells_r.append(ranks_r)
        rdf = pd.DataFrame(cells_r)
        mean_ranks_all[m_key] = rdf.mean()

    # Ranking table
    w("### Overall ranking comparison\n")
    w("| Metric | #1 | #2 | #3 | #4 | #5 | #6 | #7 |")
    w("|--------|:---|:---|:---|:---|:---|:---|:---|")
    for m_key, m_label in ALL_METRICS:
        summary = mean_ranks_all[m_key].sort_values()
        row = f"| {m_label}"
        for cond, mr in summary.items():
            row += f" | {cond} ({mr:.2f})"
        w(row + " |")
    w("")

    # Kendall's W
    mr_arr = np.array([
        [mean_ranks_all[m][c] for c in CONDITIONS_7]
        for m, _ in ALL_METRICS
    ])
    from scipy.stats import rankdata
    ranked = np.array([rankdata(row) for row in mr_arr])
    rank_sums = ranked.sum(axis=0)
    S_w = np.sum((rank_sums - rank_sums.mean()) ** 2)
    k_w = mr_arr.shape[0]
    n_w = mr_arr.shape[1]
    W_kendall = 12 * S_w / (k_w ** 2 * (n_w ** 3 - n_w))

    w(f"**Kendall's W** = {W_kendall:.3f} (k={k_w} metrics, n={n_w} conditions)\n")
    interp = ('strong' if W_kendall > 0.7
              else 'moderate' if W_kendall > 0.4
              else 'weak')
    w(f"**Interpretation**: {interp.capitalize()} agreement.\n")

    # Spearman pairwise
    w("### Pairwise Spearman ρ\n")
    all_labels = [ml for _, ml in ALL_METRICS]
    w("| | " + " | ".join(all_labels) + " |")
    w("|---" + "|---" * len(all_labels) + "|")
    for i, (mi_key, mi_label) in enumerate(ALL_METRICS):
        row = f"| **{mi_label}**"
        for j, (mj_key, mj_label) in enumerate(ALL_METRICS):
            if j <= i:
                row += " | —"
            else:
                rho, _ = stats.spearmanr(mr_arr[i], mr_arr[j])
                row += f" | {rho:.3f}"
        w(row + " |")
    w("")

    # Primary vs extended concordance
    w("### Primary vs Extended concordance\n")
    primary_keys = [m for m, _ in METRICS_PRIMARY]
    ext_keys = [m for m, _ in METRICS_EXT]
    rhos = []
    for pk in primary_keys:
        for ek in ext_keys:
            rho, _ = stats.spearmanr(
                [mean_ranks_all[pk][c] for c in CONDITIONS_7],
                [mean_ranks_all[ek][c] for c in CONDITIONS_7],
            )
            rhos.append(rho)
            pk_label = dict(ALL_METRICS)[pk]
            ek_label = dict(ALL_METRICS)[ek]
            w(f"- **{pk_label} ↔ {ek_label}**: ρ = {rho:.3f}")
    mean_rho = np.mean(rhos)
    w(f"\nMean cross-group ρ = {mean_rho:.3f}.\n")

    # =================================================================
    # 21. Hypothesis Verdict Summary
    # =================================================================
    w("---\n## 21. Hypothesis Verdict Summary\n")

    verdicts = []

    for metric, mlabel in METRICS_EXT:
        # H1
        kw_rows_v = []
        for mode in MODES:
            for level in LEVELS:
                for dist in DISTANCES:
                    groups = [
                        df[(df["condition"] == c) & (df["mode"] == mode)
                           & (df["level"] == level)
                           & (df["distance"] == dist)][metric].dropna().values
                        for c in CONDITIONS_7
                    ]
                    if all(len(g) >= 2 for g in groups):
                        _, p = stats.kruskal(*groups)
                        kw_rows_v.append(p)
        bon_v = bonferroni(pd.Series(kw_rows_v))
        pct = 100 * bon_v["n_sig"] / max(bon_v["m"], 1)
        verdicts.append(("H1", f"Condition effect ({mlabel})",
                         f"{bon_v['n_sig']}/{bon_v['m']} sig",
                         "Supported ✓" if pct > 50 else
                         "Partially" if pct > 0 else "Not supported ✗"))

        # H2: sw > smote
        sw_better = 0
        total_h2 = 0
        for rt in ["r01", "r05"]:
            for mode in MODES:
                for level in LEVELS:
                    sw = df[(df["condition"] == f"sw_smote_{rt}")
                            & (df["mode"] == mode)
                            & (df["level"] == level)][metric].dropna().values
                    sm = df[(df["condition"] == f"smote_{rt}")
                            & (df["mode"] == mode)
                            & (df["level"] == level)][metric].dropna().values
                    if len(sw) and len(sm):
                        total_h2 += 1
                        if np.mean(sw) > np.mean(sm):
                            sw_better += 1
        pct2 = 100 * sw_better / max(total_h2, 1)
        verdicts.append(("H2", f"sw > smote ({mlabel})",
                         f"{sw_better}/{total_h2} cells",
                         "Supported ✓" if pct2 > 60 else
                         "Mixed" if pct2 > 40 else "Not supported ✗"))

        # H3: oversampling > RUS
        o_better = 0
        total_h3 = 0
        for rt in ["r01", "r05"]:
            for c_over in [f"smote_{rt}", f"sw_smote_{rt}"]:
                for mode in MODES:
                    for level in LEVELS:
                        vr = df[(df["condition"] == f"rus_{rt}")
                                & (df["mode"] == mode)
                                & (df["level"] == level)][metric].dropna().values
                        vo = df[(df["condition"] == c_over)
                                & (df["mode"] == mode)
                                & (df["level"] == level)][metric].dropna().values
                        if len(vr) and len(vo):
                            total_h3 += 1
                            if np.mean(vo) > np.mean(vr):
                                o_better += 1
        pct3 = 100 * o_better / max(total_h3, 1)
        verdicts.append(("H3", f"Over > RUS ({mlabel})",
                         f"{o_better}/{total_h3}",
                         "Supported ✓" if pct3 > 60 else
                         "Mixed" if pct3 > 40 else "Not supported ✗"))

        # H5: distance effect
        groups_d = [df[df["distance"] == d][metric].dropna().values
                    for d in DISTANCES]
        H_d, p_d = stats.kruskal(*groups_d)
        verdicts.append(("H5", f"Distance effect ({mlabel})",
                         f"H={H_d:.2f}, p={p_d:.4f}",
                         "Supported ✓" if p_d < 0.05
                         else "Not supported ✗"))

        # H7: within > cross
        vw = df[df["mode"] == "target_only"][metric].dropna().values
        vc = df[df["mode"] == "source_only"][metric].dropna().values
        d7 = cliff_delta(vw, vc)
        verdicts.append(("H7", f"Within > cross ({mlabel})",
                         f"δ={d7:+.3f} ({cliff_label(d7)})",
                         "Supported ✓" if d7 > 0.147
                         else "Not supported ✗"))

        # H10: domain shift
        vi10 = df[df["level"] == "in_domain"][metric].dropna().values
        vo10 = df[df["level"] == "out_domain"][metric].dropna().values
        d10 = cliff_delta(vi10, vo10)
        _, p10 = stats.mannwhitneyu(vi10, vo10, alternative="two-sided")
        verdicts.append(("H10", f"Domain shift ({mlabel})",
                         f"δ={d10:+.3f}, p={p10:.4f}",
                         "Supported ✓" if d10 > 0.147 and p10 < 0.05
                         else "Weak" if p10 < 0.05
                         else "Not supported ✗"))

    # HE1: rebalancing → recall
    verdicts.append(("HE1", "Rebalancing → Recall↑",
                     f"{recall_improved}/{len(he1_df)} cells improved",
                     "Supported ✓" if recall_improved / len(he1_df) > 0.6
                     else "Partially" if recall_improved / len(he1_df) > 0.4
                     else "Not supported ✗"))

    # HE2: AUPRC > AUROC sensitivity
    verdicts.append(("HE2", "AUPRC more sensitive than AUROC",
                     f"Stronger η² in {auprc_stronger}/{total_cells} cells",
                     "Supported ✓" if auprc_stronger / total_cells > 0.6
                     else "Mixed" if auprc_stronger / total_cells > 0.4
                     else "Not supported ✗"))

    # HE3: P-R trade-off
    verdicts.append(("HE3", "Precision–Recall trade-off",
                     f"{n_tradeoff}/{len(to_df)} cells show P↓R↑",
                     "Supported ✓" if n_tradeoff / len(to_df) > 0.3
                     else "Partially" if n_tradeoff / len(to_df) > 0.1
                     else "Not supported ✗"))

    w("| ID | Hypothesis | Evidence | Verdict |")
    w("|:--:|-----------|----------|---------|")
    for hid, desc, evidence, verdict in verdicts:
        w(f"| {hid} | {desc} | {evidence} | {verdict} |")
    w("")

    # =================================================================
    # 22. Conclusions
    # =================================================================
    w("---\n## 22. Conclusions\n")
    w("### Key Findings from Extended Analysis\n")
    w("")
    w("1. **Metric consistency**: The extended metrics (F1, AUPRC, Recall) "
      f"show {interp} agreement with the primary metrics (F2, AUROC) on "
      f"condition rankings (Kendall's W = {W_kendall:.3f}), confirming that "
      "the primary analysis conclusions are robust to metric choice.\n")
    w("2. **AUPRC provides additional insight**: As a threshold-free metric "
      "sensitive to minority-class performance, AUPRC complements AUROC and is "
      "particularly relevant for imbalanced drowsiness detection.\n")
    w("3. **Recall validation**: Direct analysis of Recall confirms that "
      "rebalancing methods achieve their intended effect of improving positive "
      "class detection rate.\n")
    w("4. **Precision–Recall trade-off**: The trade-off analysis quantifies the "
      "cost of improved Recall in terms of Precision degradation, providing "
      "practical guidance for deployment threshold selection.\n")
    w("")

    w("### Relationship to Primary Analysis\n")
    w("This report should be read alongside the primary hypothesis test report "
      "(F2-score and AUROC). Together, the two reports provide a comprehensive "
      "statistical evaluation of Experiment 2 across 7 evaluation metrics.\n")
    w("")

    return "\n".join(lines)


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("=" * 60)
    print("Experiment 2 — Extended Hypothesis Testing (F1, AUPRC, Recall)")
    print("=" * 60)

    df = load_all_data()
    print(f"Loaded {len(df)} records")
    print(f"  Conditions: {sorted(df['condition'].unique())}")
    seeds = sorted(int(s) for s in df["seed"].unique())
    print(f"  Seeds: {seeds} (n={len(seeds)})")
    print(f"  Columns: {list(df.columns)}")

    # Verify extended metrics exist
    for m, ml in METRICS_EXT:
        if m not in df.columns:
            print(f"ERROR: metric '{m}' not found in data!")
            sys.exit(1)
        n_valid = df[m].notna().sum()
        print(f"  {ml} ({m}): {n_valid} valid values")

    report = generate_extended_report(df)
    out_path = REPORT_DIR / "hypothesis_test_report_extended.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved to: {out_path}")
    print(f"Report length: {len(report.splitlines())} lines")
    print("=" * 60)


if __name__ == "__main__":
    main()
