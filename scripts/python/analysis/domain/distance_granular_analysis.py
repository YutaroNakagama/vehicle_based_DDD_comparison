#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
distance_granular_analysis.py
=============================
Granular Kruskal-Wallis analysis of distance metric effect,
stratified by mode, condition, and domain level.

Tests whether the distance metric (MMD/DTW/Wasserstein) has a
significant effect on each metric when broken down by:
  - mode alone (3 groups)
  - condition alone (7 groups, incl. baseline)
  - level alone (2 groups)
  - mode × level (6 cells)
  - condition × level (14 cells)
  - condition × mode (21 cells)
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# reuse from main analysis
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stat_analysis_exp2_v2 import (
    load_all_data,
    eta_squared_from_H,
    bonferroni,
    cliff_delta,
    cliff_label,
    MODES, DISTANCES, LEVELS, CONDITIONS_7,
    MODE_LABEL, LEVEL_LABEL,
)

METRICS = [("f2", "F2-score"), ("auc", "AUROC"), ("f1", "F1-score"),
           ("auc_pr", "AUPRC"), ("recall", "Recall")]


def kw_test(groups):
    """KW test with guard for insufficient data."""
    valid = [g for g in groups if len(g) >= 2]
    if len(valid) < 2:
        return np.nan, np.nan, np.nan
    n = sum(len(g) for g in valid)
    H, p = stats.kruskal(*valid)
    eta2 = eta_squared_from_H(H, n, len(valid))
    return H, p, eta2


def dunn_posthoc(df_sub, metric):
    """Dunn's test between distance groups within df_sub."""
    from scikit_posthocs import posthoc_dunn
    # need at least 2 groups with >=2 obs
    groups = df_sub.groupby("distance")[metric]
    valid_groups = {k: v.values for k, v in groups if len(v) >= 2}
    if len(valid_groups) < 2:
        return None
    sub = df_sub[df_sub["distance"].isin(valid_groups.keys())]
    return posthoc_dunn(sub, val_col=metric, group_col="distance", p_adjust="bonferroni")


def main():
    df = load_all_data()
    lines = []
    w = lines.append

    w("# Distance Metric Granular Analysis\n")
    w(f"Records: {len(df)}, Distances: {DISTANCES}\n")
    w("H₀ for each cell: performance under MMD = DTW = Wasserstein (Kruskal-Wallis)\n")
    w("---\n")

    for metric, mlabel in METRICS:
        w(f"\n## {mlabel}\n")

        # ── 1. By Mode (pooling conditions & levels) ──
        w(f"### 1. Distance effect by Mode (pooling conditions & levels)\n")
        w("| Mode | N | H | p | η² | Sig? |")
        w("|------|--:|--:|--:|---:|:----:|")
        for mode in MODES:
            sub = df[df["mode"] == mode]
            groups = [sub[sub["distance"] == d][metric].dropna().values for d in DISTANCES]
            H, p, eta2 = kw_test(groups)
            sig = "✓" if (not np.isnan(p) and p < 0.05) else "✗"
            n = sum(len(g) for g in groups)
            w(f"| {MODE_LABEL[mode]} | {n} | {H:.2f} | {p:.4f} | {eta2:.4f} | {sig} |")
        w("")

        # Means per mode × distance
        w("**Mean ± SD per mode × distance:**\n")
        w("| Mode | MMD | DTW | Wasserstein |")
        w("|------|----:|----:|------------:|")
        for mode in MODES:
            vals = []
            for d in DISTANCES:
                s = df[(df["mode"] == mode) & (df["distance"] == d)][metric]
                vals.append(f"{s.mean():.4f}±{s.std():.4f}")
            w(f"| {MODE_LABEL[mode]} | {' | '.join(vals)} |")
        w("")

        # ── 2. By Condition (pooling modes & levels) ──
        w(f"### 2. Distance effect by Condition (pooling modes & levels)\n")
        w("| Condition | N | H | p | η² | Sig? |")
        w("|-----------|--:|--:|--:|---:|:----:|")
        for cond in CONDITIONS_7:
            sub = df[df["condition"] == cond]
            groups = [sub[sub["distance"] == d][metric].dropna().values for d in DISTANCES]
            H, p, eta2 = kw_test(groups)
            sig = "✓" if (not np.isnan(p) and p < 0.05) else "✗"
            n = sum(len(g) for g in groups)
            w(f"| {cond} | {n} | {H:.2f} | {p:.4f} | {eta2:.4f} | {sig} |")
        w("")

        # ── 3. By Level (pooling modes & conditions) ──
        w(f"### 3. Distance effect by Level (pooling modes & conditions)\n")
        w("| Level | N | H | p | η² | Sig? |")
        w("|-------|--:|--:|--:|---:|:----:|")
        for level in LEVELS:
            sub = df[df["level"] == level]
            groups = [sub[sub["distance"] == d][metric].dropna().values for d in DISTANCES]
            H, p, eta2 = kw_test(groups)
            sig = "✓" if (not np.isnan(p) and p < 0.05) else "✗"
            n = sum(len(g) for g in groups)
            w(f"| {LEVEL_LABEL[level]} | {n} | {H:.2f} | {p:.4f} | {eta2:.4f} | {sig} |")
        w("")

        # ── 4. By Mode × Level (pooling conditions) ──
        w(f"### 4. Distance effect by Mode × Level (pooling conditions)\n")
        w("| Mode | Level | N | H | p | η² | Sig? |")
        w("|------|-------|--:|--:|--:|---:|:----:|")
        for mode in MODES:
            for level in LEVELS:
                sub = df[(df["mode"] == mode) & (df["level"] == level)]
                groups = [sub[sub["distance"] == d][metric].dropna().values for d in DISTANCES]
                H, p, eta2 = kw_test(groups)
                sig = "✓" if (not np.isnan(p) and p < 0.05) else "✗"
                n = sum(len(g) for g in groups)
                w(f"| {MODE_LABEL[mode]} | {LEVEL_LABEL[level]} | {n} | {H:.2f} | {p:.4f} | {eta2:.4f} | {sig} |")
        w("")

        # ── 5. By Condition × Level (pooling modes) ──
        w(f"### 5. Distance effect by Condition × Level (pooling modes)\n")
        w("| Condition | Level | N | H | p | η² | Sig? |")
        w("|-----------|-------|--:|--:|--:|---:|:----:|")
        for cond in CONDITIONS_7:
            for level in LEVELS:
                sub = df[(df["condition"] == cond) & (df["level"] == level)]
                groups = [sub[sub["distance"] == d][metric].dropna().values for d in DISTANCES]
                H, p, eta2 = kw_test(groups)
                sig = "✓" if (not np.isnan(p) and p < 0.05) else "✗"
                n = sum(len(g) for g in groups)
                w(f"| {cond} | {LEVEL_LABEL[level]} | {n} | {H:.2f} | {p:.4f} | {eta2:.4f} | {sig} |")
        w("")

        # ── 6. By Condition × Mode (pooling levels) ──
        w(f"### 6. Distance effect by Condition × Mode (pooling levels)\n")
        w("| Condition | Mode | N | H | p | η² | Sig? |")
        w("|-----------|------|--:|--:|--:|---:|:----:|")
        for cond in CONDITIONS_7:
            for mode in MODES:
                sub = df[(df["condition"] == cond) & (df["mode"] == mode)]
                groups = [sub[sub["distance"] == d][metric].dropna().values for d in DISTANCES]
                H, p, eta2 = kw_test(groups)
                sig = "✓" if (not np.isnan(p) and p < 0.05) else "✗"
                n = sum(len(g) for g in groups)
                w(f"| {cond} | {MODE_LABEL[mode]} | {n} | {H:.2f} | {p:.4f} | {eta2:.4f} | {sig} |")
        w("")

    # ── Summary ──
    w("\n---\n## Summary\n")
    w("The above tables show Kruskal-Wallis tests for distance metric effect "
      "at each level of stratification. Significance is assessed at α=0.05 "
      "(uncorrected per-test). For multiple testing across cells within a section, "
      "apply Bonferroni correction as needed.\n")

    report = "\n".join(lines)
    out_path = Path(__file__).resolve().parents[4] / "results" / "analysis" / "exp2_domain_shift" / "distance_granular_report.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"Report written to {out_path}")
    print(f"Lines: {len(lines)}")

    # Also print to stdout for quick review
    print("\n" + report)


if __name__ == "__main__":
    main()
