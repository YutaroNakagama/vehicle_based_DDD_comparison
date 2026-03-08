#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_journal_figures.py
=======================
Generate publication-quality figures for international journal submission.

Figures generated:
  1. cd_diagram_f2.png / cd_diagram_auc.png     — Critical Difference diagrams
  2. violin_f2.png / violin_auc.png             — Violin plots (condition × mode)
  3. ci_forest_f2.png / ci_forest_auc.png       — Bootstrap CI forest plots
  4. convergence_f2.png / convergence_auc.png   — Seed convergence σ_rank(k) plots

Output:
  results/analysis/exp2_domain_shift/figures/png/split2/journal/

Usage:
    python scripts/python/analysis/domain/plot_journal_figures.py
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy import stats

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

CSV_BASE = (
    PROJECT_ROOT / "results" / "analysis" / "exp2_domain_shift"
    / "figures" / "csv" / "split2"
)
OUT_DIR = (
    PROJECT_ROOT / "results" / "analysis" / "exp2_domain_shift"
    / "figures" / "png" / "split2" / "journal"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants (matching stat_analysis_exp2_v2.py)
# ---------------------------------------------------------------------------
CONDITIONS_7 = [
    "baseline", "rus_r01", "rus_r05",
    "smote_r01", "smote_r05", "sw_smote_r01", "sw_smote_r05",
]
COND_LABELS = {
    "baseline":      "Baseline",
    "rus_r01":       "RUS r=0.1",
    "rus_r05":       "RUS r=0.5",
    "smote_r01":     "SMOTE r=0.1",
    "smote_r05":     "SMOTE r=0.5",
    "sw_smote_r01":  "SW-SMOTE r=0.1",
    "sw_smote_r05":  "SW-SMOTE r=0.5",
}
COND_COLORS = {
    "baseline":      "#95a5a6",
    "rus_r01":       "#27ae60",
    "rus_r05":       "#1e8449",
    "smote_r01":     "#3498db",
    "smote_r05":     "#2980b9",
    "sw_smote_r01":  "#e67e22",
    "sw_smote_r05":  "#d35400",
}
MODE_LABEL = {
    "source_only": "Cross-domain",
    "target_only": "Within-domain",
    "mixed": "Mixed",
}
LEVEL_LABEL = {"in_domain": "In-domain", "out_domain": "Out-domain"}
MODES = ["source_only", "target_only", "mixed"]
DISTANCES = ["mmd", "dtw", "wasserstein"]
LEVELS = ["in_domain", "out_domain"]
METRICS = [("f2", "F2-score"), ("auc", "AUROC")]


# ---------------------------------------------------------------------------
# Data loading (same as stat_analysis_exp2_v2.py)
# ---------------------------------------------------------------------------
def load_all_data() -> pd.DataFrame:
    files = {
        "baseline":  CSV_BASE / "baseline"        / "baseline_domain_split2_metrics_v2.csv",
        "smote":     CSV_BASE / "smote_plain"      / "smote_plain_split2_metrics_v2.csv",
        "rus":       CSV_BASE / "undersample_rus"  / "undersample_rus_split2_metrics_v2.csv",
        "sw_smote":  CSV_BASE / "sw_smote"         / "sw_smote_split2_metrics_v2.csv",
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
    merged = merged[merged["condition"].isin(CONDITIONS_7)].copy()
    return merged


def _save(fig, name: str):
    out = OUT_DIR / name
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Figure 1: Critical Difference Diagram
# ---------------------------------------------------------------------------
def _draw_cd_diagram(ax, ranks: dict, cd: float, title: str):
    """Draw a Critical Difference diagram on the given axes.

    Args:
        ranks: {condition_name: mean_rank}
        cd: critical difference value
        title: subplot title
    """
    k = len(ranks)
    sorted_items = sorted(ranks.items(), key=lambda x: x[1])
    names = [COND_LABELS.get(n, n) for n, _ in sorted_items]
    rvals = [r for _, r in sorted_items]

    # Layout
    lo, hi = 1, k
    ax.set_xlim(lo - 0.5, hi + 0.5)
    ax.set_ylim(-0.3, 1.6)
    ax.invert_xaxis()  # rank 1 on right

    # Horizontal axis
    ax.hlines(1.4, lo, hi, color="black", linewidth=0.8)
    for r in range(lo, hi + 1):
        ax.vlines(r, 1.35, 1.45, color="black", linewidth=0.8)
        ax.text(r, 1.52, str(r), ha="center", va="bottom", fontsize=8)

    # CD bar
    cd_x0 = lo
    ax.hlines(1.7, cd_x0, cd_x0 + cd, color="red", linewidth=2)
    ax.text(cd_x0 + cd / 2, 1.78, f"CD={cd:.2f}", ha="center", va="bottom",
            fontsize=8, color="red", fontweight="bold")

    # Place labels: left side (low rank = good) and right side (high rank = bad)
    mid = k / 2.0
    left_items = [(n, r) for n, r in zip(names, rvals) if r <= mid + 0.5]
    right_items = [(n, r) for n, r in zip(names, rvals) if r > mid + 0.5]

    y_step = 0.18
    # Left side (top performers, rank 1 = best, drawn on right of plot since inverted)
    for i, (name, r) in enumerate(left_items):
        y = 1.2 - (i + 1) * y_step
        ax.hlines(y, r, r, color="none")  # placeholder
        ax.plot(r, 1.4, "o", color="black", markersize=4, zorder=5)
        ax.vlines(r, y, 1.4, color="black", linewidth=0.6)
        ax.text(r + 0.15, y, f"{name} ({r:.2f})", ha="left", va="center",
                fontsize=7.5)

    # Right side (worse performers)
    for i, (name, r) in enumerate(right_items):
        y = 1.2 - (i + 1) * y_step
        ax.plot(r, 1.4, "o", color="black", markersize=4, zorder=5)
        ax.vlines(r, y, 1.4, color="black", linewidth=0.6)
        ax.text(r - 0.15, y, f"({r:.2f}) {name}", ha="right", va="center",
                fontsize=7.5)

    # Connect groups that are NOT significantly different (|rank_i - rank_j| < CD)
    raw_names = [n for n, _ in sorted_items]
    groups = []
    for i in range(k):
        for j in range(i + 1, k):
            if abs(rvals[j] - rvals[i]) < cd:
                # Check if already part of an existing group
                merged = False
                for g in groups:
                    if i in g or j in g:
                        g.add(i)
                        g.add(j)
                        merged = True
                        break
                if not merged:
                    groups.append({i, j})

    # Merge overlapping groups
    changed = True
    while changed:
        changed = False
        new_groups = []
        used = set()
        for i, g1 in enumerate(groups):
            if i in used:
                continue
            for j, g2 in enumerate(groups):
                if j <= i or j in used:
                    continue
                if g1 & g2:
                    g1 = g1 | g2
                    used.add(j)
                    changed = True
            new_groups.append(g1)
            used.add(i)
        groups = new_groups

    # Draw group bars
    bar_y_start = -0.05
    for gi, g in enumerate(groups):
        g_sorted = sorted(g)
        r_min = rvals[g_sorted[0]]
        r_max = rvals[g_sorted[-1]]
        bar_y = bar_y_start - gi * 0.1
        ax.hlines(bar_y, r_min, r_max, color="black", linewidth=3, alpha=0.7)

    ax.set_title(title, fontsize=11, fontweight="bold", pad=35)
    ax.axis("off")


def plot_cd_diagrams(df: pd.DataFrame):
    """Generate Critical Difference diagrams for F2 and AUROC."""
    for metric, mlabel in METRICS:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        for ax_idx, level in enumerate(LEVELS):
            # Build pivot: rows=seeds, columns=conditions
            sub = df[df["level"] == level]
            pivot = sub.groupby(["seed", "condition"])[metric].mean().reset_index()
            pivot_wide = pivot.pivot(index="seed", columns="condition", values=metric)
            pivot_wide = pivot_wide[CONDITIONS_7].dropna()

            if len(pivot_wide) < 3:
                axes[ax_idx].text(0.5, 0.5, "Insufficient data",
                                  ha="center", va="center", transform=axes[ax_idx].transAxes)
                continue

            # Friedman test
            arrays = [pivot_wide[c].values for c in CONDITIONS_7]
            chi2, p_fri = stats.friedmanchisquare(*arrays)

            # Mean ranks (higher metric = rank 1)
            mean_ranks = pivot_wide.rank(axis=1, ascending=False).mean()
            ranks_dict = {c: mean_ranks[c] for c in CONDITIONS_7}

            # Critical Difference (Nemenyi)
            k = len(CONDITIONS_7)
            n_blocks = len(pivot_wide)
            from scipy.stats import studentized_range
            q_crit = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)
            cd = q_crit * np.sqrt(k * (k + 1) / (6 * n_blocks))

            title = (f"{LEVEL_LABEL[level]} — {mlabel}\n"
                     f"Friedman χ²={chi2:.2f}, p={p_fri:.4f}, CD={cd:.2f}")
            _draw_cd_diagram(axes[ax_idx], ranks_dict, cd, title)

        fig.suptitle(f"Critical Difference Diagram — {mlabel}",
                     fontsize=14, fontweight="bold", y=1.02)
        fig.tight_layout()
        _save(fig, f"cd_diagram_{metric}.png")


# ---------------------------------------------------------------------------
# Figure 2: Violin / Box Plots
# ---------------------------------------------------------------------------
def plot_violin(df: pd.DataFrame):
    """Violin + box plots: condition × mode, split by level."""
    for metric, mlabel in METRICS:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)

        for row_idx, level in enumerate(LEVELS):
            for col_idx, mode in enumerate(MODES):
                ax = axes[row_idx, col_idx]
                sub = df[(df["mode"] == mode) & (df["level"] == level)].copy()
                sub["cond_label"] = sub["condition"].map(COND_LABELS)

                order = [COND_LABELS[c] for c in CONDITIONS_7]
                palette = [COND_COLORS[c] for c in CONDITIONS_7]

                # Violin
                parts = ax.violinplot(
                    [sub[sub["condition"] == c][metric].dropna().values
                     for c in CONDITIONS_7],
                    positions=range(len(CONDITIONS_7)),
                    showmeans=False, showmedians=False, showextrema=False,
                )
                for pc, color in zip(parts["bodies"], palette):
                    pc.set_facecolor(color)
                    pc.set_alpha(0.35)

                # Box plot overlay
                bp = ax.boxplot(
                    [sub[sub["condition"] == c][metric].dropna().values
                     for c in CONDITIONS_7],
                    positions=range(len(CONDITIONS_7)),
                    widths=0.3, showfliers=True,
                    patch_artist=True,
                    flierprops=dict(marker="o", markersize=3, alpha=0.5),
                    medianprops=dict(color="black", linewidth=1.5),
                )
                for patch, color in zip(bp["boxes"], palette):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

                ax.set_xticks(range(len(CONDITIONS_7)))
                ax.set_xticklabels(order, rotation=45, ha="right", fontsize=7)
                ax.set_title(f"{MODE_LABEL[mode]} — {LEVEL_LABEL[level]}",
                             fontsize=10, fontweight="bold")
                if col_idx == 0:
                    ax.set_ylabel(mlabel, fontsize=10)
                ax.grid(axis="y", alpha=0.3, linestyle="--")

        fig.suptitle(f"Distribution of {mlabel} by Condition, Mode, and Domain Level",
                     fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        _save(fig, f"violin_{metric}.png")


# ---------------------------------------------------------------------------
# Figure 3: CI Forest Plot
# ---------------------------------------------------------------------------
def bootstrap_ci_bca(data, B=10000, alpha=0.05, rng=None):
    """BCa bootstrap CI (simplified from stat_analysis_exp2_v2.py)."""
    if rng is None:
        rng = np.random.RandomState(42)
    n = len(data)
    if n < 2:
        return np.nan, np.nan, np.nan
    theta_hat = float(np.mean(data))
    boot = np.array([np.mean(data[rng.randint(0, n, n)]) for _ in range(B)])
    # Bias correction
    prop = np.clip(np.mean(boot < theta_hat), 1e-10, 1 - 1e-10)
    z0 = stats.norm.ppf(prop)
    # Acceleration (jackknife)
    jk = np.array([np.mean(np.delete(data, i)) for i in range(n)])
    jk_mean = jk.mean()
    diffs = jk_mean - jk
    num = np.sum(diffs ** 3)
    den = 6.0 * (np.sum(diffs ** 2) ** 1.5)
    a = num / den if abs(den) > 1e-15 else 0.0
    z_lo, z_hi = stats.norm.ppf(alpha / 2), stats.norm.ppf(1 - alpha / 2)

    def adj(z):
        numer = z0 + z
        d = 1 - a * numer
        return stats.norm.cdf(z0 + numer / d) if abs(d) > 1e-15 else 0.5

    lo = np.nanpercentile(boot, 100 * np.clip(adj(z_lo), 0.001, 0.999))
    hi = np.nanpercentile(boot, 100 * np.clip(adj(z_hi), 0.001, 0.999))
    return theta_hat, lo, hi


def plot_ci_forest(df: pd.DataFrame):
    """Forest plot of bootstrap CIs by condition × mode × level."""
    rng = np.random.RandomState(42)

    for metric, mlabel in METRICS:
        fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)

        for ax_idx, level in enumerate(LEVELS):
            ax = axes[ax_idx]
            y_pos = 0
            y_ticks = []
            y_labels = []

            for mode in reversed(MODES):
                for cond in reversed(CONDITIONS_7):
                    sub = df[(df["condition"] == cond) & (df["mode"] == mode)
                             & (df["level"] == level)]
                    seed_means = sub.groupby("seed")[metric].mean().dropna().values

                    if len(seed_means) >= 3:
                        est, lo, hi = bootstrap_ci_bca(seed_means, B=10000, rng=rng)
                    else:
                        est = np.mean(seed_means) if len(seed_means) else np.nan
                        lo, hi = np.nan, np.nan

                    color = COND_COLORS.get(cond, "#999")
                    ax.plot(est, y_pos, "o", color=color, markersize=6, zorder=5)
                    if not np.isnan(lo):
                        ax.hlines(y_pos, lo, hi, color=color, linewidth=2, zorder=4)

                    y_ticks.append(y_pos)
                    y_labels.append(f"{COND_LABELS[cond]}")
                    y_pos += 1

                # Add mode separator
                if mode != MODES[0]:
                    ax.axhline(y_pos - 0.5, color="gray", linewidth=0.5,
                               linestyle="--", alpha=0.5)
                y_pos += 0.5

            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_labels, fontsize=7.5)
            ax.set_xlabel(mlabel, fontsize=10)
            ax.set_title(f"{LEVEL_LABEL[level]}", fontsize=11, fontweight="bold")
            ax.grid(axis="x", alpha=0.3, linestyle="--")

            # Add mode labels on right side
            y_curr = 0
            for mode in reversed(MODES):
                mid_y = y_curr + (len(CONDITIONS_7) - 1) / 2
                ax.text(ax.get_xlim()[1], mid_y, f" {MODE_LABEL[mode]}",
                        ha="left", va="center", fontsize=8, fontstyle="italic",
                        color="gray")
                y_curr += len(CONDITIONS_7) + 0.5

        fig.suptitle(f"Bootstrap 95% CI (BCa, B=10,000) — {mlabel}",
                     fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 0.92, 0.96])
        _save(fig, f"ci_forest_{metric}.png")


# ---------------------------------------------------------------------------
# Figure 4: Seed Convergence Plot
# ---------------------------------------------------------------------------
def seed_convergence_data(df, metric, seeds, max_subsets=500, rng=None):
    """Compute ranking std for each k (same logic as stat_analysis_exp2_v2.py)."""
    if rng is None:
        rng = np.random.RandomState(42)
    all_seeds = sorted(seeds)
    n_seeds = len(all_seeds)
    ks = [k for k in [3, 5, 7, 9, 11] if k < n_seeds]
    results = {}
    for k in ks:
        all_combos = list(combinations(all_seeds, k))
        if len(all_combos) > max_subsets:
            idx = rng.choice(len(all_combos), max_subsets, replace=False)
            combos_used = [all_combos[i] for i in idx]
        else:
            combos_used = all_combos
        rankings = []
        for seed_subset in combos_used:
            sub_df = df[df["seed"].isin(seed_subset)]
            means = {}
            for cond in CONDITIONS_7:
                v = sub_df[sub_df["condition"] == cond][metric].dropna()
                means[cond] = v.mean() if len(v) > 0 else np.nan
            sorted_conds = sorted(means, key=lambda c: means[c], reverse=True)
            ranks = {c: r + 1 for r, c in enumerate(sorted_conds)}
            rankings.append(ranks)
        rank_df = pd.DataFrame(rankings)
        results[k] = {
            "mean_std": rank_df.std().mean(),
            "max_std": rank_df.std().max(),
            "per_cond_std": rank_df.std().to_dict(),
        }
    return results


def plot_convergence(df: pd.DataFrame):
    """Plot σ_rank(k) vs k for both metrics."""
    seeds = sorted(int(s) for s in df["seed"].unique())
    rng = np.random.RandomState(42)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax_idx, (metric, mlabel) in enumerate(METRICS):
        ax = axes[ax_idx]
        print(f"  Computing convergence for {mlabel}...")
        conv = seed_convergence_data(df, metric, seeds, max_subsets=500, rng=rng)

        ks = sorted(conv.keys())
        mean_stds = [conv[k]["mean_std"] for k in ks]
        max_stds = [conv[k]["max_std"] for k in ks]

        ax.plot(ks, mean_stds, "o-", color="#2c3e50", linewidth=2,
                markersize=8, label="Mean σ_rank", zorder=5)
        ax.fill_between(ks, 0, max_stds, alpha=0.15, color="#3498db",
                        label="Max σ_rank")
        ax.plot(ks, max_stds, "s--", color="#3498db", linewidth=1,
                markersize=6, alpha=0.7)

        # Per-condition lines
        for cond in CONDITIONS_7:
            per_cond = [conv[k]["per_cond_std"].get(cond, np.nan) for k in ks]
            ax.plot(ks, per_cond, "-", color=COND_COLORS[cond],
                    linewidth=0.8, alpha=0.5)

        # Reference lines
        ax.axhline(0.5, color="red", linewidth=0.8, linestyle=":",
                   alpha=0.6, label="σ=0.5 threshold")

        ax.set_xlabel("Number of Seeds (k)", fontsize=10)
        ax.set_ylabel("σ_rank (ranking standard deviation)", fontsize=10)
        ax.set_title(mlabel, fontsize=12, fontweight="bold")
        ax.set_xticks(ks)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3, linestyle="--")
        ax.set_ylim(bottom=0)

    fig.suptitle("Seed Count Convergence Analysis\n"
                 "Ranking stability as a function of seed subset size",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _save(fig, "convergence_seed.png")


# ---------------------------------------------------------------------------
# Figure 5: Multiple Comparison Correction Flow Diagram
# ---------------------------------------------------------------------------
def plot_mc_flow(df: pd.DataFrame):
    """Generate a visual diagram of the multiple comparison correction strategy."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    box_kw = dict(boxstyle="round,pad=0.4", facecolor="#ecf0f1", edgecolor="#2c3e50",
                  linewidth=1.5)
    box_blue = dict(boxstyle="round,pad=0.4", facecolor="#d4e6f1", edgecolor="#2980b9",
                    linewidth=1.5)
    box_green = dict(boxstyle="round,pad=0.4", facecolor="#d5f5e3", edgecolor="#27ae60",
                     linewidth=1.5)
    box_orange = dict(boxstyle="round,pad=0.4", facecolor="#fdebd0", edgecolor="#e67e22",
                      linewidth=1.5)

    # Title
    ax.text(5, 9.5, "Multiple Comparison Correction Framework",
            ha="center", va="center", fontsize=14, fontweight="bold")

    # Box 1: Raw p-values
    ax.text(5, 8.5, "Raw p-values from\nhypothesis test families\n(H1–H14)",
            ha="center", va="center", fontsize=10, bbox=box_kw)
    ax.annotate("", xy=(5, 7.6), xytext=(5, 7.9),
                arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1.5))

    # Decision diamond
    ax.text(5, 7.2, "Family-wise\ntesting?",
            ha="center", va="center", fontsize=10, fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fef9e7",
                      edgecolor="#f39c12", linewidth=1.5))

    # YES branch (left)
    ax.annotate("YES", xy=(2.5, 6.3), xytext=(4.0, 6.8),
                fontsize=9, color="#27ae60", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.5))
    ax.text(2.5, 5.8, "Primary: Bonferroni\n(FWER control)\nα' = α / m",
            ha="center", va="center", fontsize=9, bbox=box_blue)

    # NO branch (right)
    ax.annotate("NO", xy=(7.5, 6.3), xytext=(6.0, 6.8),
                fontsize=9, color="#e74c3c", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.5))
    ax.text(7.5, 5.8, "No correction\n(single test)\nα = 0.05",
            ha="center", va="center", fontsize=9, bbox=box_kw)

    # Arrow down from Bonferroni
    ax.annotate("", xy=(2.5, 4.6), xytext=(2.5, 5.1),
                arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1.5))

    # Sensitivity analysis
    ax.text(2.5, 4.1, "Sensitivity: BH-FDR\n(false discovery rate)\np(i) ≤ (i/m)·α",
            ha="center", va="center", fontsize=9, bbox=box_green)

    # Arrow down from BH-FDR
    ax.annotate("", xy=(2.5, 3.0), xytext=(2.5, 3.4),
                arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1.5))

    # Post-hoc
    ax.text(2.5, 2.5, "Post-hoc: Nemenyi\n(Friedman + CD)\nFor overall ranking",
            ha="center", va="center", fontsize=9, bbox=box_orange)

    # Families table (right side, lower)
    families = [
        ("H1 KW", "18/metric", "Bonf.+BH"),
        ("H1 pairwise", "36/metric", "Bonf.+BH"),
        ("H2 sw vs smote", "12/metric", "Bonf.+BH"),
        ("H3 over vs RUS", "48/metric", "Bonf.+BH"),
        ("H4 ratio", "18/metric", "Bonf.+BH"),
        ("H10 domain", "63/metric", "Bonf.+BH"),
        ("Ranking", "C(7,2)=21", "Nemenyi CD"),
    ]
    y_start = 4.5
    ax.text(7.5, y_start + 0.5, "Applied Corrections:", ha="center",
            fontsize=9, fontweight="bold")
    for i, (hyp, m, corr) in enumerate(families):
        y = y_start - i * 0.4
        ax.text(6.0, y, hyp, fontsize=7.5, va="center")
        ax.text(7.8, y, f"m={m}", fontsize=7.5, va="center", ha="center")
        ax.text(9.2, y, corr, fontsize=7.5, va="center", ha="center",
                color="#2980b9" if "Bonf" in corr else "#e67e22")

    # Bottom summary
    ax.text(5, 0.8, "Conservative (Bonferroni) → confirmatory conclusions\n"
            "Liberal (BH-FDR) → exploratory additional discoveries\n"
            "Nemenyi CD → overall ranking with pairwise guarantees",
            ha="center", va="center", fontsize=8.5, fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fadbd8",
                      edgecolor="#e74c3c", linewidth=1))

    _save(fig, "mc_correction_flow.png")


# ---------------------------------------------------------------------------
# Figure 6: Effect Size Heatmap
# ---------------------------------------------------------------------------
def cliff_delta(x, y):
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return np.nan
    more = sum(1 for xi in x for yj in y if xi > yj)
    less = sum(1 for xi in x for yj in y if xi < yj)
    return (more - less) / (nx * ny)


def plot_effect_heatmap(df: pd.DataFrame):
    """7×7 heatmap of pairwise Cliff's δ between conditions."""
    for metric, mlabel in METRICS:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax_idx, level in enumerate(LEVELS):
            ax = axes[ax_idx]
            n = len(CONDITIONS_7)
            delta_mat = np.zeros((n, n))
            for i, ci in enumerate(CONDITIONS_7):
                for j, cj in enumerate(CONDITIONS_7):
                    if i == j:
                        delta_mat[i, j] = 0.0
                    elif i < j:
                        vi = df[(df["condition"] == ci) & (df["level"] == level)][metric].dropna().values
                        vj = df[(df["condition"] == cj) & (df["level"] == level)][metric].dropna().values
                        d = cliff_delta(vi, vj)
                        delta_mat[i, j] = d
                        delta_mat[j, i] = -d

            im = ax.imshow(delta_mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
            ax.set_xticks(range(n))
            ax.set_xticklabels([COND_LABELS[c] for c in CONDITIONS_7],
                               rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(n))
            ax.set_yticklabels([COND_LABELS[c] for c in CONDITIONS_7], fontsize=8)
            ax.set_title(LEVEL_LABEL[level], fontsize=11, fontweight="bold")

            for i in range(n):
                for j in range(n):
                    val = delta_mat[i, j]
                    color = "white" if abs(val) > 0.5 else "black"
                    ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                            fontsize=6.5, color=color)

        fig.colorbar(im, ax=axes, label="Cliff's δ (row − column)",
                     fraction=0.02, pad=0.04)
        fig.suptitle(f"Pairwise Cliff's δ Effect Size — {mlabel}",
                     fontsize=13, fontweight="bold")
        fig.subplots_adjust(top=0.90, right=0.92)
        _save(fig, f"effect_heatmap_{metric}.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Journal Figure Generation")
    print("=" * 60)

    df = load_all_data()
    print(f"Loaded {len(df)} records")

    print("\n[1/6] Critical Difference diagrams...")
    plot_cd_diagrams(df)

    print("[2/6] Violin + box plots...")
    plot_violin(df)

    print("[3/6] CI forest plots...")
    plot_ci_forest(df)

    print("[4/6] Seed convergence plots...")
    plot_convergence(df)

    print("[5/6] Multiple comparison flow diagram...")
    plot_mc_flow(df)

    print("[6/6] Effect size heatmap...")
    plot_effect_heatmap(df)

    print("\n" + "=" * 60)
    print(f"All figures saved to: {OUT_DIR.relative_to(PROJECT_ROOT)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
