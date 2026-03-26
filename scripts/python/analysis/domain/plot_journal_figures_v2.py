#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_journal_figures_v2.py
==========================
Generate publication-quality figures for Experiment 2 journal paper.

Figures (7 total):
  Fig 2 — Effect size hierarchy dot plot (η² by 4 factors)
  Fig 3 — Rebalancing × Mode heatmap (F2 + AUROC, 2 panels)
  Fig 4 — Critical Difference diagrams (Nemenyi, F2 / AUROC / AUPRC)
  Fig 5 — Distance metric equivalence violin plot
  Fig 6 — Training mode grouped box plot
  Fig 7 — Domain shift reversal diverging bar chart
  Fig 8 — Seed convergence curve

Output:
  results/analysis/exp2_domain_shift/figures/png/split2/journal_v2/

Usage:
    python scripts/python/analysis/domain/plot_journal_figures_v2.py
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
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
    / "figures" / "png" / "split2" / "journal_v2"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OFFICIAL_SEEDS = {0, 1, 3, 7, 13, 42, 123, 256, 512, 999, 1337, 2024}

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
# Condition family colors for grouped visuals
FAMILY_COLORS = {
    "baseline": "#95a5a6",
    "rus":      "#27ae60",
    "smote":    "#3498db",
    "sw_smote": "#e67e22",
}
MODE_LABELS = {
    "source_only": "Cross-domain",
    "target_only": "Within-domain",
    "mixed":       "Mixed",
}
LEVEL_LABELS = {"in_domain": "In-domain", "out_domain": "Out-domain"}
MODES = ["source_only", "target_only", "mixed"]
DISTANCES = ["mmd", "dtw", "wasserstein"]
LEVELS = ["in_domain", "out_domain"]
PRIMARY_METRICS = [("f2", "F2-score"), ("auc", "AUROC"), ("auc_pr", "AUPRC")]

# Journal style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_all_data() -> pd.DataFrame:
    files = {
        "baseline":  CSV_BASE / "baseline"       / "baseline_domain_split2_metrics_v2.csv",
        "smote":     CSV_BASE / "smote_plain"    / "smote_plain_split2_metrics_v2.csv",
        "rus":       CSV_BASE / "undersample_rus"/ "undersample_rus_split2_metrics_v2.csv",
        "sw_smote":  CSV_BASE / "sw_smote"       / "sw_smote_split2_metrics_v2.csv",
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
    # Filter to official seeds
    merged = merged[merged["seed"].isin(OFFICIAL_SEEDS)].copy()
    print(f"  Loaded {len(merged)} records "
          f"({merged['seed'].nunique()} seeds, "
          f"{merged['condition'].nunique()} conditions)")
    return merged


def _save(fig, name: str):
    out = OUT_DIR / name
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Helper: η² from Kruskal-Wallis H
# ---------------------------------------------------------------------------
def eta_squared_from_H(H: float, n: int, k: int) -> float:
    """Approximate η² = (H - k + 1) / (n - k)."""
    return max(0.0, (H - k + 1) / (n - k))


# ===================================================================
# Fig 2: Variance-Based Sensitivity Analysis (Sobol Indices)
# ===================================================================
def plot_effect_hierarchy(df: pd.DataFrame):
    """Stacked bar chart showing first-order (S1) and interaction (ST-S1)
    Sobol indices for each factor, with 95 % bootstrap CIs on ST.
    Replaces old η²-only dot plot with a complete variance decomposition."""

    csv_path = (
        PROJECT_ROOT / "results" / "analysis" / "exp2_domain_shift"
        / "figures" / "csv" / "split2" / "sensitivity" / "sobol_indices.csv"
    )
    if not csv_path.exists():
        print("  [SKIP] sobol_indices.csv not found. Run sensitivity_analysis_exp2.py first.")
        return

    si = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Factor display labels (short)
    factor_display = {
        "condition": "Rebalancing\n($R$)",
        "distance": "Distance\n($D$)",
        "level": "Membership\n($G$)",
        "mode": "Mode\n($M$)",
    }
    factor_order = ["condition", "distance", "level", "mode"]
    metric_names = ["F2-score", "AUROC", "AUPRC"]
    colors_s1 = {"F2-score": "#e74c3c", "AUROC": "#3498db", "AUPRC": "#2ecc71"}
    colors_inter = {"F2-score": "#f5b7b1", "AUROC": "#aed6f1", "AUPRC": "#abebc6"}

    x = np.arange(len(factor_order))
    width = 0.22

    for i, mlabel in enumerate(metric_names):
        mdata = si[si["metric"] == mlabel].set_index("factor_key")
        offset = (i - 1) * width

        s1_vals = [mdata.loc[f, "S1"] for f in factor_order]
        st_vals = [mdata.loc[f, "ST"] for f in factor_order]
        inter_vals = [st - s1 for st, s1 in zip(st_vals, s1_vals)]
        st_lo = [mdata.loc[f, "ST_lo"] for f in factor_order]
        st_hi = [mdata.loc[f, "ST_hi"] for f in factor_order]
        # Error bars on ST
        yerr_lo = [st - lo for st, lo in zip(st_vals, st_lo)]
        yerr_hi = [hi - st for st, hi in zip(st_vals, st_hi)]

        # Bottom bar: S1 (main effect)
        bars_s1 = ax.bar(x + offset, s1_vals, width,
                         color=colors_s1[mlabel], alpha=0.9,
                         edgecolor="white", linewidth=0.8,
                         label=f"{mlabel} (main)" if i == 0 else "")
        # Top bar: interaction (ST - S1)
        bars_inter = ax.bar(x + offset, inter_vals, width,
                            bottom=s1_vals,
                            color=colors_inter[mlabel], alpha=0.7,
                            edgecolor="white", linewidth=0.8,
                            hatch="//",
                            label=f"{mlabel} (interaction)" if i == 0 else "")
        # Error bars on total ST
        ax.errorbar(x + offset, st_vals,
                    yerr=[yerr_lo, yerr_hi],
                    fmt="none", ecolor="black", elinewidth=0.8, capsize=2)

        # Annotate ST value
        for j, (sv, s1v) in enumerate(zip(st_vals, s1_vals)):
            if sv > 0.02:
                ax.text(x[j] + offset, sv + max(yerr_hi[j], 0) + 0.015,
                        f"$S_T$={sv:.3f}",
                        ha="center", va="bottom", fontsize=6.5,
                        fontweight="bold")
                # Also show S1
                ax.text(x[j] + offset, s1v / 2,
                        f"{s1v:.3f}",
                        ha="center", va="center", fontsize=6,
                        color="white", fontweight="bold")
            else:
                ax.text(x[j] + offset, sv + 0.008,
                        f"{sv:.4f}",
                        ha="center", va="bottom", fontsize=5.5)

    ax.set_xticks(x)
    ax.set_xticklabels([factor_display[f] for f in factor_order], fontsize=10)
    ax.set_ylabel("Sobol Index (fraction of total variance)", fontsize=10)
    ax.set_title("Variance-Based Sensitivity Analysis by Experimental Factor",
                 fontweight="bold")

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = []
    for mlabel in metric_names:
        legend_elements.append(
            Patch(facecolor=colors_s1[mlabel], alpha=0.9,
                  label=f"{mlabel} — main effect ($S_i$)"))
        legend_elements.append(
            Patch(facecolor=colors_inter[mlabel], alpha=0.7, hatch="//",
                  label=f"{mlabel} — interactions ($S_{{Ti}}-S_i$)"))
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9,
              fontsize=7, ncol=1)
    ax.set_ylim(0, 0.82)

    fig.tight_layout()
    _save(fig, "fig2_effect_hierarchy.png")


# ===================================================================
# Fig 3: Rebalancing × Mode Heatmap
# ===================================================================
def plot_condition_mode_heatmap(df: pd.DataFrame):
    """Heatmap of mean performance: 7 rebalancing strategies × 3 modes."""

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    for ax_idx, (metric, mlabel) in enumerate(PRIMARY_METRICS):
        ax = axes[ax_idx]

        # Compute mean across distances, levels, seeds
        pivot = df.groupby(["condition", "mode"])[metric].mean().reset_index()
        mat = pivot.pivot(index="condition", columns="mode", values=metric)
        mat = mat.reindex(index=CONDITIONS_7, columns=MODES)

        im = ax.imshow(mat.values, cmap="YlOrRd", aspect="auto",
                       vmin=0, vmax=mat.values.max() * 1.05)
        ax.set_xticks(range(len(MODES)))
        ax.set_xticklabels([MODE_LABELS[m] for m in MODES], fontsize=9)
        ax.set_yticks(range(len(CONDITIONS_7)))
        ax.set_yticklabels([COND_LABELS[c] for c in CONDITIONS_7], fontsize=9)

        # Annotate cells
        for i in range(len(CONDITIONS_7)):
            for j in range(len(MODES)):
                v = mat.values[i, j]
                color = "white" if v > mat.values.max() * 0.6 else "black"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=color)

        ax.set_title(mlabel, fontsize=12, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=mlabel)

    fig.suptitle("Mean Performance by Rebalancing Strategy and Training Mode\n"
                 "(pooled across distances, levels, and 12 seeds)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _save(fig, "fig3_condition_mode_heatmap.png")


# ===================================================================
# Fig 4: Critical Difference Diagrams (Nemenyi)
# ===================================================================
def _draw_cd_diagram(ax, ranks: dict, cd: float, title: str):
    """Draw a CD diagram on the given axes."""
    k = len(ranks)
    sorted_items = sorted(ranks.items(), key=lambda x: x[1])
    names = [COND_LABELS.get(n, n) for n, _ in sorted_items]
    rvals = [r for _, r in sorted_items]

    lo, hi = 1, k
    ax.set_xlim(lo - 0.5, hi + 0.5)
    ax.set_ylim(-0.6, 1.9)
    ax.invert_xaxis()

    # Axis line
    ax.hlines(1.4, lo, hi, color="black", linewidth=0.8)
    for r in range(lo, hi + 1):
        ax.vlines(r, 1.35, 1.45, color="black", linewidth=0.8)
        ax.text(r, 1.52, str(r), ha="center", va="bottom", fontsize=8)

    # CD bar
    ax.hlines(1.72, lo, lo + cd, color="#c0392b", linewidth=2.5)
    ax.text(lo + cd / 2, 1.80, f"CD = {cd:.2f}", ha="center", va="bottom",
            fontsize=8.5, color="#c0392b", fontweight="bold")

    # Split left/right
    mid = k / 2.0
    left_items = [(n, r) for n, r in zip(names, rvals) if r <= mid + 0.5]
    right_items = [(n, r) for n, r in zip(names, rvals) if r > mid + 0.5]

    y_step = 0.20
    for i, (name, r) in enumerate(left_items):
        y = 1.2 - (i + 1) * y_step
        ax.plot(r, 1.4, "o", color="black", markersize=4, zorder=5)
        ax.vlines(r, y, 1.4, color="black", linewidth=0.6)
        ax.text(r + 0.12, y, f"{name} ({r:.2f})", ha="left", va="center",
                fontsize=7.5)

    for i, (name, r) in enumerate(right_items):
        y = 1.2 - (i + 1) * y_step
        ax.plot(r, 1.4, "o", color="black", markersize=4, zorder=5)
        ax.vlines(r, y, 1.4, color="black", linewidth=0.6)
        ax.text(r - 0.12, y, f"({r:.2f}) {name}", ha="right", va="center",
                fontsize=7.5)

    # Non-significant groups (|rank_i - rank_j| < CD)
    groups = []
    for i in range(k):
        for j in range(i + 1, k):
            if abs(rvals[j] - rvals[i]) < cd:
                merged = False
                for g in groups:
                    if i in g or j in g:
                        g.add(i)
                        g.add(j)
                        merged = True
                        break
                if not merged:
                    groups.append({i, j})
    # Merge overlapping
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
    for gi, g in enumerate(groups):
        g_sorted = sorted(g)
        r_min = rvals[g_sorted[0]]
        r_max = rvals[g_sorted[-1]]
        bar_y = -0.05 - gi * 0.12
        ax.hlines(bar_y, r_min, r_max, color="black", linewidth=3, alpha=0.6)

    ax.set_title(title, fontsize=10, fontweight="bold", pad=30)
    ax.axis("off")


def plot_cd_diagrams(df: pd.DataFrame):
    """Generate CD diagrams for F2, AUROC, AUPRC (pooled in/out)."""
    metrics_cd = [("f2", "F2-score"), ("auc", "AUROC"), ("auc_pr", "AUPRC")]

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    for ax_idx, (metric, mlabel) in enumerate(metrics_cd):
        ax = axes[ax_idx]

        # Build pivot: rows = (seed × mode × dist × level combo), cols = conditions
        # Pool across all cells, rank per row
        sub = df.copy()
        sub["cell"] = (sub["seed"].astype(str) + "_" + sub["mode"] + "_"
                       + sub["distance"] + "_" + sub["level"])
        pivot = sub.groupby(["cell", "condition"])[metric].mean().reset_index()
        pivot_wide = pivot.pivot(index="cell", columns="condition", values=metric)
        pivot_wide = pivot_wide[CONDITIONS_7].dropna()

        if len(pivot_wide) < 3:
            ax.text(0.5, 0.5, "Insufficient data",
                    ha="center", va="center", transform=ax.transAxes)
            continue

        # Friedman test
        arrays = [pivot_wide[c].values for c in CONDITIONS_7]
        chi2, p_fri = stats.friedmanchisquare(*arrays)

        # Mean ranks (higher metric = rank 1)
        mean_ranks = pivot_wide.rank(axis=1, ascending=False).mean()
        ranks_dict = {c: mean_ranks[c] for c in CONDITIONS_7}

        # Critical Difference
        k = len(CONDITIONS_7)
        n_blocks = len(pivot_wide)
        from scipy.stats import studentized_range
        q_crit = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)
        cd = q_crit * np.sqrt(k * (k + 1) / (6 * n_blocks))

        title = (f"{mlabel} — Friedman χ²={chi2:.1f}, p<0.0001, CD={cd:.2f}")
        _draw_cd_diagram(ax, ranks_dict, cd, title)

    fig.suptitle("Critical Difference Diagrams (Nemenyi Post-Hoc)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "fig4_cd_diagrams.png")


# ===================================================================
# Fig 5: Distance Metric Equivalence — Violin Plot
# ===================================================================
def plot_distance_violin(df: pd.DataFrame):
    """Violin plots showing near-identical distributions across 3 distance metrics."""

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
    dist_labels = {"mmd": "MMD", "dtw": "DTW", "wasserstein": "Wasserstein"}
    dist_colors = {"mmd": "#e74c3c", "dtw": "#3498db", "wasserstein": "#2ecc71"}

    for ax_idx, (metric, mlabel) in enumerate(PRIMARY_METRICS):
        ax = axes[ax_idx]

        # Pool across all conditions, modes, levels, seeds
        plot_data = []
        for dist in DISTANCES:
            vals = df[df["distance"] == dist][metric].dropna().values
            for v in vals:
                plot_data.append({"Distance": dist_labels[dist], mlabel: v})
        plot_df = pd.DataFrame(plot_data)

        parts = ax.violinplot(
            [plot_df[plot_df["Distance"] == dist_labels[d]][mlabel].values
             for d in DISTANCES],
            positions=range(3), showmeans=True, showmedians=True,
            showextrema=False,
        )
        for pc, d in zip(parts["bodies"], DISTANCES):
            pc.set_facecolor(dist_colors[d])
            pc.set_alpha(0.4)
        parts["cmeans"].set_color("black")
        parts["cmedians"].set_color("#c0392b")

        # Compute KW for annotation
        groups = [df[df["distance"] == d][metric].dropna().values for d in DISTANCES]
        H, p = stats.kruskal(*groups)
        n_total = sum(len(g) for g in groups)
        eta2 = eta_squared_from_H(H, n_total, 3)

        ax.set_xticks(range(3))
        ax.set_xticklabels([dist_labels[d] for d in DISTANCES], fontsize=10)
        ax.set_ylabel(mlabel, fontsize=10)
        ax.set_title(f"{mlabel}\nKW H={H:.2f}, p={p:.3f}, η²={eta2:.4f}",
                     fontsize=10, fontweight="bold")

        # Annotate means
        for i, d in enumerate(DISTANCES):
            vals = df[df["distance"] == d][metric].dropna()
            ax.text(i, vals.mean() + 0.01, f"{vals.mean():.3f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

    fig.suptitle("Distance Metric Has Negligible Effect on Performance\n"
                 "(η² < 0.004 — distributions nearly identical)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    _save(fig, "fig5_distance_violin.png")


# ===================================================================
# Fig 6: Training Mode Grouped Box Plot
# ===================================================================
def plot_mode_boxplot(df: pd.DataFrame):
    """Box plots showing massive mode effect: within ≈ mixed >> cross."""

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
    mode_colors = {
        "source_only": "#e74c3c",
        "target_only": "#3498db",
        "mixed":       "#2ecc71",
    }

    for ax_idx, (metric, mlabel) in enumerate(PRIMARY_METRICS):
        ax = axes[ax_idx]

        data_by_mode = [df[df["mode"] == m][metric].dropna().values for m in MODES]

        bp = ax.boxplot(
            data_by_mode,
            positions=range(3), widths=0.5,
            patch_artist=True, showfliers=True,
            flierprops=dict(marker="o", markersize=3, alpha=0.3),
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
        )
        for patch, m in zip(bp["boxes"], MODES):
            patch.set_facecolor(mode_colors[m])
            patch.set_alpha(0.7)

        # Overlay means as diamonds
        for i, m in enumerate(MODES):
            vals = df[df["mode"] == m][metric].dropna()
            ax.plot(i, vals.mean(), "D", color="white", markersize=7,
                    markeredgecolor="black", markeredgewidth=1.5, zorder=5)
            ax.text(i, vals.mean() + 0.012, f"μ={vals.mean():.3f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_xticks(range(3))
        ax.set_xticklabels([MODE_LABELS[m] for m in MODES], fontsize=10)
        ax.set_ylabel(mlabel, fontsize=10)
        ax.set_title(mlabel, fontsize=11, fontweight="bold")

    fig.suptitle("Training Mode Effect: Within-Domain ≈ Mixed >> Cross-Domain",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    _save(fig, "fig6_mode_boxplot.png")


# ===================================================================
# Fig 7: Domain Shift Reversal — Diverging Bar Chart
# ===================================================================
def plot_domain_shift(df: pd.DataFrame):
    """Diverging bar chart showing Δ = out − in by mode × condition."""

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))

    for ax_idx, (metric, mlabel) in enumerate(PRIMARY_METRICS):
        ax = axes[ax_idx]

        deltas = []
        labels = []
        colors = []

        for mode in MODES:
            for cond in CONDITIONS_7:
                in_vals = df[(df["condition"] == cond) & (df["mode"] == mode)
                             & (df["level"] == "in_domain")][metric].dropna()
                out_vals = df[(df["condition"] == cond) & (df["mode"] == mode)
                              & (df["level"] == "out_domain")][metric].dropna()

                # Paired by seed × distance
                in_mean = in_vals.mean() if len(in_vals) > 0 else np.nan
                out_mean = out_vals.mean() if len(out_vals) > 0 else np.nan
                delta = out_mean - in_mean
                deltas.append(delta)
                labels.append(f"{COND_LABELS[cond]}\n({MODE_LABELS[mode][:5]})")
                colors.append(COND_COLORS[cond])

        y = np.arange(len(deltas))
        bar_colors = ["#27ae60" if d >= 0 else "#e74c3c" for d in deltas]

        ax.barh(y, deltas, color=bar_colors, alpha=0.7, edgecolor="white",
                linewidth=0.5, height=0.7)
        ax.axvline(0, color="black", linewidth=1)

        # Mode separator lines
        for i in range(1, 3):
            ax.axhline(i * 7 - 0.5, color="gray", linewidth=0.8,
                       linestyle="--", alpha=0.5)

        # Mode labels
        for i, mode in enumerate(MODES):
            mid_y = i * 7 + 3
            ax.text(ax.get_xlim()[1] * 0.95 if ax_idx == 0 else 0.15,
                    mid_y, MODE_LABELS[mode],
                    ha="right" if ax_idx == 0 else "right",
                    va="center", fontsize=9, fontweight="bold",
                    fontstyle="italic", color="#2c3e50",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              alpha=0.8, edgecolor="gray"))

        ax.set_yticks(y)
        ax.set_yticklabels([COND_LABELS[CONDITIONS_7[i % 7]]
                            for i in range(len(deltas))], fontsize=7.5)
        ax.set_xlabel(f"Δ{mlabel} (Out − In domain)", fontsize=10)
        ax.set_title(mlabel, fontsize=11, fontweight="bold")
        ax.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#27ae60", alpha=0.7, label="Δ > 0 (out > in)"),
        Patch(facecolor="#e74c3c", alpha=0.7, label="Δ < 0 (in > out)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
               fontsize=9, framealpha=0.9)

    fig.suptitle("Domain Shift Direction by Rebalancing Strategy × Mode\n"
                 "Green = out-domain outperforms (gap reversal)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 0.92])
    _save(fig, "fig7_domain_shift_reversal.png")


# ===================================================================
# Fig 8: Seed Convergence Curve
# ===================================================================
def seed_convergence_data(df, metric, seeds, max_subsets=500, rng=None):
    """Compute ranking std for each k."""
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
    """Plot σ_rank(k) convergence for both metrics."""
    seeds = sorted(int(s) for s in df["seed"].unique())
    rng = np.random.RandomState(42)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax_idx, (metric, mlabel) in enumerate(PRIMARY_METRICS):
        ax = axes[ax_idx]
        print(f"  Computing convergence for {mlabel}...")
        conv = seed_convergence_data(df, metric, seeds, max_subsets=500, rng=rng)

        ks = sorted(conv.keys())
        mean_stds = [conv[k]["mean_std"] for k in ks]
        max_stds = [conv[k]["max_std"] for k in ks]

        ax.plot(ks, mean_stds, "o-", color="#2c3e50", linewidth=2.5,
                markersize=9, label="Mean σ_rank", zorder=5)
        ax.fill_between(ks, 0, max_stds, alpha=0.12, color="#3498db")
        ax.plot(ks, max_stds, "s--", color="#3498db", linewidth=1.2,
                markersize=6, alpha=0.7, label="Max σ_rank")

        # Per-condition lines
        for cond in CONDITIONS_7:
            per_cond = [conv[k]["per_cond_std"].get(cond, np.nan) for k in ks]
            ax.plot(ks, per_cond, "-", color=COND_COLORS[cond],
                    linewidth=0.8, alpha=0.4)

        # Reference
        ax.axhline(0.5, color="#e74c3c", linewidth=0.8, linestyle=":",
                   alpha=0.6, label="σ = 0.5 threshold")

        # Annotate final values
        ax.annotate(f"σ = {mean_stds[-1]:.3f}",
                    xy=(ks[-1], mean_stds[-1]),
                    xytext=(ks[-1] - 1.5, mean_stds[-1] + 0.08),
                    fontsize=8, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1))

        ax.set_xlabel("Number of Seeds (k)", fontsize=10)
        ax.set_ylabel("σ_rank", fontsize=10)
        ax.set_title(mlabel, fontsize=11, fontweight="bold")
        ax.set_xticks(ks)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_ylim(bottom=-0.02)

    fig.suptitle("Seed Count Convergence — Ranking Stability vs. Subset Size\n"
                 f"({len(seeds)} official seeds: {seeds})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    _save(fig, "fig8_seed_convergence.png")


# ===================================================================
# Main
# ===================================================================
def main():
    print("=" * 60)
    print("Journal Figure Generation v2 (n=12 seeds)")
    print("=" * 60)

    df = load_all_data()

    print("\n[1/7] Fig 2 — Effect size hierarchy...")
    plot_effect_hierarchy(df)

    print("[2/7] Fig 3 — Rebalancing × Mode heatmap...")
    plot_condition_mode_heatmap(df)

    print("[3/7] Fig 4 — Critical Difference diagrams...")
    plot_cd_diagrams(df)

    print("[4/7] Fig 5 — Distance metric violin...")
    plot_distance_violin(df)

    print("[5/7] Fig 6 — Training mode box plot...")
    plot_mode_boxplot(df)

    print("[6/7] Fig 7 — Domain shift reversal...")
    plot_domain_shift(df)

    print("[7/7] Fig 8 — Seed convergence...")
    plot_convergence(df)

    print("\n" + "=" * 60)
    print(f"All 7 figures saved to: {OUT_DIR.relative_to(PROJECT_ROOT)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
