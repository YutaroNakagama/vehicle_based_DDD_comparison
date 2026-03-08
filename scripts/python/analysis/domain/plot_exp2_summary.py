#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_exp2_summary.py
====================
Generate summary comparison plots for Experiment 2 (Domain Shift)
across all imbalance handling conditions.

Plots generated:
  1. overall_f2_comparison.png       — F2 score overview (all conditions, seed-averaged)
  2. mode_comparison_f2.png          — F2 by training mode (cross/single/mixed)
  3. mode_comparison_recall.png      — Recall by training mode
  4. mode_comparison_f1.png          — F1 by training mode
  5. domain_gap_f2.png               — In-domain vs Out-domain F2 gap
  6. distance_comparison_f2.png      — F2 by distance metric (mmd/dtw/wasserstein)
  7. heatmap_f2_mode_condition.png   — Heatmap: F2 by (condition × mode)
  8. heatmap_f2_mode_domain.png      — Heatmap: F2 by (condition × mode × domain)
  9. radar_top_conditions.png        — Radar chart: multi-metric comparison of top conditions
 10. cross_domain_degradation.png    — Cross-domain degradation (Δ from single-domain)

Output:
  results/analysis/exp2_domain_shift/figures/png/split2/summary/

Usage:
    python scripts/python/analysis/domain/plot_exp2_summary.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
CSV_BASE = (
    PROJECT_ROOT
    / "results/analysis/exp2_domain_shift/figures/csv/split2"
)
OUT_DIR = (
    PROJECT_ROOT
    / "results/analysis/exp2_domain_shift/figures/png/split2/summary"
)

METRICS = ["f2", "recall", "precision", "f1", "auc", "auc_pr"]

# Display order & labels
COND_ORDER = [
    "baseline",
    "undersample_rus_r01", "undersample_rus_r05",
    "smote_plain_r01", "smote_plain_r05",
    "sw_smote_r01", "sw_smote_r05",
]
COND_LABELS = {
    "baseline":             "Baseline\n(class_weight)",
    "balanced_rf":          "BalancedRF",
    "smote_plain_r01":      "SMOTE\nr=0.1",
    "smote_plain_r05":      "SMOTE\nr=0.5",
    "sw_smote_r01":         "SW-SMOTE\nr=0.1",
    "sw_smote_r05":         "SW-SMOTE\nr=0.5",
    "undersample_rus_r01":  "RUS\nr=0.1",
    "undersample_rus_r05":  "RUS\nr=0.5",
}
COND_SHORT = {
    "baseline":             "Baseline",
    "balanced_rf":          "BRF",
    "smote_plain_r01":      "SMOTE r01",
    "smote_plain_r05":      "SMOTE r05",
    "sw_smote_r01":         "SW-SMOTE r01",
    "sw_smote_r05":         "SW-SMOTE r05",
    "undersample_rus_r01":  "RUS r01",
    "undersample_rus_r05":  "RUS r05",
}
MODE_LABELS = {
    "source_only": "Cross-domain",
    "target_only": "Single-domain",
    "mixed":       "Mixed",
}
MODE_ORDER = ["source_only", "target_only", "mixed"]
MODE_COLORS = {
    "source_only": "#6699cc",   # dark blue — matches per-seed bar plots
    "target_only": "#ff9966",   # orange
    "mixed":       "#cc99ff",   # purple
}
COND_COLORS = {
    "baseline":             "#95a5a6",
    "balanced_rf":          "#9b59b6",
    "smote_plain_r01":      "#3498db",
    "smote_plain_r05":      "#2980b9",
    "sw_smote_r01":         "#e67e22",
    "sw_smote_r05":         "#d35400",
    "undersample_rus_r01":  "#27ae60",
    "undersample_rus_r05":  "#1e8449",
}

DIST_ORDER = ["mmd", "dtw", "wasserstein"]


# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------
def load_all_data() -> pd.DataFrame:
    """Load and merge all split2 CSVs."""
    specs = [
        ("baseline", "baseline/baseline_domain_split2_metrics_v2.csv"),
        ("smote_plain", "smote_plain/smote_plain_split2_metrics_v2.csv"),
        ("sw_smote", "sw_smote/sw_smote_split2_metrics_v2.csv"),
        ("undersample_rus", "undersample_rus/undersample_rus_split2_metrics_v2.csv"),
        ("balanced_rf", "balanced_rf/balancedrf_split2_metrics_v2.csv"),
    ]
    frames = []
    for cond, rel in specs:
        path = CSV_BASE / rel
        if not path.exists():
            print(f"[WARN] Missing: {path}")
            continue
        df = pd.read_csv(path)
        df["condition"] = cond
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)

    # Create unified condition label including ratio
    def _label(row):
        c = row["condition"]
        r = row.get("ratio", np.nan)
        if pd.isna(r):
            return c
        return f"{c}_r{str(r).replace('.', '')}"

    all_df["cond_label"] = all_df.apply(_label, axis=1)
    return all_df


def _save(fig, name: str) -> Path:
    out = OUT_DIR / name
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out.relative_to(PROJECT_ROOT)}")
    return out


# -------------------------------------------------------------------
# Plot 1: Overall F2 comparison (bar + individual seed points)
# -------------------------------------------------------------------
def plot_overall_metric(df: pd.DataFrame, metric: str = "f2"):
    """Bar chart of mean metric across all conditions with seed scatter."""
    label = metric.upper() if metric != "auc" else "AUROC"
    agg = df.groupby("cond_label")[metric].agg(["mean", "std"]).reindex(COND_ORDER)
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(COND_ORDER))
    colors = [COND_COLORS.get(c, "#999") for c in COND_ORDER]
    bars = ax.bar(x, agg["mean"], yerr=agg["std"], capsize=4,
                  color=colors, edgecolor="white", linewidth=0.8, alpha=0.85,
                  error_kw=dict(lw=1.2, capthick=1.2))

    # Overlay individual points (per distance × domain × seed × mode)
    for i, c in enumerate(COND_ORDER):
        vals = df[df["cond_label"] == c][metric].values
        jitter = np.random.default_rng(42).uniform(-0.2, 0.2, len(vals))
        ax.scatter(i + jitter, vals, color="black", s=8, alpha=0.18, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([COND_LABELS[c] for c in COND_ORDER], fontsize=9)
    ax.set_ylabel(f"{label} Score", fontsize=12)
    ax.set_title(f"Experiment 2: Overall {label} by Imbalance Method\n"
                 "(mean ± std across modes, distances, domains, seeds)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add value labels
    for i, c in enumerate(COND_ORDER):
        ax.text(i, agg.loc[c, "mean"] + agg.loc[c, "std"] + 0.008,
                f"{agg.loc[c, 'mean']:.3f}", ha="center", va="bottom", fontsize=8)

    _save(fig, f"overall_{metric}_comparison.png")


def plot_overall_f2(df: pd.DataFrame):
    """Bar chart of mean F2 across all conditions with seed scatter."""
    plot_overall_metric(df, "f2")


# -------------------------------------------------------------------
# Plot 2-4: Mode comparison (grouped bar for F2 / Recall / F1)
# -------------------------------------------------------------------
def plot_mode_comparison(df: pd.DataFrame, metric: str, title_suffix: str = ""):
    """Grouped bar: conditions on x-axis, bars colored by mode."""
    agg = (df.groupby(["cond_label", "mode"])[metric]
             .mean().unstack("mode")
             .reindex(index=COND_ORDER, columns=MODE_ORDER))

    fig, ax = plt.subplots(figsize=(13, 5.5))
    n_cond = len(COND_ORDER)
    n_mode = len(MODE_ORDER)
    bar_w = 0.25
    x = np.arange(n_cond)

    for j, mode in enumerate(MODE_ORDER):
        vals = agg[mode].values
        offset = (j - (n_mode - 1) / 2) * bar_w
        bars = ax.bar(x + offset, vals, bar_w,
                      label=MODE_LABELS[mode], color=MODE_COLORS[mode],
                      edgecolor="white", linewidth=0.6, alpha=0.85)
        for k, v in enumerate(vals):
            ax.text(x[k] + offset, v + 0.005, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7, rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels([COND_LABELS[c] for c in COND_ORDER], fontsize=9)
    ax.set_ylabel(f"{metric.upper()} Score", fontsize=12)
    ax.set_title(f"Experiment 2: {metric.upper()} by Training Mode{title_suffix}\n"
                 f"(mean across distances, domains, seeds)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.set_ylim(0, agg.max().max() * 1.3)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _save(fig, f"mode_comparison_{metric}.png")


# -------------------------------------------------------------------
# Plot 5: In-domain vs Out-domain gap
# -------------------------------------------------------------------
def plot_domain_gap(df: pd.DataFrame, metric: str = "f2",
                    distance_filter: str | None = None,
                    ylim: tuple | None = None):
    """Grouped bar showing metric for in_domain vs out_domain per condition, by mode.

    Parameters
    ----------
    distance_filter : str or None
        If provided, restrict data to this distance metric (e.g. "mmd", "dtw", "wasserstein").
    ylim : tuple or None
        If provided, (ymin, ymax) to fix the y-axis range across plots.
    """
    label = metric.upper() if metric != "auc" else "AUROC"
    plot_df = df.copy()
    if distance_filter is not None:
        plot_df = plot_df[plot_df["distance"] == distance_filter]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)

    for ax, mode in zip(axes, MODE_ORDER):
        sub = (plot_df[plot_df["mode"] == mode]
               .groupby(["cond_label", "level"])[metric]
               .mean().unstack("level")
               .reindex(COND_ORDER))

        x = np.arange(len(COND_ORDER))
        bar_w = 0.35
        ax.bar(x - bar_w / 2, sub["in_domain"], bar_w,
               label="In-domain", color="#3498db", alpha=0.85, edgecolor="white")
        ax.bar(x + bar_w / 2, sub["out_domain"], bar_w,
               label="Out-domain", color="#e74c3c", alpha=0.85, edgecolor="white")

        # Add gap annotation
        for i, c in enumerate(COND_ORDER):
            gap = sub.loc[c, "in_domain"] - sub.loc[c, "out_domain"]
            ymax = max(sub.loc[c, "in_domain"], sub.loc[c, "out_domain"])
            color = "#27ae60" if gap >= 0 else "#c0392b"
            ax.text(i, ymax + 0.008, f"Δ={gap:+.3f}",
                    ha="center", va="bottom", fontsize=7, color=color)

        ax.set_xticks(x)
        ax.set_xticklabels([COND_SHORT[c] for c in COND_ORDER],
                           fontsize=8, rotation=45, ha="right")
        ax.set_title(MODE_LABELS[mode], fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if ylim is not None:
        axes[0].set_ylim(ylim)
    axes[0].set_ylabel(f"{label} Score", fontsize=12)
    axes[0].legend(fontsize=9)
    dist_note = f" [{distance_filter.upper()}]" if distance_filter else ""
    avg_note = "(mean across distances, seeds)" if not distance_filter else "(mean across seeds)"
    fig.suptitle(f"Experiment 2: In-domain vs Out-domain {label} by Training Mode{dist_note}\n"
                 f"{avg_note}",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    suffix = f"_{distance_filter}" if distance_filter else ""
    _save(fig, f"domain_gap_{metric}{suffix}.png")


# -------------------------------------------------------------------
# Plot 6: Distance comparison
# -------------------------------------------------------------------
def plot_distance_comparison(df: pd.DataFrame, metric: str = "f2"):
    """Heatmap-style grouped bar: metric by distance metric for each condition."""
    label = metric.upper() if metric != "auc" else "AUROC"
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)

    for ax, mode in zip(axes, MODE_ORDER):
        sub = (df[df["mode"] == mode]
               .groupby(["cond_label", "distance"])[metric]
               .mean().unstack("distance")
               .reindex(index=COND_ORDER, columns=DIST_ORDER))

        x = np.arange(len(COND_ORDER))
        bar_w = 0.25
        dist_colors = {"mmd": "#e74c3c", "dtw": "#3498db", "wasserstein": "#2ecc71"}

        for j, dist in enumerate(DIST_ORDER):
            offset = (j - 1) * bar_w
            vals = sub[dist].values
            ax.bar(x + offset, vals, bar_w,
                   label=dist.upper(), color=dist_colors[dist],
                   edgecolor="white", linewidth=0.5, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([COND_SHORT[c] for c in COND_ORDER],
                           fontsize=8, rotation=45, ha="right")
        ax.set_title(MODE_LABELS[mode], fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(f"{label} Score", fontsize=12)
    axes[0].legend(fontsize=9)
    fig.suptitle(f"Experiment 2: {label} by Distance Metric and Training Mode\n"
                 "(mean across domains, seeds)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, f"distance_comparison_{metric}.png")


# -------------------------------------------------------------------
# Plot 7: Heatmap — F2 by (condition × mode)
# -------------------------------------------------------------------
def plot_heatmap_mode_condition(df: pd.DataFrame, metric: str = "f2"):
    """Annotated heatmap of metric (condition rows × mode columns)."""
    label = metric.upper() if metric != "auc" else "AUROC"
    pivot = (df.groupby(["cond_label", "mode"])[metric]
               .mean().unstack("mode")
               .reindex(index=COND_ORDER, columns=MODE_ORDER))

    vmin_map = {"f2": 0.10, "auc": 0.50}
    vmax_map = {"f2": 0.42, "auc": 0.90}
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd",
                linewidths=0.8, linecolor="white",
                xticklabels=[MODE_LABELS[m] for m in MODE_ORDER],
                yticklabels=[COND_SHORT[c] for c in COND_ORDER],
                ax=ax, vmin=vmin_map.get(metric, 0.10),
                vmax=vmax_map.get(metric, 0.90),
                cbar_kws={"label": f"{label} Score"})
    ax.set_title(f"Experiment 2: {label} Heatmap\n(Condition × Training Mode)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("")
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    _save(fig, f"heatmap_{metric}_mode_condition.png")


# -------------------------------------------------------------------
# Plot 8: Heatmap — F2 by (condition × mode × domain)
# -------------------------------------------------------------------
def plot_heatmap_mode_domain(df: pd.DataFrame, metric: str = "f2"):
    """Annotated heatmap with mode×domain columns."""
    label = metric.upper() if metric != "auc" else "AUROC"
    pivot = (df.groupby(["cond_label", "mode", "level"])[metric]
               .mean().unstack(["mode", "level"]))

    col_order = [(m, l) for m in MODE_ORDER for l in ["in_domain", "out_domain"]]
    pivot = pivot.reindex(index=COND_ORDER, columns=pd.MultiIndex.from_tuples(col_order))

    # Flatten column labels
    col_labels = [f"{MODE_LABELS[m]}\n{'In' if l=='in_domain' else 'Out'}"
                  for m, l in col_order]

    vmin_map = {"f2": 0.10, "auc": 0.50}
    vmax_map = {"f2": 0.42, "auc": 0.90}
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivot.values, annot=True, fmt=".3f", cmap="YlOrRd",
                linewidths=0.8, linecolor="white",
                xticklabels=col_labels,
                yticklabels=[COND_SHORT[c] for c in COND_ORDER],
                ax=ax, vmin=vmin_map.get(metric, 0.10),
                vmax=vmax_map.get(metric, 0.90),
                cbar_kws={"label": f"{label} Score"})
    ax.set_title(f"Experiment 2: {label} Heatmap\n(Condition × Mode × Domain)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("")
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    _save(fig, f"heatmap_{metric}_mode_domain.png")


# -------------------------------------------------------------------
# Plot 9: Radar chart — multi-metric for top conditions
# -------------------------------------------------------------------
def plot_radar(df: pd.DataFrame):
    """Radar chart comparing top conditions across multiple metrics."""
    radar_metrics = ["f2", "recall", "precision", "f1", "auc", "auc_pr"]
    # Select top-4 + baseline for comparison
    f2_rank = df.groupby("cond_label")["f2"].mean().sort_values(ascending=False)
    top_conds = list(f2_rank.head(4).index)
    if "baseline" not in top_conds:
        top_conds.append("baseline")

    agg = df[df["cond_label"].isin(top_conds)].groupby("cond_label")[radar_metrics].mean()

    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = plt.cm.Set2(np.linspace(0, 1, len(top_conds)))

    for i, c in enumerate(top_conds):
        vals = agg.loc[c].values.tolist()
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2, label=COND_SHORT.get(c, c),
                color=colors[i], markersize=5)
        ax.fill(angles, vals, alpha=0.08, color=colors[i])

    ax.set_thetagrids(np.degrees(angles[:-1]),
                      [m.upper() for m in radar_metrics], fontsize=10)
    ax.set_rlabel_position(30)
    ax.set_title("Experiment 2: Multi-Metric Radar\n(Top conditions vs Baseline)",
                 fontsize=13, fontweight="bold", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    fig.tight_layout()
    _save(fig, "radar_top_conditions.png")


# -------------------------------------------------------------------
# Plot 10: Cross-domain degradation (delta from single-domain)
# -------------------------------------------------------------------
def plot_cross_domain_degradation(df: pd.DataFrame, metric: str = "f2"):
    """Waterfall-style chart showing metric degradation from single to cross-domain."""
    label = metric.upper() if metric != "auc" else "AUROC"
    mode_means = (df.groupby(["cond_label", "mode"])[metric]
                    .mean().unstack("mode")
                    .reindex(index=COND_ORDER, columns=MODE_ORDER))

    delta_cross = mode_means["source_only"] - mode_means["target_only"]
    delta_mixed = mode_means["mixed"] - mode_means["target_only"]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(COND_ORDER))
    bar_w = 0.35

    bars1 = ax.bar(x - bar_w / 2, delta_cross, bar_w,
                   label="Δ Cross−Single", color="#6699cc", alpha=0.85,
                   edgecolor="white")
    bars2 = ax.bar(x + bar_w / 2, delta_mixed, bar_w,
                   label="Δ Mixed−Single", color="#cc99ff", alpha=0.85,
                   edgecolor="white")

    # Value labels
    for bar_set in [bars1, bars2]:
        for bar in bar_set:
            h = bar.get_height()
            y_pos = h + 0.003 if h >= 0 else h - 0.012
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f"{h:+.3f}", ha="center", va="bottom" if h >= 0 else "top",
                    fontsize=7)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([COND_LABELS[c] for c in COND_ORDER], fontsize=9)
    ax.set_ylabel(f"{label} Difference", fontsize=12)
    ax.set_title(f"Experiment 2: {label} Gap vs Single-domain\n"
                 "(Negative = degradation from within-domain baseline)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _save(fig, f"cross_domain_degradation_{metric}.png")


# -------------------------------------------------------------------
# NEW: Distance × Domain × Mode summary plot (F2/AUC, In/Out, Cross/Single/Mixed)
# -------------------------------------------------------------------
def plot_distance_domain_gap_summary(df: pd.DataFrame):
    """
    For each distance (mmd/dtw/wasserstein), plot:
      - Bar: mean F2 (and AUC) for In/Out × Cross/Single/Mixed
      - Line: Out−In gap for each distance, mode
    2-row subplot: F2 (top), AUC (bottom)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    dist_order = ["mmd", "dtw", "wasserstein"]
    mode_order = ["source_only", "target_only", "mixed"]
    mode_labels = [MODE_LABELS[m] for m in mode_order]
    # Colors: different color for each In/Out×Mode combination
    bar_colors = {
        ("in_domain", "source_only"): "#6baed6",
        ("out_domain", "source_only"): "#2171b5",
        ("in_domain", "target_only"): "#74c476",
        ("out_domain", "target_only"): "#238b45",
        ("in_domain", "mixed"): "#fd8d3c",
        ("out_domain", "mixed"): "#e6550d",
    }
    gap_colors = {"source_only": "#6699cc", "target_only": "#ff9966", "mixed": "#cc99ff"}
    metrics = ["f2", "auc"]
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    width = 0.08  # narrow
    group_gap = 0.10  # gap between mode groups
    x = np.arange(len(dist_order))
    for row, metric in enumerate(metrics):
        ax = axes[row]
        bar_handles = []
        max_bar = 0
        for i, mode in enumerate(mode_order):
            vals_in = []
            vals_out = []
            for dist in dist_order:
                sub = df[(df["distance"]==dist)&(df["mode"]==mode)]
                in_v = sub[sub["level"]=="in_domain"][metric].mean()
                out_v = sub[sub["level"]=="out_domain"][metric].mean()
                vals_in.append(in_v)
                vals_out.append(out_v)
            # Group by mode with wider spacing
            base = x + (i-1)*group_gap + (i-1)*2*width
            h1 = ax.bar(base, vals_in, width, label=f"In-{mode_labels[i]}", color=bar_colors[("in_domain", mode)], edgecolor="white")
            h2 = ax.bar(base + width, vals_out, width, label=f"Out-{mode_labels[i]}", color=bar_colors[("out_domain", mode)], edgecolor="white")
            if row==0:
                bar_handles.append(h1[0])
                bar_handles.append(h2[0])
            max_bar = max(max_bar, max(vals_in + vals_out))
        # Reduce top margin of Y-axis
        ax.set_ylim(0, max_bar * 1.12)
        ax.set_ylabel(f"{metric.upper() if metric!='auc' else 'AUROC'}", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([d.upper() for d in dist_order], fontsize=12)
        ax.set_title(f"{metric.upper() if metric!='auc' else 'AUROC'}: Distance × Domain × Mode", fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if row==0:
            from matplotlib.lines import Line2D
            legend_elems = [bar_handles[0], bar_handles[1], bar_handles[2], bar_handles[3], bar_handles[4], bar_handles[5]]
            legend_labels = ["In-Cross", "Out-Cross", "In-Single", "Out-Single", "In-Mixed", "Out-Mixed"]
            ax.legend(legend_elems, legend_labels, fontsize=9, ncol=1, loc="upper left")
    fig.suptitle("Experiment 2: Distance Metric × Domain × Mode Summary\n(In/Out mean only; F2 & AUROC)", fontsize=15, fontweight="bold", y=0.97)
    fig.tight_layout(rect=[0,0,1,0.96])
    _save(fig, "distance_domain_gap_summary.png")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Experiment 2 Summary Plots")
    print("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_all_data()
    print(f"Loaded {len(df)} records, {df['cond_label'].nunique()} conditions\n")

    print("Generating plots (F2-based):")
    plot_overall_f2(df)                           # 1
    plot_mode_comparison(df, "f2")                 # 2
    # plot_mode_comparison(df, "recall")             # 3  — removed
    plot_mode_comparison(df, "f1")                 # 4
    plot_domain_gap(df, "f2")                      # 5
    plot_distance_comparison(df, "f2")              # 6
    # plot_heatmap_mode_condition(df, "f2")           # 7  (removed — redundant)
    # plot_heatmap_mode_domain(df, "f2")              # 8  (removed — redundant)
    plot_radar(df)                                 # 9
    # plot_cross_domain_degradation(df, "f2")         # 10 (removed — redundant)

    print("\nGenerating plots (AUROC-based):")
    plot_overall_metric(df, "auc")                 # 11
    plot_mode_comparison(df, "auc")                 # 12
    plot_domain_gap(df, "auc")                      # 13
    plot_distance_comparison(df, "auc")              # 14
    # plot_heatmap_mode_condition(df, "auc")           # 15 (removed — redundant)
    # plot_heatmap_mode_domain(df, "auc")              # 16 (removed — redundant)
    # plot_cross_domain_degradation(df, "auc")         # 17 (removed — redundant)

    print("\nGenerating plots (Domain gap per distance):")
    # Compute shared y-axis ranges across all distances + overall
    def _domain_gap_ylim(df, metric):
        vals = (df.groupby(["cond_label", "mode", "level", "distance"])[metric]
                  .mean().values)
        ymin = 0
        ymax = float(np.nanmax(vals)) * 1.20  # 20% headroom for Δ labels
        return (ymin, ymax)
    ylim_f2 = _domain_gap_ylim(df, "f2")
    ylim_auc = _domain_gap_ylim(df, "auc")
    # Re-generate overall with same scale
    plot_domain_gap(df, "f2", ylim=ylim_f2)
    plot_domain_gap(df, "auc", ylim=ylim_auc)
    for dist in DIST_ORDER:
        plot_domain_gap(df, "f2", distance_filter=dist, ylim=ylim_f2)
        plot_domain_gap(df, "auc", distance_filter=dist, ylim=ylim_auc)

    print("\nGenerating summary plot: Distance × Domain × Mode (F2/AUC)")
    plot_distance_domain_gap_summary(df)

    n_plots = len(list(OUT_DIR.glob("*.png")))
    print(f"\nDONE — {n_plots} plots saved to {OUT_DIR.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
