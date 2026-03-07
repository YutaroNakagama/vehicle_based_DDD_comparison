#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_baseline_seed_summary.py
=============================
Generate summary plots aggregating 10 seeds for the baseline_domain condition.

Reads the baseline CSV and produces the following plots in
  results/analysis/exp2_domain_shift/figures/png/split2/baseline/

Output plots:
  1. baseline_seed_summary_bar.png
       Grouped bar chart (mean ± std across 10 seeds)
       Rows: 3 distances (MMD, DTW, Wasserstein)
       Cols: key metrics (F2, AUROC, AUPRC, Recall, Precision, F1)
       Groups: mode × domain

  2. baseline_seed_summary_boxplot.png
       Box plots showing seed-to-seed variability
       Rows: 3 distances
       Cols: F2, AUROC, AUPRC
       Groups: mode × domain

  3. baseline_seed_summary_heatmap.png
       Heatmaps of mean F2 / AUROC for each domain
       Shows mode × distance matrix with color-coded values

  4. baseline_seed_summary_table.png
       Table figure showing exact mean ± std for all combinations

Usage:
    python scripts/python/analysis/domain/plot_baseline_seed_summary.py
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[4]
CSV_PATH = (
    PROJECT_ROOT
    / "results/analysis/exp2_domain_shift/figures/csv/split2/baseline"
    / "baseline_domain_split2_metrics_v2.csv"
)
OUT_DIR = (
    PROJECT_ROOT
    / "results/analysis/exp2_domain_shift/figures/png/split2/baseline"
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
METRICS = ["f2", "auc", "auc_pr", "recall", "precision", "f1"]
METRIC_LABELS = {
    "f2": "F2",
    "auc": "AUROC",
    "auc_pr": "AUPRC",
    "recall": "Recall",
    "precision": "Precision",
    "f1": "F1",
    "accuracy": "Accuracy",
}
MODE_ORDER = ["source_only", "target_only", "mixed"]
MODE_LABELS = {
    "source_only": "Cross-domain",
    "target_only": "Within-domain",
    "mixed": "Multi-domain",
}
DIST_ORDER = ["mmd", "dtw", "wasserstein"]
DIST_LABELS = {"mmd": "MMD", "dtw": "DTW", "wasserstein": "Wasserstein"}
DOMAIN_ORDER = ["in_domain", "out_domain"]
DOMAIN_LABELS = {"in_domain": "In-domain", "out_domain": "Out-domain"}

MODE_COLORS = {
    "source_only": "#6699cc",
    "target_only": "#ff9966",
    "mixed": "#cc99ff",
}


# ===== Plot 1: Grouped bar chart (mean ± std) =====
def plot_summary_bar(df: pd.DataFrame, out_path: Path) -> None:
    """Grouped bar chart: mean ± std across seeds.

    Layout:
      Rows = 3 distances (MMD, DTW, Wasserstein)
      Cols = 6 key metrics
      Within each panel: 6 bars = 3 modes × 2 domains
    """
    metrics = ["f2", "auc", "auc_pr", "recall", "precision", "f1"]
    n_rows = len(DIST_ORDER)
    n_cols = len(metrics)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows))

    for i, dist in enumerate(DIST_ORDER):
        sub = df[df["distance"] == dist]
        for j, metric in enumerate(metrics):
            ax = axes[i, j]

            # Build grouped data: mode × domain
            group_labels = []
            means = []
            stds = []
            colors = []
            hatches = []

            for mode in MODE_ORDER:
                for domain in DOMAIN_ORDER:
                    cell = sub[(sub["mode"] == mode) & (sub["level"] == domain)]
                    vals = cell[metric].dropna()
                    group_labels.append(
                        f"{MODE_LABELS[mode]}\n{DOMAIN_LABELS[domain]}"
                    )
                    means.append(vals.mean() if len(vals) > 0 else 0)
                    stds.append(vals.std() if len(vals) > 1 else 0)
                    colors.append(MODE_COLORS[mode])
                    hatches.append("" if domain == "in_domain" else "//")

            x = np.arange(len(group_labels))
            bars = ax.bar(
                x,
                means,
                yerr=stds,
                capsize=3,
                color=colors,
                edgecolor="black",
                linewidth=0.5,
                error_kw={"linewidth": 1},
            )
            # Apply hatching for out_domain
            for bar, hatch in zip(bars, hatches):
                bar.set_hatch(hatch)

            # Value labels
            for xi, (m, s) in enumerate(zip(means, stds)):
                if m > 0:
                    ax.text(
                        xi, m + s + 0.01, f"{m:.3f}", ha="center", va="bottom",
                        fontsize=6, rotation=45,
                    )

            ax.set_xticks(x)
            ax.set_xticklabels(group_labels, fontsize=7, rotation=0)
            ax.set_ylim(0, min(1.0, max(means) * 1.5 + 0.1) if means else 1.0)
            ax.set_title(
                f"{DIST_LABELS[dist]} — {METRIC_LABELS.get(metric, metric)}",
                fontsize=10, fontweight="bold",
            )
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
            ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    # Legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=MODE_COLORS[m], edgecolor="black", label=MODE_LABELS[m])
        for m in MODE_ORDER
    ] + [
        Patch(facecolor="white", edgecolor="black", label="In-domain"),
        Patch(facecolor="white", edgecolor="black", hatch="//", label="Out-domain"),
    ]
    fig.legend(
        handles=legend_handles, loc="upper center", ncol=5,
        fontsize=9, bbox_to_anchor=(0.5, 1.02),
    )

    fig.suptitle(
        "Baseline Domain — Mean ± Std across 10 Seeds",
        fontsize=14, fontweight="bold", y=1.05,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ===== Plot 2: Box plots =====
def plot_summary_boxplot(df: pd.DataFrame, out_path: Path) -> None:
    """Box plots showing seed variability.

    Layout:
      Rows = 3 distances
      Cols = 3 metrics (F2, AUROC, AUPRC)
      Within each panel: 6 boxes = 3 modes × 2 domains
    """
    box_metrics = ["f2", "auc", "auc_pr"]
    n_rows = len(DIST_ORDER)
    n_cols = len(box_metrics)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    for i, dist in enumerate(DIST_ORDER):
        sub = df[df["distance"] == dist].copy()
        for j, metric in enumerate(box_metrics):
            ax = axes[i, j]

            # Build box data
            box_data = []
            tick_labels = []
            positions = []
            colors_list = []

            pos = 0
            for mode in MODE_ORDER:
                for domain in DOMAIN_ORDER:
                    cell = sub[(sub["mode"] == mode) & (sub["level"] == domain)]
                    vals = cell[metric].dropna().values
                    box_data.append(vals)
                    short_domain = "In" if domain == "in_domain" else "Out"
                    tick_labels.append(f"{MODE_LABELS[mode][:5]}\n{short_domain}")
                    positions.append(pos)
                    colors_list.append(MODE_COLORS[mode])
                    pos += 1
                pos += 0.5  # gap between modes

            bp = ax.boxplot(
                box_data,
                positions=positions,
                widths=0.6,
                patch_artist=True,
                showmeans=True,
                meanprops=dict(marker="D", markerfacecolor="red", markersize=4),
                medianprops=dict(color="black", linewidth=1.5),
            )
            for patch, color in zip(bp["boxes"], colors_list):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            # Scatter individual seeds
            for idx, (vals, p) in enumerate(zip(box_data, positions)):
                jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
                ax.scatter(
                    p + jitter, vals,
                    alpha=0.5, s=15, color="black", zorder=3,
                )

            ax.set_xticks(positions)
            ax.set_xticklabels(tick_labels, fontsize=7)
            ax.set_title(
                f"{DIST_LABELS[dist]} — {METRIC_LABELS.get(metric, metric)}",
                fontsize=10, fontweight="bold",
            )
            ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    fig.suptitle(
        "Baseline Domain — Seed Variability (Box Plots, n=10 seeds)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ===== Plot 3: Heatmaps =====
def plot_summary_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    """Heatmaps: mean F2 / AUROC / AUPRC by mode × distance, per domain.

    Layout: 2 rows (in_domain, out_domain) × 3 cols (F2, AUROC, AUPRC)
    Each cell: mode (row) × distance (col) matrix
    """
    heat_metrics = ["f2", "auc", "auc_pr"]
    n_rows = len(DOMAIN_ORDER)
    n_cols = len(heat_metrics)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))

    for i, domain in enumerate(DOMAIN_ORDER):
        sub = df[df["level"] == domain]
        for j, metric in enumerate(heat_metrics):
            ax = axes[i, j]

            # Build pivot: mode × distance
            pivot = (
                sub.groupby(["mode", "distance"])[metric]
                .mean()
                .unstack(fill_value=0)
            )
            # Reorder
            pivot = pivot.reindex(index=MODE_ORDER, columns=DIST_ORDER)
            pivot.index = [MODE_LABELS.get(m, m) for m in pivot.index]
            pivot.columns = [DIST_LABELS.get(d, d) for d in pivot.columns]

            sns.heatmap(
                pivot,
                annot=True,
                fmt=".3f",
                cmap="YlOrRd",
                ax=ax,
                vmin=0,
                vmax=max(0.5, pivot.values.max() * 1.2),
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
            )
            ax.set_title(
                f"{DOMAIN_LABELS[domain]} — {METRIC_LABELS.get(metric, metric)}",
                fontsize=11, fontweight="bold",
            )
            ax.set_ylabel("")
            ax.set_xlabel("")

    fig.suptitle(
        "Baseline Domain — Mean Metrics (10 Seeds)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ===== Plot 4: Summary table =====
def plot_summary_table(df: pd.DataFrame, out_path: Path) -> None:
    """Render a table figure of mean ± std for all combinations.

    Rows: distance × domain
    Cols: mode × metric
    """
    table_metrics = ["f2", "auc", "auc_pr", "recall", "precision"]

    rows = []
    row_labels = []
    for dist in DIST_ORDER:
        for domain in DOMAIN_ORDER:
            sub = df[(df["distance"] == dist) & (df["level"] == domain)]
            row_label = f"{DIST_LABELS[dist]} / {DOMAIN_LABELS[domain]}"
            row_labels.append(row_label)
            row = {}
            for mode in MODE_ORDER:
                cell = sub[sub["mode"] == mode]
                for metric in table_metrics:
                    vals = cell[metric].dropna()
                    m = vals.mean() if len(vals) > 0 else 0
                    s = vals.std() if len(vals) > 1 else 0
                    col_label = f"{MODE_LABELS[mode][:5]}\n{METRIC_LABELS[metric]}"
                    row[col_label] = f"{m:.3f}±{s:.3f}"
            rows.append(row)

    table_df = pd.DataFrame(rows, index=row_labels)

    fig, ax = plt.subplots(figsize=(max(18, len(table_df.columns) * 2.2), len(row_labels) * 0.7 + 2))
    ax.axis("off")

    tbl = ax.table(
        cellText=table_df.values,
        rowLabels=table_df.index,
        colLabels=table_df.columns,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.5)

    # Color header
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")
        elif c == -1:
            cell.set_facecolor("#D6E4F0")
            cell.set_text_props(fontweight="bold")
        else:
            cell.set_facecolor("#F2F2F2" if r % 2 == 0 else "white")

    fig.suptitle(
        "Baseline Domain — Mean ± Std (10 Seeds)",
        fontsize=13, fontweight="bold", y=0.98,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    print("=" * 60)
    print("Baseline Domain — Seed Summary Plots")
    print("=" * 60)

    if not CSV_PATH.exists():
        print(f"ERROR: CSV not found: {CSV_PATH}")
        return 1

    df = pd.read_csv(CSV_PATH)
    seeds = sorted(df["seed"].unique())
    print(f"Loaded {len(df)} records, {len(seeds)} seeds: {seeds}")
    print(f"Modes: {sorted(df['mode'].unique())}")
    print(f"Distances: {sorted(df['distance'].unique())}")
    print(f"Domains: {sorted(df['level'].unique())}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Plot 1: Grouped bar chart (mean ± std)
    plot_summary_bar(df, OUT_DIR / "baseline_seed_summary_bar.png")

    # Plot 2: Box plots
    plot_summary_boxplot(df, OUT_DIR / "baseline_seed_summary_boxplot.png")

    # Plot 3: Heatmaps
    plot_summary_heatmap(df, OUT_DIR / "baseline_seed_summary_heatmap.png")

    # Plot 4: Summary table
    plot_summary_table(df, OUT_DIR / "baseline_seed_summary_table.png")

    print(f"\nDONE — 4 summary plots saved to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
