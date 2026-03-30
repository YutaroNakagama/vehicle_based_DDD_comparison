#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_strategy_comparison.py
============================
Generate Fig. 4 — 7-strategy performance comparison by training mode.

Layout: 3 rows (F2, AUROC, AUPRC) × 3 columns (Cross-domain, Within-domain, Mixed).
Each panel shows grouped box plots for the 7 rebalancing strategies, with
individual observations as jittered dots.

Usage:
    python scripts/python/analysis/domain/plot_strategy_comparison.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

CSV_BASE = (
    PROJECT_ROOT / "results" / "analysis" / "exp2_domain_shift"
    / "figures" / "csv" / "split2"
)
OUT_DIR = (
    PROJECT_ROOT / "results" / "analysis" / "exp2_domain_shift"
    / "figures" / "svg" / "split2" / "journal_v2"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants (consistent with plot_journal_figures_v2.py)
# ---------------------------------------------------------------------------
OFFICIAL_SEEDS = {0, 1, 3, 7, 13, 42, 123, 256, 512, 999, 1337, 2024}

CONDITIONS_7 = [
    "baseline", "rus_r01", "rus_r05",
    "smote_r01", "smote_r05", "sw_smote_r01", "sw_smote_r05",
]
COND_LABELS = {
    "baseline":      "BL",
    "rus_r01":       "RUS\nr=0.1",
    "rus_r05":       "RUS\nr=0.5",
    "smote_r01":     "SM\nr=0.1",
    "smote_r05":     "SM\nr=0.5",
    "sw_smote_r01":  "SW-SM\nr=0.1",
    "sw_smote_r05":  "SW-SM\nr=0.5",
}
COND_COLORS = {
    "baseline":      "#999999",   # grey
    "rus_r01":       "#E69F00",   # orange
    "rus_r05":       "#D55E00",   # vermilion
    "smote_r01":     "#0072B2",   # blue
    "smote_r05":     "#56B4E9",   # sky blue
    "sw_smote_r01":  "#CC79A7",   # reddish purple
    "sw_smote_r05":  "#F0E442",   # yellow
}
MODE_ORDER = ["source_only", "target_only", "mixed"]
MODE_LABELS = {
    "source_only": "Cross-domain",
    "target_only": "Within-domain",
    "mixed":       "Mixed",
}
PRIMARY_METRICS = [("f2", "F2-score"), ("auc", "AUROC"), ("auc_pr", "AUPRC")]

# IEEE T-IV style
_TIV_TEXT_WIDTH = 7.16   # double-column inches
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "mathtext.fontset": "stix",
    "svg.fonttype": "none",
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


# ---------------------------------------------------------------------------
# Data loading (same logic as plot_journal_figures_v2.py)
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
    merged = merged[merged["seed"].isin(OFFICIAL_SEEDS)].copy()
    print(f"  Loaded {len(merged)} records "
          f"({merged['seed'].nunique()} seeds, "
          f"{merged['condition'].nunique()} conditions)")
    return merged


# ---------------------------------------------------------------------------
# Aggregate to block-level means for cleaner box plots
# ---------------------------------------------------------------------------
def aggregate_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean over (distance × level × seed) within each
    (condition, mode) cell — gives 72 observations per condition per mode
    (3 distances × 2 levels × 12 seeds)."""
    return (
        df.groupby(["condition", "mode", "distance", "level", "seed"])
        [["f2", "auc", "auc_pr"]]
        .mean()
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------
def plot_strategy_comparison(df: pd.DataFrame):
    """3×3 grid: rows = metrics, cols = modes. Box + strip plot."""
    agg = aggregate_blocks(df)

    fig, axes = plt.subplots(
        3, 3, figsize=(_TIV_TEXT_WIDTH, 6.0),
        sharey="row", sharex=True,
    )

    for row, (mcol, mlabel) in enumerate(PRIMARY_METRICS):
        for col, mode in enumerate(MODE_ORDER):
            ax = axes[row, col]
            subset = agg[agg["mode"] == mode]

            # Prepare data for each condition
            data_list = []
            for cond in CONDITIONS_7:
                vals = subset.loc[subset["condition"] == cond, mcol].values
                data_list.append(vals)

            positions = np.arange(len(CONDITIONS_7))

            # Box plots
            bp = ax.boxplot(
                data_list,
                positions=positions,
                widths=0.55,
                patch_artist=True,
                showfliers=False,
                medianprops=dict(color="black", linewidth=1.2),
                whiskerprops=dict(color="gray", linewidth=0.8),
                capprops=dict(color="gray", linewidth=0.8),
            )
            for patch, cond in zip(bp["boxes"], CONDITIONS_7):
                patch.set_facecolor(COND_COLORS[cond])
                patch.set_alpha(0.7)
                patch.set_edgecolor("black")
                patch.set_linewidth(0.6)

            # Jittered strip overlay
            rng = np.random.default_rng(42)
            for i, (cond, vals) in enumerate(zip(CONDITIONS_7, data_list)):
                jitter = rng.uniform(-0.15, 0.15, size=len(vals))
                ax.scatter(
                    positions[i] + jitter, vals,
                    s=6, alpha=0.35,
                    color=COND_COLORS[cond],
                    edgecolors="none",
                    zorder=3,
                )

            # Axis formatting
            ax.set_xticks(positions)
            if row == 2:
                ax.set_xticklabels(
                    [COND_LABELS[c] for c in CONDITIONS_7],
                    fontsize=6, ha="center",
                )
            else:
                ax.set_xticklabels([])

            if col == 0:
                ax.set_ylabel(mlabel, fontweight="bold")

            if row == 0:
                ax.set_title(MODE_LABELS[mode], fontweight="bold")

            ax.grid(axis="y", alpha=0.3, linestyle="--")
            ax.grid(axis="x", visible=False)

    # Legend
    handles = [
        mpatches.Patch(
            facecolor=COND_COLORS[c], edgecolor="black",
            linewidth=0.6, alpha=0.7,
            label=COND_LABELS[c].replace("\n", " "),
        )
        for c in CONDITIONS_7
    ]
    fig.legend(
        handles=handles, loc="lower center",
        ncol=7, fontsize=6.5, frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout(rect=[0, 0.03, 1, 1])

    out = OUT_DIR / "fig4_strategy_comparison.svg"
    fig.savefig(out, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading data...")
    df = load_all_data()
    print("Plotting Fig. 4 — Strategy comparison...")
    plot_strategy_comparison(df)
    print("Done.")
