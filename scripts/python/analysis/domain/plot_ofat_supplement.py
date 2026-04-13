#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_ofat_supplement.py
=======================
Supplementary OFAT (One-Factor-At-a-Time) analysis to complement
the Sobol sensitivity analysis in Fig. 2.

For each of the four experimental factors (R, D, G, M), all other
factors are fixed at every possible combination, and the mean
performance across the target factor's levels is plotted as a line.
This produces a "spaghetti plot" showing:
  - Whether the factor's effect is consistent across fixed conditions
  - How much the OFAT estimate varies depending on which condition is fixed

Output:
  results/analysis/exp2_domain_shift/figures/png/split2/journal_v2/
    fig_s1_ofat_rebalancing.png
    fig_s2_ofat_distance.png
    fig_s3_ofat_membership.png
    fig_s4_ofat_mode.png

Usage:
    python scripts/python/analysis/domain/plot_ofat_supplement.py
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
OFFICIAL_SEEDS = {0, 1, 3, 7, 13, 42, 123, 256, 512, 999, 1337, 2024}

CONDITIONS_7 = [
    "baseline", "rus_r01", "rus_r05",
    "smote_r01", "smote_r05", "sw_smote_r01", "sw_smote_r05",
]
COND_LABELS = {
    "baseline":      "BL",
    "rus_r01":       "RUS .1",
    "rus_r05":       "RUS .5",
    "smote_r01":     "SM .1",
    "smote_r05":     "SM .5",
    "sw_smote_r01":  "SW .1",
    "sw_smote_r05":  "SW .5",
}
MODE_LABELS  = {"source_only": "Cross", "target_only": "Within", "mixed": "Mixed"}
DIST_LABELS  = {"mmd": "MMD", "dtw": "DTW", "wasserstein": "Wass."}
LEVEL_LABELS = {"in_domain": "In", "out_domain": "Out"}

# Colour-coding by the most informative other factor
# IEEE CVD-accessible: colour + line style + marker shape
# Palette: Bang Wong (Nature Methods 2011) — NO blue-green pair;
# grey-scale luminance spread verified: vermilion ≈ 47 %, sky blue ≈ 72 %,
# reddish purple ≈ 55 %  →  all separable in B/W.
MODE_COLORS = {
    "source_only": "#D55E00",   # vermilion  (dark,  lum ≈ 47 %)
    "target_only": "#56B4E9",   # sky blue   (light, lum ≈ 72 %)
    "mixed":       "#CC79A7",   # reddish purple (medium, lum ≈ 55 %)
}
MODE_LINESTYLES = {
    "source_only": ("-",  "o"),   # solid, circle
    "target_only": ("--", "s"),   # dashed, square
    "mixed":       (":",  "^"),   # dotted, triangle up
}
FAMILY_COLORS = {
    "baseline":      "#666666",   # dark grey
    "rus_r01":       "#E69F00",   # orange  (bright)
    "rus_r05":       "#D55E00",   # vermilion (darker orange)
    "smote_r01":     "#56B4E9",   # sky blue (light)
    "smote_r05":     "#0072B2",   # blue     (darker sky blue)
    "sw_smote_r01":  "#CC79A7",   # reddish purple (light)
    "sw_smote_r05":  "#882255",   # dark purple
}
# Map each condition to its family style (linestyle, marker)
FAMILY_LINESTYLES = {
    "baseline":      ("-",  "o"),   # solid, circle
    "rus_r01":       ("--", "s"),   # dashed, square
    "rus_r05":       ("--", "s"),
    "smote_r01":     (":",  "D"),   # dotted, diamond
    "smote_r05":     (":",  "D"),
    "sw_smote_r01":  ("-.", "^"),   # dashdot, triangle
    "sw_smote_r05":  ("-.", "^"),
}

MODES     = ["source_only", "target_only", "mixed"]
DISTANCES = ["mmd", "dtw", "wasserstein"]
LEVELS    = ["in_domain", "out_domain"]
PRIMARY_METRICS = [("f2", "F2-score"), ("auc", "AUROC"), ("auc_pr", "AUPRC")]

# Factor definitions:  (column_name, level_list, label_map, display_name)
FACTORS = {
    "condition": (CONDITIONS_7, COND_LABELS, "Rebalancing ($R$)"),
    "distance":  (DISTANCES,    DIST_LABELS,  "Distance ($D$)"),
    "level":     (LEVELS,       LEVEL_LABELS, "Membership ($G$)"),
    "mode":      (MODES,        MODE_LABELS,  "Mode ($M$)"),
}

# Distinct colours for fixed-condition lines (cycling palette)
LINE_CMAP = plt.cm.tab20

# IEEE T-IV style — full-width figure (figure*)
_TIV_TEXT_WIDTH = 7.16   # IEEE two-column text width (inches)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Liberation Sans",
    "mathtext.it": "Liberation Sans:italic",
    "mathtext.bf": "Liberation Sans:bold",
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "figure.dpi": 150,
    "axes.grid": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
})


# ---------------------------------------------------------------------------
def load_all_data() -> pd.DataFrame:
    files = {
        "baseline": CSV_BASE / "baseline"       / "baseline_domain_split2_metrics_v2.csv",
        "smote":    CSV_BASE / "smote_plain"     / "smote_plain_split2_metrics_v2.csv",
        "rus":      CSV_BASE / "undersample_rus" / "undersample_rus_split2_metrics_v2.csv",
        "sw_smote": CSV_BASE / "sw_smote"        / "sw_smote_split2_metrics_v2.csv",
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
    print(f"  Loaded {len(merged)} records")
    return merged


def _save(fig, name: str):
    out = OUT_DIR / name
    fig.savefig(out.with_suffix('.pdf'), format="pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Column order: dominant factors first, then negligible
FACTOR_ORDER = ["condition", "mode", "distance", "level"]


def _compute_ofat_lines(df, target_factor):
    """Return per-metric OFAT lines for one factor."""
    levels, _, _ = FACTORS[target_factor]
    other_factors = [f for f in FACTORS if f != target_factor]

    other_combos = (
        df.groupby(other_factors).size().reset_index()
        [other_factors].drop_duplicates()
    )

    results = {}
    combo_labels_raw = []

    for metric, _ in PRIMARY_METRICS:
        all_lines = []
        clr = []
        for _, combo_row in other_combos.iterrows():
            combo_dict = combo_row.to_dict()
            mask = pd.Series(True, index=df.index)
            for f, v in combo_dict.items():
                mask &= (df[f] == v)
            sub = df[mask]
            means = []
            for lev in levels:
                vals = sub[sub[target_factor] == lev][metric].dropna()
                means.append(vals.mean() if len(vals) > 0 else np.nan)
            all_lines.append(means)
            if metric == PRIMARY_METRICS[0][0]:
                clr.append(combo_dict)
        all_lines = np.array(all_lines)
        results[metric] = (all_lines, np.nanmean(all_lines, axis=0),
                           np.nanstd(all_lines, axis=0))
        if metric == PRIMARY_METRICS[0][0]:
            combo_labels_raw = clr

    return results, combo_labels_raw


def plot_ofat_combined(df: pd.DataFrame):
    """Combined 3-row × 4-column OFAT spaghetti plot."""
    n_rows = len(PRIMARY_METRICS)
    n_cols = len(FACTOR_ORDER)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(_TIV_TEXT_WIDTH, 4.2),
        sharey=True,
    )

    for col_idx, target_factor in enumerate(FACTOR_ORDER):
        levels, label_map, display_name = FACTORS[target_factor]

        if target_factor != "mode":
            color_factor = "mode"
            color_map, style_map = MODE_COLORS, MODE_LINESTYLES
        else:
            color_factor = "condition"
            color_map, style_map = FAMILY_COLORS, FAMILY_LINESTYLES

        results, combo_labels_raw = _compute_ofat_lines(df, target_factor)

        for row_idx, (metric, mlabel) in enumerate(PRIMARY_METRICS):
            ax = axes[row_idx, col_idx]
            all_lines, grand_mean, grand_std = results[metric]

            for i in range(len(all_lines)):
                cf_val = combo_labels_raw[i][color_factor]
                ls, mk = style_map[cf_val]
                ax.plot(range(len(levels)), all_lines[i],
                        linestyle=ls, marker=mk,
                        color=color_map[cf_val], alpha=0.45,
                        linewidth=1.0, markersize=3.0, zorder=2)

            ax.plot(range(len(levels)), grand_mean,
                    linestyle="-", marker="P",
                    color="black", linewidth=2.2, markersize=5.5, zorder=5,
                    label="OFAT mean")
            ax.fill_between(range(len(levels)),
                            grand_mean - grand_std, grand_mean + grand_std,
                            alpha=0.15, color="black", zorder=3,
                            label="$\\pm$1 SD")

            ax.set_ylim(0, 1)

            # Column title — top row only
            if row_idx == 0:
                ax.set_title(display_name, fontsize=9, fontweight="bold")

            # Legend in F2-score row (row_idx==0) for each column
            if row_idx == 0:
                from matplotlib.lines import Line2D
                _leg_kw = dict(
                    loc="upper left", fontsize=5.5, handlelength=2.0,
                    framealpha=1.0, fancybox=False,
                    edgecolor="black", facecolor="white",
                )
                if target_factor != "mode":
                    # Mode legend for R, D, G columns
                    handles = [
                        Line2D([0], [0], color=MODE_COLORS[m],
                               linestyle=ls, marker=mk,
                               markersize=3.5, linewidth=1.0,
                               label=MODE_LABELS[m])
                        for m, (ls, mk) in MODE_LINESTYLES.items()
                    ]
                    ax.legend(handles=handles, **_leg_kw)
                else:
                    # Family legend for M column (one entry per family)
                    _family_repr = {
                        "BL":       ("baseline",     "-",  "o"),
                        "RUS":      ("rus_r01",      "--", "s"),
                        "SMOTE":    ("smote_r01",    ":",  "D"),
                        "SW-SMOTE": ("sw_smote_r01", "-.", "^"),
                    }
                    handles = [
                        Line2D([0], [0],
                               color=FAMILY_COLORS[cond],
                               linestyle=ls, marker=mk,
                               markersize=3.5, linewidth=1.0,
                               label=fname)
                        for fname, (cond, ls, mk) in _family_repr.items()
                    ]
                    ax.legend(handles=handles, **_leg_kw)

            # Y-axis label — leftmost column only
            if col_idx == 0:
                ax.set_ylabel(mlabel)
            else:
                ax.set_ylabel("")
                ax.tick_params(axis="y", labelleft=False)

            # X-axis ticks — bottom row only
            ax.set_xticks(range(len(levels)))
            if row_idx == n_rows - 1:
                ax.set_xticklabels(
                    [label_map[lev] for lev in levels],
                    fontsize=8,
                    rotation=35 if len(levels) > 4 else 0,
                    ha="right" if len(levels) > 4 else "center",
                )
            else:
                ax.set_xticklabels([])

    fig.tight_layout(h_pad=0.4, w_pad=0.4)
    _save(fig, "fig_ofat_combined.svg")


# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("OFAT Combined Analysis (3×4 grid)")
    print("=" * 60)

    df = load_all_data()
    plot_ofat_combined(df)

    print("\n" + "=" * 60)
    print("Combined OFAT figure saved.")
    print("=" * 60)


if __name__ == "__main__":
    main()
