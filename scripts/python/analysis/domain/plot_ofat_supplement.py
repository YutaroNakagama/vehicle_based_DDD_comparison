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
MODE_COLORS = {
    "source_only": "#D55E00",   # vermilion
    "target_only": "#0072B2",   # blue
    "mixed":       "#009E73",   # bluish green
}
FAMILY_COLORS = {
    "baseline":      "#999999",   # grey
    "rus_r01":       "#E69F00",   # orange
    "rus_r05":       "#D55E00",   # vermilion
    "smote_r01":     "#0072B2",   # blue
    "smote_r05":     "#56B4E9",   # sky blue
    "sw_smote_r01":  "#CC79A7",   # reddish purple
    "sw_smote_r05":  "#F0E442",   # yellow
}
FAMILY_LEGEND = {
    "BL":       "#999999",
    "RUS":      "#E69F00",
    "SMOTE":    "#0072B2",
    "SW-SMOTE": "#CC79A7",
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

# IEEE T-IV style
_TIV_TEXT_WIDTH = 7.16
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "mathtext.fontset": "stix",
    "svg.fonttype": "none",
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
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
    fig.savefig(out, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
def plot_ofat(df: pd.DataFrame, target_factor: str):
    """OFAT spaghetti plot for one factor.

    For each combination of the *other* three factors (averaged over seeds),
    plot the mean metric across the target factor's levels as one line.
    Also overlay a bold line for the grand OFAT mean ± SD band.
    """
    levels, label_map, display_name = FACTORS[target_factor]
    other_factors = [f for f in FACTORS if f != target_factor]

    # Choose a colour-coding factor: Mode when available, else Rebalancing
    if target_factor != "mode":
        color_factor = "mode"
        color_map = MODE_COLORS
        color_labels = MODE_LABELS
    else:
        color_factor = "condition"
        color_map = FAMILY_COLORS
        color_labels = COND_LABELS

    fig, axes = plt.subplots(1, 3, figsize=(_TIV_TEXT_WIDTH, 3.0), sharey=False)

    for ax_idx, (metric, mlabel) in enumerate(PRIMARY_METRICS):
        ax = axes[ax_idx]

        # Build all fixed conditions (cartesian product of other factors)
        other_combos = (
            df.groupby(other_factors)
            .size()
            .reset_index()
            [other_factors]
            .drop_duplicates()
        )

        all_lines = []   # list of arrays, each shape = (n_levels,)
        combo_labels = []
        combo_labels_raw = []  # raw dict per combo for colour lookup

        for _, combo_row in other_combos.iterrows():
            combo_dict = combo_row.to_dict()
            mask = pd.Series(True, index=df.index)
            for f, v in combo_dict.items():
                mask &= (df[f] == v)
            sub = df[mask]

            # Mean over seeds for each level of the target factor
            means = []
            for lev in levels:
                vals = sub[sub[target_factor] == lev][metric].dropna()
                means.append(vals.mean() if len(vals) > 0 else np.nan)
            all_lines.append(means)

            # Build a short label for this fixed condition
            parts = []
            for f in other_factors:
                flevels, flabels, _ = FACTORS[f]
                parts.append(flabels[combo_dict[f]])
            combo_labels.append("/".join(parts))
            combo_labels_raw.append(combo_dict)

        all_lines = np.array(all_lines)  # shape (n_combos, n_levels)

        # Plot each fixed-condition line, coloured by color_factor
        n_combos = len(all_lines)
        for i in range(n_combos):
            cf_val = combo_labels_raw[i][color_factor]
            color = color_map[cf_val]
            ax.plot(range(len(levels)), all_lines[i], "-o",
                    color=color, alpha=0.40, linewidth=0.8, markersize=3,
                    zorder=2)

        # Compute grand OFAT mean and SD across fixed conditions
        grand_mean = np.nanmean(all_lines, axis=0)
        grand_std  = np.nanstd(all_lines, axis=0)

        # Bold mean line
        ax.plot(range(len(levels)), grand_mean, "-s",
                color="black", linewidth=2.5, markersize=7, zorder=5,
                label="OFAT mean")

        # ±1 SD band
        ax.fill_between(range(len(levels)),
                        grand_mean - grand_std, grand_mean + grand_std,
                        alpha=0.15, color="black", zorder=3,
                        label="±1 SD across\nfixed conditions")

        # Annotate the range (max - min across fixed conditions) per level
        for j in range(len(levels)):
            col = all_lines[:, j]
            rng = np.nanmax(col) - np.nanmin(col)
            ax.annotate(f"$\Delta$={rng:.3f}",
                        xy=(j, grand_mean[j] + grand_std[j]),
                        xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=5.5, fontweight="bold",
                        color="#D55E00")

        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([label_map[lev] for lev in levels],
                           fontsize=7, rotation=30 if len(levels) > 4 else 0,
                           ha="right" if len(levels) > 4 else "center")
        ax.set_ylabel(mlabel)
        ax.set_title(mlabel, fontweight="bold")
        if ax_idx == 0:
            # Build legend: OFAT mean + SD band + colour-factor entries
            handles = ax.get_legend_handles_labels()[0][:2]  # mean + band
            labels_leg = ["OFAT mean", "$\pm$1 SD"]
            if target_factor != "mode":
                # Colour by mode
                for m_key, m_label in MODE_LABELS.items():
                    handles.append(mpatches.Patch(color=MODE_COLORS[m_key], alpha=0.6))
                    labels_leg.append(m_label)
            else:
                # Colour by rebalancing family
                for fam_label, fam_color in FAMILY_LEGEND.items():
                    handles.append(mpatches.Patch(color=fam_color, alpha=0.6))
                    labels_leg.append(fam_label)
            ax.legend(handles, labels_leg, loc="best", framealpha=0.9,
                      fontsize=6, ncol=1)

    fig.tight_layout()

    fname = f"fig_s_ofat_{target_factor}.svg"
    _save(fig, fname)


# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("OFAT Supplementary Analysis")
    print("=" * 60)

    df = load_all_data()

    for i, factor in enumerate(["condition", "distance", "level", "mode"], 1):
        _, _, display = FACTORS[factor]
        print(f"\n[{i}/4] OFAT for {display}...")
        plot_ofat(df, factor)

    print("\n" + "=" * 60)
    print("All 4 OFAT figures saved.")
    print("=" * 60)


if __name__ == "__main__":
    main()
