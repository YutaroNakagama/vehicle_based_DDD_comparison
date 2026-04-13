#!/usr/bin/env python3
"""Bump chart: strategy Friedman mean-ranks across training modes.

Visualises the ranking reversal (Eq. 5 in §5.2) —
cross-domain vs within-domain Spearman ρ = −0.79 to −0.86.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ── paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[4]
CSV_BASE = ROOT / "results/analysis/exp2_domain_shift/figures/csv/split2"
OUT_DIR = ROOT / "results/analysis/exp2_domain_shift/figures/svg/split2/journal_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── constants ──────────────────────────────────────────────────────────
OFFICIAL_SEEDS = [0, 1, 3, 7, 13, 42, 123, 256, 512, 999, 1337, 2024]

CONDITIONS_7 = [
    "baseline", "rus_r01", "rus_r05",
    "smote_r01", "smote_r05", "sw_smote_r01", "sw_smote_r05",
]

LABELS = {
    "baseline":      "BL",
    "rus_r01":       "RUS r=0.1",
    "rus_r05":       "RUS r=0.5",
    "smote_r01":     "SMOTE r=0.1",
    "smote_r05":     "SMOTE r=0.5",
    "sw_smote_r01":  "SW-SM r=0.1",
    "sw_smote_r05":  "SW-SM r=0.5",
}

COND_COLORS = {
    "baseline":      "#999999",   # grey
    "rus_r01":       "#E69F00",   # orange
    "rus_r05":       "#D55E00",   # vermilion
    "smote_r01":     "#56B4E9",   # sky blue
    "smote_r05":     "#0072B2",   # blue
    "sw_smote_r01":  "#CC79A7",   # reddish purple
    "sw_smote_r05":  "#332288",   # dark indigo
}
# CVD-safe: distinct markers + line styles per strategy
COND_MARKERS = {
    "baseline":      "s",
    "rus_r01":       "^",
    "rus_r05":       "v",
    "smote_r01":     "D",
    "smote_r05":     "o",
    "sw_smote_r01":  "P",
    "sw_smote_r05":  "X",
}
COND_LINESTYLES = {
    "baseline":      "-",
    "rus_r01":       "--",
    "rus_r05":       "--",
    "smote_r01":     "-.",
    "smote_r05":     "-.",
    "sw_smote_r01":  ":",
    "sw_smote_r05":  ":",
}

MODE_ORDER = ["source_only", "target_only", "mixed"]
MODE_LABELS = {"source_only": "Cross", "target_only": "Within", "mixed": "Mixed"}

METRICS = {"f2": "F2-score", "auc": "AUROC", "auc_pr": "AUPRC"}

# ── IEEE T-IV style ────────────────────────────────────────────────────
_TIV_COLUMN_WIDTH = 3.5   # inches (single column)
_TIV_TEXT_WIDTH   = 7.16  # inches (double column)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Liberation Sans",
    "mathtext.it": "Liberation Sans:italic",
    "mathtext.bf": "Liberation Sans:bold",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 7,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.grid": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
})


# ── data loading ───────────────────────────────────────────────────────
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
                lambda r, m=method: f"{m}_r{str(r).replace('.', '')}"
                if pd.notna(r) else m
            )
        dfs.append(df)
    merged = pd.concat(dfs, ignore_index=True)
    merged = merged[merged["condition"].isin(CONDITIONS_7)].copy()
    merged = merged[merged["seed"].isin(OFFICIAL_SEEDS)].copy()
    return merged


def compute_per_mode_ranks(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Compute Friedman mean-ranks for each strategy within each mode."""
    records = []
    for mode in MODE_ORDER:
        sub = df[df["mode"] == mode].copy()
        sub["cell"] = (
            sub["seed"].astype(str) + "_"
            + sub["distance"] + "_"
            + sub["level"]
        )
        pivot = sub.groupby(["cell", "condition"])[metric].mean().reset_index()
        wide = pivot.pivot(index="cell", columns="condition", values=metric)
        wide = wide[CONDITIONS_7].dropna()
        mean_ranks = wide.rank(axis=1, ascending=False).mean()
        for cond in CONDITIONS_7:
            records.append({
                "mode": mode,
                "condition": cond,
                "mean_rank": mean_ranks[cond],
            })
    return pd.DataFrame(records)


def compute_spearman(rank_df: pd.DataFrame):
    """Cross-mode Spearman ρ on strategy ranking vectors."""
    results = {}
    for m1, m2 in [
        ("source_only", "target_only"),
        ("source_only", "mixed"),
        ("target_only", "mixed"),
    ]:
        r1 = rank_df[rank_df["mode"] == m1].sort_values("condition")["mean_rank"].values
        r2 = rank_df[rank_df["mode"] == m2].sort_values("condition")["mean_rank"].values
        rho, p = stats.spearmanr(r1, r2)
        results[f"{MODE_LABELS[m1]} vs {MODE_LABELS[m2]}"] = (rho, p)
    return results


# ── plotting ───────────────────────────────────────────────────────────
def plot_bump_chart(df: pd.DataFrame):
    # Column-width figure (3.5 in) — displayed at \columnwidth, no scaling
    fig, axes = plt.subplots(3, 1, figsize=(_TIV_COLUMN_WIDTH, 5.0))

    for ax_idx, (ax, (metric, metric_label)) in enumerate(
        zip(axes, METRICS.items())
    ):
        x_positions = np.arange(len(MODE_ORDER))

        # Compute mean metric value per (condition, mode)
        for cond in CONDITIONS_7:
            means = []
            for mode in MODE_ORDER:
                val = df[(df["condition"] == cond) & (df["mode"] == mode)][metric].mean()
                means.append(val)
            ax.plot(
                x_positions, means,
                marker=COND_MARKERS[cond], markersize=5,
                color=COND_COLORS[cond],
                linestyle=COND_LINESTYLES[cond],
                linewidth=1.5, alpha=0.85,
            )

        ax.set_xticks(x_positions)
        # Only show x-tick labels on the bottom subplot
        if ax_idx == len(METRICS) - 1:
            ax.set_xticklabels([MODE_LABELS[m] for m in MODE_ORDER])
        else:
            ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)  # categorical axis
        ax.set_ylabel(metric_label)

    plt.tight_layout()

    out_path = OUT_DIR / "fig_ranking_reversal.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_path}")

    # Print Spearman results for verification
    for metric, metric_label in METRICS.items():
        rank_df = compute_per_mode_ranks(df, metric)
        spearman = compute_spearman(rank_df)
        print(f"\n{metric_label} Spearman ρ:")
        for pair, (rho, p) in spearman.items():
            print(f"  {pair}: ρ = {rho:.3f}, p = {p:.4f}")

    plt.close()
    return out_path


if __name__ == "__main__":
    data = load_all_data()
    print(f"Loaded {len(data)} rows, {data['condition'].nunique()} conditions")
    plot_bump_chart(data)
