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
    "smote_r01":     "#0072B2",   # blue
    "smote_r05":     "#56B4E9",   # sky blue
    "sw_smote_r01":  "#CC79A7",   # reddish purple
    "sw_smote_r05":  "#F0E442",   # yellow
}

MODE_ORDER = ["source_only", "target_only", "mixed"]
MODE_LABELS = {"source_only": "Cross", "target_only": "Within", "mixed": "Mixed"}

METRICS = {"f2": "F2-score", "auc": "AUROC", "auc_pr": "AUPRC"}

# ── IEEE T-IV style ────────────────────────────────────────────────────
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
    "legend.fontsize": 7,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
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
    fig, axes = plt.subplots(1, 3, figsize=(_TIV_TEXT_WIDTH, 2.5), sharey=True)

    for ax, (metric, metric_label) in zip(axes, METRICS.items()):
        rank_df = compute_per_mode_ranks(df, metric)
        x_positions = np.arange(len(MODE_ORDER))

        for cond in CONDITIONS_7:
            cond_data = rank_df[rank_df["condition"] == cond]
            ranks = [
                cond_data[cond_data["mode"] == m]["mean_rank"].values[0]
                for m in MODE_ORDER
            ]
            ax.plot(
                x_positions, ranks,
                marker="o", markersize=4,
                color=COND_COLORS[cond],
                linewidth=1.5, alpha=0.85,
                label=LABELS[cond],
            )
            # Label on right side
            ax.annotate(
                LABELS[cond],
                xy=(x_positions[-1], ranks[-1]),
                xytext=(4, 0),
                textcoords="offset points",
                fontsize=5.5,
                color=COND_COLORS[cond],
                va="center",
                fontweight="bold",
            )

        ax.set_xticks(x_positions)
        ax.set_xticklabels([MODE_LABELS[m] for m in MODE_ORDER])
        ax.set_title(metric_label)
        ax.set_ylim(7.5, 0.5)  # rank 1 at top
        ax.set_yticks(range(1, 8))

        # Spearman annotation
        spearman = compute_spearman(rank_df)
        rho_cw = spearman["Cross vs Within"][0]
        p_cw = spearman["Cross vs Within"][1]
        sig = "*" if p_cw < 0.05 else ""
        ax.text(
            0.5, 0.02,
            f"$\\rho_{{C,W}}$ = {rho_cw:.2f}{sig}",
            transform=ax.transAxes,
            ha="center", fontsize=6.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.7),
        )

    axes[0].set_ylabel("Friedman Mean Rank")

    # Single legend at bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.02),
        frameon=True,
        fontsize=6.5,
    )

    plt.tight_layout()

    out_path = OUT_DIR / "fig_ranking_reversal.svg"
    fig.savefig(out_path, format="svg", bbox_inches="tight", facecolor="white")
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
