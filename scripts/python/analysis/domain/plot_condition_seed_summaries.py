#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_condition_seed_summaries.py
================================
Generate seed-aggregated summary plots (mean ± std) for each experimental
condition, matching the layout style of plot_grouped_bar_chart_raw (4 rows
× 7 metric columns).

Conditions handled:
  - smote_plain     (ratio 0.1, 0.5) → smote_summary_r01.png, smote_summary_r05.png
  - undersample_rus (ratio 0.1, 0.5) → rus_summary_r01.png, rus_summary_r05.png
  - sw_smote        (ratio 0.1, 0.5) → sw_smote_summary_r01.png, sw_smote_summary_r05.png
  - balanced_rf     (no ratio)        → brf_summary.png

Pooled data availability:
  - smote_plain, undersample_rus: pooled row available (seed 42, 123)
  - sw_smote, balanced_rf: no pooled data → 3-row plot

Usage:
    python scripts/python/analysis/domain/plot_condition_seed_summaries.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[4]
CSV_BASE = (
    PROJECT_ROOT
    / "results/analysis/exp2_domain_shift/figures/csv/split2"
)
PNG_BASE = (
    PROJECT_ROOT
    / "results/analysis/exp2_domain_shift/figures/png/split2"
)
EVAL_DIR = PROJECT_ROOT / "results/outputs/evaluation/RF"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
METRICS = ["accuracy", "precision", "recall", "f1", "f2", "auc", "auc_pr"]
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision (pos)",
    "recall": "Recall (pos)",
    "f1": "F1",
    "f2": "F2",
    "auc": "AUROC",
    "auc_pr": "AUPRC",
}
MODE_ORDER = ["source_only", "target_only", "mixed"]
MODE_LABELS = {
    "source_only": "Cross-domain",
    "target_only": "Within-domain",
    "mixed": "Multi-domain",
    "pooled": "Pooled",
}
DIST_ORDER = ["dtw", "mmd", "wasserstein"]
DIST_LABELS = {"dtw": "DTW", "mmd": "MMD", "wasserstein": "Wasserstein"}
DOMAIN_ORDER = ["out_domain", "in_domain"]
DOMAIN_LABELS = {"out_domain": "Out-domain", "in_domain": "In-domain"}

COLORS = {
    "pooled": "#66cc99",
    "source_only": "#6699cc",
    "target_only": "#ff9966",
    "mixed": "#cc99ff",
}

# ---------------------------------------------------------------------------
# Pooled data loading
# ---------------------------------------------------------------------------
POOLED_RF_PATTERNS = [
    (re.compile(r"eval_results_RF_pooled_baseline_s(?P<seed>\d+)\.json$"),
     "baseline_domain", None),
    (re.compile(r"eval_results_RF_pooled_undersample_rus_ratio(?P<ratio>[0-9.]+)_s(?P<seed>\d+)\.json$"),
     "undersample_rus", "ratio"),
    (re.compile(r"eval_results_RF_pooled_smote_ratio(?P<ratio>[0-9.]+)_s(?P<seed>\d+)\.json$"),
     "smote_plain", "ratio"),
    (re.compile(r"eval_results_RF_pooled_subjectwise_smote_ratio(?P<ratio>[0-9.]+)_s(?P<seed>\d+)\.json$"),
     "sw_smote", "ratio"),
]


def load_eval_json(path: Path) -> dict:
    """Load evaluation JSON and extract relevant metrics."""
    with open(path) as f:
        d = json.load(f)
    prec = d.get("precision", 0.0)
    rec = d.get("recall", 0.0)
    f1 = d.get("f1", 0.0)
    f2 = (5 * prec * rec) / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0
    return {
        "accuracy": d.get("accuracy", 0.0),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "f2": f2,
        "auc": d.get("roc_auc", d.get("auc", np.nan)),
        "auc_pr": d.get("auc_pr", d.get("average_precision", np.nan)),
    }


def collect_pooled_rf() -> pd.DataFrame:
    """Collect pooled RF evaluation data."""
    records = []
    for json_path in sorted(EVAL_DIR.rglob("eval_results_RF_pooled_*.json")):
        for pat, cond, ratio_key in POOLED_RF_PATTERNS:
            m = pat.match(json_path.name)
            if m:
                d = m.groupdict()
                metrics = load_eval_json(json_path)
                row = {
                    "mode": "pooled",
                    "distance": "pooled",
                    "level": "pooled",
                    "seed": int(d["seed"]),
                    "condition": cond,
                    "ratio": d.get("ratio", ""),
                }
                row.update(metrics)
                records.append(row)
                break
    df = pd.DataFrame(records)
    if not df.empty:
        key_cols = ["condition", "seed", "ratio"]
        df = df.sort_values("seed").groupby(key_cols).last().reset_index()
    return df


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------
def plot_condition_summary(
    df: pd.DataFrame,
    df_pooled: pd.DataFrame,
    condition: str,
    ratio: str,
    out_path: Path,
    title_prefix: str,
) -> None:
    """Generate a seed-aggregated summary plot (mean ± std).

    Layout matches plot_grouped_bar_chart_raw:
      Rows 1-3: DTW / MMD / Wasserstein
      Row 4: Pooled (if data available)
      Cols: 7 metrics (Accuracy, Precision, Recall, F1, F2, AUROC, AUPRC)
    """
    has_pooled = not df_pooled.empty
    n_rows = 4 if has_pooled else 3
    n_cols = len(METRICS)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 3 * n_rows),
        squeeze=False,
    )

    # Compute pooled baselines (mean across pooled seeds) for dashed line
    pooled_baselines = {}
    if has_pooled:
        for metric in METRICS:
            vals = df_pooled[metric].dropna()
            if len(vals) > 0:
                pooled_baselines[metric] = vals.mean()

    n_seeds = df["seed"].nunique()

    # --- Rows 1-3: Distance metrics ---
    for i, dist in enumerate(DIST_ORDER):
        sub = df[df["distance"] == dist]

        for j, metric in enumerate(METRICS):
            ax = axes[i, j]

            # Group: 3 modes × 2 domains = 6 bars
            levels_present = [lv for lv in DOMAIN_ORDER if lv in sub["level"].unique()]
            x_positions = np.arange(len(levels_present))
            width = 0.25
            comparison_modes = MODE_ORDER

            for idx, mode in enumerate(comparison_modes):
                means = []
                stds = []
                for lvl in levels_present:
                    cell = sub[(sub["mode"] == mode) & (sub["level"] == lvl)]
                    vals = cell[metric].dropna()
                    means.append(vals.mean() if len(vals) > 0 else 0)
                    stds.append(vals.std() if len(vals) > 1 else 0)

                offset = (idx - 1) * width
                bars = ax.bar(
                    x_positions + offset,
                    means,
                    width,
                    yerr=stds,
                    capsize=3,
                    label=MODE_LABELS.get(mode, mode),
                    color=COLORS[mode],
                    edgecolor="black",
                    linewidth=0.3,
                    error_kw={"linewidth": 0.8},
                )

                # Value labels
                for pos_idx, (m, s) in enumerate(zip(means, stds)):
                    if m > 0:
                        ax.text(
                            pos_idx + offset, m + s + 0.005,
                            f"{m:.2f}",
                            ha="center", va="bottom",
                            fontsize=7, color="black",
                        )

            # Add pooled baseline dashed line
            if metric in pooled_baselines:
                pooled_val = pooled_baselines[metric]
                ax.axhline(
                    pooled_val, color="black", linestyle="--",
                    linewidth=2, alpha=0.7, zorder=3,
                )
                ax.text(
                    len(levels_present) - 0.1, pooled_val,
                    f"Pooled ({pooled_val:.3f})",
                    fontsize=8, color="black",
                    ha="right", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8),
                )

            ax.set_xticks(x_positions)
            ax.set_xticklabels(
                [DOMAIN_LABELS.get(lv, lv) for lv in levels_present],
                fontsize=9,
            )
            ax.set_ylim(0, 1.0)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
            ax.grid(axis="y", alpha=0.3, linewidth=0.5)

            # Title on top row
            if i == 0:
                ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=11)

            # Distance label on left column
            if j == 0:
                ax.text(
                    0.02, 0.95, DIST_LABELS.get(dist, dist),
                    transform=ax.transAxes,
                    ha="left", va="top",
                    fontsize=12, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6),
                )

            # Legend on top-right subplot
            if i == 0 and j == n_cols - 1:
                ax.legend(
                    loc="upper right", fontsize=8, frameon=False,
                )

    # --- Row 4: Pooled ---
    if has_pooled:
        for j, metric in enumerate(METRICS):
            ax = axes[3, j]

            pooled_vals = df_pooled[metric].dropna()
            if len(pooled_vals) == 0:
                ax.axis("off")
                continue

            pooled_mean = pooled_vals.mean()
            pooled_std = pooled_vals.std() if len(pooled_vals) > 1 else 0

            bar = ax.bar(
                [0], [pooled_mean],
                0.6,
                yerr=[pooled_std],
                capsize=4,
                color=COLORS["pooled"],
                edgecolor="black",
                linewidth=0.3,
                error_kw={"linewidth": 0.8},
            )

            ax.text(
                0, pooled_mean + pooled_std + 0.005,
                f"{pooled_mean:.2f}",
                ha="center", va="bottom",
                fontsize=7, color="black",
            )

            ax.set_xticks([0])
            ax.set_xticklabels(["Pooled\nBaseline"])
            ax.set_xlim(-0.5, 0.5)
            ax.set_ylim(0, 1.0)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
            ax.grid(axis="y", alpha=0.3, linewidth=0.5)

            # "Pooled" label on left column
            if j == 0:
                ax.text(
                    0.02, 0.95, "Pooled",
                    transform=ax.transAxes,
                    ha="left", va="top",
                    fontsize=12, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6),
                )

    # Super title
    ratio_txt = f" (ratio={ratio})" if ratio else ""
    fig.suptitle(
        f"{title_prefix}{ratio_txt} — Mean ± Std across {n_seeds} Seeds",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Condition specifications
# ---------------------------------------------------------------------------
CONDITION_SPECS = [
    {
        "condition": "baseline_domain",
        "csv_dir": "baseline",
        "csv_file": "baseline_domain_split2_metrics_v2.csv",
        "png_dir": "baseline",
        "ratios": [None],  # No ratio
        "out_prefix": "baseline_summary",
        "title_prefix": "Baseline",
        "has_ratio_col": False,
        "pooled_condition": "baseline_domain",
    },
    {
        "condition": "smote_plain",
        "csv_dir": "smote_plain",
        "csv_file": "smote_plain_split2_metrics_v2.csv",
        "png_dir": "smote_plain",
        "ratios": ["0.1", "0.5"],
        "out_prefix": "smote_summary",
        "title_prefix": "SMOTE (Plain)",
        "has_ratio_col": True,
        "pooled_condition": "smote_plain",
    },
    {
        "condition": "undersample_rus",
        "csv_dir": "undersample_rus",
        "csv_file": "undersample_rus_split2_metrics_v2.csv",
        "png_dir": "undersample_rus",
        "ratios": ["0.1", "0.5"],
        "out_prefix": "rus_summary",
        "title_prefix": "Random Under-Sampling",
        "has_ratio_col": True,
        "pooled_condition": "undersample_rus",
    },
    {
        "condition": "sw_smote",
        "csv_dir": "sw_smote",
        "csv_file": "sw_smote_split2_metrics_v2.csv",
        "png_dir": "sw_smote",
        "ratios": ["0.1", "0.5"],
        "out_prefix": "sw_smote_summary",
        "title_prefix": "Subject-Wise SMOTE",
        "has_ratio_col": True,
        "pooled_condition": "sw_smote",
    },
    {
        "condition": "balanced_rf",
        "csv_dir": "balanced_rf",
        "csv_file": "balancedrf_split2_metrics_v2.csv",
        "png_dir": "balanced_rf",
        "ratios": [None],  # No ratio
        "out_prefix": "brf_summary",
        "title_prefix": "Balanced RF",
        "has_ratio_col": False,
        "pooled_condition": None,  # No pooled data
    },
]


def main():
    print("=" * 60)
    print("Generate seed-aggregated summary plots for all conditions")
    print("=" * 60)

    # Load pooled data once
    df_pooled_all = collect_pooled_rf()
    print(f"Pooled data: {len(df_pooled_all)} records")
    if not df_pooled_all.empty:
        print(f"  Conditions: {sorted(df_pooled_all['condition'].unique())}")
        print(f"  Seeds: {sorted(df_pooled_all['seed'].unique())}")

    generated = []

    for spec in CONDITION_SPECS:
        csv_path = CSV_BASE / spec["csv_dir"] / spec["csv_file"]
        if not csv_path.exists():
            print(f"\n[SKIP] CSV not found: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        print(f"\n--- {spec['condition']} ---")
        print(f"  CSV: {csv_path} ({len(df)} rows)")
        print(f"  Seeds: {sorted(df['seed'].unique())}")

        for ratio in spec["ratios"]:
            # Filter by ratio
            if ratio is not None and spec["has_ratio_col"]:
                ratio_val = float(ratio)
                df_ratio = df[df["ratio"].apply(lambda x: abs(float(x) - ratio_val) < 0.01)]
                ratio_tag = f"_r{ratio.replace('.', '')}"
                if ratio_tag.startswith("_r0"):
                    ratio_tag = f"_r{ratio.replace('.', '')}"
            else:
                df_ratio = df.copy()
                ratio_tag = ""
                ratio = ""

            if df_ratio.empty:
                print(f"  [SKIP] No data for ratio={ratio}")
                continue

            n_seeds = df_ratio["seed"].nunique()
            print(f"  Ratio={ratio or 'N/A'}: {len(df_ratio)} rows, {n_seeds} seeds")

            # Get pooled data for this condition & ratio
            df_pooled_cond = pd.DataFrame()
            if spec["pooled_condition"] and not df_pooled_all.empty:
                mask = df_pooled_all["condition"] == spec["pooled_condition"]
                if ratio:
                    mask = mask & (df_pooled_all["ratio"].apply(
                        lambda x: abs(float(x) - float(ratio)) < 0.01 if x else False
                    ))
                df_pooled_cond = df_pooled_all[mask].copy()
                if not df_pooled_cond.empty:
                    print(f"    Pooled: {len(df_pooled_cond)} records, "
                          f"seeds={sorted(df_pooled_cond['seed'].unique())}")
                else:
                    # Fallback to baseline pooled
                    baseline_pooled = df_pooled_all[
                        df_pooled_all["condition"] == "baseline_domain"
                    ]
                    if not baseline_pooled.empty:
                        df_pooled_cond = baseline_pooled.copy()
                        print(f"    Pooled (fallback to baseline): "
                              f"{len(df_pooled_cond)} records")

            # Output path — ratio tag goes BEFORE _summary
            #   e.g. smote_r01_summary.png, rus_r05_summary.png, brf_summary.png
            base_prefix = spec["out_prefix"].replace("_summary", "")
            out_name = f"{base_prefix}{ratio_tag}_summary.png"
            out_path = PNG_BASE / spec["png_dir"] / out_name

            plot_condition_summary(
                df=df_ratio,
                df_pooled=df_pooled_cond,
                condition=spec["condition"],
                ratio=ratio,
                out_path=out_path,
                title_prefix=spec["title_prefix"],
            )
            generated.append(out_path)

    print(f"\n{'=' * 60}")
    print(f"DONE — {len(generated)} summary plots generated")
    for p in generated:
        print(f"  {p}")
    print("=" * 60)


if __name__ == "__main__":
    sys.exit(main() or 0)
