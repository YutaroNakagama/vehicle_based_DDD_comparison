#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_split2_metrics.py
=========================
Collect evaluation metrics from split2 domain analysis experiments (BalancedRF)
and generate summary_metrics_bar plots.

Output locations (per-condition subdirectory):
  - CSV: results/analysis/exp2_domain_shift/figures/csv/split2/{condition}/*.csv
  - PNG: results/analysis/exp2_domain_shift/figures/png/split2/{condition}/*.png

Pipeline:
  1. Scan BalancedRF evaluation JSONs with *split2* tag
  2. Parse condition / mode / distance / domain / seed from filename
  3. Build per-condition CSV  (figures/csv/split2/{condition}/)
  4. Generate per-seed bar plots (figures/png/split2/{condition}/)

Usage:
    python scripts/python/analysis/domain/collect_split2_metrics.py
    python scripts/python/analysis/domain/collect_split2_metrics.py --dry-run
    python scripts/python/analysis/domain/collect_split2_metrics.py --condition baseline
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src import config as cfg
from src.utils.visualization.visualization import (
    plot_grouped_bar_chart_raw,
    save_figure,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- Paths -----------------------------------------------------------
EVAL_DIR = Path(cfg.RESULTS_EVALUATION_PATH) / "BalancedRF"
FIG_BASE = Path(cfg.RESULTS_DOMAIN_ANALYSIS_PATH) / "figures"
CSV_BASE = FIG_BASE / "csv" / "split2"
PNG_BASE = FIG_BASE / "png" / "split2"

# Condition → subdirectory name mapping
CONDITION_DIR_MAP = {
    "baseline":     "balanced_rf",
    "balanced_rf":  "balanced_rf",
    "smote_plain":  "smote_plain",
    "smote":        "smote_plain",   # subject-wise variant goes to smote_plain/
    "undersample":  "undersample_rus",
}

METRICS = ["accuracy", "precision", "recall", "f1", "f2", "auc", "auc_pr"]

# --- Filename parsing ------------------------------------------------
# Example: eval_results_BalancedRF_source_only_baseline_knn_mmd_out_domain_split2_s42.json
# Older:   eval_results_BalancedRF_source_only_balanced_rf_knn_wasserstein_out_domain_source_only_s42.json
SPLIT2_PATTERN = re.compile(
    r"eval_results_BalancedRF_"
    r"(?P<mode>source_only|target_only|mixed)_"
    r"(?P<condition>[a-z_]+?)_"        # greedy‐minimal before 'knn'
    r"knn_"
    r"(?P<distance>mmd|dtw|wasserstein)_"
    r"(?P<domain>in_domain|out_domain)"
    r"(?:_split2)?(?:_(?:source_only|target_only|mixed))?"  # optional trailing tags
    r"_s(?P<seed>\d+)"
    r"\.json$"
)


def parse_eval_filename(name: str) -> dict | None:
    """Return parsed metadata dict or None if filename does not match."""
    m = SPLIT2_PATTERN.match(name)
    if not m:
        return None
    return m.groupdict()


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


# --- Step 1: Collect -------------------------------------------------
def collect_all_split2(condition_filter: str | None = None) -> pd.DataFrame:
    """Scan BalancedRF eval JSONs and return a DataFrame."""
    records = []
    for json_path in sorted(EVAL_DIR.rglob("eval_results_*knn*.json")):
        meta = parse_eval_filename(json_path.name)
        if meta is None:
            continue
        # Skip baseline condition — RF baseline is the canonical one (in baseline/)
        if meta["condition"] == "baseline":
            continue
        if condition_filter and meta["condition"] != condition_filter:
            continue

        # Determine job_id from directory structure (.../JOBID/JOBID[n]/file)
        job_id = json_path.parent.parent.name

        metrics = load_eval_json(json_path)
        row = {
            "job_id": job_id,
            "mode": meta["mode"],
            "distance": meta["distance"],
            "level": meta["domain"],        # level == domain for split2
            "seed": int(meta["seed"]),
            "condition": meta["condition"],
        }
        row.update(metrics)
        records.append(row)

    df = pd.DataFrame(records)
    if df.empty:
        logger.warning("No split2 evaluation JSONs found!")
        return df

    # Deduplicate: keep only the latest job_id per unique key
    key_cols = ["condition", "mode", "distance", "level", "seed"]
    before = len(df)
    df = df.sort_values("job_id").groupby(key_cols).last().reset_index()
    if len(df) < before:
        logger.info(f"Dedup: {before} → {len(df)} rows (removed {before - len(df)} duplicates)")

    logger.info(f"Collected {len(df)} records across conditions: "
                f"{sorted(df['condition'].unique())}")
    return df


# --- Step 1b: Collect pooled (all-subjects) reference data -----------
# Example: eval_results_BalancedRF_pooled_balanced_rf_s42.json
POOLED_PATTERN = re.compile(
    r"eval_results_BalancedRF_pooled_(?P<condition>[a-z_]+?)_s(?P<seed>\d+)\.json$"
)


def collect_pooled_data() -> pd.DataFrame:
    """Scan BalancedRF pooled evaluation JSONs (mode=pooled, no domain split).

    These serve as the 'all subjects' baseline for Row 4 and the dashed
    horizontal reference lines in Rows 1-3 of the bar-chart grid.
    """
    records = []
    for json_path in sorted(EVAL_DIR.rglob("eval_results_BalancedRF_pooled_*_s*.json")):
        m = POOLED_PATTERN.match(json_path.name)
        if m is None:
            continue

        job_id = json_path.parent.parent.name
        metrics = load_eval_json(json_path)
        row = {
            "job_id": job_id,
            "mode": "pooled",
            "distance": "pooled",   # placeholder — not used per-distance
            "level": "pooled",      # placeholder
            "seed": int(m.group("seed")),
            "condition": m.group("condition"),
        }
        row.update(metrics)
        records.append(row)

    df = pd.DataFrame(records)
    if not df.empty:
        # Keep only the latest run per (condition, seed) — highest job_id
        df = df.sort_values("job_id").groupby(["condition", "seed"]).last().reset_index()
        logger.info(f"Collected {len(df)} pooled records: "
                    f"{sorted(df['condition'].unique())}")
    else:
        logger.warning("No pooled evaluation JSONs found.")
    return df


# --- Step 2: Save CSVs -----------------------------------------------
CONDITION_CSV_MAP = {
    "baseline":     "baseline_split2_metrics_v2.csv",
    "balanced_rf":  "balancedrf_split2_metrics_v2.csv",
    "smote_plain":  "smote_split2_metrics.csv",
    "smote":        "swsmote_split2_metrics.csv",
    "undersample":  "rus_split2_metrics.csv",
}


def save_csvs(df: pd.DataFrame) -> dict[str, Path]:
    """Write one CSV per condition, returning {condition: path}."""
    paths = {}

    for cond, cond_df in df.groupby("condition"):
        cond_dir = CONDITION_DIR_MAP.get(cond, cond)
        csv_dir = CSV_BASE / cond_dir
        csv_dir.mkdir(parents=True, exist_ok=True)
        fname = CONDITION_CSV_MAP.get(cond, f"{cond}_split2_metrics.csv")
        out = csv_dir / fname

        # Add ratio column if ratio-based condition (match existing format)
        if cond in ("smote_plain", "smote", "undersample"):
            if "ratio" not in cond_df.columns:
                cond_df = cond_df.copy()
                cond_df["ratio"] = ""

        cols = ["job_id", "mode", "distance", "level", "seed"]
        if "ratio" in cond_df.columns:
            cols.append("ratio")
        cols += METRICS
        cols = [c for c in cols if c in cond_df.columns]

        cond_df[cols].to_csv(out, index=False)
        logger.info(f"  CSV saved: {out}  ({len(cond_df)} rows)")
        paths[cond] = out

    return paths


# --- Step 3: Generate plots ------------------------------------------
def generate_plots(df: pd.DataFrame, df_pooled: pd.DataFrame) -> list[Path]:
    """Generate summary_metrics_bar_*.png per condition × seed.

    Pooled data (mode='pooled') is merged into each condition so that
    ``plot_grouped_bar_chart_raw`` can draw the 4th row and dashed
    baseline lines.
    """
    generated = []

    for (cond, seed), sub in df.groupby(["condition", "seed"]):
        # Resolve output subdirectory
        cond_dir = CONDITION_DIR_MAP.get(cond, cond)
        png_dir = PNG_BASE / cond_dir
        png_dir.mkdir(parents=True, exist_ok=True)

        # --- Merge pooled reference rows for this seed ---------------
        # Use condition-matched pooled if available, otherwise fall back
        # to 'balanced_rf' pooled (common BalancedRF baseline).
        pooled_rows = pd.DataFrame()
        if not df_pooled.empty:
            pooled_cond = df_pooled[
                (df_pooled["condition"] == cond) & (df_pooled["seed"] == seed)
            ]
            if pooled_cond.empty:
                # Fallback: use balanced_rf pooled as generic reference
                pooled_cond = df_pooled[
                    (df_pooled["condition"] == "balanced_rf") & (df_pooled["seed"] == seed)
                ]
            if not pooled_cond.empty:
                pooled_rows = pooled_cond.copy()
                pooled_rows["condition"] = cond  # align condition label

        if not pooled_rows.empty:
            common_cols = sub.columns.intersection(pooled_rows.columns)
            sub_with_pooled = pd.concat(
                [sub, pooled_rows[common_cols]], ignore_index=True
            )
        else:
            sub_with_pooled = sub

        # Build a short label for filename
        cond_label = cond.replace("_plain", "").replace("balanced_rf", "balancedrf")
        out_name = f"summary_metrics_bar_{cond_label}_v2_s{seed}.png"
        out_path = png_dir / out_name

        # plot_grouped_bar_chart_raw expects modes and levels
        fig = plot_grouped_bar_chart_raw(
            data=sub_with_pooled,
            metrics=METRICS,
            modes=["pooled", "source_only", "target_only", "mixed"],
            distance_col="distance",
            level_col="level",
            baseline_rates={"auc_pr": 0.033},
        )
        if fig is not None:
            save_figure(fig, str(out_path), dpi=200)
            plt.close(fig)
            generated.append(out_path)
            logger.info(f"  PNG saved: {out_path}")

    return generated


# --- Main -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Collect split2 BalancedRF metrics → CSV + plots"
    )
    parser.add_argument("--condition", default=None,
                        help="Filter to single condition (e.g., baseline)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Collect & print but do not write files")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Collect split2 domain metrics (BalancedRF)")
    logger.info("=" * 60)
    logger.info(f"Eval directory : {EVAL_DIR}")
    logger.info(f"CSV output     : {CSV_BASE}/<condition>/")
    logger.info(f"PNG output     : {PNG_BASE}/<condition>/")

    # Step 1a: Collect domain-split data
    df = collect_all_split2(condition_filter=args.condition)
    if df.empty:
        logger.error("No data found. Exiting.")
        return 1

    # Step 1b: Collect pooled (all-subjects) reference data
    df_pooled = collect_pooled_data()

    logger.info(f"\nConditions : {sorted(df['condition'].unique())}")
    logger.info(f"Modes      : {sorted(df['mode'].unique())}")
    logger.info(f"Distances  : {sorted(df['distance'].unique())}")
    logger.info(f"Domains    : {sorted(df['level'].unique())}")
    logger.info(f"Seeds      : {sorted(df['seed'].unique())}")
    logger.info(f"Total rows : {len(df)}  (+{len(df_pooled)} pooled)")

    if args.dry_run:
        logger.info("\n[DRY-RUN] Would write the following:")
        for cond, cond_df in df.groupby("condition"):
            cond_dir = CONDITION_DIR_MAP.get(cond, cond)
            fname = CONDITION_CSV_MAP.get(cond, f"{cond}_split2_metrics.csv")
            logger.info(f"  {CSV_BASE / cond_dir / fname}  ({len(cond_df)} rows)")
        return 0

    # Step 2: Save CSVs
    logger.info("\n--- Saving CSVs ---")
    save_csvs(df)

    # Step 3: Generate plots (with pooled reference)
    logger.info("\n--- Generating plots ---")
    plots = generate_plots(df, df_pooled)

    logger.info("\n" + "=" * 60)
    logger.info(f"DONE — {len(df)} records, {len(plots)} plots generated")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
