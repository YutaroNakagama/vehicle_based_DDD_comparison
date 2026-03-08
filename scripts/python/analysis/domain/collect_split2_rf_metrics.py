#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_split2_rf_metrics.py
============================
Collect evaluation metrics from split2 domain analysis experiments (RF model)
and generate summary_metrics_bar plots.

This script complements collect_split2_metrics.py (BalancedRF) by handling
RF-based experiment 2 conditions:
  - baseline_domain  (RF, no ratio)
  - smote_plain      (RF, ratios 0.1, 0.5)
  - undersample_rus  (RF, ratios 0.1, 0.5)
  - sw_smote         (RF, ratios 0.1, 0.5)  [subject-wise SMOTE]

Output locations (per-condition subdirectory):
  - CSV: results/analysis/exp2_domain_shift/figures/csv/split2/{condition}/*.csv
  - PNG: results/analysis/exp2_domain_shift/figures/png/split2/{condition}/*.png

Pipeline:
  1. Scan RF evaluation JSONs for experiment 2 conditions
  2. Parse condition / mode / distance / domain / ratio / seed from filename
  3. Build per-condition CSV
  4. Generate per-(condition × ratio × seed) bar plots

Usage:
    python scripts/python/analysis/domain/collect_split2_rf_metrics.py
    python scripts/python/analysis/domain/collect_split2_rf_metrics.py --dry-run
    python scripts/python/analysis/domain/collect_split2_rf_metrics.py --condition smote_plain
"""

from __future__ import annotations

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
EVAL_DIR = Path(cfg.RESULTS_EVALUATION_PATH) / "RF"
FIG_BASE = Path(cfg.RESULTS_DOMAIN_ANALYSIS_PATH) / "figures"
CSV_BASE = FIG_BASE / "csv" / "split2"
PNG_BASE = FIG_BASE / "png" / "split2"

# Condition → subdirectory name mapping
CONDITION_DIR_MAP = {
    "baseline_domain":  "baseline",
    "smote_plain":      "smote_plain",
    "undersample_rus":  "undersample_rus",
    "sw_smote":         "sw_smote",
}

METRICS = ["accuracy", "precision", "recall", "f1", "f2", "auc", "auc_pr"]

# --- Conditions handled -----------------------------------------------
RF_CONDITIONS = ["baseline_domain", "smote_plain", "undersample_rus", "sw_smote"]

# --- Filename parsing ------------------------------------------------
# Handles:
#   eval_results_RF_{mode}_{condition}_knn_{distance}_{domain}_{mode_tag}[_split2][_ratio{r}]_s{seed}.json
SPLIT2_RF_PATTERN = re.compile(
    r"eval_results_RF_"
    r"(?P<mode>source_only|target_only|mixed)_"
    r"(?P<condition>baseline_domain|smote_plain|undersample_rus)_"
    r"knn_"
    r"(?P<distance>mmd|dtw|wasserstein)_"
    r"(?P<domain>in_domain|out_domain)_"
    r"(?:source_only|target_only|mixed)"
    r"(?:_split2)?"
    r"(?:_ratio(?P<ratio>[0-9.]+))?"
    r"_s(?P<seed>\d+)"
    r"\.json$"
)

# Pooled patterns
# eval_results_RF_pooled_baseline_s{seed}.json
# eval_results_RF_pooled_undersample_rus_ratio{r}_s{seed}.json
# eval_results_RF_pooled_smote_ratio{r}_s{seed}.json   (fallback for smote_plain)
POOLED_RF_PATTERNS = [
    re.compile(
        r"eval_results_RF_pooled_baseline_s(?P<seed>\d+)\.json$"
    ),
    re.compile(
        r"eval_results_RF_pooled_undersample_rus_ratio(?P<ratio>[0-9.]+)_s(?P<seed>\d+)\.json$"
    ),
    re.compile(
        r"eval_results_RF_pooled_smote_ratio(?P<ratio>[0-9.]+)_s(?P<seed>\d+)\.json$"
    ),
    re.compile(
        r"eval_results_RF_pooled_subjectwise_smote_ratio(?P<ratio>[0-9.]+)_s(?P<seed>\d+)\.json$"
    ),
]

# --- sw_smote filename patterns (3 naming variants) --------------------
# Variant 1: swsmote  (no ratio, s42 only — oldest runs)
SW_SMOTE_PAT1 = re.compile(
    r"eval_results_RF_"
    r"(?P<mode>source_only|target_only|mixed)_"
    r"swsmote_knn_"
    r"(?P<distance>mmd|dtw|wasserstein)_"
    r"(?P<domain>in_domain|out_domain)"
    r"(?:_split2)"
    r"_s(?P<seed>\d+)"
    r"\.json$"
)
# Variant 2: smote_subjectwise  (ratio with dot, e.g. ratio0.1)
SW_SMOTE_PAT2 = re.compile(
    r"eval_results_RF_"
    r"(?P<mode>source_only|target_only|mixed)_"
    r"smote_subjectwise_knn_"
    r"(?P<distance>mmd|dtw|wasserstein)_"
    r"(?P<domain>in_domain|out_domain)"
    r"(?:_split2)"
    r"_ratio(?P<ratio>[0-9.]+)"
    r"_s(?P<seed>\d+)"
    r"\.json$"
)
# Variant 3: imbalv3  (latest runs, ratio with dot, has _subjectwise tag)
SW_SMOTE_PAT3 = re.compile(
    r"eval_results_RF_"
    r"(?P<mode>source_only|target_only|mixed)_"
    r"imbalv3_knn_"
    r"(?P<distance>mmd|dtw|wasserstein)_"
    r"(?P<domain>in_domain|out_domain)_"
    r"(?:source_only|target_only|mixed)"
    r"(?:_split2)"
    r"(?:_subjectwise)"
    r"_ratio(?P<ratio>[0-9.]+)"
    r"_s(?P<seed>\d+)"
    r"\.json$"
)
# Variant 4: smote  (ratio with underscore, e.g. ratio0_5)
SW_SMOTE_PAT4 = re.compile(
    r"eval_results_RF_"
    r"(?P<mode>source_only|target_only|mixed)_"
    r"smote_knn_"
    r"(?P<distance>mmd|dtw|wasserstein)_"
    r"(?P<domain>in_domain|out_domain)"
    r"(?:_split2)"
    r"_ratio(?P<ratio>[0-9_]+)"
    r"_s(?P<seed>\d+)"
    r"\.json$"
)
SW_SMOTE_PATTERNS = [SW_SMOTE_PAT2, SW_SMOTE_PAT3, SW_SMOTE_PAT4]


POOLED_CONDITION_MAP = {
    0: "baseline_domain",
    1: "undersample_rus",
    2: "smote_plain",
    3: "sw_smote",
}


def parse_eval_filename(name: str) -> dict | None:
    """Return parsed metadata dict or None if filename does not match."""
    # Try main pattern first (baseline_domain, smote_plain, undersample_rus)
    m = SPLIT2_RF_PATTERN.match(name)
    if m:
        d = m.groupdict()
        if d["ratio"] is None:
            d["ratio"] = ""
        return d

    # Try sw_smote patterns
    for pat in SW_SMOTE_PATTERNS:
        m = pat.match(name)
        if m:
            d = m.groupdict()
            d["condition"] = "sw_smote"
            # Normalize ratio: None → "", underscore → dot
            ratio = d.get("ratio") or ""
            d["ratio"] = ratio.replace("_", ".")
            return d

    return None


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
def collect_all_split2_rf(condition_filter: str | None = None) -> pd.DataFrame:
    """Scan RF eval JSONs and return a DataFrame."""
    records = []
    for json_path in sorted(EVAL_DIR.rglob("eval_results_RF_*knn*.json")):
        meta = parse_eval_filename(json_path.name)
        if meta is None:
            continue
        if condition_filter and meta["condition"] != condition_filter:
            continue

        # Determine job_id from directory structure
        job_id = json_path.parent.parent.name

        metrics = load_eval_json(json_path)
        row = {
            "job_id": job_id,
            "mode": meta["mode"],
            "distance": meta["distance"],
            "level": meta["domain"],
            "seed": int(meta["seed"]),
            "condition": meta["condition"],
            "ratio": meta["ratio"],
        }
        row.update(metrics)
        records.append(row)

    df = pd.DataFrame(records)
    if df.empty:
        logger.warning("No RF split2 evaluation JSONs found!")
        return df

    # De-duplicate: keep latest job_id per unique combination
    key_cols = ["mode", "distance", "level", "seed", "condition", "ratio"]
    df = df.sort_values("job_id").groupby(key_cols).last().reset_index()

    logger.info(f"Collected {len(df)} records across conditions: "
                f"{sorted(df['condition'].unique())}")
    return df


# --- Step 1b: Collect pooled reference data ---------------------------
def collect_pooled_rf() -> pd.DataFrame:
    """Scan RF pooled evaluation JSONs for baseline / undersample_rus / smote."""
    records = []
    for json_path in sorted(EVAL_DIR.rglob("eval_results_RF_pooled_*.json")):
        matched = False
        for idx, pat in enumerate(POOLED_RF_PATTERNS):
            m = pat.match(json_path.name)
            if m:
                cond = POOLED_CONDITION_MAP[idx]
                d = m.groupdict()
                job_id = json_path.parent.parent.name
                metrics = load_eval_json(json_path)
                row = {
                    "job_id": job_id,
                    "mode": "pooled",
                    "distance": "pooled",
                    "level": "pooled",
                    "seed": int(d["seed"]),
                    "condition": cond,
                    "ratio": d.get("ratio", ""),
                }
                row.update(metrics)
                records.append(row)
                matched = True
                break

    df = pd.DataFrame(records)
    if not df.empty:
        key_cols = ["condition", "seed", "ratio"]
        df = df.sort_values("job_id").groupby(key_cols).last().reset_index()
        logger.info(f"Collected {len(df)} pooled records: "
                    f"{sorted(df['condition'].unique())}")
    else:
        logger.warning("No RF pooled evaluation JSONs found.")
    return df


# --- Step 2: Save CSVs -----------------------------------------------
CONDITION_CSV_MAP = {
    "baseline_domain":  "baseline_domain_split2_metrics_v2.csv",
    "smote_plain":      "smote_plain_split2_metrics_v2.csv",
    "undersample_rus":  "undersample_rus_split2_metrics_v2.csv",
    "sw_smote":         "sw_smote_split2_metrics_v2.csv",
}


def save_csvs(df: pd.DataFrame) -> dict[str, Path]:
    """Write one CSV per condition, returning {condition: path}."""
    paths = {}

    for cond, cond_df in df.groupby("condition"):
        cond_dir = CONDITION_DIR_MAP.get(cond, cond)
        csv_dir = CSV_BASE / cond_dir
        csv_dir.mkdir(parents=True, exist_ok=True)
        fname = CONDITION_CSV_MAP.get(cond, f"{cond}_split2_metrics_v2.csv")
        out = csv_dir / fname

        cols = ["job_id", "mode", "distance", "level", "seed", "ratio"] + METRICS
        cols = [c for c in cols if c in cond_df.columns]

        cond_df[cols].to_csv(out, index=False)
        logger.info(f"  CSV saved: {out}  ({len(cond_df)} rows)")
        paths[cond] = out

    return paths


# --- Step 3: Generate plots ------------------------------------------
def _ratio_label(ratio: str) -> str:
    """Convert ratio string to label: '0.1' → 'r01', '0.5' → 'r05'."""
    if not ratio:
        return ""
    r = ratio.replace(".", "")
    if r.startswith("0"):
        return f"_r{r}"
    return f"_r{r}"


def generate_plots(df: pd.DataFrame, df_pooled: pd.DataFrame) -> list[Path]:
    """Generate summary_metrics_bar_*.png per condition × ratio × seed."""
    generated = []

    # Group by (condition, ratio, seed)
    group_cols = ["condition", "ratio", "seed"]
    for (cond, ratio, seed), sub in df.groupby(group_cols):
        # Resolve output subdirectory
        cond_dir = CONDITION_DIR_MAP.get(cond, cond)
        png_dir = PNG_BASE / cond_dir
        png_dir.mkdir(parents=True, exist_ok=True)

        # --- Merge pooled reference rows ---------------------------------
        pooled_rows = pd.DataFrame()
        if not df_pooled.empty:
            # 1) Exact match: same condition, seed, ratio
            pooled_cond = df_pooled[
                (df_pooled["condition"] == cond)
                & (df_pooled["seed"] == seed)
                & (df_pooled["ratio"] == ratio)
            ]
            # 2) Fallback: same condition, any seed (mean across seeds)
            if pooled_cond.empty:
                pooled_cond_all = df_pooled[
                    (df_pooled["condition"] == cond)
                    & (df_pooled["ratio"] == ratio)
                ]
                if not pooled_cond_all.empty:
                    mean_row = pooled_cond_all.select_dtypes(include="number").mean()
                    template = pooled_cond_all.iloc[[0]].copy()
                    for col in mean_row.index:
                        template[col] = mean_row[col]
                    template["seed"] = seed
                    pooled_cond = template
            # 3) Fallback: baseline_domain pooled, same seed
            if pooled_cond.empty:
                pooled_cond = df_pooled[
                    (df_pooled["condition"] == "baseline_domain")
                    & (df_pooled["seed"] == seed)
                ]
            # 4) Fallback: baseline_domain pooled, any seed (mean)
            if pooled_cond.empty:
                bl_all = df_pooled[
                    df_pooled["condition"] == "baseline_domain"
                ]
                if not bl_all.empty:
                    mean_row = bl_all.select_dtypes(include="number").mean()
                    template = bl_all.iloc[[0]].copy()
                    for col in mean_row.index:
                        template[col] = mean_row[col]
                    template["seed"] = seed
                    pooled_cond = template
            if not pooled_cond.empty:
                pooled_rows = pooled_cond.copy()
                pooled_rows["condition"] = cond
                pooled_rows["ratio"] = ratio

        if not pooled_rows.empty:
            common_cols = sub.columns.intersection(pooled_rows.columns)
            sub_with_pooled = pd.concat(
                [sub, pooled_rows[common_cols]], ignore_index=True
            )
        else:
            sub_with_pooled = sub

        # Build output filename (short: condition + ratio + seed)
        cond_short = {
            "baseline_domain": "baseline",
            "smote_plain": "smote",
            "undersample_rus": "rus",
            "sw_smote": "sw_smote",
        }.get(cond, cond)
        ratio_tag = _ratio_label(ratio)
        out_name = f"{cond_short}{ratio_tag}_s{seed}.png"
        out_path = png_dir / out_name

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
        description="Collect split2 RF metrics → CSV + plots"
    )
    parser.add_argument("--condition", default=None,
                        help="Filter to single condition (e.g., smote_plain)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Collect & print but do not write files")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Collect split2 domain metrics (RF)")
    logger.info("=" * 60)
    logger.info(f"Eval directory : {EVAL_DIR}")
    logger.info(f"CSV output     : {CSV_BASE}/<condition>/")
    logger.info(f"PNG output     : {PNG_BASE}/<condition>/")

    # Step 1a: Collect domain-split data
    df = collect_all_split2_rf(condition_filter=args.condition)
    if df.empty:
        logger.error("No data found. Exiting.")
        return 1

    # Step 1b: Collect pooled reference data
    df_pooled = collect_pooled_rf()

    logger.info(f"\nConditions : {sorted(df['condition'].unique())}")
    logger.info(f"Modes      : {sorted(df['mode'].unique())}")
    logger.info(f"Distances  : {sorted(df['distance'].unique())}")
    logger.info(f"Domains    : {sorted(df['level'].unique())}")
    logger.info(f"Seeds      : {sorted(df['seed'].unique())}")
    if "ratio" in df.columns:
        logger.info(f"Ratios     : {sorted(df['ratio'].unique())}")
    logger.info(f"Total rows : {len(df)}  (+{len(df_pooled)} pooled)")

    if args.dry_run:
        logger.info("\n[DRY-RUN] Would write the following:")
        for cond, cond_df in df.groupby("condition"):
            cond_dir = CONDITION_DIR_MAP.get(cond, cond)
            fname = CONDITION_CSV_MAP.get(cond, f"{cond}_split2_metrics_v2.csv")
            logger.info(f"  {CSV_BASE / cond_dir / fname}  ({len(cond_df)} rows)")
        logger.info("\n[DRY-RUN] Would generate plots for:")
        for (cond, ratio, seed), sub in df.groupby(["condition", "ratio", "seed"]):
            ratio_tag = _ratio_label(ratio)
            cond_short = {"baseline_domain": "baseline", "smote_plain": "smote", "undersample_rus": "rus", "sw_smote": "sw_smote"}.get(cond, cond)
            name = f"{cond_short}{ratio_tag}_s{seed}.png"
            logger.info(f"  {name}  ({len(sub)} rows)")
        return 0

    # Step 2: Save CSVs
    logger.info("\n--- Saving CSVs ---")
    save_csvs(df)

    # Step 3: Generate plots
    logger.info("\n--- Generating plots ---")
    plots = generate_plots(df, df_pooled)

    logger.info("\n" + "=" * 60)
    logger.info(f"DONE — {len(df)} records, {len(plots)} plots generated")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
