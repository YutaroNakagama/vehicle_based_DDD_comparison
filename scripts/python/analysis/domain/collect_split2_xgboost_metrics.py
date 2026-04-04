#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_split2_xgboost_metrics.py
==================================
Collect evaluation metrics from split2 domain analysis experiments
using XGBoost model and generate aggregated CSVs for Sobol analysis.

Mirrors collect_split2_rf_metrics.py but for XGBoost validation runs.

Conditions handled (same 7 as RF):
  - baseline         (XGBoost, no ratio)
  - smote_plain      (XGBoost, ratios 0.1, 0.5)
  - undersample_rus  (XGBoost, ratios 0.1, 0.5)
  - sw_smote         (XGBoost, ratios 0.1, 0.5)  [subject-wise SMOTE]

Output:
  results/analysis/exp2_domain_shift/figures/csv/split2/xgboost/{condition}/*.csv

Usage:
    python scripts/python/analysis/domain/collect_split2_xgboost_metrics.py
    python scripts/python/analysis/domain/collect_split2_xgboost_metrics.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src import config as cfg

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- Paths -----------------------------------------------------------
EVAL_DIR = Path(cfg.RESULTS_EVALUATION_PATH) / "XGBoost"
FIG_BASE = Path(cfg.RESULTS_DOMAIN_ANALYSIS_PATH) / "figures"
CSV_BASE = FIG_BASE / "csv" / "split2" / "xgboost"

METRICS = ["accuracy", "precision", "recall", "f1", "f2", "auc", "auc_pr"]

# --- Filename parsing ------------------------------------------------
# XGBoost eval filenames follow the same pattern as RF but with XGBoost model,
# and tags prefixed with xgb_
# Pattern: eval_results_XGBoost_{mode}_{tag_details}.json

# Baseline: xgb_baseline_domain_knn_{distance}_{domain}_{mode}_split2_s{seed}
XGB_BASELINE_PAT = re.compile(
    r"eval_results_XGBoost_"
    r"(?P<mode>source_only|target_only|mixed)_"
    r"xgb_baseline_domain_knn_"
    r"(?P<distance>mmd|dtw|wasserstein)_"
    r"(?P<domain>in_domain|out_domain)_"
    r"(?:source_only|target_only|mixed)"
    r"(?:_split2)"
    r"_s(?P<seed>\d+)"
    r"\.json$"
)

# SMOTE plain: xgb_smote_plain_knn_{distance}_{domain}_{mode}_split2_ratio{r}_s{seed}
XGB_SMOTE_PLAIN_PAT = re.compile(
    r"eval_results_XGBoost_"
    r"(?P<mode>source_only|target_only|mixed)_"
    r"xgb_smote_plain_knn_"
    r"(?P<distance>mmd|dtw|wasserstein)_"
    r"(?P<domain>in_domain|out_domain)_"
    r"(?:source_only|target_only|mixed)"
    r"(?:_split2)"
    r"_ratio(?P<ratio>[0-9.]+)"
    r"_s(?P<seed>\d+)"
    r"\.json$"
)

# Undersample RUS: xgb_undersample_rus_knn_{distance}_{domain}_{mode}_split2_ratio{r}_s{seed}
XGB_UNDERSAMPLE_PAT = re.compile(
    r"eval_results_XGBoost_"
    r"(?P<mode>source_only|target_only|mixed)_"
    r"xgb_undersample_rus_knn_"
    r"(?P<distance>mmd|dtw|wasserstein)_"
    r"(?P<domain>in_domain|out_domain)_"
    r"(?:source_only|target_only|mixed)"
    r"(?:_split2)"
    r"_ratio(?P<ratio>[0-9.]+)"
    r"_s(?P<seed>\d+)"
    r"\.json$"
)

# SW-SMOTE (subject-wise): xgb_imbalv3_knn_{distance}_{domain}_{mode}_split2_subjectwise_ratio{r}_s{seed}
XGB_SW_SMOTE_PAT = re.compile(
    r"eval_results_XGBoost_"
    r"(?P<mode>source_only|target_only|mixed)_"
    r"xgb_imbalv3_knn_"
    r"(?P<distance>mmd|dtw|wasserstein)_"
    r"(?P<domain>in_domain|out_domain)_"
    r"(?:source_only|target_only|mixed)"
    r"(?:_split2)"
    r"(?:_subjectwise)"
    r"_ratio(?P<ratio>[0-9.]+)"
    r"_s(?P<seed>\d+)"
    r"\.json$"
)

PATTERNS = [
    ("baseline", XGB_BASELINE_PAT),
    ("smote_plain", XGB_SMOTE_PLAIN_PAT),
    ("undersample_rus", XGB_UNDERSAMPLE_PAT),
    ("sw_smote", XGB_SW_SMOTE_PAT),
]

# Condition → subdirectory mapping
CONDITION_DIR_MAP = {
    "baseline": "baseline",
    "smote_plain": "smote_plain",
    "undersample_rus": "undersample_rus",
    "sw_smote": "sw_smote",
}

CONDITION_CSV_MAP = {
    "baseline": "xgb_baseline_domain_split2_metrics_v2.csv",
    "smote_plain": "xgb_smote_plain_split2_metrics_v2.csv",
    "undersample_rus": "xgb_undersample_rus_split2_metrics_v2.csv",
    "sw_smote": "xgb_sw_smote_split2_metrics_v2.csv",
}


def parse_eval_filename(name: str) -> dict | None:
    """Return parsed metadata dict or None if filename does not match."""
    for condition, pat in PATTERNS:
        m = pat.match(name)
        if m:
            d = m.groupdict()
            d["condition"] = condition
            if "ratio" not in d or d["ratio"] is None:
                d["ratio"] = ""
            return d
    return None


def load_eval_json(path: Path) -> dict:
    """Load evaluation JSON and extract relevant metrics."""
    with open(path) as f:
        d = json.load(f)

    prec = d.get("precision", 0.0)
    rec = d.get("recall", 0.0)
    f2 = (5 * prec * rec) / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0

    return {
        "accuracy": d.get("accuracy", 0.0),
        "precision": prec,
        "recall": rec,
        "f1": d.get("f1", 0.0),
        "f2": f2,
        "auc": d.get("roc_auc", d.get("auc", np.nan)),
        "auc_pr": d.get("auc_pr", d.get("average_precision", np.nan)),
    }


def collect_all_split2_xgboost() -> pd.DataFrame:
    """Scan XGBoost eval JSONs and return a DataFrame."""
    if not EVAL_DIR.exists():
        logger.warning(f"XGBoost eval directory not found: {EVAL_DIR}")
        return pd.DataFrame()

    records = []
    for json_path in sorted(EVAL_DIR.rglob("eval_results_XGBoost_*.json")):
        meta = parse_eval_filename(json_path.name)
        if meta is None:
            continue

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
        logger.warning("No XGBoost split2 evaluation JSONs found!")
        return df

    # De-duplicate: keep latest job_id per unique combination
    key_cols = ["mode", "distance", "level", "seed", "condition", "ratio"]
    df = df.sort_values("job_id").groupby(key_cols).last().reset_index()

    logger.info(f"Collected {len(df)} XGBoost records across conditions: "
                f"{sorted(df['condition'].unique())}")
    return df


def save_csvs(df: pd.DataFrame) -> dict[str, Path]:
    """Write one CSV per condition, returning {condition: path}."""
    paths = {}
    for cond, cond_df in df.groupby("condition"):
        cond_dir = CONDITION_DIR_MAP.get(cond, cond)
        csv_dir = CSV_BASE / cond_dir
        csv_dir.mkdir(parents=True, exist_ok=True)
        fname = CONDITION_CSV_MAP.get(cond, f"xgb_{cond}_split2_metrics_v2.csv")
        out = csv_dir / fname

        cols = ["job_id", "mode", "distance", "level", "seed", "ratio"] + METRICS
        cols = [c for c in cols if c in cond_df.columns]

        cond_df[cols].to_csv(out, index=False)
        logger.info(f"  CSV saved: {out}  ({len(cond_df)} rows)")
        paths[cond] = out

    return paths


def print_summary(df: pd.DataFrame):
    """Print factorial coverage summary."""
    if df.empty:
        return

    print("\n" + "=" * 60)
    print("XGBoost Factorial Coverage Summary")
    print("=" * 60)

    conditions = sorted(df["condition"].unique())
    modes = sorted(df["mode"].unique())
    distances = sorted(df["distance"].unique())
    levels = sorted(df["level"].unique())
    seeds = sorted(df["seed"].unique())

    print(f"  Conditions: {len(conditions)}  {conditions}")
    print(f"  Modes:      {len(modes)}  {modes}")
    print(f"  Distances:  {len(distances)}  {distances}")
    print(f"  Levels:     {len(levels)}  {levels}")
    print(f"  Seeds:      {len(seeds)}  (total: {len(seeds)})")

    # Map conditions to the 7-level factor
    cond_7 = set()
    for _, row in df.iterrows():
        cond = row["condition"]
        ratio = row.get("ratio", "")
        if cond == "baseline":
            cond_7.add("baseline")
        elif ratio:
            r = str(ratio).replace(".", "")
            if r.startswith("0"):
                r = r  # e.g., "01", "05"
            cond_7.add(f"{cond}_r{r}")

    expected = 7 * 3 * 2 * 3  # 126 cells
    actual_cells = len(df.groupby(["condition", "ratio", "mode", "distance", "level"]))
    print(f"\n  Factorial cells: {actual_cells} / {expected} expected")
    print(f"  Total records: {len(df)} / {expected * 12} expected (12 seeds)")

    # Coverage matrix per condition
    print("\n  Per-condition records:")
    for cond in conditions:
        n = len(df[df["condition"] == cond])
        ratios = sorted(df[df["condition"] == cond]["ratio"].unique())
        print(f"    {cond:20s}: {n:4d} rows  (ratios: {ratios})")

    print("=" * 60)


# --- Main -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Collect XGBoost split2 evaluation metrics."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Only count matches, don't write CSVs")
    args = parser.parse_args()

    print("=" * 60)
    print("Collecting XGBoost Split2 Metrics")
    print("=" * 60)

    df = collect_all_split2_xgboost()
    if df.empty:
        print("\nNo data found. Check that XGBoost experiments have completed.")
        return

    print_summary(df)

    if args.dry_run:
        print("\n[DRY-RUN] Skipping CSV write.")
        return

    paths = save_csvs(df)
    print(f"\nSaved {len(paths)} CSV files to {CSV_BASE}")


if __name__ == "__main__":
    main()
