#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
regenerate_all_conditions_may2026.py
====================================
Regenerate per-condition CSV + summary plots using ONLY May-2026 manual_*
evaluation JSONs (post 90→145 RF feature pool + post split.py model_name fix).

Conditions covered:
  - baseline_domain
  - smote_plain         (ratios 0.1, 0.5)
  - undersample_rus     (ratios 0.1, 0.5)
  - sw_smote (imbalv3)  (ratios 0.1, 0.5)

Outputs:
  CSV: results/analysis/exp2_domain_shift/figures/csv/split2/<cond>_NEW_may2026/...csv
  PNG: results/analysis/exp2_domain_shift/figures/png/split2/<cond>_NEW_may2026/*.png
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts/python/analysis/domain"))

from src import config as cfg

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

EVAL_DIR = Path(cfg.RESULTS_EVALUATION_PATH) / "RF"
FIG_BASE = Path(cfg.RESULTS_DOMAIN_ANALYSIS_PATH) / "figures"

# (condition_key, regex, has_ratio, csv_dir_name, csv_filename, png_dir_name, plotter_csv_dir, plotter_csv_file, plotter_png_dir)
SPECS = [
    {
        "key": "baseline_domain",
        "pattern": re.compile(
            r"eval_results_RF_(?P<mode>source_only|target_only|mixed)_"
            r"baseline_domain_knn_"
            r"(?P<distance>mmd|dtw|wasserstein)_"
            r"(?P<domain>in_domain|out_domain)_"
            r"(?:source_only|target_only|mixed)_split2"
            r"_s(?P<seed>\d+)\.json$"
        ),
        "has_ratio": False,
        "csv_subdir": "baseline_NEW_may2026",
        "csv_name": "baseline_domain_split2_metrics_NEW_may2026.csv",
        "png_subdir": "baseline_NEW_may2026",
        # spec for plot_condition_seed_summaries
        "spec": {
            "condition": "baseline_domain",
            "csv_dir": "baseline_NEW_may2026",
            "csv_file": "baseline_domain_split2_metrics_NEW_may2026.csv",
            "png_dir": "baseline_NEW_may2026",
            "ratios": [None],
            "out_prefix": "baseline_summary",
            "title_prefix": "Baseline (May 2026)",
            "has_ratio_col": False,
            "pooled_condition": "baseline_domain",
        },
    },
    {
        "key": "smote_plain",
        "pattern": re.compile(
            r"eval_results_RF_(?P<mode>source_only|target_only|mixed)_"
            r"smote_plain_knn_"
            r"(?P<distance>mmd|dtw|wasserstein)_"
            r"(?P<domain>in_domain|out_domain)_"
            r"(?:source_only|target_only|mixed)_split2"
            r"_ratio(?P<ratio>[0-9.]+)_s(?P<seed>\d+)\.json$"
        ),
        "has_ratio": True,
        "csv_subdir": "smote_plain_NEW_may2026",
        "csv_name": "smote_plain_split2_metrics_NEW_may2026.csv",
        "png_subdir": "smote_plain_NEW_may2026",
        "spec": {
            "condition": "smote_plain",
            "csv_dir": "smote_plain_NEW_may2026",
            "csv_file": "smote_plain_split2_metrics_NEW_may2026.csv",
            "png_dir": "smote_plain_NEW_may2026",
            "ratios": ["0.1", "0.5"],
            "out_prefix": "smote_summary",
            "title_prefix": "SMOTE (Plain, May 2026)",
            "has_ratio_col": True,
            "pooled_condition": "smote_plain",
        },
    },
    {
        "key": "undersample_rus",
        "pattern": re.compile(
            r"eval_results_RF_(?P<mode>source_only|target_only|mixed)_"
            r"undersample_rus_knn_"
            r"(?P<distance>mmd|dtw|wasserstein)_"
            r"(?P<domain>in_domain|out_domain)_"
            r"(?:source_only|target_only|mixed)_split2"
            r"_ratio(?P<ratio>[0-9.]+)_s(?P<seed>\d+)\.json$"
        ),
        "has_ratio": True,
        "csv_subdir": "undersample_rus_NEW_may2026",
        "csv_name": "undersample_rus_split2_metrics_NEW_may2026.csv",
        "png_subdir": "undersample_rus_NEW_may2026",
        "spec": {
            "condition": "undersample_rus",
            "csv_dir": "undersample_rus_NEW_may2026",
            "csv_file": "undersample_rus_split2_metrics_NEW_may2026.csv",
            "png_dir": "undersample_rus_NEW_may2026",
            "ratios": ["0.1", "0.5"],
            "out_prefix": "rus_summary",
            "title_prefix": "Random Under-Sampling (May 2026)",
            "has_ratio_col": True,
            "pooled_condition": "undersample_rus",
        },
    },
    {
        "key": "sw_smote",
        "pattern": re.compile(
            r"eval_results_RF_(?P<mode>source_only|target_only|mixed)_"
            r"imbalv3_knn_"
            r"(?P<distance>mmd|dtw|wasserstein)_"
            r"(?P<domain>in_domain|out_domain)_"
            r"(?:source_only|target_only|mixed)_split2_subjectwise"
            r"_ratio(?P<ratio>[0-9.]+)_s(?P<seed>\d+)\.json$"
        ),
        "has_ratio": True,
        "csv_subdir": "sw_smote_NEW_may2026",
        "csv_name": "sw_smote_split2_metrics_NEW_may2026.csv",
        "png_subdir": "sw_smote_NEW_may2026",
        "spec": {
            "condition": "sw_smote",
            "csv_dir": "sw_smote_NEW_may2026",
            "csv_file": "sw_smote_split2_metrics_NEW_may2026.csv",
            "png_dir": "sw_smote_NEW_may2026",
            "ratios": ["0.1", "0.5"],
            "out_prefix": "sw_smote_summary",
            "title_prefix": "Subject-Wise SMOTE (May 2026)",
            "has_ratio_col": True,
            "pooled_condition": "sw_smote",
        },
    },
]


def _collect(spec) -> pd.DataFrame:
    pat = spec["pattern"]
    has_ratio = spec["has_ratio"]
    rows = []
    for root, _, files in os.walk(EVAL_DIR):
        if "manual_202605" not in root:
            continue
        for f in files:
            m = pat.match(f)
            if not m:
                continue
            p = os.path.join(root, f)
            try:
                d = json.load(open(p))
            except Exception:
                continue
            md = m.groupdict()
            prec = d.get("precision", 0.0)
            rec = d.get("recall", 0.0)
            f1 = d.get("f1", 0.0)
            f2 = (5 * prec * rec) / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0
            rows.append({
                "job_id": Path(root).parent.name,
                "mode": md["mode"],
                "distance": md["distance"],
                "level": md["domain"],
                "seed": int(md["seed"]),
                "ratio": md.get("ratio", "") if has_ratio else "",
                "accuracy": d.get("accuracy", 0.0),
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "f2": f2,
                "auc": d.get("roc_auc", np.nan),
                "auc_pr": d.get("auc_pr", np.nan),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    keys = ["mode", "distance", "level", "seed"] + (["ratio"] if has_ratio else [])
    df = df.sort_values("job_id").groupby(keys).last().reset_index()
    return df


def main() -> int:
    csv_base = FIG_BASE / "csv" / "split2"
    png_base = FIG_BASE / "png" / "split2"

    all_specs_for_plot = []

    for s in SPECS:
        log.info(f"==== {s['key']} ====")
        df = _collect(s)
        if df.empty:
            log.warning(f"  No records found.")
            continue
        log.info(f"  records={len(df)} seeds={sorted(df['seed'].unique())} "
                 f"distances={sorted(df['distance'].unique())} "
                 f"modes={sorted(df['mode'].unique())} "
                 f"domains={sorted(df['level'].unique())}")
        out_dir = csv_base / s["csv_subdir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        cols = ["job_id", "mode", "distance", "level", "seed", "ratio",
                "accuracy", "precision", "recall", "f1", "f2", "auc", "auc_pr"]
        out_csv = out_dir / s["csv_name"]
        df[cols].to_csv(out_csv, index=False)
        log.info(f"  CSV → {out_csv}")
        g = df.groupby(["mode", "level"])["auc"].agg(["mean", "std", "count"])
        log.info(f"  AUROC by mode×domain:\n{g.to_string()}")
        all_specs_for_plot.append(s["spec"])

    # Generate plots by monkey-patching plot_condition_seed_summaries
    log.info("==== Generating plots ====")
    import plot_condition_seed_summaries as pcss
    pcss.CSV_BASE = csv_base
    pcss.PNG_BASE = png_base
    pcss.CONDITION_SPECS = all_specs_for_plot
    rc = pcss.main()
    if rc:
        log.error("Plot generation reported failure")
        return rc
    log.info("ALL DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
