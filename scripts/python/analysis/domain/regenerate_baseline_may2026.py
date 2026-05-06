#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
regenerate_baseline_may2026.py
==============================
Regenerate baseline_domain split2 CSV and seed-summary plots using ONLY the
May-2026 manual_* evaluation JSONs (the post 90→145 RF feature pool runs
for exp2 round 2).

Outputs:
  CSV: results/analysis/exp2_domain_shift/figures/csv/split2/baseline/
       baseline_domain_split2_metrics_NEW_may2026.csv
  PNG: results/analysis/exp2_domain_shift/figures/png/split2/baseline_NEW_may2026/
       baseline_seed_summary_bar.png
       baseline_seed_summary_boxplot.png
       baseline_seed_summary_heatmap.png
       baseline_seed_summary_table.png

Usage:
    python scripts/python/analysis/domain/regenerate_baseline_may2026.py
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
CSV_DIR = FIG_BASE / "csv" / "split2" / "baseline"
PNG_DIR = FIG_BASE / "png" / "split2" / "baseline_NEW_may2026"
CSV_NAME = "baseline_domain_split2_metrics_NEW_may2026.csv"

# baseline_domain naming pattern (no ratio)
PAT = re.compile(
    r"eval_results_RF_"
    r"(?P<mode>source_only|target_only|mixed)_"
    r"baseline_domain_knn_"
    r"(?P<distance>mmd|dtw|wasserstein)_"
    r"(?P<domain>in_domain|out_domain)_"
    r"(?:source_only|target_only|mixed)"
    r"(?:_split2)"
    r"_s(?P<seed>\d+)"
    r"\.json$"
)


def _collect_may() -> pd.DataFrame:
    """Walk EVAL_DIR; keep baseline_domain split2 records produced under
    manual_202605* batch directories only."""
    rows = []
    for root, dirs, files in os.walk(EVAL_DIR):
        # filter: must be under manual_202605*
        if "manual_202605" not in root:
            continue
        for f in files:
            m = PAT.match(f)
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
                "ratio": "",
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
    # de-dup: keep latest job_id per combination
    keys = ["mode", "distance", "level", "seed"]
    df = df.sort_values("job_id").groupby(keys).last().reset_index()
    return df


def main() -> int:
    log.info("Scanning May-2026 manual_* evaluation JSONs (baseline_domain only)…")
    df = _collect_may()
    if df.empty:
        log.error("No baseline_domain May-2026 records found.")
        return 1
    log.info(f"Collected {len(df)} records  "
             f"(seeds={sorted(df['seed'].unique())}, "
             f"distances={sorted(df['distance'].unique())}, "
             f"modes={sorted(df['mode'].unique())}, "
             f"domains={sorted(df['level'].unique())})")

    CSV_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = CSV_DIR / CSV_NAME
    cols = ["job_id", "mode", "distance", "level", "seed", "ratio",
            "accuracy", "precision", "recall", "f1", "f2", "auc", "auc_pr"]
    df[cols].to_csv(out_csv, index=False)
    log.info(f"CSV written: {out_csv}")

    # Quick sanity print
    g = df.groupby(["mode", "level"])["auc"].agg(["mean", "std", "count"])
    log.info("AUROC by mode×domain:\n" + g.to_string())

    # ------------------------------------------------------------------
    # Re-use plot_baseline_seed_summary by monkey-patching its constants
    # ------------------------------------------------------------------
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    import plot_baseline_seed_summary as pbss
    pbss.CSV_PATH = out_csv
    pbss.OUT_DIR = PNG_DIR
    log.info(f"Generating plots into {PNG_DIR}")
    rc = pbss.main()
    if rc:
        log.error("Plot generation reported failure.")
        return rc
    log.info("DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
