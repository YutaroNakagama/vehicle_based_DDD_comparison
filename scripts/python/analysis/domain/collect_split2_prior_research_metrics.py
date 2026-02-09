#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_split2_prior_research_metrics.py
=========================================
Collect evaluation metrics from split2 prior-research experiments
(SvmW, SvmA, Lstm) for **all conditions** (baseline, imbalv3 = subject-wise
SMOTE, smote_plain, undersample_rus) and generate summary_metrics_bar plots
in the SAME layout as the Exp2 (BalancedRF) pipeline.

Output structure (mirrors Exp2):
  results/analysis/exp3_prior_research/figures/
  ├── csv/split2/{Model}/{condition}/  ← per-condition CSVs
  └── png/split2/{Model}/{condition}/  ← per-condition × ratio × seed PNGs

Pipeline:
  1. Scan SvmW / SvmA / Lstm evaluation JSONs with *split2* tag
  2. Parse model / condition / mode / distance / domain / ratio / seed
  3. Build per-model × per-condition CSV
  4. Generate per-model × per-condition × per-ratio × per-seed bar plots

Usage:
    python scripts/python/analysis/domain/collect_split2_prior_research_metrics.py
    python scripts/python/analysis/domain/collect_split2_prior_research_metrics.py --model Lstm
    python scripts/python/analysis/domain/collect_split2_prior_research_metrics.py --model SvmA --condition baseline
    python scripts/python/analysis/domain/collect_split2_prior_research_metrics.py --dry-run
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

# --- Constants --------------------------------------------------------
PRIOR_MODELS = ["SvmW", "SvmA", "Lstm"]
METRICS = ["accuracy", "precision", "recall", "f1", "f2", "auc", "auc_pr"]

# --- Paths ------------------------------------------------------------
EVAL_BASE = Path(cfg.RESULTS_EVALUATION_PATH)
FIG_BASE = Path(cfg.RESULTS_PRIOR_RESEARCH_ANALYSIS_PATH) / "figures"
CSV_BASE = FIG_BASE / "csv" / "split2"
PNG_BASE = FIG_BASE / "png" / "split2"

# Condition subdirectory naming
# Maps internal tag names (from eval filenames) → user-facing directory names
CONDITION_DIR_MAP = {
    "baseline":        "baseline",
    "imbalv3":         "subject_wise_smote",   # internal tag → readable name
    "smote_plain":     "smote_plain",
    "undersample_rus": "rus",                   # internal tag → readable name
}

# Maps internal tag names → filename-safe short labels (used in PNG/CSV names)
CONDITION_FILE_LABEL = {
    "baseline":        "baseline",
    "imbalv3":         "subject_wise_smote",
    "smote_plain":     "smote_plain",
    "undersample_rus": "rus",
}

# Human-friendly condition labels for plot titles
CONDITION_LABELS = {
    "baseline":        "Baseline",
    "imbalv3":         "Subject-wise SMOTE",
    "smote_plain":     "SMOTE (plain)",
    "undersample_rus": "Random Under-Sampling (RUS)",
}

# =====================================================================
# Filename parsing
# =====================================================================
# Pattern 1 – Baseline
#   eval_results_Lstm_source_only_baseline_knn_dtw_in_domain_split2_s42.json
BASELINE_PATTERN = re.compile(
    r"eval_results_(?P<model>SvmW|SvmA|Lstm)_"
    r"(?P<mode>source_only|target_only)_"
    r"baseline_knn_"
    r"(?P<distance>mmd|dtw|wasserstein)_"
    r"(?P<domain>in_domain|out_domain)"
    r"(?:_split2)?"
    r"_s(?P<seed>\d+)"
    r"\.json$"
)

# Pattern 2 – Prior research (imbalv3 / smote_plain / undersample_rus)
#   eval_results_Lstm_source_only_prior_Lstm_imbalv3_knn_mmd_in_domain_source_only_split2_subjectwise_ratio0.5_s42.json
#   eval_results_Lstm_source_only_prior_Lstm_smote_plain_knn_mmd_out_domain_source_only_split2_ratio0.5_s42.json
#   eval_results_Lstm_target_only_prior_Lstm_undersample_rus_knn_mmd_out_domain_target_only_split2_ratio0.5_s42.json
PRIOR_PATTERN = re.compile(
    r"eval_results_(?P<model>SvmW|SvmA|Lstm)_"
    r"(?P<mode>source_only|target_only)_"
    r"prior_(?:SvmW|SvmA|Lstm)_"
    r"(?P<condition>imbalv3|smote_plain|undersample_rus)_"
    r"knn_"
    r"(?P<distance>mmd|dtw|wasserstein)_"
    r"(?P<domain>in_domain|out_domain)"
    r"(?:_(?:source_only|target_only))?"       # optional redundant mode suffix
    r"(?:_split2)?"
    r"(?:_subjectwise)?"                       # imbalv3 has subjectwise tag
    r"_ratio(?P<ratio>[0-9.]+)"
    r"_s(?P<seed>\d+)"
    r"\.json$"
)


# Pattern 3 – Pooled (all subjects, no domain split)
#   eval_results_Lstm_pooled_prior_research_s42.json
#   eval_results_SvmA_pooled_prior_research_s123.json
POOLED_PATTERN = re.compile(
    r"eval_results_(?P<model>SvmW|SvmA|Lstm)_"
    r"pooled_prior_research_"
    r"s(?P<seed>\d+)"
    r"\.json$"
)


def parse_eval_filename(name: str) -> dict | None:
    """Return parsed metadata dict or None."""
    # Try baseline first
    m = BASELINE_PATTERN.match(name)
    if m:
        d = m.groupdict()
        d["condition"] = "baseline"
        d["ratio"] = ""
        return d
    # Try prior-research pattern
    m = PRIOR_PATTERN.match(name)
    if m:
        return m.groupdict()
    return None


# =====================================================================
# Metric extraction (handles heterogeneous JSON formats)
# =====================================================================
def _to_pos_scalar(val, default: float = 0.0) -> float:
    """Extract positive-class scalar from a value that might be a list.

    SvmA stores precision/recall as [neg, pos]; Lstm uses precision_pos/recall_pos.
    """
    if val is None:
        return default
    if isinstance(val, list):
        return float(val[1]) if len(val) > 1 else default
    return float(val)


def load_eval_json(path: Path) -> dict:
    """Load evaluation JSON and normalise to a common metric dict.

    Handles three metric layouts:
      - SvmA:  precision=[neg,pos], recall=[neg,pos], f1_score=[neg,pos], thr based
      - SvmW:  precision/recall/f1 as scalars or _pos keys
      - Lstm:  precision_pos/recall_pos/f1_pos + classification_report
    """
    with open(path) as f:
        d = json.load(f)

    # --- Precision / Recall / F1 (positive class) ---------------------
    prec = None
    rec = None
    f1 = None

    # Priority 1: explicit _pos keys (Lstm)
    if d.get("precision_pos") is not None:
        prec = float(d["precision_pos"])
    if d.get("recall_pos") is not None:
        rec = float(d["recall_pos"])
    if d.get("f1_pos") is not None:
        f1 = float(d["f1_pos"])

    # Priority 2: list-valued keys (SvmA)
    if prec is None and d.get("precision") is not None:
        prec = _to_pos_scalar(d["precision"])
    if rec is None and d.get("recall") is not None:
        rec = _to_pos_scalar(d["recall"])
    if f1 is None:
        f1_raw = d.get("f1_score", d.get("f1"))
        if f1_raw is not None:
            f1 = _to_pos_scalar(f1_raw)

    # Priority 3: classification_report (Lstm fallback)
    cr = d.get("classification_report", {})
    if isinstance(cr, dict):
        pos_cls = cr.get("1.0") or cr.get("1") or cr.get(1)
        if pos_cls and isinstance(pos_cls, dict):
            if prec is None:
                prec = float(pos_cls.get("precision", 0.0))
            if rec is None:
                rec = float(pos_cls.get("recall", 0.0))
            if f1 is None:
                f1 = float(pos_cls.get("f1-score", 0.0))

    # Fallback to 0
    prec = prec if prec is not None else 0.0
    rec = rec if rec is not None else 0.0
    f1 = f1 if f1 is not None else 0.0

    # F2 (beta=2, weights recall higher)
    f2 = (5 * prec * rec) / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0

    # AUC metrics (may be absent → NaN)
    roc_auc = d.get("roc_auc", d.get("auc"))
    auc_pr = d.get("auc_pr", d.get("average_precision"))

    return {
        "accuracy":  float(d.get("accuracy", 0.0)),
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
        "f2":        f2,
        "auc":       float(roc_auc) if roc_auc is not None else np.nan,
        "auc_pr":    float(auc_pr) if auc_pr is not None else np.nan,
    }


# =====================================================================
# Step 1: Collect evaluation results
# =====================================================================
def collect_all_split2(
    model_filter: str | None = None,
    condition_filter: str | None = None,
) -> pd.DataFrame:
    """Scan evaluation JSONs and return a tidy DataFrame."""
    records = []
    models_to_scan = [model_filter] if model_filter else PRIOR_MODELS

    for model_name in models_to_scan:
        eval_dir = EVAL_BASE / model_name
        if not eval_dir.exists():
            logger.warning(f"Eval directory not found: {eval_dir}")
            continue

        for json_path in sorted(eval_dir.rglob("eval_results_*.json")):
            meta = parse_eval_filename(json_path.name)
            if meta is None:
                continue
            if meta["model"] != model_name:
                continue
            if condition_filter and meta["condition"] != condition_filter:
                continue

            job_id = json_path.parent.parent.name
            metrics = load_eval_json(json_path)
            row = {
                "job_id":    job_id,
                "model":     meta["model"],
                "mode":      meta["mode"],
                "condition": meta["condition"],
                "distance":  meta["distance"],
                "level":     meta["domain"],
                "ratio":     meta.get("ratio", ""),
                "seed":      int(meta["seed"]),
            }
            row.update(metrics)
            records.append(row)

    df = pd.DataFrame(records)
    if df.empty:
        logger.warning("No split2 evaluation JSONs found!")
    else:
        # Deduplicate: keep latest job per unique condition tuple
        key_cols = ["model", "mode", "condition", "distance", "level", "ratio", "seed"]
        df = df.sort_values("job_id").drop_duplicates(subset=key_cols, keep="last")
        logger.info(
            f"Collected {len(df)} records – models={sorted(df['model'].unique())}, "
            f"conditions={sorted(df['condition'].unique())}"
        )
    return df


# =====================================================================
# Step 1b: Collect pooled (all-subjects) reference data
# =====================================================================
def collect_pooled_data(
    model_filter: str | None = None,
) -> pd.DataFrame:
    """Scan pooled evaluation JSONs for prior-research models.

    These serve as the 'all subjects' baseline for Row 4 (Pooled) and the
    dashed horizontal reference lines in Rows 1-3 of the bar-chart grid.
    """
    records = []
    models_to_scan = [model_filter] if model_filter else PRIOR_MODELS

    for model_name in models_to_scan:
        eval_dir = EVAL_BASE / model_name
        if not eval_dir.exists():
            continue

        for json_path in sorted(eval_dir.rglob("eval_results_*pooled*prior_research*.json")):
            m = POOLED_PATTERN.match(json_path.name)
            if m is None:
                continue
            if m.group("model") != model_name:
                continue

            job_id = json_path.parent.parent.name
            metrics = load_eval_json(json_path)
            row = {
                "job_id":    job_id,
                "model":     m.group("model"),
                "mode":      "pooled",
                "condition":  "baseline",   # pooled is the baseline reference
                "distance":  "pooled",      # placeholder
                "level":     "pooled",      # placeholder
                "ratio":     "",
                "seed":      int(m.group("seed")),
            }
            row.update(metrics)
            records.append(row)

    df = pd.DataFrame(records)
    if not df.empty:
        # Keep latest job per (model, seed)
        df = df.sort_values("job_id").drop_duplicates(
            subset=["model", "seed"], keep="last"
        )
        logger.info(
            f"Collected {len(df)} pooled records: "
            f"models={sorted(df['model'].unique())}, "
            f"seeds={sorted(df['seed'].unique())}"
        )
    else:
        logger.warning("No pooled evaluation JSONs found.")
    return df


# =====================================================================
# Step 2: Save CSVs
# =====================================================================
def save_csvs(df: pd.DataFrame) -> dict[str, Path]:
    """Write per-model × per-condition CSVs."""
    paths = {}
    for (model, cond), sub in df.groupby(["model", "condition"]):
        cond_dir = CONDITION_DIR_MAP.get(cond, cond)
        csv_dir = CSV_BASE / model / cond_dir
        csv_dir.mkdir(parents=True, exist_ok=True)
        cond_label = CONDITION_FILE_LABEL.get(cond, cond)
        fname = f"{model.lower()}_{cond_label}_split2_metrics.csv"
        out = csv_dir / fname

        cols = ["job_id", "mode", "distance", "level", "ratio", "seed"] + METRICS
        cols = [c for c in cols if c in sub.columns]
        sub[cols].to_csv(out, index=False)
        logger.info(f"  CSV saved: {out}  ({len(sub)} rows)")
        paths[(model, cond)] = out
    return paths


# =====================================================================
# Step 3: Generate plots
# =====================================================================
def generate_plots(df: pd.DataFrame, df_pooled: pd.DataFrame) -> list[Path]:
    """Generate summary bar-chart PNGs per model × condition × ratio × seed.

    For baseline (no ratio), one plot per seed.
    For other conditions, one plot per (ratio, seed).

    Pooled data (mode='pooled') is merged into each plot so that
    ``plot_grouped_bar_chart_raw`` can draw the 4th row (Pooled baseline)
    and dashed horizontal reference lines in Rows 1-3.
    """
    generated = []

    # Group keys depend on whether ratio is present
    for (model, cond), model_cond_df in df.groupby(["model", "condition"]):
        cond_dir = CONDITION_DIR_MAP.get(cond, cond)
        png_dir = PNG_BASE / model / cond_dir
        png_dir.mkdir(parents=True, exist_ok=True)

        # Determine grouping columns
        ratios = sorted(model_cond_df["ratio"].unique())
        has_ratio = any(r != "" for r in ratios)

        if has_ratio:
            group_cols = ["ratio", "seed"]
        else:
            group_cols = ["seed"]

        for group_key, sub in model_cond_df.groupby(group_cols):
            if has_ratio:
                ratio_val, seed = group_key
                ratio_tag = f"_r{str(ratio_val).replace('.', '')}"
            else:
                seed = group_key if not isinstance(group_key, tuple) else group_key[0]
                ratio_val = ""
                ratio_tag = ""

            # Build filename (use user-facing label, not internal tag)
            cond_label = CONDITION_FILE_LABEL.get(cond, cond)
            out_name = f"summary_metrics_bar_{model.lower()}_{cond_label}{ratio_tag}_v2_s{seed}.png"
            out_path = png_dir / out_name

            # --- Merge pooled reference rows for this model + seed ----
            sub_with_pooled = sub
            if not df_pooled.empty:
                pooled_rows = df_pooled[
                    (df_pooled["model"] == model) & (df_pooled["seed"] == seed)
                ]
                if not pooled_rows.empty:
                    pooled_rows = pooled_rows.copy()
                    pooled_rows["condition"] = cond  # align condition label
                    common_cols = sub.columns.intersection(pooled_rows.columns)
                    sub_with_pooled = pd.concat(
                        [sub, pooled_rows[common_cols]], ignore_index=True
                    )

            # plot_grouped_bar_chart_raw expects modes (include pooled)
            fig = plot_grouped_bar_chart_raw(
                data=sub_with_pooled,
                metrics=METRICS,
                modes=["pooled", "source_only", "target_only"],
                distance_col="distance",
                level_col="level",
                baseline_rates={"auc_pr": 0.033},
            )
            if fig is not None:
                # Add suptitle
                label = CONDITION_LABELS.get(cond, cond)
                ratio_str = f" (ratio={ratio_val})" if ratio_val else ""
                fig.suptitle(
                    f"{model} – {label}{ratio_str} – seed={seed}",
                    fontsize=14, fontweight="bold", y=1.01,
                )
                save_figure(fig, str(out_path), dpi=200)
                plt.close(fig)
                generated.append(out_path)
                logger.info(f"  PNG saved: {out_path}")

    return generated


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Collect split2 prior-research (SvmW/SvmA/Lstm) metrics → CSV + plots"
    )
    parser.add_argument(
        "--model", default=None, choices=PRIOR_MODELS,
        help="Filter to single model (e.g., Lstm)",
    )
    parser.add_argument(
        "--condition", default=None,
        choices=list(CONDITION_DIR_MAP.keys()),
        help="Filter to single condition (e.g., baseline, imbalv3, smote_plain, undersample_rus)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Collect & print summary but do not write files",
    )
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("Collect split2 prior-research metrics (Exp3: SvmW/SvmA/Lstm)")
    logger.info("=" * 70)
    logger.info(f"Eval base  : {EVAL_BASE}")
    logger.info(f"CSV output : {CSV_BASE}/<Model>/<condition>/")
    logger.info(f"PNG output : {PNG_BASE}/<Model>/<condition>/")

    # Step 1a: Collect domain-split data
    df = collect_all_split2(
        model_filter=args.model,
        condition_filter=args.condition,
    )
    if df.empty:
        logger.error("No data found. Exiting.")
        return 1

    # Step 1b: Collect pooled (all-subjects) reference data
    df_pooled = collect_pooled_data(model_filter=args.model)

    logger.info(f"\nModels     : {sorted(df['model'].unique())}")
    logger.info(f"Conditions : {sorted(df['condition'].unique())}")
    logger.info(f"Modes      : {sorted(df['mode'].unique())}")
    logger.info(f"Distances  : {sorted(df['distance'].unique())}")
    logger.info(f"Domains    : {sorted(df['level'].unique())}")
    logger.info(f"Ratios     : {sorted(df['ratio'].unique())}")
    logger.info(f"Seeds      : {sorted(df['seed'].unique())}")
    logger.info(f"Total rows : {len(df)}  (+{len(df_pooled)} pooled)")

    # Summary table
    summary = df.groupby(["model", "condition", "ratio"]).size().reset_index(name="count")
    logger.info(f"\nBreakdown:\n{summary.to_string(index=False)}")

    if args.dry_run:
        logger.info("\n[DRY-RUN] No files written.")
        return 0

    # Step 2: Save CSVs
    logger.info("\n--- Saving CSVs ---")
    save_csvs(df)

    # Step 3: Generate plots (with pooled reference)
    logger.info("\n--- Generating plots ---")
    plots = generate_plots(df, df_pooled)

    logger.info("\n" + "=" * 70)
    logger.info(f"DONE — {len(df)} records, {len(plots)} plots generated")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
