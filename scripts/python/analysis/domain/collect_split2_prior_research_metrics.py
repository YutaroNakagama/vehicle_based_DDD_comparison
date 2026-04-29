#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
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

# Canonical seeds for Exp3 prior-research replication (per design doc).
# Older exploratory runs may have left eval JSONs for other seeds (0, 1, 3, ...)
# that contain only partial coverage; they would render as mostly-empty plots.
# Filter to canonical seeds by default.
CANONICAL_SEEDS = (42, 123)

# --- Paths ------------------------------------------------------------
EVAL_BASE = Path(cfg.RESULTS_EVALUATION_PATH)
FIG_BASE = Path(cfg.RESULTS_PRIOR_RESEARCH_ANALYSIS_PATH) / "figures"
CSV_BASE = FIG_BASE / "csv" / "split2"
PNG_BASE = FIG_BASE / "png" / "split2"

# Condition subdirectory naming
# Maps internal tag names (from eval filenames) → user-facing directory names
# Aligned with Exp2 (collect_split2_rf_metrics.py) conventions
CONDITION_DIR_MAP = {
    "baseline":        "baseline",
    "imbalv3":         "sw_smote",              # aligned with Exp2 sw_smote
    "smote_plain":     "smote_plain",
    "undersample_rus": "undersample_rus",        # aligned with Exp2
}

# Maps internal tag names → filename-safe short labels (used in PNG/CSV names)
# Aligned with Exp2: baseline→baseline, sw_smote→sw_smote, smote_plain→smote, rus→rus
CONDITION_FILE_LABEL = {
    "baseline":        "baseline",
    "imbalv3":         "sw_smote",
    "smote_plain":     "smote",
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
# Pattern 1 – Baseline (legacy: source_only/target_only/mixed)
#   eval_results_Lstm_source_only_baseline_knn_dtw_in_domain_split2_s42.json
BASELINE_PATTERN = re.compile(
    r"eval_results_(?P<model>SvmW|SvmA|Lstm)_"
    r"(?P<mode>source_only|target_only|mixed)_"
    r"(?:prior_(?:SvmW|SvmA|Lstm)_)?"
    r"baseline_knn_"
    r"(?P<distance>mmd|dtw|wasserstein)_"
    r"(?P<domain>in_domain|out_domain)"
    r"(?:_(?:source_only|target_only|mixed))?"
    r"(?:_split2)?"
    r"_s(?P<seed>\d+)"
    r"\.json$"
)

# Pattern 1b – Baseline (domain_train mode)
#   eval_results_SvmW_domain_train_prior_SvmW_baseline_knn_wasserstein_in_domain_domain_train_split2_s123_within.json
BASELINE_DT_PATTERN = re.compile(
    r"eval_results_(?P<model>SvmW|SvmA|Lstm)_"
    r"domain_train_"
    r"prior_(?:SvmW|SvmA|Lstm)_"
    r"baseline_knn_"
    r"(?P<distance>mmd|dtw|wasserstein)_"
    r"(?P<domain>in_domain|out_domain)"
    r"(?:_domain_train)?"
    r"(?:_split2)?"
    r"_s(?P<seed>\d+)"
    r"_(?P<eval_type>within|cross)"
    r"\.json$"
)

# Pattern 2 – Prior research (legacy: source_only/target_only/mixed)
#   eval_results_Lstm_source_only_prior_Lstm_imbalv3_knn_mmd_in_domain_source_only_split2_subjectwise_ratio0.5_s42.json
PRIOR_PATTERN = re.compile(
    r"eval_results_(?P<model>SvmW|SvmA|Lstm)_"
    r"(?P<mode>source_only|target_only|mixed)_"
    r"prior_(?:SvmW|SvmA|Lstm)_"
    r"(?P<condition>imbalv3|smote_plain|undersample_rus)_"
    r"knn_"
    r"(?P<distance>mmd|dtw|wasserstein)_"
    r"(?P<domain>in_domain|out_domain)"
    r"(?:_(?:source_only|target_only|mixed))?"  # optional redundant mode suffix
    r"(?:_split2)?"
    r"(?:_subjectwise)?"                       # imbalv3 has subjectwise tag
    r"_ratio(?P<ratio>[0-9.]+)"
    r"_s(?P<seed>\d+)"
    r"\.json$"
)

# Pattern 2b – Prior research (domain_train mode)
#   eval_results_SvmW_domain_train_prior_SvmW_undersample_rus_knn_wasserstein_out_domain_domain_train_split2_ratio0.5_s42_within.json
#   eval_results_SvmW_domain_train_prior_SvmW_imbalv3_knn_mmd_in_domain_domain_train_split2_subjectwise_ratio0.5_s42_within.json
PRIOR_DT_PATTERN = re.compile(
    r"eval_results_(?P<model>SvmW|SvmA|Lstm)_"
    r"domain_train_"
    r"prior_(?:SvmW|SvmA|Lstm)_"
    r"(?P<condition>imbalv3|smote_plain|undersample_rus)_"
    r"knn_"
    r"(?P<distance>mmd|dtw|wasserstein)_"
    r"(?P<domain>in_domain|out_domain)"
    r"(?:_domain_train)?"
    r"(?:_split2)?"
    r"(?:_subjectwise)?"                       # imbalv3 has subjectwise tag
    r"_ratio(?P<ratio>[0-9.]+)"
    r"_s(?P<seed>\d+)"
    r"_(?P<eval_type>within|cross)"
    r"\.json$"
)


# Pattern 3 – Pooled (all subjects, no domain split)
# --- Legacy baseline-only (no condition in tag) ---
#   eval_results_Lstm_pooled_prior_research_s42.json
#   eval_results_SvmA_pooled_prior_research_s123.json
POOLED_LEGACY_PATTERN = re.compile(
    r"eval_results_(?P<model>SvmW|SvmA|Lstm)_"
    r"pooled_prior_research_"
    r"s(?P<seed>\d+)"
    r"\.json$"
)

# --- Condition-aware pooled patterns (new) ---
# Baseline:
#   eval_results_SvmW_pooled_prior_SvmW_baseline_s42.json
POOLED_BASELINE_PATTERN = re.compile(
    r"eval_results_(?P<model>SvmW|SvmA|Lstm)_"
    r"pooled_prior_(?:SvmW|SvmA|Lstm)_"
    r"baseline_"
    r"s(?P<seed>\d+)"
    r"\.json$"
)

# Imbalv3 (subject-wise SMOTE):
#   eval_results_SvmW_pooled_prior_SvmW_imbalv3_subjectwise_ratio0.5_s42.json
POOLED_IMBALV3_PATTERN = re.compile(
    r"eval_results_(?P<model>SvmW|SvmA|Lstm)_"
    r"pooled_prior_(?:SvmW|SvmA|Lstm)_"
    r"(?P<condition>imbalv3)_"
    r"(?:subjectwise_)?"
    r"ratio(?P<ratio>[0-9.]+)_"
    r"s(?P<seed>\d+)"
    r"\.json$"
)

# SMOTE plain / undersample_rus:
#   eval_results_SvmW_pooled_prior_SvmW_smote_plain_ratio0.5_s42.json
#   eval_results_SvmW_pooled_prior_SvmW_undersample_rus_ratio0.5_s42.json
POOLED_CONDITION_PATTERN = re.compile(
    r"eval_results_(?P<model>SvmW|SvmA|Lstm)_"
    r"pooled_prior_(?:SvmW|SvmA|Lstm)_"
    r"(?P<condition>smote_plain|undersample_rus)_"
    r"ratio(?P<ratio>[0-9.]+)_"
    r"s(?P<seed>\d+)"
    r"\.json$"
)


# Mapping: domain_train eval_type → legacy mode name for plot compatibility
_EVAL_TYPE_TO_MODE = {
    "within": "target_only",   # within-domain eval → "Within-domain" bar
    "cross":  "source_only",   # cross-domain eval  → "Cross-domain" bar
}


def parse_eval_filename(name: str) -> dict | None:
    """Return parsed metadata dict or None."""
    # Try domain_train baseline first (new format)
    m = BASELINE_DT_PATTERN.match(name)
    if m:
        d = m.groupdict()
        d["condition"] = "baseline"
        d["ratio"] = ""
        d["mode"] = _EVAL_TYPE_TO_MODE.get(d.pop("eval_type", "within"), "target_only")
        return d
    # Try domain_train prior-research pattern (new format)
    m = PRIOR_DT_PATTERN.match(name)
    if m:
        d = m.groupdict()
        d["mode"] = _EVAL_TYPE_TO_MODE.get(d.pop("eval_type", "within"), "target_only")
        return d
    # Try legacy baseline
    m = BASELINE_PATTERN.match(name)
    if m:
        d = m.groupdict()
        d["condition"] = "baseline"
        d["ratio"] = ""
        return d
    # Try legacy prior-research pattern
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
    keep_mixed: bool = False,
) -> pd.DataFrame:
    """Scan evaluation JSONs and return a tidy DataFrame."""
    records = []
    # Track which models have domain_train-format eval files.
    # Only those models should have their legacy 'mixed' rows dropped,
    # because domain_train does not produce a 'mixed' evaluation.
    models_with_domain_train: set[str] = set()
    models_to_scan = [model_filter] if model_filter else PRIOR_MODELS

    for model_name in models_to_scan:
        eval_dir = EVAL_BASE / model_name
        if not eval_dir.exists():
            logger.warning(f"Eval directory not found: {eval_dir}")
            continue

        for json_path in sorted(eval_dir.rglob("eval_results_*.json")):
            # Skip invalidated results
            if "_invalidated" in str(json_path):
                continue
            meta = parse_eval_filename(json_path.name)
            if meta is None:
                continue
            if meta["model"] != model_name:
                continue
            if condition_filter and meta["condition"] != condition_filter:
                continue

            # Detect domain_train-format files (they had eval_type key
            # before it was popped by parse_eval_filename, and their
            # filenames contain '_within.json' or '_cross.json').
            if json_path.name.endswith(("_within.json", "_cross.json")):
                models_with_domain_train.add(model_name)

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
        # Deduplicate: keep latest job per unique condition tuple.
        # 'mixed' (Multi-domain) is a distinct evaluation mode from
        # source_only (Cross-domain) and target_only (Within-domain),
        # so it is always preserved — even for models that also have
        # domain_train data (which maps to source_only / target_only).
        key_cols = ["model", "mode", "condition", "distance", "level", "ratio", "seed"]
        df = df.sort_values("job_id").drop_duplicates(subset=key_cols, keep="last")

        # For models with domain_train data (canonical Exp3 setup),
        # the legacy 'mixed' mode is orphaned: domain_train only produces
        # source_only (cross) and target_only (within) evaluations, and
        # the surviving 'mixed' rows are from old failed runs that render
        # as degenerate base-rate predictors. Drop them by default.
        if not keep_mixed and models_with_domain_train:
            before = len(df)
            mask_drop = (
                df["model"].isin(models_with_domain_train) & (df["mode"] == "mixed")
            )
            n_drop = int(mask_drop.sum())
            if n_drop:
                df = df.loc[~mask_drop].reset_index(drop=True)
                logger.info(
                    f"Dropped {n_drop} legacy 'mixed'-mode rows for models with "
                    f"domain_train data: {sorted(models_with_domain_train)} "
                    f"({before}→{len(df)} rows). Use --keep-mixed to retain."
                )

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

    Supports both legacy (baseline-only) and condition-aware pooled files.
    Each condition produces its own pooled reference for Row 4 (Pooled)
    and dashed horizontal reference lines in Rows 1-3 of the bar-chart grid.
    """
    records = []
    models_to_scan = [model_filter] if model_filter else PRIOR_MODELS

    # Ordered list of (pattern, condition_source) tuples.
    # condition_source: "fixed_baseline" means always baseline,
    # "from_match" means extract from regex named group.
    _pooled_patterns = [
        (POOLED_BASELINE_PATTERN, "fixed_baseline"),
        (POOLED_IMBALV3_PATTERN, "from_match"),
        (POOLED_CONDITION_PATTERN, "from_match"),
        (POOLED_LEGACY_PATTERN, "fixed_baseline"),  # legacy fallback
    ]

    for model_name in models_to_scan:
        eval_dir = EVAL_BASE / model_name
        if not eval_dir.exists():
            continue

        for json_path in sorted(eval_dir.rglob("eval_results_*pooled*prior*.json")):
            # Skip invalidated results
            if "_invalidated" in str(json_path):
                continue
            # Try each pattern in order; first match wins
            matched = False
            for pat, cond_source in _pooled_patterns:
                m = pat.match(json_path.name)
                if m is None:
                    continue
                if m.group("model") != model_name:
                    break  # wrong model, skip file

                # Determine condition
                if cond_source == "fixed_baseline":
                    condition = "baseline"
                else:
                    condition = m.groupdict().get("condition", "baseline")

                job_id = json_path.parent.parent.name
                metrics = load_eval_json(json_path)
                row = {
                    "job_id":    job_id,
                    "model":     m.group("model"),
                    "mode":      "pooled",
                    "condition": condition,
                    "distance":  "pooled",      # placeholder
                    "level":     "pooled",      # placeholder
                    "ratio":     m.groupdict().get("ratio", ""),
                    "seed":      int(m.group("seed")),
                }
                row.update(metrics)
                records.append(row)
                matched = True
                break  # matched, no need to try other patterns

    df = pd.DataFrame(records)
    if not df.empty:
        # Keep latest job per (model, condition, seed)
        df = df.sort_values("job_id").drop_duplicates(
            subset=["model", "condition", "seed"], keep="last"
        )
        logger.info(
            f"Collected {len(df)} pooled records: "
            f"models={sorted(df['model'].unique())}, "
            f"conditions={sorted(df['condition'].unique())}, "
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
def generate_plots(
    df: pd.DataFrame,
    df_pooled: pd.DataFrame,
    min_coverage: float = 0.5,
) -> list[Path]:
    """Generate summary bar-chart PNGs per model × condition × ratio × seed.

    For baseline (no ratio), one plot per seed.
    For other conditions, one plot per (ratio, seed).

    Pooled data (mode='pooled') is merged into each plot so that
    ``plot_grouped_bar_chart_raw`` can draw the 4th row (Pooled baseline)
    and dashed horizontal reference lines in Rows 1-3.

    Args:
        min_coverage: minimum fraction of expected cells
            (3 distances × 2 domains × 2 modes = 12) that must be present;
            plots below this threshold are skipped with a warning.
    """
    EXPECTED_CELLS = 3 * 2 * 2  # distance × domain × (source_only/target_only)
    generated = []
    skipped = []

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

            # --- Coverage check: skip plots that lack enough data ---
            n_cells = len(sub)
            coverage = n_cells / EXPECTED_CELLS
            if coverage < min_coverage:
                skipped.append(
                    (model, cond, ratio_val, seed, n_cells, EXPECTED_CELLS)
                )
                logger.warning(
                    f"  SKIP plot (coverage {coverage:.0%} = {n_cells}/{EXPECTED_CELLS} < "
                    f"{min_coverage:.0%}): {model}/{cond} ratio={ratio_val} seed={seed}"
                )
                continue

            # Build filename – Exp2-aligned short style:
            #   {cond_short}{ratio_tag}_s{seed}.png
            cond_short = CONDITION_FILE_LABEL.get(cond, cond)
            out_name = f"{cond_short}{ratio_tag}_s{seed}.png"
            out_path = png_dir / out_name

            # --- Merge pooled reference rows for this model + condition + seed ----
            sub_with_pooled = sub
            if not df_pooled.empty:
                pooled_rows = df_pooled[
                    (df_pooled["model"] == model)
                    & (df_pooled["condition"] == cond)
                    & (df_pooled["seed"] == seed)
                ]
                if not pooled_rows.empty:
                    pooled_rows = pooled_rows.copy()
                    common_cols = sub.columns.intersection(pooled_rows.columns)
                    sub_with_pooled = pd.concat(
                        [sub, pooled_rows[common_cols]], ignore_index=True
                    )

            # plot_grouped_bar_chart_raw expects modes (include pooled)
            # Determine active comparison modes from the data
            data_modes = sorted(sub_with_pooled["mode"].unique())
            comparison_modes = [m for m in data_modes if m != "pooled"]
            all_modes = ["pooled"] + comparison_modes
            fig = plot_grouped_bar_chart_raw(
                data=sub_with_pooled,
                metrics=METRICS,
                modes=all_modes,
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

    if skipped:
        logger.warning(
            f"\n  Skipped {len(skipped)} plot(s) due to insufficient coverage "
            f"(< {min_coverage:.0%}). Re-run after pending jobs complete."
        )
        for model, cond, ratio_val, seed, got, exp in skipped:
            logger.warning(
                f"    - {model}/{cond}  ratio={ratio_val}  seed={seed}  : {got}/{exp} cells"
            )

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
    parser.add_argument(
        "--seeds", default=",".join(str(s) for s in CANONICAL_SEEDS),
        help=(
            "Comma-separated list of seeds to keep "
            f"(default: canonical {CANONICAL_SEEDS}). "
            "Use 'all' to disable filtering and include every seed found on disk."
        ),
    )
    parser.add_argument(
        "--min-coverage", type=float, default=0.5,
        help=(
            "Minimum fraction of expected cells (12 = 3 distances × 2 domains × "
            "2 modes) required to render a plot. Default 0.5. Use 0.0 to render all."
        ),
    )
    parser.add_argument(
        "--keep-mixed", action="store_true",
        help=(
            "Keep legacy 'mixed' (Multi-domain) mode rows even for models with "
            "domain_train data. Default: drop them (they are orphaned from "
            "older failed runs and render as degenerate base-rate bars)."
        ),
    )
    args = parser.parse_args()

    if args.seeds.strip().lower() == "all":
        seed_filter = None
    else:
        seed_filter = {int(s) for s in args.seeds.split(",") if s.strip()}

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
        keep_mixed=args.keep_mixed,
    )
    if df.empty:
        logger.error("No data found. Exiting.")
        return 1

    # Apply seed filter (default: canonical seeds only)
    if seed_filter is not None:
        before = len(df)
        df = df[df["seed"].astype(int).isin(seed_filter)].reset_index(drop=True)
        logger.info(
            f"Seed filter {sorted(seed_filter)}: kept {len(df)}/{before} rows "
            f"(use --seeds all to disable)"
        )

    # Step 1b: Collect pooled (all-subjects) reference data
    df_pooled = collect_pooled_data(model_filter=args.model)
    if seed_filter is not None and not df_pooled.empty and "seed" in df_pooled.columns:
        df_pooled = df_pooled[df_pooled["seed"].astype(int).isin(seed_filter)].reset_index(drop=True)

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
    plots = generate_plots(df, df_pooled, min_coverage=args.min_coverage)

    logger.info("\n" + "=" * 70)
    logger.info(f"DONE — {len(df)} records, {len(plots)} plots generated")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
