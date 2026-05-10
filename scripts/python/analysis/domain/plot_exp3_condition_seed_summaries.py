#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_exp3_condition_seed_summaries.py
======================================
Generate seed-aggregated summary plots (mean ± std) for each Exp3
prior-research (model, condition, ratio) cell, mirroring the Exp2 script
``plot_condition_seed_summaries.py`` (4 rows × 7 metric columns).

Inputs (per-seed CSVs already produced by ``collect_split2_prior_research_metrics.py``):
    results/analysis/exp3_prior_research/figures/csv/split2/{Model}/{cond_dir}/
        {model_lower}_{cond_label}_split2_metrics.csv

Outputs:
    results/analysis/exp3_prior_research/figures/png/split2/{Model}/{cond_dir}/
        {cond_label}{ratio_tag}_summary.png

For each cell, all available seeds are aggregated (mean ± std). Pooled rows
(if any) are placed on Row 4 and used for dashed horizontal references on
Rows 1-3, exactly like Exp2.

Usage:
    python scripts/python/analysis/domain/plot_exp3_condition_seed_summaries.py
    python scripts/python/analysis/domain/plot_exp3_condition_seed_summaries.py --model Lstm
    python scripts/python/analysis/domain/plot_exp3_condition_seed_summaries.py --model SvmW --condition smote_plain
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the Exp2 plotting routine — schema is identical.
from scripts.python.analysis.domain.plot_condition_seed_summaries import (
    plot_condition_summary,
)
# Reuse Exp3 pooled-data collector.
from scripts.python.analysis.domain.collect_split2_prior_research_metrics import (
    collect_pooled_data,
    CONDITION_DIR_MAP,
    CONDITION_FILE_LABEL,
    CONDITION_LABELS,
    PRIOR_MODELS,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CSV_BASE = PROJECT_ROOT / "results/analysis/exp3_prior_research/figures/csv/split2"
PNG_BASE = PROJECT_ROOT / "results/analysis/exp3_prior_research/figures/png/split2"

# Condition meta. Each entry maps the canonical condition tag (used in eval
# JSON / CSV rows) to the per-cell plotting parameters.
#   ratios: list of ratio strings to render (None = no ratio filter)
#   title : human-friendly title prefix for the suptitle
COND_SPECS = [
    {"condition": "baseline",        "ratios": [None]},
    {"condition": "imbalv3",         "ratios": ["0.1", "0.5"]},   # subject-wise SMOTE
    {"condition": "smote_plain",     "ratios": ["0.1", "0.5"]},
    {"condition": "undersample_rus", "ratios": ["0.1", "0.5"]},
]


def _csv_path(model: str, cond: str) -> Path:
    cond_dir = CONDITION_DIR_MAP.get(cond, cond)
    cond_label = CONDITION_FILE_LABEL.get(cond, cond)
    return CSV_BASE / model / cond_dir / f"{model.lower()}_{cond_label}_split2_metrics.csv"


def _ratio_match(series: pd.Series, ratio_str: str) -> pd.Series:
    """Match ratio as float (CSV may store '0.1' or 0.1 or '0.10')."""
    target = float(ratio_str)
    def ok(x):
        if pd.isna(x) or x == "":
            return False
        try:
            return abs(float(x) - target) < 1e-6
        except (TypeError, ValueError):
            return False
    return series.apply(ok)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate seed-aggregated summary plots for Exp3 prior-research"
    )
    parser.add_argument("--model", default=None, choices=PRIOR_MODELS,
                        help="Filter to a single model")
    parser.add_argument("--condition", default=None,
                        choices=[c["condition"] for c in COND_SPECS],
                        help="Filter to a single condition")
    parser.add_argument("--min-seeds", type=int, default=12,
                        help="Skip cells whose seed count is below this (default 12)")
    parser.add_argument("--strict-subcell", action="store_true", default=True,
                        help="Require min-seeds in EVERY (distance, level, mode) sub-cell, "
                             "not just overall (default: on)")
    args = parser.parse_args()

    models = [args.model] if args.model else PRIOR_MODELS
    cond_specs = (
        [c for c in COND_SPECS if c["condition"] == args.condition]
        if args.condition else COND_SPECS
    )

    # Pooled reference data — one global load, filtered per (model, cond, ratio).
    df_pooled_all = collect_pooled_data(model_filter=args.model)
    if df_pooled_all.empty:
        logger.warning("No pooled prior-research data found — Row 4 will be omitted.")

    generated: list[Path] = []
    skipped: list[tuple[str, str, str | None, str]] = []

    for model in models:
        for spec in cond_specs:
            cond = spec["condition"]
            csv_path = _csv_path(model, cond)
            if not csv_path.exists():
                logger.info(f"[SKIP] CSV not found: {csv_path}")
                continue
            df = pd.read_csv(csv_path)
            if df.empty:
                logger.info(f"[SKIP] empty CSV: {csv_path}")
                continue

            for ratio in spec["ratios"]:
                if ratio is None:
                    df_r = df
                    ratio_tag = ""
                    ratio_for_title = ""
                else:
                    df_r = df[_ratio_match(df["ratio"], ratio)]
                    ratio_tag = f"_r{ratio.replace('.', '')}"
                    ratio_for_title = ratio

                if df_r.empty:
                    skipped.append((model, cond, ratio, "no rows"))
                    continue
                n_seeds = df_r["seed"].nunique()
                if n_seeds < args.min_seeds:
                    skipped.append((model, cond, ratio, f"only {n_seeds} seed(s) overall"))
                    continue

                # Strict: each (distance, level, mode) sub-cell must also
                # have >= min-seeds, otherwise the mean/std for that bar
                # would be drawn from < min-seeds runs and the "summary"
                # is not really comparable across cells.
                if args.strict_subcell:
                    sub_counts = (
                        df_r.groupby(["distance", "level", "mode"])["seed"]
                            .nunique()
                    )
                    if sub_counts.empty or sub_counts.min() < args.min_seeds:
                        worst = int(sub_counts.min()) if not sub_counts.empty else 0
                        skipped.append(
                            (model, cond, ratio,
                             f"sub-cell minimum {worst} seed(s) < {args.min_seeds}")
                        )
                        continue

                # Pooled subset for this (model, cond[, ratio])
                df_pooled = pd.DataFrame()
                if not df_pooled_all.empty:
                    mask = (
                        (df_pooled_all["model"] == model)
                        & (df_pooled_all["condition"] == cond)
                    )
                    if ratio is not None:
                        mask = mask & _ratio_match(df_pooled_all["ratio"], ratio)
                    df_pooled = df_pooled_all[mask].copy()

                cond_dir = CONDITION_DIR_MAP.get(cond, cond)
                cond_label = CONDITION_FILE_LABEL.get(cond, cond)
                out_path = PNG_BASE / model / cond_dir / f"{cond_label}{ratio_tag}_summary.png"

                title_prefix = f"{model} – {CONDITION_LABELS.get(cond, cond)}"

                logger.info(
                    f"  Plot: {model}/{cond} ratio={ratio or 'N/A'} "
                    f"({len(df_r)} rows, {n_seeds} seeds, "
                    f"{len(df_pooled)} pooled rows) → {out_path.name}"
                )

                plot_condition_summary(
                    df=df_r,
                    df_pooled=df_pooled,
                    condition=cond,
                    ratio=ratio_for_title,
                    out_path=out_path,
                    title_prefix=title_prefix,
                )
                generated.append(out_path)

    logger.info("=" * 60)
    logger.info(f"DONE — {len(generated)} summary plot(s) generated, {len(skipped)} skipped")
    for model, cond, ratio, reason in skipped:
        logger.info(f"  SKIP {model}/{cond} ratio={ratio}: {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
