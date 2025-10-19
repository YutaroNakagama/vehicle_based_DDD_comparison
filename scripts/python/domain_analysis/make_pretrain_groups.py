#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a wide-format comparison table for only10 vs finetune results.

This script creates a wide-form summary table of evaluation metrics for
driver drowsiness detection experiments. It ensures a metrics summary
CSV is available, generating one if necessary, and then produces the
wide comparison table.

Examples
--------
Generate a wide comparison table for Random Forest results in
``model/common``:

    $ python make_pretrain_groups.py --model_dir model/common --model_tag RF
"""

import sys
from pathlib import Path
import argparse

# add project root
THIS = Path(__file__).resolve()
PRJ = THIS.parents[1]
sys.path.append(str(PRJ))

from src.analysis.metrics_tables import summarize_metrics, make_comparison_table

def main():
    """Parse CLI arguments and generate the wide comparison table.

    Parameters
    ----------
    None

    Other Parameters
    ----------------
    --model_dir : str, optional
        Path to the model directory (default: ``model/common``).
    --model_tag : str, optional
        Model tag identifying the architecture (default: ``RF``).
    --split : str, optional
        Dataset split to summarize (default: ``test``).
    --summary_csv : str, optional
        Path to an existing summary CSV. If not provided, a new summary
        will be generated in ``model_dir``.
    --out_csv : str, optional
        Path to save the wide comparison table. Defaults to
        ``table_only10_vs_finetune_wide.csv`` in ``model_dir``.

    Returns
    -------
    None

    Raises
    ------
    SystemExit
        If invalid arguments are passed.
    """
    ap = argparse.ArgumentParser(description="Make wide comparison table (only10 vs finetune).")
    ap.add_argument("--model_dir", default=str(PRJ / "model" / "common"))
    ap.add_argument("--model_tag", default="RF")
    ap.add_argument("--split", default="test")
    ap.add_argument("--summary_csv", default=None, help="If omitted, will generate from model_dir.")
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    summary_csv = Path(args.summary_csv) if args.summary_csv else (model_dir / "summary_only10_vs_finetune.csv")
    out_csv = Path(args.out_csv) if args.out_csv else (model_dir / "table_only10_vs_finetune_wide.csv")

    if not summary_csv.exists():
        # build summary first
        summarize_metrics(model_dir=model_dir, model_tag=args.model_tag, split=args.split, out_csv=summary_csv)

    wide = make_comparison_table(summary_df_or_path=summary_csv, out_csv=out_csv)
    print(f"Saved CSV: {out_csv}")
    print(wide)

if __name__ == "__main__":
    main()

