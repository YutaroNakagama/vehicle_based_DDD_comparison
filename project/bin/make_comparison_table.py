#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import argparse

# add project root
THIS = Path(__file__).resolve()
PRJ = THIS.parents[1]
sys.path.append(str(PRJ))

from src.analysis.metrics_tables import summarize_metrics, make_comparison_table

def main():
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

