#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
from pathlib import Path
import argparse

# add project root
THIS = Path(__file__).resolve()
PRJ = THIS.parents[1]
sys.path.append(str(PRJ))

from src.analysis.metrics_tables import summarize_metrics

def main():
    ap = argparse.ArgumentParser(description="Summarize metrics_* CSVs under model_dir.")
    ap.add_argument("--model_dir", default=str(PRJ / "model" / "common"))
    ap.add_argument("--model_tag", default="RF")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    out_csv = Path(args.out_csv) if args.out_csv else (model_dir / "summary_only10_vs_finetune.csv")

    df = summarize_metrics(
        model_dir=model_dir,
        model_tag=args.model_tag,
        split=args.split,
        out_csv=out_csv,
    )
    print(f"Saved: {out_csv}")
    print(df)

if __name__ == "__main__":
    main()

