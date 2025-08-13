#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import argparse

# add project root to import path
THIS = Path(__file__).resolve()
PRJ = THIS.parents[1]
sys.path.insert(0, str(PRJ))

from src.analysis.pretrain_groups_report import run_report_pretrain_groups

def main():
    ap = argparse.ArgumentParser(
        description="Report Intra/Inter/NN for 10 vs 78 using pretrain groups and existing distance artifacts."
    )
    ap.add_argument("--group_dir", default=str(PRJ / "misc" / "pretrain_groups"))
    ap.add_argument("--out_summary_json", default=str(PRJ / "misc" / "pretrain_groups" / "summary_report_ext.json"))
    ap.add_argument("--out_summary_csv",  default=str(PRJ / "misc" / "pretrain_groups" / "summary_report_ext.csv"))

    # optional overrides for artifact locations
    ap.add_argument("--mmd_matrix", default=str(PRJ / "results" / "mmd" / "mmd_matrix.npy"))
    ap.add_argument("--mmd_subjects", default=str(PRJ / "results" / "mmd" / "mmd_subjects.json"))
    ap.add_argument("--wasserstein_matrix", default=str(PRJ / "results" / "distances" / "wasserstein_matrix.npy"))
    ap.add_argument("--dtw_matrix", default=str(PRJ / "results" / "distances" / "dtw_matrix.npy"))
    ap.add_argument("--dist_subjects", default=str(PRJ / "results" / "distances" / "subjects.json"))
    args = ap.parse_args()

    run_report_pretrain_groups(
        group_dir=Path(args.group_dir),
        out_summary_json=Path(args.out_summary_json),
        out_summary_csv=Path(args.out_summary_csv),
        mmd_matrix=Path(args.mmd_matrix),
        mmd_subjects=Path(args.mmd_subjects),
        wass_matrix=Path(args.wasserstein_matrix),
        dtw_matrix=Path(args.dtw_matrix),
        dist_subjects=Path(args.dist_subjects),
    )
    print(f"[OK] Wrote {args.out_summary_json} and {args.out_summary_csv}")

if __name__ == "__main__":
    main()

