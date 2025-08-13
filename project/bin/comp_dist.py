#!/usr/bin/env python3
# Backward-compatible wrapper for domain distance computations.
# English comments only.

import os, sys
# add project root so that "import src.***" works when running from bin/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.analysis.distances import run_comp_dist
import argparse

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute MMD/Wasserstein/DTW matrices and group summaries.")
    p.add_argument("--subject_list", default="../../dataset/mdapbe/subject_list.txt",
                   help="Path to subject_list.txt (default keeps current behavior)")
    p.add_argument("--data_root", default="data/processed/common",
                   help="Root directory of processed CSVs (default keeps current behavior)")
    p.add_argument("--out_mmd_dir", default="results/mmd",
                   help="Output directory for MMD results")
    p.add_argument("--out_dist_dir", default="results/distances",
                   help="Output directory for Wasserstein/DTW results")
    p.add_argument("--groups_file", default="../misc/target_groups.txt",
                   help="Path to group definition file")
    return p

def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return run_comp_dist(
        subject_list_path=args.subject_list,
        data_root=args.data_root,
        out_mmd_dir=args.out_mmd_dir,
        out_dist_dir=args.out_dist_dir,
        groups_file=args.groups_file,
    )

if __name__ == "__main__":
    raise SystemExit(main())

