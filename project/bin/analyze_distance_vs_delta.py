#!/usr/bin/env python3
# bin/analyze_distance_vs_delta.py
import os, sys, argparse
# add project root for "src.***"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.analysis.correlation import run_distance_vs_delta

def main():
    ap = argparse.ArgumentParser(
        description="Correlate group distances (d(U,G), disp(G)) with delta metrics."
    )
    ap.add_argument("--summary_csv", required=True,
                    help="Wide CSV with columns: group, accuracy_delta, f1_delta, auc_delta, precision_delta, recall_delta")
    ap.add_argument("--distance", required=True,
                    help="Distance matrix path (.csv or .npy)")
    ap.add_argument("--subjects_json", default=None,
                    help="Required when --distance is .npy (JSON list of subject IDs)")
    ap.add_argument("--groups_dir", required=True,
                    help="Directory containing group member txt files (e.g., misc/pretrain_groups)")
    ap.add_argument("--group_names_file", required=True,
                    help="A file listing group txt basenames (one per line)")
    ap.add_argument("--outdir", default="model/common/dist_corr",
                    help="Output directory")
    ap.add_argument("--subject_list", default=None,
                    help="Optional txt of all subject IDs (otherwise inferred from distance matrix)")
    args = ap.parse_args()

    return run_distance_vs_delta(
        summary_csv=args.summary_csv,
        distance_path=args.distance,
        groups_dir=args.groups_dir,
        group_names_file=args.group_names_file,
        outdir=args.outdir,
        subjects_json=args.subjects_json,
        subject_list=args.subject_list,
    )

if __name__ == "__main__":
    raise SystemExit(main())

