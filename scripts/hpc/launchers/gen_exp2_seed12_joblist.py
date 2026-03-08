#!/usr/bin/env python3
"""Generate the job list for Exp2 12-seed completion.

Reads existing CSV results to find missing seeds, then adds seed 999 for all cells.
Outputs a job list file for the launcher script.

Usage:
    python scripts/hpc/launchers/gen_exp2_seed12_joblist.py [--output FILE]
"""
import argparse
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CSV_BASE = PROJECT_ROOT / "results/analysis/exp2_domain_shift/figures/csv/split2"

FILES = {
    "baseline": CSV_BASE / "baseline" / "baseline_domain_split2_metrics_v2.csv",
    "smote_plain": CSV_BASE / "smote_plain" / "smote_plain_split2_metrics_v2.csv",
    "undersample": CSV_BASE / "undersample_rus" / "undersample_rus_split2_metrics_v2.csv",
    "sw_smote": CSV_BASE / "sw_smote" / "sw_smote_split2_metrics_v2.csv",
}

CONDITIONS_7 = [
    "baseline", "rus_r01", "rus_r05",
    "smote_r01", "smote_r05",
    "sw_smote_r01", "sw_smote_r05",
]
COND_TO_PBS = {
    "baseline":      ("baseline",    ""),
    "rus_r01":       ("undersample", "0.1"),
    "rus_r05":       ("undersample", "0.5"),
    "smote_r01":     ("smote_plain", "0.1"),
    "smote_r05":     ("smote_plain", "0.5"),
    "sw_smote_r01":  ("smote",       "0.1"),
    "sw_smote_r05":  ("smote",       "0.5"),
}

MODES = ["source_only", "target_only", "mixed"]
DISTANCES = ["mmd", "dtw", "wasserstein"]
LEVELS = ["in_domain", "out_domain"]
EXISTING_SEEDS = [0, 1, 3, 7, 13, 42, 123, 256, 512, 1337, 2024]
NEW_SEED = 999


def load_data():
    dfs = []
    for key, path in FILES.items():
        df = pd.read_csv(path)
        if key == "baseline":
            df["condition"] = "baseline"
        elif key == "sw_smote":
            df["condition"] = df["ratio"].apply(
                lambda r: f'sw_smote_r{str(r).replace(".", "")}'
            )
        elif key == "smote_plain":
            df["condition"] = df["ratio"].apply(
                lambda r: f'smote_r{str(r).replace(".", "")}'
            )
        elif key == "undersample":
            df["condition"] = df["ratio"].apply(
                lambda r: f'rus_r{str(r).replace(".", "")}'
            )
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", "-o",
        default=str(PROJECT_ROOT / "scripts/hpc/launchers/exp2_seed12_joblist.txt"),
    )
    args = parser.parse_args()

    df = load_data()
    jobs = []

    # 1. Find missing seeds in existing data
    for cond in CONDITIONS_7:
        for mode in MODES:
            for dist in DISTANCES:
                for level in LEVELS:
                    cell = df[
                        (df["condition"] == cond)
                        & (df["mode"] == mode)
                        & (df["distance"] == dist)
                        & (df["level"] == level)
                    ]
                    present = set(cell["seed"].values)
                    for s in EXISTING_SEEDS:
                        if s not in present:
                            pbs_cond, ratio = COND_TO_PBS[cond]
                            jobs.append(f"{pbs_cond}|{ratio}|{mode}|{dist}|{level}|{s}")

    n_missing = len(jobs)

    # 2. Add new seed for all 126 cells
    for mode in MODES:
        for dist in DISTANCES:
            for level in LEVELS:
                for cond in CONDITIONS_7:
                    pbs_cond, ratio = COND_TO_PBS[cond]
                    jobs.append(f"{pbs_cond}|{ratio}|{mode}|{dist}|{level}|{NEW_SEED}")

    n_new = len(jobs) - n_missing

    with open(args.output, "w") as f:
        f.write(f"# Exp2 12-seed completion job list\n")
        f.write(f"# Generated from CSV data\n")
        f.write(f"# Missing seed fill: {n_missing} jobs\n")
        f.write(f"# New seed ({NEW_SEED}): {n_new} jobs\n")
        f.write(f"# Total: {len(jobs)} jobs\n")
        f.write(f"# Format: CONDITION|RATIO|MODE|DISTANCE|DOMAIN|SEED\n")
        for line in jobs:
            f.write(line + "\n")

    print(f"Missing seed fill: {n_missing}")
    print(f"New seed ({NEW_SEED}):    {n_new}")
    print(f"Total:             {len(jobs)}")
    print(f"Written to: {args.output}")


if __name__ == "__main__":
    main()
