#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""                                                                                                                                                                                                                                                                                                     
subject_group_generator.py
==========================

Generate ranked subject group files (out_domain / mid_domain / in_domain)
based on domain distance matrices (MMD, Wasserstein, DTW).

This module is automatically called after distance computation to create
subject groups for domain generalization experiments.

Outputs are saved under:
    results/analysis/exp2_domain_shift/distance/ranks{N}/{metric}_mean_{level}.txt
"""import argparse
import json
import sys
import numpy as np
from pathlib import Path

from src import config as cfg

import logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ============================================================
# Configuration
# ============================================================
METRICS = cfg.DISTANCE_METRICS
ROOT = Path(cfg.RESULTS_DOMAIN_ANALYSIS_PATH) / "distance" / "subject-wise"
RANKS_ROOT = Path(cfg.RESULTS_DOMAIN_ANALYSIS_PATH) / "distance" / "subject-wise" / "ranks"
# Output to mean_distance subfolder (new structure)
OUT_DIR = RANKS_ROOT / "ranks29" / "mean_distance"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Utility
# ============================================================
def safe_relpath(path: Path) -> str:
    """Return a relative path if possible, else absolute path."""
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path.resolve())


def save_group(metric: str, name: str, subjects: list[str]) -> None:
    """Save a group of subject IDs to a text file.
    
    New format: {metric}_{level}.txt (without 'mean_' prefix)
    """
    path = OUT_DIR / f"{metric}_{name}.txt"
    path.write_text("\n".join(subjects) + "\n", encoding="utf-8")
    print(f"[SAVED] {safe_relpath(path)}")


# ============================================================
# Main
# ============================================================
def main() -> int:
    """Generate top/middle/low 10-subject groups for selected metric(s)."""

    parser = argparse.ArgumentParser(description="Generate ranked groups from distance matrices.")
    parser.add_argument(
        "--metric",
        choices=["mmd", "wasserstein", "dtw", "all"],
        default="all",
        help="Which metric(s) to process (default: all)."
    )
    args = parser.parse_args()

    metrics = [args.metric] if args.metric != "all" else METRICS

    for metric in metrics:
        matrix_path = ROOT / metric / f"{metric}_matrix.npy"
        subjects_path = ROOT / metric / f"{metric}_subjects.json"

        # --- Check files ---
        if not matrix_path.exists() or not subjects_path.exists():
            logging.warning(f"Skipping {metric}: missing matrix or subject file.")
            return 1

        # --- Load matrix and subjects ---
        M = np.load(matrix_path)
        subjects = json.loads(subjects_path.read_text())

        if M.shape[0] != M.shape[1] or len(subjects) != M.shape[0]:
            logging.error(f"Shape mismatch in {metric}: {M.shape} vs {len(subjects)}")
            return 1

        # --- Compute mean distance for each subject ---
        np.fill_diagonal(M, np.nan)
        mean_dist = np.nanmean(M, axis=1)
        ranked_idx = np.argsort(-mean_dist)

#        # --- Split into High / Middle / Low (10 each) ---
#        n = len(ranked_idx)
#        high = [subjects[i] for i in ranked_idx[:10]]
#        middle = [subjects[i] for i in ranked_idx[n // 2 - 5 : n // 2 + 5]]
#        low = [subjects[i] for i in ranked_idx[-10:]]

        # --- Split into High / Middle / Low (29 each) ---
        group_size = 29
        half = group_size // 2   # = 14
        n = len(ranked_idx)

        # High: top 29
        high = [subjects[i] for i in ranked_idx[:group_size]]

        # Middle: centred 29 around the median
        mid_start = max(0, n // 2 - half)
        mid_end   = min(n, n // 2 + half + 1)  # +1 to make total 29
        middle = [subjects[i] for i in ranked_idx[mid_start:mid_end]]

        # Low: bottom 29
        low = [subjects[i] for i in ranked_idx[-group_size:]]

        # --- Save outputs ---
        save_group(metric, "out_domain", high)
        save_group(metric, "mid_domain", middle)
        save_group(metric, "in_domain", low)

    logging.info("All ranked groups generated successfully.")
    return 0


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    sys.exit(main())
