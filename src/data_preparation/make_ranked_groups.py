#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_ranked_groups.py
=====================

Generate ranked 10-subject group files (high / middle / low)
based on domain distance matrices (MMD, Wasserstein, DTW).

Outputs are saved under:
    results/domain_analysis/group_distances/ranks10/{metric}_mean_{level}.txt
"""

import json
import numpy as np
from pathlib import Path

# ============================================================
# Configuration
# ============================================================
METRICS = ["wasserstein", "mmd", "dtw"]
ROOT = Path("results/domain_analysis/distance")
OUT_DIR = ROOT / "ranks10"
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
    """Save a group of subject IDs to a text file."""
    path = OUT_DIR / f"{metric}_mean_{name}.txt"
    path.write_text("\n".join(subjects) + "\n", encoding="utf-8")
    print(f"[SAVED] {safe_relpath(path)}")


# ============================================================
# Main
# ============================================================
def main() -> None:
    """Generate top/middle/low 10-subject groups for each distance metric."""
    for metric in METRICS:
        matrix_path = ROOT / metric / f"{metric}_matrix.npy"
        subjects_path = ROOT / metric / f"{metric}_subjects.json"

        # --- Check files ---
        if not matrix_path.exists() or not subjects_path.exists():
            print(f"[WARN] Skipping {metric}: missing matrix or subject file.")
            continue

        # --- Load matrix and subjects ---
        M = np.load(matrix_path)
        subjects = json.loads(subjects_path.read_text())

        if M.shape[0] != M.shape[1] or len(subjects) != M.shape[0]:
            print(f"[ERROR] Shape mismatch in {metric}: {M.shape} vs {len(subjects)}")
            continue

        # --- Compute mean distance for each subject ---
        np.fill_diagonal(M, np.nan)
        mean_dist = np.nanmean(M, axis=1)
        ranked_idx = np.argsort(mean_dist)

        # --- Split into High / Middle / Low (10 each) ---
        n = len(ranked_idx)
        high = [subjects[i] for i in ranked_idx[:10]]
        middle = [subjects[i] for i in ranked_idx[n // 2 - 5 : n // 2 + 5]]
        low = [subjects[i] for i in ranked_idx[-10:]]

        # --- Save outputs ---
        save_group(metric, "high", high)
        save_group(metric, "middle", middle)
        save_group(metric, "low", low)

    print("\nAll ranked groups generated successfully.")


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    main()

