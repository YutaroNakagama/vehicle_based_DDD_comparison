#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
regenerate_split2_normalized.py
================================
Regenerate split2 KNN domain groups using NORMALIZED features.

Steps:
  1. Load raw features for all 87 subjects
  2. Apply global z-score normalization (StandardScaler)
  3. Compute pairwise distance matrices (MMD, DTW, Wasserstein)
  4. Rank subjects using KNN (k=5) average distance
  5. Split into in_domain (44) and out_domain (43) groups
  6. Save to results/analysis/exp2_domain_shift/distance/rankings/split2/knn/
  7. Also save distance matrices to distance/matrices/ for reference

Before overwriting, backs up the existing group files.

Usage:
  python scripts/python/analysis/domain/regenerate_split2_normalized.py
  python scripts/python/analysis/domain/regenerate_split2_normalized.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.domain.distance import (
    _extract_features,
    _compute_distance_matrix,
)

# ── Configuration ──
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "common"
SUBJECT_LIST = PROJECT_ROOT / "config" / "subjects" / "subject_list.txt"
METRICS = ["mmd", "wasserstein", "dtw"]
KNN_K = 5
IN_DOMAIN_SIZE = 44   # split2: 44 in_domain + 43 out_domain = 87
OUT_DOMAIN_SIZE = 43

RANKINGS_DIR = (
    PROJECT_ROOT
    / "results"
    / "analysis"
    / "exp2_domain_shift"
    / "distance"
    / "rankings"
    / "split2"
    / "knn"
)
MATRICES_DIR = (
    PROJECT_ROOT
    / "results"
    / "analysis"
    / "exp2_domain_shift"
    / "distance"
    / "matrices_normalized"
)


def load_subjects() -> list[str]:
    """Load subject list from config."""
    return [
        line.strip()
        for line in SUBJECT_LIST.read_text().splitlines()
        if line.strip()
    ]


def normalize_features(
    features: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Apply global z-score normalization across all subjects.

    Matches the normalization in run_comp_dist() (distance.py):
    - Pad shorter subjects to the max column count with zeros
    - Then global z-score (same as np.vstack -> mean/std)
    """
    max_cols = max(X.shape[1] for X in features.values())
    min_cols = min(X.shape[1] for X in features.values())

    if max_cols != min_cols:
        mismatched = {s: X.shape[1] for s, X in features.items() if X.shape[1] != max_cols}
        print(f"  [WARN] Dimension mismatch: max={max_cols}, padding {len(mismatched)} subjects")
        for s, d in mismatched.items():
            print(f"    {s}: {d} → {max_cols} (zero-padding {max_cols - d} cols)")
        features = {
            s: np.pad(X, ((0, 0), (0, max_cols - X.shape[1])), mode="constant")
            if X.shape[1] < max_cols else X
            for s, X in features.items()
        }

    # Global z-score (same as run_comp_dist: vstack → mean/std)
    all_data = np.vstack(list(features.values()))
    mu = all_data.mean(axis=0, keepdims=True)
    sigma = all_data.std(axis=0, keepdims=True) + 1e-12
    return {s: ((X - mu) / sigma).astype(np.float32) for s, X in features.items()}


def rank_by_knn(matrix: np.ndarray, k: int = KNN_K) -> np.ndarray:
    """Compute KNN-based outlier score for each subject.

    Returns an array of mean distance to k nearest neighbors.
    """
    n = matrix.shape[0]
    scores = np.zeros(n)
    for i in range(n):
        dists = matrix[i].copy()
        dists[i] = np.inf
        k_actual = min(k, n - 1)
        k_nearest = np.partition(dists, k_actual - 1)[:k_actual]
        scores[i] = np.mean(k_nearest)
    return scores


def split_groups(
    knn_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Split into out_domain (top 43) and in_domain (bottom 44) by KNN score.

    Returns (out_domain_indices, in_domain_indices).
    """
    sorted_idx = np.argsort(-knn_scores)  # descending: highest = outlier
    out_domain = sorted_idx[:OUT_DOMAIN_SIZE]
    in_domain = sorted_idx[-IN_DOMAIN_SIZE:]
    return out_domain, in_domain


def backup_existing(dry_run: bool = False) -> Path | None:
    """Backup existing group files. Returns backup path."""
    if not RANKINGS_DIR.exists():
        return None
    backup_dir = RANKINGS_DIR.parent / f"knn_backup_{datetime.now():%Y%m%d_%H%M%S}"
    if dry_run:
        print(f"[DRY-RUN] Would backup {RANKINGS_DIR} → {backup_dir}")
        return None
    shutil.copytree(RANKINGS_DIR, backup_dir)
    print(f"Backed up existing groups → {backup_dir}")
    return backup_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print what would happen")
    parser.add_argument("--n-jobs", type=int, default=4, help="Parallel workers")
    args = parser.parse_args()

    # 1. Load subjects
    subjects = load_subjects()
    print(f"Subjects: {len(subjects)}")

    # 2. Load raw features
    print("Loading features...")
    raw_features = _extract_features(subjects, DATA_DIR)
    loaded = [s for s in subjects if s in raw_features]
    print(f"Loaded: {len(loaded)} subjects, dims={next(iter(raw_features.values())).shape[1]}")

    if len(loaded) != len(subjects):
        missing = set(subjects) - set(loaded)
        print(f"[WARN] Missing subjects: {missing}")

    # 3. Normalize
    print("Normalizing features (global z-score)...")
    norm_features = normalize_features(raw_features)
    n_features = next(iter(norm_features.values())).shape[1]
    print(f"Normalized: {n_features} features")

    # Feature scale verification
    all_data = np.vstack(list(norm_features.values()))
    col_std = np.std(all_data, axis=0)
    print(f"Post-normalization std: min={col_std.min():.4f}, max={col_std.max():.4f} (should be ~1.0)")

    # 4. Backup existing groups
    backup_dir = backup_existing(dry_run=args.dry_run)

    # 5. Compute distance matrices and generate groups for each metric
    subject_list = list(norm_features.keys())

    for metric in METRICS:
        print(f"\n{'='*60}")
        print(f"Computing {metric.upper()} distance matrix...")
        matrix, subjects_valid = _compute_distance_matrix(
            norm_features, metric, n_jobs=args.n_jobs
        )
        print(f"  Matrix shape: {matrix.shape}")

        # Save distance matrix
        if not args.dry_run:
            MATRICES_DIR.mkdir(parents=True, exist_ok=True)
            np.save(MATRICES_DIR / f"{metric}_matrix.npy", matrix)
            with open(MATRICES_DIR / f"{metric}_subjects.json", "w") as f:
                json.dump(subjects_valid, f)

        # KNN ranking
        knn_scores = rank_by_knn(matrix, k=KNN_K)
        out_idx, in_idx = split_groups(knn_scores)

        out_subjects = [subjects_valid[i] for i in out_idx]
        in_subjects = [subjects_valid[i] for i in in_idx]

        print(f"  out_domain: {len(out_subjects)} subjects")
        print(f"  in_domain:  {len(in_subjects)} subjects")

        # Compare with existing groups (use backup to avoid comparing with overwritten files)
        compare_dir = backup_dir if backup_dir else RANKINGS_DIR
        existing_in = compare_dir / f"{metric}_in_domain.txt"
        if existing_in.exists():
            old_in = set(l.strip() for l in existing_in.read_text().splitlines() if l.strip())
            new_in = set(in_subjects)
            overlap = old_in & new_in
            switched = len(old_in - new_in)
            print(f"  vs existing: {len(overlap)}/{len(old_in)} overlap, {switched} subjects switched")

        # Save group files
        if not args.dry_run:
            RANKINGS_DIR.mkdir(parents=True, exist_ok=True)
            (RANKINGS_DIR / f"{metric}_out_domain.txt").write_text(
                "\n".join(out_subjects) + "\n"
            )
            (RANKINGS_DIR / f"{metric}_in_domain.txt").write_text(
                "\n".join(in_subjects) + "\n"
            )
            print(f"  Saved: {RANKINGS_DIR / f'{metric}_*.txt'}")
        else:
            print(f"  [DRY-RUN] Would save to {RANKINGS_DIR}")

    # 6. Save summary
    if not args.dry_run:
        summary = RANKINGS_DIR / "ranks43_summary.txt"
        lines = [
            "ranks43 KNN Ranking Summary (NORMALIZED features)",
            "=" * 50,
            f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
            f"Normalization: global z-score (StandardScaler)",
            f"KNN k: {KNN_K}",
            f"in_domain size:  {IN_DOMAIN_SIZE}",
            f"out_domain size: {OUT_DOMAIN_SIZE}",
            f"K neighbors:     {KNN_K}",
            "",
        ]
        for metric in METRICS:
            lines.append(f"\n{metric.upper()}:")
            lines.append(f"  in_domain:  {IN_DOMAIN_SIZE}")
            lines.append(f"  out_domain: {OUT_DOMAIN_SIZE}")
        summary.write_text("\n".join(lines) + "\n")
        print(f"\nSummary saved: {summary}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
