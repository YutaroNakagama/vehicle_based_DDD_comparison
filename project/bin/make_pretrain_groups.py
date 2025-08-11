#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_pretrain_groups.py

Build 10-subject groups that are (1) most likely to BENEFIT from pretraining
("friendly") and (2) least likely to benefit ("hard") using only existing
distance artifacts for each metric: MMD, Wasserstein, DTW.

Artifacts expected:
  - MMD:
      matrix:   results/mmd/mmd_matrix.npy
      subjects: results/mmd/mmd_subjects.json
  - Wasserstein / DTW:
      matrix:   results/distances/wasserstein_matrix.npy / dtw_matrix.npy
      subjects: results/distances/subjects.json

Criteria (all computed robustly with NaN-safe stats):
  - Intra(T): mean of pairwise distances within T (i<j).  (smaller is better)
  - Inter(T,S): mean of cross distances between T and S.   (smaller is better)
  - NN(T→S): average over t in T of min_{s in S} dist(t,s).(smaller is better)

Objectives:
  Friendly (maximize pretrain gain): minimize Intra + Inter + NN
  Hard     (minimize pretrain gain): maximize (-Intra) + Inter + NN
           with a soft penalty if Intra explodes (avoid degenerate, unlearnable sets).

Greedy procedure:
  - Seed:
      Friendly: subject with smallest global mean distance (closest to source).
      Hard    : subject with largest global mean distance  (farthest from source).
  - Iteratively add one subject that optimizes the objective with scores
    min-max normalized across candidates at each step.
  - Locked / banned subjects can be supported if needed (simple hooks provided).

Outputs:
  misc/pretrain_groups/
      <metric>_friendly.txt   # one line with 10 IDs
      <metric>_hard.txt       # one line with 10 IDs
      <metric>_report.json    # quick metrics for both groups

Author: you
"""

import os
import json
import argparse
from typing import List, Set, Tuple, Dict
import numpy as np

# -------- I/O helpers --------

def load_matrix_subjects(matrix_path: str, subjects_path: str) -> Tuple[np.ndarray, List[str]]:
    M = np.load(matrix_path)
    with open(subjects_path, "r") as f:
        subs = json.load(f)
    if M.shape[0] != M.shape[1] or M.shape[0] != len(subs):
        raise ValueError(f"Matrix/subjects mismatch: {M.shape} vs {len(subs)}")
    np.fill_diagonal(M, 0.0)
    return M, subs

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_group_line(ids: List[str], path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        f.write(" ".join(ids) + "\n")
    print(f"[OK] saved {path}")

# -------- metrics on a set T vs complement S --------

def global_mean_dist(M: np.ndarray) -> np.ndarray:
    n = M.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return np.nanmean(np.where(mask, M, np.nan), axis=1)

def intra_mean(M: np.ndarray, T: Set[int]) -> float:
    idx = list(T)
    if len(idx) < 2:
        return 0.0
    vals = [M[i, j] for i in idx for j in idx if i < j]
    arr = np.array(vals, dtype=float)
    return float(np.nanmean(arr)) if arr.size else np.nan

def inter_mean(M: np.ndarray, T: Set[int], S: Set[int]) -> float:
    if not T or not S:
        return np.nan
    vals = [M[i, j] for i in T for j in S]
    arr = np.array(vals, dtype=float)
    return float(np.nanmean(arr)) if arr.size else np.nan

def nn_T_to_S(M: np.ndarray, T: Set[int], S: Set[int]) -> float:
    """Average nearest-neighbor distance from each t in T to S."""
    if not T or not S:
        return np.nan
    S_list = list(S)
    dists = []
    for t in T:
        arr = np.array([M[t, s] for s in S_list], dtype=float)
        if arr.size:
            dists.append(np.nanmin(arr))
    return float(np.nanmean(dists)) if dists else np.nan

# -------- normalization utils --------

def _nanmin(x: np.ndarray) -> float:
    v = np.nanmin(x)
    return float(v) if np.isfinite(v) else 0.0

def _nanmax(x: np.ndarray) -> float:
    v = np.nanmax(x)
    return float(v) if np.isfinite(v) else 1.0

def minmax_norm(arr: List[float]) -> np.ndarray:
    a = np.array(arr, dtype=float)
    lo, hi = _nanmin(a), _nanmax(a)
    denom = (hi - lo) if (hi - lo) > 1e-12 else 1.0
    return (a - lo) / denom

# -------- greedy builders --------

def build_group(
    M: np.ndarray,
    k: int,
    friendly: bool,
    seed_idx: int
) -> List[int]:
    """
    Greedy construction:
      friendly=True  -> minimize Intra + Inter + NN
      friendly=False -> maximize (-Intra) + Inter + NN (with Intra soft penalty)
    """
    n = M.shape[0]
    T: Set[int] = {seed_idx}
    while len(T) < k:
        candidates = [i for i in range(n) if i not in T]
        intra_vals, inter_vals, nn_vals, obj_vals = [], [], [], []
        for c in candidates:
            T_new = set(T) | {c}
            S_new = set(range(n)) - T_new
            intra = intra_mean(M, T_new)
            inter = inter_mean(M, T_new, S_new)
            nnavg = nn_T_to_S(M, T_new, S_new)
            intra_vals.append(intra)
            inter_vals.append(inter)
            nn_vals.append(nnavg)
        # normalize across candidates
        intra_n = minmax_norm(intra_vals)      # smaller is better
        inter_n = minmax_norm(inter_vals)      # smaller is better
        nn_n    = minmax_norm(nn_vals)         # smaller is better

        # objective per candidate
        # weights can be tuned; start equal
        if friendly:
            # minimize all three -> use their sum (already in [0,1])
            obj = -(1.0* (1.0 - intra_n) + 1.0* (1.0 - inter_n) + 1.0* (1.0 - nn_n))
            # equivalent: minimize (intra_n + inter_n + nn_n)
        else:
            # maximize Inter and NN, but avoid exploding Intra:
            # gain = (inter + nn) - soft_penalty(intra)
            # soft penalty: linear on normalized intra; can be reduced if desired
            penalty = 0.5 * intra_n
            gain = (inter_n + nn_n) - penalty
            obj = gain  # larger is better

        # choose best candidate per objective
        if friendly:
            # most negative obj (smallest normalized sum)
            best_idx = int(np.nanargmin(obj))
        else:
            best_idx = int(np.nanargmax(obj))

        T.add(candidates[best_idx])

    return list(T)

def pick_seed(M: np.ndarray, friendly: bool) -> int:
    g = global_mean_dist(M)
    # friendly: start from the most "central" (smallest global mean)
    # hard    : start from the most "peripheral" (largest global mean)
    order = np.argsort(g) if friendly else np.argsort(-g)
    return int(order[0])

# -------- end-to-end per metric --------

def make_groups_for_metric(
    metric_name: str,
    matrix_path: str,
    subjects_path: str,
    k: int = 10,
    out_dir: str = "misc/pretrain_groups"
) -> Dict[str, Dict[str, float]]:
    M, subs = load_matrix_subjects(matrix_path, subjects_path)
    seed_friendly = pick_seed(M, friendly=True)
    seed_hard     = pick_seed(M, friendly=False)

    T_f = set(build_group(M, k, friendly=True,  seed_idx=seed_friendly))
    T_h = set(build_group(M, k, friendly=False, seed_idx=seed_hard))

    S_f = set(range(len(subs))) - T_f
    S_h = set(range(len(subs))) - T_h

    # quick stats
    stats = {
        "friendly": {
            "intra": float(intra_mean(M, T_f)),
            "inter": float(inter_mean(M, T_f, S_f)),
            "nn":    float(nn_T_to_S(M, T_f, S_f)),
            "ids":   [subs[i] for i in T_f]
        },
        "hard": {
            "intra": float(intra_mean(M, T_h)),
            "inter": float(inter_mean(M, T_h, S_h)),
            "nn":    float(nn_T_to_S(M, T_h, S_h)),
            "ids":   [subs[i] for i in T_h]
        }
    }

    ensure_dir(out_dir)
    save_group_line(stats["friendly"]["ids"], os.path.join(out_dir, f"{metric_name}_friendly.txt"))
    save_group_line(stats["hard"]["ids"],     os.path.join(out_dir, f"{metric_name}_hard.txt"))
    # per-metric json report
    with open(os.path.join(out_dir, f"{metric_name}_report.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[{metric_name}] Friendly stats:", {k:v for k,v in stats["friendly"].items() if k!='ids'})
    print(f"[{metric_name}] Hard     stats:", {k:v for k,v in stats["hard"].items() if k!='ids'})
    return stats

# -------- CLI --------

def main():
    ap = argparse.ArgumentParser(description="Make 10-subject groups maximizing/minimizing pretrain benefit per metric.")
    ap.add_argument("--k", type=int, default=10, help="Group size (default: 10).")
    ap.add_argument("--out", default="misc/pretrain_groups", help="Output directory.")
    args = ap.parse_args()

    # paths for each metric
    metrics = [
        ("mmd",         "results/mmd/mmd_matrix.npy",                 "results/mmd/mmd_subjects.json"),
        ("wasserstein", "results/distances/wasserstein_matrix.npy",   "results/distances/subjects.json"),
        ("dtw",         "results/distances/dtw_matrix.npy",           "results/distances/subjects.json"),
    ]

    all_stats = {}
    for name, mpath, spath in metrics:
        print(f"\n=== {name.upper()} ===")
        all_stats[name] = make_groups_for_metric(name, mpath, spath, k=args.k, out_dir=args.out)

    # combined report
    with open(os.path.join(args.out, "summary_all_metrics.json"), "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\n[OK] Summary saved to {os.path.join(args.out, 'summary_all_metrics.json')}")

if __name__ == "__main__":
    main()

