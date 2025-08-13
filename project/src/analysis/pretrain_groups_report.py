# src/analysis/pretrain_groups_report.py
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
import json
import numpy as np
import pandas as pd

# HPC/非GUI環境でも安全
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------- I/O helpers ----------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def save_json(obj, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def load_matrix_subjects(matrix_path: Path, subjects_path: Path) -> Tuple[np.ndarray, List[str]]:
    """Load (N,N) distance matrix and subject id list; enforce zero diagonal."""
    M = np.load(matrix_path)
    subs = json.loads(subjects_path.read_text())
    if M.shape[0] != M.shape[1] or M.shape[0] != len(subs):
        raise ValueError(f"Matrix/subjects mismatch: {M.shape} vs {len(subs)}")
    np.fill_diagonal(M, 0.0)
    return M, subs

def load_group_ids(file: Path) -> List[str]:
    """Read one-line text: 'Sxxxx_1 Syyyy_2 ...'"""
    return file.read_text(encoding="utf-8").strip().split()


# ---------- metrics on T vs S ----------

def intra_mean(M: np.ndarray, T: Set[int]) -> float:
    idx = list(T)
    if len(idx) < 2:
        return float("nan")
    vals = [M[i, j] for i in idx for j in idx if i < j]
    arr = np.array(vals, dtype=float)
    return float(np.nanmean(arr)) if arr.size else float("nan")

def inter_mean(M: np.ndarray, T: Set[int], S: Set[int]) -> float:
    if not T or not S:
        return float("nan")
    vals = [M[i, j] for i in T for j in S]
    arr = np.array(vals, dtype=float)
    return float(np.nanmean(arr)) if arr.size else float("nan")

def nn_T_to_S(M: np.ndarray, T: Set[int], S: Set[int]) -> float:
    """Average over t in T of min distance to S (NaN-safe)."""
    if not T or not S:
        return float("nan")
    S_list = list(S)
    mins: List[float] = []
    for t in T:
        arr = np.array([M[t, s] for s in S_list], dtype=float)
        if arr.size:
            mins.append(float(np.nanmin(arr)))
    return float(np.nanmean(mins)) if mins else float("nan")


# ---------- plotting ----------

def plot_bars(
    metric_name: str,
    stats: Dict[str, Dict[str, float]],
    out_png: Path,
    ylabel: str = "Distance",
) -> None:
    """
    stats = {
      "friendly": {"intra": x, "inter": y, "nn": z},
      "hard":     {"intra": a, "inter": b, "nn": c}
    }
    """
    labels = ["Intra", "Inter", "NN"]
    friendly_vals = [stats["friendly"]["intra"], stats["friendly"]["inter"], stats["friendly"]["nn"]]
    hard_vals     = [stats["hard"]["intra"],     stats["hard"]["inter"],     stats["hard"]["nn"]]

    x = np.arange(len(labels))
    width = 0.36

    plt.figure(figsize=(7.5, 4.5))
    plt.bar(x - width/2, friendly_vals, width, label="Friendly")
    plt.bar(x + width/2, hard_vals,     width, label="Hard")
    plt.xticks(x, labels)
    plt.ylabel(ylabel)
    plt.title(f"{metric_name.upper()}: Intra / Inter / NN (10 vs 78)")
    plt.legend()
    plt.tight_layout()
    ensure_dir(out_png.parent)
    plt.savefig(out_png, dpi=180)
    plt.close()


# ---------- per-metric core ----------

def report_for_metric(
    metric_name: str,
    matrix_path: Path,
    subjects_path: Path,
    group_friendly_file: Path,
    group_hard_file: Path,
    out_dir: Path,
) -> Dict[str, Dict[str, float]]:
    """
    Compute stats for one metric and write per-metric JSON/CSV/PNG into out_dir.
    Returns the stats dict.
    """
    M, subs = load_matrix_subjects(matrix_path, subjects_path)
    id2idx = {sid: i for i, sid in enumerate(subs)}

    ids_f = load_group_ids(group_friendly_file)
    ids_h = load_group_ids(group_hard_file)

    T_f = {id2idx[s] for s in ids_f if s in id2idx}
    T_h = {id2idx[s] for s in ids_h if s in id2idx}
    S_f = set(range(len(subs))) - T_f
    S_h = set(range(len(subs))) - T_h

    stats = {
        "friendly": {
            "intra": intra_mean(M, T_f),
            "inter": inter_mean(M, T_f, S_f),
            "nn":    nn_T_to_S(M, T_f, S_f),
            "ids":   [subs[i] for i in sorted(T_f)],
        },
        "hard": {
            "intra": intra_mean(M, T_h),
            "inter": inter_mean(M, T_h, S_h),
            "nn":    nn_T_to_S(M, T_h, S_h),
            "ids":   [subs[i] for i in sorted(T_h)],
        },
    }

    # outputs
    ensure_dir(out_dir)
    save_json(stats, out_dir / f"{metric_name}_report_ext.json")
    pd.DataFrame.from_dict(stats, orient="index")[["intra","inter","nn"]].to_csv(
        out_dir / f"{metric_name}_report_ext.csv"
    )
    plot_bars(metric_name, stats, out_dir / f"{metric_name}_bars.png")
    return stats


# ---------- high-level orchestrator ----------

def run_report_pretrain_groups(
    group_dir: Path = Path("misc/pretrain_groups"),
    out_summary_json: Path = Path("misc/pretrain_groups/summary_report_ext.json"),
    out_summary_csv: Path  = Path("misc/pretrain_groups/summary_report_ext.csv"),
    # default artifact locations
    mmd_matrix: Path = Path("results/mmd/mmd_matrix.npy"),
    mmd_subjects: Path = Path("results/mmd/mmd_subjects.json"),
    wass_matrix: Path = Path("results/distances/wasserstein_matrix.npy"),
    dtw_matrix: Path  = Path("results/distances/dtw_matrix.npy"),
    dist_subjects: Path = Path("results/distances/subjects.json"),
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Run report for {mmd, wasserstein, dtw} using 10人グループ (friendly/hard) in `group_dir`.
    Writes per-metric json/csv/png into group_dir, plus combined summary json/csv.
    Returns a nested dict: metric -> {'friendly': {...}, 'hard': {...}}.
    """
    metrics = [
        ("mmd",         mmd_matrix,  mmd_subjects),
        ("wasserstein", wass_matrix, dist_subjects),
        ("dtw",         dtw_matrix,  dist_subjects),
    ]

    all_stats: Dict[str, Dict[str, Dict[str, float]]] = {}
    for name, mpath, spath in metrics:
        gf = group_dir / f"{name}_friendly.txt"
        gh = group_dir / f"{name}_hard.txt"
        if not (gf.exists() and gh.exists()):
            raise FileNotFoundError(f"Missing group files for {name}: {gf} / {gh}")
        all_stats[name] = report_for_metric(
            metric_name=name,
            matrix_path=mpath,
            subjects_path=spath,
            group_friendly_file=gf,
            group_hard_file=gh,
            out_dir=group_dir,
        )

    # combined JSON
    save_json(all_stats, out_summary_json)

    # combined CSV
    rows = []
    for metric, st in all_stats.items():
        for kind in ("friendly", "hard"):
            rows.append({
                "metric": metric,
                "group": kind,
                "intra": st[kind]["intra"],
                "inter": st[kind]["inter"],
                "nn":    st[kind]["nn"],
            })
    pd.DataFrame(rows).to_csv(out_summary_csv, index=False)
    return all_stats

