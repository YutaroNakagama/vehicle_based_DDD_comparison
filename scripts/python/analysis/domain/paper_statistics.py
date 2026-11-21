#!/usr/bin/env python3
"""Statistics of the paper that are not produced by the other analysis scripts.

Reads the per-evaluation metric tables (1,512 rows: 126 factor cells x 12 seeds) and
prints:

* the interaction index S_{M,R} with its 95 % percentile bootstrap interval
  (B = 2,000 resamples of the seeds) and the M-R dominance ratio
  (S_M + S_R + S_{M,R}) / (1 - S_eps);
* Cliff's delta between the levels of the training mode M, computed on the seed
  means over the 3 x 2 x 7 combinations of the other factors;
* Spearman's rank correlation of the ranking of the rebalancing methods between
  the levels of M and between the levels of the distance metric D;
* the seed convergence: the standard deviation of the rank of each rebalancing
  method over all k-element subsets of the 12 seeds, k = 3, 5, 7, 9, 11.

The performance table and the total/first-order Sobol indices are produced by
``compute_paper_table_may2026.py`` and ``sensitivity_analysis_exp2.py``.

Usage:
    python scripts/python/analysis/domain/paper_statistics.py [--bootstrap 2000]
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CSV = PROJECT_ROOT / "results/analysis/exp2_domain_shift/figures/csv/split2"
FILES = {
    "BL": CSV / "baseline_NEW_may2026/baseline_domain_split2_metrics_NEW_may2026.csv",
    "RUS": CSV / "undersample_rus_NEW_may2026/undersample_rus_split2_metrics_NEW_may2026.csv",
    "SMOTE": CSV / "smote_plain_NEW_may2026/smote_plain_split2_metrics_NEW_may2026.csv",
    "SW-SMOTE": CSV / "sw_smote_NEW_may2026/sw_smote_split2_metrics_NEW_may2026.csv",
}
SEEDS = [0, 1, 3, 7, 13, 42, 123, 256, 512, 999, 1337, 2024]
MODES = {"source_only": "Cross", "target_only": "Within", "mixed": "Mixed"}
DISTS = {"dtw": "DTW", "mmd": "MMD", "wasserstein": "W1"}
METRICS = {"f2": "F2", "auc": "AUROC", "auc_pr": "AUPRC"}
R_ORDER = ["BL", "RUS 0.1", "RUS 0.5", "SMOTE 0.1", "SMOTE 0.5", "SW-SMOTE 0.1", "SW-SMOTE 0.5"]
FACTORS = ["D", "G", "M", "R"]


def load() -> pd.DataFrame:
    parts = []
    for family, path in FILES.items():
        d = pd.read_csv(path)
        d["R"] = "BL" if family == "BL" else family + " " + d["ratio"].map(lambda r: f"{r:g}")
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df = df[df["seed"].isin(SEEDS)].copy()
    df["M"] = df["mode"].map(MODES)
    df["D"] = df["distance"].map(DISTS)
    df["G"] = df["level"].map({"in_domain": "In", "out_domain": "Out"})
    counts = df.groupby(FACTORS).size()
    if len(df) != 1512 or len(counts) != 126 or counts.min() != 12 or counts.max() != 12:
        raise SystemExit(f"unexpected layout: {len(df)} rows, {len(counts)} cells")
    return df


def sobol(d: pd.DataFrame, metric: str) -> dict:
    """Plug-in Sobol indices of the balanced design (Appendix of the paper)."""
    y = d[metric].to_numpy()
    var = y.var()
    hbar = d.groupby(FACTORS)[metric].transform("mean").to_numpy()
    var_eps = ((y - hbar) ** 2).mean()
    grand = hbar.mean()
    tmp = d[FACTORS].assign(_h=hbar)
    closed = {}
    for k in range(1, 5):
        for sub in itertools.combinations(FACTORS, k):
            cm = tmp.groupby(list(sub))["_h"].transform("mean").to_numpy()
            closed[sub] = ((cm - grand) ** 2).mean()
    v = {}
    for k in range(1, 5):
        for sub in itertools.combinations(FACTORS, k):
            v[sub] = closed[sub] - sum(v[s] for j in range(1, k) for s in itertools.combinations(sub, j))
    s = {key: val / var for key, val in v.items()}
    eps = var_eps / var
    return {
        "S_M": s[("M",)], "S_R": s[("R",)], "S_MR": s[("M", "R")], "S_eps": eps,
        "dominance": (s[("M",)] + s[("R",)] + s[("M", "R")]) / (1 - eps),
    }


def bootstrap_smr(df: pd.DataFrame, metric: str, n_boot: int, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    by_seed = {s: g for s, g in df.groupby("seed")}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(SEEDS, size=len(SEEDS), replace=True)
        bd = pd.concat([by_seed[s].assign(seed=i) for i, s in enumerate(pick)], ignore_index=True)
        vals.append(sobol(bd, metric)["S_MR"])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def cliff(a, b) -> float:
    a = np.asarray(a)[:, None]
    b = np.asarray(b)[None, :]
    return float(((a > b).sum() - (a < b).sum()) / (a.size * b.size))


def magnitude(delta: float) -> str:
    x = abs(delta)
    return "N" if x < 0.147 else "S" if x < 0.33 else "M" if x < 0.474 else "L"


def rank_desc(values) -> np.ndarray:
    """rank = 1 + number of levels that perform better; ties share the rank."""
    values = np.asarray(values)
    return np.array([1 + (values > v).sum() for v in values])


def mean_ranks(df: pd.DataFrame, factor: str, level: str, others: list[str], metric: str) -> np.ndarray:
    ranks = []
    for _, g in df[df[factor] == level].groupby(others):
        ranks.append(rank_desc(g.set_index("R").loc[R_ORDER, metric].to_numpy()))
    return np.mean(ranks, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bootstrap", type=int, default=2000, help="bootstrap resamples (default 2000)")
    args = ap.parse_args()
    df = load()
    print(f"Loaded {len(df)} evaluations (126 cells x 12 seeds)\n")

    print("== Interaction index and M-R dominance ratio ==")
    for m, name in METRICS.items():
        s = sobol(df, m)
        lo, hi = bootstrap_smr(df, m, args.bootstrap)
        print(f"{name:6s} S_MR={s['S_MR']:.3f} [{lo:.3f}, {hi:.3f}]  S_eps={s['S_eps']:.3f}  "
              f"dominance={s['dominance']:.3f}")

    print("\n== Cliff's delta between the levels of M (seed means, 42 combinations) ==")
    cells = df.groupby(FACTORS)[list(METRICS)].mean().reset_index()
    for x, y in [("Within", "Mixed"), ("Cross", "Within"), ("Cross", "Mixed")]:
        row = []
        for m in METRICS:
            a = cells[cells.M == x].sort_values(["D", "G", "R"])[m]
            b = cells[cells.M == y].sort_values(["D", "G", "R"])[m]
            d = cliff(a, b)
            row.append(f"{d:+.2f} ({magnitude(d)})")
        print(f"{x}--{y}: " + "  ".join(f"{METRICS[m]}={r}" for m, r in zip(METRICS, row)))

    for factor, levels, others in [("M", ["Within", "Mixed", "Cross"], ["D", "G", "seed"]),
                                   ("D", ["DTW", "MMD", "W1"], ["G", "M", "seed"])]:
        print(f"\n== Spearman's rank correlation between the levels of {factor} ==")
        for a, b in itertools.combinations(levels, 2):
            row = []
            for m in METRICS:
                rho, p = spearmanr(mean_ranks(df, factor, a, others, m), mean_ranks(df, factor, b, others, m))
                row.append(f"{METRICS[m]}={rho:+.2f} (p={p:.3f})")
            print(f"{a}--{b}: " + "  ".join(row))

    print("\n== Seed convergence: sigma_rank over k-element subsets of the 12 seeds ==")
    for m, name in METRICS.items():
        per_seed = df.groupby(["seed", "R"])[m].mean().unstack()[R_ORDER]
        print(f"-- {name}")
        for k in (3, 5, 7, 9, 11):
            ranks = [rank_desc(per_seed.loc[list(sub)].mean().to_numpy())
                     for sub in itertools.combinations(SEEDS, k)]
            sd = np.array(ranks).std(axis=0)
            per = " ".join(f"{r}={v:.2f}" for r, v in zip(R_ORDER, sd))
            print(f"   k={k:2d}: mean={sd.mean():.3f} max={sd.max():.3f} | {per}")


if __name__ == "__main__":
    main()
