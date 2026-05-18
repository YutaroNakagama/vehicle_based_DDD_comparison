#!/usr/bin/env python3
"""Seed-count rank-stability analysis for Exp3 prior-method replication.

Computes sigma_rank(k) — the average standard deviation of each condition's
rank position across all C(N_seeds, k) random k-seed subsets — as a function
of k.  Monotonic convergence toward zero confirms that N_seeds is sufficient
to determine condition rankings.

Mirrors the Exp2 TIV2026 analysis (fig8_seed_convergence) but applied to
Exp3 Lstm SW-SMOTE results (15 seeds x 2 ratios x 2 domain groups = 4
conditions, each averaged over 3 distance metrics).

Usage (from repo root, WSL2 venv):
    /home/ynakagama/.venv_tf_gpu/bin/python \\
        scripts/python/analysis/domain/seed_rank_stability_exp3.py

Outputs:
    results/analysis/exp3_prior_research/figures/
        seed_convergence_Lstm_swsmote.pdf
        seed_convergence_Lstm_swsmote.png
        seed_convergence_Lstm_swsmote.csv
"""
from __future__ import annotations

import glob
import json
import logging
import re
import sys
from itertools import combinations
from pathlib import Path

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[4]
EVAL_BASE = REPO / "results" / "outputs" / "evaluation"
OUT_DIR = REPO / "results" / "analysis" / "exp3_prior_research" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = {
    "AUROC":  "roc_auc",
    "AUPRC":  "auc_pr",
    "F2":     "f2_thr",
}

# Condition label → (domain, ratio)
CONDITIONS = {
    "in-domain\nr=0.3":  ("in_domain",  "0.3"),
    "in-domain\nr=0.5":  ("in_domain",  "0.5"),
    "out-domain\nr=0.3": ("out_domain", "0.3"),
    "out-domain\nr=0.5": ("out_domain", "0.5"),
}

DISTANCES = ["mmd", "dtw", "wasserstein"]

# Regex for Lstm SW-SMOTE within-domain eval JSONs
_PAT = re.compile(
    r"eval_results_Lstm_domain_train_prior_Lstm_imbalv3_knn_"
    r"(?P<distance>mmd|dtw|wasserstein)_"
    r"(?P<domain>in_domain|out_domain)"
    r"(?:_domain_train)?(?:_split2)?(?:_subjectwise)?"
    r"_ratio(?P<ratio>[0-9.]+)"
    r"_s(?P<seed>\d+)_within\.json$"
)

# ---------------------------------------------------------------------------

def load_lstm_swsmote() -> pd.DataFrame:
    """Load all Lstm SW-SMOTE within-domain eval JSONs; return tidy DataFrame."""
    rows = []
    for path in glob.glob(
        str(EVAL_BASE / "Lstm" / "**" / "*.json"), recursive=True
    ):
        name = Path(path).name
        m = _PAT.match(name)
        if not m:
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception as e:
            log.warning("Skipping %s: %s", path, e)
            continue
        row = {
            "distance":  m.group("distance"),
            "domain":    m.group("domain"),
            "ratio":     m.group("ratio"),
            "seed":      int(m.group("seed")),
            "timestamp": d.get("timestamp", ""),
        }
        for metric_label, json_key in METRICS.items():
            row[metric_label] = d.get(json_key, np.nan)
        rows.append(row)

    df = pd.DataFrame(rows)
    log.info(
        "Loaded %d Lstm SW-SMOTE within JSONs | seeds=%s",
        len(df),
        sorted(df["seed"].unique()),
    )
    return df


def sigma_rank(
    df: pd.DataFrame,
    metric: str,
    seeds: list[int],
    conditions: dict[str, tuple[str, str]],
    max_combinations: int = 10_000,
) -> dict[int, tuple[float, float]]:
    """Return {k: (mean_sigma_rank, max_sigma_rank)} for k=1..N.

    For each k, samples up to max_combinations subsets uniformly if C(N,k)
    exceeds the budget; otherwise enumerates exactly.
    """
    N = len(seeds)
    results: dict[int, tuple[float, float]] = {}
    rng = np.random.default_rng(0)

    for k in range(1, N + 1):
        total_combos = math.comb(N, k)
        if total_combos <= max_combinations:
            subsets = list(combinations(seeds, k))
        else:
            # Sample without replacement from the seed list
            subsets = [
                tuple(rng.choice(seeds, k, replace=False).tolist())
                for _ in range(max_combinations)
            ]

        # ranks[i][c] = rank of condition c in subset i  (1 = best AUROC)
        n_cond = len(conditions)
        rank_matrix = np.empty((len(subsets), n_cond), dtype=float)

        for i, subset in enumerate(subsets):
            means = []
            for cond_label, (domain, ratio) in conditions.items():
                mask = (
                    (df["domain"] == domain)
                    & (df["ratio"] == ratio)
                    & (df["seed"].isin(subset))
                )
                val = df.loc[mask, metric].mean()
                means.append(val if not np.isnan(val) else 0.0)
            # Higher metric = better; rank 1 = highest
            order = np.argsort(means)[::-1]
            ranks = np.empty(n_cond, dtype=float)
            ranks[order] = np.arange(1, n_cond + 1)
            rank_matrix[i] = ranks

        per_cond_std = rank_matrix.std(axis=0, ddof=0)
        results[k] = (float(per_cond_std.mean()), float(per_cond_std.max()))

    return results


def bootstrap_ci_width(
    df: pd.DataFrame,
    metric: str,
    seeds: list[int],
    conditions: dict[str, tuple[str, str]],
    n_resamples: int = 2_000,
) -> dict[str, list[tuple[int, float]]]:
    """Return {cond_label: [(k, ci_width), ...]} for k=1..N.

    Uses percentile bootstrap (B=2000) over seed subsets to estimate how CI
    width shrinks as k grows.
    """
    N = len(seeds)
    rng = np.random.default_rng(42)
    out: dict[str, list] = {c: [] for c in conditions}

    for cond_label, (domain, ratio) in conditions.items():
        cond_df = df[(df["domain"] == domain) & (df["ratio"] == ratio)]
        # One mean per seed (averaged over distances)
        seed_means = (
            cond_df.groupby("seed")[metric].mean().reindex(seeds).values
        )

        for k in range(1, N + 1):
            # Bootstrap: pick k seeds with replacement, take mean
            draws = seed_means[
                rng.integers(0, N, size=(n_resamples, k))
            ].mean(axis=1)
            lo, hi = np.nanpercentile(draws, [2.5, 97.5])
            out[cond_label].append((k, hi - lo))

    return out


# ---------------------------------------------------------------------------

def plot_convergence(
    sigma_results: dict[str, dict[int, tuple[float, float]]],
    seeds: list[int],
    out_prefix: str,
) -> None:
    """3-row figure (one per metric) matching Exp2 fig8 style."""
    metric_names = list(sigma_results.keys())
    N = len(seeds)
    ks = list(range(1, N + 1))

    fig, axes = plt.subplots(len(metric_names), 1, figsize=(7, 3.2 * len(metric_names)),
                              sharex=True)
    if len(metric_names) == 1:
        axes = [axes]

    cond_labels = list(CONDITIONS.keys())
    colors = plt.cm.tab10(np.linspace(0, 0.7, len(cond_labels)))

    for ax, metric in zip(axes, metric_names):
        res = sigma_results[metric]
        mean_sigma = [res[k][0] for k in ks]
        max_sigma  = [res[k][1] for k in ks]

        ax.plot(ks, mean_sigma, "k-", lw=2, label=r"Mean $\sigma_{\rm rank}$")
        ax.fill_between(ks, 0, max_sigma, alpha=0.15, color="gray",
                        label=r"Max $\sigma_{\rm rank}$")
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.set_ylabel(r"$\sigma_{\rm rank}$", fontsize=11)
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_ylim(-0.02, max(max_sigma) * 1.15 + 0.02)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3)

        # Annotate final value
        ax.annotate(
            f"{mean_sigma[-1]:.3f}",
            xy=(N, mean_sigma[-1]),
            xytext=(N - 1.5, mean_sigma[-1] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.07),
            fontsize=8, color="k",
        )

    axes[-1].set_xlabel("Number of seeds $k$", fontsize=11)
    axes[-1].set_xticks(range(1, N + 1))

    fig.suptitle(
        "Exp3 Lstm SW-SMOTE — Ranking Stability ($\\sigma_{\\rm rank}$)\n"
        "4 conditions: in/out-domain × r=0.3/0.5, averaged over 3 distances",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    for ext in ("pdf", "png"):
        p = f"{out_prefix}.{ext}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        log.info("Saved %s", p)
    plt.close(fig)


def save_csv(
    sigma_results: dict[str, dict[int, tuple[float, float]]],
    seeds: list[int],
    out_prefix: str,
) -> None:
    N = len(seeds)
    rows = []
    for metric, res in sigma_results.items():
        for k in range(1, N + 1):
            rows.append({
                "metric": metric,
                "k": k,
                "mean_sigma_rank": res[k][0],
                "max_sigma_rank":  res[k][1],
            })
    df = pd.DataFrame(rows)
    p = f"{out_prefix}.csv"
    df.to_csv(p, index=False)
    log.info("Saved %s", p)
    return df


# ---------------------------------------------------------------------------

def main() -> None:
    df = load_lstm_swsmote()
    if df.empty:
        log.error("No Lstm SW-SMOTE JSONs found. Check EVAL_BASE path.")
        sys.exit(1)

    seeds = sorted(df["seed"].unique().tolist())
    N = len(seeds)
    log.info("Seeds (%d): %s", N, seeds)

    # Deduplication: for any (seed, distance, domain, ratio) with > 1 row,
    # keep the row with the most recent timestamp.
    df["timestamp"] = pd.to_datetime(df.get("timestamp", "1970-01-01"), errors="coerce")
    df = (
        df.sort_values("timestamp", na_position="first")
        .drop_duplicates(subset=["seed", "distance", "domain", "ratio"], keep="last")
        .reset_index(drop=True)
    )

    # Check coverage: seeds that have at least 1 evaluation per condition
    seed_cond_counts = df.groupby(["seed", "domain", "ratio"]).size()
    complete_seeds = []
    for s in seeds:
        ok = all(
            seed_cond_counts.get((s, dom, rat), 0) >= 1
            for dom, rat in CONDITIONS.values()
        )
        if ok:
            complete_seeds.append(s)
        else:
            missing_conds = [
                f"{dom}/{rat}"
                for dom, rat in CONDITIONS.values()
                if seed_cond_counts.get((s, dom, rat), 0) < 1
            ]
            log.warning("Seed %s missing conditions %s — excluded", s, missing_conds)
    seeds = complete_seeds
    N = len(seeds)
    log.info("Seeds with full 4-condition coverage (%d): %s", N, seeds)

    # ---- sigma_rank for each metric ----------------------------------------
    sigma_results: dict[str, dict] = {}
    for metric_label in METRICS:
        log.info("Computing sigma_rank for %s ...", metric_label)
        sigma_results[metric_label] = sigma_rank(
            df, metric_label, seeds, CONDITIONS, max_combinations=5_000
        )

    # ---- Print summary table ------------------------------------------------
    print("\n=== sigma_rank convergence (mean / max) ===")
    print(f"{'k':>3}  " + "  ".join(f"{m:>20}" for m in sigma_results))
    for k in range(1, N + 1):
        row = f"{k:>3}  "
        for m in sigma_results:
            mu, mx = sigma_results[m][k]
            row += f"  {mu:6.3f} / {mx:6.3f}  "
        print(row)

    # ---- Plots & CSV --------------------------------------------------------
    out_prefix = str(OUT_DIR / "seed_convergence_Lstm_swsmote")
    plot_convergence(sigma_results, seeds, out_prefix)
    csv_df = save_csv(sigma_results, seeds, out_prefix)

    # ---- Conclusion ---------------------------------------------------------
    print("\n=== Conclusion ===")
    for metric_label in METRICS:
        res = sigma_results[metric_label]
        # Find smallest k where mean_sigma < 0.2 (stable enough)
        stable_k = next(
            (k for k in range(1, N + 1) if res[k][0] < 0.2), N
        )
        final_sigma = res[N][0]
        print(
            f"  {metric_label}: sigma_rank at k={N} = {final_sigma:.4f} | "
            f"stable (sigma<0.2) from k={stable_k}"
        )


if __name__ == "__main__":
    main()
