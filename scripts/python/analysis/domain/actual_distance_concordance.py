#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
actual_distance_concordance.py
===============================
Compute domain groupings using the ACTUAL distance metrics from the experiment
(MMD, DTW, Wasserstein) and compare rank concordance and group membership overlap.

This corrects the earlier analysis that incorrectly used Euclidean/Manhattan/Cosine
as proxies.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.domain.distance import (
    compute_mmd,
    _load_features_one,
    _extract_features,
    _compute_distance_matrix,
)

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "common"
SUBJECT_LIST = PROJECT_ROOT / "config" / "subjects" / "subject_list.txt"
GROUP_SIZE = 29

OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis" / "exp2_domain_shift"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def rank_subjects(matrix: np.ndarray) -> np.ndarray:
    """Return descending rank of each subject by mean distance to all others."""
    masked = matrix.copy()
    np.fill_diagonal(masked, np.nan)
    mean_d = np.nanmean(masked, axis=1)
    return np.argsort(np.argsort(-mean_d))  # rank 0 = highest mean distance


def get_domain_groups(matrix: np.ndarray) -> dict:
    """Assign domain groups exactly as clustering_projection_ranked.py does."""
    n = matrix.shape[0]
    masked = matrix.copy()
    np.fill_diagonal(masked, np.nan)
    mean_d = np.nanmean(masked, axis=1)

    sorted_idx = np.argsort(-mean_d)  # descending

    out_domain = set(sorted_idx[:GROUP_SIZE].tolist())
    mid_start = n // 2 - GROUP_SIZE // 2
    mid_domain = set(sorted_idx[mid_start:mid_start + GROUP_SIZE].tolist())
    in_domain = set(sorted_idx[-GROUP_SIZE:].tolist())

    return {"out_domain": out_domain, "mid_domain": mid_domain, "in_domain": in_domain}


def main():
    subjects = [l.strip() for l in SUBJECT_LIST.read_text().splitlines() if l.strip()]
    print(f"Loading features for {len(subjects)} subjects...")

    features = _extract_features(subjects, DATA_DIR)
    loaded = [s for s in subjects if s in features]
    print(f"Loaded: {len(loaded)} subjects")

    # Compute the three actual distance matrices
    metrics = ["mmd", "wasserstein", "dtw"]
    matrices = {}
    subj_orders = {}

    for m in metrics:
        print(f"Computing {m.upper()} distance matrix...")
        mat, subj_list = _compute_distance_matrix(features, m, n_jobs=4)
        matrices[m] = mat
        subj_orders[m] = subj_list
        print(f"  {m.upper()} done: shape={mat.shape}, "
              f"mean={np.nanmean(mat):.4f}, max={np.nanmax(mat):.4f}")

    # Ensure subject order is consistent
    assert subj_orders["mmd"] == subj_orders["wasserstein"] == subj_orders["dtw"], \
        "Subject order mismatch between metrics!"
    subj_list = subj_orders["mmd"]
    n_subjects = len(subj_list)

    # Compute ranks
    ranks = {m: rank_subjects(matrices[m]) for m in metrics}

    # Compute domain groups
    groups = {m: get_domain_groups(matrices[m]) for m in metrics}

    # ── Build report ──
    lines = []
    w = lines.append

    w("# Domain Group Concordance: MMD vs DTW vs Wasserstein\n")
    w(f"**Subjects**: {n_subjects}  ")
    w(f"**GROUP_SIZE**: {GROUP_SIZE}  ")
    w(f"**Feature dimensions**: {next(iter(features.values())).shape[1]}\n")

    # Distance matrix summary
    w("## 1. Distance Matrix Summary\n")
    w("| Metric | Mean | Std | Min (off-diag) | Max |")
    w("|--------|:----:|:---:|:--------------:|:---:|")
    for m in metrics:
        mat = matrices[m]
        off = mat[np.triu_indices(n_subjects, k=1)]
        w(f"| {m.upper()} | {np.mean(off):.4f} | {np.std(off):.4f} | "
          f"{np.min(off):.4f} | {np.max(off):.4f} |")
    w("")

    # Rank concordance (Spearman)
    w("## 2. Subject Rank Concordance (Spearman ρ)\n")
    w("Ranking = mean distance to all other subjects (descending = out_domain first).\n")
    w("| Metric Pair | Spearman ρ | p-value |")
    w("|-------------|:----------:|:-------:|")
    pairs = [("mmd", "wasserstein"), ("mmd", "dtw"), ("wasserstein", "dtw")]
    for m1, m2 in pairs:
        rho, pval = stats.spearmanr(ranks[m1], ranks[m2])
        w(f"| {m1.upper()} vs {m2.upper()} | {rho:.4f} | {pval:.2e} |")
    w("")

    # Kendall's tau (alternative rank correlation)
    w("## 3. Kendall τ Rank Correlation\n")
    w("| Metric Pair | Kendall τ | p-value |")
    w("|-------------|:---------:|:-------:|")
    for m1, m2 in pairs:
        tau, pval = stats.kendalltau(ranks[m1], ranks[m2])
        w(f"| {m1.upper()} vs {m2.upper()} | {tau:.4f} | {pval:.2e} |")
    w("")

    # Domain group membership overlap
    w("## 4. Domain Group Membership Overlap\n")
    w(f"Each group has {GROUP_SIZE} subjects.\n")

    for domain in ["out_domain", "mid_domain", "in_domain"]:
        w(f"### {domain}\n")
        w(f"| Pair | Overlap | Jaccard |")
        w(f"|------|:-------:|:-------:|")
        for m1, m2 in pairs:
            g1 = groups[m1][domain]
            g2 = groups[m2][domain]
            overlap = len(g1 & g2)
            jaccard = len(g1 & g2) / len(g1 | g2)
            w(f"| {m1.upper()} ∩ {m2.upper()} | {overlap}/{GROUP_SIZE} ({overlap/GROUP_SIZE*100:.0f}%) | {jaccard:.3f} |")

        all_agree = groups["mmd"][domain] & groups["wasserstein"][domain] & groups["dtw"][domain]
        w(f"\n**All three agree**: {len(all_agree)}/{GROUP_SIZE} ({len(all_agree)/GROUP_SIZE*100:.0f}%)\n")

    # Per-group mean distance comparison
    w("## 5. Mean Distance by Domain Group\n")
    w("Do in_domain subjects have lower mean distance than out_domain subjects? (Sanity check)\n")
    w("| Metric | out_domain mean_d | mid_domain mean_d | in_domain mean_d | out/in ratio |")
    w("|--------|:-----------------:|:-----------------:|:----------------:|:------------:|")
    for m in metrics:
        mat = matrices[m].copy()
        np.fill_diagonal(mat, np.nan)
        mean_d = np.nanmean(mat, axis=1)

        out_idx = list(groups[m]["out_domain"])
        mid_idx = list(groups[m]["mid_domain"])
        in_idx = list(groups[m]["in_domain"])

        out_mean = np.mean(mean_d[out_idx])
        mid_mean = np.mean(mean_d[mid_idx])
        in_mean = np.mean(mean_d[in_idx])
        ratio = out_mean / in_mean if in_mean > 0 else float("inf")

        w(f"| {m.upper()} | {out_mean:.4f} | {mid_mean:.4f} | {in_mean:.4f} | {ratio:.2f} |")
    w("")

    # Subjects that switch groups across metrics
    w("## 6. Subjects That Switch Groups\n")
    w("Subjects assigned to different domain groups depending on distance metric.\n")

    switch_count = 0
    switch_details = []
    for i, s in enumerate(subj_list):
        assignments = {}
        for m in metrics:
            for domain in ["out_domain", "mid_domain", "in_domain"]:
                if i in groups[m][domain]:
                    assignments[m] = domain
                    break
            else:
                assignments[m] = "Other"

        unique_assignments = set(assignments.values())
        if len(unique_assignments) > 1:
            switch_count += 1
            switch_details.append((s, assignments))

    w(f"**Total subjects that switch**: {switch_count}/{n_subjects}\n")
    if switch_details:
        w("| Subject | MMD | Wasserstein | DTW |")
        w("|---------|-----|-------------|-----|")
        for s, a in switch_details:
            w(f"| {s} | {a.get('mmd', '-')} | {a.get('wasserstein', '-')} | {a.get('dtw', '-')} |")
    else:
        w("All subjects have identical group assignments across all three metrics.\n")
    w("")

    # Detailed rank comparison: top 10 and bottom 10
    w("## 7. Rank Comparison: Top 10 (most out_domain) and Bottom 10 (most in_domain)\n")
    w("### Top 10 (highest mean distance)\n")
    w("| Rank | MMD | Wasserstein | DTW |")
    w("|:----:|-----|-------------|-----|")
    for rank_pos in range(10):
        row = [str(rank_pos + 1)]
        for m in metrics:
            idx = np.where(ranks[m] == rank_pos)[0][0]
            row.append(subj_list[idx])
        w(f"| {' | '.join(row)} |")
    w("")

    w("### Bottom 10 (lowest mean distance)\n")
    w("| Rank | MMD | Wasserstein | DTW |")
    w("|:----:|-----|-------------|-----|")
    for rank_pos in range(n_subjects - 10, n_subjects):
        row = [str(rank_pos + 1)]
        for m in metrics:
            idx = np.where(ranks[m] == rank_pos)[0][0]
            row.append(subj_list[idx])
        w(f"| {' | '.join(row)} |")
    w("")

    # Summary
    w("## 8. Summary\n")

    # Compute summary stats
    rho_mw, _ = stats.spearmanr(ranks["mmd"], ranks["wasserstein"])
    rho_md, _ = stats.spearmanr(ranks["mmd"], ranks["dtw"])
    rho_wd, _ = stats.spearmanr(ranks["wasserstein"], ranks["dtw"])
    min_rho = min(rho_mw, rho_md, rho_wd)
    max_rho = max(rho_mw, rho_md, rho_wd)

    all_out = groups["mmd"]["out_domain"] & groups["wasserstein"]["out_domain"] & groups["dtw"]["out_domain"]
    all_in = groups["mmd"]["in_domain"] & groups["wasserstein"]["in_domain"] & groups["dtw"]["in_domain"]

    w(f"1. **Rank concordance**: Spearman ρ ranges from {min_rho:.4f} to {max_rho:.4f} "
      f"across the three actual distance metrics (MMD, DTW, Wasserstein).\n")
    w(f"2. **out_domain overlap**: {len(all_out)}/{GROUP_SIZE} "
      f"({len(all_out)/GROUP_SIZE*100:.0f}%) subjects are classified as out_domain by all three metrics.\n")
    w(f"3. **in_domain overlap**: {len(all_in)}/{GROUP_SIZE} "
      f"({len(all_in)/GROUP_SIZE*100:.0f}%) subjects are classified as in_domain by all three metrics.\n")
    w(f"4. **Total group switchers**: {switch_count}/{n_subjects} "
      f"({switch_count/n_subjects*100:.1f}%) subjects change domain assignment depending on distance metric.\n")

    report = "\n".join(lines)
    out_path = OUTPUT_DIR / "actual_distance_concordance.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved to: {out_path}")
    print("\n" + report)


if __name__ == "__main__":
    main()
