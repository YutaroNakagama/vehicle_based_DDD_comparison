#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
normalized_distance_analysis.py
================================
Compare domain groupings computed on UNNORMALIZED vs STANDARDIZED features.

Purpose:
  - Address the reviewer concern: "If lane offset dominates at O(10^3),
    the H3 finding (distance metrics are equivalent) is trivially true."
  - Recompute all three distance matrices (MMD, DTW, Wasserstein) after
    StandardScaler normalisation and compare:
      1. Rank concordance (Spearman ρ) between metrics (normalised)
      2. Group membership overlap (normalised vs unnormalised)
      3. Whether metrics differentiate after normalisation
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.domain.distance import (
    _extract_features,
    _compute_distance_matrix,
)

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "common"
SUBJECT_LIST = PROJECT_ROOT / "config" / "subjects" / "subject_list.txt"
GROUP_SIZE = 29
METRICS = ["mmd", "wasserstein", "dtw"]

OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis" / "exp2_domain_shift"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def rank_subjects(matrix: np.ndarray) -> np.ndarray:
    masked = matrix.copy()
    np.fill_diagonal(masked, np.nan)
    mean_d = np.nanmean(masked, axis=1)
    return np.argsort(np.argsort(-mean_d))


def get_domain_groups(matrix: np.ndarray) -> dict:
    n = matrix.shape[0]
    masked = matrix.copy()
    np.fill_diagonal(masked, np.nan)
    mean_d = np.nanmean(masked, axis=1)
    sorted_idx = np.argsort(-mean_d)
    out_domain = set(sorted_idx[:GROUP_SIZE].tolist())
    half = GROUP_SIZE // 2
    mid_start = n // 2 - half
    mid_domain = set(sorted_idx[mid_start:mid_start + GROUP_SIZE].tolist())
    in_domain = set(sorted_idx[-GROUP_SIZE:].tolist())
    return {"out_domain": out_domain, "mid_domain": mid_domain, "in_domain": in_domain}


def align_feature_dims(features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Truncate all subjects to the minimum common column count."""
    min_cols = min(X.shape[1] for X in features.values())
    return {s: X[:, :min_cols] for s, X in features.items()}


def standardize_features(features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Fit StandardScaler on pooled features and transform each subject."""
    features = align_feature_dims(features)
    all_data = np.vstack(list(features.values()))
    scaler = StandardScaler()
    scaler.fit(all_data)
    return {s: scaler.transform(X) for s, X in features.items()}


def feature_scale_summary(features: dict[str, np.ndarray], label: str) -> list[str]:
    """Return lines describing feature scale statistics."""
    all_data = np.vstack(list(features.values()))
    col_std = np.std(all_data, axis=0)
    col_range = np.ptp(all_data, axis=0)
    lines = [
        f"\n### Feature Scale Summary ({label})\n",
        f"- Dimensions: {all_data.shape[1]}",
        f"- Std  — min: {col_std.min():.4f}, median: {np.median(col_std):.4f}, "
        f"max: {col_std.max():.4f}, ratio(max/min): {col_std.max()/max(col_std.min(), 1e-12):.1f}",
        f"- Range — min: {col_range.min():.4f}, median: {np.median(col_range):.4f}, "
        f"max: {col_range.max():.4f}, ratio(max/min): {col_range.max()/max(col_range.min(), 1e-12):.1f}",
        "",
    ]
    return lines


def main():
    subjects = [l.strip() for l in SUBJECT_LIST.read_text().splitlines() if l.strip()]
    print(f"Loading features for {len(subjects)} subjects...")
    raw_features = _extract_features(subjects, DATA_DIR)
    loaded = [s for s in subjects if s in raw_features]
    print(f"Loaded: {len(loaded)} subjects, dims={next(iter(raw_features.values())).shape[1]}")

    # Align to common column count
    raw_features = align_feature_dims(raw_features)
    print(f"Aligned to {next(iter(raw_features.values())).shape[1]} common columns")

    # Standardize
    print("Standardizing features...")
    std_features = standardize_features(raw_features)

    # Compute distance matrices: unnormalised and normalised
    results = {}
    for label, feats in [("raw", raw_features), ("standardized", std_features)]:
        results[label] = {}
        for m in METRICS:
            print(f"  Computing {m.upper()} ({label})...")
            mat, subj_list = _compute_distance_matrix(feats, m, n_jobs=4)
            results[label][m] = {"matrix": mat, "subjects": subj_list}

    subj_list = results["raw"]["mmd"]["subjects"]

    # ── Build report ──
    lines = []
    w = lines.append

    w("# Normalised vs Unnormalised Distance Analysis\n")
    w(f"**Subjects**: {len(subj_list)}  |  **Group size**: {GROUP_SIZE}  |  "
      f"**Features**: {next(iter(raw_features.values())).shape[1]}\n")

    # Feature scale summary
    lines.extend(feature_scale_summary(raw_features, "Raw"))
    lines.extend(feature_scale_summary(std_features, "Standardized"))

    # --- Section 1: Distance matrix summary ---
    w("## 1. Distance Matrix Summary\n")
    w("| Normalisation | Metric | Mean | Std | Min | Max |")
    w("|:------------:|--------|:----:|:---:|:---:|:---:|")
    for label in ["raw", "standardized"]:
        for m in METRICS:
            mat = results[label][m]["matrix"]
            off = mat[np.triu_indices(len(subj_list), k=1)]
            w(f"| {label} | {m.upper()} | {np.mean(off):.4f} | {np.std(off):.4f} | "
              f"{np.min(off):.6f} | {np.max(off):.4f} |")
    w("")

    # --- Section 2: Within-condition rank concordance ---
    w("## 2. Rank Concordance Between Metrics (Spearman ρ)\n")
    w("| Normalisation | Pair | Spearman ρ | p-value |")
    w("|:------------:|------|:----------:|:-------:|")
    pairs = [("mmd", "wasserstein"), ("mmd", "dtw"), ("wasserstein", "dtw")]
    for label in ["raw", "standardized"]:
        ranks = {m: rank_subjects(results[label][m]["matrix"]) for m in METRICS}
        for m1, m2 in pairs:
            rho, pval = stats.spearmanr(ranks[m1], ranks[m2])
            w(f"| {label} | {m1.upper()} vs {m2.upper()} | {rho:.4f} | {pval:.2e} |")
    w("")

    # --- Section 3: Group membership overlap (normalised vs raw) ---
    w("## 3. Group Overlap: Standardized vs Raw (Same Metric)\n")
    w("How many subjects stay in the same domain group after normalisation?\n")
    w("| Metric | Domain | Overlap | Jaccard |")
    w("|--------|--------|:-------:|:-------:|")
    for m in METRICS:
        raw_groups = get_domain_groups(results["raw"][m]["matrix"])
        std_groups = get_domain_groups(results["standardized"][m]["matrix"])
        for domain in ["out_domain", "mid_domain", "in_domain"]:
            g_raw = raw_groups[domain]
            g_std = std_groups[domain]
            overlap = len(g_raw & g_std)
            jaccard = len(g_raw & g_std) / len(g_raw | g_std)
            w(f"| {m.upper()} | {domain} | {overlap}/{GROUP_SIZE} ({overlap/GROUP_SIZE*100:.0f}%) | {jaccard:.3f} |")
    w("")

    # --- Section 4: Cross-metric group overlap (standardised) ---
    w("## 4. Cross-Metric Group Overlap (Standardized)\n")
    w("Do metrics DIFFERENTIATE after normalisation?\n")
    w("| Domain | Pair | Overlap | Jaccard |")
    w("|--------|------|:-------:|:-------:|")
    for domain in ["out_domain", "in_domain"]:
        std_groups_all = {m: get_domain_groups(results["standardized"][m]["matrix"]) for m in METRICS}
        for m1, m2 in pairs:
            g1 = std_groups_all[m1][domain]
            g2 = std_groups_all[m2][domain]
            overlap = len(g1 & g2)
            jaccard = len(g1 & g2) / len(g1 | g2)
            w(f"| {domain} | {m1.upper()} vs {m2.upper()} | {overlap}/{GROUP_SIZE} ({overlap/GROUP_SIZE*100:.0f}%) | {jaccard:.3f} |")
    w("")

    # --- Section 5: Subject switching summary ---
    w("## 5. Subject Switching Summary\n")
    for label in ["raw", "standardized"]:
        all_groups = {m: get_domain_groups(results[label][m]["matrix"]) for m in METRICS}
        switch = 0
        for i in range(len(subj_list)):
            assignments = set()
            for m in METRICS:
                for domain in ["out_domain", "mid_domain", "in_domain"]:
                    if i in all_groups[m][domain]:
                        assignments.add(domain)
            if len(assignments) > 1:
                switch += 1
        w(f"- **{label}**: {switch}/{len(subj_list)} subjects switch groups across metrics "
          f"({switch/len(subj_list)*100:.1f}%)")
    w("")

    # --- Section 6: Conclusion ---
    w("## 6. Interpretation\n")

    # Compute key numbers for auto-conclusion
    raw_rho_vals = []
    std_rho_vals = []
    for m1, m2 in pairs:
        raw_ranks = {m: rank_subjects(results["raw"][m]["matrix"]) for m in METRICS}
        rho_raw, _ = stats.spearmanr(raw_ranks[m1], raw_ranks[m2])
        raw_rho_vals.append(rho_raw)
        std_ranks = {m: rank_subjects(results["standardized"][m]["matrix"]) for m in METRICS}
        rho_std, _ = stats.spearmanr(std_ranks[m1], std_ranks[m2])
        std_rho_vals.append(rho_std)

    w(f"- Raw ρ range: {min(raw_rho_vals):.3f}–{max(raw_rho_vals):.3f}")
    w(f"- Standardized ρ range: {min(std_rho_vals):.3f}–{max(std_rho_vals):.3f}")
    concordance_change = np.mean(std_rho_vals) - np.mean(raw_rho_vals)
    w(f"- Mean ρ change: {concordance_change:+.3f}")
    w("")
    if concordance_change < -0.1:
        w("**Normalisation REDUCES cross-metric concordance** → metrics capture different "
          "distributional properties when lane offset dominance is removed. The H3 finding "
          "is partially explained by feature scale heterogeneity.")
    elif concordance_change > 0.1:
        w("**Normalisation INCREASES cross-metric concordance** → the underlying subject "
          "structure is robust to feature scaling; metric equivalence is not an artifact "
          "of lane offset dominance.")
    else:
        w("**Normalisation has MINIMAL effect on cross-metric concordance** → metric "
          "equivalence is robust to feature scaling and NOT solely driven by lane offset dominance.")
    w("")

    # Write report
    report = "\n".join(lines)
    out_path = OUTPUT_DIR / "normalized_distance_analysis.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved to: {out_path}")
    print(report)


if __name__ == "__main__":
    main()
