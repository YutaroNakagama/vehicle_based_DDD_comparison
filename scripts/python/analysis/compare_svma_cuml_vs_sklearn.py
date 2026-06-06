"""Compare cuML SvmA validation output against the existing sklearn baseline.

Reads both within JSONs for the same tag and prints roc_auc, auc_pr, recall,
precision, f1_score diffs. Intended as a single-job GO/NoGo gate before
committing to a full 144-job cuML re-run.

Usage:
    python scripts/python/analysis/compare_svma_cuml_vs_sklearn.py \
        --tag prior_SvmA_imbalv3_knn_mmd_in_domain_domain_train_split2_subjectwise_ratio0.3_s42
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EVAL_DIR = REPO / "results" / "outputs" / "evaluation" / "SvmA"

TARGET_METRICS = ["roc_auc", "auc_pr", "accuracy", "f1_pos", "recall_pos", "precision_pos"]
GO_TOLERANCE = 0.02  # AUROC diff threshold


def find_within_jsons(tag: str) -> list[Path]:
    fname = f"eval_results_SvmA_domain_train_{tag}_within.json"
    return sorted(EVAL_DIR.rglob(fname))


def load(p: Path) -> dict:
    return json.loads(p.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    jsons = find_within_jsons(args.tag)
    if len(jsons) < 2:
        print(f"Need >=2 within JSONs for tag={args.tag}, found {len(jsons)}.", file=sys.stderr)
        for j in jsons:
            print(f"  {j}", file=sys.stderr)
        sys.exit(2)

    # newest = cuML (just produced), older = sklearn (baseline)
    jsons.sort(key=lambda p: p.stat().st_mtime)
    sk_path, cu_path = jsons[0], jsons[-1]
    sk, cu = load(sk_path), load(cu_path)

    print(f"sklearn (older): {sk_path}")
    print(f"   mtime: {sk.get('timestamp', '?')}")
    print(f"cuML    (newer): {cu_path}")
    print(f"   mtime: {cu.get('timestamp', '?')}")
    print()
    print(f"{'metric':<20s} {'sklearn':>10s} {'cuML':>10s} {'Δ':>10s}")
    print("-" * 54)
    for m in TARGET_METRICS:
        sv = sk.get(m, float("nan"))
        cv = cu.get(m, float("nan"))
        try:
            d = cv - sv
            print(f"{m:<20s} {sv:>10.4f} {cv:>10.4f} {d:>+10.4f}")
        except Exception:
            print(f"{m:<20s} {sv!s:>10s} {cv!s:>10s} {'n/a':>10s}")

    print()
    auroc_diff = abs((cu.get("roc_auc", 0) or 0) - (sk.get("roc_auc", 0) or 0))
    verdict = "GO" if auroc_diff <= GO_TOLERANCE else "NoGo"
    print(f"|Δ roc_auc| = {auroc_diff:.4f}  (tolerance {GO_TOLERANCE})  →  {verdict}")


if __name__ == "__main__":
    main()
