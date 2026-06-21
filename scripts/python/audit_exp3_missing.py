#!/usr/bin/env python3
"""Audit exp3 four-factor grid completeness.

Scans results/outputs/evaluation/<MODEL>/**/eval_results_*.json and emits
a tab-separated missing-cell list with 7 columns:

    MODEL  COND  DIST  DOM  MODE  RATIO  SEED

where:
  MODE  in {dt, mx}        # dt = domain_train (within+cross), mx = mixed
  RATIO is empty string for COND=baseline, otherwise '0.1' or '0.5'

Lstm + (smote_plain | undersample_rus) + ratio=0.1 cells are excluded
(infeasible; natural minority of Lstm event labels ~0.27).

Eval-JSON naming (canonical):
  dt: eval_results_<M>_domain_train_prior_<M>_<COND>_knn_<DIST>_<DOM>_domain_train_split2[..._subjectwise][_ratio<R>]_s<SEED>_<within|cross>.json
  mx: eval_results_<M>_mixed_prior_<M>_<COND>_knn_<DIST>_<DOM>_mixed_split2[..._subjectwise][_ratio<R>]_s<SEED>.json

A dt cell is complete only when BOTH within+cross JSONs exist.
A mx cell is complete when its single JSON exists.
"""
from __future__ import annotations

import argparse
import re
import sys
from itertools import product
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = PROJECT_ROOT / "results" / "outputs" / "evaluation"

MODELS = ["SvmA", "SvmW", "Lstm"]
CONDITIONS = ["baseline", "smote", "smote_plain", "undersample_rus"]
DISTS = ["mmd", "dtw", "wasserstein"]
DOMS = ["in_domain", "out_domain"]
RATIOS = ["0.1", "0.5"]
SEEDS = [0, 1, 3, 7, 13, 42, 99, 123, 256, 512, 777, 999, 1234, 1337, 2024]
MODES = ["dt", "mx"]

# CONDITION env-value used by submitter (-> tag fragment in eval JSON)
COND_TAG_FRAG = {
    "baseline": "baseline",
    "smote": "imbalv3",          # subject-wise SMOTE
    "smote_plain": "smote_plain",
    "undersample_rus": "undersample_rus",
}
MODE_FRAG = {"dt": "domain_train", "mx": "mixed"}


def is_infeasible(model: str, cond: str, ratio: str) -> bool:
    """Lstm + smote_plain/undersample_rus + r=0.1 is infeasible (natural minority ~0.27)."""
    return (
        model == "Lstm"
        and cond in ("smote_plain", "undersample_rus")
        and ratio == "0.1"
    )


def expected_eval_basenames(
    model: str, cond: str, dist: str, dom: str, mode: str, ratio: str, seed: int
) -> list[str]:
    """Return the eval-JSON basename(s) that must exist for a complete cell."""
    mode_frag = MODE_FRAG[mode]
    cond_frag = COND_TAG_FRAG[cond]
    parts = [
        f"eval_results_{model}_{mode_frag}_prior_{model}_{cond_frag}",
        f"knn_{dist}_{dom}_{mode_frag}_split2",
    ]
    tag_body = "_".join(parts)
    if cond == "baseline":
        suffix = f"s{seed}"
    elif cond == "smote":
        suffix = f"subjectwise_ratio{ratio}_s{seed}"
    else:
        suffix = f"ratio{ratio}_s{seed}"

    base = f"{tag_body}_{suffix}"
    if mode == "dt":
        return [f"{base}_within.json", f"{base}_cross.json"]
    else:
        return [f"{base}.json"]


def scan_existing_evals() -> set[str]:
    """Return the set of eval-JSON basenames present under EVAL_ROOT."""
    if not EVAL_ROOT.exists():
        print(f"[FATAL] {EVAL_ROOT} not found", file=sys.stderr)
        sys.exit(1)
    found = set()
    for model in MODELS:
        mdir = EVAL_ROOT / model
        if not mdir.exists():
            continue
        for p in mdir.rglob("eval_results_*.json"):
            found.add(p.name)
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="Path to write missing list (7-col TSV)")
    ap.add_argument("--summary", action="store_true", help="Print per-(model, cond, mode) summary")
    args = ap.parse_args()

    existing = scan_existing_evals()
    print(f"[INFO] Found {len(existing)} eval JSONs under {EVAL_ROOT}", file=sys.stderr)

    missing: list[tuple[str, str, str, str, str, str, int]] = []
    expected_total = 0
    complete_total = 0

    cell_iter = product(MODELS, CONDITIONS, DISTS, DOMS, MODES, SEEDS)
    for model, cond, dist, dom, mode, seed in cell_iter:
        # Build ratio iterator
        if cond == "baseline":
            ratio_choices = [""]
        else:
            ratio_choices = RATIOS
        for ratio in ratio_choices:
            if cond != "baseline" and is_infeasible(model, cond, ratio):
                continue
            expected_total += 1
            basenames = expected_eval_basenames(
                model, cond, dist, dom, mode, ratio if ratio else "0.5", seed
            )
            # baseline ignores ratio in basename so above arg is unused for baseline
            if cond == "baseline":
                basenames = expected_eval_basenames(
                    model, cond, dist, dom, mode, "", seed
                )
            if all(b in existing for b in basenames):
                complete_total += 1
            else:
                missing.append((model, cond, dist, dom, mode, ratio, seed))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fp:
        for row in missing:
            # bash IFS=$'\t' read collapses consecutive tabs, so emit '-'
            # for empty RATIO (baseline) instead of an empty field.
            row_out = tuple("-" if x == "" else x for x in row)
            fp.write("\t".join(str(x) for x in row_out) + "\n")

    pct = 100.0 * complete_total / expected_total if expected_total else 0.0
    print(
        f"[INFO] Expected={expected_total} Complete={complete_total} "
        f"Missing={len(missing)} ({pct:.1f}% complete)",
        file=sys.stderr,
    )
    print(f"[INFO] Wrote {len(missing)} rows to {out_path}", file=sys.stderr)

    if args.summary:
        from collections import Counter
        c = Counter((m, cd, md) for m, cd, _, _, md, _, _ in missing)
        print("\n[SUMMARY] Missing cells by (model, cond, mode):", file=sys.stderr)
        for k, v in sorted(c.items()):
            print(f"  {k[0]:<5} {k[1]:<16} {k[2]:<3} {v}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
