"""Summarise SvmA local cuML evaluation results for paper main.tex update.

Reads every SvmA within/cross JSON under results/outputs/evaluation/SvmA and
prints per-condition AUROC/AUPRC mean+std, plus a paper-ready sentence.
"""
from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EVAL_DIR = REPO / "results" / "outputs" / "evaluation" / "SvmA"

TAG_RE = re.compile(
    r"prior_SvmA_imbalv3_knn_"
    r"(?P<dist>\w+?)_"
    r"(?P<dom>in_domain|out_domain)_"
    r"domain_train_split2_subjectwise_ratio(?P<r>[\d.]+)_s(?P<seed>\d+)_"
    r"(?P<split>within|cross)\.json$"
)


def main():
    rows = []
    for p in EVAL_DIR.rglob("eval_results_SvmA_*.json"):
        m = TAG_RE.search(p.name)
        if not m:
            continue
        d = json.loads(p.read_text())
        rows.append({
            "dist": m.group("dist"),
            "dom": m.group("dom"),
            "r": float(m.group("r")),
            "seed": int(m.group("seed")),
            "split": m.group("split"),
            "roc_auc": d.get("roc_auc"),
            "auc_pr": d.get("auc_pr"),
            "size_bytes": p.stat().st_size,
        })
    print(f"Found {len(rows)} JSONs ({sum(1 for r in rows if r['split']=='within')} within, "
          f"{sum(1 for r in rows if r['split']=='cross')} cross)")
    print()

    print("=" * 80)
    print("Per-condition aggregates (within eval only)")
    print("=" * 80)
    print(f"{'dist':<14}{'dom':<12}{'r':<6}{'n':<4}{'AUROC mean':>12}{'AUROC sd':>10}{'AUPRC mean':>12}")
    print("-" * 80)
    groups = defaultdict(list)
    for r in rows:
        if r["split"] != "within":
            continue
        key = (r["dist"], r["dom"], r["r"])
        groups[key].append(r)
    for key in sorted(groups):
        items = groups[key]
        aurocs = [it["roc_auc"] for it in items if it["roc_auc"] is not None]
        auprcs = [it["auc_pr"]  for it in items if it["auc_pr"]  is not None]
        if not aurocs:
            continue
        m_au = statistics.mean(aurocs)
        s_au = statistics.stdev(aurocs) if len(aurocs) > 1 else 0.0
        m_pr = statistics.mean(auprcs) if auprcs else float("nan")
        print(f"{key[0]:<14}{key[1]:<12}{key[2]:<6.1f}{len(items):<4}"
              f"{m_au:>12.4f}{s_au:>10.4f}{m_pr:>12.4f}")
    print()

    print("=" * 80)
    print("Same for cross eval")
    print("=" * 80)
    print(f"{'dist':<14}{'dom':<12}{'r':<6}{'n':<4}{'AUROC mean':>12}{'AUROC sd':>10}{'AUPRC mean':>12}")
    print("-" * 80)
    groups = defaultdict(list)
    for r in rows:
        if r["split"] != "cross":
            continue
        key = (r["dist"], r["dom"], r["r"])
        groups[key].append(r)
    for key in sorted(groups):
        items = groups[key]
        aurocs = [it["roc_auc"] for it in items if it["roc_auc"] is not None]
        auprcs = [it["auc_pr"]  for it in items if it["auc_pr"]  is not None]
        if not aurocs:
            continue
        m_au = statistics.mean(aurocs)
        s_au = statistics.stdev(aurocs) if len(aurocs) > 1 else 0.0
        m_pr = statistics.mean(auprcs) if auprcs else float("nan")
        print(f"{key[0]:<14}{key[1]:<12}{key[2]:<6.1f}{len(items):<4}"
              f"{m_au:>12.4f}{s_au:>10.4f}{m_pr:>12.4f}")
    print()

    within_aurocs = [r["roc_auc"] for r in rows if r["split"] == "within" and r["roc_auc"] is not None]
    if within_aurocs:
        lo, hi = min(within_aurocs), max(within_aurocs)
        print(f"All-within AUROC range: {lo:.4f}–{hi:.4f}, "
              f"mean={statistics.mean(within_aurocs):.4f}, "
              f"sd={statistics.stdev(within_aurocs) if len(within_aurocs)>1 else 0.0:.4f}, "
              f"N={len(within_aurocs)}")


if __name__ == "__main__":
    main()
