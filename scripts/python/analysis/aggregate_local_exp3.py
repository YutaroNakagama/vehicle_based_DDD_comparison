"""Aggregate local exp3 evaluation JSONs and print summary statistics.

Handles both Lstm and SvmW eval JSON formats.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from collections import defaultdict
import statistics

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

REPO = Path(__file__).resolve().parents[3]
EVAL_DIR = REPO / "results" / "outputs" / "evaluation"

TAG_RE = re.compile(
    r"eval_results_(?P<model>\w+)_domain_train_"
    r"prior_\w+_imbalv3_knn_(?P<distance>\w+)_(?P<domain>\w+_domain)_"
    r"domain_train_split2_subjectwise_ratio(?P<ratio>[\d.]+)_s(?P<seed>\d+)_"
    r"(?P<eval_type>within|cross)\.json$"
)

def compute_auroc_trapz(fpr, tpr):
    """Trapezoidal AUC from fpr/tpr lists."""
    if HAS_NUMPY:
        return float(np.trapz(tpr, fpr))
    # Manual trapz
    area = 0.0
    for i in range(1, len(fpr)):
        area += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2
    return abs(area)

def parse_svmw(data: dict) -> dict:
    """Extract metrics from SvmW JSON format."""
    recall = data.get("recall", float("nan"))
    precision = data.get("precision", float("nan"))
    f1 = data.get("f1", float("nan"))
    # Compute F2
    if precision + recall > 0:
        f2 = (1 + 4) * precision * recall / (4 * precision + recall)
    else:
        f2 = float("nan")
    # Compute AUROC from roc_curve if present
    auroc = float("nan")
    auprc = float("nan")
    rc = data.get("roc_curve", {})
    if rc:
        fpr = rc.get("fpr", [])
        tpr = rc.get("tpr", [])
        if fpr and tpr:
            auroc = compute_auroc_trapz(fpr, tpr)
        pr = rc.get("precision", [])
        rec = rc.get("recall", [])
        if pr and rec:
            # sort by recall ascending for AUPRC
            pairs = sorted(zip(rec, pr))
            recs = [p[0] for p in pairs]
            prs = [p[1] for p in pairs]
            auprc = compute_auroc_trapz(recs, prs)
    return {"recall": recall, "precision": precision, "f1": f1, "f2": f2, "auroc": auroc, "auprc": auprc}

def parse_lstm(data: dict) -> dict:
    """Extract metrics from Lstm JSON format."""
    recall = data.get("recall_pos", float("nan"))
    precision = data.get("precision_pos", float("nan"))
    f1 = data.get("f1_pos", float("nan"))
    auroc = data.get("roc_auc", float("nan"))
    auprc = data.get("auc_pr", float("nan"))
    if precision is None: precision = float("nan")
    if recall is None: recall = float("nan")
    if auroc is None: auroc = float("nan")
    if auprc is None: auprc = float("nan")
    if str(precision) != "nan" and str(recall) != "nan":
        denom = 4 * float(precision) + float(recall)
        f2 = (5 * float(precision) * float(recall) / denom) if denom > 0 else float("nan")
    else:
        f2 = float("nan")
    return {"recall": recall, "precision": precision, "f1": f1, "f2": f2, "auroc": auroc, "auprc": auprc}

def load_all():
    records = []
    for model_dir in EVAL_DIR.iterdir():
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        for jf in model_dir.rglob("*.json"):
            m = TAG_RE.match(jf.name)
            if not m:
                continue
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"  SKIP {jf.name}: {e}")
                continue
            if model == "Lstm":
                metrics = parse_lstm(data)
            elif model in ("SvmW", "SvmA"):
                metrics = parse_svmw(data)
            else:
                metrics = parse_svmw(data)
            rec = {
                "model": m.group("model"),
                "distance": m.group("distance"),
                "domain": m.group("domain"),
                "ratio": float(m.group("ratio")),
                "seed": int(m.group("seed")),
                "eval_type": m.group("eval_type"),
                **metrics,
            }
            records.append(rec)
    return records

def safe_mean(vals):
    v = [x for x in vals if str(x) != "nan"]
    return statistics.mean(v) if v else float("nan")

def safe_std(vals):
    v = [x for x in vals if str(x) != "nan"]
    return statistics.stdev(v) if len(v) >= 2 else float("nan")

def fmt(v):
    if str(v) == "nan":
        return "   N/A"
    return f"{v:.4f}"

def main():
    records = load_all()
    print(f"Loaded {len(records)} records from {EVAL_DIR}\n")

    # Group by model / eval_type / ratio / domain
    groups = defaultdict(list)
    for r in records:
        key = (r["model"], r["eval_type"], r["domain"], r["ratio"])
        groups[key].append(r)

    print(f"{'Model':<8} {'EvalType':<8} {'Domain':<12} {'Ratio':<6} {'N':>3}  "
          f"{'AUROC':>8} {'AUPRC':>8} {'Recall':>8} {'Precision':>10} {'F2':>8}")
    print("-" * 85)

    for key in sorted(groups.keys()):
        model, eval_type, domain, ratio = key
        recs = groups[key]
        n = len(recs)
        auroc_m = safe_mean([r["auroc"] for r in recs])
        auprc_m = safe_mean([r["auprc"] for r in recs])
        rec_m = safe_mean([r["recall"] for r in recs])
        pre_m = safe_mean([r["precision"] for r in recs])
        f2_m = safe_mean([r["f2"] for r in recs])
        print(f"{model:<8} {eval_type:<8} {domain:<12} {ratio:<6.1f} {n:>3}  "
              f"{fmt(auroc_m):>8} {fmt(auprc_m):>8} {fmt(rec_m):>8} {fmt(pre_m):>10} {fmt(f2_m):>8}")

    print()
    # Also show per-distance breakdown for Lstm within
    print("=== Lstm within: per-distance summary ===")
    for dist in ("mmd", "wasserstein", "dtw"):
        recs = [r for r in records if r["model"] == "Lstm" and r["eval_type"] == "within" and r["distance"] == dist]
        if not recs:
            print(f"  {dist}: no records")
            continue
        auroc_m = safe_mean([r["auroc"] for r in recs])
        rec_m = safe_mean([r["recall"] for r in recs])
        f2_m = safe_mean([r["f2"] for r in recs])
        print(f"  {dist:<15} N={len(recs):>3}  AUROC={fmt(auroc_m)}  Recall={fmt(rec_m)}  F2={fmt(f2_m)}")

    print()
    print("=== SvmW within: per-distance breakdown ===")
    for dist in ("mmd", "wasserstein", "dtw"):
        recs = [r for r in records if r["model"] == "SvmW" and r["eval_type"] == "within" and r["distance"] == dist]
        if not recs:
            print(f"  {dist}: no records")
            continue
        auroc_m = safe_mean([r["auroc"] for r in recs])
        rec_m = safe_mean([r["recall"] for r in recs])
        f2_m = safe_mean([r["f2"] for r in recs])
        print(f"  {dist:<15} N={len(recs):>3}  AUROC={fmt(auroc_m)}  Recall={fmt(rec_m)}  F2={fmt(f2_m)}")

if __name__ == "__main__":
    main()
