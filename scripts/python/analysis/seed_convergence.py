"""Seed-convergence analysis — justifies the per-method seed counts.

For each model/experiment, computes the running mean within-AUROC and its 95% CI
as seeds are added (TIV2026-style, fig8). The chosen seed count is justified when
the running mean has stabilised and the CI half-width meets the target:
  * discriminating methods (RF/SvmW/Lstm): CI half-width small enough to separate
    from / overlap with each other meaningfully;
  * chance method (SvmA): CI upper bound < 0.60 (excludes weak signal).

Reads the b1cmp_ (target_only) and iv25base_ (pooled) eval JSONs. Robust to
partial completion — run anytime to see current convergence.

Usage: python scripts/python/analysis/seed_convergence.py
"""
from __future__ import annotations
import glob, json, re, math
from collections import defaultdict

SEED_ORDER = [0, 1, 7, 13, 42, 123, 256, 512, 1337, 2024, 2025]


def collect(model, kind):
    """kind: 'b1' (target_only within) or 'iv' (pooled). Returns {seed: [auroc,...]}."""
    by_seed = defaultdict(list)
    if kind == "b1":
        pat = f"results/outputs/evaluation/{model}/**/*b1cmp_{model}*within.json"
    else:
        pat = f"results/outputs/evaluation/{model}/**/eval_results_{model}_pooled_iv25base_{model}*.json"
    for f in glob.glob(pat, recursive=True):
        m = re.search(r"_s(\d+)", f)
        if not m:
            continue
        try:
            a = json.load(open(f)).get("roc_auc")
        except Exception:
            a = None
        if a is not None:
            by_seed[int(m.group(1))].append(a)
    return by_seed


def ci_half(vals):
    n = len(vals)
    if n < 2:
        return float("nan")
    sd = (sum((v - sum(vals) / n) ** 2 for v in vals) / (n - 1)) ** 0.5
    # t(0.975, n-1) approx via normal for n>=2 quick table
    tt = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36,
          9: 2.31, 10: 2.26, 11: 2.23}.get(n, 1.96)
    return tt * sd / math.sqrt(n)


def report(kind, label):
    print(f"\n===== {label} =====")
    for model in ["RF", "SvmW", "SvmA", "Lstm"]:
        by_seed = collect(model, kind)
        # average across ratios within a seed, then order seeds
        seed_means = {s: sum(v) / len(v) for s, v in by_seed.items() if v}
        ordered = [seed_means[s] for s in SEED_ORDER if s in seed_means]
        ordered += [seed_means[s] for s in seed_means if s not in SEED_ORDER]
        if not ordered:
            print(f"  {model:5s}: (no results yet)")
            continue
        # running mean + CI
        run = []
        for k in range(1, len(ordered) + 1):
            sub = ordered[:k]
            run.append((k, sum(sub) / k, ci_half(sub)))
        n, mean, ci = run[-1]
        conv = "  ".join(f"n{k}:{mu:.3f}" for k, mu, _ in run)
        ci_s = f"+/-{ci:.3f}" if not math.isnan(ci) else "n/a"
        print(f"  {model:5s}: n={n}  mean={mean:.3f} {ci_s}   running[{conv}]")


if __name__ == "__main__":
    report("b1", "B1 (target_only + SW-SMOTE, within-AUROC)")
    report("iv", "IV2025 baseline (pooled, no imbalance)")
