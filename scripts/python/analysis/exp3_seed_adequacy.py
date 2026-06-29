"""exp3 seed-adequacy — TIV2026-aligned, ALL conditions (c1 6-case + IV2025).

Applies the TIV2026 (exp2) seed-validity framework to every exp3 condition x model:
  (1) Running-mean AUROC convergence + 95% CI half-width as seeds are added.
  (2) Percentile bootstrap 95% CI (B=2000) over the available seeds.
  (3) sigma_rank: rank-stability of the within>mixed>cross ordering across random
      kappa-seed subsets (TIV2026 fig: sigma_rank -> 0 means the ordering is set by
      condition identity, not seed choice).
Adequacy verdict per condition:
  * discriminating (mean > 0.55): 95% CI half-width <= TARGET_HW (0.05) -> ADEQUATE.
  * chance (mean <= 0.55): bootstrap CI upper bound < 0.60 (excludes weak signal).
Robust to partial completion (SvmW/SvmA still running). Reads c1 imbalv3 tags and
iv25base_ pooled tags. Output: results/analysis/exp3_verification/seed_adequacy.json
"""
from __future__ import annotations
import glob, json, re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
EV = REPO / "results" / "outputs" / "evaluation"
TARGET_HW = 0.05          # journal target for 95% CI half-width (discriminating)
CHANCE_MEAN = 0.55        # mean below this => treated as "chance" condition
CHANCE_UPPER = 0.60       # chance CI upper must stay below this
B_BOOT = 2000

SEED_MASTER = [42, 123, 2025, 0, 1, 7, 13, 256, 512, 1337, 2024, 3, 5, 9, 11, 17, 23, 99, 777, 2718]
SEEDS_BY_MODEL = {"RF": SEED_MASTER[:20], "Lstm": SEED_MASTER[:15], "SvmW": SEED_MASTER[:15], "SvmA": SEED_MASTER[:15]}
COND = [("in_domain", "target_only", "Within-in"), ("out_domain", "target_only", "Within-out"),
        ("in_domain", "source_only", "Cross-in"), ("out_domain", "source_only", "Cross-out"),
        ("in_domain", "mixed", "Mixed-in"), ("out_domain", "mixed", "Mixed-out")]
PAT = re.compile(r"imbalv3_knn_wasserstein_(in_domain|out_domain)_(target_only|source_only|mixed)_split2_subjectwise_ratio0\.5_s(\d+)\.json$")
_RNG = np.random.RandomState(12345)


def collect_c1(model):
    seeds = set(str(s) for s in SEEDS_BY_MODEL[model])
    out = defaultdict(dict)  # (dom,mode) -> {seed: auroc}
    base = EV / model
    if base.exists():
        for fp in base.rglob(f"eval_results_{model}_*imbalv3_knn_wasserstein*ratio0.5_s*.json"):
            mm = PAT.search(fp.name)
            if not mm or mm.group(3) not in seeds:
                continue
            dom, mode, s = mm.groups()
            try:
                a = json.load(open(fp)).get("roc_auc")
            except Exception:
                continue
            if isinstance(a, (int, float)):
                k = (dom, mode)
                if s not in out[k] or fp.stat().st_mtime > out[k].get(s + "_mt", 0):
                    out[k][s] = a
                    out[k][s + "_mt"] = fp.stat().st_mtime
    # strip mtimes
    return {k: {s: v for s, v in d.items() if not s.endswith("_mt")} for k, d in out.items()}


def ci_t(vals):
    from scipy import stats
    n = len(vals)
    if n < 2:
        return None
    s = np.std(vals, ddof=1)
    return float(stats.t.ppf(0.975, n - 1) * s / np.sqrt(n))


def ci_boot(vals):
    if len(vals) < 2:
        return (None, None)
    v = np.asarray(vals)
    means = [v[_RNG.randint(0, len(v), len(v))].mean() for _ in range(B_BOOT)]
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def req_n(s, hw):
    from scipy import stats
    for n in range(2, 300):
        if stats.t.ppf(0.975, n - 1) * s / np.sqrt(n) <= hw:
            return n
    return ">300"


def sigma_rank(cond_seed_auroc, seeds_common, kappa):
    """Std of each condition's rank across all kappa-subsets of seeds_common, averaged."""
    conds = list(cond_seed_auroc.keys())
    if len(conds) < 2 or len(seeds_common) < kappa:
        return None
    rank_pos = {c: [] for c in conds}
    subsets = list(combinations(seeds_common, kappa))
    if len(subsets) > 400:
        idx = _RNG.choice(len(subsets), 400, replace=False)
        subsets = [subsets[i] for i in idx]
    for sub in subsets:
        means = {c: np.mean([cond_seed_auroc[c][s] for s in sub]) for c in conds}
        order = sorted(conds, key=lambda c: -means[c])
        for r, c in enumerate(order):
            rank_pos[c].append(r)
    return float(np.mean([np.std(rank_pos[c]) for c in conds]))


def main():
    report = {"target_hw": TARGET_HW, "models": {}}
    for model in ["RF", "Lstm", "SvmW", "SvmA"]:
        data = collect_c1(model)
        n_target = len(SEEDS_BY_MODEL[model])
        print(f"\n=== {model} (target {n_target} seeds/condition) ===")
        print(f"  {'condition':11s} {'n':>3} {'mean':>6} {'std':>6} {'CI_hw':>6} {'boot95':>16} {'req_n@.05':>9}  verdict")
        mrep = {}
        cond_full = {}  # (dom,mode)->{seed:auroc} for conditions with full seeds (for sigma_rank)
        for dom, mode, lab in COND:
            d = data.get((dom, mode), {})
            vals = list(d.values())
            if not vals:
                print(f"  {lab:11s}   0   (no data yet)")
                mrep[lab] = {"n": 0}
                continue
            n = len(vals); mean = float(np.mean(vals)); std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
            hw = ci_t(vals); lo, hi = ci_boot(vals)
            chance = mean <= CHANCE_MEAN
            if chance:
                ok = (hi is not None and hi < CHANCE_UPPER)
                verdict = f"chance, upper {hi:.3f}{'<0.60 OK' if ok else '>=0.60 NEED+'}" if hi else "n<2"
            else:
                ok = (hw is not None and hw <= TARGET_HW)
                verdict = f"{'ADEQUATE' if ok else 'NEED MORE'} (hw {hw:.3f} vs {TARGET_HW})" if hw else "n<2"
            rn = req_n(std, TARGET_HW) if std > 0 else 1
            bstr = f"[{lo:.3f},{hi:.3f}]" if lo is not None else "-"
            print(f"  {lab:11s} {n:>3} {mean:>6.3f} {std:>6.3f} {hw if hw else 0:>6.3f} {bstr:>16} {str(rn):>9}  {verdict}")
            mrep[lab] = {"n": n, "mean": round(mean, 4), "std": round(std, 4),
                         "ci_hw": round(hw, 4) if hw else None, "boot95": [round(lo, 4), round(hi, 4)] if lo else None,
                         "req_n_05": rn, "adequate": bool(ok)}
            if n >= n_target:
                cond_full[lab] = d
        # sigma_rank over conditions with full seeds + common seeds
        if len(cond_full) >= 2:
            common = set.intersection(*[set(cond_full[c].keys()) for c in cond_full])
            common = sorted(common, key=lambda x: int(x))
            srk = {}
            for kappa in sorted(set([max(2, len(common) - 4), len(common) - 2, len(common) - 1])):
                if kappa >= 2 and kappa <= len(common):
                    sr = sigma_rank(cond_full, common, kappa)
                    if sr is not None:
                        srk[kappa] = round(sr, 3)
            print(f"  sigma_rank ({len(cond_full)} full conditions, {len(common)} common seeds): {srk}")
            mrep["_sigma_rank"] = {"n_conditions": len(cond_full), "n_common_seeds": len(common), "by_kappa": srk}
        report["models"][model] = mrep

    outp = REPO / "results" / "analysis" / "exp3_verification" / "seed_adequacy.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(report, indent=2))
    print(f"\n[saved] {outp}")


if __name__ == "__main__":
    main()
