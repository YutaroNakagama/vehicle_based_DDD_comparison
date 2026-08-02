"""exp3 c1 — seed-count convergence for every paper-target case (TIV2026 / exp2 fig8 style).

For each (method x mode) the paper reports, add seeds one at a time (fixed master order) and
track the running mean AUROC and its 95% t-CI half-width. A case's seed count is justified when:
  * discriminating (running mean > 0.55): 95% CI half-width <= 0.05 (TARGET_HW), and the running
    mean has flattened (last-3 span <= 0.01);
  * chance (running mean <= 0.55): bootstrap 95% CI upper bound < 0.60 (excludes weak signal).

Paper-target modes (Within retired): Pooled-base, Pooled-SW-SMOTE, Mixed-in, Mixed-out.
Methods: RF (fs), RF (nofs), SvmW, SvmA, Lstm.

Outputs
-------
- figures/c1_recorded/fig_seed_convergence.png  (grid: methods x modes; running mean +/- CI vs k)
- results/analysis/exp3_verification/c1_seed_convergence.json  (per-case verdicts)
CPU-only, reads recorded eval JSONs; run anytime.
"""
from __future__ import annotations
import glob, json, os, re, sys
from collections import defaultdict
import numpy as np
from scipy import stats

REPO = r"c:/git/work/vehicle_ddd_eval/vehicle_based_DDD_comparison"; os.chdir(REPO)
TARGET_HW = 0.05          # 95% CI half-width target (discriminating cases)
CHANCE_MEAN = 0.55        # running mean <= this -> "chance" case
CHANCE_UPPER = 0.60       # chance bootstrap CI upper must stay below this
FLAT_SPAN = 0.01          # last-3 running-mean span for "flattened"
B_BOOT = 2000
# augmentation / master seed order (extra present seeds appended, sorted)
SEED_ORDER = [42, 123, 2025, 0, 1, 7, 13, 256, 512, 1337, 2024, 3, 5, 9, 11]
MODES = ["Pooled-base", "Pooled-SW-SMOTE", "Mixed-in", "Mixed-out"]
METHODS = ["RF (fs)", "RF (nofs)", "SvmW", "SvmA", "Lstm"]


def mode_of(b):
    if "iv25smote" in b or "pooled_swsmote" in b: return "Pooled-SW-SMOTE"
    if "pooled" in b and "smote" not in b.lower() and "swsmote" not in b.lower(): return "Pooled-base"
    if "mixed" in b and "out_domain" in b: return "Mixed-out"
    if "mixed" in b and "in_domain" in b: return "Mixed-in"
    return None


def method_of(m, b):
    if m == "RF": return "RF (nofs)" if "nofs" in b else "RF (fs)"
    return m


# collect {(method,mode): {seed: auroc}}
cells = defaultdict(dict)
for m in ["RF", "SvmW", "SvmA", "Lstm"]:
    for f in glob.glob(f"results/outputs/evaluation/{m}/**/eval_results_*.json", recursive=True):
        b = os.path.basename(f); md = mode_of(b)
        if md is None: continue
        s = re.search(r"_s(\d+)\.json$", b)
        if not s: continue
        try: a = json.load(open(f)).get("roc_auc")
        except Exception: continue
        if isinstance(a, (int, float)):
            cells[(method_of(m, b), md)][int(s.group(1))] = a


def ordered_seeds(sd):
    present = list(sd)
    ordered = [s for s in SEED_ORDER if s in sd] + sorted(s for s in present if s not in SEED_ORDER)
    return ordered


def running(vals):
    """running mean and 95% t-CI half-width for k=2..N."""
    ks, means, hws = [], [], []
    for k in range(2, len(vals) + 1):
        v = np.array(vals[:k]); sd = v.std(ddof=1)
        hw = stats.t.ppf(0.975, k - 1) * sd / np.sqrt(k)
        ks.append(k); means.append(v.mean()); hws.append(hw)
    return ks, means, hws


def boot_upper(vals):
    rng = np.random.RandomState(12345); v = np.array(vals)
    bs = [rng.choice(v, len(v), replace=True).mean() for _ in range(B_BOOT)]
    return float(np.percentile(bs, 97.5))


results = {}
for me in METHODS:
    for md in MODES:
        sd = cells.get((me, md), {})
        if len(sd) < 2:
            results[f"{me} | {md}"] = {"n": len(sd), "verdict": "n<2" if sd else "absent"}
            continue
        seeds = ordered_seeds(sd)
        vals = [sd[s] for s in seeds]
        ks, means, hws = running(vals)
        fmean = means[-1]; fhw = hws[-1]
        flat = (max(means[-3:]) - min(means[-3:])) <= FLAT_SPAN if len(means) >= 3 else False
        if fmean > CHANCE_MEAN:
            adeq = (fhw <= TARGET_HW) and flat
            crit = f"CI-hw {fhw:.3f}<=0.05 & flat"
        else:
            up = boot_upper(vals); adeq = up < CHANCE_UPPER
            crit = f"chance boot-upper {up:.3f}<0.60"
        results[f"{me} | {md}"] = {"n": len(vals), "mean": round(fmean, 3), "ci_hw": round(fhw, 3),
                                   "flattened": bool(flat), "verdict": "ADEQUATE" if adeq else "SHORT",
                                   "criterion": crit, "ks": ks, "means": means, "hws": hws}

# ---- plot grid ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, axes = plt.subplots(len(METHODS), len(MODES), figsize=(15, 13), sharex=False)
for i, me in enumerate(METHODS):
    for j, md in enumerate(MODES):
        ax = axes[i][j]; r = results.get(f"{me} | {md}", {})
        if "means" not in r:
            ax.text(0.5, 0.5, r.get("verdict", "absent"), ha="center", va="center", fontsize=9, color="gray")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0: ax.set_title(md, fontsize=10)
            if j == 0: ax.set_ylabel(me, fontsize=10)
            continue
        ks = np.array(r["ks"]); mn = np.array(r["means"]); hw = np.array(r["hws"])
        chance = r["mean"] <= CHANCE_MEAN
        col = "#d62728" if chance else "#1f77b4"
        ax.plot(ks, mn, "-o", ms=3, color=col)
        ax.fill_between(ks, mn - hw, mn + hw, color=col, alpha=0.18)
        ax.axhline(0.5, ls="--", c="gray", lw=0.8)
        ok = r["verdict"] == "ADEQUATE"
        ax.text(0.97, 0.06, f"n={r['n']}  hw={r['ci_hw']:.3f}\n{'✓ '+r['verdict'] if ok else '✗ '+r['verdict']}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
                color=("#2a7" if ok else "#c33"))
        ax.set_ylim(0.44, 1.0); ax.grid(alpha=0.25)
        if i == 0: ax.set_title(md, fontsize=10)
        if j == 0: ax.set_ylabel(me, fontsize=10)
        if i == len(METHODS) - 1: ax.set_xlabel("number of seeds (k)", fontsize=8)
fig.suptitle("exp3 c1 — seed-count convergence: running mean AUROC ± 95% CI vs seeds (paper-target cases)\n"
             "blue = discriminating (target CI half-width ≤ 0.05); red = near-0.5 baseline (bootstrap CI upper < 0.60)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.965])
outdir = "docs/experiments/results/exp3-analysis/figures/c1_recorded"; os.makedirs(outdir, exist_ok=True)
fig.savefig(f"{outdir}/fig_seed_convergence.png", dpi=120, bbox_inches="tight"); plt.close(fig)

os.makedirs("results/analysis/exp3_verification", exist_ok=True)
clean = {k: {kk: vv for kk, vv in v.items() if kk not in ("ks", "means", "hws")} for k, v in results.items()}
json.dump(clean, open("results/analysis/exp3_verification/c1_seed_convergence.json", "w"), indent=2)

print(f"{'case':28s} {'n':>3} {'mean':>6} {'CI-hw':>7}  verdict")
for k in sorted(results):
    r = results[k]
    if "mean" in r: print(f"  {k:28s} {r['n']:>3} {r['mean']:>6.3f} {r['ci_hw']:>7.3f}  {r['verdict']}  ({r['criterion']})")
    else: print(f"  {k:28s} {r.get('n',0):>3}    --      --  {r['verdict']}")
n_ok = sum(1 for r in results.values() if r.get("verdict") == "ADEQUATE")
print(f"\nADEQUATE: {n_ok}/{sum(1 for r in results.values() if 'mean' in r)} present cases")
print(f"figure: {outdir}/fig_seed_convergence.png")
