"""Statistical characterisation of the exp3 c1 recorded evaluation metrics.

Design: 5 methods {RF (fs, top-10), RF-nofs (all 165 features), SvmW, SvmA, Lstm}
        x 6 evaluation modes {Pooled-base, Pooled-SW-SMOTE, Within-in, Within-out,
        Mixed-in, Mixed-out} (Cross excluded), per-seed AUROC.

This summarises the model-dependent structure of the recorded AUROC metrics under the
within-domain / pooled evaluation protocol. It reports, without asserting absolute
generalisation performance:
  (A) descriptives + 95% CIs (t and percentile bootstrap)
  (B) between-method Kruskal-Wallis per mode (+ eta^2_H, Dunn/Holm post-hoc, Cliff's delta)
  (C) within-method mode contrasts (unpaired Mann-Whitney + Hodges-Lehmann + Cliff's delta,
      with the retired seed-paired Wilcoxon and the seed correlation kept alongside as a
      diagnostic -- see docs/experiments/results/exp3-analysis/unpaired_contrast_verification.md)
  (D) model-characteristic quantities: RF feature-count effect (fs vs nofs); decision spread /
      specificity (all-positive collapse vs de-collapse under SW-SMOTE); across-seed stability
      (Brown-Forsythe); SvmA feature-informativeness probe
  (E) balanced two-way (method x mode) Scheirer-Ray-Hare test

Only scipy/numpy/pandas/matplotlib are used (Dunn, Cliff's delta, Scheirer-Ray-Hare implemented
here). Figures are written to docs/experiments/results/exp3-analysis/figures/c1_recorded/ and
referenced from c1_recorded_value_analysis.md.
"""
import glob, os, re, json, math, itertools
import numpy as np
from scipy import stats

REPO = r"c:/git/work/vehicle_ddd_eval/vehicle_based_DDD_comparison"
os.chdir(REPO)
EVAL = "results/outputs/evaluation"
FIGDIR = "docs/experiments/results/exp3-analysis/figures/c1_recorded"
RNG = np.random.default_rng(20260726)

METHODS = ["RF_fs", "RF_nofs", "SvmW", "SvmA", "Lstm"]
MODES = ["pooled_base", "pooled_smote", "mixed_in", "mixed_out"]  # within_* retired (within_retirement_plan.md)
MODE_LABEL = {"pooled_base": "Pooled-base", "pooled_smote": "Pooled-SW-SMOTE",
              "within_in": "Within-in", "within_out": "Within-out",
              "mixed_in": "Mixed-in", "mixed_out": "Mixed-out"}
COL = {"RF_fs": "#1f77b4", "RF_nofs": "#17becf", "SvmW": "#ff7f0e", "SvmA": "#d62728", "Lstm": "#2ca02c"}

def classify(model, b):
    is_nofs = "_nofs_s" in b
    method = ("RF_nofs" if is_nofs else "RF_fs") if model == "RF" else model
    if "pooled_iv25base" in b: mode = "pooled_base"
    elif "pooled_iv25smote" in b: mode = "pooled_smote"
    elif "target_only_imbalv3_knn_wasserstein_in_domain" in b: mode = "within_in"
    elif "target_only_imbalv3_knn_wasserstein_out_domain" in b: mode = "within_out"
    elif "mixed_imbalv3_knn_wasserstein_in_domain" in b: mode = "mixed_in"
    elif "mixed_imbalv3_knn_wasserstein_out_domain" in b: mode = "mixed_out"
    else: return None
    m = re.search(r"_s(\d+)\.json$", b)
    return (method, mode, m.group(1)) if m else None

def cell_metrics(path):
    try: d = json.load(open(path))
    except Exception: return None
    auc = d.get("roc_auc")
    if auc is None or not (0 <= float(auc) <= 1): return None
    cm = d.get("confusion_matrix"); spec = pred_pos = np.nan
    if cm and len(cm) == 2:
        tn, fp = cm[0]; fn, tp = cm[1]; n = tn + fp + fn + tp
        spec = tn / (tn + fp) if (tn + fp) else np.nan
        pred_pos = (fp + tp) / n if n else np.nan
    proba = d.get("y_pred_proba")
    pstd = float(np.std(proba)) if isinstance(proba, list) and proba else np.nan
    return dict(roc_auc=float(auc), specificity=spec, pred_pos_rate=pred_pos, proba_std=pstd)

best = {}
for model in ["RF", "SvmW", "SvmA", "Lstm"]:
    for f in glob.glob(f"{EVAL}/{model}/**/eval_results_{model}_*.json", recursive=True):
        c = classify(model, os.path.basename(f))
        if not c: continue
        mt = os.path.getmtime(f)
        if c not in best or mt > best[c][0]: best[c] = (mt, f)
data = {}
for (method, mode, seed), (mt, f) in best.items():
    mm = cell_metrics(f)
    if mm: data.setdefault((method, mode), {})[seed] = mm

def arr(method, mode, key="roc_auc"):
    d = data.get((method, mode), {})
    return {s: v[key] for s, v in d.items() if not (isinstance(v[key], float) and math.isnan(v[key]))}

def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0: return np.nan
    gt = sum((x > b).sum() for x in a); lt = sum((x < b).sum() for x in a)
    return (gt - lt) / (len(a) * len(b))
def cliff_mag(d):
    ad = abs(d); return "negligible" if ad < 0.147 else "small" if ad < 0.33 else "medium" if ad < 0.474 else "large"
def boot_ci(x, n=2000):
    x = np.asarray(x, float)
    if len(x) < 3: return (np.nan, np.nan)
    r = stats.bootstrap((x,), np.mean, n_resamples=n, method="percentile", confidence_level=0.95, random_state=RNG)
    return (float(r.confidence_interval.low), float(r.confidence_interval.high))
def t_ci(x):
    x = np.asarray(x, float); n = len(x)
    if n < 2: return (np.nan, np.nan)
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / math.sqrt(n)
    return (x.mean() - h, x.mean() + h)
def dunn_holm(groups):
    names = [k for k in groups if len(groups[k]) > 0]
    allv = np.concatenate([np.asarray(groups[k], float) for k in names]); ranks = stats.rankdata(allv)
    idx = 0; rmean = {}; nsize = {}
    for k in names:
        n = len(groups[k]); rmean[k] = ranks[idx:idx + n].mean(); nsize[k] = n; idx += n
    N = len(allv); _, cnt = np.unique(allv, return_counts=True); ties = sum(t**3 - t for t in cnt)
    sigma2 = (N * (N + 1) / 12) - ties / (12 * (N - 1)); out = []
    for a, b in itertools.combinations(names, 2):
        se = math.sqrt(sigma2 * (1 / nsize[a] + 1 / nsize[b])); z = (rmean[a] - rmean[b]) / se if se else 0.0
        out.append([a, b, z, 2 * (1 - stats.norm.cdf(abs(z)))])
    order = sorted(range(len(out)), key=lambda i: out[i][3]); m = len(out)
    for rank, i in enumerate(order): out[i].append(min(1.0, out[i][3] * (m - rank)))
    return out
def scheirer_ray_hare(recs):
    A = sorted(set(r[0] for r in recs)); B = sorted(set(r[1] for r in recs))
    R = stats.rankdata(np.array([r[2] for r in recs], float)); N = len(R); MS = np.var(R) * N / (N - 1); grand = R.mean()
    def ss(g):
        s = 0.0
        for gv in set(g):
            mask = np.array([g[i] == gv for i in range(N)]); s += mask.sum() * (R[mask].mean() - grand) ** 2
        return s
    ga = [r[0] for r in recs]; gb = [r[1] for r in recs]; gab = [(r[0], r[1]) for r in recs]
    SS_A = ss(ga); SS_B = ss(gb); SS_AB = ss(gab) - SS_A - SS_B
    dfA, dfB = len(A) - 1, len(B) - 1; dfAB = dfA * dfB
    return {"A_H": SS_A / MS, "A_df": dfA, "A_p": 1 - stats.chi2.cdf(SS_A / MS, dfA),
            "B_H": SS_B / MS, "B_df": dfB, "B_p": 1 - stats.chi2.cdf(SS_B / MS, dfB),
            "AB_H": SS_AB / MS, "AB_df": dfAB, "AB_p": 1 - stats.chi2.cdf(SS_AB / MS, dfAB)}
def fmt(x, p=3): return "nan" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:.{p}f}"

# ---- print tables (A/B/C/D/E) ----
print("(A) DESCRIPTIVES + 95% CI (AUROC)")
A = {}
for method in METHODS:
    A[method] = {}
    for mode in MODES:
        v = np.array(list(arr(method, mode).values()), float)
        if len(v) == 0: A[method][mode] = None; continue
        A[method][mode] = dict(n=len(v), mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                               t_ci=list(map(float, t_ci(v))), boot_ci=list(map(float, boot_ci(v))))
        print(f"  {method:8} {MODE_LABEL[mode]:16}: {v.mean():.3f}+/-{(v.std(ddof=1) if len(v)>1 else 0):.3f} (n={len(v)})")

print("\n(B) BETWEEN-METHOD Kruskal-Wallis per mode")
for mode in MODES:
    g = {m: np.array(list(arr(m, mode).values()), float) for m in METHODS}
    g = {m: v for m, v in g.items() if len(v) >= 2}
    if len(g) < 2: continue
    H, p = stats.kruskal(*g.values()); k = len(g); N = sum(len(v) for v in g.values())
    eta2 = (H - k + 1) / (N - k) if N > k else float('nan')
    order = sorted(g, key=lambda m: -g[m].mean())
    print(f"  {MODE_LABEL[mode]:16}: H={H:.2f} p={p:.3g} eta2={eta2:.2f}  rank: " + " > ".join(f"{m}({g[m].mean():.3f})" for m in order))

print("\n(C) WITHIN-METHOD mode contrasts (unpaired Mann-Whitney + paired diagnostic)")
CONTR = [("imbalance(base->SWSMOTE)", "pooled_base", "pooled_smote"),
         ("population(pooled->mixed)", "pooled_smote", "mixed_in"),
         ("target_group(in->out)",     "mixed_in",    "mixed_out")]

def hodges_lehmann(x, y):
    """Median of all pairwise differences y_j - x_i: the Mann-Whitney-consistent estimate."""
    return float(np.median([b - a for a in x for b in y]))

print(f"  {'method':8} {'contrast':26} {'n1':>3}{'n2':>4} {'HL':>7} {'p_unpair':>9} "
      f"{'delta':>6} | {'nc':>3} {'rho':>6} {'dmed':>7} {'p_pair':>8}")
for method in METHODS:
    for name, m1, m2 in CONTR:
        d1, d2 = arr(method, m1), arr(method, m2)
        x = np.array(list(d1.values()), float)
        y = np.array(list(d2.values()), float)
        if len(x) < 2 or len(y) < 2:
            print(f"  {method:8} {name:26}: NA (n1={len(x)}, n2={len(y)})"); continue
        # --- unpaired: what the paper will report ---
        U, p = stats.mannwhitneyu(y, x, alternative="two-sided")   # 'auto' -> exact when it can
        hl = hodges_lehmann(x, y); dl = cliffs_delta(y, x)
        # --- diagnostic: the pairing the switch is retiring ---
        common = sorted(set(d1) & set(d2))
        xc = np.array([d1[s] for s in common], float)
        yc = np.array([d2[s] for s in common], float)
        rho = stats.spearmanr(xc, yc)[0] if len(common) >= 4 else float('nan')
        if len(common) >= 3:
            try: _, p_pair = stats.wilcoxon(xc, yc)
            except ValueError: p_pair = float('nan')
            d_pair = float(np.median(yc - xc))
        else:
            p_pair = d_pair = float('nan')
        print(f"  {method:8} {name:26} {len(x):3d}{len(y):4d} {hl:+7.3f} {p:9.3g} "
              f"{dl:+6.2f} | {len(common):3d} {rho:+6.2f} {d_pair:+7.3f} {p_pair:8.3g}")

print("\n(D1) RF feature-count effect (fs vs nofs)")
D1 = {}
for mode in MODES:
    a = np.array(list(arr("RF_fs", mode).values()), float); b = np.array(list(arr("RF_nofs", mode).values()), float)
    if len(a) >= 2 and len(b) >= 2:
        U, p = stats.mannwhitneyu(b, a, alternative="two-sided"); dl = cliffs_delta(b, a)
        D1[mode] = dict(fs=float(a.mean()), nofs=float(b.mean()), p=float(p), cliff=float(dl))
        print(f"  {MODE_LABEL[mode]:16}: fs={a.mean():.3f} nofs={b.mean():.3f} d={b.mean()-a.mean():+.3f} p={p:.3g} cliff={dl:+.2f}({cliff_mag(dl)})")

print("\n(D2) decision spread/specificity (Pooled)")
D2 = {}
for method in METHODS:
    D2[method] = {}
    for mode in MODES:
        d = data.get((method, mode), {})
        if not d: continue
        D2[method][mode] = dict(proba_std=float(np.nanmean([v["proba_std"] for v in d.values()])),
                                specificity=float(np.nanmean([v["specificity"] for v in d.values()])),
                                pred_pos_rate=float(np.nanmean([v["pred_pos_rate"] for v in d.values()])))
for method in ["SvmW", "Lstm", "RF_fs"]:
    for mode in ["pooled_base", "pooled_smote"]:
        r = D2.get(method, {}).get(mode)
        if r: print(f"  {method:6} {MODE_LABEL[mode]:15}: proba_std={r['proba_std']:.3f} spec={fmt(r['specificity'])} pred_pos={fmt(r['pred_pos_rate'])}")

print("\n(D3) across-seed SD + Brown-Forsythe")
SD = {m: {md: (float(np.std(list(arr(m, md).values()), ddof=1)) if len(arr(m, md)) > 1 else None) for md in MODES} for m in METHODS}
for mode in MODES:
    gs = [np.array(list(arr(m, mode).values()), float) for m in METHODS if len(arr(m, mode)) >= 2]
    if len(gs) >= 2:
        W, p = stats.levene(*gs, center="median")
        print(f"  {MODE_LABEL[mode]:16}: BF W={W:.2f} p={p:.3g}")

print("\n(E) two-way SRH (balanced, Within/Mixed x in/out)")
core = ["mixed_in", "mixed_out"]  # within_* retired; SRH mode factor now 2 levels (df=1)
psets = {m: set.intersection(*[set(arr(m, md)) for md in core]) if all(arr(m, md) for md in core) else set() for m in METHODS}
common = set.intersection(*psets.values()) if all(psets.values()) else set()
recs = [(m, md, arr(m, md)[s]) for m in METHODS for md in core for s in common if s in arr(m, md)]
srh = None
if common and len(recs) == len(METHODS) * len(core) * len(common):
    srh = scheirer_ray_hare(recs)
    print(f"  balanced {len(METHODS)}x{len(core)}x{len(common)}={len(recs)}: method H={srh['A_H']:.1f} p={srh['A_p']:.3g} | mode p={srh['B_p']:.3g} | interaction p={srh['AB_p']:.3g}")

# ---- figures ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
os.makedirs(FIGDIR, exist_ok=True)

# Fig 1: AUROC by method x mode. Pooled-base dropped (all remaining modes are SW-SMOTE-treated).
# Error bars = mean +/- SD (matching the table's +/- column), clipped to [0,1] since AUROC is
# bounded (a symmetric bar can otherwise overshoot 1 for small-n, near-ceiling cells).
MODES_F1 = [md for md in MODES if md != "pooled_base"]
fig, ax = plt.subplots(figsize=(10, 5)); x = np.arange(len(MODES_F1)); w = 0.16
for i, method in enumerate(METHODS):
    m = []; lo = []; hi = []
    for mode in MODES_F1:
        r = A[method][mode]
        if r:
            mu, sd = r["mean"], r["sd"]
            m.append(mu); lo.append(min(sd, mu)); hi.append(min(sd, 1.0 - mu))
        else: m.append(np.nan); lo.append(0); hi.append(0)
    ax.bar(x + (i - 2) * w, m, w, yerr=[lo, hi], capsize=2, label=method, color=COL[method])
ax.axhline(0.5, ls="--", c="gray", lw=1); ax.text(len(MODES_F1) - 0.5, 0.505, "0.5", color="gray", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([MODE_LABEL[md] for md in MODES_F1], rotation=15)
ax.set_ylabel("AUROC (mean $\\pm$ SD, clipped to [0,1])"); ax.set_ylim(0.45, 1.0)
ax.set_title("Recorded AUROC by method $\\times$ evaluation mode (SW-SMOTE modes)")
ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14))
fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig1_auroc_method_mode.png", dpi=130, bbox_inches="tight"); plt.close(fig)

# Fig 2: RF feature-count effect
fig, ax = plt.subplots(figsize=(8.5, 4.5)); mfc = ["mixed_in", "mixed_out", "pooled_smote"]
x = np.arange(len(mfc)); w = 0.38
ax.bar(x - w / 2, [A["RF_fs"][md]["mean"] if A["RF_fs"][md] else np.nan for md in mfc], w, label="RF fs (top-10)", color=COL["RF_fs"])
ax.bar(x + w / 2, [A["RF_nofs"][md]["mean"] if A["RF_nofs"][md] else np.nan for md in mfc], w, label="RF nofs (all 165)", color=COL["RF_nofs"])
ax.axhline(0.5, ls="--", c="gray", lw=1); ax.set_xticks(x); ax.set_xticklabels([MODE_LABEL[md] for md in mfc], rotation=18)
ax.set_ylabel("AUROC"); ax.set_ylim(0.45, 1.0); ax.set_title("RF feature-count effect (all 165 features vs top-10)"); ax.legend()
fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig2_rf_feature_count.png", dpi=130, bbox_inches="tight"); plt.close(fig)

# Fig 3: specificity (degeneracy) Pooled
fig, ax = plt.subplots(figsize=(7.5, 4.5)); md_deg = ["SvmW", "Lstm", "RF_fs"]; x = np.arange(len(md_deg)); w = 0.38
ax.bar(x - w / 2, [D2[m].get("pooled_base", {}).get("specificity", np.nan) for m in md_deg], w, label="Pooled-base", color="#8c564b")
ax.bar(x + w / 2, [D2[m].get("pooled_smote", {}).get("specificity", np.nan) for m in md_deg], w, label="Pooled-SW-SMOTE", color="#2ca02c")
ax.set_xticks(x); ax.set_xticklabels(md_deg); ax.set_ylim(0, 1.0)
ax.set_ylabel("specificity ($\\approx$0 = all-positive)"); ax.set_title("Decision spread under Pooled: SW-SMOTE restores SvmW specificity"); ax.legend()
fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig3_specificity.png", dpi=130, bbox_inches="tight"); plt.close(fig)

# Fig 4: seed instability
fig, ax = plt.subplots(figsize=(7, 4)); mv = ["mixed_in", "mixed_out"]; methods_v = ["RF_fs", "RF_nofs", "SvmA", "SvmW", "Lstm"]
ax.bar(methods_v, [np.nanmean([SD[m][md] for md in mv if SD[m][md] is not None]) for m in methods_v], color=[COL[m] for m in methods_v])
ax.set_ylabel("mean across-seed SD of AUROC"); ax.set_title("Across-seed variability by method (Within/Mixed)")
fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig4_seed_variability.png", dpi=130, bbox_inches="tight"); plt.close(fig)

print(f"\nfigures written to {FIGDIR}/ (fig1..fig4)")
