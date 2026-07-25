"""Academic statistical analysis of the exp3 c1 RECORDED evaluation results.

Design: 5 methods {RF (fs, top-10), RF (nofs, all-165), SvmW, SvmA, Lstm}
        x 6 modes {Pooled-base, Pooled-SW-SMOTE, Within-in, Within-out, Mixed-in, Mixed-out}
        (Cross excluded by request), with per-seed AUROC/AUPRC + degeneracy metrics.

This does NOT follow the exp2/TIV2026 Sobol variance-decomposition approach. It instead
characterises each MODEL x each experiment MODE with standard inferential statistics:
  (A) descriptives + 95% CIs (t and percentile bootstrap)
  (B) between-method Kruskal-Wallis per mode (+ eta^2_H, Dunn post-hoc w/ Holm, Cliff's delta)
  (C) within-method mode contrasts (seed-paired Wilcoxon signed-rank + rank-biserial effect)
  (D) model-characteristic quantities: RF fs-vs-nofs feature-count effect; decision-degeneracy
      metrics (predicted-probability spread, specificity, predicted-positive rate); seed-variance
      / stability (Brown-Forsythe); SvmA feature-signal probe (cited)
  (E) non-parametric two-way (method x mode) Scheirer-Ray-Hare test on a balanced common-seed subset
  (F) leaked-vs-honest validity contrast (recorded values carry a documented train/eval row-overlap
      leak; honest point estimates + EEG positive control are tabulated for correct attribution)

Only scipy/numpy/pandas are used (statsmodels/scikit-posthocs are unavailable); Dunn, Cliff's
delta and Scheirer-Ray-Hare are implemented here. Results are written to
results/analysis/exp3_verification/c1_statistical_analysis.json and printed as summary tables.
"""
import glob, os, re, json, math, itertools
import numpy as np
from scipy import stats

REPO = r"c:/git/work/vehicle_ddd_eval/vehicle_based_DDD_comparison"
os.chdir(REPO)
EVAL = "results/outputs/evaluation"
OUT = "results/analysis/exp3_verification/c1_statistical_analysis.json"
RNG = np.random.default_rng(20260726)

METHODS = ["RF_fs", "RF_nofs", "SvmW", "SvmA", "Lstm"]
MODES = ["pooled_base", "pooled_smote", "within_in", "within_out", "mixed_in", "mixed_out"]
MODE_LABEL = {"pooled_base": "Pooled-base", "pooled_smote": "Pooled-SW-SMOTE",
              "within_in": "Within-in", "within_out": "Within-out",
              "mixed_in": "Mixed-in", "mixed_out": "Mixed-out"}

# ---------------------------------------------------------------- data collection
def classify(model, basename):
    b = basename
    is_nofs = "_nofs_s" in b
    if model == "RF":
        method = "RF_nofs" if is_nofs else "RF_fs"
    else:
        method = model
    if "pooled_iv25base" in b: mode = "pooled_base"
    elif "pooled_iv25smote" in b: mode = "pooled_smote"
    elif "target_only_imbalv3_knn_wasserstein_in_domain" in b: mode = "within_in"
    elif "target_only_imbalv3_knn_wasserstein_out_domain" in b: mode = "within_out"
    elif "mixed_imbalv3_knn_wasserstein_in_domain" in b: mode = "mixed_in"
    elif "mixed_imbalv3_knn_wasserstein_out_domain" in b: mode = "mixed_out"
    else: return None
    m = re.search(r"_s(\d+)\.json$", b)
    if not m: return None
    return method, mode, m.group(1)

def cell_metrics(path):
    try:
        d = json.load(open(path))
    except Exception:
        return None
    auc = d.get("roc_auc")
    if auc is None or not (0 <= float(auc) <= 1):
        return None
    cm = d.get("confusion_matrix")
    spec = pred_pos = recall = np.nan
    if cm and len(cm) == 2:
        tn, fp = cm[0]; fn, tp = cm[1]
        n = tn + fp + fn + tp
        spec = tn / (tn + fp) if (tn + fp) else np.nan
        recall = tp / (tp + fn) if (tp + fn) else np.nan
        pred_pos = (fp + tp) / n if n else np.nan
    proba = d.get("y_pred_proba")
    pstd = float(np.std(proba)) if isinstance(proba, list) and proba else np.nan
    return dict(roc_auc=float(auc), auc_pr=(float(d["auc_pr"]) if d.get("auc_pr") is not None else np.nan),
                specificity=spec, recall_pos=recall, pred_pos_rate=pred_pos, proba_std=pstd)

# best (newest) per (method, mode, seed)
best = {}
for model in ["RF", "SvmW", "SvmA", "Lstm"]:
    for f in glob.glob(f"{EVAL}/{model}/**/eval_results_{model}_*.json", recursive=True):
        c = classify(model, os.path.basename(f))
        if not c: continue
        method, mode, seed = c
        mt = os.path.getmtime(f)
        k = (method, mode, seed)
        if k not in best or mt > best[k][0]:
            best[k] = (mt, f)

data = {}  # (method,mode) -> {seed: metrics}
for (method, mode, seed), (mt, f) in best.items():
    mm = cell_metrics(f)
    if mm is None: continue
    data.setdefault((method, mode), {})[seed] = mm

def arr(method, mode, key="roc_auc"):
    d = data.get((method, mode), {})
    return {s: v[key] for s, v in d.items() if not (isinstance(v[key], float) and math.isnan(v[key]))}

# ---------------------------------------------------------------- helpers
def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0: return np.nan
    gt = sum((x > b).sum() for x in a); lt = sum((x < b).sum() for x in a)
    return (gt - lt) / (len(a) * len(b))

def cliff_mag(d):
    ad = abs(d)
    return "negligible" if ad < 0.147 else "small" if ad < 0.33 else "medium" if ad < 0.474 else "large"

def boot_ci(x, n=2000, alpha=0.05):
    x = np.asarray(x, float)
    if len(x) < 2: return (np.nan, np.nan)
    res = stats.bootstrap((x,), np.mean, n_resamples=n, method="percentile",
                          confidence_level=1 - alpha, random_state=RNG)
    return (float(res.confidence_interval.low), float(res.confidence_interval.high))

def t_ci(x, alpha=0.05):
    x = np.asarray(x, float); n = len(x)
    if n < 2: return (np.nan, np.nan)
    m, se = x.mean(), x.std(ddof=1) / math.sqrt(n)
    h = stats.t.ppf(1 - alpha / 2, n - 1) * se
    return (m - h, m + h)

def dunn_holm(groups):
    """groups: dict name->array. Returns list of (a,b,z,p_holm)."""
    names = [k for k in groups if len(groups[k]) > 0]
    allv = np.concatenate([np.asarray(groups[k], float) for k in names])
    ranks = stats.rankdata(allv)
    idx = 0; rmean = {}; nsize = {}
    for k in names:
        n = len(groups[k]); rmean[k] = ranks[idx:idx + n].mean(); nsize[k] = n; idx += n
    N = len(allv)
    # tie correction
    _, cnt = np.unique(allv, return_counts=True)
    ties = sum(t**3 - t for t in cnt)
    sigma2 = (N * (N + 1) / 12) - ties / (12 * (N - 1))
    out = []
    for a, b in itertools.combinations(names, 2):
        se = math.sqrt(sigma2 * (1 / nsize[a] + 1 / nsize[b]))
        z = (rmean[a] - rmean[b]) / se if se else 0.0
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        out.append([a, b, z, p])
    # Holm
    order = sorted(range(len(out)), key=lambda i: out[i][3]); m = len(out)
    for rank, i in enumerate(order):
        out[i].append(min(1.0, out[i][3] * (m - rank)))
    return out

def scheirer_ray_hare(recs):
    """recs: list of (factorA, factorB, value). Balanced two-way SRH on ranks."""
    A = sorted(set(r[0] for r in recs)); B = sorted(set(r[1] for r in recs))
    vals = np.array([r[2] for r in recs], float)
    R = stats.rankdata(vals); N = len(R)
    MS_total = R.var(ddof=0) * N / (N - 1) if N > 1 else 1.0  # == variance of ranks *N/(N-1)
    MS = np.var(R) * N / (N - 1)
    grand = R.mean()
    def ss(group_key):
        s = 0.0
        for g in set(group_key):
            m = np.array([group_key[i] == g for i in range(N)])
            s += m.sum() * (R[m].mean() - grand) ** 2
        return s
    ga = [r[0] for r in recs]; gb = [r[1] for r in recs]; gab = [(r[0], r[1]) for r in recs]
    SS_A = ss(ga); SS_B = ss(gb); SS_cells = ss(gab); SS_AB = SS_cells - SS_A - SS_B
    dfA, dfB = len(A) - 1, len(B) - 1; dfAB = dfA * dfB
    H_A, H_B, H_AB = SS_A / MS, SS_B / MS, SS_AB / MS
    return {
        "factorA_H": H_A, "factorA_df": dfA, "factorA_p": 1 - stats.chi2.cdf(H_A, dfA),
        "factorB_H": H_B, "factorB_df": dfB, "factorB_p": 1 - stats.chi2.cdf(H_B, dfB),
        "interaction_H": H_AB, "interaction_df": dfAB, "interaction_p": 1 - stats.chi2.cdf(H_AB, dfAB),
    }

def fmt(x, p=3):
    return "nan" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:.{p}f}"

RESULTS = {}

# ---------------------------------------------------------------- (A) descriptives + CI
print("=" * 100)
print("(A) DESCRIPTIVES + 95% CI (AUROC).  cell = mean +/- SD (n) [t-CI] {bootstrap-CI}")
A = {}
for method in METHODS:
    row = {}
    for mode in MODES:
        d = arr(method, mode); v = np.array(list(d.values()), float)
        if len(v) == 0: row[mode] = None; continue
        tci = t_ci(v); bci = boot_ci(v) if len(v) >= 3 else (np.nan, np.nan)
        row[mode] = dict(n=len(v), mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                         t_ci=[float(tci[0]), float(tci[1])], boot_ci=[float(bci[0]), float(bci[1])],
                         seeds=sorted(d.keys()))
    A[method] = row
    print(f"\n  {method}")
    for mode in MODES:
        r = row[mode]
        if r is None: print(f"    {MODE_LABEL[mode]:16}: --"); continue
        print(f"    {MODE_LABEL[mode]:16}: {r['mean']:.3f}+/-{r['sd']:.3f} (n={r['n']:2}) "
              f"t[{fmt(r['t_ci'][0])},{fmt(r['t_ci'][1])}] boot[{fmt(r['boot_ci'][0])},{fmt(r['boot_ci'][1])}]")
RESULTS["A_descriptives"] = A

# ---------------------------------------------------------------- (B) between-method KW per mode
print("\n" + "=" * 100)
print("(B) BETWEEN-METHOD Kruskal-Wallis per mode (AUROC) + eta^2_H, Dunn(Holm), Cliff's delta")
B = {}
for mode in MODES:
    groups = {m: np.array(list(arr(m, mode).values()), float) for m in METHODS}
    groups = {m: v for m, v in groups.items() if len(v) >= 2}
    if len(groups) < 2:
        B[mode] = dict(note="insufficient groups"); print(f"\n  {MODE_LABEL[mode]}: insufficient data"); continue
    H, p = stats.kruskal(*groups.values())
    k = len(groups); N = sum(len(v) for v in groups.values())
    eta2 = (H - k + 1) / (N - k) if N > k else np.nan
    dunn = dunn_holm(groups) if p < 0.05 else []
    B[mode] = dict(H=float(H), p=float(p), k=k, N=N, eta2_H=float(eta2),
                   group_means={m: float(v.mean()) for m, v in groups.items()},
                   dunn=[[a, b, float(z), float(ph)] for a, b, z, p, ph in dunn])
    print(f"\n  {MODE_LABEL[mode]:16}: H={H:.2f} p={p:.4g} eta^2_H={fmt(eta2)} (k={k}, N={N})")
    order = sorted(groups, key=lambda m: -groups[m].mean())
    print("     rank(mean): " + " > ".join(f"{m}({groups[m].mean():.3f})" for m in order))
    if dunn:
        sig = [(a, b, ph) for a, b, z, p, ph in dunn if ph < 0.05]
        for a, b, ph in sig[:8]:
            print(f"       Dunn {a} vs {b}: p_holm={ph:.4g}, cliff_delta={fmt(cliffs_delta(groups[a],groups[b]),2)} ({cliff_mag(cliffs_delta(groups[a],groups[b]))})")
RESULTS["B_between_method"] = B

# ---------------------------------------------------------------- (C) within-method mode contrasts
print("\n" + "=" * 100)
print("(C) WITHIN-METHOD mode contrasts (seed-paired Wilcoxon signed-rank + rank-biserial r)")
CONTRASTS = [("imbalance_effect", "pooled_base", "pooled_smote"),
             ("domain_restriction_within", "pooled_smote", "within_in"),
             ("domain_restriction_mixed", "pooled_smote", "mixed_in"),
             ("domain_shift_within", "within_in", "within_out"),
             ("domain_shift_mixed", "mixed_in", "mixed_out")]
C = {}
for method in METHODS:
    C[method] = {}
    for name, m1, m2 in CONTRASTS:
        d1, d2 = arr(method, m1), arr(method, m2)
        common = sorted(set(d1) & set(d2))
        if len(common) < 3:
            C[method][name] = dict(n_pair=len(common), note="insufficient paired seeds"); continue
        x = np.array([d1[s] for s in common]); y = np.array([d2[s] for s in common])
        try:
            W, p = stats.wilcoxon(x, y)
        except ValueError:
            W, p = np.nan, np.nan
        n = len(common); rb = 1 - (2 * W) / (n * (n + 1)) if not math.isnan(W) else np.nan  # rank-biserial
        C[method][name] = dict(n_pair=n, median_diff=float(np.median(y - x)),
                               mean_diff=float((y - x).mean()), W=float(W) if not math.isnan(W) else None,
                               p=float(p) if not math.isnan(p) else None, rank_biserial=float(rb) if not math.isnan(rb) else None)
    parts = []
    for name, _, _ in CONTRASTS:
        r = C[method][name]
        if r.get("p") is None: parts.append(f"{name}=NA"); continue
        parts.append(f"{name}: dmed={r['median_diff']:+.3f} p={r['p']:.3g}")
    print(f"  {method:8}: " + " | ".join(parts))
RESULTS["C_mode_contrasts"] = C

# ---------------------------------------------------------------- (D) model-characteristic quantities
print("\n" + "=" * 100)
print("(D) MODEL CHARACTERISTICS")
D = {}
# D1 RF fs vs nofs per mode
print("  D1: RF feature-count effect (fs top-10 vs nofs all-165), Mann-Whitney + Cliff's delta")
d1 = {}
for mode in MODES:
    a = np.array(list(arr("RF_fs", mode).values()), float); b = np.array(list(arr("RF_nofs", mode).values()), float)
    if len(a) >= 2 and len(b) >= 2:
        U, p = stats.mannwhitneyu(b, a, alternative="two-sided")  # b=nofs vs a=fs
        dl = cliffs_delta(b, a)
        d1[mode] = dict(fs_mean=float(a.mean()), nofs_mean=float(b.mean()), delta_mean=float(b.mean() - a.mean()),
                        U=float(U), p=float(p), cliff=float(dl))
        print(f"    {MODE_LABEL[mode]:16}: fs={a.mean():.3f} nofs={b.mean():.3f} d={b.mean()-a.mean():+.3f} "
              f"p={p:.3g} cliff={dl:+.2f}({cliff_mag(dl)})")
D["D1_RF_feature_count"] = d1
# D2 degeneracy metrics
print("  D2: decision degeneracy — mean predicted-proba spread / specificity / predicted-positive rate")
d2 = {}
for method in METHODS:
    d2[method] = {}
    for mode in MODES:
        d = data.get((method, mode), {})
        if not d: continue
        ps = np.nanmean([v["proba_std"] for v in d.values()])
        sp = np.nanmean([v["specificity"] for v in d.values()])
        pp = np.nanmean([v["pred_pos_rate"] for v in d.values()])
        d2[method][mode] = dict(proba_std=float(ps), specificity=float(sp), pred_pos_rate=float(pp), n=len(d))
D["D2_degeneracy"] = d2
for method in ["SvmW", "SvmA", "RF_fs", "Lstm"]:
    for mode in ["pooled_base", "pooled_smote"]:
        r = d2.get(method, {}).get(mode)
        if r: print(f"    {method:6} {MODE_LABEL[mode]:15}: proba_std={r['proba_std']:.3f} spec={fmt(r['specificity'])} pred_pos={fmt(r['pred_pos_rate'])}")
# D3 seed-variance / stability (Brown-Forsythe across methods, per mode)
print("  D3: seed-variance / stability — per-method SD and Brown-Forsythe equal-variance test per mode")
d3 = {"per_method_sd": {}, "brown_forsythe": {}}
for method in METHODS:
    d3["per_method_sd"][method] = {mode: (float(np.std(list(arr(method, mode).values()), ddof=1))
                                          if len(arr(method, mode)) > 1 else None) for mode in MODES}
for mode in MODES:
    groups = [np.array(list(arr(m, mode).values()), float) for m in METHODS if len(arr(m, mode)) >= 2]
    if len(groups) >= 2:
        W, p = stats.levene(*groups, center="median")
        d3["brown_forsythe"][mode] = dict(W=float(W), p=float(p))
        print(f"    {MODE_LABEL[mode]:16}: Brown-Forsythe W={W:.2f} p={p:.3g}  (SD by method: " +
              ", ".join(f"{m}={fmt(np.std(list(arr(m,mode).values()),ddof=1) if len(arr(m,mode))>1 else float('nan'))}" for m in METHODS) + ")")
D["D3_seed_variance"] = d3
# D4 SvmA feature-signal probe (established)
D["D4_SvmA_feature_signal_probe"] = dict(univariate_max=0.515, multivariate_rbf=0.509, RF_on_SvmA_feats=0.496,
                                         note="Established probe (c1_results.md/domain_imbalance_factor_analysis.md): SvmA's 18 steering statistics carry no drowsiness signal even under a stronger learner.")
RESULTS["D_model_characteristics"] = D

# ---------------------------------------------------------------- (E) two-way SRH (balanced common-seed subset)
print("\n" + "=" * 100)
print("(E) TWO-WAY (method x mode) Scheirer-Ray-Hare on a balanced common-seed subset")
# Balanced factorial on the modes ALL 5 methods share (RF_nofs has no Pooled-base; Pooled-SW-SMOTE
# is still regenerating) -> the four Within/Mixed x in/out cells, on seeds common to every method.
core_modes = ["within_in", "within_out", "mixed_in", "mixed_out"]
# common seeds across all methods within each core mode, then intersect
per_method_seeds = {}
for method in METHODS:
    s = None
    for mode in core_modes:
        ss = set(arr(method, mode).keys())
        s = ss if s is None else (s & ss)
    per_method_seeds[method] = s or set()
common_seeds = set.intersection(*per_method_seeds.values()) if all(per_method_seeds.values()) else set()
recs = []
for method in METHODS:
    for mode in core_modes:
        d = arr(method, mode)
        for s in common_seeds:
            if s in d: recs.append((method, mode, d[s]))
E = {"core_modes": core_modes, "common_seeds": sorted(common_seeds), "n_records": len(recs)}
if len(common_seeds) >= 3 and len(recs) == len(METHODS) * len(core_modes) * len(common_seeds):
    srh = scheirer_ray_hare(recs); E["srh"] = srh
    print(f"  balanced design: {len(METHODS)} methods x {len(core_modes)} modes x {len(common_seeds)} seeds = {len(recs)} obs")
    print(f"    method   : H={srh['factorA_H']:.1f} df={srh['factorA_df']} p={srh['factorA_p']:.3g}")
    print(f"    mode     : H={srh['factorB_H']:.1f} df={srh['factorB_df']} p={srh['factorB_p']:.3g}")
    print(f"    method*mode: H={srh['interaction_H']:.1f} df={srh['interaction_df']} p={srh['interaction_p']:.3g}")
else:
    E["note"] = f"insufficient balanced data (common_seeds={sorted(common_seeds)})"
    print(f"  {E['note']}")
RESULTS["E_two_way_SRH"] = E

# ---------------------------------------------------------------- (F) leaked vs honest validity contrast
print("\n" + "=" * 100)
print("(F) LEAKED-vs-HONEST validity contrast (recorded values carry train/eval row-overlap leak)")
F = {
 "note": ("Recorded AUROCs above are computed under the paper's within-domain / pooled protocols "
          "which share a documented train/eval row overlap (~60% pooled, ~69% within). Honest = "
          "leakage-free subject-disjoint re-evaluation (0% overlap). Point estimates from c1_results.md "
          "3.6-3.7 + EEG positive control."),
 "honest": {"RF_fs": 0.517, "RF_nofs": 0.534, "SvmW": 0.520, "SvmA": 0.500,
            "Lstm_KSS": 0.51, "Lstm_DRT_crosssubject": 0.74, "EEG_bandpower_control": 0.61},
 "leaked_within_in_mean": {m: (A[m]["within_in"]["mean"] if A[m]["within_in"] else None) for m in METHODS},
}
print("   method   leaked(Within-in)  honest(leakage-free)")
for m in METHODS:
    lk = F["leaked_within_in_mean"][m]
    hon = F["honest"].get(m if m != "Lstm" else "Lstm_KSS")
    print(f"   {m:8}  {fmt(lk):>10}        {fmt(hon):>6}")
print(f"   EEG positive control (honest): {F['honest']['EEG_bandpower_control']}  |  Lstm-DRT cross-subject (honest): {F['honest']['Lstm_DRT_crosssubject']}")
RESULTS["F_leaked_vs_honest"] = F

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(RESULTS, open(OUT, "w"), indent=1, default=lambda o: None if (isinstance(o, float) and math.isnan(o)) else o)
print("\n" + "=" * 100)
print(f"written: {OUT}")
