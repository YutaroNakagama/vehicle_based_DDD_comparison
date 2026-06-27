"""exp3 T2 probe — is SvmW's 0.79 honest feature signal or split-dependent?

verification_tasks.md T2: SvmW's 8 GHM steering-wheel wavelet band-energies are
univariate ~0.510 / multivariate random-split ~0.485 (chance), yet B1 (within-
subject temporal split) reaches 0.79. Hypothesis: the 0.79 rides on within-subject
temporal structure, not latent drowsiness signal in the 8 bands.

This probe isolates the SPLIT effect with everything else held fixed (same 8
features, same SMOTE, same RBF-SVM + RF):
  A) within-subject temporal : per subject, sort by Timestamp, first 70% -> train,
     last 30% -> test (subjects appear in both, different time segments) -- mimics
     time_stratified_three_way_split, the B1 target_only path.
  B) subject-disjoint        : GroupShuffleSplit by subject_id (no subject overlap).

Verdict: A >> B (B ~ chance) -> 0.79 is split-dependent (write the SvmW conclusion
as "SMOTE restores the decision function but only exploits within-domain temporal
structure"). A ~ B and both lift -> genuine latent signal.

CPU-only, self-contained. Output: results/analysis/exp3_verification/t2_svmw_split_probe.json
"""
from __future__ import annotations
import glob, json, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[3]
import sys; sys.path.insert(0, str(REPO))

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from src.utils.io.split_helpers import _prepare_df_with_label_and_features

SW = ['SteeringWheel_DDD', 'SteeringWheel_DDA', 'SteeringWheel_DAD', 'SteeringWheel_DAA',
      'SteeringWheel_ADD', 'SteeringWheel_ADA', 'SteeringWheel_AAD', 'SteeringWheel_AAA']


def fit_eval(Xtr, ytr, Xte, yte, seed):
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    out = {}
    # cap SVM train for tractability (stratified)
    if len(ytr) > 8000:
        rng = np.random.RandomState(seed); idx = []
        for cl in np.unique(ytr):
            ci = np.where(ytr == cl)[0]; take = max(1, int(round(8000 * len(ci) / len(ytr))))
            idx += list(rng.choice(ci, min(take, len(ci)), replace=False))
        idx = np.array(sorted(idx)); Xs, ys = Xtr_s[idx], ytr[idx]
    else:
        Xs, ys = Xtr_s, ytr
    try:
        k = max(1, min(5, int(pd.Series(ys).value_counts().min()) - 1))
        Xr, yr = SMOTE(random_state=seed, k_neighbors=k).fit_resample(Xs, ys)
        svm = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed).fit(Xr, yr)
        out["svm_smote"] = round(roc_auc_score(yte, svm.predict_proba(Xte_s)[:, 1]), 4)
    except Exception as e:
        out["svm_smote"] = f"err:{e}"
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                random_state=seed, n_jobs=-1).fit(Xtr, ytr)
    out["rf"] = round(roc_auc_score(yte, rf.predict_proba(Xte)[:, 1]), 4)
    return out


def temporal_split(d):
    tr_idx, te_idx = [], []
    for sid, g in d.groupby("subject_id"):
        g = g.sort_values("Timestamp")
        cut = int(len(g) * 0.70)
        tr_idx += list(g.index[:cut]); te_idx += list(g.index[cut:])
    return np.array(tr_idx), np.array(te_idx)


def main():
    fr = []
    for fp in sorted(glob.glob(str(REPO / "data/processed/common/processed_*.csv"))):
        df = pd.read_csv(fp); df["subject_id"] = Path(fp).stem.replace("processed_", ""); fr.append(df)
    d, _ = _prepare_df_with_label_and_features(pd.concat(fr, ignore_index=True), model_name="SvmW")
    d = d.reset_index(drop=True)
    X = d[SW].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = d["label"].values
    print(f"[data] n={len(d)} subj={d['subject_id'].nunique()} pos={y.sum()} ({y.mean():.3%})")

    rep = {"n": len(d), "pos_rate": round(float(y.mean()), 4), "features": SW, "splits": {}}

    # A) within-subject temporal (deterministic)
    tr, te = temporal_split(d)
    a = fit_eval(X.iloc[tr].values, y[tr], X.iloc[te].values, y[te], 42)
    a["test_pos_rate"] = round(float(y[te].mean()), 4)
    rep["splits"]["within_subject_temporal"] = a
    print(f"[A within-subject temporal] {a}")

    # B) subject-disjoint (3 seeds -> mean)
    bs = []
    for s in (42, 123, 2025):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=s)
        tr, te = next(gss.split(d, y, groups=d["subject_id"]))
        r = fit_eval(X.iloc[tr].values, y[tr], X.iloc[te].values, y[te], s)
        bs.append(r); print(f"[B subject-disjoint s{s}] {r}")
    def _m(key):
        vals = [b[key] for b in bs if isinstance(b[key], (int, float))]
        return round(float(np.mean(vals)), 4) if vals else None
    rep["splits"]["subject_disjoint"] = {"per_seed": bs,
                                         "svm_smote_mean": _m("svm_smote"), "rf_mean": _m("rf")}
    print(f"[B subject-disjoint MEAN] svm_smote={_m('svm_smote')} rf={_m('rf')}")

    outp = REPO / "results/analysis/exp3_verification/t2_svmw_split_probe.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(rep, indent=2))
    print(f"[saved] {outp}")


if __name__ == "__main__":
    main()
