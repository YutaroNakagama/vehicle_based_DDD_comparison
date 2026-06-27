"""SvmW (Zhao 2009) faithfulness check — per-EXPERIMENT (per-session) labeling.

IV2025's framework classifies per time-window for ALL methods (fair comparison).
Zhao's ORIGINAL, however, assigned ONE label per entire experiment (no windows).
This script reproduces that literal variant for SvmW and reports AUROC, to check
whether SvmW's advantage survives Zhao's own per-session labeling.

Method (per session = subject_id):
  * Features: mean of the 8 GHM multiwavelet-packet band energies over the
    session's windows (one 8-dim vector per session) — wavelet-only, faithful.
  * Label:   session is "drowsy" if its fraction of drowsy windows is above the
    cohort median (balanced binary; the per-window positive rate is too low for a
    majority vote, which is itself a finding about the MMDAP dataset).
  * Eval:    leave-one-subject-out (LOSO) — each session predicted once; AUROC
    over all sessions. (SVM is near-deterministic, so seeds add little; reported.)

Run:  python scripts/python/analysis/svmw_zhao_persession.py
"""
from __future__ import annotations
import sys; sys.path.insert(0, ".")
import numpy as np, glob, re
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
from src.utils.io.loaders import load_subject_csvs
from src.utils.io.split_helpers import _prepare_df_with_label_and_features

WAVELET = ["SteeringWheel_DDD","SteeringWheel_DDA","SteeringWheel_DAD","SteeringWheel_DAA",
           "SteeringWheel_ADD","SteeringWheel_ADA","SteeringWheel_AAD","SteeringWheel_AAA"]

def main():
    subs=[l.strip() for l in open("results/analysis/exp2_domain_shift/distance/rankings/split2/knn/wasserstein_in_domain.txt") if l.strip()]
    data,_=load_subject_csvs(subs, model_name=None, add_subject_id=True, base_path="data/processed/common")
    df,_=_prepare_df_with_label_and_features(data, model_name="SvmW")  # adds binary 'label' (KSS, paper mapping)
    # aggregate to one row per session
    rows=[]
    for sid,g in df.groupby("subject_id"):
        feats=g[WAVELET].mean().values
        pos_rate=g["label"].mean()
        rows.append((sid, feats, pos_rate))
    X=np.array([r[1] for r in rows]); rates=np.array([r[2] for r in rows]); groups=np.array([r[0] for r in rows])
    y=(rates>np.median(rates)).astype(int)   # balanced session-level label
    print(f"[per-session] sessions={len(y)}  drowsy={y.sum()}  (window pos-rate range {rates.min():.3f}-{rates.max():.3f})")

    # LOSO AUROC across seeds (SVM near-deterministic; vary seed for robustness)
    aurocs=[]
    for seed in [0,1,7,13,42,123,256,512,1337,2024,2025]:
        logo=LeaveOneGroupOut(); proba=np.zeros(len(y))
        for tr,te in logo.split(X,y,groups):
            sc=MinMaxScaler().fit(X[tr])
            clf=SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=seed).fit(sc.transform(X[tr]), y[tr])
            proba[te]=clf.predict_proba(sc.transform(X[te]))[:,1]
        try: aurocs.append(roc_auc_score(y, proba))
        except Exception: pass
    import statistics as st
    print(f"[per-session SvmW LOSO] AUROC mean={st.mean(aurocs):.3f} sd={st.pstdev(aurocs):.3f} n_seeds={len(aurocs)}")
    print("  (compare: windowed SvmW B1 target_only ~0.78)")

if __name__=="__main__":
    main()
