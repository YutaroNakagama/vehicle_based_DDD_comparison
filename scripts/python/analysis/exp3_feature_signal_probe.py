"""exp3 T1 feature-signal probe — does the FAITHFUL Arefnezhad feature set carry signal?

verification_tasks.md T1: the prior "SvmA has no signal" (univariate 0.515 /
multivariate 0.509) was measured on an UNFAITHFUL 14-suffix filter that dropped
Arefnezhad's main features (Sample Entropy, Katz FD, Shannon Entropy, Spectral
Flux, Q1/Q2/Q3) and added 4 non-paper ones (Mean/Var/Max/Min). The SvmA.py fix
(2026-06-27) restores the faithful 18 features/signal (36 total). This probe
re-measures feature signal on the FAITHFUL set, classifier-light:

  (a) univariate directionless AUROC per feature (max(auc, 1-auc))
  (b) multivariate RBF-SVM  (raw + SW-SMOTE)   on a SUBJECT-DISJOINT split
  (c) multivariate RF (class_weight=balanced)  -> classifier-independent check (T4 half)

Run for BOTH the faithful-18 and the legacy-14 sets for direct comparison (plot #1).
Verdict: if faithful set still <0.55 -> "no signal" is STRENGTHENED (paper features
included, still chance). If it jumps -> the old null was a feature-filter artifact.

Output: results/analysis/exp3_verification/t1_feature_signal_probe.json (+ stdout).
CPU-only, self-contained (reuses the pipeline's SvmA label mapping). Does not touch
the running c1 jobs.
"""
from __future__ import annotations
import glob, json, logging, os, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE

from src.utils.io.split_helpers import _prepare_df_with_label_and_features
from src.models.architectures.SvmA import SVMA_PAPER_FEATURE_SUFFIXES

# Legacy 14-suffix set (pre-fix) for the comparison arm.
LEGACY_14 = ['Mean', 'Variance', 'StdDev', 'Max', 'Min', 'Range', 'Energy',
             'Skewness', 'Kurtosis', 'ZeroCrossingRate',
             'DominantFreq', 'FreqCOG', 'SpectralEntropy', 'AvgPSD']
SEED = 42


def load_all_common() -> pd.DataFrame:
    frames = []
    for fp in sorted(glob.glob(str(REPO / "data" / "processed" / "common" / "processed_*.csv"))):
        sid = Path(fp).stem.replace("processed_", "")
        df = pd.read_csv(fp)
        df["subject_id"] = sid
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No data/processed/common/processed_*.csv found.")
    return pd.concat(frames, ignore_index=True)


def select(cols, suffixes):
    sig = [c for c in cols if c.startswith("Steering_") or c.startswith("SteeringSpeed_")]
    return [c for c in sig if any(c.endswith(s) for s in suffixes)]


def subject_disjoint_split(df, feat_cols):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    tr, te = next(gss.split(df, df["label"], groups=df["subject_id"]))
    Xtr = df.iloc[tr][feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    Xte = df.iloc[te][feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    ytr = df.iloc[tr]["label"].values
    yte = df.iloc[te]["label"].values
    return Xtr, Xte, ytr, yte


def univariate(df, feat_cols):
    """Directionless univariate AUROC per feature on the full pooled set."""
    y = df["label"].values
    out = {}
    for c in feat_cols:
        x = df[c].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
        try:
            a = roc_auc_score(y, x)
            out[c] = round(max(a, 1 - a), 4)
        except Exception:
            out[c] = None
    return out


def _subsample(X, y, cap=8000):
    """Stratified subsample to keep RBF-SVM (O(n^2)) tractable."""
    if len(y) <= cap:
        return X, y
    rng = np.random.RandomState(SEED)
    idx = []
    for cls in np.unique(y):
        ci = np.where(y == cls)[0]
        take = max(1, int(round(cap * len(ci) / len(y))))
        idx.extend(rng.choice(ci, size=min(take, len(ci)), replace=False))
    idx = np.array(sorted(idx))
    return X[idx], y[idx]


def multivariate(Xtr, Xte, ytr, yte):
    res = {}
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    Xsv, ysv = _subsample(Xtr_s, ytr)  # cap for SVM only
    res["svm_train_n"] = int(len(ysv))
    # RBF-SVM raw
    svm = SVC(kernel="rbf", probability=True, random_state=SEED, class_weight=None).fit(Xsv, ysv)
    res["svm_rbf_raw"] = round(roc_auc_score(yte, svm.predict_proba(Xte_s)[:, 1]), 4)
    # RBF-SVM + SW-SMOTE (train only); guard tiny minority
    try:
        k = max(1, min(5, int(pd.Series(ysv).value_counts().min()) - 1))
        Xr, yr = SMOTE(random_state=SEED, k_neighbors=k).fit_resample(Xsv, ysv)
        svm2 = SVC(kernel="rbf", probability=True, random_state=SEED).fit(Xr, yr)
        res["svm_rbf_smote"] = round(roc_auc_score(yte, svm2.predict_proba(Xte_s)[:, 1]), 4)
    except Exception as e:
        res["svm_rbf_smote"] = f"err:{e}"
    # RF (classifier-independent check) on the SAME SvmA features
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
    res["rf_on_svma_feats"] = round(roc_auc_score(yte, rf.predict_proba(Xte)[:, 1]), 4)
    return res


def main():
    df = load_all_common()
    d, _ = _prepare_df_with_label_and_features(df, model_name="SvmA")
    pos = int(d["label"].sum()); n = len(d)
    print(f"[data] n={n} subjects={d['subject_id'].nunique()} pos={pos} ({pos/n:.3%})")

    report = {"n": n, "n_subjects": int(d["subject_id"].nunique()),
              "pos_rate": round(pos / n, 4), "seed": SEED, "sets": {}}
    for name, suff in [("faithful_18", SVMA_PAPER_FEATURE_SUFFIXES), ("legacy_14", LEGACY_14)]:
        cols = select(d.columns, suff)
        uni = univariate(d, cols)
        Xtr, Xte, ytr, yte = subject_disjoint_split(d, cols)
        mv = multivariate(Xtr, Xte, ytr, yte)
        top = sorted([(v, k) for k, v in uni.items() if v is not None], reverse=True)[:6]
        report["sets"][name] = {
            "n_features": len(cols),
            "univariate_max": max([v for v in uni.values() if v is not None], default=None),
            "univariate_top6": [{"feature": k, "auroc": v} for v, k in top],
            "univariate_all": uni,
            "multivariate": mv,
            "split": {"train_pos_rate": round(float(ytr.mean()), 4),
                      "test_pos_rate": round(float(yte.mean()), 4),
                      "n_train": int(len(ytr)), "n_test": int(len(yte))},
        }
        print(f"\n=== {name} ({len(cols)} cols) ===")
        print(f"  univariate max         : {report['sets'][name]['univariate_max']}")
        print(f"  univariate top6        : {[(k, v) for v, k in top]}")
        print(f"  multivariate           : {mv}")

    outdir = REPO / "results" / "analysis" / "exp3_verification"
    outdir.mkdir(parents=True, exist_ok=True)
    outp = outdir / "t1_feature_signal_probe.json"
    outp.write_text(json.dumps(report, indent=2))
    print(f"\n[saved] {outp}")


if __name__ == "__main__":
    main()
