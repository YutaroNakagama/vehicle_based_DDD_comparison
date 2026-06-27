#!/usr/bin/env python3
"""verify_rf_feature_parity.py
================================
Decide whether exp3's local RF collapse (within-domain AUROC ~0.52) is caused
by an INCOMPLETE local `data/processed/common/*.csv` (only steering-signal
statistical + wavelet features present) versus the full 145-feature set that
exp2's HPC RF used.

Run BEFORE and AFTER copying the HPC 145-feature `common` CSVs into
`data/processed/common/`:

    python scripts/shell/verify_rf_feature_parity.py
    python scripts/shell/verify_rf_feature_parity.py --domain out_domain --distance wasserstein

What it does (no Optuna, fast):
  1. Reports whether the full RF feature range end-column `LaneOffset_AAA`
     (and other non-steering statistical/wavelet blocks) is present.
  2. Reproduces the exp3 domain_train within split via the project's own
     `split_data_domain_train`, trains a default RF, reports within AUROC/AUPRC.
  3. Also trains on the explicit Wang-15 fallback set for reference.

Interpretation:
  * If, after copying HPC data, full-feature RF within AUROC jumps to ~0.8+,
    exp2's RF=0.890 is legitimate and exp3 RF must be re-run on the full data.
  * If it stays ~0.52 even with the full 145 features, exp2's 0.87 is suspect.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

# Columns that should exist ONLY when the full 145-feature CSV is present.
FULL_FEATURE_SENTINELS = [
    "LaneOffset_AAA",        # lane-offset wavelet octet end (RF range end col)
    "LatAcc_Range", "LongAcc_Range", "LaneOffset_Range",  # non-steering statistical
]


def _clean(df):
    num = df.select_dtypes(include=[np.number]).fillna(0.0)
    num = num.replace([np.inf, -np.inf], 0.0).clip(-1e18, 1e18)
    return num


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="in_domain", choices=["in_domain", "out_domain"])
    ap.add_argument("--distance", default="wasserstein", choices=["mmd", "dtw", "wasserstein"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    target_file = (
        REPO / "results/analysis/exp2_domain_shift/distance/rankings/split2"
        / "knn" / f"{args.distance}_{args.domain}.txt"
    )
    subjects = [l.strip() for l in open(target_file) if l.strip()]
    print(f"[INFO] {args.domain}/{args.distance}: {len(subjects)} subjects")

    # Column presence check on one CSV
    import pandas as pd
    one = next((REPO / "data/processed/common").glob("processed_*.csv"))
    cols = pd.read_csv(one, nrows=1).columns.tolist()
    print(f"[INFO] common CSV has {len(cols)} columns ({one.name})")
    for c in FULL_FEATURE_SENTINELS:
        print(f"   {'OK ' if c in cols else 'MISSING'}  {c}")
    full_present = all(c in cols for c in FULL_FEATURE_SENTINELS)
    print(f"[INFO] Full 145-feature set present: {full_present}\n")

    from src.utils.io.split_helpers import split_data_domain_train

    Xtr, Xva, Xte, ytr, yva, yte = split_data_domain_train(
        subjects=subjects, model_name="RF", seed=args.seed,
        train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, keep_subject_id=False,
    )

    def fit_eval(feat_cols, label):
        cols_use = [c for c in feat_cols if c in Xtr.columns]
        if not cols_use:
            print(f"   [{label}] no usable columns present — skipped")
            return
        a = _clean(Xtr[cols_use]); b = _clean(Xte[cols_use])
        rf = RandomForestClassifier(
            n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=42
        ).fit(a, ytr)
        p = rf.predict_proba(b)[:, 1]
        print(f"   [{label}] nfeat={len(cols_use):3d}  within AUROC={roc_auc_score(yte, p):.3f}"
              f"  AUPRC={average_precision_score(yte, p):.3f}")

    print(f"[RESULT] test n={len(yte)} pos%={yte.mean()*100:.1f}")
    # What the pipeline actually resolved for RF (whatever split returned):
    fit_eval(list(Xtr.columns), "RF resolved feature set (pipeline)")
    # Wang-15 fallback (current local behaviour):
    wang = [c for c in Xtr.columns if c.endswith(("_std_dev", "_pred_error", "_mean"))
            and c[0].islower()]
    fit_eval(wang, "Wang-15 fallback (snake_case)")
    print("\n[DONE] If full set present and AUROC jumps to ~0.8+, exp2 RF is legit "
          "and exp3 RF must be re-run on the full-feature data.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
