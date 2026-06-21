# exp2 RF=0.89 vs exp3 RF=0.52 — HPC Verification Runbook

**Created:** 2026-06-21  ·  **Run on:** HAKUSAN HPC (`~/git/ddd/vehicle_based_DDD_comparison`)
**Owner question:** *Can exp3 RF legitimately equal exp2 RF (≈0.89)?*

---

## 0. TL;DR — the ONE decisive test

Run **Test A** (standalone sklearn, no `train.py`/TensorFlow needed) on the HPC `data/processed/common/`.
Read the two AUROC numbers it prints and apply the decision table:

| VEHICLE AUROC | EEG (Channel_*) AUROC | Verdict | Action |
|---|---|---|---|
| **≈ 0.50–0.55** | **≈ 0.9+** | exp2's 0.89 is a **leaky-split/protocol artifact**; data is honest | **exp3 = ~0.52 (honest). Correct exp2's 0.89.** Do NOT match. |
| **≈ 0.85–0.90** | ≈ 0.9+ | HPC data genuinely carries signal → **local 0.52 is a local-data defect** | **Re-run exp3 RF on HPC data** → it legitimately matches exp2. |
| ≈ 0.52 | **≈ 0.52** | label/eval setup broken (not a real result) | Fix label/columns first (send column list). |

> The EEG column is a **self-check**: EEG is the *source* of the drowsiness label, so a healthy eval pipeline MUST score ~0.9+ on EEG. If EEG is high and vehicle is ~0.5, then 0.52 is the genuine vehicle result, not a broken pipeline.

---

## 1. Why we are doing this (context established 2026-06-21)

Two bugs were found and **fixed locally** (uncommitted working-tree edits in `src/data_pipeline/features/`):
1. **`kss.py` label corruption** — `remove_outliers()` dropped rows then re-attached KSS labels *by position* → ~46% of labels mis-aligned. Affects **RF / SvmW / SvmA** (KSS label); **Lstm uses `event_label` → unaffected**. After fix: spearman(label, source ratio) 0.585→0.970; EEG→label AUROC 0.523→0.974.
2. **Feature-gen regression** — Feb-2026 "steering-only" commits restricted shared `time_freq_domain_process`/`wavelet_process` for `model_name="common"`, starving RF's feature pool. Fixed (model-aware).

**The open puzzle:** exp2 RF (HPC, stored) = **0.89**, but exp3 / current-local RF = **0.52**. We must know which is real before re-running, because the user wants the two papers (IV2015 + TIV2026) consistent and rigorous (no compromise).

### What has ALREADY been ruled out (do not re-investigate)
- ❌ **EEG feature leak** — exp2 RF `selected_features_*.pkl` checked on HPC: **0/40 contain `Channel_*`**; all are 10 steering features (`Steering_DominantFreq`, `SteeringSpeed_*`, …). exp2 did NOT use EEG.
- ❌ **Random-split fallback leak** — `grep "Class collapsed|fallback to random" logs/` = **0 firings**.
- ❌ **Mode (target_only vs domain_train)** — locally these give an *identical* split and AUROC≈0.49–0.52. So the 0.89↔0.52 gap is **HPC-data vs local-data**, NOT the eval mode.

### What we know about exp2's 0.89 (HPC stored eval JSONs)
`imbalv3 / wasserstein / in_domain`, by mode (median AUROC): **target_only = 0.900**, mixed = 0.839, source_only = 0.625.
→ Only the within-same-domain mode (`target_only`) is high. This *pattern* (target_only≫source_only) is what a clean subject-held-out split should NOT produce if the signal were real and stable; it is consistent with some within-domain leak/optimism on HPC data — hence Test A on the actual HPC data.

---

## 2. Test A — standalone sklearn RF (PRIMARY, no TF, no `train.py`)

`train.py` currently fails on HPC because `dispatch.py` eagerly imports Lstm → TensorFlow, and base `python3.13` TF is broken (`No module named 'tensorflow.python'`). This test avoids all of that.

```bash
cd ~/git/ddd/vehicle_based_DDD_comparison
python3 - <<'PY'
import glob, pandas as pd, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

files = sorted(glob.glob('data/processed/common/*.csv'))
print("common CSVs:", len(files))
dfs=[]
for f in files:
    d=pd.read_csv(f); d['__subj']=f.split('/')[-1]; dfs.append(d)
df=pd.concat(dfs, ignore_index=True)
print("rows:", len(df), " cols:", df.shape[1])

cands=[c for c in df.columns if ('KSS' in c or 'label' in c.lower()) and df[c].dropna().nunique()==2]
label='KSS_Theta_Alpha_Beta' if ('KSS_Theta_Alpha_Beta' in df.columns and df['KSS_Theta_Alpha_Beta'].dropna().nunique()==2) else (cands[0] if cands else None)
if label is None:
    print("NO binary label. KSS/label cols:", [c for c in df.columns if 'KSS' in c or 'label' in c.lower()]); raise SystemExit
print("LABEL:", label, " positive_rate:", round(df[label].mean(),3))

veh=[c for c in df.columns if c.startswith(('Steering','SteeringSpeed')) and 'Channel' not in c]
eeg=[c for c in df.columns if c.startswith('Channel')]
print(f"vehicle(steering) feats={len(veh)}  EEG(Channel_*) feats={len(eeg)}")

subs=sorted(df['__subj'].unique()); np.random.seed(42); np.random.shuffle(subs)
k=int(len(subs)*0.7); trs=set(subs[:k])
tr=df[df['__subj'].isin(trs)]; te=df[~df['__subj'].isin(trs)]
print(f"clean subject split -> train={len(tr)}(pos {tr[label].mean():.3f}) test={len(te)}(pos {te[label].mean():.3f})")

def auc(cols):
    if not cols: return None
    c=RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight='balanced')
    c.fit(tr[cols].fillna(0), tr[label].astype(int))
    return round(roc_auc_score(te[label].astype(int), c.predict_proba(te[cols].fillna(0))[:,1]),4)

print("\n=== within-domain (CLEAN subject-held-out) RF AUROC ===")
print("VEHICLE (steering):", auc(veh))
print("EEG (Channel_*)   :", auc(eeg), "  <- sanity: ~0.9+ means label/eval pipeline is healthy")
PY
```

**Record:** `LABEL`, `positive_rate`, `vehicle feats`, `EEG feats`, and the two AUROCs → apply the §0 table.

---

## 3. Test B — real pipeline RF in `domain_train` (CONFIRMATORY, needs a working env)

Only if you want the exact production-pipeline number. Requires a python env where `import tensorflow` works (the env that ran the HPC Lstm jobs) + sklearn/imblearn.

```bash
# find the working env / module setup the HPC jobs used:
conda env list
grep -rhE "conda activate|source .*activate|module load" scripts/hpc/ 2>/dev/null | sort -u | head

# activate it, then:
cd ~/git/ddd/vehicle_based_DDD_comparison
export PYTHONPATH=$PWD            # REQUIRED (else: No module named 'src')
export N_TRIALS_OVERRIDE=10       # fast
JID=$(date +%s)                   # numeric only (train.py strips non-numeric -> eval mismatch)
TF=results/analysis/exp2_domain_shift/distance/rankings/split2/knn/wasserstein_in_domain.txt

PBS_JOBID=$JID python3 scripts/python/train/train.py --model RF --mode domain_train --seed 42 \
  --target_file "$TF" --tag rfck_dt --time_stratify_labels \
  --use_oversampling --oversample_method smote --target_ratio 0.5 --subject_wise_oversampling

PBS_JOBID=$JID python3 scripts/python/evaluation/evaluate.py --model RF --tag rfck_dt \
  --mode domain_train --target_file "$TF" --eval_type within --jobid $JID 2>&1 | grep -iE "ROC AUC|No model|Found model"
```
The final `ROC AUC:` is the production within-domain RF on HPC data.

### Optional Test B2 — same data, `target_only` mode (isolate mode vs data)
Re-run Test B with `--mode target_only` (eval likewise `--mode target_only`). If `target_only` ≈ 0.90 but `domain_train` ≈ 0.52 on the SAME HPC data → the gap is the *mode/split protocol* (target_only leaks) → exp2's number is the suspect one.

---

## 4. Supporting checks (already run on HPC — kept for the record)

```bash
# (a) exp2 RF used EEG? -> answer was 0/40, n_features=10 (steering only)
python3 - <<'PY'
import joblib, glob, os, random
files = glob.glob('models/RF/**/selected_features_RF_*_split2_*.pkl', recursive=True)
print("exp2-era RF selected_features:", len(files))
s=random.Random(0).sample(files, min(40,len(files)))
nch=sum(any('Channel' in str(c) for c in joblib.load(f)) for f in s)
print(f"with Channel_*(EEG): {nch}/{len(s)} ; n_features(sample): {len(joblib.load(s[0]))}")
PY

# (b) exp2 RF AUROC by mode (imbalv3/wasserstein/in_domain) -> target_only median ~0.90
# (c) random-split fallback firing -> 0
grep -rh "Class collapsed\|fallback to random" logs/ 2>/dev/null | wc -l
```

---

## 5. Decision & next actions

- **If Test A VEHICLE ≈ 0.5 (and EEG ≈ 0.9+):**
  exp2's 0.89 is not a real vehicle→drowsiness signal. **Plan:** report exp3 RF = ~0.52; correct exp2's RF=0.890 claim; reframe the paper around the *honest* finding — *classical baselines (RF/SvmW/SvmA) ≈ chance on EEG-drowsiness; only Lstm with DRT-event labels (~0.82) transfers; prior high RF numbers were label/split artifacts*. Then re-run exp3 RF/SvmW/SvmA locally on the **fixed labels** (Lstm unaffected).

- **If Test A VEHICLE ≈ 0.85–0.90:**
  HPC data genuinely supports it. **Plan:** the local 0.52 came from a **local data defect** (feature/label regen). Re-run exp3 RF on the (correct) HPC data → it will legitimately match exp2; then reconcile local data generation.

- **Always:** the `kss.py` label-alignment fix and the feature-gen fix are correct and should be committed regardless (they were genuine bugs). Lstm does not need re-evaluation (uses `event_label`).

---

## 6. Status of the local re-run effort (other session)
- The local SvmA cuML run was **stopped** (it had run on the pre-fix corrupted labels). RF/SvmW/SvmA all need re-running on fixed labels once data is confirmed; Lstm is valid as-is.
- Local `data/processed/common` was being regenerated (fixed features+labels); model-specific `data/processed/{SvmA,SvmW}/` need regeneration before those re-runs (RF reads `common` directly).

---

**Report back:** paste the Test A output (LABEL line + the two AUROCs). That single result selects the branch in §5 and ends the investigation.
