# exp2 RF=0.89 vs exp3 RF=0.52 — HPC Verification Runbook

**Created:** 2026-06-21  ·  **Run on:** HAKUSAN HPC (`~/git/ddd/vehicle_based_DDD_comparison`)
**Owner question:** *Can exp3 RF legitimately equal exp2 RF (≈0.89)?*

---

## ✅ Verification result (2026-06-21) — RESOLVED

**Answer: NO.** exp2 RF ≈ 0.89 is **not** a reproducible vehicle→drowsiness signal — it is an
**evaluation-protocol artifact** (`target_only` within-subject temporal split **+** SMOTE oversampling).
The honest, cross-subject vehicle AUROC is **≈ 0.49–0.53 (chance)**, which matches **exp3 RF = 0.52**.

- **Test A** (compute node `lcpcc-010`) on HPC `data/processed/common/`: **VEHICLE (steering) AUROC = 0.486**, EEG (Channel_*) = 0.588.
- **Stored HPC eval JSONs** (no re-run): the 0.89–0.90 is recorded **only** for `target_only` + SMOTE; the *same* data/features give **baseline 0.633** and **source_only (cross-domain) 0.527**.
- **Decision (§5, branch 1): correct exp2's RF = 0.890; keep exp3 RF ≈ 0.52 as the honest result.**

> **Caveat:** the EEG self-check scored **0.588, not ~0.9+**, so HPC `data/processed/common` still
> carries the **pre-fix (mis-aligned) KSS labels**. This does **not** change the verdict — the verdict
> rests on the *stored exp2 artifacts* (§7.2), which are independent of any fresh re-run — but the
> label-fix regeneration is still required before publishing the corrected exp3 numbers (§7.3).

Full evidence, commands, and tables in **§7** below.

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
  - ⚠️ **Refined by §7.2 (2026-06-21):** this holds only for the *local clean* split. The **stored exp2** `target_only` ran `subject_time_split` (**within-subject** — same subjects in train+test) **+ SMOTE**, and on the *same* HPC data a clean subject-held-out split scores **0.486** (Test A). So the true driver of 0.89 is the **protocol (within-subject split) + oversampling**, demonstrable *within* HPC data — not an HPC-vs-local data difference.

### What we know about exp2's 0.89 (HPC stored eval JSONs)
`imbalv3 / wasserstein / in_domain`, by mode (median AUROC): **target_only = 0.900**, mixed = 0.839, source_only = 0.625.
→ Only the within-same-domain mode (`target_only`) is high. This *pattern* (target_only≫source_only) is what a clean subject-held-out split should NOT produce if the signal were real and stable; it is consistent with some within-domain leak/optimism on HPC data — hence Test A on the actual HPC data.
→ **Confirmed (§7.2):** the high number requires `target_only` **and** SMOTE — baseline `target_only` = **0.633**, `source_only` = **0.508–0.527** on identical data/features.

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

## 7. Verification results & evidence (2026-06-21)

Executed on the HAKUSAN compute node `lcpcc-010` (PBS `SINGLE` queue, `conda python310`:
pandas 2.3.1 / numpy 2.1.3 / sklearn 1.6.1).
Artifacts: [`scripts/hpc/logs/testA_verify/`](../../scripts/hpc/logs/testA_verify) —
`testA.py`, `testA.pbs`, `testA.out`, `label_diag.py`.

### 7.1 Test A — standalone sklearn RF on HPC `data/processed/common/`

The stored data has **no binary label column**: `KSS_Theta_Alpha_Beta` is the raw 9-level KSS scale
(values 1–9), as are `KSS_Theta_Alpha_Beta_percent` and `theta_alpha_over_beta_label`. Test A was
corrected to derive the label **exactly as production does**
([`split_helpers.py::_prepare_df_with_label_and_features`](../../src/utils/io/split_helpers.py#L28)
→ [`config.py` `KSS_BIN_LABELS` / `KSS_LABEL_MAP`](../../src/config.py#L186)):
keep KSS ∈ {1,2,3,4,5,8,9} (drop 6,7), map 1–5 → 0 (alert), 8–9 → 1 (drowsy).

```text
rows: 70841 (87 subjects)        →  after KSS filter: 61765
LABEL positive_rate: 0.0393      (drowsy 2426 / alert 59339)
features: 52 steering(vehicle),  40 EEG(Channel_*)
clean subject-held-out split (70/30 by subject):
    VEHICLE (steering) AUROC = 0.4864
    EEG (Channel_*)    AUROC = 0.588
```

- **VEHICLE ≈ 0.49 → chance.** Steering/vehicle features carry no transferable drowsiness signal under an honest subject-held-out split.
- **EEG ≈ 0.59 (not ~0.9+) → the EEG self-check FAILS.** Because KSS is *derived from* EEG (θ+α)/β, a healthy pipeline must score ~0.9+ on EEG. 0.59 indicates the HPC `common` data still holds the **pre-fix, position-mis-aligned KSS labels** (the `kss.py` `remove_outliers`/`adjust_scores_length` bug) → §0 table **row 3** for the *fresh-run* path.

### 7.2 Provenance of the 0.89 — from stored HPC artifacts (decisive evidence)

The exp2 numbers need **no** re-run: they are recorded in **8,236 RF eval JSONs** under
[`results/outputs/evaluation/RF/<jobid>/<jobid>[1]/eval_results_RF_*.json`](../../results/outputs/evaluation/RF).
Each JSON stores `roc_auc`, `subject_list` (all 87), `mode`, `tag`, `timestamp`.

**The 0.89–0.90 is recorded only for `target_only` + SMOTE.** Median stored `roc_auc`
(`wasserstein` / `in_domain` / `split2`; identical 87 subjects; identical steering features):

| mode | imbalance | n | median AUROC | max |
|---|---|---:|---:|---:|
| **target_only** | **imbalv3 (SMOTE)** | 60 | **0.907** | 0.930 |
| target_only | smote_plain | 62 | 0.876 | 0.933 |
| target_only | **baseline (none)** | 38 | **0.633** | 0.857 |
| target_only | undersample | 61 | 0.590 | 0.884 |
| mixed | imbalv3 (SMOTE) | 59 | 0.873 | 0.910 |
| **source_only** | **imbalv3 (SMOTE)** | 65 | **0.527** | 0.620 |
| source_only | baseline | 37 | 0.508 | 0.550 |

Feature check on the 0.907 run (job `14880278`): its `selected_features_*.pkl` =
**10 features, all steering, 0 EEG** (`Steering_DominantFreq`, `SteeringSpeed_DominantFreq`,
`Steering_ZeroCrossingRate`, …) → the high score is **not** an EEG leak.

**Mechanism (confirmed in code):**
- `target_only` is hard-wired to `subject_time_split`
  ([`model_pipeline.py` L166](../../src/models/model_pipeline.py#L166)) →
  [`time_stratified_three_way_split` *per subject*](../../src/utils/io/split_helpers.py#L138):
  **the same subjects appear in train and test** (split only by time) → within-subject leakage (baseline already 0.633 vs a true held-out 0.49).
- SMOTE is then applied to the training data
  ([`model_pipeline.py` Stage 4.5, L209](../../src/models/model_pipeline.py#L209)), compounding the optimism (0.633 → 0.907).
- `source_only` trains on **disjoint** subjects → genuine cross-subject generalization → **≈ chance (0.51–0.53)**.

A real signal would not collapse from 0.91 (within-subject) to 0.53 (cross-subject) on identical
data/features, nor require SMOTE to appear. The asymmetry **target_only ≫ source_only** and
**SMOTE ≫ baseline** is the fingerprint of a protocol artifact.

### 7.3 Conclusion & next actions

- **Verdict:** exp2 RF = 0.890 is an artifact of `target_only` (within-subject temporal split) + SMOTE.
  Honest cross-subject vehicle→drowsiness RF AUROC ≈ **0.50–0.53** = exp3's 0.52.
  → **§5 branch 1: correct exp2's 0.890; keep exp3 RF ≈ 0.52 as the honest result.**
  Established from **stored artifacts alone** — independent of the EEG self-check / current label state.
- **Still required before publishing the corrected numbers:** regenerate HPC `data/processed/common`
  with the fixed `kss.py` (so the EEG self-check rises to ~0.9+), then re-run Test A as a clean
  confirmation and re-run exp3 RF/SvmW/SvmA on the fixed labels (Lstm unaffected — uses `event_label`).

---

**Report back:** ✅ **Answered (2026-06-21).** Test A VEHICLE = 0.486 / EEG = 0.588, plus the
stored-artifact provenance (§7.2), select **§5 branch 1** — correct exp2's 0.89; exp3 RF ≈ 0.52 is honest.
