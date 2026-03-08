# Experiment 3 (Prior Research Replication) Results

> **Revision**: Status after migration to the domain_train unified version (252 jobs).
> Data from the old split2 version (source_only/target_only, 504 jobs) is included as reference.

---

## Experiment Status

**Status:** Running (as of 2026-02-20)

### Job Submission and Completion Status (domain_train unified version)

| Model | Target | Completed (exit=0) | Running | Queued | Not submitted | Daemon |
|---|---|---|---|---|---|---|
| SvmW | 84 | 84 | 0 | 0 | 0 | Complete ✅ |
| SvmA | 84 | 0 | — | — | — | Running (auto_resub_unified_v2.sh) |
| Lstm | 84 | 0 | — | — | — | Running (auto_resub_unified_v2.sh) |

> Check SvmA / Lstm detailed progress with `qstat -u $USER`.

### SvmW Completion Verification (84 jobs exit=0): All normal ✅

Verification checklist:
- Model files (`SvmW_*.pkl`): 84/84 ✅
- Optuna study (`optuna_*.pkl`): 84/84 ✅
- Scaler (`scaler_*.pkl`): 84/84 ✅
- Evaluation result JSONs (`*_within.json` + `*_cross.json`): 168/168 ✅ (84 x 2 eval types)
- Training result JSONs: 84/84 ✅

> SvmW domain_train average execution time: ~15 min (significantly reduced from ~5 hours in the old split2 version)

---

## Known Issues

### 1. Lstm `seq_len` Bug (Resolved — old split2 version)

- **Cause:** Commit 278697c introduced a typo: `seg_len` -> `seq_len`
- **Impact:** All Lstm jobs submitted before the fix failed with `NameError: name 'seq_len' is not defined`
- **Fix:** Commit 49cf96e corrected `seq_len` -> `seg_len`
- **Resolution:** domain_train unified version uses the fixed code

### 2. SMOTE Condition Walltime Exceeded (Addressed)

- **Cause:** SMOTE/smote_plain conditions are computationally intensive, exceeding the original walltime
- **Fix:** Walltime increased (SvmW: 24h, SvmA: 48h, Lstm: 24h)

### 3. Evaluation Result Filename Overwrite Bug (Resolved)

- **Cause:** In domain_train mode, within and cross evaluations are run, but
  the old `save_eval_results` did not include `eval_type` in filenames,
  causing cross evaluation results to overwrite within evaluation results
- **Fix:** Added `eval_type` suffix to `src/utils/io/savers.py`
  - Results saved as separate files: `eval_results_*_within.json` / `eval_results_*_cross.json`
- **Resolution:** 83 SvmW results were fixed via manual rename + 1 job re-evaluated

### 4. Old split2 Version Training Duplication Issue (Resolved by domain_train)

- **Cause:** source_only and target_only were training the same model twice
  (e.g., in_domain target_only and out_domain source_only both trained on in_domain data)
- **Fix:** Unified to domain_train mode: train once per domain, evaluate twice

---

## Old split2 Version Reference Data (as of 2026-02-17)

<details>
<summary>Old split2 version status (reference)</summary>

### Job Submission and Completion Status (old split2 version)

| Model | Target | Completed (exit=0) | Notes |
|---|---|---|---|
| SvmW | 168 | 92 | Migrated to domain_train unified version |
| SvmA | 168 | 0 | Migrated to domain_train unified version |
| Lstm | 168 | 4 | Migrated to domain_train unified version |

### Accumulated eval Result Count (split2, including old jobs)

| Model | Eval count | Target |
|---|---|---|
| SvmW | 274 | 168 |
| SvmA | 233 | 168 |
| Lstm | 323 | 168 |

> Values exceeding the target include results from old settings/old code (before code fixes).
> Final analysis uses only domain_train unified version results.

</details>

---

## Related Documents

- [Experiment conditions](../conditions/03-prior-research-conditions.md) — Condition matrix and HPC resource settings
- [Reproducibility guide](../reproducibility.md) — How to reproduce experiments
- [Prior research models](../../architecture/prior_research.md) — Model architecture details
- [Optimization methods](exp3-analysis/optimization_methods.md) — Hyperparameter optimization summary
