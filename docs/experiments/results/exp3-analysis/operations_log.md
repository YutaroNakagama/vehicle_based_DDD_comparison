# Experiment 3 — Operations Log (Known Issues & Job Resubmissions)

> Spun out of `03-prior-research-results.md` (2026-04-29) so the main results doc
> can focus on findings rather than HPC operational history.

## Final Batch (24 jobs submitted 2026-04-23, canonical 2-seed scope)

- **SvmA × 1**: `prior_SvmA_smote_plain_knn_wasserstein_out_domain_..._ratio0.5_s42` → PBS 15101 (SINGLE, walltime=48h)
- **Lstm × 21**: all `ratio=0.1` conditions for `smote_plain` (10) and `undersample_rus` (11) → PBS 15102–15122 (GPU queues, walltime=24h)
- **Lstm × 2 (cross-only re-runs)**: `smote_plain mmd in_domain s123` and `smote_plain mmd out_domain s42` had `_within.json` only (cross eval previously failed) → PBS 15123–15124 (GPU-1A/GPU-S, walltime=24h)

## Re-submission of Missing 24 Tags (2026-04-29)

Triggered after audit revealed the 2026-04-23 batch silently lost most Lstm jobs (Issue 6 below).

- Submitter: [`scripts/hpc/launchers/submit_exp3_missing_24_v2.sh`](../../../../scripts/hpc/launchers/submit_exp3_missing_24_v2.sh)
- Composition: 1 SvmA (timed-out re-try) + 23 Lstm (CUDA init failures)
- PBS IDs: **20199 (SvmA, LONG, 48h)** and **20200–20236 (Lstm, GPU-1/1A/L/LA/S, 24h)**
- Required pre-step: ~75 non-canonical extension Lstm jobs were `qdel`'d to free per-QOS submit limits (`gpu-1` 30, `gpu-1a` 20, `gpu-s` 15, `gpu-l` 5, `gpu-la` 5).

---

## Known Issues

### 1. Lstm `seq_len` Bug (Resolved — old split2 version)

- **Cause:** Commit 278697c introduced a typo: `seg_len` -> `seq_len`
- **Impact:** All Lstm jobs submitted before the fix failed with `NameError: name 'seq_len' is not defined`
- **Fix:** Commit 49cf96e corrected `seq_len` -> `seg_len`
- **Resolution:** domain_train unified version uses the fixed code

### 2. SMOTE Condition Walltime Exceeded (Addressed)

- **Cause:** SMOTE/smote_plain conditions are computationally intensive, exceeding the original walltime
- **Fix:** Walltime increased (SvmW: 24h, SvmA: 48h, Lstm: 24h) — see conditions doc table

### 3. Evaluation Result Filename Overwrite Bug (Resolved)

- **Cause:** In domain_train mode, within and cross evaluations are run, but
  the old `save_eval_results` did not include `eval_type` in filenames,
  causing cross evaluation results to overwrite within evaluation results
- **Fix:** Added `eval_type` suffix to `src/utils/io/savers.py`
  - Results saved as separate files: `eval_results_*_within.json` / `eval_results_*_cross.json`
- **Resolution:** 83 SvmW results were fixed via manual rename + 1 job re-evaluated

### 4. Old split2 Version Training Duplication Issue (Resolved by domain_train)

- **Cause:** source_only and target_only were training the same model twice
- **Fix:** Unified to domain_train mode: train once per domain, evaluate twice

### 5. Lstm Cross-Eval Silent Failures (Detected 2026-04-23)

- **Symptom:** 2 Lstm tags (`smote_plain mmd in_domain s123`, `smote_plain mmd out_domain s42`)
  had `_within.json` produced but `_cross.json` was missing.
- **Detection:** Audit by tag (rather than by file count) revealed the gap;
  raw file counts (within=63, cross=61) did not match.
- **Fix:** Re-submitted as PBS 15123–15124 (GPU-1A/GPU-S, walltime=24h).
  Both completed with exit 0 and produced the missing `_cross.json` files.

### 6. Lstm CUDA Init Failures Reported as exit 0 (Detected 2026-04-29, Fixed 2026-04-29)

- **Symptom:** All 21 Lstm jobs in the 2026-04-23 final batch (PBS 15102–15122)
  printed `Error loading CUDA libraries. GPU will not be used.` early in the
  log, then the wrapper printed `[DONE] Unified GPU job completed (exit code: 0)`.
  No `*.keras` model was saved, and the subsequent eval steps logged
  `[EVAL] No model file found … Aborted.`
- **Impact:** No `_within.json` / `_cross.json` was produced, so the canonical
  exp3 set still missed 23 Lstm tags after the batch.
- **Detection:** Re-audit on 2026-04-29 by reproducing the missing-tag list.
- **Root cause:** `hpc_sdk/22.2` module was removed from the cluster and replaced
  by `nvhpc/26.3` family. The GPU PBS wrapper silently ignored the missing module
  (`|| true`), leaving `libcudart.so.12` and `libcudnn.so.9` unavailable.
  Conda env `python310` only has `cudatoolkit 11.8` which is incompatible with
  TF 2.19.0 (built against CUDA 12.5.1 + cuDNN 9).
- **Fix (2026-04-29):**
  - Updated [`pbs_prior_research_unified_gpu.sh`](../../../../scripts/hpc/jobs/train/pbs_prior_research_unified_gpu.sh)
    to set `LD_LIBRARY_PATH` directly to the CUDA 12.8u1 installation at
    `/app/kagayaki/CUDA/12.8u1` (provides `libcudart.so.12` + `libcudnn.so.9.10.0`
    + all other CUDA 12 math libs).
  - Added a hard GPU count check (`tf.config.list_physical_devices('GPU')`) that
    fails with `exit 1` if no GPU is detected, preventing silent CPU fallback.
- **Resubmission (2026-04-29):** All 23 queued Lstm jobs (PBS 20200–20236) that
  had the old broken script were `qdel`'d and resubmitted. Full missing-job audit
  produced 478 tuples; submitted 477 (1 SvmA already running). Cluster per-user
  limit ~113 means the remaining ~364 Lstm jobs will be batched as slots free up.
  Current queue: **75 Lstm + 38 SvmA** (PBS ~20152–20629).

---

## Old split2 Version Reference Data (as of 2026-02-17)

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
