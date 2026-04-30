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

## Mixed-Domain 15-seed Expansion (2026-04-29)

Triggered by user requirement: every exp3 condition should have **mixed-mode** runs across all 15 official seeds (in addition to `domain_train` which was already 100%/94%/30% covered for SvmW/SvmA/Lstm).

- Audit baseline (pre-submission, mixed mode only):
  - SvmW: 93 active tags (mostly `s42` + `s123`, 2 legacy seeds)
  - SvmA: 0 active tags
  - Lstm: 0 active tags
- Target per model: 7 cond×ratio × 3 dist × 2 dom × 15 seeds = **630 tags**
- Total target: **1890 mixed jobs**; missing at start: **1791** (saved as
  `scripts/hpc/launchers/exp3_mixed_missing_15seeds.txt`).
- Submitter: [`scripts/hpc/launchers/submit_exp3_mixed_15seeds.sh`](../../../../scripts/hpc/launchers/submit_exp3_mixed_15seeds.sh)
  - Reads TAB-delimited `MODEL\tCOND\tDIST\tDOM\tRATIO\tSEED` (RATIO=`-` for baseline).
  - Routes Lstm → split2 GPU PBS, SvmW/SvmA → split2 CPU PBS, round-robin queues.
  - **Note:** uses partition name `DEF` (renamed from `DEFAULT` on the cluster).
- Driver: [`scripts/hpc/launchers/auto_resub_exp3_mixed_15seeds.sh`](../../../../scripts/hpc/launchers/auto_resub_exp3_mixed_15seeds.sh)
  - Runs the submitter every 10 minutes (configurable) until the missing list drains.
  - Skips iterations when active jobs ≥ 280 to keep headroom for other experiments.
- First submission attempt (PID 3149507): 0 OK / 483 errors — **all `QOSMaxSubmitJobPerUserLimit`** because user queue was already at 309 active jobs. Auto-resub daemon (PID 3165339) launched after fix to drain progressively.

## Known Issues (Mixed Expansion)

### M1. Bash `IFS=$'\t' read` collapses consecutive tabs
- Symptom: empty `RATIO` field for baseline rows caused `SEED` to shift into `RATIO`,
  yielding malformed Job_Name like `Sv_bs_mi_m_r0_s` (no seed).
- Fix: write the missing list with a `-` sentinel for the empty RATIO column;
  the submitter restores empty after parsing.

### M2. Slurm partition rename `DEFAULT` → `DEF`
- Symptom: `sbatch: error: invalid partition specified: DEFAULT` for every CPU job.
- Verified via `scontrol show partition` (DEFAULT not found, DEF is `Default=YES`).
- Fix in this submitter: `CPU_QUEUES=("SINGLE" "DEF" "SMALL" "LONG" "LARGE")`.
- **Other launchers in `scripts/hpc/launchers/` still reference `DEFAULT` and should
  be patched before reuse** (e.g. `auto_resub_domain_train*.sh`, `auto_resub_unified*.sh`,
  `launch_prior_research_mixed.sh`, `auto_resub_mixed_exp3.sh`).

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

### 7. PBS_JOBID Format Mismatch Between Train and Eval (Detected 2026-04-29, Fixed 2026-04-29)

- **Symptom:** SvmA jobs ran train successfully, then eval aborted with
  `[EVAL] No model file found for mode=domain_train, tag=... in models/SvmA/manual_20260429_183104`.
  All 11 in-flight SvmA jobs and the 75 queued Lstm jobs from Issue 6's
  fix were affected.
- **Root cause:** The PBS wrappers set `PBS_JOBID="${PBS_JOBID:-manual_$(date +%Y%m%d_%H%M%S)}"`
  when the env var was unset. `src/utils/io/savers.py` constructs
  `save_mode = f"train_{pbs_jobid}[{array_idx}]"` then extracts the jobid via
  `re.search(r"(?P<jobid>\d{5,})\[(?P<fold>\d+)\]", mode)`. The regex matches
  the LAST 5+-digit run before `[N]`, so it captured only `183104` (the hh:mm:ss
  portion) and saved to `models/SvmA/183104/183104[1]/`. Meanwhile evaluate.py
  was passed the full `--jobid manual_20260429_183104` and looked under
  `models/SvmA/manual_20260429_183104/` which was empty.
- **Fix (2026-04-29):** All three PBS wrappers (`pbs_prior_research_unified.sh`,
  `pbs_prior_research_unified_gpu.sh`, `pbs_prior_research_split2.sh`) now
  prefer `SLURM_JOB_ID` / `SLURM_JOBID` and fall back to `$(date +%s)$$` —
  always pure numeric. Save and eval paths now agree.
- **In-flight job recovery:** [`scripts/hpc/launchers/eval_retry_inflight_svma.sh`](../../../../scripts/hpc/launchers/eval_retry_inflight_svma.sh)
  auto-discovers `models/SvmA/<6digit>/` dirs that lack matching eval JSONs,
  parses the tag from the .pkl filename, and re-runs `evaluate.py` with the
  6-digit jobid so the existing models are not wasted.

### 9. Lstm Eval Threshold Tuned on Test Set (Detected 2026-04-30, Open)

- **Symptom:** 99.8% of Lstm evaluations report `recall_pos = 1.0` (584 / 585
  files), regardless of condition (baseline / imbalv3 / smote_plain /
  undersample_rus). The model essentially predicts the positive class for every
  segment.
- **Mechanism:** [`src/evaluation/models/lstm.py:196`](../../../../src/evaluation/models/lstm.py#L196)
  calls `find_optimal_threshold(y_test, y_score, beta=2.0)` — i.e. the F2
  threshold is selected on the **test labels themselves** and then applied
  back to those same test predictions. This is test-set leakage and it
  artificially favours degenerate "predict all positive" thresholds because
  F2 emphasises recall (β=2). The cleaner Stage 8 path used by SvmW/SvmA
  (`load_or_optimize_threshold` on `X_val`/`y_val`) is bypassed for Lstm
  because Keras models have neither `predict_proba` nor `decision_function`.
- **Impact:**
  - Lstm `accuracy` and `f1_pos` numbers in the saved JSONs are biased
    upward (over-optimistic) compared with SvmW/SvmA which tune on validation.
  - The "domain-shift" gap (within vs cross) is masked because both sides
    pick saturating thresholds; the residual difference traces the
    positive-class prevalence difference between in_domain and out_domain
    test holdouts (in_domain ≈ 11.1 %, out_domain ≈ 9.7 %), not real
    cross-domain degradation.
  - Even with the leakage, no Lstm condition beats predict-all-positive
    on F2 → the underlying model genuinely fails to discriminate on these
    features.
- **Remediation (proposed):** route Lstm through the same validation-set
  threshold optimisation that SvmW/SvmA use. The minimal change is to
  expose `(X_val, y_val)` into `lstm_eval` and replace the
  `find_optimal_threshold(y_test, …)` call with a validation-set version.
  *Not yet applied — requires re-running every Lstm tag once the fix lands.*

### 8. GPU CUDA Path Visible Only From Login Node (Detected 2026-04-30, Fixed 2026-04-30)

- **Symptom:** After Issue 6's fix, every Lstm GPU job still failed:
  `nvidia-smi` reported the H100, but TF said `TF GPU count: 0` and the
  hard GPU check exited with rc=1. 100+ failures over a few hours.
- **Root cause:** `/app/kagayaki/CUDA/12.8u1/` is only mounted on the login
  node (`hakusan1`); GPU compute nodes (`spcc-cld-gl0X`) have CUDA at
  `/app/CUDA/12.8u1/` and access it via `module load cuda/12.8u1` (which
  bundles cuDNN 9.10.0 at `/app/CUDA/12.8u1/lib64/libcudnn.so.9`). Setting
  `LD_LIBRARY_PATH` to the kagayaki path on a GPU node points at non-existent
  files, so libcudart can't be loaded.
- **Fix (2026-04-30):** Both GPU PBS wrappers now run `module load cuda/12.8u1`
  on the compute node (with `set +u` around it because the `module` function
  references unset internals under strict mode). The kagayaki path remains as
  a fallback.
- **Diagnostic:** [`scripts/hpc/logs/train/gpu_diag_v4.log`](../../../../scripts/hpc/logs/train/gpu_diag_v4.log)
  confirmed that on `spcc-cld-gl04`, after `module load cuda/12.8u1`,
  `tf.config.list_physical_devices('GPU')` returns the H100 device and
  `tf.constant([1,2]) * tf.constant([3,4])` runs on GPU.

---

## Mixed-Domain 15-seed Operational Status (as of 2026-04-30)

The 15-seed mixed expansion runs concurrently with the in-flight domain_train
backfill. Submitters and retry loops form a layered pipeline so each type of
work has a dedicated submitter + auto-resubmitter.

### Job-Name Prefix Convention

| Prefix | Model | Mode | Compute | Submitter |
|---|---|---|---|---|
| `Sv_` | SvmA | domain_train | CPU | submit_exp3_final_remaining.sh |
| `Ls_` | Lstm | domain_train | GPU | submit_exp3_final_remaining.sh |
| `Lc_` | Lstm | domain_train | CPU (hedge) | submit_lstm_cpu_hedge.sh |
| `Sw_` | SvmW | mixed | CPU | submit_svmw_mixed_seeds.sh |
| `Sa_` | SvmA | mixed | CPU | submit_svma_mixed_seeds.sh |
| `Ls_*_mx_` | Lstm | mixed | GPU | submit_lstm_mixed_seeds.sh |
| `Lm_` | Lstm | mixed | CPU (fallback) | submit_lstm_mixed_cpu.sh |

### Active Auto-Resubmit Loops

Each loop is a long-running bash daemon that calls its submitter every 15 min
and exits when the missing list drains. Loops dedup against the live `qstat`
queue + completed eval JSONs, so re-runs are cheap.

| Script | Purpose |
|---|---|
| `auto_resubmit_exp3.sh` | SvmA dt + Lstm dt GPU |
| `auto_resubmit_lstm_cpu_hedge.sh` | Lstm dt CPU hedge (Lc_*) |
| `auto_resubmit_svmw_mixed.sh` | SvmW mixed |
| `auto_resubmit_svma_mixed.sh` | SvmA mixed |
| `auto_resubmit_lstm_mixed.sh` | Lstm mixed GPU (Ls_*_mx_) |
| `auto_resubmit_lstm_mixed_cpu.sh` | Lstm mixed CPU fallback (Lm_*) |

### Per-User QOS Limits Observed (2026-04-29)

`QOSMaxSubmitJobPerUserLimit` triggers on these per-queue caps:

| Queue | Limit | Queue | Limit |
|---|---|---|---|
| GPU-1 | 30 | SINGLE | 40 |
| GPU-1A | 20 | SMALL | 30 |
| GPU-S | 15 | LONG | 15 |
| GPU-L | 5 | LARGE | 15 |
| GPU-LA | 5 | DEF | 40 |
| VM-GPU-L | 3 | VM-LM | 60 |
|  |  | XLARGE / X2LARGE | ~7 |
|  |  | VM-CPU | 10 |
|  |  | LONG-L | 10 |

Total CPU concurrency cap ≈ 230 jobs; total GPU cap ≈ 78 jobs.

### Throughput Observations (2026-04-30)

| Workload | Per-job runtime | Sustained rate |
|---|---|---|
| SvmW mixed (CPU) | 30–90 min | 5–10/h (decaying) |
| SvmA dt (CPU) | 30 min – 12 h (Optuna+SMOTE tail) | 1–6/h |
| SvmA mixed (CPU) | 8–15 h (subjectwise SMOTE) | 1–3/h |
| Lstm dt CPU hedge | 1.5–3 h | 2–7/h |
| Lstm dt GPU (post-fix) | ~5 min | 30–60/h once running |
| Lstm mixed CPU | 1.5–3 h | (matches dt CPU) |

### Decision: Optuna `n_trials` Held at 100

User confirmed (2026-04-30) the SvmA mixed workload should keep
`N_TRIALS=100` even though its 8–15 h/job runtime is the dominant ETA
bottleneck (~13–20 days for 612 remaining tags at 1.3/h sustained).
Reducing trials would risk reproducibility against the canonical 2-seed
runs already saved.

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
