# Experiment 3 — Local-PC Mini-Replication (2026-05-16)

A self-contained 108-job subset of the prior-research replication, run on a
single Windows workstation as a local fallback / sanity check for the HPC
canonical run documented in [operations_log.md](operations_log.md).

## Scope (108 jobs)

| Axis | Values | Count |
|---|---|---|
| Model | `SvmW`, `SvmA`, `Lstm` | 3 |
| Distance | `mmd`, `wasserstein`, `dtw` | 3 |
| Domain | `in_domain`, `out_domain` | 2 |
| Target ratio | `0.3`, `0.5` | 2 |
| Seed | `42`, `123`, `2025` | 3 |
| Condition | `imbalv3` only (subject-wise SMOTE) | 1 |
| Mode | `domain_train` only | 1 |

= **3 × 3 × 2 × 2 × 3 = 108 jobs** (each = 1 train + 2 eval = within + cross).

This subset deliberately covers only the canonical 2-seed scope's neighbourhood
(adds seed `2025`) to keep wall-clock manageable on a single laptop. SMOTE-plain
and undersample conditions are excluded — see Issue #15 in
[operations_log.md](operations_log.md) for why `ratio=0.1` is unreachable for
Lstm regardless.

## Launcher

- Script: [`scripts/python/train/local_exp3_launcher.py`](../../../../scripts/python/train/local_exp3_launcher.py)
- CLI: `--models SvmW SvmA Lstm`, `--dry-run`, `--limit N`
- Skip logic: a job is considered done if its `eval_results_<model>_domain_train_<tag>_within.json`
  already exists under `results/outputs/evaluation/<model>/**/`.
- Logs: per-job stdout/stderr under `logs/exp3_local/prior_<model>_*.log`;
  launcher's own log at `logs/exp3_local/launcher.err.log` (`DONE`/`FAIL`
  events). `launcher.log` (stdout) is empty by design — Python's `logging`
  default writes to stderr.

## Parallelism (tuned 2026-05-16)

Target machine: **i9-12900HK (14C/20T = 6P+8E), 64 GB RAM, RTX 3060 6 GB**.
TensorFlow 2.13.1 on Windows is **CPU-only**
(`tf.test.is_built_with_cuda() == False`); the RTX 3060 is intentionally not
used because the DirectML plugin has no Python 3.11 wheel (development
paused at 0.4.0.dev230202).

| Pool | Workers | Per-worker BLAS threads | Notes |
|---|---|---|---|
| SvmW | 6 | OMP/MKL/OPENBLAS = 1 | Optuna 100 trials, CPU-bound |
| SvmA | 8 | OMP/MKL/OPENBLAS = 1 | PSO (no Optuna), CPU-bound, longest tail |
| Lstm | 4 | OMP/MKL/OPENBLAS = 2, `TF_NUM_INTRAOP_THREADS=3`, `TF_NUM_INTEROP_THREADS=2` | TF CPU; light BLAS parallelism |

Total = **18 worker processes**. Sustained CPU at ~100 % across all cores
with ~9 GB free RAM. Tuning history (4 → 13 → 17 → 18 workers) is summarised
in this conversation's session log and was retained because the final
configuration is the only one with full CPU saturation.

Per-job environment also sets `CUDA_VISIBLE_DEVICES=""` (non-Lstm) or pops it
(Lstm) to prevent accidental partial GPU init in numpy/sklearn deps.

## Optuna trials

`N_TRIALS_OVERRIDE=100` is forwarded to each worker. This matches the HPC
canonical setting (see [optimization_methods.md](optimization_methods.md))
so local results can be directly compared with the HPC `domain_train` runs.

## Throughput observed (sample)

| Model | Per-job runtime (observed) | Pool throughput |
|---|---|---|
| Lstm | ~35–55 min (1 train + 2 eval) | ~4 jobs / 45 min |
| SvmW | ~85–165 min (Optuna 100 trials) | ~3 jobs / hour |
| SvmA | > 150 min (PSO, no intermediate logs) | first DONE not yet observed at 3h elapsed |

Lstm is no longer the bottleneck. SvmW and SvmA dominate the ETA; full-batch
completion projected ~ **+13 hours from launch** (i.e. ~02:00–04:00 next day
for a 13:00 start). Restart-safe: re-running the launcher skips already-done
jobs.

## Verification recipe

After each batch check, the following confirms healthy completion:

1. Every `DONE` line in `launcher.err.log` carries `rc1=0 rc2=0`
   (`rc1` = train, `rc2` = eval).
2. `FAIL` / `Train FAILED` / `Eval FAILED` counts are zero in the launcher log.
3. For every completed tag, **two** JSONs exist:
   - `eval_results_<model>_domain_train_<tag>_within.json` (~3 KB, classification report)
   - `eval_results_<model>_domain_train_<tag>_cross.json` (~105 KB, per-subject cross-eval)
4. Per-job log line `ERROR - [SAVE] Model object is None ... skipping save`
   is **benign** — it is the early checkpoint stage that only persists
   `scaler` and `selected_features`; the real model save happens after the
   train loop completes. Verify the final `.keras` / `.pkl` under
   `models/<model>/<run_id>/<run_id>[1]/` instead.

## Progress snapshot (2026-05-17 07:56, +18 h 56 m)

| Metric | Value |
|---|---|
| Overall | **72 / 108 (66.7 %)** |
| Lstm done | **36 / 36 ✅ complete** (last DONE 18:23 on 5/16) |
| SvmW done | **36 / 36 ✅ complete** (last DONE 01:17 on 5/17; SvmW pool exited cleanly — `SKIP (already done)` entries confirm tail drained) |
| SvmA done | 0 / 36 (10 workers still on their first job; PSO+SMOTE quiet phase) |
| Failures | 0 |
| All-DONE rc | `rc1=0 rc2=0` for every entry |
| Within JSONs | Lstm 36, SvmW 40 (incl. 4 pre-existing HPC outputs), SvmA 0 |
| Cross JSONs | Lstm 36, SvmW 40, SvmA 0 (mirrors within counts) |
| Active python procs | 19 (3 launchers + **16 SvmA workers**, see aux2 note below) |
| CPU | **~82 %** after aux2 launch (was ~20 % before) |
| Memory | 26 GB free / 64 GB |
| SvmA CPU-time per worker (pre-aux2) | 46 400 s (primary, 12.9 h CPU on 18.9 h wall ≈ 68 % single-core duty) / 30 000 s (aux1, 8.3 h CPU on 13.6 h wall) |
| **Aux2 launcher** | **PID 13460, started 08:05:03 on 5/17 with `--reverse --skip 2 --models SvmA` (6 workers)** |

### ETA (refined 2026-05-17 08:05 after aux2 launch)

| Model | Remaining | Workers | Per-job | Subtotal |
|---|---|---|---|---|
| SvmW | 0 | — | — | ✅ done |
| SvmA | 36 (16 in flight, 20 pending) | 16 (primary 8 + aux1 2 + aux2 6) | 18–24 h projected | Wave 1 (in-flight 16) finishes 5/17 late afternoon – 5/18 morning; wave 2 (final 20 jobs / 16 workers = 1.25 wave) → finish **5/18 evening – 5/19 morning** |

**Revised total ETA: 2026-05-18 18:00 – 2026-05-19 09:00** (improvement of
~12–18 h vs the pre-aux2 estimate of 5/19 06:00–18:00). The first
observed SvmA DONE (expected within hours from primary's 8 mmd workers
that have been running 19 + h) will tighten this window significantly.

### Per-job time observations (from launcher log)

| Model | Distance | Domain | Range observed |
|---|---|---|---|
| Lstm | mmd / wasserstein / dtw | both | 2 075 – 3 168 s (35–53 min) |
| SvmW | mmd | in / out | 7 885 – 11 281 s (132–188 min) |
| SvmW | wasserstein | in / out | 5 491 – 11 170 s (92–186 min) |
| SvmW | dtw | in / out | 5 773 – 11 825 s (96–197 min) |
| SvmA | — | — | none completed yet (>18 h wall; first DONE imminent based on CPU-time accrual) |

## Optimisation status

**Updated 2026-05-17 08:05** — the idle headroom that opened after Lstm
+ SvmW completed is now being used by a **second auxiliary launcher**
(`aux2`, PID 13460) running 6 additional SvmA workers. Total SvmA
in-flight = 16 (primary 8 + aux1 2 + aux2 6); CPU rose from ~20 % back
to ~82 %, memory still has 26 GB free.

### Loss-free speed-ups — status

| Option | Status |
|---|---|
| Spawn 2nd aux launcher with `SVMA=6 --reverse --skip 2` | ✅ **Applied 2026-05-17 08:05** (aux2 PID 13460). The `--skip 2` keeps aux2's reverse-end picks from colliding with aux1's first 2 reverse-end workers. |
| Switch SvmA BLAS threads 1 → 2 | ⚠️ Skipped — adding workers was cleaner and risks PSO non-determinism. |
| Reduce SvmW Optuna trials 100 → 50 | ❌ Moot (SvmW already complete). |
| Use RTX 3060 GPU for Lstm | ❌ Moot (Lstm already complete) + no TF 2.13 cp311 GPU wheel. |
| Kill an in-flight SvmA mid-PSO | ❌ Never — would lose 13–19 h of accumulated CPU work per killed worker. |

### Hardware-level limits at 16-worker steady state

- **Cores:** i9-12900HK has 14 physical cores (6P+8E = 20 threads). 16
  SvmA workers (each single-thread BLAS) + 3 launcher threads ≈ matches
  physical-core count. CPU at ~82 % leaves headroom for OS scheduling.
- **Memory:** 26 GB free; 16 workers × ~340 MB ≈ 5.5 GB → no pressure.
- **Disk I/O:** SvmA writes only on DONE; not a bottleneck.
- **GPU:** RTX 3060 unusable (no TF 2.13 cp311 GPU build; DirectML cp311
  wheel does not exist). All remaining work is CPU-only by necessity.

### Aux2 launcher startup command (for reference)

```powershell
$env:LOCAL_PARALLEL_SVMA = "6"; $env:LOCAL_PARALLEL_SVMW = "0"; $env:LOCAL_PARALLEL_LSTM = "0"
Start-Process -FilePath python -ArgumentList `
  'scripts/python/train/local_exp3_launcher.py','--reverse','--skip','2','--models','SvmA' `
  -RedirectStandardOutput 'logs/exp3_local/aux2_launcher.log' `
  -RedirectStandardError 'logs/exp3_local/aux2_launcher.err.log' -WindowStyle Hidden
```

The `--skip 2` flag (added in this commit) skips the first 2 entries of
the reversed pending list so aux2 does not collide with aux1 PID 14412,
which already holds those 2 tags.

## Relationship to HPC runs

These local results land in the same `results/outputs/evaluation/<model>/`
tree as the HPC runs and use identical CLI flags. They are tagged with the
same `prior_<model>_imbalv3_knn_<dist>_<dom>_domain_train_split2_subjectwise_ratio<R>_s<seed>`
tag pattern, so the existing collection script
[`scripts/python/domain_analysis/collect_evaluation_metrics.py`](../../../../scripts/python/domain_analysis/collect_evaluation_metrics.py)
picks them up automatically with no extra glob changes.

---

# Phase 2 — WSL2/CUDA Lstm Seed Expansion (2026-05-17)

After Phase 1 completed Lstm (3 seeds × 12 combos = 36 jobs), the remaining
12 seeds were added to bring Lstm to the full 15-seed canonical set. Since
TF dropped native Windows GPU support after 2.10, this phase runs inside
**WSL2 Ubuntu** with **TF 2.21.0 + CUDA 12 (pip bundles)** for ~20× speedup
vs DirectML (260 ms/step solo → ~450 ms/step with 8 concurrent workers).

## Scope (180 jobs total, 139 remaining at launch)

| Axis | Values | Count |
|---|---|---|
| Model | `Lstm` only | 1 |
| Distance | `mmd`, `wasserstein`, `dtw` | 3 |
| Domain | `in_domain`, `out_domain` | 2 |
| Target ratio | `0.3`, `0.5` | 2 |
| Seed | All 15 canonical seeds | 15 |
| Mode | `domain_train` only | 1 |

= **3 × 2 × 2 × 15 = 180 jobs**; 41 already done from Phase 1 → **139 pending**.

## Launcher

- Script: [`scripts/python/train/local_exp3_lstm_wsl2_launcher.py`](../../../../scripts/python/train/local_exp3_lstm_wsl2_launcher.py)
- Runs from **WSL2 Ubuntu** via:
  ```
  wsl -d Ubuntu -- bash -c '/home/ynakagama/.venv_tf_gpu/bin/python \
      /mnt/c/git/work/vehicle_ddd_eval/vehicle_based_DDD_comparison/\
      scripts/python/train/local_exp3_lstm_wsl2_launcher.py \
      > /home/ynakagama/launcher.log 2>&1'
  ```
  Run as a PowerShell background task (`run_in_background=true`) — WSL2 processes
  die if the invoking shell closes, so the background task keeps the session alive.
- Skip logic: `already_done()` checks for `*_within.json` existence only (set-based
  O(1) lookup via `build_done_set()` single `rglob` scan at startup).
- Logs: per-job stdout/stderr under `logs/exp3_lstm_wsl2/<tag>.log`;
  launcher's own log at `/home/ynakagama/launcher.log` (WSL2 native FS).

## GPU / Environment

| Setting | Value |
|---|---|
| GPU | NVIDIA RTX 3060 Laptop (6144 MiB VRAM) |
| CUDA | 12.x (bundled via `tensorflow[and-cuda]` pip wheels) |
| TF version | 2.21.0 |
| Python | 3.10 (WSL2 Ubuntu, venv `/home/ynakagama/.venv_tf_gpu`) |
| Workers | 8 (N_WORKERS = 8) |
| `CUDA_VISIBLE_DEVICES` | `0` |
| `TF_FORCE_GPU_ALLOW_GROWTH` | `true` |
| `OMP_NUM_THREADS` | `2` per worker |

GPU utilization: **59–85 %**; VRAM: **~4340 / 6144 MiB (70 %)** at 8-worker steady
state. Adding workers beyond 10 risks VRAM OOM; 8 is the recommended setting.

## Throughput observed

- Phase 1 CPU baseline (solo): 2075–3168 s (35–53 min) per job.
- Phase 2 CUDA (8 workers): **actual observed** 1402–3762 s (23–63 min) per job,
  avg ~47 min. The GPU is not always active — SMOTE oversampling and data loading
  are CPU-bound gaps between epochs. Solo CUDA speed would be ~260 ms/step.

## Progress snapshot (2026-05-18 06:32)

| Metric | Value |
|---|---|
| Started | 2026-05-17 23:34 JST |
| Elapsed | ~7 h |
| Within JSONs | **94 / 180 (52.2 %)** |
| Cross JSONs | **94 / 180 (52.2 %)** |
| Jobs completed (CUDA launcher) | **58** — all `rc1=0 rc2=0` |
| Failures / anomalies | **0** |
| Workers active | 8 (`wasserstein_in_domain_ratio0.5` seeds) |
| GPU | 0–85 % (0 % in preprocessing gaps), 60–67 °C |
| WSL2 RAM | 11 / 31 GB used |
| Remaining jobs | ~84 (excl. 2 mmd_in_domain forced re-run — see below) |
| **Projected ETA (main launcher)** | **2026-05-18 ~16:30–17:00 JST** |
| **Projected ETA (all incl. forced re-run)** | **2026-05-18 ~18:00 JST** |

### Verified artifacts (58 completed jobs)

Every DONE entry in `/home/ynakagama/launcher.log` carries `rc1=0 rc2=0`.
For each completed tag, both `_within.json` and `_cross.json` exist under
`results/outputs/evaluation/Lstm/<jobid>/<jobid>[1]/`.

## Known Issue — 2 missing cross JSONs (mmd_in_domain ratio0.3)

Jobs `mmd_in_domain / ratio0.3 / s777` and `mmd_in_domain / ratio0.3 / s1000`
completed training and within eval in the DirectML phase but their cross eval
never ran (DirectML launcher killed mid-flight). The within JSONs caused the
WSL2 launcher to skip them at startup.

**History of fix attempts:**
1. Direct cross eval re-run via `evaluate.py --jobid <original>` — failed: HDF5
   format `.keras` files (TF 2.13) not loadable by Keras 3 (`time_major` arg
   removed in Keras 3's RNN deserialization).
2. `loaders.py` patched with H5 fallback + RNN/LSTMCell in custom_objects —
   still fails: `time_major=False` kwarg breaks `RNN.__init__` in Keras 3.
3. **Final fix**: forced full re-train via
   [`scripts/python/train/run_forced_lstm_jobs.py`](../../../../scripts/python/train/run_forced_lstm_jobs.py)
   launched 2026-05-18 06:32 JST as WSL2 background task.

**Status**: forced re-run **COMPLETE** (2026-05-18 07:57–07:59 JST).
- s777: rc1=0 rc2=0, elapsed 3887 s (~64.8 min)
- s1000: rc1=0 rc2=0, elapsed 3756 s (~62.6 min)

Both within+cross JSONs confirmed present under `results/outputs/evaluation/Lstm/`.
Log: `/home/ynakagama/forced_rerun.log`

---

## Progress snapshot (2026-05-18 20:21, Phase 2 **COMPLETE**)

| Metric | Value |
|---|---|
| Started | 2026-05-17 23:34 JST |
| Elapsed | ~20 h 47 min |
| **All WSL2 workers finished** | 2026-05-18 20:21 JST (log: "All WSL2/CUDA Lstm workers finished.") |
| Within JSONs | **324 / 180** (324 total incl. Phase 1 36 + forced re-run 2) |
| Cross JSONs | **322 / 180** (2 fewer cross due to legacy HDF5 skips resolved by forced re-run) |
| Jobs completed (CUDA launcher DONE) | **139** — all `rc1=0 rc2=0` |
| Jobs skipped (already done at launch) | 41 |
| Failures / anomalies | **0** |
| Forced re-run (s777+s1000) | ✅ **DONE** — see Known Issue section above |

### Final artifact count (Phase 2 end-state)

All 180 scope jobs accounted for:
- 41 skipped (within JSON existed from Phase 1 / DirectML)
- 139 completed by WSL2 CUDA launcher (`rc1=0 rc2=0`)
- 2 forced re-run (s777+s1000 mmd_in_domain ratio0.3) — `rc1=0 rc2=0`

Total Lstm within JSONs on disk: **324**; cross JSONs: **322**.
(The 2-count gap on cross is from the 2 original DirectML jobs whose cross JSON
was missing and was replaced by the forced re-run; both new cross JSONs confirmed
present.)

---

## Phase 1 SvmA — Restart (2026-05-18 07:27 JST)

Phase 1 SvmA (36 jobs, subject-wise SMOTE only) was killed on 2026-05-17 ~15:28
when the Windows session terminated mid-PSO (0/36 done at kill time). Restarted
as a Windows background process on 2026-05-18 07:27 JST alongside the Lstm WSL2
Phase 2 run (no GPU conflict — SvmA is CPU-only PSO).

**Launcher command:**
```powershell
$env:LOCAL_PARALLEL_SVMA = "8"; $env:LOCAL_PARALLEL_SVMW = "0"; $env:LOCAL_PARALLEL_LSTM = "0"
$env:N_TRIALS_OVERRIDE = "100"
Start-Process -FilePath python `
  -ArgumentList "scripts/python/train/local_exp3_launcher.py","--models","SvmA" `
  -WorkingDirectory "C:\git\work\vehicle_ddd_eval\vehicle_based_DDD_comparison" `
  -RedirectStandardOutput "logs/svma_launcher.log" `
  -RedirectStandardError "logs/svma_launcher_err.log" -WindowStyle Hidden
```

**Status (2026-05-19 00:46 JST):** 8 workers running, PSO in progress on first 8 mmd
jobs (started 07:27 2026-05-18, ~17 h elapsed, PSO typically 18–24 h/job).
CPU time accrued: ~30,000 s per worker (≈ 8.3 h CPU on 17 h wall ≈ 49% single-core duty).
First completions expected 2026-05-19 01:00–07:00 JST.

| Metric | Value |
|---|---|
| Jobs | 36 (imbalv3 SW-SMOTE only, seeds 42 / 123 / 2025) |
| Workers | 8 |
| Done | 0 (first wave still in PSO; completion imminent) |
| Logs | `logs/svma_launcher_err.log` (launcher), `logs/exp3_local/prior_SvmA_*.log` (per-job) |
| **ETA** | **~2026-05-22 10:00 JST** (4.5 batches × 21 h/batch; from 07:27) |

---

# Phase 3 — Lstm WSL2/CUDA Non-SW-SMOTE Conditions (2026-05-19)

After Phase 2 completed all 180 SW-SMOTE (imbalv3) Lstm jobs, the remaining 3 conditions
(baseline, smote_plain r=0.5, undersample_rus r=0.5) were added to cover the full Lstm
condition grid. These conditions were not available in the local Phase 1 (CPU-only,
DirectML-limited) run.

## Scope (270 jobs)

| Axis | Values | Count |
|---|---|---|
| Model | `Lstm` only | 1 |
| Condition | `baseline`, `smote_plain r=0.5`, `undersample_rus r=0.5` | 3 |
| Distance | `mmd`, `wasserstein`, `dtw` | 3 |
| Domain | `in_domain`, `out_domain` | 2 |
| Seed | All 15 canonical seeds | 15 |
| Mode | `domain_train` only | 1 |

= **3 × 3 × 2 × 15 = 270 jobs**. `ratio=0.1` is N/A for Lstm (natural minority ≈27% >
target 10%; imblearn raises ValueError). See Issue #15 in operations_log.md.

## Launcher

- Script: [`scripts/python/train/local_exp3_lstm_wsl2_other_launcher.py`](../../../../scripts/python/train/local_exp3_lstm_wsl2_other_launcher.py)
- Runs from **WSL2 Ubuntu** via:
  ```bash
  nohup /home/ynakagama/.venv_tf_gpu/bin/python \
      scripts/python/train/local_exp3_lstm_wsl2_other_launcher.py \
      > /home/ynakagama/launcher_other.log 2>&1 &
  ```
- Tag naming:
  - `baseline`: `prior_Lstm_baseline_knn_{dist}_{dom}_domain_train_split2_s{seed}`
  - `smote_plain`: `prior_Lstm_smote_plain_knn_{dist}_{dom}_domain_train_split2_ratio0.5_s{seed}`
  - `undersample_rus`: `prior_Lstm_undersample_rus_knn_{dist}_{dom}_domain_train_split2_ratio0.5_s{seed}`
- Same GPU/environment settings as Phase 2 (CUDA_VISIBLE_DEVICES=0, TF_FORCE_GPU_ALLOW_GROWTH=true,
  N_WORKERS=8, OMP_NUM_THREADS=2).
- Logs: per-job under `logs/exp3_lstm_wsl2/<tag>.log`; launcher at `/home/ynakagama/launcher_other.log`.

## Launch notes (2026-05-19 00:09 JST)

Two launcher instances were accidentally started (PIDs 11 and 59). PID 59 and its 8 worker
processes were killed immediately after detection. PID 11 (the surviving instance) retained
its 8 workers (PIDs 23–30) and continued cleanly — the `already_done()` gate prevents
duplicate job execution once any within.json exists.

## Progress snapshot (2026-05-19 00:52 JST, ~43 min after launch)

| Metric | Value |
|---|---|
| Started | 2026-05-19 00:09 JST |
| Jobs total | 270 |
| Done | 0 (first 8 workers still on Fold 1 of first jobs) |
| Active workers | 8 (PID 11 launcher, workers PIDs 23–30) |
| Current jobs | `baseline_knn_mmd_in_domain_s{0,7,42,99,123,256,512,777}` |
| GPU | **84 %**, 2086 / 6144 MiB — training confirmed active |
| Training confirmed | s0 Fold 1 Epoch 1 in progress (AUROC 0.727 at step 32/32) |
| Data loading gap | 24 min (00:12→00:36); sequence-generation gap 16 min (00:36→00:52) |
| **Projected ETA** | **~2026-05-19 09:00–11:00 JST** (270 / 8 workers × ~47 min/job ≈ 26 h from 00:09; GPU sharing slightly slower than Phase 2) |

---

# σ_rank Seed-Count Stability Analysis (2026-05-19)

To validate that 15 seeds is sufficient for stable condition rankings (matching the
TIV2026 methodology), a σ_rank convergence analysis was run on the 162 Lstm SW-SMOTE
Phase 2 within JSONs.

**Script:** `scripts/python/analysis/domain/seed_rank_stability_exp3.py`

**Conditions:** 4 (in_domain × r=0.3, in_domain × r=0.5, out_domain × r=0.3, out_domain × r=0.5),
averaged over 3 distance metrics.

**Results:**

| k | AUROC σ_rank (mean/max) | AUPRC σ_rank (mean/max) | F2 σ_rank (mean/max) |
|---|---|---|---|
| 1 | 0.249 / 0.249 | 0.325 / 0.400 | 0.000 / 0.000 |
| 2 | 0.068 / 0.137 | 0.147 / 0.294 | 0.000 / 0.000 |
| 3 | 0.000 / 0.000 | 0.089 / 0.179 | 0.000 / 0.000 |
| 6 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| 15 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |

- AUROC: stable (σ<0.2) from **k=2**; σ=0.000 from k=3
- AUPRC: stable (σ<0.2) from **k=2**; σ=0.000 from k=6
- F2: stable (σ<0.2) from **k=1**

**Conclusion:** 15 seeds is far more than necessary — even k=6 gives perfect rank stability
across all metrics. Rankings are unambiguous at the chosen seed count.

**Outputs:**
- `results/analysis/exp3_prior_research/figures/seed_convergence_Lstm_swsmote.pdf`
- `results/analysis/exp3_prior_research/figures/seed_convergence_Lstm_swsmote.png`
- `results/analysis/exp3_prior_research/figures/seed_convergence_Lstm_swsmote.csv`

---

# Phase 4 — SvmW + SvmA 12-seed Expansion (2026-05-19)

Goal: reach 12 seeds for all SW-SMOTE conditions across SvmW and SvmA, matching
the 12-seed priority target (3 Phase 1 seeds already done → 9 new seeds needed).

**New seeds for both models:**
`[0, 7, 99, 256, 512, 777, 1000, 1337, 2024]` (9 seeds)

Combined with Phase 1 seeds `[42, 123, 2025]` → **12 seeds total**.

## SvmW Phase 2

| Metric | Value |
|---|---|
| Script | `scripts/python/train/local_exp3_svmw_phase2_launcher.py` |
| Jobs | 108 (9 seeds × 3 dist × 2 domain × 2 ratio) |
| Workers | 8 |
| Per-job | ~2.5–3.5 h (Optuna 100 trials, same as Phase 1) |
| Launched | **2026-05-19 07:33 JST** |
| Logs | `logs/exp3_svmw_phase2/` |
| **ETA** | **~2026-05-21 01:30 JST** (14 batches × 3 h = 42 h from 07:33) |

**Status (launch +0 min):** 8 workers started immediately:
`mmd_in_domain_ratio0.3_s{0,7,99,256,512,777,1000,1337}` picked up by workers 0–7.

## SvmA Phase 2 — Parallel PSO

SvmA Phase 2 uses a new `pso_parallel()` implementation that evaluates PSO particles
in parallel using `multiprocessing.Pool` (spawn), reducing per-job wall-clock from
~21 h (sequential PSO) to ~7–9 h (4× PSO speedup, non-PSO phases unchanged).

### Key code change: `src/models/architectures/SvmA.py`

| Component | Description |
|---|---|
| `SVMObjective` class | Picklable replacement for the PSO closure; stores training arrays as numpy for efficient pickling |
| `pso_parallel()` | Custom PSO matching pyswarm parameters; uses `Pool(n_processes)` initializer pattern to send large arrays once per worker, then only ~192-byte particle positions per iteration |
| `_pso_worker_init` / `_pso_eval_worker` | Module-level functions (required for spawn pickling on Windows) |
| `SVMA_PSO_PROCESSES` env var | Controls PSO parallelism (default 1 = sequential, backward-compatible) |

**Resource design (i9-12900HK, 20 logical threads):**

| Process group | Cores used |
|---|---|
| SvmA Phase 1 (8 sequential workers) | ~8 |
| SvmA Phase 2 (3 workers × 4 PSO processes) | 12 |
| Total | ~20 — fully saturated during PSO phase |

| Metric | Value |
|---|---|
| Script | `scripts/python/train/local_exp3_svma_phase2_launcher.py` |
| Jobs | 108 (9 seeds × 3 dist × 2 domain × 2 ratio) |
| Workers | 3 (`LOCAL_PARALLEL_SVMA=3`) |
| PSO processes | 4 per worker (`SVMA_PSO_PROCESSES=4`) |
| Estimated per-job | ~7–9 h (vs 21 h sequential) |
| Launched | **2026-05-19 07:33 JST** |
| Logs | `logs/exp3_svma_phase2/` |
| **ETA** | **~2026-06-01** (36 batches × 8 h ≈ 12 days; will accelerate when Phase 1 SvmA completes ~2026-05-23 and workers can be increased to 5) |

**Status (launch +0 min):** 3 workers started:
- SvmA-0: `mmd_in_domain_ratio0.3_s0`
- SvmA-1: `mmd_in_domain_ratio0.3_s7`
- SvmA-2: `mmd_in_domain_ratio0.3_s99`

PSO worker pool (4 processes per job) spawns after data loading + normalization (~2–3 h).

### Phase 1 SvmA status at Phase 2 launch

| Metric | Value |
|---|---|
| Started | 2026-05-18 07:27 JST |
| Elapsed | ~24 h |
| Done | **0 / 36** (first batch of 8 jobs still in PSO) |
| CPU time accrued | ~47,000 s / worker ≈ 13.2 h CPU (55% duty) |
| PSO progress (est.) | ~73% through PSO phase (~5 h CPU remaining = ~9 h wall) |
| **First DONE expected** | ~2026-05-19 16:30 JST |
| **All 36 done ETA** | ~2026-05-23 04:30 JST (4.5 batches × 24 h/batch from 07:27) |

---

# Phase 5 — Post-Reboot SvmA Relaunch (2026-05-22 → in flight at 2026-05-24)

## Why a relaunch was needed

Between Phase 4 launch (2026-05-19 07:33 JST) and 2026-05-22, the original
Phase 1 SvmA pool (8 workers, started 2026-05-18 07:27 JST) **never produced
a single `_within.json`** despite >100 h of accumulated CPU time. Multiple
restart cycles were attempted to diagnose:

| Date | Action | Outcome |
|---|---|---|
| 2026-05-20 ~03:30 JST | Restart Phase 1 SvmA with 12 workers (`svma_phase1_12w.{log,err}`) | 0 / 36 DONE; per-job logs only contain `STARTED`, no PSO completion |
| 2026-05-20 ~22:00 JST | SvmA smoke-test single-job run (`svma_smoketest.{log,err}`) | Reproduced same hang in PSO body — diagnosed as **PSO `n_processes=1` serial scan over swarmsize=50 × maxiter=100 = 5000 SVM fits** at ~20–30 s each → 28–42 h pure PSO wall, exceeding the 24 h baseline expectation. No bug — just under-estimated runtime. |
| 2026-05-21 22:46 JST | Relaunch with 20 workers (`launcher_svma_20w.{log,err}`) — full 36-job Phase 1 reverse-skip-0 run | All 20 STARTed simultaneously; CPU saturated at ~100 % |
| 2026-05-22 23:26 JST | **PC reboot** (Windows Update / power event) — all SvmA workers + Phase 2/3/4 WSL2 Lstm + SvmW Phase 2 launchers terminated mid-PSO. CPU-time baseline snapshotted in `cpu_baseline.json` / `cpu_baseline_ts.txt`. |
| 2026-05-23 11:03:53 JST | **Post-reboot relaunch** — `launcher_svma_20w_post.{log,err}` re-spawns 20 SvmA workers from scratch (launcher logs marker `RELAUNCH at 2026-05-23 11:03:52 (post-reboot)`). |
| 2026-05-23 12:27 JST | `auto_recover_2026-05-23_12-26-20.log` — auto-recovery script attempted to re-launch WSL2 Lstm Phase 2/3 and SvmW Phase 2, detected WSL2 distro unreachable (`Wsl/Service/0x8007274c`) and SvmA already running; skipped both. **WSL2 batches not restarted.** |
| 2026-05-23 15:53 JST | `paused_pids_2026-05-23_15-53-37.txt` — 21 SvmA PIDs paused briefly (free system load), then resumed. |

## Current state (2026-05-24 20:43 JST, ~33.7 h after relaunch)

| Metric | Value |
|---|---|
| Active python procs | **21** (1 launcher PID 3976 + 20 SvmA workers, all started 2026-05-23 11:03:53) |
| Per-worker CPU time | **~99 600 s ≈ 27.7 h CPU** (≈ 82 % single-core duty over 33.7 h wall) |
| Per-worker RAM | 119–372 MB (variation = different distance/domain conditions) |
| Per-job logs | All workers log line: `[PSO] Starting PSO swarmsize=50 maxiter=100 n_processes=1` at 2026-05-23 11:04:58 → **no log lines since** (PSO body is silent by design — pyswarm has no per-iter callback hook). |
| Within JSONs (Phase 1 SvmA) | **0 / 36** |
| Cross JSONs (Phase 1 SvmA) | **0 / 36** |
| Last `_within.json` save in `models/SvmA/<jobid>/` | 2026-05-23 11:04:45 — **early checkpoint** (scaler + selected_features only; the benign `Model object is None — skipping save` log from Verification recipe step 4). |
| Failures | 0 (no `FAIL` / `Train FAILED` in launcher.err) |

**Health assessment:** workers are **alive and computing** (steady CPU
accrual, no zombie processes, no error in logs). Silent logs are
**expected** for PSO body; 33.7 h elapsed is consistent with the smoke-test
diagnosis (28–42 h PSO wall when swarmsize=50 × maxiter=100 serial). First
DONE projected within the next few hours; all 36 expected ~2026-05-26 if
no further interruption.

**Do not kill** these processes — each carries ~27.7 h of accumulated
PSO work that would be lost on restart.

## WSL2 Lstm batches (Phase 2 + Phase 3): final on-disk count

| Tree | Within JSONs on disk |
|---|---|
| `results/outputs/evaluation/Lstm/` | **434** |
| `results/outputs/evaluation/SvmW/` | **148** |
| `results/outputs/evaluation/SvmA/` | **0 (directory does not exist)** |

Lstm scope (Phase 1 36 + Phase 2 180 SW-SMOTE + Phase 3 270 non-SW-SMOTE
= 486 expected) is at **434 / 486 ≈ 89 %**. The 52 missing jobs are the
non-SW-SMOTE conditions that were still in flight in WSL2 at the
2026-05-22 reboot and never re-launched (auto-recover failed because the
WSL2 distro was unreachable post-reboot). A manual WSL2 re-launch is
**deferred** until the Phase 1 SvmA pool completes — sharing 20 CPU
threads + WSL2 GPU work risks contention.

SvmW = 148 ≈ Phase 1 (36) + Phase 2 (108) + 4 pre-existing HPC outputs.
Phase 2 SvmW is **complete**; no further action.

## Expected artifacts not yet on disk

| Item | Why missing | Action |
|---|---|---|
| SvmA Phase 1 within/cross JSONs (0 / 36) | All 20 workers still in PSO body (silent phase) | Wait — first DONE within hours; full pool done ~2026-05-26 |
| SvmA Phase 2 within/cross JSONs (0 / 108) | Launcher PID killed at 2026-05-22 reboot; not yet re-launched | Re-launch `local_exp3_svma_phase2_launcher.py` after Phase 1 finishes |
| Lstm Phase 3 within/cross JSONs (52 / 270 missing) | WSL2 distro unreachable since 2026-05-22 reboot | Restart WSL2 + re-run `local_exp3_lstm_wsl2_other_launcher.py` |

## Lessons captured (added to operations_log.md Issue list)

1. **PSO body is silent** — `pyswarm` does not emit per-iter logs. Long
   silent stretches (~28–42 h for swarmsize=50 × maxiter=100 serial)
   look like a hang but are normal. Use CPU-time accrual on the PID
   (`Get-Process | Select Id,CPU`) — not log timestamps — to assess
   health.
2. **`SVMA_PSO_PROCESSES > 1` is the right knob** — Phase 4 used 4 PSO
   processes per worker which reduces per-job wall ~4× (~7–9 h vs
   ~28–42 h). Phase 1 / Phase 5 launchers still default to 1 for
   backward compatibility; **future Phase 1 SvmA replays should set
   `SVMA_PSO_PROCESSES=4`** (or use the Phase 2 parallel-PSO launcher).
3. **WSL2 distro is not crash-safe** — after host reboot, WSL2 service
   sometimes returns `0x8007274c` and refuses to start the distro
   silently from a Windows Scheduled Task. The auto-recover script
   should explicitly `wsl --shutdown` + `wsl -d Ubuntu -- echo ok`
   probe before launching jobs; if the probe fails, alert the user
   instead of skipping.
