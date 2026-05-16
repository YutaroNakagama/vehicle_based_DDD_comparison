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
