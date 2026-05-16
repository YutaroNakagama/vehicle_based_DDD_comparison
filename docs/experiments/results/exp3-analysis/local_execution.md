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

## Progress snapshot (2026-05-16 18:55, +5 h 55 m)

| Metric | Value |
|---|---|
| Overall | **52 / 108 (48.1 %)** |
| Lstm done | **36 / 36 ✅ complete** |
| SvmW done | 16 / 36 (44 %) |
| SvmA done | 0 / 36 (PSO+SMOTE first-batch tail still in flight) |
| Failures | 0 |
| All-DONE rc | `rc1=0 rc2=0` for every entry |
| Within JSONs | 52 (matches DONE count) |
| Cross JSONs | 52 (matches DONE count) |
| Active python procs | 20 (1 primary + 1 aux launcher + 18 workers) |
| CPU | ~97 % sustained |
| Memory | 18 GB free / 64 GB |
| **Aux launcher** | **PID 14412, auto-started 18:21:49 by watcher Job 3 ✅** |

### ETA (refined 2026-05-16 18:55)

| Model | Remaining | Workers | Per-job | Subtotal |
|---|---|---|---|---|
| SvmW | 20 | 8 (primary 6 + aux 2) | 1.7–3.1 h | 6–8 h → finish ≈ 02:00–04:00 next day |
| SvmA | 36 | 10 (primary 8 + aux 2) | 6–15 h (PSO+SMOTE) | 12–22 h → finish **≈ 14:00–22:00 next day** |

**Total ETA: 2026-05-17 14:00–22:00.** This is slower than the previous
estimate because SvmW per-job time settled at 1.7–3.1 h (vs the initial
1.5 h projection) and SvmA still has not produced a first DONE by which
to calibrate the PSO+SMOTE tail.

### Per-job time observations (from launcher log)

| Model | Distance | Domain | Range observed |
|---|---|---|---|
| Lstm | mmd / wasserstein / dtw | both | 2 075 – 3 168 s (35–53 min) |
| SvmW | mmd | in / out | 7 885 – 11 281 s (132–188 min) |
| SvmW | wasserstein | in / out | 6 079 – 11 097 s (101–185 min) |
| SvmA | — | — | none completed yet (>5 h per job, first DONE pending) |

## Optimisation status

CPU is the binding constraint: all 6 P-cores (12 logical) + 8 E-cores are
pinned at ~97 % by the 18-worker pool. Further raw parallelism would
oversubscribe and slow the per-job completions. The only loss-free
speed-up — filling the slots Lstm vacates after it finishes its 36 jobs —
is already applied via the **auxiliary launcher** that auto-spawned at
18:21:49 (PID 14412). The watcher PowerShell `BackgroundJob` (5-min poll)
detected the Lstm-36/36 milestone, then ran:

```powershell
$env:LOCAL_PARALLEL_LSTM = "0"
$env:LOCAL_PARALLEL_SVMW = "2"
$env:LOCAL_PARALLEL_SVMA = "2"
python scripts/python/train/local_exp3_launcher.py --reverse --models SvmW SvmA
```

with logs at `logs/exp3_local/aux_watcher.log` and
`logs/exp3_local/aux_launcher.{log,err.log}`. The `--reverse` flag (added in
commit `5532d8d`) makes the auxiliary process the pending list from the
opposite end, so the two launchers can co-run without tag collisions until
they meet in the middle. Net effect: **+4 workers (2 SvmW + 2 SvmA)**
without disturbing the in-flight 14 primary workers.

The launcher itself supports per-model parallelism overrides via
`LOCAL_PARALLEL_{SVMW,SVMA,LSTM}` env vars — no code edits required to
reshape the pool. The primary launcher's PARALLELISM dict is the default
when those env vars are unset.

### Why no further parallelism is loss-free

| Option | Effect |
|---|---|
| Add more workers beyond 18 | CPU already 97 % saturated → context-switch overhead degrades per-job time, net throughput drops. Rejected. |
| Reduce SvmW Optuna trials 100 → 50 | Halves SvmW time, but breaks comparability with HPC results. Rejected (see [optimization_methods.md](optimization_methods.md) 2026-04-30 decision). |
| Skip SvmA SMOTE / change PSO | Same comparability concern. Rejected. |
| Use RTX 3060 GPU for Lstm | TF 2.13 cp311 is CPU-only build; DirectML wheel is not available for Python 3.11. Already moot since Lstm is complete. |
| Increase BLAS threads per worker | CPU already saturated → oversubscription. No gain. |
| Re-balance aux launcher (e.g. SvmW=3 SvmA=2) | CPU saturated → no measurable gain. Skip. |

The aux launcher is the only meaningful headroom that existed, and it is
now active.

## Relationship to HPC runs

These local results land in the same `results/outputs/evaluation/<model>/`
tree as the HPC runs and use identical CLI flags. They are tagged with the
same `prior_<model>_imbalv3_knn_<dist>_<dom>_domain_train_split2_subjectwise_ratio<R>_s<seed>`
tag pattern, so the existing collection script
[`scripts/python/domain_analysis/collect_evaluation_metrics.py`](../../../../scripts/python/domain_analysis/collect_evaluation_metrics.py)
picks them up automatically with no extra glob changes.
