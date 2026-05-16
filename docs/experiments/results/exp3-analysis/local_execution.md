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

## Progress snapshot (2026-05-16 16:35, +3 h 35 m)

| Metric | Value |
|---|---|
| Overall | **34 / 108 (31.5 %)** |
| Lstm done | 25 (incl. 8 pre-restart) — 11 more before Lstm pool drains |
| SvmW done | 9 (incl. 3 pre-restart) |
| SvmA done | 0 (still in PSO; first DONE expected ~ +5–7 h from launch) |
| Failures | 0 |
| All-DONE rc | `rc1=0 rc2=0` for every entry |
| Within JSONs | 34 (matches DONE count) |
| Cross JSONs | 34 (matches DONE count) |
| Active python procs | 19 (1 launcher + 18 workers) |
| CPU | ~100 % sustained |

ETA (refined): all 108 jobs complete around **翌 02:00–04:00** (≈ 12–14 h
remaining), gated by SvmA's PSO tail. The Lstm pool is expected to drain
around **18:00–18:30**; when that happens the **aux launcher** (see below)
auto-starts and adds 4 more CPU workers for the remaining SvmW + SvmA work.

## Optimisation status

CPU is the binding constraint: all 6 P-cores (12 logical) + 8 E-cores are
pinned at ~100 % by the 18-worker pool. Further raw parallelism would
oversubscribe and slow the per-job completions, so the only loss-free
speed-up available is **filling the slots Lstm vacates after it finishes
its 36 jobs**. That is implemented as an *auxiliary launcher* scheduled by
a PowerShell `BackgroundJob` (5-min poll). When the Lstm within-JSON count
reaches 36, the watcher spawns:

```powershell
$env:LOCAL_PARALLEL_LSTM = "0"
$env:LOCAL_PARALLEL_SVMW = "2"
$env:LOCAL_PARALLEL_SVMA = "2"
python scripts/python/train/local_exp3_launcher.py --reverse --models SvmW SvmA
```

with logs at `logs/exp3_local/aux_watcher.log` and
`logs/exp3_local/aux_launcher.{log,err.log}`. The `--reverse` flag (added in
commit `5532d8d`) makes the auxiliary process the pending list from the
opposite end, so two launchers can co-run without tag collisions until they
meet in the middle. Expected speed-up: ETA shortens by ~2–3 hours.

The launcher itself supports per-model parallelism overrides via
`LOCAL_PARALLEL_{SVMW,SVMA,LSTM}` env vars — no code edits required to
reshape the pool. The primary launcher's PARALLELISM dict is the default
when those env vars are unset.

## Relationship to HPC runs

These local results land in the same `results/outputs/evaluation/<model>/`
tree as the HPC runs and use identical CLI flags. They are tagged with the
same `prior_<model>_imbalv3_knn_<dist>_<dom>_domain_train_split2_subjectwise_ratio<R>_s<seed>`
tag pattern, so the existing collection script
[`scripts/python/domain_analysis/collect_evaluation_metrics.py`](../../../../scripts/python/domain_analysis/collect_evaluation_metrics.py)
picks them up automatically with no extra glob changes.
