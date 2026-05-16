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
| Active python procs | 12 (1 primary + 1 aux launcher + **10 SvmA workers**) |
| CPU | **~20 %** (only 10 single-thread SvmA workers active; the freed Lstm + SvmW slots are now idle) |
| Memory | 26 GB free / 64 GB |
| SvmA CPU-time per worker | 46 400 s (primary, 12.9 h CPU on 18.9 h wall ≈ 68 % single-core duty) / 30 000 s (aux, 8.3 h CPU on 13.6 h wall) |

### ETA (refined 2026-05-17 07:56)

| Model | Remaining | Workers | Per-job | Subtotal |
|---|---|---|---|---|
| SvmW | 0 | — | — | ✅ done |
| SvmA | 36 (10 in flight, 26 pending) | 10 | 18–24 h projected from CPU-time profile | Wave 1 finishes ~late 5/17; wave 2 + 3 → finish **5/19 morning – afternoon** with current 10-worker config |

**Revised total ETA: 2026-05-19 06:00–18:00** (slip of ~+1.5 days vs the
previous "5/17 14:00–22:00" estimate because each SvmA job is now
projected at ~18–24 h, not 6–15 h — see optimisation section below for
why more workers can shorten this). The PSO+SMOTE inner loop produces no
intermediate log lines so the first observed DONE will tighten this
window significantly.

### Per-job time observations (from launcher log)

| Model | Distance | Domain | Range observed |
|---|---|---|---|
| Lstm | mmd / wasserstein / dtw | both | 2 075 – 3 168 s (35–53 min) |
| SvmW | mmd | in / out | 7 885 – 11 281 s (132–188 min) |
| SvmW | wasserstein | in / out | 5 491 – 11 170 s (92–186 min) |
| SvmW | dtw | in / out | 5 773 – 11 825 s (96–197 min) |
| SvmA | — | — | none completed yet (>18 h wall; first DONE imminent based on CPU-time accrual) |

## Optimisation status

**Updated 2026-05-17 07:56** — the situation has flipped from CPU-bound
to **idle-bound**. With Lstm (36) and SvmW (36) both complete, only the
10 SvmA workers remain, each pinned to a single core (BLAS=1 by design
for PSO determinism). CPU utilisation is now ~20 % and ~10 cores' worth
of headroom is going unused.

### Loss-free speed-ups still available

| Option | Effect | Recommendation |
|---|---|---|
| **Spawn 2nd aux launcher with `SVMA=6 SVMW=0 LSTM=0 --reverse`** | Adds up to 6 more SvmA workers; each picks an unstarted tag from the opposite end of the pending list. Cuts SvmA remaining-work wall time from ~3 waves × 18-24 h to ~2 waves. **Saves ~18–24 h of wall time.** | ✅ **Recommended** — CPU has the headroom; SvmA workers are independent processes; SMOTE memory footprint is ~340 MB/worker so total RAM for 16 workers ≈ 5.5 GB (well within 26 GB free). |
| Reduce SvmW Optuna trials 100 → 50 | N/A — SvmW already complete. | ❌ Moot. |
| Use RTX 3060 GPU for Lstm | N/A — Lstm already complete. | ❌ Moot. |
| Switch SvmA BLAS threads 1 → 2 | Each worker uses 2 cores instead of 1; throughput per worker rises ~1.3× but only if BLAS is on the hot path. PSO inner loop is mostly Python-level (numpy reduces dominated by N≈30k samples). Marginal gain (~10–20 %), risks PSO non-determinism if any BLAS path is randomised. | ⚠️ Skip — adding workers is cleaner. |
| Kill an in-flight SvmA mid-PSO to redistribute | Loses ~13–19 h of accumulated CPU work. | ❌ Never. |

### Hardware-level limits

- **Cores:** i9-12900HK has 14 physical cores (6P+8E = 20 threads). 10 in
  use, 4–10 headroom depending on whether we count E-core threads as full
  slots. Spawning 6 more SvmA workers brings total to 16 ≈ matches
  physical-core count, leaving 4 threads for OS/launchers.
- **Memory:** 26 GB free, ~340 MB per SvmA worker → 16 workers ≈ 5.5 GB,
  no pressure.
- **Disk I/O:** SvmA writes only on DONE; not a bottleneck.
- **GPU:** RTX 3060 unusable (no TF 2.13 cp311 GPU build; DirectML cp311
  wheel does not exist). All remaining work is CPU-only by necessity.

### Why this was not done earlier

At 18:55 on 5/16, all 18 worker slots were CPU-saturated, so spawning
more would have caused oversubscription. The window only opened once
Lstm (4 workers) and SvmW (8 workers) drained — i.e. **now**.

If the user authorises the additional aux launcher, the command is:

```powershell
$env:LOCAL_PARALLEL_SVMA = "6"; $env:LOCAL_PARALLEL_SVMW = "0"; $env:LOCAL_PARALLEL_LSTM = "0"
python scripts/python/train/local_exp3_launcher.py --reverse --models SvmA `
  *> logs/exp3_local/aux2_launcher.log
```

## Relationship to HPC runs

These local results land in the same `results/outputs/evaluation/<model>/`
tree as the HPC runs and use identical CLI flags. They are tagged with the
same `prior_<model>_imbalv3_knn_<dist>_<dom>_domain_train_split2_subjectwise_ratio<R>_s<seed>`
tag pattern, so the existing collection script
[`scripts/python/domain_analysis/collect_evaluation_metrics.py`](../../../../scripts/python/domain_analysis/collect_evaluation_metrics.py)
picks them up automatically with no extra glob changes.
