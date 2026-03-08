# Experiment 3 (Prior Research Replication) Conditions

This file lists the experiment conditions used in "Experiment 3: Prior research model replication (domain_train unified version)."

> **Revision history:** Migrated from the old split2 version (source_only/target_only modes, 504 jobs) to the domain_train unified version (252 jobs).
> The old version trained the same model twice for each domain using source_only and target_only,
> whereas domain_train trains each domain only once with 70/15/15 split and evaluates twice (within-domain / cross-domain).

---

## Overview

- **Objective**: Replicate prior research models (SvmA, SvmW, Lstm) with split2 domain splitting to evaluate domain shift robustness
- **Models**: SvmW, SvmA, Lstm (3 types)
- **Training mode**: `domain_train` (split each domain's data 70/15/15, train once, evaluate twice)
- **Data split**: `split2` (`in_domain`: 44 subjects, `out_domain`: 43 subjects)
- **Launcher**: `scripts/hpc/launchers/launch_prior_research_unified.sh`
- **Job scripts**:
  - CPU (SvmW / SvmA): `scripts/hpc/jobs/train/pbs_prior_research_unified.sh`
  - GPU (Lstm): `scripts/hpc/jobs/train/pbs_prior_research_unified_gpu.sh`
- **Auto-submission daemon**: `scripts/hpc/launchers/auto_resub_unified_v2.sh` (unified for all models, automatic GPU queue routing)

---

## Experiment Matrix

| Parameter | Values |
|---|---|
| Models | SvmW, SvmA, Lstm (3) |
| Distance metrics | mmd, dtw, wasserstein (3) |
| Domain groups | in_domain (44 subjects), out_domain (43 subjects) (2) |
| Training mode | domain_train (1) |
| Seeds | 42, 123 (2) |
| Imbalance methods | baseline, smote_plain, smote, undersample (4) |
| Target ratios | 0.1, 0.5 (for ratio-based methods only) (2) |
| Distance ranking | knn |
| Optuna trials | 100 (SvmW only) |

### Training Mode

| Mode | Description | Training data | Evaluation (within) | Evaluation (cross) |
|---|---|---|---|---|
| domain_train | Unified mode | Target domain train (70%) | Same domain test (15%) | Opposite domain test (15%) |

> **Difference from old split2 version:**
> - Old: `source_only` (train on opposite domain -> evaluate on target) + `target_only` (train on same domain -> evaluate on same) as 2 modes -> same model trained twice
> - New: single `domain_train` mode, each domain's data split 70/15/15 -> train once, evaluate twice (within + cross)
> - Effect: Job count halved (504 -> 252), enables within/cross comparison from the same trained model

### Imbalance Handling Methods

| Method | Ratio parameter | Jobs per metric combo |
|---|---|---|
| baseline | N/A | 1 |
| smote_plain | 0.1, 0.5 | 2 |
| smote | 0.1, 0.5 | 2 |
| undersample | 0.1, 0.5 | 2 |

1 metric combo = 1 distance x 1 domain x 1 mode x 1 seed

### Job Count Calculation

All models use the same 4 conditions (baseline, smote_plain, smote, undersample).
domain_train mode uses 1 mode (unified from the old version's 2 modes: source_only/target_only).

```
baseline:    3 dist x 2 dom x 1 mode x 2 seed x 1           = 12
smote_plain: 3 dist x 2 dom x 1 mode x 2 seed x 2 ratios    = 24
smote:       3 dist x 2 dom x 1 mode x 2 seed x 2 ratios    = 24
undersample: 3 dist x 2 dom x 1 mode x 2 seed x 2 ratios    = 24
------------------------------------------------------------
Total: 84 jobs/model x 3 models = 252 jobs
```

> **Note:** The old split2 version had 168 jobs/model x 3 = 504 jobs, halved by the domain_train migration.

---

## HPC Resource Settings

### Unified Launcher Default Settings

| Model | CPUs | Memory | Walltime | GPU | Notes |
|---|---|---|---|---|---|
| SvmW | 8 | 16 GB | 12:00:00 | None | Optuna optimization |
| SvmA | 8 | 32 GB | 24:00:00 | None | PSO optimization |
| Lstm | 4 | 16 GB | 16:00:00 | 1 (A40/A100) | GPU acceleration |

> **Lstm GPU support:** Lstm runs with the GPU PBS script (`pbs_prior_research_unified_gpu.sh`).
> Loads CUDA libraries with `module load hpc_sdk/22.2` and configures TensorFlow GPU memory growth with `configure_gpu()`.

### SMOTE Condition Walltime Increases

Walltime exceeded frequently for SMOTE conditions (smote, smote_plain), so the following increases were applied.

| Model | Condition | Old walltime | New walltime | CPUs | Memory |
|---|---|---|---|---|---|
| SvmW | baseline, undersample | 12:00:00 | 12:00:00 | 8 | 16 GB |
| SvmW | smote, smote_plain | 12:00:00 | **24:00:00** | 8 | 16 GB |
| SvmA | baseline, undersample | 24:00:00 | 24:00:00 | 8 | 32 GB |
| SvmA | smote, smote_plain | 24:00:00 | **48:00:00** | 8 | 32 GB |
| Lstm | baseline | 16:00:00 | 16:00:00 | 4 | 16 GB |
| Lstm | smote, smote_plain, undersample | 16:00:00 | **24:00:00** | 4 | 16 GB |

### Queue Allocation

The auto-submission daemon (`auto_resub_unified_v2.sh`) automatically routes to CPU/GPU queues based on the model.

**CPU queues (SvmW / SvmA):**

| Queue | max_run/user | Purpose |
|---|---|---|
| SINGLE | 10 | Primary submission target |
| DEFAULT | 20 | Primary submission target |
| SMALL | 7 | Auxiliary |
| LONG | 2 | Long-running jobs |

**GPU queues (Lstm):**

| Queue | GPU | max_run/user | Purpose |
|---|---|---|---|
| GPU-1 | A40 | 4 | Primary GPU |
| GPU-1A | A100 | 2 | High-speed GPU |
| GPU-S | A40 | 2 | Auxiliary GPU |
| GPU-L | A40 | 1 | Auxiliary GPU |
| GPU-LA | A100 | 1 | Auxiliary GPU |

> The daemon checks queue availability with `qstat` and submits to available queues in round-robin fashion.
> CPU max concurrent: 39 jobs, GPU max concurrent: 10 jobs.

---

## Tag Naming Convention

```
prior_{MODEL}_{CONDITION}_{RANKING}_{DISTANCE}_{DOMAIN}_domain_train_split2_s{SEED}
```

Example: `prior_SvmW_baseline_knn_mmd_out_domain_domain_train_split2_s42`

With ratio specification:
```
prior_{MODEL}_{CONDITION}_{RATIO}_{RANKING}_{DISTANCE}_{DOMAIN}_domain_train_split2_s{SEED}
```

Example: `prior_SvmA_smote_0.1_knn_dtw_in_domain_domain_train_split2_s123`

---

## Output Artifacts

### Model Files

```
models/{MODEL}/{JOB_ID}/{JOB_ID}[1]/
├── {MODEL}_{mode}_{tag}_{jobid}_1.keras      # Final model (Lstm: .keras)
├── {MODEL}_{mode}_{tag}_{jobid}_1.pkl         # Final model (SvmW/SvmA: .pkl)
├── {MODEL}_fold{N}_{jobid}_1.keras            # Fold model (Lstm only, N=1-5)
├── scaler_{MODEL}_{mode}_{tag}_{jobid}_1.pkl  # Scaler
├── scaler_{MODEL}_fold{N}_{jobid}_1.pkl       # Fold scaler (Lstm)
├── selected_features_*.pkl                    # Selected features
├── feature_meta_*.json                        # Feature metadata
├── threshold_*.json                           # Classification threshold (SvmW)
└── training_history_*.json                    # Training history (Lstm)
```

### Optuna Study (SvmW only)

```
models/SvmW/{JOB_ID}/
├── optuna_SvmW_{mode}_{tag}_study.pkl         # Optuna study
├── optuna_SvmW_{mode}_{tag}_trials.csv        # Trial history
└── optuna_SvmW_{mode}_{tag}_convergence.json  # Convergence data
```

### Evaluation Results

In domain_train mode, two evaluations (within / cross) are run for each job.
Filenames include a `_within` / `_cross` suffix.

```
results/outputs/evaluation/{MODEL}/{JOB_ID}/{JOB_ID}[1]/
├── eval_results_{MODEL}_{mode}_{tag}_within.json   # Within-domain evaluation (same domain test 15%)
├── eval_results_{MODEL}_{mode}_{tag}_within.csv
├── eval_results_{MODEL}_{mode}_{tag}_cross.json    # Cross-domain evaluation (opposite domain test 15%)
└── eval_results_{MODEL}_{mode}_{tag}_cross.csv
```

> **Note:** The old split2 version used a single file without suffix.
> When migrating to domain_train, `savers.py` was modified to include `eval_type` in filenames.

---

## Related Documents

- [Reproducibility guide](../reproducibility.md) — How to reproduce experiments
- [Results](../results/03-prior-research-results.md) — Experiment 3 results
- [Prior research models](../../architecture/prior_research.md) — Model architecture details
