# Experiments and Reproducibility

This guide explains how to reproduce the experiments from our research.

## Experiment Overview

The repository supports three main experiment types:

| # | Experiment | Model | Description |
|---|-----------|-------|-------------|
| 1 | **Imbalance Analysis** | RF | Compare oversampling/undersampling strategies |
| 2 | **Domain Shift** | RF | Cross-domain generalization with split2 grouping |
| 3 | **Prior Research Replication** | SvmA, SvmW, Lstm | Replicate baselines with domain split2 |

## Experiment 1: Imbalance Analysis

Compare imbalance handling methods in isolation (pooled training without domain generalization).

**詳細な実験条件一覧:** [conditions/experiment_1_conditions.md](conditions/experiment_1_conditions.md)

**Launcher:** `scripts/hpc/launchers/launch_imbalance.sh`

```bash
# 8 methods × 2 seeds = 16 experiments
bash scripts/hpc/launchers/launch_imbalance.sh
```

| Method | Description |
|--------|-------------|
| Baseline | RF with class_weight only |
| SMOTE 0.1 / 0.5 | SMOTE with sampling_ratio |
| SW-SMOTE 0.1 / 0.5 | Subject-wise SMOTE |
| RUS 0.1 / 0.5 | Random Under-Sampling |
| Balanced RF | BalancedRandomForestClassifier |

**Key Parameters:**
- Model: RF
- Seeds: 42, 123
- Optuna trials: 100
- Optimization metric: F2 score

## Experiment 2: Domain Shift (RF, Split2)

Evaluate cross-domain generalization using 2-group domain splitting.

**Launcher:** `scripts/hpc/launchers/launch_paper_domain_split2.sh`

```bash
# Dry run first
bash scripts/hpc/launchers/launch_paper_domain_split2.sh --dry-run

# Submit all jobs
bash scripts/hpc/launchers/launch_paper_domain_split2.sh
```

**Experiment Matrix:**
**詳細な実験条件一覧:** [conditions/experiment_2_conditions.md](conditions/experiment_2_conditions.md)

| Parameter | Values |
|-----------|--------|
| Distance metrics | mmd, dtw, wasserstein (3) |
| Domain groups | in_domain (44), out_domain (43) |
| Training modes | source_only, target_only (2) |
| Seeds | 42, 123 (2) |
| Imbalance methods | baseline, smote_plain, smote, undersample, balanced_rf |
| Target ratios | 0.1, 0.5 (for ratio-based methods) |

**Total:** 3 × 2 × 2 × 2 × 8 conditions = 192 jobs

## Experiment 3: Prior Research Replication (Split2)

Replicate prior research baselines with domain split2 grouping.

**詳細な実験条件一覧:** [conditions/experiment_3_conditions.md](conditions/experiment_3_conditions.md)

**Launcher:** `scripts/hpc/launchers/launch_prior_research_split2.sh`

```bash
# Dry run first
bash scripts/hpc/launchers/launch_prior_research_split2.sh --dry-run

# Submit all jobs
bash scripts/hpc/launchers/launch_prior_research_split2.sh
```

**Experiment Matrix:**

| Parameter | Values |
|-----------|--------|
| Models | SvmW, SvmA, Lstm (3) |
| Distance metrics | mmd, dtw, wasserstein (3) |
| Domain groups | in_domain, out_domain (2) |
| Training modes | source_only, target_only (2) |
| Seeds | 42, 123 (2) |
| Imbalance methods | model-dependent (see below) |

**Methods per model:**
- SvmW: baseline, smote_plain, smote, undersample, balanced_rf (5 methods)
- SvmA: baseline, smote_plain, smote, undersample (4 methods)
- Lstm: baseline, smote_plain, smote, undersample (4 methods)

**Total:** 552 jobs (SvmW: 216 + SvmA: 168 + Lstm: 168)

## HPC Batch Execution

### Submit Jobs

```bash
# Single job
qsub -N job_name -l select=1:ncpus=4:mem=8gb -l walltime=08:00:00 -q SINGLE \
    -v CONDITION=baseline,MODE=source_only,DISTANCE=mmd,DOMAIN=out_domain,SEED=42 \
    scripts/hpc/jobs/domain_analysis/pbs_domain_comparison_split2.sh

# Bulk submission across multiple queues
bash scripts/hpc/launchers/massive_submit.sh
```

### Monitor Jobs

```bash
# Check queue status
qstat -u $USER

# Count running/queued jobs
qstat -u $USER | grep -c " R "
qstat -u $USER | grep -c " Q "

# View job output
tail -50 scripts/hpc/logs/train/*.OU
```

### Check Results

```bash
# Count result files by model
find results/outputs/evaluation/RF -name "*.json" | wc -l
find results/outputs/evaluation/SvmA -name "*.json" | wc -l

# Check for errors in recent logs
grep -l "Traceback\|ERROR\|KeyError" scripts/hpc/logs/train/*.OU | tail -5
```

## Results Structure

```
results/
├── analysis/
│   └── domain/
│       ├── distance/           # Distance matrices and rankings
│       └── summary/            # Aggregated analysis
├── outputs/
│   ├── evaluation/             # Test metrics (JSON/CSV)
│   │   ├── RF/
│   │   ├── BalancedRF/
│   │   ├── SvmA/
│   │   ├── SvmW/
│   │   └── Lstm/
│   └── training/               # Training metrics
└── README.md
```

## Saved Artifacts

Each experiment saves:

| File | Description |
|------|-------------|
| `{model}_{mode}.pkl` | Trained model |
| `scaler_{mode}.pkl` | Feature scaler |
| `features_{mode}.json` | Selected features |
| `metrics_*.json` | Evaluation metrics |
| `optuna_study_*.pkl` | Optuna study object (RF/SvmW) |

## Reproducibility Checklist

✅ **Fixed random seeds** — Set in config and CLI (`--seed 42`)

✅ **Versioned dependencies** — `requirements.txt` with pinned versions

✅ **Documented preprocessing** — Feature extraction parameters in `src/config.py`

✅ **Saved hyperparameters** — Optuna study objects preserved

✅ **Git versioning** — All code changes tracked

✅ **Thread control** — HPC jobs set `OMP_NUM_THREADS=1` etc. for determinism

✅ **CPU mode** — TensorFlow forced to CPU (`CUDA_VISIBLE_DEVICES=""`)

---

## Related Documents

- [Developer Guide](../architecture/developer_guide.md) — Repository architecture
- [Domain Generalization Pipeline](../architecture/domain_generalization.md) — Domain analysis details
- [Imbalance Methods](../reference/imbalance_methods.md) — Method descriptions
- [Experiment Results](results/) — Historical experiment results
