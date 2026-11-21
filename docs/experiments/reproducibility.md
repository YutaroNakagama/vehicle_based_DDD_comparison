# Experiments and Reproducibility

This guide explains how the experiments of the paper were run and how to reproduce
them. The scripts that regenerate the paper's tables and figures from the tracked
CSVs are listed in the top-level `README.md`.

## Experiment Overview

| # | Experiment | Model | Description |
|---|-----------|-------|-------------|
| 1 | **Imbalance Analysis** | RF | Compare oversampling/undersampling strategies (pooled training) |
| 2 | **Domain Shift** (the paper) | RF | Full factorial design over distance metric, subject group, training mode and rebalancing, with split2 grouping |

## Experiment 1: Imbalance Analysis

Compare imbalance handling methods in isolation (pooled training without domain generalization).

**Detailed experiment conditions:** [conditions/01-imbalance-conditions.md](conditions/01-imbalance-conditions.md)

```bash
bash scripts/hpc/launchers/launch_imbalance.sh
```

## Experiment 2: Domain Shift (RF, Split2) — the paper

**Detailed experiment conditions:** [conditions/02-domain-conditions.md](conditions/02-domain-conditions.md)

| Factor | Levels |
|-----------|--------|
| Distance metric `D` | mmd, dtw, wasserstein (3) |
| Subject group `G` | in_domain (44), out_domain (43) |
| Training mode `M` | source_only (Cross), target_only (Within), mixed (Mixed) |
| Rebalancing `R` | baseline, undersample 0.1/0.5 (RUS), smote_plain 0.1/0.5 (SMOTE), smote 0.1/0.5 (SW-SMOTE) |
| Seed `S` | 0, 1, 3, 7, 13, 42, 123, 256, 512, 999, 1337, 2024 |

3 × 2 × 3 × 7 = 126 factor cells × 12 seeds = **1,512 runs**. Each run is one PBS job
of `scripts/hpc/jobs/domain_analysis/pbs_domain_comparison_split2.sh`, which performs
the HPO (100 Optuna trials), the calibration, the final fit and the evaluation.

**Launchers:**
- Cross/Within: `scripts/hpc/launchers/launch_paper_domain_split2.sh`
- Mixed: `scripts/hpc/launchers/launch_exp2_mixed.sh`
- Completion to the 12 seeds: `scripts/hpc/launchers/gen_exp2_seed12_joblist.py` (job list)
  and `scripts/hpc/launchers/launch_exp2_seed12_fill.sh` (submission)

```bash
# single cell
qsub -v CONDITION=baseline,MODE=source_only,DISTANCE=mmd,DOMAIN=out_domain,SEED=42 \
    scripts/hpc/jobs/domain_analysis/pbs_domain_comparison_split2.sh
```

The job scripts are specific to the cluster used for the paper (JAIST HAKUSAN, PBS):
replace the account name, project path, notification address and queue names before
use.

**Training modes:**

| Mode | Description | Training Data | Evaluation Data |
|------|-------------|---------------|------------------|
| source_only | Cross | Opposite group | Test group |
| target_only | Within | Same group | Test group |
| mixed | Mixed | All 87 subject-sessions | Test group |

### Condition Naming Convention

| CONDITION | Tag prefix | Processing |
|-----------|-------------|----------|
| `baseline` | `baseline_domain_*` | No rebalancing (class weights only) |
| `smote_plain` | `smote_plain_*` | SMOTE on the pooled training data |
| `smote` | `imbalv3_*` | Subject-wise SMOTE (SMOTE applied per subject) |
| `undersample` | `undersample_rus_*` | Random under-sampling |

### Rebalancing implementation

Condition → CLI flags (`scripts/hpc/jobs/domain_analysis/pbs_domain_comparison_split2.sh`):

| Paper level | Launcher condition | CLI flags |
|---|---|---|
| SMOTE | `smote_plain` | `--use_oversampling --oversample_method smote --target_ratio {0.1,0.5}` |
| SW-SMOTE | `smote` | `--use_oversampling --oversample_method smote --target_ratio {0.1,0.5} --subject_wise_oversampling` |
| RUS | `undersample` | `--use_oversampling --oversample_method undersample_rus --target_ratio {0.1,0.5}` |

- `k_neighbors = min(5, minority_count - 1)` for both SMOTE variants
  (`src/models/sampling/oversampling.py`).
- SMOTE (pooled) is applied to the RF-selected top-10 features, unscaled, before
  `StandardScaler.transform` (`src/models/training/pipeline.py`).
- SW-SMOTE runs before the RF feature selection, on the full cleaned numeric feature
  matrix, unscaled, looping over the subjects with `random_state + i`; a subject with
  fewer than two minority samples is left unchanged (`src/models/model_pipeline.py`).

## Monitoring jobs

```bash
qstat -u $USER
tail -50 scripts/hpc/logs/train/*.OU
grep -l "Traceback\|ERROR" scripts/hpc/logs/train/*.OU | tail -5
```

## Results Structure

```
results/
├── analysis/exp2_domain_shift/
│   ├── distance/           # distance matrices and In/Out subject lists
│   └── figures/            # metric tables (csv/), figures (png/, svg/)
└── outputs/
    ├── training/RF/        # per-run training records (JSON/CSV)
    └── evaluation/RF/      # per-run evaluation records (JSON/CSV)
```

Each run additionally saves the trained model (`*.pkl`), the scaler, the selected
features and the Optuna study under `models/RF/`; these are not distributed.

## Reproducibility Checklist

- **Fixed random seeds** — passed to the rebalancing, the TPE sampler and the RF (`--seed`).
- **Dependencies** — `requirements.txt` (ranges) and `requirements-lock.txt` (versions used for the paper).
- **Preprocessing** — feature extraction parameters in `src/config.py`.
- **Thread control** — HPC jobs set `OMP_NUM_THREADS=1` etc.

## Related Documents

- [Developer Guide](../architecture/developer_guide.md) — Repository architecture
- [Domain Generalization Pipeline](../architecture/domain_generalization.md) — Domain analysis details
- [Imbalance Methods](../reference/imbalance_methods.md) — Method descriptions
