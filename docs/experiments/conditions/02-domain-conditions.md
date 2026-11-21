# Experiment 2 (Domain Shift) Conditions

This file lists the experiment conditions of the paper ("Experiment 2: Domain Shift, split2").

---

## Overview

- **Objective**: Evaluate the effect of rebalancing and domain shift on an RF detector
- **Model**: RF
- **Data split**: `split2` (`in_domain`: 44 subject-sessions, `out_domain`: 43 subject-sessions)
- **Launchers**:
  - Cross/Within: `scripts/hpc/launchers/launch_paper_domain_split2.sh`
  - Mixed: `scripts/hpc/launchers/launch_exp2_mixed.sh`
  - Completion to the 12 seeds: `scripts/hpc/launchers/gen_exp2_seed12_joblist.py` (job list)
    and `scripts/hpc/launchers/launch_exp2_seed12_fill.sh` (submission)
- **Job script**: `scripts/hpc/jobs/domain_analysis/pbs_domain_comparison_split2.sh`

## Experiment Parameters

| Parameter | Values |
|-----------|--------|
| Distance metrics | mmd, dtw, wasserstein (3) |
| Domain groups | in_domain (44), out_domain (43) (2) |
| Training modes | source_only, target_only, mixed (3) |
| Rebalancing | 7 levels (below) |
| Seeds | 0, 1, 3, 7, 13, 42, 123, 256, 512, 999, 1337, 2024 (12) |
| Ranking method | knn (k = 5) |
| Optuna trials | 100 |
| HPO cross-validation | StratifiedKFold, 3 folds (`CV_FOLDS_OPTUNA` in `src/config.py`) |
| HPO objective | F2 at the fold-optimal threshold (`src/models/training/optuna_tuning.py`) |

## Rebalancing Conditions (7 levels)

| # | CONDITION | Paper level | ratio | jobs per (D, G, M, seed) |
|---|-----------|-------------|-------|------------|
| 1 | `baseline` | Baseline (no rebalancing) | N/A | 1 |
| 2 | `undersample` | RUS | 0.1, 0.5 | 2 |
| 3 | `smote_plain` | SMOTE (on the pooled training data) | 0.1, 0.5 | 2 |
| 4 | `smote` | SW-SMOTE (SMOTE applied per subject) | 0.1, 0.5 | 2 |

The CLI flags of each condition are listed in the
[reproducibility guide](../reproducibility.md#rebalancing-implementation).

## Job Count

```
3 dist x 2 group x 3 mode x 7 rebalancing levels = 126 factor cells
126 cells x 12 seeds                               = 1,512 jobs
```

The runs were first submitted with fewer seeds; `gen_exp2_seed12_joblist.py` and
`launch_exp2_seed12_fill.sh` completed every cell to the 12 seeds above.

## Training Mode Definitions

| Mode | Description | Training data | Evaluation data |
|------|-------------|---------------|-----------------|
| `source_only` | Cross | Opposite group | Test group |
| `target_only` | Within | Same group | Test group |
| `mixed` | Mixed | All 87 subject-sessions (pooled) | Test group |

### split2 Data Split Details

| Mode | Group | Training data | Evaluation data |
|------|--------|---------------|-----------------|
| source_only | out_domain | in_domain (44) | out_domain (43) |
| source_only | in_domain | out_domain (43) | in_domain (44) |
| target_only | out_domain | out_domain (43) | out_domain (43) |
| target_only | in_domain | in_domain (44) | in_domain (44) |
| mixed | out_domain | All 87 | out_domain (43) |
| mixed | in_domain | All 87 | in_domain (44) |

## HPC Resource Settings

As requested by the launchers (`launch_paper_domain_split2.sh`, `launch_exp2_mixed.sh`):

| Condition | CPUs | Memory | Walltime | Queue |
|-----------|------|--------|----------|-------|
| baseline / undersample | 4 | 8 GB | 06:00:00 | SINGLE |
| smote / smote_plain | 4 | 10 GB | 08:00:00 | SINGLE |

`launch_exp2_seed12_fill.sh` requests longer walltimes (10 h; 20 h for the SMOTE
variants).

## Related Documents

- [Reproducibility guide](../reproducibility.md)
- [Domain Generalization Pipeline](../../architecture/domain_generalization.md)
- [Imbalance methods](../../reference/imbalance_methods.md)
