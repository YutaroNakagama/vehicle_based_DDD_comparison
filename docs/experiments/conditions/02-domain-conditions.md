# Experiment 2 (Domain Shift) Conditions

This file lists the experiment conditions used in "Experiment 2: Domain Shift (split2)."

---

## Overview

- **Objective**: Evaluate domain shift robustness of RF models using split2 domain splitting
- **Model**: RF (BalancedRF included as an imbalance handling method)
- **Data split**: `split2` (`in_domain`: 44 subjects, `out_domain`: 43 subjects)
- **Launchers**:
  - Cross/Within-domain: `scripts/hpc/launchers/launch_paper_domain_split2.sh`
  - Multi-domain: `scripts/hpc/launchers/launch_exp2_mixed.sh`
  - Additional seed submission: `scripts/hpc/launchers/exp2_10seeds_submit.sh`
  - Failed job resubmission: `scripts/hpc/launchers/resubmit_failed_exp2.sh`
- **Job script**: `scripts/hpc/jobs/domain_analysis/pbs_domain_comparison_split2.sh`

## Experiment Parameters

| Parameter | Values |
|-----------|--------|
| Distance metrics | mmd, dtw, wasserstein (3) |
| Domain groups | in_domain (44 subjects), out_domain (43 subjects) (2) |
| Training modes | source_only, target_only, mixed (3) |
| Seeds | 0, 1, 7, 13, 42, 123, 256, 512, 1337, 2024 (10) |
| Ranking method | knn |
| Optuna trials | 100 |
| CV strategy | StratifiedKFold (3-fold) |
| Optimization metric | F2 score |

> **Note:** Initial experiments used 2 seeds (42, 123). Eight additional seeds were added later,
> completing the stability evaluation with 10 seeds (balanced_rf: 2 seeds only).

## Imbalance Handling Conditions (5 types -> 8 jobs/combo)

| # | CONDITION | Description | ratio | jobs/combo |
|---|-----------|-------------|-------|------------|
| 1 | `baseline` | No imbalance handling (class_weight only) | N/A | 1 |
| 2 | `smote_plain` | Global SMOTE (applied after pooling all subjects) | 0.1, 0.5 | 2 |
| 3 | `smote` | Subject-wise SMOTE (SMOTE applied per subject) | 0.1, 0.5 | 2 |
| 4 | `undersample` | Random Under-Sampling (RUS) | 0.1, 0.5 | 2 |
| 5 | `balanced_rf` | BalancedRandomForestClassifier (internal balancing) | N/A | 1 |

> **Difference between smote_plain and smote:**
> - `smote_plain`: SMOTE applied after pooling all subjects' data (standard SMOTE)
> - `smote`: SMOTE applied to each subject's data individually before pooling (Subject-wise SMOTE)
>
> Subject-wise SMOTE has the advantage of preserving inter-subject data distribution differences.

## Job Count Calculation

```
10-seed version (baseline, smote_plain, sw_smote, undersample_rus):
  Cross/Within-domain:  3 dist x 2 dom x 2 mode x 10 seed x 7 cond  = 840 jobs
  Multi-domain (mixed): 3 dist x 2 dom x 1 mode x 10 seed x 7 cond  = 420 jobs
  Subtotal: 1,260 jobs

2-seed version (balanced_rf only):
  Cross/Within-domain:  3 dist x 2 dom x 2 mode x 2 seed x 1 cond   = 24 jobs
  Multi-domain (mixed): 3 dist x 2 dom x 1 mode x 2 seed x 1 cond   = 12 jobs
  Subtotal: 36 jobs

Total: 1,260 + 36 = 1,296 jobs
```

> **Note:** Initially executed as 288 jobs with 2 seeds (42, 123).
> Additional seeds were submitted later, completing ~1,296 jobs in total.

## Training Mode Definitions

| Mode | Description | Training data | Evaluation data |
|------|-------------|---------------|-----------------|
| `source_only` | Cross-domain | Opposite domain | Target domain |
| `target_only` | Within-domain | Same domain | Same domain |
| `mixed` | Multi-domain | All 87 subjects (pooled) | Target domain |

### split2 Data Split Details

| Mode | Domain | Training data | Evaluation data |
|------|--------|---------------|-----------------|
| source_only | out_domain | in_domain (44 subjects) | out_domain (43 subjects) |
| source_only | in_domain | out_domain (43 subjects) | in_domain (44 subjects) |
| target_only | out_domain | out_domain (43 subjects) | out_domain (43 subjects) |
| target_only | in_domain | in_domain (44 subjects) | in_domain (44 subjects) |
| mixed | out_domain | All 87 subjects | out_domain (43 subjects) |
| mixed | in_domain | All 87 subjects | in_domain (44 subjects) |

## HPC Resource Settings

| Condition | CPUs | Memory | Walltime | Queue |
|-----------|------|--------|----------|-------|
| baseline / undersample | 4 | 8 GB | 06:00:00 | SINGLE |
| smote / smote_plain | 4 | 10 GB | 08:00:00 | SINGLE |
| balanced_rf | 8 | 12 GB | 08:00:00 | LONG |

> **Known issue:** smote_plain with ratio=0.5 + out_domain + seed=123 exceeded
> the DEFAULT queue (10h) walltime, requiring resubmission to the LONG queue (15h).

## Related Documents

- [Experiment results](../results/02-domain-results.md)
- [Reproducibility guide](../reproducibility.md)
- [Domain Generalization Pipeline](../../architecture/domain_generalization.md)
- [Imbalance methods](../../reference/imbalance_methods.md)
