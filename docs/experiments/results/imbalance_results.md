# Imbalance Experiment Results (2026-01-10)

> **Moved:** This file has been renamed to `02-imbalance-results.md`. Please see `docs/experiments/results/02-imbalance-results.md`.
>
> **Note:** This document records experiment results from a specific date. For method descriptions, see [Imbalance Methods](../../reference/imbalance_methods.md).

## Experiment Overview

**Objective:** Compare imbalance handling methods in isolation (pooled training without domain generalization).

**Conditions:**
- Model: Random Forest (RF)
- Seeds: 42, 123
- Optuna trials: 100
- Optimization metric: F2 score (validation set)
- 8 methods × 2 seeds = 16 experiments

| Method | Description |
|--------|-------------|
| Baseline | RF with class_weight only |
| SMOTE 0.1 | SMOTE with sampling_ratio=0.1 |
| SMOTE 0.5 | SMOTE with sampling_ratio=0.5 |
| SW-SMOTE 0.1 | Subject-wise SMOTE with ratio=0.1 |
| SW-SMOTE 0.5 | Subject-wise SMOTE with ratio=0.5 |
| RUS 0.1 | Random Under-Sampling with ratio=0.1 |
| RUS 0.5 | Random Under-Sampling with ratio=0.5 |
| Balanced RF | BalancedRandomForestClassifier |

## Results Summary

### Validation Set Performance (Optuna Best F2)

| Method | Best F2 (seed 42) | Best F2 (seed 123) | Improvement |
|--------|-------------------|--------------------| ------------|
| **SW-SMOTE 0.5** | **0.931** | **0.929** | +13.2% |
| SW-SMOTE 0.1 | 0.605 | 0.591 | +14.5% |
| SMOTE 0.5 | 0.779 | 0.804 | +8.4% |
| SMOTE 0.1 | 0.406 | 0.434 | +8.3% |
| RUS 0.5 | 0.716 | 0.715 | +0.2% |
| RUS 0.1 | 0.337 | 0.335 | +0.1% |
| Balanced RF | 0.173 | 0.171 | +0.1% |
| Baseline | 0.172 | 0.171 | +0.2% |

**Key Finding:** Subject-wise SMOTE with ratio=0.5 achieved the best performance, significantly outperforming all other methods.

### Test Set Performance (Generalization)

| Method | ROC AUC | AUPRC | Recall |
|--------|---------|-------|--------|
| All methods | ~0.50-0.54 | ~0.04 | 0-50% |

**Observation:** Severe generalization gap between validation and test sets across all methods. This suggests that the imbalance handling alone cannot solve the domain shift problem.

## Optuna Hyperparameter Optimization Analysis

### Trial Count Evaluation (100 trials)

| Aspect | Evaluation | Details |
|--------|------------|---------|
| Best Trial Position | ⚠️ Late discovery | Many best trials found late (76, 77, 88, 93, 94, 96, 99) |
| Convergence | ✅ Converged | RUS/Baseline converged early (11, 61 trials) |
| High-improvement methods | ⚠️ More trials recommended | SW-SMOTE (+13-14%), SMOTE (+6-8%) may benefit from more trials |

**Recommendation:** For SW-SMOTE and SMOTE methods, consider increasing trials to **150-200** for more precise optimization.

### Search Range Evaluation

| Parameter | Range | Evaluation |
|-----------|-------|------------|
| n_estimators | [100, 1500] | ✅ Appropriate (no boundary saturation) |
| max_depth | [5,10,20,30,50,100,None] | ✅ Appropriate (wide range explored) |
| min_samples_split | [10, 100] | ✅ Appropriate |
| min_samples_leaf | [5, 50] | ✅ Appropriate |
| max_features | [sqrt,log2,0.05-0.5] | ✅ Appropriate (0.05 addition effective) |
| max_samples | [None, 0.5, 0.7, 0.9] | ⚠️ Consider adding 0.3 |

## Visualization Files

- **Metrics comparison:** `results/analysis/imbalance/metrics/`
- **Optuna convergence:** `results/analysis/imbalance/optuna/`
- **Sampling distribution:** `results/analysis/imbalance/sampling/`
