# Domain Shift Experiment Results (2025-01-10)

> **Note:** This document records experiment results from a specific date. For pipeline details, see [Domain Generalization Pipeline](../../architecture/domain_generalization.md).

## Experiment Status

| Category | Count |
|----------|-------|
| **Running (R)** | 39 jobs |
| **Queued (Q)** | 77 jobs |
| **Total Submitted** | ~116 jobs |
| **Completed Results** | 20 files (domain experiments) |

## Job Distribution

| Job Type | Running | Queued | Description |
|----------|---------|--------|-------------|
| `dom_smot_*` | 34 | 43 | SMOTE + BalancedRF |
| `dom_unde_*` | 2 | 15 | Undersample RUS |
| `dom_bala_*` | 1 | 12 | BalancedRF (no resampling) |
| `dom_base_*` | 2 | 4 | Baseline |

## Preliminary Results

### Best Performing Configurations (by test_F1)

| Experiment | val_F1 | test_F1 | Notes |
|------------|--------|---------|-------|
| `source_only_balanced_rf_knn_dtw_mid_domain_s123` | 0.077 | 0.083 | Mid-domain source |
| `target_only_balanced_rf_knn_dtw_mid_domain_s42` | 0.078 | 0.081 | Mid-domain target |
| `target_only_balanced_rf_knn_dtw_in_domain_s123` | 0.064 | 0.073 | In-domain target |
| `source_only_balanced_rf_knn_dtw_in_domain_s123` | 0.086 | 0.072 | In-domain source |
| `source_only_balanced_rf_knn_mmd_in_domain_s42` | 0.071 | 0.071 | MMD-based in-domain |
| `source_only_balanced_rf_knn_dtw_out_domain_s123` | 0.068 | 0.055 | Out-domain source |

### Observations

1. **SMOTE + Ranking experiments failed**: All `rank_knn_mmd` and `rank_lof_mmd` with SMOTE produced test_F1 = 0.0
2. **Baseline (balanced_rf) relatively stable**: F1 scores around 0.05-0.08
3. **Validation-Test Gap**: Some experiments show extreme gaps (e.g., val_F1=0.52 → test_F1=0.0)
4. **Distance metric comparison**: DTW-based ranking appears more stable than MMD

## Experiment Design Matrix

| Dimension | Values |
|-----------|--------|
| **Data Split** | source_only, target_only |
| **Distance Metric** | knn_dtw, knn_mmd, knn_wasserstein, lof_mmd |
| **Domain Group** | in_domain, mid_domain, out_domain |
| **Imbalance Method** | baseline, balanced_rf, undersample_rus, smote |
| **Seeds** | 42, 123 |

**Total Combinations**: 2 × 4 × 3 × 4 × 2 = 192 experiments
