# Prior Research Experiment Results (2025-01-10)

> **Moved:** This file has been renamed to `03-prior-research-results.md`. Please see `docs/experiments/results/03-prior-research-results.md`.
>
> **Note:** This document records experiment results from a specific date. For model architecture details, see [Prior Research](../../architecture/prior_research.md).

## SvmW Results (Completed)

| Seed | val_F1 | test_F1 | test_Recall | test_Precision |
|------|--------|---------|-------------|----------------|
| 42   | 0.076  | 0.076   | 1.000       | 0.039          |
| 123  | 0.076  | 0.076   | 1.000       | 0.039          |

**Observations:**
- High test recall (1.0) indicates the model detects all drowsiness events
- Low precision indicates many false positives
- Consistent results across seeds

## SvmA and Lstm (Pending Re-run)

Previous runs failed due to model saving issue (returned `None`). Code has been fixed:
- `SvmA.py`: Now returns `(model, scaler, selected_features, results)`
- `lstm.py`: Tracks best model across folds, returns results
- `dispatch.py`: Updated to handle new return format

## Baseline Comparison

| Method | Model | val_F1 | test_F1 | Notes |
|--------|-------|--------|---------|-------|
| Prior Research | SvmW | 0.076 | 0.076 | High recall, low precision |
| Prior Research | SvmA | — | — | Pending re-run |
| Prior Research | Lstm | — | — | Pending re-run |
| Proposed (Imbalance) | BalancedRF + SW-SMOTE 0.5 | **0.931** | 0.022 | Best validation, generalization gap |
| Proposed (Domain) | BalancedRF + domain ranking | — | — | In progress |

## Implemented Methods and Job IDs

| Method | Job ID | Tag |
|--------|--------|-----|
| Baseline RF (class_weight) | 14468417 | `imbal_v2_baseline` |
| SMOTE + Tomek | 14468418 | `imbal_v2_smote_tomek` |
| SMOTE + ENN | 14468419 | `imbal_v2_smote_enn` |
| SMOTE + RUS | 14468421 | `imbal_v2_smote_rus` |
| EasyEnsemble | 14468501 | `imbal_v2_easyensemble` |
| Jittering + Scaling | 14471460 | `imbal_v2_jitter_scale` |
| Undersample RUS | 14471478 | `imbal_v2_undersample_rus` |
| Undersample Tomek | 14471479 | `imbal_v2_undersample_tomek` |
