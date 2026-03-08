# Domain Shift Experiment Results (Experiment 2)

> **Note:** For pipeline details, see [Domain Generalization Pipeline](../../architecture/domain_generalization.md).
> For condition definitions, see [Domain Conditions](../conditions/02-domain-conditions.md).

---

## Experiment 2: Domain Generalization with Subject-Split (split2)

**Last updated:** 2026-03-07

### Overview

Experiment 2 investigates domain generalization performance of **RF models** under a vehicle-based subject split (split2).
Training and evaluation are performed for all combinations of training modes, distance metrics, domain groups,
imbalance handling methods, and random seeds (10 seeds for main conditions, 2 seeds for balanced_rf).

### Experiment Matrix

| Parameter | Values | Count |
|-----------|--------|-------|
| Distance metrics | mmd, dtw, wasserstein | 3 |
| Domain groups | in_domain (44), out_domain (43) | 2 |
| Training modes | source_only, target_only, mixed | 3 |
| Seeds | 0, 1, 7, 13, 42, 123, 256, 512, 1337, 2024 | 10 (balanced_rf: 2) |
| Conditions | baseline, smote_plain, smote (sw_smote), undersample, balanced_rf | 5 (-> 8 jobs/combo) |

**Total configurations:** ~1,296 (1,260 for 10-seed conditions + 36 for balanced_rf)

### Experiment Status (2026-03-07)

| Condition | Ratio | Model | Seeds | Expected | Completed | Status |
|-----------|-------|-------|-------|----------|-----------|--------|
| baseline | — | RF | 10 | 180 | 179 | ✅ (99.4%) |
| smote_plain | 0.1 | RF | 10 | 180 | 180 | ✅ |
| smote_plain | 0.5 | RF | 10 | 180 | 179 | ✅ (99.4%) |
| sw_smote | 0.1 | RF | 10 | 180 | 179 | ✅ (99.4%) |
| sw_smote | 0.5 | RF | 10 | 180 | 179 | ✅ (99.4%) |
| undersample (RUS) | 0.1 | RF | 10 | 180 | 179 | ✅ (99.4%) |
| undersample (RUS) | 0.5 | RF | 10 | 180 | 179 | ✅ (99.4%) |
| balanced_rf | — | BalancedRF | 2 | 36 | 36 | ✅ |
| **Total** | | | | **~1,296** | **1,290** | **99.5%** |

> At most 1 missing result per condition (specific seed x distance x domain combination).
> This does not affect the statistical conclusions of the experiment.

### Key Results Summary

#### Overall Mean Across All Conditions (mean +/- std across all distance/mode/domain)

| Condition | N | F2 | AUROC | AUPRC | Recall | Precision |
|-----------|---|-----|-------|-------|--------|-----------|
| baseline | 179 | 0.216+/-0.059 | 0.634+/-0.128 | 0.143+/-0.179 | 0.642+/-0.158 | 0.059+/-0.017 |
| smote_plain (r=0.1) | 180 | 0.352+/-0.164 | 0.768+/-0.178 | 0.426+/-0.297 | 0.657+/-0.279 | 0.126+/-0.066 |
| smote_plain (r=0.5) | 179 | 0.380+/-0.209 | 0.757+/-0.173 | 0.401+/-0.281 | 0.577+/-0.286 | 0.170+/-0.117 |
| sw_smote (r=0.1) | 179 | **0.409+/-0.241** | 0.762+/-0.182 | **0.435+/-0.336** | 0.575+/-0.318 | 0.197+/-0.135 |
| sw_smote (r=0.5) | 179 | 0.318+/-0.244 | 0.749+/-0.169 | 0.290+/-0.235 | 0.324+/-0.251 | 0.308+/-0.236 |
| undersample_rus (r=0.1) | 179 | 0.184+/-0.048 | 0.594+/-0.092 | 0.108+/-0.122 | 0.537+/-0.148 | 0.051+/-0.013 |
| undersample_rus (r=0.5) | 179 | 0.160+/-0.037 | 0.564+/-0.065 | 0.073+/-0.055 | 0.451+/-0.139 | 0.045+/-0.010 |
| balanced_rf | 36 | 0.208+/-0.165 | 0.757+/-0.129 | 0.427+/-0.254 | 0.534+/-0.391 | 0.064+/-0.059 |

#### Out-domain F2 (Primary Domain Shift Robustness Metric)

| Condition | source_only (Cross) | target_only (Within) | mixed (Multi) |
|-----------|---------------------|----------------------|---------------|
| baseline | 0.157+/-0.012 | 0.223+/-0.053 | 0.308+/-0.030 |
| smote_plain (r=0.1) | 0.138+/-0.013 | 0.455+/-0.067 | 0.496+/-0.049 |
| smote_plain (r=0.5) | 0.118+/-0.017 | 0.504+/-0.091 | **0.565+/-0.120** |
| sw_smote (r=0.1) | 0.101+/-0.013 | **0.558+/-0.107** | **0.611+/-0.165** |
| sw_smote (r=0.5) | 0.038+/-0.012 | 0.479+/-0.180 | 0.464+/-0.164 |
| undersample_rus (r=0.1) | 0.156+/-0.014 | 0.210+/-0.043 | 0.223+/-0.043 |
| undersample_rus (r=0.5) | 0.154+/-0.017 | 0.184+/-0.040 | 0.190+/-0.020 |
| balanced_rf | 0.103+/-0.113 | 0.223+/-0.188 | 0.396+/-0.113 |

#### Out-domain AUROC (Discrimination Performance Under Domain Shift)

| Condition | source_only (Cross) | target_only (Within) | mixed (Multi) |
|-----------|---------------------|----------------------|---------------|
| baseline | 0.519+/-0.013 | 0.668+/-0.117 | 0.840+/-0.063 |
| smote_plain (r=0.1) | 0.523+/-0.008 | 0.905+/-0.024 | **0.912+/-0.016** |
| smote_plain (r=0.5) | 0.522+/-0.007 | 0.889+/-0.034 | 0.898+/-0.033 |
| sw_smote (r=0.1) | 0.512+/-0.009 | **0.906+/-0.056** | 0.889+/-0.062 |
| sw_smote (r=0.5) | 0.515+/-0.012 | 0.871+/-0.059 | 0.883+/-0.042 |
| undersample_rus (r=0.1) | 0.526+/-0.020 | 0.652+/-0.091 | 0.661+/-0.087 |
| undersample_rus (r=0.5) | 0.528+/-0.023 | 0.602+/-0.066 | 0.591+/-0.049 |
| balanced_rf | 0.603+/-0.035 | 0.831+/-0.022 | 0.893+/-0.014 |

### Key Findings

1. **Training mode impact:** Across all conditions: `mixed` (Multi-domain) >= `target_only` (Within-domain) >> `source_only` (Cross-domain).
   Cross-domain alone degrades to near chance-level (AUROC ~ 0.52).

2. **Imbalance handling effectiveness:**
   - **SMOTE-based methods (smote_plain, sw_smote)** substantially improve both F2 and AUROC over baseline.
   - **sw_smote (r=0.1)** achieves the best out-domain F2 scores (within: 0.558, mixed: 0.611).
   - **undersample_rus** performs similarly to or slightly worse than baseline. RUS is not effective for this task.
   - **balanced_rf** uses only 2 seeds but achieves relatively high AUROC=0.893 in mixed mode.

3. **SMOTE ratio impact:**
   - smote_plain: ratio=0.5 slightly outperforms in F2.
   - sw_smote: ratio=0.1 is more stable overall than ratio=0.5 (especially in mixed mode).
     ratio=0.5 tends to increase variance due to excessive oversampling.

4. **Domain shift robustness:**
   - The performance gap between in-domain and out-domain varies by condition.
   - SMOTE-based + mixed mode combinations achieve out-domain AUROC ~ 0.89-0.91,
     showing relatively suppressed domain shift impact.

### Visualizations

Summary plots (seed-aggregated, mean +/- std):

```
results/analysis/exp2_domain_shift/figures/png/split2/
├── baseline/
│   ├── baseline_summary.png          # 10-seed aggregated (4 rows: DTW/MMD/Wasserstein/Pooled)
│   └── baseline_s{seed}.png          # Per-seed plots (10 files)
├── smote_plain/
│   ├── smote_r01_summary.png         # ratio=0.1, 10-seed aggregated
│   ├── smote_r05_summary.png         # ratio=0.5, 10-seed aggregated
│   └── smote_r{01,05}_s{seed}.png    # Per-seed (20 files)
├── undersample_rus/
│   ├── rus_r01_summary.png
│   ├── rus_r05_summary.png
│   └── rus_r{01,05}_s{seed}.png      # Per-seed (20 files)
├── sw_smote/
│   ├── sw_smote_r01_summary.png
│   ├── sw_smote_r05_summary.png
│   └── sw_smote_r{01,05}_s{seed}.png # Per-seed (20 files)
└── balanced_rf/
    ├── brf_summary.png               # 2-seed aggregated
    └── brf_s{seed}.png               # Per-seed (2 files)
```

### Condition Naming Convention

Mapping between the `CONDITION` parameter in code and actual tag names:

| CONDITION (launcher) | Tag in eval files | Processing |
|---------------------|---------------------|---------|
| `baseline` | `baseline_domain_*` | No imbalance handling (class_weight only) |
| `smote_plain` | `smote_plain_*` | Global SMOTE |
| `smote` | `imbalv3_*` / `swsmote_*` | Subject-wise SMOTE (per subject) |
| `undersample` | `undersample_rus_*` | Random Under-Sampling |
| `balanced_rf` | `balanced_rf_*` | BalancedRandomForestClassifier |

> **Note:** The sw_smote tag name changed during development: `swsmote_*` -> `smote_subjectwise_*` -> `imbalv3_*`.
> Multiple naming conventions coexist in result directories, but they refer to the same processing.

### Analysis Scripts

| Script | Purpose |
|--------|---------|
| `scripts/python/analysis/domain/collect_split2_rf_metrics.py` | Eval JSON -> per-condition CSV + per-seed bar plots |
| `scripts/python/analysis/domain/plot_exp2_summary.py` | Cross-condition comparison summary plots (24 plots) |
| `scripts/python/analysis/domain/plot_condition_seed_summaries.py` | Seed-aggregated summary plots (all conditions) |
| `scripts/python/analysis/domain/plot_baseline_seed_summary.py` | Baseline-specific summary (bar/boxplot/heatmap/table) |

### Output Structure

```
results/outputs/evaluation/RF/         # Eval JSONs for all conditions
│   └── {JOB_ID}/{JOB_ID}[1]/eval_results_*.json
results/analysis/exp2_domain_shift/
├── figures/csv/split2/{condition}/     # Aggregated CSVs
│   └── {condition}_split2_metrics_v2.csv
└── figures/png/split2/{condition}/     # Plots
    ├── {prefix}_summary.png           # Seed-aggregated summary
    └── {prefix}_s{seed}.png           # Per-seed plots
```

### Related Documents

- [Domain Conditions](../conditions/02-domain-conditions.md)
- [Reproducibility Guide](../reproducibility.md)
- [Domain Generalization Pipeline](../../architecture/domain_generalization.md)
- [Imbalance Methods](../../reference/imbalance_methods.md)

### Statistical Analysis Reports

- [Hypothesis-Driven Analysis](exp2-analysis/hypothesis_test_report.md) — Full 24-section statistical report (F2-score, AUROC)
- [Extended Hypothesis Testing](exp2-analysis/hypothesis_test_report_extended.md) — F1, AUPRC, Recall analysis with Precision–Recall trade-off
- [Statistical Report](exp2-analysis/statistical_report.md) — Domain shift statistical analysis
- [Statistical Report (ratio=0.5)](exp2-analysis/statistical_report_ratio05.md) — Ratio 0.5 subset analysis
- [Ratio Sensitivity Analysis](exp2-analysis/ratio_sensitivity_report.md) — Ranking consistency across ratios
- [Domain Distance Analysis](exp2-analysis/domain_distances.md) — MDS/t-SNE/UMAP distance analysis
