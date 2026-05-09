# Sensitivity Analysis Report — Experiment 2

## Variance-Based Sensitivity Analysis (Functional ANOVA Decomposition)

This analysis decomposes the total variance of each performance metric into
contributions from the four experimental factors and their interactions.

- **First-order index ($S_i$)**: fraction of variance explained by factor $i$ alone
- **Total-order index ($S_{Ti}$)**: fraction explained by $i$ including all interactions
- **$S_{Ti} - S_i$**: variance due to interactions involving factor $i$
- Bootstrap CIs: 95% percentile, $B = 2000$, resampling over seeds


## F2-score

### First-Order and Total-Order Indices

| Factor | $S_i$ | 95% CI | $S_{Ti}$ | 95% CI | Interaction ($S_{Ti}-S_i$) |
|--------|-------|--------|----------|--------|---------------------------|
| Rebalancing ($R$) | 0.1672 | [0.1463, 0.2027] | 0.3701 | [0.3440, 0.4404] | 0.2029 |
| Distance ($D$) | 0.0002 | [0.0001, 0.0006] | 0.0101 | [0.0092, 0.0214] | 0.0099 |
| Membership ($G$) | 0.0001 | [0.0000, 0.0015] | 0.0179 | [0.0165, 0.0376] | 0.0179 |
| Mode ($M$) | 0.3744 | [0.3598, 0.3956] | 0.5758 | [0.5578, 0.6198] | 0.2014 |

Residual (seed variation): 0.2496 (25.0%)

### Complete Variance Decomposition

| Effect | $S$ | % of total |
|--------|-----|-----------|
| Mode ($M$) | 0.3744 | 37.4% |
| Rebalancing ($R$) × Mode ($M$) | 0.1863 | 18.6% |
| Rebalancing ($R$) | 0.1672 | 16.7% |
| Rebalancing ($R$) × Membership ($G$) × Mode ($M$) | 0.0048 | 0.5% |
| Membership ($G$) × Mode ($M$) | 0.0038 | 0.4% |
| Rebalancing ($R$) × Membership ($G$) | 0.0037 | 0.4% |
| Rebalancing ($R$) × Distance ($D$) × Membership ($G$) × Mode ($M$) | 0.0031 | 0.3% |
| Rebalancing ($R$) × Distance ($D$) × Mode ($M$) | 0.0027 | 0.3% |
| Rebalancing ($R$) × Distance ($D$) | 0.0014 | 0.1% |
| Distance ($D$) × Membership ($G$) | 0.0013 | 0.1% |
| Rebalancing ($R$) × Distance ($D$) × Membership ($G$) | 0.0008 | 0.1% |
| Distance ($D$) × Mode ($M$) | 0.0003 | 0.0% |
| Distance ($D$) × Membership ($G$) × Mode ($M$) | 0.0003 | 0.0% |
| Distance ($D$) | 0.0002 | 0.0% |
| Membership ($G$) | 0.0001 | 0.0% |
| Residual (seed) | 0.2496 | 25.0% |
| **Total** | **1.0000** | **100.0%** |

## AUROC

### First-Order and Total-Order Indices

| Factor | $S_i$ | 95% CI | $S_{Ti}$ | 95% CI | Interaction ($S_{Ti}-S_i$) |
|--------|-------|--------|----------|--------|---------------------------|
| Rebalancing ($R$) | 0.1367 | [0.1243, 0.1549] | 0.2715 | [0.2659, 0.2997] | 0.1348 |
| Distance ($D$) | 0.0009 | [0.0003, 0.0019] | 0.0195 | [0.0171, 0.0281] | 0.0186 |
| Membership ($G$) | 0.0050 | [0.0021, 0.0087] | 0.0292 | [0.0262, 0.0454] | 0.0242 |
| Mode ($M$) | 0.5838 | [0.5596, 0.6102] | 0.7121 | [0.6922, 0.7491] | 0.1283 |

Residual (seed variation): 0.1319 (13.2%)

### Complete Variance Decomposition

| Effect | $S$ | % of total |
|--------|-----|-----------|
| Mode ($M$) | 0.5838 | 58.4% |
| Rebalancing ($R$) | 0.1367 | 13.7% |
| Rebalancing ($R$) × Mode ($M$) | 0.1113 | 11.1% |
| Rebalancing ($R$) × Distance ($D$) × Membership ($G$) × Mode ($M$) | 0.0062 | 0.6% |
| Rebalancing ($R$) × Membership ($G$) | 0.0060 | 0.6% |
| Membership ($G$) | 0.0050 | 0.5% |
| Rebalancing ($R$) × Membership ($G$) × Mode ($M$) | 0.0034 | 0.3% |
| Rebalancing ($R$) × Distance ($D$) × Mode ($M$) | 0.0031 | 0.3% |
| Distance ($D$) × Membership ($G$) | 0.0026 | 0.3% |
| Rebalancing ($R$) × Distance ($D$) | 0.0024 | 0.2% |
| Rebalancing ($R$) × Distance ($D$) × Membership ($G$) | 0.0024 | 0.2% |
| Membership ($G$) × Mode ($M$) | 0.0023 | 0.2% |
| Distance ($D$) × Membership ($G$) × Mode ($M$) | 0.0013 | 0.1% |
| Distance ($D$) | 0.0009 | 0.1% |
| Distance ($D$) × Mode ($M$) | 0.0006 | 0.1% |
| Residual (seed) | 0.1319 | 13.2% |
| **Total** | **1.0000** | **100.0%** |

## AUPRC

### First-Order and Total-Order Indices

| Factor | $S_i$ | 95% CI | $S_{Ti}$ | 95% CI | Interaction ($S_{Ti}-S_i$) |
|--------|-------|--------|----------|--------|---------------------------|
| Rebalancing ($R$) | 0.1314 | [0.1123, 0.1651] | 0.2917 | [0.2923, 0.3548] | 0.1603 |
| Distance ($D$) | 0.0007 | [0.0002, 0.0019] | 0.0231 | [0.0178, 0.0433] | 0.0224 |
| Membership ($G$) | 0.0000 | [0.0000, 0.0019] | 0.0376 | [0.0340, 0.0666] | 0.0376 |
| Mode ($M$) | 0.3547 | [0.3205, 0.3988] | 0.4994 | [0.4788, 0.5658] | 0.1446 |

Residual (seed variation): 0.3497 (35.0%)

### Complete Variance Decomposition

| Effect | $S$ | % of total |
|--------|-----|-----------|
| Mode ($M$) | 0.3547 | 35.5% |
| Rebalancing ($R$) | 0.1314 | 13.1% |
| Rebalancing ($R$) × Mode ($M$) | 0.1190 | 11.9% |
| Rebalancing ($R$) × Membership ($G$) × Mode ($M$) | 0.0110 | 1.1% |
| Rebalancing ($R$) × Membership ($G$) | 0.0108 | 1.1% |
| Rebalancing ($R$) × Distance ($D$) × Membership ($G$) × Mode ($M$) | 0.0087 | 0.9% |
| Rebalancing ($R$) × Distance ($D$) × Mode ($M$) | 0.0045 | 0.4% |
| Rebalancing ($R$) × Distance ($D$) × Membership ($G$) | 0.0041 | 0.4% |
| Rebalancing ($R$) × Distance ($D$) | 0.0022 | 0.2% |
| Distance ($D$) × Membership ($G$) | 0.0017 | 0.2% |
| Distance ($D$) × Membership ($G$) × Mode ($M$) | 0.0010 | 0.1% |
| Distance ($D$) | 0.0007 | 0.1% |
| Membership ($G$) × Mode ($M$) | 0.0002 | 0.0% |
| Distance ($D$) × Mode ($M$) | 0.0001 | 0.0% |
| Membership ($G$) | 0.0000 | 0.0% |
| Residual (seed) | 0.3497 | 35.0% |
| **Total** | **1.0000** | **100.0%** |

---

*Generated by `scripts/python/analysis/domain/sensitivity_analysis_exp2.py`*
