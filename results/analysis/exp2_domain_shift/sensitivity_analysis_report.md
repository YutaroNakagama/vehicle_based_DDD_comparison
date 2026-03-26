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
| Rebalancing ($R$) | 0.2425 | [0.2334, 0.2612] | 0.4639 | [0.4597, 0.4975] | 0.2215 |
| Distance ($D$) | 0.0004 | [0.0002, 0.0009] | 0.0080 | [0.0067, 0.0168] | 0.0076 |
| Membership ($G$) | 0.0039 | [0.0017, 0.0074] | 0.0175 | [0.0145, 0.0334] | 0.0136 |
| Mode ($M$) | 0.3675 | [0.3547, 0.3833] | 0.5933 | [0.5852, 0.6201] | 0.2258 |

Residual (seed variation): 0.1567 (15.7%)

### Complete Variance Decomposition

| Effect | $S$ | % of total |
|--------|-----|-----------|
| Mode ($M$) | 0.3675 | 36.7% |
| Rebalancing ($R$) | 0.2425 | 24.2% |
| Rebalancing ($R$) × Mode ($M$) | 0.2124 | 21.2% |
| Membership ($G$) × Mode ($M$) | 0.0057 | 0.6% |
| Membership ($G$) | 0.0039 | 0.4% |
| Rebalancing ($R$) × Membership ($G$) × Mode ($M$) | 0.0031 | 0.3% |
| Rebalancing ($R$) × Distance ($D$) × Membership ($G$) × Mode ($M$) | 0.0019 | 0.2% |
| Rebalancing ($R$) × Distance ($D$) × Mode ($M$) | 0.0017 | 0.2% |
| Rebalancing ($R$) × Distance ($D$) × Membership ($G$) | 0.0012 | 0.1% |
| Distance ($D$) × Membership ($G$) | 0.0010 | 0.1% |
| Rebalancing ($R$) × Distance ($D$) | 0.0009 | 0.1% |
| Distance ($D$) × Mode ($M$) | 0.0005 | 0.0% |
| Distance ($D$) × Membership ($G$) × Mode ($M$) | 0.0004 | 0.0% |
| Distance ($D$) | 0.0004 | 0.0% |
| Rebalancing ($R$) × Membership ($G$) | 0.0002 | 0.0% |
| Residual (seed) | 0.1567 | 15.7% |
| **Total** | **1.0000** | **100.0%** |

## AUROC

### First-Order and Total-Order Indices

| Factor | $S_i$ | 95% CI | $S_{Ti}$ | 95% CI | Interaction ($S_{Ti}-S_i$) |
|--------|-------|--------|----------|--------|---------------------------|
| Rebalancing ($R$) | 0.2396 | [0.2243, 0.2569] | 0.3966 | [0.3798, 0.4272] | 0.1570 |
| Distance ($D$) | 0.0010 | [0.0005, 0.0017] | 0.0149 | [0.0111, 0.0245] | 0.0139 |
| Membership ($G$) | 0.0093 | [0.0073, 0.0118] | 0.0310 | [0.0271, 0.0432] | 0.0218 |
| Mode ($M$) | 0.5045 | [0.4911, 0.5197] | 0.6636 | [0.6578, 0.6803] | 0.1591 |

Residual (seed variation): 0.0789 (7.9%)

### Complete Variance Decomposition

| Effect | $S$ | % of total |
|--------|-----|-----------|
| Mode ($M$) | 0.5045 | 50.4% |
| Rebalancing ($R$) | 0.2396 | 24.0% |
| Rebalancing ($R$) × Mode ($M$) | 0.1382 | 13.8% |
| Membership ($G$) | 0.0093 | 0.9% |
| Membership ($G$) × Mode ($M$) | 0.0074 | 0.7% |
| Rebalancing ($R$) × Distance ($D$) × Membership ($G$) × Mode ($M$) | 0.0045 | 0.4% |
| Rebalancing ($R$) × Membership ($G$) | 0.0043 | 0.4% |
| Rebalancing ($R$) × Distance ($D$) × Mode ($M$) | 0.0037 | 0.4% |
| Rebalancing ($R$) × Membership ($G$) × Mode ($M$) | 0.0030 | 0.3% |
| Rebalancing ($R$) × Distance ($D$) | 0.0024 | 0.2% |
| Distance ($D$) × Membership ($G$) × Mode ($M$) | 0.0016 | 0.2% |
| Distance ($D$) | 0.0010 | 0.1% |
| Rebalancing ($R$) × Distance ($D$) × Membership ($G$) | 0.0009 | 0.1% |
| Distance ($D$) × Mode ($M$) | 0.0008 | 0.1% |
| Distance ($D$) × Membership ($G$) | 0.0001 | 0.0% |
| Residual (seed) | 0.0789 | 7.9% |
| **Total** | **1.0000** | **100.0%** |

## AUPRC

### First-Order and Total-Order Indices

| Factor | $S_i$ | 95% CI | $S_{Ti}$ | 95% CI | Interaction ($S_{Ti}-S_i$) |
|--------|-------|--------|----------|--------|---------------------------|
| Rebalancing ($R$) | 0.2890 | [0.2616, 0.3285] | 0.4602 | [0.4437, 0.5186] | 0.1712 |
| Distance ($D$) | 0.0004 | [0.0000, 0.0011] | 0.0103 | [0.0076, 0.0227] | 0.0099 |
| Membership ($G$) | 0.0021 | [0.0005, 0.0054] | 0.0257 | [0.0172, 0.0611] | 0.0236 |
| Mode ($M$) | 0.3090 | [0.2932, 0.3312] | 0.4832 | [0.4771, 0.5189] | 0.1742 |

Residual (seed variation): 0.2186 (21.9%)

### Complete Variance Decomposition

| Effect | $S$ | % of total |
|--------|-----|-----------|
| Mode ($M$) | 0.3090 | 30.9% |
| Rebalancing ($R$) | 0.2890 | 28.9% |
| Rebalancing ($R$) × Mode ($M$) | 0.1531 | 15.3% |
| Membership ($G$) × Mode ($M$) | 0.0086 | 0.9% |
| Rebalancing ($R$) × Membership ($G$) × Mode ($M$) | 0.0051 | 0.5% |
| Rebalancing ($R$) × Membership ($G$) | 0.0041 | 0.4% |
| Rebalancing ($R$) × Distance ($D$) × Membership ($G$) × Mode ($M$) | 0.0040 | 0.4% |
| Rebalancing ($R$) × Distance ($D$) × Mode ($M$) | 0.0025 | 0.2% |
| Membership ($G$) | 0.0021 | 0.2% |
| Rebalancing ($R$) × Distance ($D$) | 0.0014 | 0.1% |
| Rebalancing ($R$) × Distance ($D$) × Membership ($G$) | 0.0009 | 0.1% |
| Distance ($D$) × Membership ($G$) × Mode ($M$) | 0.0005 | 0.1% |
| Distance ($D$) × Mode ($M$) | 0.0004 | 0.0% |
| Distance ($D$) | 0.0004 | 0.0% |
| Distance ($D$) × Membership ($G$) | 0.0002 | 0.0% |
| Residual (seed) | 0.2186 | 21.9% |
| **Total** | **1.0000** | **100.0%** |

---

*Generated by `scripts/python/analysis/domain/sensitivity_analysis_exp2.py`*
