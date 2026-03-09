# Experiment B — Ratio Sensitivity Analysis

Compare condition rankings between ratio=0.1 and ratio=0.5.

If rankings are consistent, the findings are robust to ratio choice.

## 1. Overall Condition Rankings by Ratio

### F2-score

| Condition | Mean Rank (r=0.1) | Mean Rank (r=0.5) | Δ Rank |
|-----------|:-----------------:|:-----------------:|:------:|
| baseline | 2.67 | 2.44 | -0.22 |
| rus | 3.00 | 3.22 | +0.22 |
| smote | 2.33 | 1.78 | -0.56 |
| sw_smote | 2.00 | 2.56 | +0.56 |

Spearman rank correlation: $\rho_s = 0.400$ (p=0.6000)

### AUROC

| Condition | Mean Rank (r=0.1) | Mean Rank (r=0.5) | Δ Rank |
|-----------|:-----------------:|:-----------------:|:------:|
| baseline | 2.89 | 2.94 | +0.06 |
| rus | 3.06 | 3.28 | +0.22 |
| smote | 1.78 | 1.56 | -0.22 |
| sw_smote | 2.28 | 2.22 | -0.06 |

Spearman rank correlation: $\rho_s = 1.000$ (p=0.0000)

## 2. Best Condition per Cell

Does the best-performing condition change when ratio changes?

### F2-score

| Mode | Level | Distance | Best (r=0.1) | Best (r=0.5) | Match? |
|------|-------|----------|:------------:|:------------:|:------:|
| Cross-domain | In-domain | MMD | rus | baseline | ✗ |
| Cross-domain | In-domain | DTW | rus | baseline | ✗ |
| Cross-domain | In-domain | WASSERSTEIN | rus | baseline | ✗ |
| Cross-domain | Out-domain | MMD | baseline | baseline | ✓ |
| Cross-domain | Out-domain | DTW | rus | baseline | ✗ |
| Cross-domain | Out-domain | WASSERSTEIN | baseline | rus | ✗ |
| Within-domain | In-domain | MMD | sw_smote | smote | ✗ |
| Within-domain | In-domain | DTW | sw_smote | smote | ✗ |
| Within-domain | In-domain | WASSERSTEIN | sw_smote | sw_smote | ✓ |
| Within-domain | Out-domain | MMD | sw_smote | smote | ✗ |
| Within-domain | Out-domain | DTW | sw_smote | smote | ✗ |
| Within-domain | Out-domain | WASSERSTEIN | sw_smote | sw_smote | ✓ |
| Mixed | In-domain | MMD | sw_smote | smote | ✗ |
| Mixed | In-domain | DTW | sw_smote | smote | ✗ |
| Mixed | In-domain | WASSERSTEIN | sw_smote | smote | ✗ |
| Mixed | Out-domain | MMD | sw_smote | smote | ✗ |
| Mixed | Out-domain | DTW | sw_smote | smote | ✗ |
| Mixed | Out-domain | WASSERSTEIN | sw_smote | smote | ✗ |

**Agreement**: 3/18 cells (17%)

### AUROC

| Mode | Level | Distance | Best (r=0.1) | Best (r=0.5) | Match? |
|------|-------|----------|:------------:|:------------:|:------:|
| Cross-domain | In-domain | MMD | baseline | baseline | ✓ |
| Cross-domain | In-domain | DTW | rus | sw_smote | ✗ |
| Cross-domain | In-domain | WASSERSTEIN | rus | sw_smote | ✗ |
| Cross-domain | Out-domain | MMD | rus | rus | ✓ |
| Cross-domain | Out-domain | DTW | smote | rus | ✗ |
| Cross-domain | Out-domain | WASSERSTEIN | rus | smote | ✗ |
| Within-domain | In-domain | MMD | smote | smote | ✓ |
| Within-domain | In-domain | DTW | smote | smote | ✓ |
| Within-domain | In-domain | WASSERSTEIN | sw_smote | smote | ✗ |
| Within-domain | Out-domain | MMD | smote | smote | ✓ |
| Within-domain | Out-domain | DTW | smote | smote | ✓ |
| Within-domain | Out-domain | WASSERSTEIN | sw_smote | smote | ✗ |
| Mixed | In-domain | MMD | sw_smote | smote | ✗ |
| Mixed | In-domain | DTW | sw_smote | smote | ✗ |
| Mixed | In-domain | WASSERSTEIN | sw_smote | smote | ✗ |
| Mixed | Out-domain | MMD | smote | smote | ✓ |
| Mixed | Out-domain | DTW | smote | smote | ✓ |
| Mixed | Out-domain | WASSERSTEIN | smote | smote | ✓ |

**Agreement**: 9/18 cells (50%)

## 3. Effect Size Stability

Cliff's δ (baseline vs method) comparison between ratios.

### F2-score

| Comparison | Mode | Level | Dist | δ (r=0.1) | δ (r=0.5) | Δδ | Same direction? |
|------------|------|-------|------|----------:|----------:|---:|:---------------:|
| rus vs baseline | Cross-domain | In-domain | MMD | +0.880 | -0.580 | -1.460 | ✗ |
| smote vs baseline | Cross-domain | In-domain | MMD | -1.000 | -1.000 | +0.000 | ✓ |
| sw_smote vs baseline | Cross-domain | In-domain | MMD | -1.000 | -1.000 | +0.000 | ✓ |
| rus vs baseline | Cross-domain | In-domain | DTW | +0.080 | -0.240 | -0.320 | ✗ |
| smote vs baseline | Cross-domain | In-domain | DTW | -1.000 | -1.000 | +0.000 | ✓ |
| sw_smote vs baseline | Cross-domain | In-domain | DTW | -1.000 | -1.000 | +0.000 | ✓ |
| rus vs baseline | Cross-domain | In-domain | WASSERSTEIN | +0.960 | -0.160 | -1.120 | ✗ |
| smote vs baseline | Cross-domain | In-domain | WASSERSTEIN | -1.000 | -1.000 | +0.000 | ✓ |
| sw_smote vs baseline | Cross-domain | In-domain | WASSERSTEIN | -1.000 | -1.000 | +0.000 | ✓ |
| rus vs baseline | Cross-domain | Out-domain | MMD | -0.460 | -0.520 | -0.060 | ✓ |
| smote vs baseline | Cross-domain | Out-domain | MMD | -0.900 | -1.000 | -0.100 | ✓ |
| sw_smote vs baseline | Cross-domain | Out-domain | MMD | -1.000 | -1.000 | +0.000 | ✓ |
| rus vs baseline | Cross-domain | Out-domain | DTW | +0.720 | -0.180 | -0.900 | ✗ |
| smote vs baseline | Cross-domain | Out-domain | DTW | -0.700 | -1.000 | -0.300 | ✓ |
| sw_smote vs baseline | Cross-domain | Out-domain | DTW | -1.000 | -1.000 | +0.000 | ✓ |
| rus vs baseline | Cross-domain | Out-domain | WASSERSTEIN | -0.380 | -0.120 | +0.260 | ✓ |
| smote vs baseline | Cross-domain | Out-domain | WASSERSTEIN | -0.960 | -1.000 | -0.040 | ✓ |
| sw_smote vs baseline | Cross-domain | Out-domain | WASSERSTEIN | -1.000 | -1.000 | +0.000 | ✓ |
| rus vs baseline | Within-domain | In-domain | MMD | -0.200 | -1.000 | -0.800 | ✓ |
| smote vs baseline | Within-domain | In-domain | MMD | +1.000 | +1.000 | +0.000 | ✓ |
| sw_smote vs baseline | Within-domain | In-domain | MMD | +0.960 | +0.780 | -0.180 | ✓ |
| rus vs baseline | Within-domain | In-domain | DTW | -0.800 | -0.220 | +0.580 | ✓ |
| smote vs baseline | Within-domain | In-domain | DTW | +1.000 | +1.000 | +0.000 | ✓ |
| sw_smote vs baseline | Within-domain | In-domain | DTW | +1.000 | +1.000 | +0.000 | ✓ |
| rus vs baseline | Within-domain | In-domain | WASSERSTEIN | -1.000 | -1.000 | +0.000 | ✓ |
| smote vs baseline | Within-domain | In-domain | WASSERSTEIN | +1.000 | +1.000 | +0.000 | ✓ |
| sw_smote vs baseline | Within-domain | In-domain | WASSERSTEIN | +1.000 | +1.000 | +0.000 | ✓ |
| rus vs baseline | Within-domain | Out-domain | MMD | +0.400 | +0.260 | -0.140 | ✓ |
| smote vs baseline | Within-domain | Out-domain | MMD | +1.000 | +1.000 | +0.000 | ✓ |
| sw_smote vs baseline | Within-domain | Out-domain | MMD | +1.000 | +1.000 | +0.000 | ✓ |
| rus vs baseline | Within-domain | Out-domain | DTW | +0.400 | -0.640 | -1.040 | ✗ |
| smote vs baseline | Within-domain | Out-domain | DTW | +1.000 | +1.000 | +0.000 | ✓ |
| sw_smote vs baseline | Within-domain | Out-domain | DTW | +0.980 | +0.960 | -0.020 | ✓ |
| rus vs baseline | Within-domain | Out-domain | WASSERSTEIN | -0.720 | -0.840 | -0.120 | ✓ |
| smote vs baseline | Within-domain | Out-domain | WASSERSTEIN | +1.000 | +1.000 | +0.000 | ✓ |
| sw_smote vs baseline | Within-domain | Out-domain | WASSERSTEIN | +1.000 | +0.500 | -0.500 | ✓ |
| rus vs baseline | Mixed | In-domain | MMD | -0.760 | -1.000 | -0.240 | ✓ |
| smote vs baseline | Mixed | In-domain | MMD | +1.000 | +1.000 | +0.000 | ✓ |
| sw_smote vs baseline | Mixed | In-domain | MMD | +1.000 | +0.560 | -0.440 | ✓ |
| rus vs baseline | Mixed | In-domain | DTW | -0.380 | -1.000 | -0.620 | ✓ |
| smote vs baseline | Mixed | In-domain | DTW | +1.000 | +1.000 | +0.000 | ✓ |
| sw_smote vs baseline | Mixed | In-domain | DTW | +1.000 | +0.222 | -0.778 | ✓ |
| rus vs baseline | Mixed | In-domain | WASSERSTEIN | -0.720 | -1.000 | -0.280 | ✓ |
| smote vs baseline | Mixed | In-domain | WASSERSTEIN | +1.000 | +1.000 | +0.000 | ✓ |
| sw_smote vs baseline | Mixed | In-domain | WASSERSTEIN | +1.000 | +0.400 | -0.600 | ✓ |
| rus vs baseline | Mixed | Out-domain | MMD | -0.880 | -1.000 | -0.120 | ✓ |
| smote vs baseline | Mixed | Out-domain | MMD | +1.000 | +1.000 | +0.000 | ✓ |
| sw_smote vs baseline | Mixed | Out-domain | MMD | +0.980 | +0.780 | -0.200 | ✓ |
| rus vs baseline | Mixed | Out-domain | DTW | -0.880 | -1.000 | -0.120 | ✓ |
| smote vs baseline | Mixed | Out-domain | DTW | +1.000 | +1.000 | +0.000 | ✓ |
| sw_smote vs baseline | Mixed | Out-domain | DTW | +0.960 | +0.580 | -0.380 | ✓ |
| rus vs baseline | Mixed | Out-domain | WASSERSTEIN | -0.880 | -1.000 | -0.120 | ✓ |
| smote vs baseline | Mixed | Out-domain | WASSERSTEIN | +1.000 | +1.000 | +0.000 | ✓ |
| sw_smote vs baseline | Mixed | Out-domain | WASSERSTEIN | +0.978 | +0.600 | -0.378 | ✓ |

**Directional agreement**: 49/54 (91%)

### AUROC

| Comparison | Mode | Level | Dist | δ (r=0.1) | δ (r=0.5) | Δδ | Same direction? |
|------------|------|-------|------|----------:|----------:|---:|:---------------:|
| rus vs baseline | Cross-domain | In-domain | MMD | -1.000 | -0.800 | +0.200 | ✓ |
| smote vs baseline | Cross-domain | In-domain | MMD | -0.400 | -0.140 | +0.260 | ✓ |
| sw_smote vs baseline | Cross-domain | In-domain | MMD | -0.680 | -1.000 | -0.320 | ✓ |
| rus vs baseline | Cross-domain | In-domain | DTW | +0.540 | -0.900 | -1.440 | ✗ |
| smote vs baseline | Cross-domain | In-domain | DTW | -0.400 | -1.000 | -0.600 | ✓ |
| sw_smote vs baseline | Cross-domain | In-domain | DTW | -0.540 | +0.140 | +0.680 | ✗ |
| rus vs baseline | Cross-domain | In-domain | WASSERSTEIN | +0.180 | +0.080 | -0.100 | ✓ |
| smote vs baseline | Cross-domain | In-domain | WASSERSTEIN | -0.820 | -1.000 | -0.180 | ✓ |
| sw_smote vs baseline | Cross-domain | In-domain | WASSERSTEIN | -0.040 | +0.980 | +1.020 | ✗ |
| rus vs baseline | Cross-domain | Out-domain | MMD | -0.120 | -0.180 | -0.060 | ✓ |
| smote vs baseline | Cross-domain | Out-domain | MMD | -0.220 | -0.880 | -0.660 | ✓ |
| sw_smote vs baseline | Cross-domain | Out-domain | MMD | -0.640 | -0.540 | +0.100 | ✓ |
| rus vs baseline | Cross-domain | Out-domain | DTW | +0.420 | +0.460 | +0.040 | ✓ |
| smote vs baseline | Cross-domain | Out-domain | DTW | +0.560 | +0.240 | -0.320 | ✓ |
| sw_smote vs baseline | Cross-domain | Out-domain | DTW | -0.420 | -0.120 | +0.300 | ✓ |
| rus vs baseline | Cross-domain | Out-domain | WASSERSTEIN | +0.280 | +0.320 | +0.040 | ✓ |
| smote vs baseline | Cross-domain | Out-domain | WASSERSTEIN | +0.500 | +0.800 | +0.300 | ✓ |
| sw_smote vs baseline | Cross-domain | Out-domain | WASSERSTEIN | +0.000 | -0.280 | -0.280 | ✗ |
| rus vs baseline | Within-domain | In-domain | MMD | -0.020 | -0.040 | -0.020 | ✓ |
| smote vs baseline | Within-domain | In-domain | MMD | +0.960 | +0.920 | -0.040 | ✓ |
| sw_smote vs baseline | Within-domain | In-domain | MMD | +0.920 | +0.940 | +0.020 | ✓ |
| rus vs baseline | Within-domain | In-domain | DTW | +0.580 | +0.480 | -0.100 | ✓ |
| smote vs baseline | Within-domain | In-domain | DTW | +1.000 | +1.000 | +0.000 | ✓ |
| sw_smote vs baseline | Within-domain | In-domain | DTW | +1.000 | +1.000 | +0.000 | ✓ |
| rus vs baseline | Within-domain | In-domain | WASSERSTEIN | +0.040 | -0.760 | -0.800 | ✗ |
| smote vs baseline | Within-domain | In-domain | WASSERSTEIN | +1.000 | +1.000 | +0.000 | ✓ |
| sw_smote vs baseline | Within-domain | In-domain | WASSERSTEIN | +1.000 | +1.000 | +0.000 | ✓ |
| rus vs baseline | Within-domain | Out-domain | MMD | -0.200 | +0.200 | +0.400 | ✗ |
| smote vs baseline | Within-domain | Out-domain | MMD | +1.000 | +1.000 | +0.000 | ✓ |
| sw_smote vs baseline | Within-domain | Out-domain | MMD | +1.000 | +1.000 | +0.000 | ✓ |
| rus vs baseline | Within-domain | Out-domain | DTW | +0.540 | -0.420 | -0.960 | ✗ |
| smote vs baseline | Within-domain | Out-domain | DTW | +1.000 | +1.000 | +0.000 | ✓ |
| sw_smote vs baseline | Within-domain | Out-domain | DTW | +0.980 | +0.960 | -0.020 | ✓ |
| rus vs baseline | Within-domain | Out-domain | WASSERSTEIN | -0.760 | -0.860 | -0.100 | ✓ |
| smote vs baseline | Within-domain | Out-domain | WASSERSTEIN | +0.940 | +0.540 | -0.400 | ✓ |
| sw_smote vs baseline | Within-domain | Out-domain | WASSERSTEIN | +1.000 | +0.480 | -0.520 | ✓ |
| rus vs baseline | Mixed | In-domain | MMD | -0.560 | -0.840 | -0.280 | ✓ |
| smote vs baseline | Mixed | In-domain | MMD | +1.000 | +1.000 | +0.000 | ✓ |
| sw_smote vs baseline | Mixed | In-domain | MMD | +1.000 | +1.000 | +0.000 | ✓ |
| rus vs baseline | Mixed | In-domain | DTW | -0.400 | -0.940 | -0.540 | ✓ |
| smote vs baseline | Mixed | In-domain | DTW | +0.980 | +0.980 | +0.000 | ✓ |
| sw_smote vs baseline | Mixed | In-domain | DTW | +1.000 | +1.000 | +0.000 | ✓ |
| rus vs baseline | Mixed | In-domain | WASSERSTEIN | -0.620 | -0.900 | -0.280 | ✓ |
| smote vs baseline | Mixed | In-domain | WASSERSTEIN | +0.980 | +0.980 | +0.000 | ✓ |
| sw_smote vs baseline | Mixed | In-domain | WASSERSTEIN | +1.000 | +1.000 | +0.000 | ✓ |
| rus vs baseline | Mixed | Out-domain | MMD | -0.860 | -1.000 | -0.140 | ✓ |
| smote vs baseline | Mixed | Out-domain | MMD | +0.600 | +0.480 | -0.120 | ✓ |
| sw_smote vs baseline | Mixed | Out-domain | MMD | +0.520 | +0.320 | -0.200 | ✓ |
| rus vs baseline | Mixed | Out-domain | DTW | -0.880 | -1.000 | -0.120 | ✓ |
| smote vs baseline | Mixed | Out-domain | DTW | +0.540 | +0.500 | -0.040 | ✓ |
| sw_smote vs baseline | Mixed | Out-domain | DTW | +0.440 | +0.460 | +0.020 | ✓ |
| rus vs baseline | Mixed | Out-domain | WASSERSTEIN | -0.880 | -1.000 | -0.120 | ✓ |
| smote vs baseline | Mixed | Out-domain | WASSERSTEIN | +0.540 | +0.520 | -0.020 | ✓ |
| sw_smote vs baseline | Mixed | Out-domain | WASSERSTEIN | +0.556 | +0.440 | -0.116 | ✓ |

**Directional agreement**: 47/54 (87%)

## 4. Mean Performance by Ratio

### F2-score

| Condition | Mode | Mean (r=0.1) | Mean (r=0.5) | Δ |
|-----------|------|-------------:|-------------:|--:|
| baseline | Cross-domain | 0.1605 | 0.1605 | +0.0000 |
| baseline | Within-domain | 0.2143 | 0.2143 | +0.0000 |
| baseline | Mixed | 0.2739 | 0.2739 | +0.0000 |
| rus | Cross-domain | 0.1655 | 0.1557 | -0.0097 |
| rus | Within-domain | 0.1778 | 0.1574 | -0.0204 |
| rus | Mixed | 0.2089 | 0.1683 | -0.0406 |
| smote | Cross-domain | 0.1360 | 0.1132 | -0.0228 |
| smote | Within-domain | 0.4588 | 0.5056 | +0.0468 |
| smote | Mixed | 0.4619 | 0.5234 | +0.0615 |
| sw_smote | Cross-domain | 0.1025 | 0.0425 | -0.0600 |
| sw_smote | Within-domain | 0.5561 | 0.4900 | -0.0661 |
| sw_smote | Mixed | 0.5718 | 0.4239 | -0.1479 |

### AUROC

| Condition | Mode | Mean (r=0.1) | Mean (r=0.5) | Δ |
|-----------|------|-------------:|-------------:|--:|
| baseline | Cross-domain | 0.5213 | 0.5213 | +0.0000 |
| baseline | Within-domain | 0.6307 | 0.6307 | +0.0000 |
| baseline | Mixed | 0.7500 | 0.7500 | +0.0000 |
| rus | Cross-domain | 0.5255 | 0.5214 | -0.0041 |
| rus | Within-domain | 0.6256 | 0.6034 | -0.0222 |
| rus | Mixed | 0.6298 | 0.5704 | -0.0594 |
| smote | Cross-domain | 0.5207 | 0.5186 | -0.0021 |
| smote | Within-domain | 0.9034 | 0.8847 | -0.0188 |
| smote | Mixed | 0.8799 | 0.8707 | -0.0092 |
| sw_smote | Cross-domain | 0.5148 | 0.5204 | +0.0056 |
| sw_smote | Within-domain | 0.9001 | 0.8685 | -0.0316 |
| sw_smote | Mixed | 0.8717 | 0.8588 | -0.0129 |

## 5. Conclusion

- Spearman rank correlation between r=0.1 and r=0.5: F2 $\rho_s=1.000$, AUROC $\rho_s=1.000$
- **Conclusion**: Rankings are **highly consistent** across ratios. The findings are robust to ratio choice.
