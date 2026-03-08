# Distance Metric Granular Analysis

Records: 1306, Distances: ['mmd', 'dtw', 'wasserstein']

H₀ for each cell: performance under MMD = DTW = Wasserstein (Kruskal-Wallis)

---


## F2-score

### 1. Distance effect by Mode (pooling conditions & levels)

| Mode | N | H | p | η² | Sig? |
|------|--:|--:|--:|---:|:----:|
| Cross-domain | 445 | 5.04 | 0.0805 | 0.0069 | ✗ |
| Within-domain | 438 | 1.90 | 0.3877 | 0.0000 | ✗ |
| Mixed | 423 | 0.04 | 0.9810 | 0.0000 | ✗ |

**Mean ± SD per mode × distance:**

| Mode | MMD | DTW | Wasserstein |
|------|----:|----:|------------:|
| Cross-domain | 0.1293±0.0440 | 0.1210±0.0420 | 0.1259±0.0430 |
| Within-domain | 0.3648±0.1829 | 0.3501±0.1861 | 0.3743±0.1899 |
| Mixed | 0.3703±0.1775 | 0.3723±0.1789 | 0.3768±0.1830 |

### 2. Distance effect by Condition (pooling modes & levels)

| Condition | N | H | p | η² | Sig? |
|-----------|--:|--:|--:|---:|:----:|
| baseline | 186 | 5.90 | 0.0524 | 0.0213 | ✗ |
| rus_r01 | 190 | 1.09 | 0.5804 | 0.0000 | ✗ |
| rus_r05 | 191 | 2.41 | 0.2997 | 0.0022 | ✗ |
| smote_r01 | 186 | 0.62 | 0.7346 | 0.0000 | ✗ |
| smote_r05 | 185 | 1.19 | 0.5512 | 0.0000 | ✗ |
| sw_smote_r01 | 184 | 0.44 | 0.8036 | 0.0000 | ✗ |
| sw_smote_r05 | 184 | 0.67 | 0.7145 | 0.0000 | ✗ |

### 3. Distance effect by Level (pooling modes & conditions)

| Level | N | H | p | η² | Sig? |
|-------|--:|--:|--:|---:|:----:|
| In-domain | 655 | 0.57 | 0.7504 | 0.0000 | ✗ |
| Out-domain | 651 | 3.46 | 0.1776 | 0.0022 | ✗ |

### 4. Distance effect by Mode × Level (pooling conditions)

| Mode | Level | N | H | p | η² | Sig? |
|------|-------|--:|--:|--:|---:|:----:|
| Cross-domain | In-domain | 224 | 0.27 | 0.8723 | 0.0000 | ✗ |
| Cross-domain | Out-domain | 221 | 7.80 | 0.0202 | 0.0266 | ✓ |
| Within-domain | In-domain | 219 | 0.63 | 0.7299 | 0.0000 | ✗ |
| Within-domain | Out-domain | 219 | 2.53 | 0.2817 | 0.0025 | ✗ |
| Mixed | In-domain | 212 | 0.30 | 0.8605 | 0.0000 | ✗ |
| Mixed | Out-domain | 211 | 0.72 | 0.6993 | 0.0000 | ✗ |

### 5. Distance effect by Condition × Level (pooling modes)

| Condition | Level | N | H | p | η² | Sig? |
|-----------|-------|--:|--:|--:|---:|:----:|
| baseline | In-domain | 92 | 0.55 | 0.7600 | 0.0000 | ✗ |
| baseline | Out-domain | 94 | 7.16 | 0.0278 | 0.0567 | ✓ |
| rus_r01 | In-domain | 95 | 6.87 | 0.0323 | 0.0529 | ✓ |
| rus_r01 | Out-domain | 95 | 1.30 | 0.5214 | 0.0000 | ✗ |
| rus_r05 | In-domain | 96 | 14.38 | 0.0008 | 0.1331 | ✓ |
| rus_r05 | Out-domain | 95 | 15.77 | 0.0004 | 0.1497 | ✓ |
| smote_r01 | In-domain | 93 | 2.79 | 0.2478 | 0.0088 | ✗ |
| smote_r01 | Out-domain | 93 | 2.03 | 0.3629 | 0.0003 | ✗ |
| smote_r05 | In-domain | 93 | 1.63 | 0.4435 | 0.0000 | ✗ |
| smote_r05 | Out-domain | 92 | 0.22 | 0.8945 | 0.0000 | ✗ |
| sw_smote_r01 | In-domain | 93 | 0.73 | 0.6946 | 0.0000 | ✗ |
| sw_smote_r01 | Out-domain | 91 | 0.01 | 0.9949 | 0.0000 | ✗ |
| sw_smote_r05 | In-domain | 93 | 1.04 | 0.5953 | 0.0000 | ✗ |
| sw_smote_r05 | Out-domain | 91 | 0.30 | 0.8603 | 0.0000 | ✗ |

### 6. Distance effect by Condition × Mode (pooling levels)

| Condition | Mode | N | H | p | η² | Sig? |
|-----------|------|--:|--:|--:|---:|:----:|
| baseline | Cross-domain | 64 | 6.08 | 0.0478 | 0.0669 | ✓ |
| baseline | Within-domain | 62 | 29.41 | 0.0000 | 0.4646 | ✓ |
| baseline | Mixed | 60 | 2.51 | 0.2856 | 0.0089 | ✗ |
| rus_r01 | Cross-domain | 64 | 2.54 | 0.2806 | 0.0089 | ✗ |
| rus_r01 | Within-domain | 64 | 4.17 | 0.1246 | 0.0355 | ✗ |
| rus_r01 | Mixed | 62 | 1.96 | 0.3745 | 0.0000 | ✗ |
| rus_r05 | Cross-domain | 64 | 0.11 | 0.9464 | 0.0000 | ✗ |
| rus_r05 | Within-domain | 64 | 0.43 | 0.8051 | 0.0000 | ✗ |
| rus_r05 | Mixed | 63 | 1.51 | 0.4694 | 0.0000 | ✗ |
| smote_r01 | Cross-domain | 64 | 18.21 | 0.0001 | 0.2657 | ✓ |
| smote_r01 | Within-domain | 62 | 0.21 | 0.9026 | 0.0000 | ✗ |
| smote_r01 | Mixed | 60 | 0.46 | 0.7940 | 0.0000 | ✗ |
| smote_r05 | Cross-domain | 63 | 17.82 | 0.0001 | 0.2636 | ✓ |
| smote_r05 | Within-domain | 62 | 0.71 | 0.7018 | 0.0000 | ✗ |
| smote_r05 | Mixed | 60 | 0.14 | 0.9310 | 0.0000 | ✗ |
| sw_smote_r01 | Cross-domain | 63 | 5.01 | 0.0818 | 0.0501 | ✗ |
| sw_smote_r01 | Within-domain | 62 | 4.08 | 0.1300 | 0.0353 | ✗ |
| sw_smote_r01 | Mixed | 59 | 0.06 | 0.9713 | 0.0000 | ✗ |
| sw_smote_r05 | Cross-domain | 63 | 7.84 | 0.0199 | 0.0973 | ✓ |
| sw_smote_r05 | Within-domain | 62 | 1.42 | 0.4905 | 0.0000 | ✗ |
| sw_smote_r05 | Mixed | 59 | 0.04 | 0.9815 | 0.0000 | ✗ |


## AUROC

### 1. Distance effect by Mode (pooling conditions & levels)

| Mode | N | H | p | η² | Sig? |
|------|--:|--:|--:|---:|:----:|
| Cross-domain | 445 | 22.09 | 0.0000 | 0.0454 | ✓ |
| Within-domain | 438 | 4.55 | 0.1027 | 0.0059 | ✗ |
| Mixed | 423 | 0.46 | 0.7939 | 0.0000 | ✗ |

**Mean ± SD per mode × distance:**

| Mode | MMD | DTW | Wasserstein |
|------|----:|----:|------------:|
| Cross-domain | 0.5227±0.0143 | 0.5154±0.0086 | 0.5228±0.0188 |
| Within-domain | 0.7709±0.1481 | 0.7596±0.1567 | 0.7893±0.1422 |
| Mixed | 0.7710±0.1441 | 0.7699±0.1408 | 0.7775±0.1389 |

### 2. Distance effect by Condition (pooling modes & levels)

| Condition | N | H | p | η² | Sig? |
|-----------|--:|--:|--:|---:|:----:|
| baseline | 186 | 6.56 | 0.0376 | 0.0249 | ✓ |
| rus_r01 | 190 | 2.66 | 0.2640 | 0.0035 | ✗ |
| rus_r05 | 191 | 5.40 | 0.0671 | 0.0181 | ✗ |
| smote_r01 | 186 | 0.83 | 0.6616 | 0.0000 | ✗ |
| smote_r05 | 185 | 1.95 | 0.3770 | 0.0000 | ✗ |
| sw_smote_r01 | 184 | 2.69 | 0.2607 | 0.0038 | ✗ |
| sw_smote_r05 | 184 | 0.82 | 0.6650 | 0.0000 | ✗ |

### 3. Distance effect by Level (pooling modes & conditions)

| Level | N | H | p | η² | Sig? |
|-------|--:|--:|--:|---:|:----:|
| In-domain | 655 | 11.08 | 0.0039 | 0.0139 | ✓ |
| Out-domain | 651 | 2.35 | 0.3091 | 0.0005 | ✗ |

### 4. Distance effect by Mode × Level (pooling conditions)

| Mode | Level | N | H | p | η² | Sig? |
|------|-------|--:|--:|--:|---:|:----:|
| Cross-domain | In-domain | 224 | 38.65 | 0.0000 | 0.1658 | ✓ |
| Cross-domain | Out-domain | 221 | 78.93 | 0.0000 | 0.3529 | ✓ |
| Within-domain | In-domain | 219 | 3.92 | 0.1407 | 0.0089 | ✗ |
| Within-domain | Out-domain | 219 | 3.08 | 0.2144 | 0.0050 | ✗ |
| Mixed | In-domain | 212 | 0.77 | 0.6791 | 0.0000 | ✗ |
| Mixed | Out-domain | 211 | 0.77 | 0.6815 | 0.0000 | ✗ |

### 5. Distance effect by Condition × Level (pooling modes)

| Condition | Level | N | H | p | η² | Sig? |
|-----------|-------|--:|--:|--:|---:|:----:|
| baseline | In-domain | 92 | 7.67 | 0.0216 | 0.0637 | ✓ |
| baseline | Out-domain | 94 | 2.08 | 0.3531 | 0.0009 | ✗ |
| rus_r01 | In-domain | 95 | 4.49 | 0.1061 | 0.0270 | ✗ |
| rus_r01 | Out-domain | 95 | 1.00 | 0.6065 | 0.0000 | ✗ |
| rus_r05 | In-domain | 96 | 8.31 | 0.0156 | 0.0679 | ✓ |
| rus_r05 | Out-domain | 95 | 7.72 | 0.0211 | 0.0621 | ✓ |
| smote_r01 | In-domain | 93 | 1.33 | 0.5137 | 0.0000 | ✗ |
| smote_r01 | Out-domain | 93 | 0.23 | 0.8898 | 0.0000 | ✗ |
| smote_r05 | In-domain | 93 | 3.28 | 0.1942 | 0.0142 | ✗ |
| smote_r05 | Out-domain | 92 | 0.29 | 0.8653 | 0.0000 | ✗ |
| sw_smote_r01 | In-domain | 93 | 3.40 | 0.1826 | 0.0156 | ✗ |
| sw_smote_r01 | Out-domain | 91 | 1.65 | 0.4379 | 0.0000 | ✗ |
| sw_smote_r05 | In-domain | 93 | 1.39 | 0.4998 | 0.0000 | ✗ |
| sw_smote_r05 | Out-domain | 91 | 0.62 | 0.7343 | 0.0000 | ✗ |

### 6. Distance effect by Condition × Mode (pooling levels)

| Condition | Mode | N | H | p | η² | Sig? |
|-----------|------|--:|--:|--:|---:|:----:|
| baseline | Cross-domain | 64 | 19.04 | 0.0001 | 0.2793 | ✓ |
| baseline | Within-domain | 62 | 27.35 | 0.0000 | 0.4297 | ✓ |
| baseline | Mixed | 60 | 1.04 | 0.5936 | 0.0000 | ✗ |
| rus_r01 | Cross-domain | 64 | 2.64 | 0.2671 | 0.0105 | ✗ |
| rus_r01 | Within-domain | 64 | 3.80 | 0.1498 | 0.0295 | ✗ |
| rus_r01 | Mixed | 62 | 0.32 | 0.8541 | 0.0000 | ✗ |
| rus_r05 | Cross-domain | 64 | 1.77 | 0.4134 | 0.0000 | ✗ |
| rus_r05 | Within-domain | 64 | 4.40 | 0.1109 | 0.0393 | ✗ |
| rus_r05 | Mixed | 63 | 1.36 | 0.5055 | 0.0000 | ✗ |
| smote_r01 | Cross-domain | 64 | 13.79 | 0.0010 | 0.1933 | ✓ |
| smote_r01 | Within-domain | 62 | 0.40 | 0.8180 | 0.0000 | ✗ |
| smote_r01 | Mixed | 60 | 0.58 | 0.7465 | 0.0000 | ✗ |
| smote_r05 | Cross-domain | 63 | 28.55 | 0.0000 | 0.4425 | ✓ |
| smote_r05 | Within-domain | 62 | 0.66 | 0.7195 | 0.0000 | ✗ |
| smote_r05 | Mixed | 60 | 0.53 | 0.7689 | 0.0000 | ✗ |
| sw_smote_r01 | Cross-domain | 63 | 7.22 | 0.0271 | 0.0870 | ✓ |
| sw_smote_r01 | Within-domain | 62 | 6.50 | 0.0388 | 0.0762 | ✓ |
| sw_smote_r01 | Mixed | 59 | 0.45 | 0.7967 | 0.0000 | ✗ |
| sw_smote_r05 | Cross-domain | 63 | 1.33 | 0.5134 | 0.0000 | ✗ |
| sw_smote_r05 | Within-domain | 62 | 1.44 | 0.4873 | 0.0000 | ✗ |
| sw_smote_r05 | Mixed | 59 | 0.01 | 0.9937 | 0.0000 | ✗ |


## F1-score

### 1. Distance effect by Mode (pooling conditions & levels)

| Mode | N | H | p | η² | Sig? |
|------|--:|--:|--:|---:|:----:|
| Cross-domain | 445 | 17.38 | 0.0002 | 0.0348 | ✓ |
| Within-domain | 438 | 1.28 | 0.5265 | 0.0000 | ✗ |
| Mixed | 423 | 0.12 | 0.9429 | 0.0000 | ✗ |

**Mean ± SD per mode × distance:**

| Mode | MMD | DTW | Wasserstein |
|------|----:|----:|------------:|
| Cross-domain | 0.0741±0.0169 | 0.0698±0.0147 | 0.0722±0.0157 |
| Within-domain | 0.2441±0.1634 | 0.2343±0.1611 | 0.2503±0.1694 |
| Mixed | 0.2580±0.1686 | 0.2590±0.1715 | 0.2650±0.1747 |

### 2. Distance effect by Condition (pooling modes & levels)

| Condition | N | H | p | η² | Sig? |
|-----------|--:|--:|--:|---:|:----:|
| baseline | 186 | 3.61 | 0.1642 | 0.0088 | ✗ |
| rus_r01 | 190 | 0.50 | 0.7796 | 0.0000 | ✗ |
| rus_r05 | 191 | 1.92 | 0.3825 | 0.0000 | ✗ |
| smote_r01 | 186 | 0.80 | 0.6715 | 0.0000 | ✗ |
| smote_r05 | 185 | 1.34 | 0.5128 | 0.0000 | ✗ |
| sw_smote_r01 | 184 | 0.21 | 0.9017 | 0.0000 | ✗ |
| sw_smote_r05 | 184 | 0.58 | 0.7478 | 0.0000 | ✗ |

### 3. Distance effect by Level (pooling modes & conditions)

| Level | N | H | p | η² | Sig? |
|-------|--:|--:|--:|---:|:----:|
| In-domain | 655 | 3.04 | 0.2189 | 0.0016 | ✗ |
| Out-domain | 651 | 5.89 | 0.0527 | 0.0060 | ✗ |

### 4. Distance effect by Mode × Level (pooling conditions)

| Mode | Level | N | H | p | η² | Sig? |
|------|-------|--:|--:|--:|---:|:----:|
| Cross-domain | In-domain | 224 | 10.45 | 0.0054 | 0.0382 | ✓ |
| Cross-domain | Out-domain | 221 | 19.46 | 0.0001 | 0.0801 | ✓ |
| Within-domain | In-domain | 219 | 0.60 | 0.7406 | 0.0000 | ✗ |
| Within-domain | Out-domain | 219 | 2.45 | 0.2941 | 0.0021 | ✗ |
| Mixed | In-domain | 212 | 0.37 | 0.8329 | 0.0000 | ✗ |
| Mixed | Out-domain | 211 | 1.28 | 0.5284 | 0.0000 | ✗ |

### 5. Distance effect by Condition × Level (pooling modes)

| Condition | Level | N | H | p | η² | Sig? |
|-----------|-------|--:|--:|--:|---:|:----:|
| baseline | In-domain | 92 | 0.20 | 0.9037 | 0.0000 | ✗ |
| baseline | Out-domain | 94 | 5.76 | 0.0562 | 0.0413 | ✗ |
| rus_r01 | In-domain | 95 | 8.05 | 0.0179 | 0.0657 | ✓ |
| rus_r01 | Out-domain | 95 | 3.90 | 0.1423 | 0.0206 | ✗ |
| rus_r05 | In-domain | 96 | 17.73 | 0.0001 | 0.1691 | ✓ |
| rus_r05 | Out-domain | 95 | 15.28 | 0.0005 | 0.1443 | ✓ |
| smote_r01 | In-domain | 93 | 2.79 | 0.2477 | 0.0088 | ✗ |
| smote_r01 | Out-domain | 93 | 2.47 | 0.2911 | 0.0052 | ✗ |
| smote_r05 | In-domain | 93 | 1.98 | 0.3717 | 0.0000 | ✗ |
| smote_r05 | Out-domain | 92 | 0.74 | 0.6893 | 0.0000 | ✗ |
| sw_smote_r01 | In-domain | 93 | 0.30 | 0.8598 | 0.0000 | ✗ |
| sw_smote_r01 | Out-domain | 91 | 0.01 | 0.9949 | 0.0000 | ✗ |
| sw_smote_r05 | In-domain | 93 | 0.68 | 0.7115 | 0.0000 | ✗ |
| sw_smote_r05 | Out-domain | 91 | 0.41 | 0.8150 | 0.0000 | ✗ |

### 6. Distance effect by Condition × Mode (pooling levels)

| Condition | Mode | N | H | p | η² | Sig? |
|-----------|------|--:|--:|--:|---:|:----:|
| baseline | Cross-domain | 64 | 9.52 | 0.0086 | 0.1233 | ✓ |
| baseline | Within-domain | 62 | 28.51 | 0.0000 | 0.4493 | ✓ |
| baseline | Mixed | 60 | 2.06 | 0.3570 | 0.0011 | ✗ |
| rus_r01 | Cross-domain | 64 | 2.53 | 0.2823 | 0.0087 | ✗ |
| rus_r01 | Within-domain | 64 | 2.89 | 0.2357 | 0.0146 | ✗ |
| rus_r01 | Mixed | 62 | 0.93 | 0.6276 | 0.0000 | ✗ |
| rus_r05 | Cross-domain | 64 | 0.83 | 0.6614 | 0.0000 | ✗ |
| rus_r05 | Within-domain | 64 | 0.40 | 0.8170 | 0.0000 | ✗ |
| rus_r05 | Mixed | 63 | 0.39 | 0.8232 | 0.0000 | ✗ |
| smote_r01 | Cross-domain | 64 | 20.27 | 0.0000 | 0.2995 | ✓ |
| smote_r01 | Within-domain | 62 | 0.23 | 0.8935 | 0.0000 | ✗ |
| smote_r01 | Mixed | 60 | 0.26 | 0.8790 | 0.0000 | ✗ |
| smote_r05 | Cross-domain | 63 | 22.52 | 0.0000 | 0.3420 | ✓ |
| smote_r05 | Within-domain | 62 | 0.73 | 0.6933 | 0.0000 | ✗ |
| smote_r05 | Mixed | 60 | 0.08 | 0.9632 | 0.0000 | ✗ |
| sw_smote_r01 | Cross-domain | 63 | 10.58 | 0.0050 | 0.1430 | ✓ |
| sw_smote_r01 | Within-domain | 62 | 3.95 | 0.1385 | 0.0331 | ✗ |
| sw_smote_r01 | Mixed | 59 | 0.11 | 0.9461 | 0.0000 | ✗ |
| sw_smote_r05 | Cross-domain | 63 | 8.54 | 0.0140 | 0.1089 | ✓ |
| sw_smote_r05 | Within-domain | 62 | 1.03 | 0.5969 | 0.0000 | ✗ |
| sw_smote_r05 | Mixed | 59 | 0.05 | 0.9749 | 0.0000 | ✗ |


## AUPRC

### 1. Distance effect by Mode (pooling conditions & levels)

| Mode | N | H | p | η² | Sig? |
|------|--:|--:|--:|---:|:----:|
| Cross-domain | 445 | 7.12 | 0.0284 | 0.0116 | ✓ |
| Within-domain | 438 | 2.60 | 0.2722 | 0.0014 | ✗ |
| Mixed | 423 | 0.04 | 0.9810 | 0.0000 | ✗ |

**Mean ± SD per mode × distance:**

| Mode | MMD | DTW | Wasserstein |
|------|----:|----:|------------:|
| Cross-domain | 0.0485±0.0049 | 0.0487±0.0063 | 0.0516±0.0101 |
| Within-domain | 0.3595±0.2734 | 0.3562±0.2766 | 0.3890±0.2810 |
| Mixed | 0.3792±0.2814 | 0.3782±0.2835 | 0.3835±0.2840 |

### 2. Distance effect by Condition (pooling modes & levels)

| Condition | N | H | p | η² | Sig? |
|-----------|--:|--:|--:|---:|:----:|
| baseline | 186 | 2.86 | 0.2396 | 0.0047 | ✗ |
| rus_r01 | 190 | 1.67 | 0.4341 | 0.0000 | ✗ |
| rus_r05 | 191 | 1.71 | 0.4253 | 0.0000 | ✗ |
| smote_r01 | 186 | 0.14 | 0.9317 | 0.0000 | ✗ |
| smote_r05 | 185 | 0.32 | 0.8506 | 0.0000 | ✗ |
| sw_smote_r01 | 184 | 1.72 | 0.4224 | 0.0000 | ✗ |
| sw_smote_r05 | 184 | 1.19 | 0.5515 | 0.0000 | ✗ |

### 3. Distance effect by Level (pooling modes & conditions)

| Level | N | H | p | η² | Sig? |
|-------|--:|--:|--:|---:|:----:|
| In-domain | 655 | 4.21 | 0.1219 | 0.0034 | ✗ |
| Out-domain | 651 | 2.66 | 0.2646 | 0.0010 | ✗ |

### 4. Distance effect by Mode × Level (pooling conditions)

| Mode | Level | N | H | p | η² | Sig? |
|------|-------|--:|--:|--:|---:|:----:|
| Cross-domain | In-domain | 224 | 28.51 | 0.0000 | 0.1199 | ✓ |
| Cross-domain | Out-domain | 221 | 24.66 | 0.0000 | 0.1040 | ✓ |
| Within-domain | In-domain | 219 | 3.98 | 0.1364 | 0.0092 | ✗ |
| Within-domain | Out-domain | 219 | 1.91 | 0.3840 | 0.0000 | ✗ |
| Mixed | In-domain | 212 | 0.03 | 0.9836 | 0.0000 | ✗ |
| Mixed | Out-domain | 211 | 0.13 | 0.9363 | 0.0000 | ✗ |

### 5. Distance effect by Condition × Level (pooling modes)

| Condition | Level | N | H | p | η² | Sig? |
|-----------|-------|--:|--:|--:|---:|:----:|
| baseline | In-domain | 92 | 0.73 | 0.6944 | 0.0000 | ✗ |
| baseline | Out-domain | 94 | 2.37 | 0.3056 | 0.0041 | ✗ |
| rus_r01 | In-domain | 95 | 1.91 | 0.3840 | 0.0000 | ✗ |
| rus_r01 | Out-domain | 95 | 1.16 | 0.5599 | 0.0000 | ✗ |
| rus_r05 | In-domain | 96 | 3.79 | 0.1500 | 0.0193 | ✗ |
| rus_r05 | Out-domain | 95 | 1.33 | 0.5149 | 0.0000 | ✗ |
| smote_r01 | In-domain | 93 | 0.58 | 0.7493 | 0.0000 | ✗ |
| smote_r01 | Out-domain | 93 | 1.73 | 0.4215 | 0.0000 | ✗ |
| smote_r05 | In-domain | 93 | 0.46 | 0.7944 | 0.0000 | ✗ |
| smote_r05 | Out-domain | 92 | 0.33 | 0.8461 | 0.0000 | ✗ |
| sw_smote_r01 | In-domain | 93 | 3.34 | 0.1878 | 0.0149 | ✗ |
| sw_smote_r01 | Out-domain | 91 | 1.09 | 0.5802 | 0.0000 | ✗ |
| sw_smote_r05 | In-domain | 93 | 1.56 | 0.4592 | 0.0000 | ✗ |
| sw_smote_r05 | Out-domain | 91 | 1.56 | 0.4573 | 0.0000 | ✗ |

### 6. Distance effect by Condition × Mode (pooling levels)

| Condition | Mode | N | H | p | η² | Sig? |
|-----------|------|--:|--:|--:|---:|:----:|
| baseline | Cross-domain | 64 | 1.38 | 0.5023 | 0.0000 | ✗ |
| baseline | Within-domain | 62 | 18.59 | 0.0001 | 0.2813 | ✓ |
| baseline | Mixed | 60 | 0.04 | 0.9825 | 0.0000 | ✗ |
| rus_r01 | Cross-domain | 64 | 1.35 | 0.5094 | 0.0000 | ✗ |
| rus_r01 | Within-domain | 64 | 0.66 | 0.7182 | 0.0000 | ✗ |
| rus_r01 | Mixed | 62 | 0.02 | 0.9898 | 0.0000 | ✗ |
| rus_r05 | Cross-domain | 64 | 1.68 | 0.4321 | 0.0000 | ✗ |
| rus_r05 | Within-domain | 64 | 2.80 | 0.2466 | 0.0131 | ✗ |
| rus_r05 | Mixed | 63 | 0.63 | 0.7298 | 0.0000 | ✗ |
| smote_r01 | Cross-domain | 64 | 1.35 | 0.5081 | 0.0000 | ✗ |
| smote_r01 | Within-domain | 62 | 0.21 | 0.8990 | 0.0000 | ✗ |
| smote_r01 | Mixed | 60 | 0.13 | 0.9367 | 0.0000 | ✗ |
| smote_r05 | Cross-domain | 63 | 7.04 | 0.0296 | 0.0840 | ✓ |
| smote_r05 | Within-domain | 62 | 0.33 | 0.8480 | 0.0000 | ✗ |
| smote_r05 | Mixed | 60 | 0.04 | 0.9818 | 0.0000 | ✗ |
| sw_smote_r01 | Cross-domain | 63 | 3.89 | 0.1427 | 0.0316 | ✗ |
| sw_smote_r01 | Within-domain | 62 | 4.75 | 0.0929 | 0.0466 | ✗ |
| sw_smote_r01 | Mixed | 59 | 0.06 | 0.9710 | 0.0000 | ✗ |
| sw_smote_r05 | Cross-domain | 63 | 4.62 | 0.0991 | 0.0437 | ✗ |
| sw_smote_r05 | Within-domain | 62 | 1.13 | 0.5686 | 0.0000 | ✗ |
| sw_smote_r05 | Mixed | 59 | 0.03 | 0.9863 | 0.0000 | ✗ |


## Recall

### 1. Distance effect by Mode (pooling conditions & levels)

| Mode | N | H | p | η² | Sig? |
|------|--:|--:|--:|---:|:----:|
| Cross-domain | 445 | 2.15 | 0.3405 | 0.0003 | ✗ |
| Within-domain | 438 | 4.44 | 0.1088 | 0.0056 | ✗ |
| Mixed | 423 | 0.08 | 0.9614 | 0.0000 | ✗ |

**Mean ± SD per mode × distance:**

| Mode | MMD | DTW | Wasserstein |
|------|----:|----:|------------:|
| Cross-domain | 0.3050±0.1724 | 0.2820±0.1664 | 0.3033±0.1824 |
| Within-domain | 0.6804±0.1933 | 0.6327±0.2163 | 0.6725±0.2217 |
| Mixed | 0.6567±0.1941 | 0.6568±0.1902 | 0.6464±0.2033 |

### 2. Distance effect by Condition (pooling modes & levels)

| Condition | N | H | p | η² | Sig? |
|-----------|--:|--:|--:|---:|:----:|
| baseline | 186 | 12.28 | 0.0022 | 0.0562 | ✓ |
| rus_r01 | 190 | 3.26 | 0.1958 | 0.0067 | ✗ |
| rus_r05 | 191 | 1.99 | 0.3695 | 0.0000 | ✗ |
| smote_r01 | 186 | 0.59 | 0.7438 | 0.0000 | ✗ |
| smote_r05 | 185 | 1.24 | 0.5377 | 0.0000 | ✗ |
| sw_smote_r01 | 184 | 2.24 | 0.3262 | 0.0013 | ✗ |
| sw_smote_r05 | 184 | 0.80 | 0.6687 | 0.0000 | ✗ |

### 3. Distance effect by Level (pooling modes & conditions)

| Level | N | H | p | η² | Sig? |
|-------|--:|--:|--:|---:|:----:|
| In-domain | 655 | 0.42 | 0.8096 | 0.0000 | ✗ |
| Out-domain | 651 | 3.55 | 0.1692 | 0.0024 | ✗ |

### 4. Distance effect by Mode × Level (pooling conditions)

| Mode | Level | N | H | p | η² | Sig? |
|------|-------|--:|--:|--:|---:|:----:|
| Cross-domain | In-domain | 224 | 1.40 | 0.4959 | 0.0000 | ✗ |
| Cross-domain | Out-domain | 221 | 3.79 | 0.1501 | 0.0082 | ✗ |
| Within-domain | In-domain | 219 | 2.16 | 0.3391 | 0.0008 | ✗ |
| Within-domain | Out-domain | 219 | 4.74 | 0.0934 | 0.0127 | ✗ |
| Mixed | In-domain | 212 | 0.35 | 0.8412 | 0.0000 | ✗ |
| Mixed | Out-domain | 211 | 0.28 | 0.8705 | 0.0000 | ✗ |

### 5. Distance effect by Condition × Level (pooling modes)

| Condition | Level | N | H | p | η² | Sig? |
|-----------|-------|--:|--:|--:|---:|:----:|
| baseline | In-domain | 92 | 4.10 | 0.1290 | 0.0235 | ✗ |
| baseline | Out-domain | 94 | 9.18 | 0.0101 | 0.0789 | ✓ |
| rus_r01 | In-domain | 95 | 2.73 | 0.2557 | 0.0079 | ✗ |
| rus_r01 | Out-domain | 95 | 0.62 | 0.7334 | 0.0000 | ✗ |
| rus_r05 | In-domain | 96 | 9.41 | 0.0091 | 0.0796 | ✓ |
| rus_r05 | Out-domain | 95 | 26.22 | 0.0000 | 0.2632 | ✓ |
| smote_r01 | In-domain | 93 | 1.30 | 0.5226 | 0.0000 | ✗ |
| smote_r01 | Out-domain | 93 | 2.72 | 0.2570 | 0.0080 | ✗ |
| smote_r05 | In-domain | 93 | 2.13 | 0.3450 | 0.0014 | ✗ |
| smote_r05 | Out-domain | 92 | 0.00 | 0.9975 | 0.0000 | ✗ |
| sw_smote_r01 | In-domain | 93 | 1.88 | 0.3915 | 0.0000 | ✗ |
| sw_smote_r01 | Out-domain | 91 | 2.39 | 0.3029 | 0.0044 | ✗ |
| sw_smote_r05 | In-domain | 93 | 1.26 | 0.5324 | 0.0000 | ✗ |
| sw_smote_r05 | Out-domain | 91 | 0.29 | 0.8658 | 0.0000 | ✗ |

### 6. Distance effect by Condition × Mode (pooling levels)

| Condition | Mode | N | H | p | η² | Sig? |
|-----------|------|--:|--:|--:|---:|:----:|
| baseline | Cross-domain | 64 | 11.69 | 0.0029 | 0.1588 | ✓ |
| baseline | Within-domain | 62 | 30.01 | 0.0000 | 0.4747 | ✓ |
| baseline | Mixed | 60 | 1.54 | 0.4640 | 0.0000 | ✗ |
| rus_r01 | Cross-domain | 64 | 2.53 | 0.2828 | 0.0086 | ✗ |
| rus_r01 | Within-domain | 64 | 7.40 | 0.0247 | 0.0886 | ✓ |
| rus_r01 | Mixed | 62 | 3.51 | 0.1726 | 0.0257 | ✗ |
| rus_r05 | Cross-domain | 64 | 1.78 | 0.4108 | 0.0000 | ✗ |
| rus_r05 | Within-domain | 64 | 1.91 | 0.3844 | 0.0000 | ✗ |
| rus_r05 | Mixed | 63 | 4.58 | 0.1011 | 0.0431 | ✗ |
| smote_r01 | Cross-domain | 64 | 5.81 | 0.0547 | 0.0625 | ✗ |
| smote_r01 | Within-domain | 62 | 1.60 | 0.4493 | 0.0000 | ✗ |
| smote_r01 | Mixed | 60 | 0.92 | 0.6304 | 0.0000 | ✗ |
| smote_r05 | Cross-domain | 63 | 7.81 | 0.0202 | 0.0968 | ✓ |
| smote_r05 | Within-domain | 62 | 0.34 | 0.8456 | 0.0000 | ✗ |
| smote_r05 | Mixed | 60 | 1.29 | 0.5249 | 0.0000 | ✗ |
| sw_smote_r01 | Cross-domain | 63 | 2.85 | 0.2400 | 0.0142 | ✗ |
| sw_smote_r01 | Within-domain | 62 | 9.33 | 0.0094 | 0.1243 | ✓ |
| sw_smote_r01 | Mixed | 59 | 0.15 | 0.9258 | 0.0000 | ✗ |
| sw_smote_r05 | Cross-domain | 63 | 7.40 | 0.0247 | 0.0900 | ✓ |
| sw_smote_r05 | Within-domain | 62 | 1.64 | 0.4411 | 0.0000 | ✗ |
| sw_smote_r05 | Mixed | 59 | 0.05 | 0.9731 | 0.0000 | ✗ |


---
## Summary

The above tables show Kruskal-Wallis tests for distance metric effect at each level of stratification. Significance is assessed at α=0.05 (uncorrected per-test). For multiple testing across cells within a section, apply Bonferroni correction as needed.
