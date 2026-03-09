# Distance Metric Granular Analysis

Records: 1258, Distances: ['mmd', 'dtw', 'wasserstein']

H₀ for each cell: performance under MMD = DTW = Wasserstein (Kruskal-Wallis)

---


## F2-score

### 1. Distance effect by Mode (pooling conditions & levels)

| Mode | N | H | p | η² | Sig? |
|------|--:|--:|--:|---:|:----:|
| Cross-domain | 420 | 4.03 | 0.1335 | 0.0049 | ✗ |
| Within-domain | 420 | 1.43 | 0.4902 | 0.0000 | ✗ |
| Mixed | 418 | 0.05 | 0.9733 | 0.0000 | ✗ |

**Mean ± SD per mode × distance:**

| Mode | MMD | DTW | Wasserstein |
|------|----:|----:|------------:|
| Cross-domain | 0.1285±0.0443 | 0.1209±0.0418 | 0.1259±0.0430 |
| Within-domain | 0.3648±0.1829 | 0.3555±0.1868 | 0.3769±0.1916 |
| Mixed | 0.3719±0.1772 | 0.3776±0.1786 | 0.3768±0.1830 |

### 2. Distance effect by Condition (pooling modes & levels)

| Condition | N | H | p | η² | Sig? |
|-----------|--:|--:|--:|---:|:----:|
| baseline | 180 | 5.00 | 0.0823 | 0.0169 | ✗ |
| rus_r01 | 180 | 1.24 | 0.5392 | 0.0000 | ✗ |
| rus_r05 | 180 | 2.45 | 0.2940 | 0.0025 | ✗ |
| smote_r01 | 180 | 0.46 | 0.7932 | 0.0000 | ✗ |
| smote_r05 | 180 | 0.69 | 0.7069 | 0.0000 | ✗ |
| sw_smote_r01 | 179 | 0.25 | 0.8823 | 0.0000 | ✗ |
| sw_smote_r05 | 179 | 0.37 | 0.8330 | 0.0000 | ✗ |

### 3. Distance effect by Level (pooling modes & conditions)

| Level | N | H | p | η² | Sig? |
|-------|--:|--:|--:|---:|:----:|
| In-domain | 629 | 1.26 | 0.5319 | 0.0000 | ✗ |
| Out-domain | 629 | 2.04 | 0.3610 | 0.0001 | ✗ |

### 4. Distance effect by Mode × Level (pooling conditions)

| Mode | Level | N | H | p | η² | Sig? |
|------|-------|--:|--:|--:|---:|:----:|
| Cross-domain | In-domain | 210 | 0.31 | 0.8557 | 0.0000 | ✗ |
| Cross-domain | Out-domain | 210 | 5.83 | 0.0541 | 0.0185 | ✗ |
| Within-domain | In-domain | 210 | 0.43 | 0.8073 | 0.0000 | ✗ |
| Within-domain | Out-domain | 210 | 2.61 | 0.2706 | 0.0030 | ✗ |
| Mixed | In-domain | 209 | 0.49 | 0.7828 | 0.0000 | ✗ |
| Mixed | Out-domain | 209 | 0.65 | 0.7242 | 0.0000 | ✗ |

### 5. Distance effect by Condition × Level (pooling modes)

| Condition | Level | N | H | p | η² | Sig? |
|-----------|-------|--:|--:|--:|---:|:----:|
| baseline | In-domain | 90 | 0.57 | 0.7529 | 0.0000 | ✗ |
| baseline | Out-domain | 90 | 5.70 | 0.0577 | 0.0426 | ✗ |
| rus_r01 | In-domain | 90 | 5.94 | 0.0514 | 0.0452 | ✗ |
| rus_r01 | Out-domain | 90 | 0.93 | 0.6290 | 0.0000 | ✗ |
| rus_r05 | In-domain | 90 | 13.78 | 0.0010 | 0.1354 | ✓ |
| rus_r05 | Out-domain | 90 | 13.91 | 0.0010 | 0.1369 | ✓ |
| smote_r01 | In-domain | 90 | 3.10 | 0.2125 | 0.0126 | ✗ |
| smote_r01 | Out-domain | 90 | 1.84 | 0.3983 | 0.0000 | ✗ |
| smote_r05 | In-domain | 90 | 1.38 | 0.5006 | 0.0000 | ✗ |
| smote_r05 | Out-domain | 90 | 0.07 | 0.9670 | 0.0000 | ✗ |
| sw_smote_r01 | In-domain | 90 | 0.60 | 0.7405 | 0.0000 | ✗ |
| sw_smote_r01 | Out-domain | 89 | 0.00 | 0.9981 | 0.0000 | ✗ |
| sw_smote_r05 | In-domain | 89 | 0.66 | 0.7205 | 0.0000 | ✗ |
| sw_smote_r05 | Out-domain | 90 | 0.20 | 0.9033 | 0.0000 | ✗ |

### 6. Distance effect by Condition × Mode (pooling levels)

| Condition | Mode | N | H | p | η² | Sig? |
|-----------|------|--:|--:|--:|---:|:----:|
| baseline | Cross-domain | 60 | 7.33 | 0.0256 | 0.0935 | ✓ |
| baseline | Within-domain | 60 | 31.23 | 0.0000 | 0.5129 | ✓ |
| baseline | Mixed | 60 | 2.51 | 0.2856 | 0.0089 | ✗ |
| rus_r01 | Cross-domain | 60 | 2.69 | 0.2609 | 0.0121 | ✗ |
| rus_r01 | Within-domain | 60 | 4.75 | 0.0931 | 0.0482 | ✗ |
| rus_r01 | Mixed | 60 | 1.99 | 0.3695 | 0.0000 | ✗ |
| rus_r05 | Cross-domain | 60 | 0.12 | 0.9421 | 0.0000 | ✗ |
| rus_r05 | Within-domain | 60 | 0.16 | 0.9242 | 0.0000 | ✗ |
| rus_r05 | Mixed | 60 | 1.68 | 0.4312 | 0.0000 | ✗ |
| smote_r01 | Cross-domain | 60 | 15.84 | 0.0004 | 0.2428 | ✓ |
| smote_r01 | Within-domain | 60 | 0.06 | 0.9727 | 0.0000 | ✗ |
| smote_r01 | Mixed | 60 | 0.46 | 0.7940 | 0.0000 | ✗ |
| smote_r05 | Cross-domain | 60 | 14.08 | 0.0009 | 0.2119 | ✓ |
| smote_r05 | Within-domain | 60 | 0.81 | 0.6671 | 0.0000 | ✗ |
| smote_r05 | Mixed | 60 | 0.14 | 0.9310 | 0.0000 | ✗ |
| sw_smote_r01 | Cross-domain | 60 | 6.66 | 0.0358 | 0.0817 | ✓ |
| sw_smote_r01 | Within-domain | 60 | 3.70 | 0.1573 | 0.0298 | ✗ |
| sw_smote_r01 | Mixed | 59 | 0.06 | 0.9713 | 0.0000 | ✗ |
| sw_smote_r05 | Cross-domain | 60 | 7.64 | 0.0219 | 0.0990 | ✓ |
| sw_smote_r05 | Within-domain | 60 | 1.15 | 0.5622 | 0.0000 | ✗ |
| sw_smote_r05 | Mixed | 59 | 0.04 | 0.9815 | 0.0000 | ✗ |


## AUROC

### 1. Distance effect by Mode (pooling conditions & levels)

| Mode | N | H | p | η² | Sig? |
|------|--:|--:|--:|---:|:----:|
| Cross-domain | 420 | 21.64 | 0.0000 | 0.0471 | ✓ |
| Within-domain | 420 | 4.11 | 0.1282 | 0.0051 | ✗ |
| Mixed | 418 | 0.14 | 0.9303 | 0.0000 | ✗ |

**Mean ± SD per mode × distance:**

| Mode | MMD | DTW | Wasserstein |
|------|----:|----:|------------:|
| Cross-domain | 0.5229±0.0142 | 0.5155±0.0086 | 0.5228±0.0188 |
| Within-domain | 0.7709±0.1481 | 0.7607±0.1586 | 0.7898±0.1427 |
| Mixed | 0.7729±0.1428 | 0.7760±0.1380 | 0.7775±0.1389 |

### 2. Distance effect by Condition (pooling modes & levels)

| Condition | N | H | p | η² | Sig? |
|-----------|--:|--:|--:|---:|:----:|
| baseline | 180 | 5.51 | 0.0635 | 0.0198 | ✗ |
| rus_r01 | 180 | 1.70 | 0.4270 | 0.0000 | ✗ |
| rus_r05 | 180 | 6.23 | 0.0444 | 0.0239 | ✓ |
| smote_r01 | 180 | 0.81 | 0.6667 | 0.0000 | ✗ |
| smote_r05 | 180 | 1.30 | 0.5221 | 0.0000 | ✗ |
| sw_smote_r01 | 179 | 1.96 | 0.3744 | 0.0000 | ✗ |
| sw_smote_r05 | 179 | 0.39 | 0.8241 | 0.0000 | ✗ |

### 3. Distance effect by Level (pooling modes & conditions)

| Level | N | H | p | η² | Sig? |
|-------|--:|--:|--:|---:|:----:|
| In-domain | 629 | 6.70 | 0.0351 | 0.0075 | ✓ |
| Out-domain | 629 | 1.64 | 0.4413 | 0.0000 | ✗ |

### 4. Distance effect by Mode × Level (pooling conditions)

| Mode | Level | N | H | p | η² | Sig? |
|------|-------|--:|--:|--:|---:|:----:|
| Cross-domain | In-domain | 210 | 36.71 | 0.0000 | 0.1677 | ✓ |
| Cross-domain | Out-domain | 210 | 73.82 | 0.0000 | 0.3470 | ✓ |
| Within-domain | In-domain | 210 | 3.21 | 0.2005 | 0.0059 | ✗ |
| Within-domain | Out-domain | 210 | 3.20 | 0.2022 | 0.0058 | ✗ |
| Mixed | In-domain | 209 | 0.55 | 0.7611 | 0.0000 | ✗ |
| Mixed | Out-domain | 209 | 0.74 | 0.6916 | 0.0000 | ✗ |

### 5. Distance effect by Condition × Level (pooling modes)

| Condition | Level | N | H | p | η² | Sig? |
|-----------|-------|--:|--:|--:|---:|:----:|
| baseline | In-domain | 90 | 6.73 | 0.0346 | 0.0544 | ✓ |
| baseline | Out-domain | 90 | 1.50 | 0.4720 | 0.0000 | ✗ |
| rus_r01 | In-domain | 90 | 3.46 | 0.1777 | 0.0167 | ✗ |
| rus_r01 | Out-domain | 90 | 0.81 | 0.6685 | 0.0000 | ✗ |
| rus_r05 | In-domain | 90 | 7.37 | 0.0251 | 0.0617 | ✓ |
| rus_r05 | Out-domain | 90 | 8.96 | 0.0113 | 0.0800 | ✓ |
| smote_r01 | In-domain | 90 | 1.46 | 0.4823 | 0.0000 | ✗ |
| smote_r01 | Out-domain | 90 | 0.14 | 0.9312 | 0.0000 | ✗ |
| smote_r05 | In-domain | 90 | 2.83 | 0.2426 | 0.0096 | ✗ |
| smote_r05 | Out-domain | 90 | 0.09 | 0.9562 | 0.0000 | ✗ |
| sw_smote_r01 | In-domain | 90 | 2.24 | 0.3268 | 0.0027 | ✗ |
| sw_smote_r01 | Out-domain | 89 | 1.35 | 0.5086 | 0.0000 | ✗ |
| sw_smote_r05 | In-domain | 89 | 0.75 | 0.6867 | 0.0000 | ✗ |
| sw_smote_r05 | Out-domain | 90 | 0.41 | 0.8134 | 0.0000 | ✗ |

### 6. Distance effect by Condition × Mode (pooling levels)

| Condition | Mode | N | H | p | η² | Sig? |
|-----------|------|--:|--:|--:|---:|:----:|
| baseline | Cross-domain | 60 | 17.02 | 0.0002 | 0.2635 | ✓ |
| baseline | Within-domain | 60 | 28.52 | 0.0000 | 0.4653 | ✓ |
| baseline | Mixed | 60 | 1.04 | 0.5936 | 0.0000 | ✗ |
| rus_r01 | Cross-domain | 60 | 2.62 | 0.2694 | 0.0109 | ✗ |
| rus_r01 | Within-domain | 60 | 3.24 | 0.1984 | 0.0217 | ✗ |
| rus_r01 | Mixed | 60 | 0.50 | 0.7793 | 0.0000 | ✗ |
| rus_r05 | Cross-domain | 60 | 2.31 | 0.3144 | 0.0055 | ✗ |
| rus_r05 | Within-domain | 60 | 5.50 | 0.0640 | 0.0614 | ✗ |
| rus_r05 | Mixed | 60 | 1.19 | 0.5509 | 0.0000 | ✗ |
| smote_r01 | Cross-domain | 60 | 14.69 | 0.0006 | 0.2226 | ✓ |
| smote_r01 | Within-domain | 60 | 0.19 | 0.9080 | 0.0000 | ✗ |
| smote_r01 | Mixed | 60 | 0.58 | 0.7465 | 0.0000 | ✗ |
| smote_r05 | Cross-domain | 60 | 25.72 | 0.0000 | 0.4162 | ✓ |
| smote_r05 | Within-domain | 60 | 0.67 | 0.7161 | 0.0000 | ✗ |
| smote_r05 | Mixed | 60 | 0.53 | 0.7689 | 0.0000 | ✗ |
| sw_smote_r01 | Cross-domain | 60 | 7.92 | 0.0191 | 0.1038 | ✓ |
| sw_smote_r01 | Within-domain | 60 | 6.60 | 0.0368 | 0.0807 | ✓ |
| sw_smote_r01 | Mixed | 59 | 0.45 | 0.7967 | 0.0000 | ✗ |
| sw_smote_r05 | Cross-domain | 60 | 1.02 | 0.6016 | 0.0000 | ✗ |
| sw_smote_r05 | Within-domain | 60 | 1.01 | 0.6028 | 0.0000 | ✗ |
| sw_smote_r05 | Mixed | 59 | 0.01 | 0.9937 | 0.0000 | ✗ |


## F1-score

### 1. Distance effect by Mode (pooling conditions & levels)

| Mode | N | H | p | η² | Sig? |
|------|--:|--:|--:|---:|:----:|
| Cross-domain | 420 | 15.12 | 0.0005 | 0.0315 | ✓ |
| Within-domain | 420 | 0.87 | 0.6472 | 0.0000 | ✗ |
| Mixed | 418 | 0.12 | 0.9416 | 0.0000 | ✗ |

**Mean ± SD per mode × distance:**

| Mode | MMD | DTW | Wasserstein |
|------|----:|----:|------------:|
| Cross-domain | 0.0739±0.0170 | 0.0698±0.0145 | 0.0722±0.0157 |
| Within-domain | 0.2441±0.1634 | 0.2388±0.1626 | 0.2528±0.1718 |
| Mixed | 0.2593±0.1685 | 0.2637±0.1717 | 0.2650±0.1747 |

### 2. Distance effect by Condition (pooling modes & levels)

| Condition | N | H | p | η² | Sig? |
|-----------|--:|--:|--:|---:|:----:|
| baseline | 180 | 2.96 | 0.2276 | 0.0054 | ✗ |
| rus_r01 | 180 | 0.59 | 0.7447 | 0.0000 | ✗ |
| rus_r05 | 180 | 1.97 | 0.3740 | 0.0000 | ✗ |
| smote_r01 | 180 | 0.60 | 0.7407 | 0.0000 | ✗ |
| smote_r05 | 180 | 0.78 | 0.6760 | 0.0000 | ✗ |
| sw_smote_r01 | 179 | 0.17 | 0.9162 | 0.0000 | ✗ |
| sw_smote_r05 | 179 | 0.33 | 0.8494 | 0.0000 | ✗ |

### 3. Distance effect by Level (pooling modes & conditions)

| Level | N | H | p | η² | Sig? |
|-------|--:|--:|--:|---:|:----:|
| In-domain | 629 | 4.25 | 0.1192 | 0.0036 | ✗ |
| Out-domain | 629 | 3.92 | 0.1408 | 0.0031 | ✗ |

### 4. Distance effect by Mode × Level (pooling conditions)

| Mode | Level | N | H | p | η² | Sig? |
|------|-------|--:|--:|--:|---:|:----:|
| Cross-domain | In-domain | 210 | 10.44 | 0.0054 | 0.0408 | ✓ |
| Cross-domain | Out-domain | 210 | 17.00 | 0.0002 | 0.0725 | ✓ |
| Within-domain | In-domain | 210 | 0.48 | 0.7848 | 0.0000 | ✗ |
| Within-domain | Out-domain | 210 | 2.52 | 0.2832 | 0.0025 | ✗ |
| Mixed | In-domain | 209 | 0.56 | 0.7541 | 0.0000 | ✗ |
| Mixed | Out-domain | 209 | 1.22 | 0.5446 | 0.0000 | ✗ |

### 5. Distance effect by Condition × Level (pooling modes)

| Condition | Level | N | H | p | η² | Sig? |
|-----------|-------|--:|--:|--:|---:|:----:|
| baseline | In-domain | 90 | 0.30 | 0.8625 | 0.0000 | ✗ |
| baseline | Out-domain | 90 | 4.54 | 0.1031 | 0.0292 | ✗ |
| rus_r01 | In-domain | 90 | 7.05 | 0.0295 | 0.0580 | ✓ |
| rus_r01 | Out-domain | 90 | 3.40 | 0.1827 | 0.0161 | ✗ |
| rus_r05 | In-domain | 90 | 16.99 | 0.0002 | 0.1723 | ✓ |
| rus_r05 | Out-domain | 90 | 13.15 | 0.0014 | 0.1281 | ✓ |
| smote_r01 | In-domain | 90 | 3.10 | 0.2118 | 0.0127 | ✗ |
| smote_r01 | Out-domain | 90 | 2.02 | 0.3638 | 0.0003 | ✗ |
| smote_r05 | In-domain | 90 | 1.82 | 0.4024 | 0.0000 | ✗ |
| smote_r05 | Out-domain | 90 | 0.52 | 0.7716 | 0.0000 | ✗ |
| sw_smote_r01 | In-domain | 90 | 0.36 | 0.8344 | 0.0000 | ✗ |
| sw_smote_r01 | Out-domain | 89 | 0.01 | 0.9944 | 0.0000 | ✗ |
| sw_smote_r05 | In-domain | 89 | 0.46 | 0.7965 | 0.0000 | ✗ |
| sw_smote_r05 | Out-domain | 90 | 0.32 | 0.8529 | 0.0000 | ✗ |

### 6. Distance effect by Condition × Mode (pooling levels)

| Condition | Mode | N | H | p | η² | Sig? |
|-----------|------|--:|--:|--:|---:|:----:|
| baseline | Cross-domain | 60 | 10.48 | 0.0053 | 0.1487 | ✓ |
| baseline | Within-domain | 60 | 29.83 | 0.0000 | 0.4882 | ✓ |
| baseline | Mixed | 60 | 2.06 | 0.3570 | 0.0011 | ✗ |
| rus_r01 | Cross-domain | 60 | 2.70 | 0.2589 | 0.0123 | ✗ |
| rus_r01 | Within-domain | 60 | 3.27 | 0.1949 | 0.0223 | ✗ |
| rus_r01 | Mixed | 60 | 1.03 | 0.5980 | 0.0000 | ✗ |
| rus_r05 | Cross-domain | 60 | 0.95 | 0.6217 | 0.0000 | ✗ |
| rus_r05 | Within-domain | 60 | 0.06 | 0.9724 | 0.0000 | ✗ |
| rus_r05 | Mixed | 60 | 0.36 | 0.8372 | 0.0000 | ✗ |
| smote_r01 | Cross-domain | 60 | 17.78 | 0.0001 | 0.2768 | ✓ |
| smote_r01 | Within-domain | 60 | 0.07 | 0.9641 | 0.0000 | ✗ |
| smote_r01 | Mixed | 60 | 0.26 | 0.8790 | 0.0000 | ✗ |
| smote_r05 | Cross-domain | 60 | 19.07 | 0.0001 | 0.2995 | ✓ |
| smote_r05 | Within-domain | 60 | 0.80 | 0.6691 | 0.0000 | ✗ |
| smote_r05 | Mixed | 60 | 0.08 | 0.9632 | 0.0000 | ✗ |
| sw_smote_r01 | Cross-domain | 60 | 11.96 | 0.0025 | 0.1747 | ✓ |
| sw_smote_r01 | Within-domain | 60 | 3.57 | 0.1675 | 0.0276 | ✗ |
| sw_smote_r01 | Mixed | 59 | 0.11 | 0.9461 | 0.0000 | ✗ |
| sw_smote_r05 | Cross-domain | 60 | 8.21 | 0.0165 | 0.1089 | ✓ |
| sw_smote_r05 | Within-domain | 60 | 0.79 | 0.6734 | 0.0000 | ✗ |
| sw_smote_r05 | Mixed | 59 | 0.05 | 0.9749 | 0.0000 | ✗ |


## AUPRC

### 1. Distance effect by Mode (pooling conditions & levels)

| Mode | N | H | p | η² | Sig? |
|------|--:|--:|--:|---:|:----:|
| Cross-domain | 420 | 7.42 | 0.0245 | 0.0130 | ✓ |
| Within-domain | 420 | 2.16 | 0.3393 | 0.0004 | ✗ |
| Mixed | 418 | 0.03 | 0.9871 | 0.0000 | ✗ |

**Mean ± SD per mode × distance:**

| Mode | MMD | DTW | Wasserstein |
|------|----:|----:|------------:|
| Cross-domain | 0.0485±0.0051 | 0.0485±0.0058 | 0.0516±0.0101 |
| Within-domain | 0.3595±0.2734 | 0.3619±0.2803 | 0.3899±0.2819 |
| Mixed | 0.3815±0.2810 | 0.3873±0.2823 | 0.3835±0.2840 |

### 2. Distance effect by Condition (pooling modes & levels)

| Condition | N | H | p | η² | Sig? |
|-----------|--:|--:|--:|---:|:----:|
| baseline | 180 | 2.16 | 0.3388 | 0.0009 | ✗ |
| rus_r01 | 180 | 0.95 | 0.6232 | 0.0000 | ✗ |
| rus_r05 | 180 | 1.88 | 0.3907 | 0.0000 | ✗ |
| smote_r01 | 180 | 0.01 | 0.9949 | 0.0000 | ✗ |
| smote_r05 | 180 | 0.06 | 0.9695 | 0.0000 | ✗ |
| sw_smote_r01 | 179 | 1.07 | 0.5858 | 0.0000 | ✗ |
| sw_smote_r05 | 179 | 0.74 | 0.6922 | 0.0000 | ✗ |

### 3. Distance effect by Level (pooling modes & conditions)

| Level | N | H | p | η² | Sig? |
|-------|--:|--:|--:|---:|:----:|
| In-domain | 629 | 1.78 | 0.4107 | 0.0000 | ✗ |
| Out-domain | 629 | 1.51 | 0.4696 | 0.0000 | ✗ |

### 4. Distance effect by Mode × Level (pooling conditions)

| Mode | Level | N | H | p | η² | Sig? |
|------|-------|--:|--:|--:|---:|:----:|
| Cross-domain | In-domain | 210 | 25.25 | 0.0000 | 0.1123 | ✓ |
| Cross-domain | Out-domain | 210 | 23.18 | 0.0000 | 0.1023 | ✓ |
| Within-domain | In-domain | 210 | 3.13 | 0.2094 | 0.0054 | ✗ |
| Within-domain | Out-domain | 210 | 1.90 | 0.3870 | 0.0000 | ✗ |
| Mixed | In-domain | 209 | 0.10 | 0.9519 | 0.0000 | ✗ |
| Mixed | Out-domain | 209 | 0.03 | 0.9830 | 0.0000 | ✗ |

### 5. Distance effect by Condition × Level (pooling modes)

| Condition | Level | N | H | p | η² | Sig? |
|-----------|-------|--:|--:|--:|---:|:----:|
| baseline | In-domain | 90 | 0.70 | 0.7039 | 0.0000 | ✗ |
| baseline | Out-domain | 90 | 1.60 | 0.4490 | 0.0000 | ✗ |
| rus_r01 | In-domain | 90 | 1.48 | 0.4764 | 0.0000 | ✗ |
| rus_r01 | Out-domain | 90 | 0.90 | 0.6375 | 0.0000 | ✗ |
| rus_r05 | In-domain | 90 | 2.98 | 0.2253 | 0.0113 | ✗ |
| rus_r05 | Out-domain | 90 | 2.33 | 0.3121 | 0.0038 | ✗ |
| smote_r01 | In-domain | 90 | 1.00 | 0.6058 | 0.0000 | ✗ |
| smote_r01 | Out-domain | 90 | 1.17 | 0.5581 | 0.0000 | ✗ |
| smote_r05 | In-domain | 90 | 0.62 | 0.7338 | 0.0000 | ✗ |
| smote_r05 | Out-domain | 90 | 0.26 | 0.8768 | 0.0000 | ✗ |
| sw_smote_r01 | In-domain | 90 | 2.15 | 0.3421 | 0.0017 | ✗ |
| sw_smote_r01 | Out-domain | 89 | 0.76 | 0.6856 | 0.0000 | ✗ |
| sw_smote_r05 | In-domain | 89 | 1.03 | 0.5972 | 0.0000 | ✗ |
| sw_smote_r05 | Out-domain | 90 | 1.41 | 0.4949 | 0.0000 | ✗ |

### 6. Distance effect by Condition × Mode (pooling levels)

| Condition | Mode | N | H | p | η² | Sig? |
|-----------|------|--:|--:|--:|---:|:----:|
| baseline | Cross-domain | 60 | 1.50 | 0.4726 | 0.0000 | ✗ |
| baseline | Within-domain | 60 | 19.06 | 0.0001 | 0.2994 | ✓ |
| baseline | Mixed | 60 | 0.04 | 0.9825 | 0.0000 | ✗ |
| rus_r01 | Cross-domain | 60 | 1.48 | 0.4767 | 0.0000 | ✗ |
| rus_r01 | Within-domain | 60 | 0.54 | 0.7641 | 0.0000 | ✗ |
| rus_r01 | Mixed | 60 | 0.05 | 0.9775 | 0.0000 | ✗ |
| rus_r05 | Cross-domain | 60 | 0.87 | 0.6473 | 0.0000 | ✗ |
| rus_r05 | Within-domain | 60 | 4.26 | 0.1188 | 0.0397 | ✗ |
| rus_r05 | Mixed | 60 | 0.47 | 0.7893 | 0.0000 | ✗ |
| smote_r01 | Cross-domain | 60 | 1.13 | 0.5672 | 0.0000 | ✗ |
| smote_r01 | Within-domain | 60 | 0.36 | 0.8365 | 0.0000 | ✗ |
| smote_r01 | Mixed | 60 | 0.13 | 0.9367 | 0.0000 | ✗ |
| smote_r05 | Cross-domain | 60 | 6.43 | 0.0401 | 0.0778 | ✓ |
| smote_r05 | Within-domain | 60 | 0.33 | 0.8489 | 0.0000 | ✗ |
| smote_r05 | Mixed | 60 | 0.04 | 0.9818 | 0.0000 | ✗ |
| sw_smote_r01 | Cross-domain | 60 | 4.29 | 0.1173 | 0.0401 | ✗ |
| sw_smote_r01 | Within-domain | 60 | 3.74 | 0.1542 | 0.0305 | ✗ |
| sw_smote_r01 | Mixed | 59 | 0.06 | 0.9710 | 0.0000 | ✗ |
| sw_smote_r05 | Cross-domain | 60 | 5.11 | 0.0775 | 0.0546 | ✗ |
| sw_smote_r05 | Within-domain | 60 | 0.87 | 0.6473 | 0.0000 | ✗ |
| sw_smote_r05 | Mixed | 59 | 0.03 | 0.9863 | 0.0000 | ✗ |


## Recall

### 1. Distance effect by Mode (pooling conditions & levels)

| Mode | N | H | p | η² | Sig? |
|------|--:|--:|--:|---:|:----:|
| Cross-domain | 420 | 1.86 | 0.3954 | 0.0000 | ✗ |
| Within-domain | 420 | 4.23 | 0.1204 | 0.0054 | ✗ |
| Mixed | 418 | 0.18 | 0.9140 | 0.0000 | ✗ |

**Mean ± SD per mode × distance:**

| Mode | MMD | DTW | Wasserstein |
|------|----:|----:|------------:|
| Cross-domain | 0.3024±0.1740 | 0.2820±0.1664 | 0.3033±0.1824 |
| Within-domain | 0.6804±0.1933 | 0.6362±0.2122 | 0.6747±0.2209 |
| Mixed | 0.6586±0.1934 | 0.6612±0.1909 | 0.6464±0.2033 |

### 2. Distance effect by Condition (pooling modes & levels)

| Condition | N | H | p | η² | Sig? |
|-----------|--:|--:|--:|---:|:----:|
| baseline | 180 | 11.05 | 0.0040 | 0.0511 | ✓ |
| rus_r01 | 180 | 3.56 | 0.1688 | 0.0088 | ✗ |
| rus_r05 | 180 | 2.32 | 0.3134 | 0.0018 | ✗ |
| smote_r01 | 180 | 0.39 | 0.8221 | 0.0000 | ✗ |
| smote_r05 | 180 | 0.63 | 0.7284 | 0.0000 | ✗ |
| sw_smote_r01 | 179 | 1.63 | 0.4418 | 0.0000 | ✗ |
| sw_smote_r05 | 179 | 0.45 | 0.8002 | 0.0000 | ✗ |

### 3. Distance effect by Level (pooling modes & conditions)

| Level | N | H | p | η² | Sig? |
|-------|--:|--:|--:|---:|:----:|
| In-domain | 629 | 0.02 | 0.9902 | 0.0000 | ✗ |
| Out-domain | 629 | 2.63 | 0.2690 | 0.0010 | ✗ |

### 4. Distance effect by Mode × Level (pooling conditions)

| Mode | Level | N | H | p | η² | Sig? |
|------|-------|--:|--:|--:|---:|:----:|
| Cross-domain | In-domain | 210 | 1.32 | 0.5163 | 0.0000 | ✗ |
| Cross-domain | Out-domain | 210 | 2.70 | 0.2592 | 0.0034 | ✗ |
| Within-domain | In-domain | 210 | 1.48 | 0.4770 | 0.0000 | ✗ |
| Within-domain | Out-domain | 210 | 5.21 | 0.0738 | 0.0155 | ✗ |
| Mixed | In-domain | 209 | 0.39 | 0.8236 | 0.0000 | ✗ |
| Mixed | Out-domain | 209 | 0.14 | 0.9324 | 0.0000 | ✗ |

### 5. Distance effect by Condition × Level (pooling modes)

| Condition | Level | N | H | p | η² | Sig? |
|-----------|-------|--:|--:|--:|---:|:----:|
| baseline | In-domain | 90 | 3.67 | 0.1598 | 0.0192 | ✗ |
| baseline | Out-domain | 90 | 8.13 | 0.0172 | 0.0704 | ✓ |
| rus_r01 | In-domain | 90 | 2.14 | 0.3426 | 0.0016 | ✗ |
| rus_r01 | Out-domain | 90 | 1.11 | 0.5752 | 0.0000 | ✗ |
| rus_r05 | In-domain | 90 | 9.13 | 0.0104 | 0.0819 | ✓ |
| rus_r05 | Out-domain | 90 | 26.40 | 0.0000 | 0.2804 | ✓ |
| smote_r01 | In-domain | 90 | 1.29 | 0.5235 | 0.0000 | ✗ |
| smote_r01 | Out-domain | 90 | 2.71 | 0.2580 | 0.0082 | ✗ |
| smote_r05 | In-domain | 90 | 1.61 | 0.4473 | 0.0000 | ✗ |
| smote_r05 | Out-domain | 90 | 0.03 | 0.9853 | 0.0000 | ✗ |
| sw_smote_r01 | In-domain | 90 | 1.15 | 0.5616 | 0.0000 | ✗ |
| sw_smote_r01 | Out-domain | 89 | 2.43 | 0.2964 | 0.0050 | ✗ |
| sw_smote_r05 | In-domain | 89 | 0.77 | 0.6810 | 0.0000 | ✗ |
| sw_smote_r05 | Out-domain | 90 | 0.20 | 0.9055 | 0.0000 | ✗ |

### 6. Distance effect by Condition × Mode (pooling levels)

| Condition | Mode | N | H | p | η² | Sig? |
|-----------|------|--:|--:|--:|---:|:----:|
| baseline | Cross-domain | 60 | 11.61 | 0.0030 | 0.1686 | ✓ |
| baseline | Within-domain | 60 | 32.74 | 0.0000 | 0.5392 | ✓ |
| baseline | Mixed | 60 | 1.54 | 0.4640 | 0.0000 | ✗ |
| rus_r01 | Cross-domain | 60 | 2.20 | 0.3335 | 0.0034 | ✗ |
| rus_r01 | Within-domain | 60 | 7.36 | 0.0252 | 0.0940 | ✓ |
| rus_r01 | Mixed | 60 | 3.60 | 0.1654 | 0.0280 | ✗ |
| rus_r05 | Cross-domain | 60 | 1.46 | 0.4828 | 0.0000 | ✗ |
| rus_r05 | Within-domain | 60 | 1.72 | 0.4235 | 0.0000 | ✗ |
| rus_r05 | Mixed | 60 | 5.16 | 0.0758 | 0.0554 | ✗ |
| smote_r01 | Cross-domain | 60 | 4.43 | 0.1093 | 0.0426 | ✗ |
| smote_r01 | Within-domain | 60 | 1.48 | 0.4769 | 0.0000 | ✗ |
| smote_r01 | Mixed | 60 | 0.92 | 0.6304 | 0.0000 | ✗ |
| smote_r05 | Cross-domain | 60 | 5.76 | 0.0561 | 0.0660 | ✗ |
| smote_r05 | Within-domain | 60 | 0.27 | 0.8729 | 0.0000 | ✗ |
| smote_r05 | Mixed | 60 | 1.29 | 0.5249 | 0.0000 | ✗ |
| sw_smote_r01 | Cross-domain | 60 | 3.73 | 0.1548 | 0.0304 | ✗ |
| sw_smote_r01 | Within-domain | 60 | 9.59 | 0.0083 | 0.1332 | ✓ |
| sw_smote_r01 | Mixed | 59 | 0.15 | 0.9258 | 0.0000 | ✗ |
| sw_smote_r05 | Cross-domain | 60 | 7.21 | 0.0272 | 0.0914 | ✓ |
| sw_smote_r05 | Within-domain | 60 | 1.31 | 0.5193 | 0.0000 | ✗ |
| sw_smote_r05 | Mixed | 59 | 0.05 | 0.9731 | 0.0000 | ✗ |


---
## Summary

The above tables show Kruskal-Wallis tests for distance metric effect at each level of stratification. Significance is assessed at α=0.05 (uncorrected per-test). For multiple testing across cells within a section, apply Bonferroni correction as needed.
