# Experiment 2 — Domain Shift Statistical Analysis Report

**Generated**: auto  
**Records**: 719  
**Seeds**: [0, 1, 7, 13, 42, 123, 256, 512, 1337, 2024] (n=10)  
**Conditions**: ['baseline', 'rus', 'smote', 'sw_smote']  
**Ratio**: 0.1 (for rus/smote/sw_smote; n/a for baseline)  

## 1. Experimental Design

### 1.1 Factor Structure

This experiment uses a **4-factor factorial design**:

| Factor | Levels | Description |
|--------|--------|-------------|
| Condition $C$ | baseline, rus, smote, sw_smote | Imbalance handling method |
| Mode $M$ | cross-domain, within-domain, mixed | Training data composition |
| Distance $D$ | MMD, DTW, Wasserstein | Domain distance metric for grouping |
| Level $L$ | in-domain, out-domain | Target domain proximity |

Each observation is the evaluation metric (F2 or AUROC) for a specific
combination $(C, M, D, L, s)$ where $s$ is the random seed.

### 1.2 Statistical Model

The observed metric for configuration $(c, m, d, l, s)$:

$$Y_{cmdls} = \mu + \alpha_c + \beta_m + \gamma_d + \delta_l + (\alpha\beta)_{cm} + (\alpha\gamma)_{cd} + (\alpha\delta)_{cl} + \varepsilon_{cmdls}$$

where:
- $\mu$: grand mean
- $\alpha_c$: main effect of condition
- $\beta_m$: main effect of mode
- $\gamma_d$: main effect of distance metric
- $\delta_l$: main effect of domain level
- $(\alpha\beta)_{cm}$, etc.: two-way interaction terms
- $\varepsilon_{cmdls} \sim \mathcal{N}(0, \sigma^2)$: residual error

Due to non-normality of classification metrics, we employ **non-parametric** tests.

---
## 2. Analysis: F2-score

### 2.1 Descriptive Statistics

#### Mode: Cross-domain

| Condition | In-domain (mean±SD) | Out-domain (mean±SD) | Δ (out−in) |
|-----------|--------------------:|---------------------:|-----------:|
| baseline | 0.1637±0.0061 | 0.1572±0.0118 | -0.0065 |
| rus | 0.1750±0.0118 | 0.1560±0.0144 | -0.0190 |
| smote | 0.1344±0.0173 | 0.1376±0.0129 | +0.0032 |
| sw_smote | 0.1035±0.0140 | 0.1014±0.0134 | -0.0021 |

#### Mode: Within-domain

| Condition | In-domain (mean±SD) | Out-domain (mean±SD) | Δ (out−in) |
|-----------|--------------------:|---------------------:|-----------:|
| baseline | 0.2054±0.0345 | 0.2233±0.0528 | +0.0179 |
| rus | 0.1461±0.0486 | 0.2096±0.0432 | +0.0635 |
| smote | 0.4621±0.0663 | 0.4554±0.0674 | -0.0068 |
| sw_smote | 0.5541±0.0962 | 0.5582±0.1074 | +0.0041 |

#### Mode: Mixed

| Condition | In-domain (mean±SD) | Out-domain (mean±SD) | Δ (out−in) |
|-----------|--------------------:|---------------------:|-----------:|
| baseline | 0.2394±0.0221 | 0.3084±0.0299 | +0.0690 |
| rus | 0.1951±0.0551 | 0.2227±0.0428 | +0.0276 |
| smote | 0.4277±0.0778 | 0.4961±0.0494 | +0.0684 |
| sw_smote | 0.5335±0.1137 | 0.6114±0.1646 | +0.0779 |

### 2.2 Kruskal-Wallis H-test (Condition effect)

Tests whether the distribution of F2-score differs across the 4 conditions.

$$H = \frac{12}{N(N+1)} \sum_{i=1}^{k} \frac{R_i^2}{n_i} - 3(N+1)$$

where $R_i$ is the sum of ranks in group $i$, $n_i$ the group size, $N = \sum n_i$.

| Mode | Level | Distance | H | p-value | Sig (α=0.05) |
|------|-------|----------|--:|--------:|:------------:|
| Cross-domain | In-domain | MMD | 35.763 | 0.0000 | ✓ |
| Cross-domain | In-domain | DTW | 31.873 | 0.0000 | ✓ |
| Cross-domain | In-domain | WASSERSTEIN | 34.433 | 0.0000 | ✓ |
| Cross-domain | Out-domain | MMD | 28.068 | 0.0000 | ✓ |
| Cross-domain | Out-domain | DTW | 30.563 | 0.0000 | ✓ |
| Cross-domain | Out-domain | WASSERSTEIN | 29.399 | 0.0000 | ✓ |
| Within-domain | In-domain | MMD | 29.150 | 0.0000 | ✓ |
| Within-domain | In-domain | DTW | 33.203 | 0.0000 | ✓ |
| Within-domain | In-domain | WASSERSTEIN | 36.299 | 0.0000 | ✓ |
| Within-domain | Out-domain | MMD | 31.001 | 0.0000 | ✓ |
| Within-domain | Out-domain | DTW | 30.119 | 0.0000 | ✓ |
| Within-domain | Out-domain | WASSERSTEIN | 34.398 | 0.0000 | ✓ |
| Mixed | In-domain | MMD | 32.141 | 0.0000 | ✓ |
| Mixed | In-domain | DTW | 30.666 | 0.0000 | ✓ |
| Mixed | In-domain | WASSERSTEIN | 32.009 | 0.0000 | ✓ |
| Mixed | Out-domain | MMD | 32.431 | 0.0000 | ✓ |
| Mixed | Out-domain | DTW | 32.178 | 0.0000 | ✓ |
| Mixed | Out-domain | WASSERSTEIN | 31.588 | 0.0000 | ✓ |

**Summary**: 18/18 cells significant at α=0.05; 18/18 after Bonferroni correction (α'=0.0028).

### 2.3 Pairwise Comparisons (Mann-Whitney U)

Baseline vs each method, testing:

$$H_0: F_{\text{baseline}}(x) = F_{\text{method}}(x)$$
$$H_1: F_{\text{baseline}}(x) \neq F_{\text{method}}(x)$$

| Comparison | Mode | Level | Distance | U | p | Cliff's δ | Effect | Mean(method) | Mean(baseline) |
|------------|------|-------|----------|--:|--:|----------:|:------:|-------------:|---------------:|
| rus vs baseline | Cross-domain | In-domain | MMD | 6.0 | 0.0010 | +0.880 | large | 0.1704 | 0.1625 |
| smote vs baseline | Cross-domain | In-domain | MMD | 100.0 | 0.0002 * | -1.000 | large | 0.1523 | 0.1625 |
| sw_smote vs baseline | Cross-domain | In-domain | MMD | 100.0 | 0.0002 * | -1.000 | large | 0.1060 | 0.1625 |
| rus vs baseline | Cross-domain | In-domain | DTW | 46.0 | 0.7913 | +0.080 | negligible | 0.1700 | 0.1686 |
| smote vs baseline | Cross-domain | In-domain | DTW | 100.0 | 0.0002 * | -1.000 | large | 0.1281 | 0.1686 |
| sw_smote vs baseline | Cross-domain | In-domain | DTW | 100.0 | 0.0002 * | -1.000 | large | 0.0973 | 0.1686 |
| rus vs baseline | Cross-domain | In-domain | WASSERSTEIN | 2.0 | 0.0003 * | +0.960 | large | 0.1845 | 0.1600 |
| smote vs baseline | Cross-domain | In-domain | WASSERSTEIN | 100.0 | 0.0002 * | -1.000 | large | 0.1227 | 0.1600 |
| sw_smote vs baseline | Cross-domain | In-domain | WASSERSTEIN | 100.0 | 0.0002 * | -1.000 | large | 0.1071 | 0.1600 |
| rus vs baseline | Cross-domain | Out-domain | MMD | 73.0 | 0.0890 | -0.460 | medium | 0.1557 | 0.1670 |
| smote vs baseline | Cross-domain | Out-domain | MMD | 95.0 | 0.0008 * | -0.900 | large | 0.1408 | 0.1670 |
| sw_smote vs baseline | Cross-domain | Out-domain | MMD | 100.0 | 0.0002 * | -1.000 | large | 0.1110 | 0.1670 |
| rus vs baseline | Cross-domain | Out-domain | DTW | 14.0 | 0.0073 | +0.720 | large | 0.1543 | 0.1439 |
| smote vs baseline | Cross-domain | Out-domain | DTW | 85.0 | 0.0091 | -0.700 | large | 0.1313 | 0.1439 |
| sw_smote vs baseline | Cross-domain | Out-domain | DTW | 100.0 | 0.0002 * | -1.000 | large | 0.1000 | 0.1439 |
| rus vs baseline | Cross-domain | Out-domain | WASSERSTEIN | 69.0 | 0.1620 | -0.380 | medium | 0.1579 | 0.1607 |
| smote vs baseline | Cross-domain | Out-domain | WASSERSTEIN | 98.0 | 0.0003 * | -0.960 | large | 0.1405 | 0.1607 |
| sw_smote vs baseline | Cross-domain | Out-domain | WASSERSTEIN | 100.0 | 0.0002 * | -1.000 | large | 0.0933 | 0.1607 |
| rus vs baseline | Within-domain | In-domain | MMD | 60.0 | 0.4727 | -0.200 | small | 0.1973 | 0.2272 |
| smote vs baseline | Within-domain | In-domain | MMD | 0.0 | 0.0002 * | +1.000 | large | 0.4858 | 0.2272 |
| sw_smote vs baseline | Within-domain | In-domain | MMD | 2.0 | 0.0003 * | +0.960 | large | 0.5242 | 0.2272 |
| rus vs baseline | Within-domain | In-domain | DTW | 90.0 | 0.0028 | -0.800 | large | 0.1366 | 0.1720 |
| smote vs baseline | Within-domain | In-domain | DTW | 0.0 | 0.0002 * | +1.000 | large | 0.4637 | 0.1720 |
| sw_smote vs baseline | Within-domain | In-domain | DTW | 0.0 | 0.0002 * | +1.000 | large | 0.5501 | 0.1720 |
| rus vs baseline | Within-domain | In-domain | WASSERSTEIN | 100.0 | 0.0002 * | -1.000 | large | 0.1044 | 0.2169 |
| smote vs baseline | Within-domain | In-domain | WASSERSTEIN | 0.0 | 0.0002 * | +1.000 | large | 0.4370 | 0.2169 |
| sw_smote vs baseline | Within-domain | In-domain | WASSERSTEIN | 0.0 | 0.0002 * | +1.000 | large | 0.5879 | 0.2169 |
| rus vs baseline | Within-domain | Out-domain | MMD | 30.0 | 0.1405 | +0.400 | medium | 0.1935 | 0.1909 |
| smote vs baseline | Within-domain | Out-domain | MMD | 0.0 | 0.0002 * | +1.000 | large | 0.4405 | 0.1909 |
| sw_smote vs baseline | Within-domain | Out-domain | MMD | 0.0 | 0.0002 * | +1.000 | large | 0.5242 | 0.1909 |
| rus vs baseline | Within-domain | Out-domain | DTW | 30.0 | 0.1405 | +0.400 | medium | 0.2093 | 0.1939 |
| smote vs baseline | Within-domain | Out-domain | DTW | 0.0 | 0.0002 * | +1.000 | large | 0.4508 | 0.1939 |
| sw_smote vs baseline | Within-domain | Out-domain | DTW | 1.0 | 0.0002 * | +0.980 | large | 0.5424 | 0.1939 |
| rus vs baseline | Within-domain | Out-domain | WASSERSTEIN | 86.0 | 0.0073 | -0.720 | large | 0.2259 | 0.2851 |
| smote vs baseline | Within-domain | Out-domain | WASSERSTEIN | 0.0 | 0.0002 * | +1.000 | large | 0.4748 | 0.2851 |
| sw_smote vs baseline | Within-domain | Out-domain | WASSERSTEIN | 0.0 | 0.0002 * | +1.000 | large | 0.6078 | 0.2851 |
| rus vs baseline | Mixed | In-domain | MMD | 88.0 | 0.0046 | -0.760 | large | 0.1844 | 0.2264 |
| smote vs baseline | Mixed | In-domain | MMD | 0.0 | 0.0002 * | +1.000 | large | 0.4449 | 0.2264 |
| sw_smote vs baseline | Mixed | In-domain | MMD | 0.0 | 0.0002 * | +1.000 | large | 0.5517 | 0.2264 |
| rus vs baseline | Mixed | In-domain | DTW | 69.0 | 0.1620 | -0.380 | medium | 0.2231 | 0.2506 |
| smote vs baseline | Mixed | In-domain | DTW | 0.0 | 0.0002 * | +1.000 | large | 0.4231 | 0.2506 |
| sw_smote vs baseline | Mixed | In-domain | DTW | 0.0 | 0.0002 * | +1.000 | large | 0.5271 | 0.2506 |
| rus vs baseline | Mixed | In-domain | WASSERSTEIN | 86.0 | 0.0073 | -0.720 | large | 0.1777 | 0.2412 |
| smote vs baseline | Mixed | In-domain | WASSERSTEIN | 0.0 | 0.0002 * | +1.000 | large | 0.4152 | 0.2412 |
| sw_smote vs baseline | Mixed | In-domain | WASSERSTEIN | 0.0 | 0.0002 * | +1.000 | large | 0.5217 | 0.2412 |
| rus vs baseline | Mixed | Out-domain | MMD | 94.0 | 0.0010 | -0.880 | large | 0.2194 | 0.2938 |
| smote vs baseline | Mixed | Out-domain | MMD | 0.0 | 0.0002 * | +1.000 | large | 0.4693 | 0.2938 |
| sw_smote vs baseline | Mixed | Out-domain | MMD | 1.0 | 0.0002 * | +0.980 | large | 0.5885 | 0.2938 |
| rus vs baseline | Mixed | Out-domain | DTW | 94.0 | 0.0010 | -0.880 | large | 0.2183 | 0.3056 |
| smote vs baseline | Mixed | Out-domain | DTW | 0.0 | 0.0002 * | +1.000 | large | 0.5003 | 0.3056 |
| sw_smote vs baseline | Mixed | Out-domain | DTW | 2.0 | 0.0003 * | +0.960 | large | 0.6223 | 0.3056 |
| rus vs baseline | Mixed | Out-domain | WASSERSTEIN | 94.0 | 0.0010 | -0.880 | large | 0.2304 | 0.3258 |
| smote vs baseline | Mixed | Out-domain | WASSERSTEIN | 0.0 | 0.0002 * | +1.000 | large | 0.5188 | 0.3258 |
| sw_smote vs baseline | Mixed | Out-domain | WASSERSTEIN | 1.0 | 0.0004 * | +0.978 | large | 0.6248 | 0.3258 |

**Bonferroni threshold**: α'=0.00093 (m=54). **37** comparisons significant after correction.

### 2.4 Paired Comparison (Wilcoxon Signed-Rank)

Paired by seed — more powerful than Mann-Whitney when seeds are shared.

$$W = \sum_{i=1}^{n} \text{sign}(d_i) \cdot R_i, \quad d_i = Y_{\text{method},i} - Y_{\text{baseline},i}$$

| Comparison | Mode | Level | Distance | W | p | Cliff's δ | Effect | n |
|------------|------|-------|----------|--:|--:|----------:|:------:|--:|
| rus vs baseline | Cross-domain | In-domain | MMD | 1.0 | 0.0039 | +0.880 | large | 10 |
| smote vs baseline | Cross-domain | In-domain | MMD | 0.0 | 0.0020 | -1.000 | large | 10 |
| sw_smote vs baseline | Cross-domain | In-domain | MMD | 0.0 | 0.0020 | -1.000 | large | 10 |
| rus vs baseline | Cross-domain | In-domain | DTW | 24.0 | 0.7695 | +0.080 | negligible | 10 |
| smote vs baseline | Cross-domain | In-domain | DTW | 0.0 | 0.0020 | -1.000 | large | 10 |
| sw_smote vs baseline | Cross-domain | In-domain | DTW | 0.0 | 0.0020 | -1.000 | large | 10 |
| rus vs baseline | Cross-domain | In-domain | WASSERSTEIN | 0.0 | 0.0020 | +0.960 | large | 10 |
| smote vs baseline | Cross-domain | In-domain | WASSERSTEIN | 0.0 | 0.0020 | -1.000 | large | 10 |
| sw_smote vs baseline | Cross-domain | In-domain | WASSERSTEIN | 0.0 | 0.0020 | -1.000 | large | 10 |
| rus vs baseline | Cross-domain | Out-domain | MMD | 9.0 | 0.0645 | -0.460 | medium | 10 |
| smote vs baseline | Cross-domain | Out-domain | MMD | 0.0 | 0.0020 | -0.900 | large | 10 |
| sw_smote vs baseline | Cross-domain | Out-domain | MMD | 0.0 | 0.0020 | -1.000 | large | 10 |
| rus vs baseline | Cross-domain | Out-domain | DTW | 8.0 | 0.0488 | +0.720 | large | 10 |
| smote vs baseline | Cross-domain | Out-domain | DTW | 9.0 | 0.0645 | -0.700 | large | 10 |
| sw_smote vs baseline | Cross-domain | Out-domain | DTW | 0.0 | 0.0020 | -1.000 | large | 10 |
| rus vs baseline | Cross-domain | Out-domain | WASSERSTEIN | 23.0 | 0.6953 | -0.380 | medium | 10 |
| smote vs baseline | Cross-domain | Out-domain | WASSERSTEIN | 0.0 | 0.0020 | -0.960 | large | 10 |
| sw_smote vs baseline | Cross-domain | Out-domain | WASSERSTEIN | 0.0 | 0.0020 | -1.000 | large | 10 |
| rus vs baseline | Within-domain | In-domain | MMD | 11.0 | 0.1055 | -0.200 | small | 10 |
| smote vs baseline | Within-domain | In-domain | MMD | 0.0 | 0.0020 | +1.000 | large | 10 |
| sw_smote vs baseline | Within-domain | In-domain | MMD | 0.0 | 0.0020 | +0.960 | large | 10 |
| rus vs baseline | Within-domain | In-domain | DTW | 4.0 | 0.0137 | -0.800 | large | 10 |
| smote vs baseline | Within-domain | In-domain | DTW | 0.0 | 0.0020 | +1.000 | large | 10 |
| sw_smote vs baseline | Within-domain | In-domain | DTW | 0.0 | 0.0020 | +1.000 | large | 10 |
| rus vs baseline | Within-domain | In-domain | WASSERSTEIN | 0.0 | 0.0020 | -1.000 | large | 10 |
| smote vs baseline | Within-domain | In-domain | WASSERSTEIN | 0.0 | 0.0020 | +1.000 | large | 10 |
| sw_smote vs baseline | Within-domain | In-domain | WASSERSTEIN | 0.0 | 0.0020 | +1.000 | large | 10 |
| rus vs baseline | Within-domain | Out-domain | MMD | 15.0 | 0.2324 | +0.400 | medium | 10 |
| smote vs baseline | Within-domain | Out-domain | MMD | 0.0 | 0.0020 | +1.000 | large | 10 |
| sw_smote vs baseline | Within-domain | Out-domain | MMD | 0.0 | 0.0020 | +1.000 | large | 10 |
| rus vs baseline | Within-domain | Out-domain | DTW | 19.0 | 0.4316 | +0.400 | medium | 10 |
| smote vs baseline | Within-domain | Out-domain | DTW | 0.0 | 0.0020 | +1.000 | large | 10 |
| sw_smote vs baseline | Within-domain | Out-domain | DTW | 0.0 | 0.0020 | +0.980 | large | 10 |
| rus vs baseline | Within-domain | Out-domain | WASSERSTEIN | 0.0 | 0.0020 | -0.720 | large | 10 |
| smote vs baseline | Within-domain | Out-domain | WASSERSTEIN | 0.0 | 0.0020 | +1.000 | large | 10 |
| sw_smote vs baseline | Within-domain | Out-domain | WASSERSTEIN | 0.0 | 0.0020 | +1.000 | large | 10 |
| rus vs baseline | Mixed | In-domain | MMD | 11.0 | 0.1055 | -0.760 | large | 10 |
| smote vs baseline | Mixed | In-domain | MMD | 0.0 | 0.0020 | +1.000 | large | 10 |
| sw_smote vs baseline | Mixed | In-domain | MMD | 0.0 | 0.0020 | +1.000 | large | 10 |
| rus vs baseline | Mixed | In-domain | DTW | 12.0 | 0.1309 | -0.380 | medium | 10 |
| smote vs baseline | Mixed | In-domain | DTW | 0.0 | 0.0020 | +1.000 | large | 10 |
| sw_smote vs baseline | Mixed | In-domain | DTW | 0.0 | 0.0020 | +1.000 | large | 10 |
| rus vs baseline | Mixed | In-domain | WASSERSTEIN | 4.0 | 0.0137 | -0.720 | large | 10 |
| smote vs baseline | Mixed | In-domain | WASSERSTEIN | 0.0 | 0.0020 | +1.000 | large | 10 |
| sw_smote vs baseline | Mixed | In-domain | WASSERSTEIN | 0.0 | 0.0020 | +1.000 | large | 10 |
| rus vs baseline | Mixed | Out-domain | MMD | 1.0 | 0.0039 | -0.880 | large | 10 |
| smote vs baseline | Mixed | Out-domain | MMD | 0.0 | 0.0020 | +1.000 | large | 10 |
| sw_smote vs baseline | Mixed | Out-domain | MMD | 0.0 | 0.0020 | +0.980 | large | 10 |
| rus vs baseline | Mixed | Out-domain | DTW | 1.0 | 0.0039 | -0.880 | large | 10 |
| smote vs baseline | Mixed | Out-domain | DTW | 0.0 | 0.0020 | +1.000 | large | 10 |
| sw_smote vs baseline | Mixed | Out-domain | DTW | 0.0 | 0.0020 | +0.960 | large | 10 |
| rus vs baseline | Mixed | Out-domain | WASSERSTEIN | 1.0 | 0.0039 | -0.880 | large | 10 |
| smote vs baseline | Mixed | Out-domain | WASSERSTEIN | 0.0 | 0.0020 | +1.000 | large | 10 |
| sw_smote vs baseline | Mixed | Out-domain | WASSERSTEIN | 0.0 | 0.0039 | +0.975 | large | 9 |

**Bonferroni threshold**: α'=0.00093 (m=54). **0** comparisons significant after correction.

**Note**: With n=10 paired observations, the minimum achievable p-value for Wilcoxon signed-rank is $p_{\min} = 1/2^{10-1} = 0.001953$. When $p_{\min} > \alpha'$, no comparison can reach Bonferroni significance regardless of effect magnitude. This is a **floor effect** of small sample size, not evidence of no difference.

### 2.5 Friedman Test (Repeated-Measures)

Seeds serve as blocks (subjects). Tests whether at least one condition differs.

$$\chi_F^2 = \frac{12}{bk(k+1)} \sum_{j=1}^{k} R_j^2 - 3b(k+1)$$

where $b$ = number of blocks (seeds), $k$ = number of conditions, $R_j$ = rank sum.

| Mode | Level | Distance | χ² | p-value | n_seeds | Sig |
|------|-------|----------|----|--------:|--------:|:---:|
| Cross-domain | In-domain | MMD | 28.920 | 0.0000 | 10 | ✓ |
| Cross-domain | In-domain | DTW | 27.000 | 0.0000 | 10 | ✓ |
| Cross-domain | In-domain | WASSERSTEIN | 28.080 | 0.0000 | 10 | ✓ |
| Cross-domain | Out-domain | MMD | 24.360 | 0.0000 | 10 | ✓ |
| Cross-domain | Out-domain | DTW | 21.720 | 0.0001 | 10 | ✓ |
| Cross-domain | Out-domain | WASSERSTEIN | 23.160 | 0.0000 | 10 | ✓ |
| Within-domain | In-domain | MMD | 24.600 | 0.0000 | 10 | ✓ |
| Within-domain | In-domain | DTW | 27.840 | 0.0000 | 10 | ✓ |
| Within-domain | In-domain | WASSERSTEIN | 30.000 | 0.0000 | 10 | ✓ |
| Within-domain | Out-domain | MMD | 25.560 | 0.0000 | 10 | ✓ |
| Within-domain | Out-domain | DTW | 25.200 | 0.0000 | 10 | ✓ |
| Within-domain | Out-domain | WASSERSTEIN | 30.000 | 0.0000 | 10 | ✓ |
| Mixed | In-domain | MMD | 26.160 | 0.0000 | 10 | ✓ |
| Mixed | In-domain | DTW | 25.560 | 0.0000 | 10 | ✓ |
| Mixed | In-domain | WASSERSTEIN | 27.000 | 0.0000 | 10 | ✓ |
| Mixed | Out-domain | MMD | 26.400 | 0.0000 | 10 | ✓ |
| Mixed | Out-domain | DTW | 26.400 | 0.0000 | 10 | ✓ |
| Mixed | Out-domain | WASSERSTEIN | 24.600 | 0.0000 | 9 | ✓ |

### 2.6 Domain Gap Analysis

Domain gap: $\Delta = Y_{\text{out-domain}} - Y_{\text{in-domain}}$
 (paired by seed).

Negative Δ indicates performance degrades in out-domain (expected for domain shift).

| Mode | Distance | Condition | Mean Δ | SD Δ | n | KW p (across conds) |
|------|----------|-----------|-------:|-----:|--:|--------------------:|
| Cross-domain | MMD | baseline | +0.0045 | 0.0075 | 10 | 0.0117 |
| Cross-domain | MMD | rus | -0.0147 | 0.0168 | 10 | 0.0117 |
| Cross-domain | MMD | smote | -0.0115 | 0.0151 | 10 | 0.0117 |
| Cross-domain | MMD | sw_smote | +0.0050 | 0.0162 | 10 | 0.0117 |
| Cross-domain | DTW | baseline | -0.0247 | 0.0085 | 10 | 0.0005 |
| Cross-domain | DTW | rus | -0.0157 | 0.0167 | 10 | 0.0005 |
| Cross-domain | DTW | smote | +0.0032 | 0.0131 | 10 | 0.0005 |
| Cross-domain | DTW | sw_smote | +0.0026 | 0.0200 | 10 | 0.0005 |
| Cross-domain | WASSERSTEIN | baseline | +0.0007 | 0.0056 | 10 | 0.0000 |
| Cross-domain | WASSERSTEIN | rus | -0.0266 | 0.0219 | 10 | 0.0000 |
| Cross-domain | WASSERSTEIN | smote | +0.0178 | 0.0103 | 10 | 0.0000 |
| Cross-domain | WASSERSTEIN | sw_smote | -0.0138 | 0.0181 | 10 | 0.0000 |
| Within-domain | MMD | baseline | -0.0363 | 0.0376 | 10 | 0.1574 |
| Within-domain | MMD | rus | -0.0038 | 0.0398 | 10 | 0.1574 |
| Within-domain | MMD | smote | -0.0453 | 0.0552 | 10 | 0.1574 |
| Within-domain | MMD | sw_smote | +0.0000 | 0.1997 | 10 | 0.1574 |
| Within-domain | DTW | baseline | +0.0219 | 0.0339 | 10 | 0.0213 |
| Within-domain | DTW | rus | +0.0728 | 0.0490 | 10 | 0.0213 |
| Within-domain | DTW | smote | -0.0129 | 0.0664 | 10 | 0.0213 |
| Within-domain | DTW | sw_smote | -0.0077 | 0.1370 | 10 | 0.0213 |
| Within-domain | WASSERSTEIN | baseline | +0.0681 | 0.0484 | 10 | 0.0074 |
| Within-domain | WASSERSTEIN | rus | +0.1215 | 0.0585 | 10 | 0.0074 |
| Within-domain | WASSERSTEIN | smote | +0.0379 | 0.0469 | 10 | 0.0074 |
| Within-domain | WASSERSTEIN | sw_smote | +0.0199 | 0.0686 | 10 | 0.0074 |
| Mixed | MMD | baseline | +0.0674 | 0.0371 | 10 | 0.2164 |
| Mixed | MMD | rus | +0.0351 | 0.0665 | 10 | 0.2164 |
| Mixed | MMD | smote | +0.0244 | 0.0747 | 10 | 0.2164 |
| Mixed | MMD | sw_smote | +0.0368 | 0.2093 | 10 | 0.2164 |
| Mixed | DTW | baseline | +0.0550 | 0.0396 | 10 | 0.0108 |
| Mixed | DTW | rus | -0.0049 | 0.0674 | 10 | 0.0108 |
| Mixed | DTW | smote | +0.0772 | 0.0672 | 10 | 0.0108 |
| Mixed | DTW | sw_smote | +0.0952 | 0.2130 | 10 | 0.0108 |
| Mixed | WASSERSTEIN | baseline | +0.0846 | 0.0360 | 10 | 0.1674 |
| Mixed | WASSERSTEIN | rus | +0.0527 | 0.0805 | 10 | 0.1674 |
| Mixed | WASSERSTEIN | smote | +0.1037 | 0.0701 | 10 | 0.1674 |
| Mixed | WASSERSTEIN | sw_smote | +0.1098 | 0.2265 | 9 | 0.1674 |

### 2.7 Condition × Distance Interaction

Does the best-performing condition depend on which distance metric is used?

| Mode | Level | Distance | Best Condition | Mean F2-score | Consistent? |
|------|-------|----------|----------------|---:|:-----------:|
| Cross-domain | In-domain | MMD | rus | 0.1704 | ✓ |
| Cross-domain | In-domain | DTW | rus | 0.1700 | ✓ |
| Cross-domain | In-domain | WASSERSTEIN | rus | 0.1845 | ✓ |
| Cross-domain | Out-domain | MMD | baseline | 0.1670 | ✗ |
| Cross-domain | Out-domain | DTW | rus | 0.1543 | ✗ |
| Cross-domain | Out-domain | WASSERSTEIN | baseline | 0.1607 | ✗ |
| Within-domain | In-domain | MMD | sw_smote | 0.5242 | ✓ |
| Within-domain | In-domain | DTW | sw_smote | 0.5501 | ✓ |
| Within-domain | In-domain | WASSERSTEIN | sw_smote | 0.5879 | ✓ |
| Within-domain | Out-domain | MMD | sw_smote | 0.5242 | ✓ |
| Within-domain | Out-domain | DTW | sw_smote | 0.5424 | ✓ |
| Within-domain | Out-domain | WASSERSTEIN | sw_smote | 0.6078 | ✓ |
| Mixed | In-domain | MMD | sw_smote | 0.5517 | ✓ |
| Mixed | In-domain | DTW | sw_smote | 0.5271 | ✓ |
| Mixed | In-domain | WASSERSTEIN | sw_smote | 0.5217 | ✓ |
| Mixed | Out-domain | MMD | sw_smote | 0.5885 | ✓ |
| Mixed | Out-domain | DTW | sw_smote | 0.6223 | ✓ |
| Mixed | Out-domain | WASSERSTEIN | sw_smote | 0.6248 | ✓ |

### 2.8 Overall Condition Ranking

Mean rank across all 18 cells (3 modes × 2 levels × 3 distances). Rank 1 = best.

| Rank | Condition | Mean Rank |
|:----:|-----------|----------:|
| 1 | sw_smote | 2.00 |
| 2 | smote | 2.33 |
| 3 | baseline | 2.67 |
| 4 | rus | 3.00 |

---
## 3. Analysis: AUROC

### 3.1 Descriptive Statistics

#### Mode: Cross-domain

| Condition | In-domain (mean±SD) | Out-domain (mean±SD) | Δ (out−in) |
|-----------|--------------------:|---------------------:|-----------:|
| baseline | 0.5232±0.0053 | 0.5195±0.0131 | -0.0037 |
| rus | 0.5246±0.0244 | 0.5264±0.0202 | +0.0018 |
| smote | 0.5180±0.0059 | 0.5234±0.0084 | +0.0054 |
| sw_smote | 0.5172±0.0112 | 0.5123±0.0092 | -0.0049 |

#### Mode: Within-domain

| Condition | In-domain (mean±SD) | Out-domain (mean±SD) | Δ (out−in) |
|-----------|--------------------:|---------------------:|-----------:|
| baseline | 0.5933±0.0675 | 0.6680±0.1167 | +0.0747 |
| rus | 0.5997±0.0679 | 0.6515±0.0910 | +0.0518 |
| smote | 0.9015±0.0292 | 0.9054±0.0244 | +0.0038 |
| sw_smote | 0.8940±0.0538 | 0.9063±0.0564 | +0.0123 |

#### Mode: Mixed

| Condition | In-domain (mean±SD) | Out-domain (mean±SD) | Δ (out−in) |
|-----------|--------------------:|---------------------:|-----------:|
| baseline | 0.6600±0.0619 | 0.8400±0.0632 | +0.1800 |
| rus | 0.5990±0.1151 | 0.6607±0.0865 | +0.0618 |
| smote | 0.8481±0.0466 | 0.9117±0.0165 | +0.0635 |
| sw_smote | 0.8550±0.0515 | 0.8890±0.0620 | +0.0340 |

### 3.2 Kruskal-Wallis H-test (Condition effect)

Tests whether the distribution of AUROC differs across the 4 conditions.

$$H = \frac{12}{N(N+1)} \sum_{i=1}^{k} \frac{R_i^2}{n_i} - 3(N+1)$$

where $R_i$ is the sum of ranks in group $i$, $n_i$ the group size, $N = \sum n_i$.

| Mode | Level | Distance | H | p-value | Sig (α=0.05) |
|------|-------|----------|--:|--------:|:------------:|
| Cross-domain | In-domain | MMD | 24.843 | 0.0000 | ✓ |
| Cross-domain | In-domain | DTW | 12.544 | 0.0057 | ✓ |
| Cross-domain | In-domain | WASSERSTEIN | 7.117 | 0.0683 |  |
| Cross-domain | Out-domain | MMD | 7.847 | 0.0493 | ✓ |
| Cross-domain | Out-domain | DTW | 16.886 | 0.0007 | ✓ |
| Cross-domain | Out-domain | WASSERSTEIN | 6.798 | 0.0786 |  |
| Within-domain | In-domain | MMD | 28.036 | 0.0000 | ✓ |
| Within-domain | In-domain | DTW | 30.505 | 0.0000 | ✓ |
| Within-domain | In-domain | WASSERSTEIN | 29.366 | 0.0000 | ✓ |
| Within-domain | Out-domain | MMD | 29.508 | 0.0000 | ✓ |
| Within-domain | Out-domain | DTW | 28.841 | 0.0000 | ✓ |
| Within-domain | Out-domain | WASSERSTEIN | 31.787 | 0.0000 | ✓ |
| Mixed | In-domain | MMD | 28.778 | 0.0000 | ✓ |
| Mixed | In-domain | DTW | 23.975 | 0.0000 | ✓ |
| Mixed | In-domain | WASSERSTEIN | 28.332 | 0.0000 | ✓ |
| Mixed | Out-domain | MMD | 23.258 | 0.0000 | ✓ |
| Mixed | Out-domain | DTW | 22.365 | 0.0001 | ✓ |
| Mixed | Out-domain | WASSERSTEIN | 22.861 | 0.0000 | ✓ |

**Summary**: 16/18 cells significant at α=0.05; 14/18 after Bonferroni correction (α'=0.0028).

### 3.3 Pairwise Comparisons (Mann-Whitney U)

Baseline vs each method, testing:

$$H_0: F_{\text{baseline}}(x) = F_{\text{method}}(x)$$
$$H_1: F_{\text{baseline}}(x) \neq F_{\text{method}}(x)$$

| Comparison | Mode | Level | Distance | U | p | Cliff's δ | Effect | Mean(method) | Mean(baseline) |
|------------|------|-------|----------|--:|--:|----------:|:------:|-------------:|---------------:|
| rus vs baseline | Cross-domain | In-domain | MMD | 100.0 | 0.0002 * | -1.000 | large | 0.5055 | 0.5228 |
| smote vs baseline | Cross-domain | In-domain | MMD | 70.0 | 0.1405 | -0.400 | medium | 0.5203 | 0.5228 |
| sw_smote vs baseline | Cross-domain | In-domain | MMD | 84.0 | 0.0113 | -0.680 | large | 0.5132 | 0.5228 |
| rus vs baseline | Cross-domain | In-domain | DTW | 23.0 | 0.0452 | +0.540 | large | 0.5242 | 0.5186 |
| smote vs baseline | Cross-domain | In-domain | DTW | 70.0 | 0.1405 | -0.400 | medium | 0.5155 | 0.5186 |
| sw_smote vs baseline | Cross-domain | In-domain | DTW | 77.0 | 0.0452 | -0.540 | large | 0.5112 | 0.5186 |
| rus vs baseline | Cross-domain | In-domain | WASSERSTEIN | 41.0 | 0.5205 | +0.180 | small | 0.5441 | 0.5282 |
| smote vs baseline | Cross-domain | In-domain | WASSERSTEIN | 91.0 | 0.0022 | -0.820 | large | 0.5182 | 0.5282 |
| sw_smote vs baseline | Cross-domain | In-domain | WASSERSTEIN | 52.0 | 0.9097 | -0.040 | negligible | 0.5272 | 0.5282 |
| rus vs baseline | Cross-domain | Out-domain | MMD | 56.0 | 0.6776 | -0.120 | negligible | 0.5350 | 0.5344 |
| smote vs baseline | Cross-domain | Out-domain | MMD | 61.0 | 0.4274 | -0.220 | small | 0.5327 | 0.5344 |
| sw_smote vs baseline | Cross-domain | Out-domain | MMD | 82.0 | 0.0173 | -0.640 | large | 0.5216 | 0.5344 |
| rus vs baseline | Cross-domain | Out-domain | DTW | 29.0 | 0.1212 | +0.420 | medium | 0.5203 | 0.5139 |
| smote vs baseline | Cross-domain | Out-domain | DTW | 22.0 | 0.0376 | +0.560 | large | 0.5216 | 0.5139 |
| sw_smote vs baseline | Cross-domain | Out-domain | DTW | 71.0 | 0.1212 | -0.420 | medium | 0.5075 | 0.5139 |
| rus vs baseline | Cross-domain | Out-domain | WASSERSTEIN | 36.0 | 0.3075 | +0.280 | small | 0.5240 | 0.5101 |
| smote vs baseline | Cross-domain | Out-domain | WASSERSTEIN | 25.0 | 0.0640 | +0.500 | large | 0.5159 | 0.5101 |
| sw_smote vs baseline | Cross-domain | Out-domain | WASSERSTEIN | 50.0 | 1.0000 | +0.000 | negligible | 0.5078 | 0.5101 |
| rus vs baseline | Within-domain | In-domain | MMD | 51.0 | 0.9698 | -0.020 | negligible | 0.5990 | 0.6291 |
| smote vs baseline | Within-domain | In-domain | MMD | 2.0 | 0.0003 * | +0.960 | large | 0.9084 | 0.6291 |
| sw_smote vs baseline | Within-domain | In-domain | MMD | 4.0 | 0.0006 * | +0.920 | large | 0.8698 | 0.6291 |
| rus vs baseline | Within-domain | In-domain | DTW | 21.0 | 0.0312 | +0.580 | large | 0.5633 | 0.5329 |
| smote vs baseline | Within-domain | In-domain | DTW | 0.0 | 0.0002 * | +1.000 | large | 0.9012 | 0.5329 |
| sw_smote vs baseline | Within-domain | In-domain | DTW | 0.0 | 0.0002 * | +1.000 | large | 0.8943 | 0.5329 |
| rus vs baseline | Within-domain | In-domain | WASSERSTEIN | 48.0 | 0.9097 | +0.040 | negligible | 0.6368 | 0.6181 |
| smote vs baseline | Within-domain | In-domain | WASSERSTEIN | 0.0 | 0.0002 * | +1.000 | large | 0.8950 | 0.6181 |
| sw_smote vs baseline | Within-domain | In-domain | WASSERSTEIN | 0.0 | 0.0002 * | +1.000 | large | 0.9179 | 0.6181 |
| rus vs baseline | Within-domain | Out-domain | MMD | 60.0 | 0.4727 | -0.200 | small | 0.5916 | 0.5917 |
| smote vs baseline | Within-domain | Out-domain | MMD | 0.0 | 0.0002 * | +1.000 | large | 0.9000 | 0.5917 |
| sw_smote vs baseline | Within-domain | Out-domain | MMD | 0.0 | 0.0002 * | +1.000 | large | 0.8901 | 0.5917 |
| rus vs baseline | Within-domain | Out-domain | DTW | 23.0 | 0.0452 | +0.540 | large | 0.7011 | 0.6111 |
| smote vs baseline | Within-domain | Out-domain | DTW | 0.0 | 0.0002 * | +1.000 | large | 0.9042 | 0.6111 |
| sw_smote vs baseline | Within-domain | Out-domain | DTW | 1.0 | 0.0002 * | +0.980 | large | 0.8969 | 0.6111 |
| rus vs baseline | Within-domain | Out-domain | WASSERSTEIN | 88.0 | 0.0046 | -0.760 | large | 0.6618 | 0.8012 |
| smote vs baseline | Within-domain | Out-domain | WASSERSTEIN | 3.0 | 0.0004 * | +0.940 | large | 0.9119 | 0.8012 |
| sw_smote vs baseline | Within-domain | Out-domain | WASSERSTEIN | 0.0 | 0.0002 * | +1.000 | large | 0.9318 | 0.8012 |
| rus vs baseline | Mixed | In-domain | MMD | 78.0 | 0.0376 | -0.560 | large | 0.5802 | 0.6155 |
| smote vs baseline | Mixed | In-domain | MMD | 0.0 | 0.0002 * | +1.000 | large | 0.8611 | 0.6155 |
| sw_smote vs baseline | Mixed | In-domain | MMD | 0.0 | 0.0002 * | +1.000 | large | 0.8611 | 0.6155 |
| rus vs baseline | Mixed | In-domain | DTW | 70.0 | 0.1405 | -0.400 | medium | 0.6283 | 0.6877 |
| smote vs baseline | Mixed | In-domain | DTW | 1.0 | 0.0002 * | +0.980 | large | 0.8361 | 0.6877 |
| sw_smote vs baseline | Mixed | In-domain | DTW | 0.0 | 0.0002 * | +1.000 | large | 0.8490 | 0.6877 |
| rus vs baseline | Mixed | In-domain | WASSERSTEIN | 81.0 | 0.0211 | -0.620 | large | 0.5884 | 0.6769 |
| smote vs baseline | Mixed | In-domain | WASSERSTEIN | 1.0 | 0.0002 * | +0.980 | large | 0.8472 | 0.6769 |
| sw_smote vs baseline | Mixed | In-domain | WASSERSTEIN | 0.0 | 0.0002 * | +1.000 | large | 0.8549 | 0.6769 |
| rus vs baseline | Mixed | Out-domain | MMD | 93.0 | 0.0013 | -0.860 | large | 0.6662 | 0.8371 |
| smote vs baseline | Mixed | Out-domain | MMD | 20.0 | 0.0257 | +0.600 | large | 0.9060 | 0.8371 |
| sw_smote vs baseline | Mixed | Out-domain | MMD | 24.0 | 0.0539 | +0.520 | large | 0.8863 | 0.8371 |
| rus vs baseline | Mixed | Out-domain | DTW | 94.0 | 0.0010 | -0.880 | large | 0.6565 | 0.8383 |
| smote vs baseline | Mixed | Out-domain | DTW | 23.0 | 0.0452 | +0.540 | large | 0.9124 | 0.8383 |
| sw_smote vs baseline | Mixed | Out-domain | DTW | 28.0 | 0.1041 | +0.440 | medium | 0.8890 | 0.8383 |
| rus vs baseline | Mixed | Out-domain | WASSERSTEIN | 94.0 | 0.0010 | -0.880 | large | 0.6595 | 0.8446 |
| smote vs baseline | Mixed | Out-domain | WASSERSTEIN | 23.0 | 0.0452 | +0.540 | large | 0.9165 | 0.8446 |
| sw_smote vs baseline | Mixed | Out-domain | WASSERSTEIN | 20.0 | 0.0455 | +0.556 | large | 0.8919 | 0.8446 |

**Bonferroni threshold**: α'=0.00093 (m=54). **19** comparisons significant after correction.

### 3.4 Paired Comparison (Wilcoxon Signed-Rank)

Paired by seed — more powerful than Mann-Whitney when seeds are shared.

$$W = \sum_{i=1}^{n} \text{sign}(d_i) \cdot R_i, \quad d_i = Y_{\text{method},i} - Y_{\text{baseline},i}$$

| Comparison | Mode | Level | Distance | W | p | Cliff's δ | Effect | n |
|------------|------|-------|----------|--:|--:|----------:|:------:|--:|
| rus vs baseline | Cross-domain | In-domain | MMD | 0.0 | 0.0020 | -1.000 | large | 10 |
| smote vs baseline | Cross-domain | In-domain | MMD | 16.0 | 0.2754 | -0.400 | medium | 10 |
| sw_smote vs baseline | Cross-domain | In-domain | MMD | 4.0 | 0.0137 | -0.680 | large | 10 |
| rus vs baseline | Cross-domain | In-domain | DTW | 13.0 | 0.1602 | +0.540 | large | 10 |
| smote vs baseline | Cross-domain | In-domain | DTW | 16.0 | 0.2754 | -0.400 | medium | 10 |
| sw_smote vs baseline | Cross-domain | In-domain | DTW | 6.0 | 0.0273 | -0.540 | large | 10 |
| rus vs baseline | Cross-domain | In-domain | WASSERSTEIN | 14.0 | 0.1934 | +0.180 | small | 10 |
| smote vs baseline | Cross-domain | In-domain | WASSERSTEIN | 3.0 | 0.0098 | -0.820 | large | 10 |
| sw_smote vs baseline | Cross-domain | In-domain | WASSERSTEIN | 27.0 | 1.0000 | -0.040 | negligible | 10 |
| rus vs baseline | Cross-domain | Out-domain | MMD | 27.0 | 1.0000 | -0.120 | negligible | 10 |
| smote vs baseline | Cross-domain | Out-domain | MMD | 17.0 | 0.3223 | -0.220 | small | 10 |
| sw_smote vs baseline | Cross-domain | Out-domain | MMD | 4.0 | 0.0137 | -0.640 | large | 10 |
| rus vs baseline | Cross-domain | Out-domain | DTW | 15.0 | 0.2324 | +0.420 | medium | 10 |
| smote vs baseline | Cross-domain | Out-domain | DTW | 9.0 | 0.0645 | +0.560 | large | 10 |
| sw_smote vs baseline | Cross-domain | Out-domain | DTW | 11.0 | 0.1055 | -0.420 | medium | 10 |
| rus vs baseline | Cross-domain | Out-domain | WASSERSTEIN | 13.0 | 0.1602 | +0.280 | small | 10 |
| smote vs baseline | Cross-domain | Out-domain | WASSERSTEIN | 12.0 | 0.1309 | +0.500 | large | 10 |
| sw_smote vs baseline | Cross-domain | Out-domain | WASSERSTEIN | 21.0 | 0.5566 | +0.000 | negligible | 10 |
| rus vs baseline | Within-domain | In-domain | MMD | 18.0 | 0.3750 | -0.020 | negligible | 10 |
| smote vs baseline | Within-domain | In-domain | MMD | 0.0 | 0.0020 | +0.960 | large | 10 |
| sw_smote vs baseline | Within-domain | In-domain | MMD | 1.0 | 0.0039 | +0.920 | large | 10 |
| rus vs baseline | Within-domain | In-domain | DTW | 8.0 | 0.0488 | +0.580 | large | 10 |
| smote vs baseline | Within-domain | In-domain | DTW | 0.0 | 0.0020 | +1.000 | large | 10 |
| sw_smote vs baseline | Within-domain | In-domain | DTW | 0.0 | 0.0020 | +1.000 | large | 10 |
| rus vs baseline | Within-domain | In-domain | WASSERSTEIN | 23.0 | 0.6953 | +0.040 | negligible | 10 |
| smote vs baseline | Within-domain | In-domain | WASSERSTEIN | 0.0 | 0.0020 | +1.000 | large | 10 |
| sw_smote vs baseline | Within-domain | In-domain | WASSERSTEIN | 0.0 | 0.0020 | +1.000 | large | 10 |
| rus vs baseline | Within-domain | Out-domain | MMD | 26.0 | 0.9219 | -0.200 | small | 10 |
| smote vs baseline | Within-domain | Out-domain | MMD | 0.0 | 0.0020 | +1.000 | large | 10 |
| sw_smote vs baseline | Within-domain | Out-domain | MMD | 0.0 | 0.0020 | +1.000 | large | 10 |
| rus vs baseline | Within-domain | Out-domain | DTW | 9.0 | 0.0645 | +0.540 | large | 10 |
| smote vs baseline | Within-domain | Out-domain | DTW | 0.0 | 0.0020 | +1.000 | large | 10 |
| sw_smote vs baseline | Within-domain | Out-domain | DTW | 0.0 | 0.0020 | +0.980 | large | 10 |
| rus vs baseline | Within-domain | Out-domain | WASSERSTEIN | 0.0 | 0.0020 | -0.760 | large | 10 |
| smote vs baseline | Within-domain | Out-domain | WASSERSTEIN | 0.0 | 0.0020 | +0.940 | large | 10 |
| sw_smote vs baseline | Within-domain | Out-domain | WASSERSTEIN | 0.0 | 0.0020 | +1.000 | large | 10 |
| rus vs baseline | Mixed | In-domain | MMD | 14.0 | 0.1934 | -0.560 | large | 10 |
| smote vs baseline | Mixed | In-domain | MMD | 0.0 | 0.0020 | +1.000 | large | 10 |
| sw_smote vs baseline | Mixed | In-domain | MMD | 0.0 | 0.0020 | +1.000 | large | 10 |
| rus vs baseline | Mixed | In-domain | DTW | 16.0 | 0.2754 | -0.400 | medium | 10 |
| smote vs baseline | Mixed | In-domain | DTW | 0.0 | 0.0020 | +0.980 | large | 10 |
| sw_smote vs baseline | Mixed | In-domain | DTW | 0.0 | 0.0020 | +1.000 | large | 10 |
| rus vs baseline | Mixed | In-domain | WASSERSTEIN | 9.0 | 0.0645 | -0.620 | large | 10 |
| smote vs baseline | Mixed | In-domain | WASSERSTEIN | 0.0 | 0.0020 | +0.980 | large | 10 |
| sw_smote vs baseline | Mixed | In-domain | WASSERSTEIN | 0.0 | 0.0020 | +1.000 | large | 10 |
| rus vs baseline | Mixed | Out-domain | MMD | 2.0 | 0.0059 | -0.860 | large | 10 |
| smote vs baseline | Mixed | Out-domain | MMD | 6.0 | 0.0273 | +0.600 | large | 10 |
| sw_smote vs baseline | Mixed | Out-domain | MMD | 7.0 | 0.0371 | +0.520 | large | 10 |
| rus vs baseline | Mixed | Out-domain | DTW | 1.0 | 0.0039 | -0.880 | large | 10 |
| smote vs baseline | Mixed | Out-domain | DTW | 5.0 | 0.0195 | +0.540 | large | 10 |
| sw_smote vs baseline | Mixed | Out-domain | DTW | 3.0 | 0.0098 | +0.440 | medium | 10 |
| rus vs baseline | Mixed | Out-domain | WASSERSTEIN | 1.0 | 0.0039 | -0.880 | large | 10 |
| smote vs baseline | Mixed | Out-domain | WASSERSTEIN | 6.0 | 0.0273 | +0.540 | large | 10 |
| sw_smote vs baseline | Mixed | Out-domain | WASSERSTEIN | 2.0 | 0.0117 | +0.506 | large | 9 |

**Bonferroni threshold**: α'=0.00093 (m=54). **0** comparisons significant after correction.

**Note**: With n=10 paired observations, the minimum achievable p-value for Wilcoxon signed-rank is $p_{\min} = 1/2^{10-1} = 0.001953$. When $p_{\min} > \alpha'$, no comparison can reach Bonferroni significance regardless of effect magnitude. This is a **floor effect** of small sample size, not evidence of no difference.

### 3.5 Friedman Test (Repeated-Measures)

Seeds serve as blocks (subjects). Tests whether at least one condition differs.

$$\chi_F^2 = \frac{12}{bk(k+1)} \sum_{j=1}^{k} R_j^2 - 3b(k+1)$$

where $b$ = number of blocks (seeds), $k$ = number of conditions, $R_j$ = rank sum.

| Mode | Level | Distance | χ² | p-value | n_seeds | Sig |
|------|-------|----------|----|--------:|--------:|:---:|
| Cross-domain | In-domain | MMD | 22.680 | 0.0000 | 10 | ✓ |
| Cross-domain | In-domain | DTW | 7.560 | 0.0560 | 10 |  |
| Cross-domain | In-domain | WASSERSTEIN | 3.720 | 0.2933 | 10 |  |
| Cross-domain | Out-domain | MMD | 5.160 | 0.1604 | 10 |  |
| Cross-domain | Out-domain | DTW | 16.200 | 0.0010 | 10 | ✓ |
| Cross-domain | Out-domain | WASSERSTEIN | 5.640 | 0.1305 | 10 |  |
| Within-domain | In-domain | MMD | 22.200 | 0.0001 | 10 | ✓ |
| Within-domain | In-domain | DTW | 26.040 | 0.0000 | 10 | ✓ |
| Within-domain | In-domain | WASSERSTEIN | 24.480 | 0.0000 | 10 | ✓ |
| Within-domain | Out-domain | MMD | 24.480 | 0.0000 | 10 | ✓ |
| Within-domain | Out-domain | DTW | 22.680 | 0.0000 | 10 | ✓ |
| Within-domain | Out-domain | WASSERSTEIN | 27.480 | 0.0000 | 10 | ✓ |
| Mixed | In-domain | MMD | 22.440 | 0.0001 | 10 | ✓ |
| Mixed | In-domain | DTW | 18.840 | 0.0003 | 10 | ✓ |
| Mixed | In-domain | WASSERSTEIN | 22.680 | 0.0000 | 10 | ✓ |
| Mixed | Out-domain | MMD | 20.160 | 0.0002 | 10 | ✓ |
| Mixed | Out-domain | DTW | 19.800 | 0.0002 | 10 | ✓ |
| Mixed | Out-domain | WASSERSTEIN | 19.533 | 0.0002 | 9 | ✓ |

### 3.6 Domain Gap Analysis

Domain gap: $\Delta = Y_{\text{out-domain}} - Y_{\text{in-domain}}$
 (paired by seed).

Negative Δ indicates performance degrades in out-domain (expected for domain shift).

| Mode | Distance | Condition | Mean Δ | SD Δ | n | KW p (across conds) |
|------|----------|-----------|-------:|-----:|--:|--------------------:|
| Cross-domain | MMD | baseline | +0.0117 | 0.0042 | 10 | 0.0142 |
| Cross-domain | MMD | rus | +0.0295 | 0.0187 | 10 | 0.0142 |
| Cross-domain | MMD | smote | +0.0124 | 0.0083 | 10 | 0.0142 |
| Cross-domain | MMD | sw_smote | +0.0085 | 0.0132 | 10 | 0.0142 |
| Cross-domain | DTW | baseline | -0.0047 | 0.0095 | 10 | 0.0495 |
| Cross-domain | DTW | rus | -0.0039 | 0.0145 | 10 | 0.0495 |
| Cross-domain | DTW | smote | +0.0061 | 0.0053 | 10 | 0.0495 |
| Cross-domain | DTW | sw_smote | -0.0038 | 0.0085 | 10 | 0.0495 |
| Cross-domain | WASSERSTEIN | baseline | -0.0181 | 0.0086 | 10 | 0.0308 |
| Cross-domain | WASSERSTEIN | rus | -0.0201 | 0.0415 | 10 | 0.0308 |
| Cross-domain | WASSERSTEIN | smote | -0.0023 | 0.0097 | 10 | 0.0308 |
| Cross-domain | WASSERSTEIN | sw_smote | -0.0194 | 0.0114 | 10 | 0.0308 |
| Within-domain | MMD | baseline | -0.0374 | 0.0888 | 10 | 0.4061 |
| Within-domain | MMD | rus | -0.0074 | 0.0524 | 10 | 0.4061 |
| Within-domain | MMD | smote | -0.0084 | 0.0344 | 10 | 0.4061 |
| Within-domain | MMD | sw_smote | +0.0203 | 0.1169 | 10 | 0.4061 |
| Within-domain | DTW | baseline | +0.0782 | 0.0568 | 10 | 0.0003 |
| Within-domain | DTW | rus | +0.1378 | 0.1121 | 10 | 0.0003 |
| Within-domain | DTW | smote | +0.0030 | 0.0254 | 10 | 0.0003 |
| Within-domain | DTW | sw_smote | +0.0026 | 0.0751 | 10 | 0.0003 |
| Within-domain | WASSERSTEIN | baseline | +0.1832 | 0.1115 | 10 | 0.0114 |
| Within-domain | WASSERSTEIN | rus | +0.0250 | 0.1317 | 10 | 0.0114 |
| Within-domain | WASSERSTEIN | smote | +0.0169 | 0.0259 | 10 | 0.0114 |
| Within-domain | WASSERSTEIN | sw_smote | +0.0140 | 0.0152 | 10 | 0.0114 |
| Mixed | MMD | baseline | +0.2216 | 0.0894 | 10 | 0.0008 |
| Mixed | MMD | rus | +0.0860 | 0.1586 | 10 | 0.0008 |
| Mixed | MMD | smote | +0.0449 | 0.0468 | 10 | 0.0008 |
| Mixed | MMD | sw_smote | +0.0252 | 0.0900 | 10 | 0.0008 |
| Mixed | DTW | baseline | +0.1506 | 0.0948 | 10 | 0.0848 |
| Mixed | DTW | rus | +0.0281 | 0.1721 | 10 | 0.0848 |
| Mixed | DTW | smote | +0.0764 | 0.0436 | 10 | 0.0848 |
| Mixed | DTW | sw_smote | +0.0400 | 0.0848 | 10 | 0.0848 |
| Mixed | WASSERSTEIN | baseline | +0.1677 | 0.0939 | 10 | 0.0402 |
| Mixed | WASSERSTEIN | rus | +0.0711 | 0.1447 | 10 | 0.0402 |
| Mixed | WASSERSTEIN | smote | +0.0693 | 0.0452 | 10 | 0.0402 |
| Mixed | WASSERSTEIN | sw_smote | +0.0402 | 0.0906 | 9 | 0.0402 |

### 3.7 Condition × Distance Interaction

Does the best-performing condition depend on which distance metric is used?

| Mode | Level | Distance | Best Condition | Mean AUROC | Consistent? |
|------|-------|----------|----------------|---:|:-----------:|
| Cross-domain | In-domain | MMD | baseline | 0.5228 | ✗ |
| Cross-domain | In-domain | DTW | rus | 0.5242 | ✗ |
| Cross-domain | In-domain | WASSERSTEIN | rus | 0.5441 | ✗ |
| Cross-domain | Out-domain | MMD | rus | 0.5350 | ✗ |
| Cross-domain | Out-domain | DTW | smote | 0.5216 | ✗ |
| Cross-domain | Out-domain | WASSERSTEIN | rus | 0.5240 | ✗ |
| Within-domain | In-domain | MMD | smote | 0.9084 | ✗ |
| Within-domain | In-domain | DTW | smote | 0.9012 | ✗ |
| Within-domain | In-domain | WASSERSTEIN | sw_smote | 0.9179 | ✗ |
| Within-domain | Out-domain | MMD | smote | 0.9000 | ✗ |
| Within-domain | Out-domain | DTW | smote | 0.9042 | ✗ |
| Within-domain | Out-domain | WASSERSTEIN | sw_smote | 0.9318 | ✗ |
| Mixed | In-domain | MMD | sw_smote | 0.8611 | ✓ |
| Mixed | In-domain | DTW | sw_smote | 0.8490 | ✓ |
| Mixed | In-domain | WASSERSTEIN | sw_smote | 0.8549 | ✓ |
| Mixed | Out-domain | MMD | smote | 0.9060 | ✓ |
| Mixed | Out-domain | DTW | smote | 0.9124 | ✓ |
| Mixed | Out-domain | WASSERSTEIN | smote | 0.9165 | ✓ |

### 3.8 Overall Condition Ranking

Mean rank across all 18 cells (3 modes × 2 levels × 3 distances). Rank 1 = best.

| Rank | Condition | Mean Rank |
|:----:|-----------|----------:|
| 1 | smote | 1.78 |
| 2 | sw_smote | 2.28 |
| 3 | baseline | 2.89 |
| 4 | rus | 3.06 |

---
## 4. Cross-Metric Synthesis

### 4.1 Overall Rankings Comparison

| Condition | Mean Rank (F2) | Mean Rank (AUROC) | Average |
|-----------|:--------------:|:-----------------:|:-------:|
| baseline | 2.67 | 2.89 | 2.78 |
| rus | 3.00 | 3.06 | 3.03 |
| smote | 2.33 | 1.78 | 2.06 |
| sw_smote | 2.00 | 2.28 | 2.14 |

### 4.2 Summary of Significant Findings

| Test | Metric | Sig/Total (raw α=0.05) | Sig/Total (Bonferroni) |
|------|--------|:----------------------:|:----------------------:|
| Kruskal-Wallis | F2 | 18/18 | 18/18 |
| Kruskal-Wallis | AUROC | 16/18 | 14/18 |
| Mann-Whitney U | F2 | 47/54 | 37/54 |
| Mann-Whitney U | AUROC | 37/54 | 19/54 |
| Wilcoxon SR | F2 | 45/54 | 0/54 |
| Wilcoxon SR | AUROC | 34/54 | 0/54 |

### 4.3 Effect Size Summary (Cliff's δ)

Distribution of effect sizes across all pairwise comparisons:

**F2**:
  - negligible: 1/54 (2%)
  - small: 1/54 (2%)
  - medium: 5/54 (9%)
  - large: 47/54 (87%)

**AUROC**:
  - negligible: 5/54 (9%)
  - small: 4/54 (7%)
  - medium: 6/54 (11%)
  - large: 39/54 (72%)

---
## 5. Statistical Power Analysis

### 5.1 Current Design Power

- Current number of seeds: **n=10**
- Unique seeds: [0, 1, 7, 13, 42, 123, 256, 512, 1337, 2024]
- Cells per condition: 3 modes × 2 levels × 3 distances = 18
- Observations per cell per condition: ~10 seeds

### 5.2 Power Considerations

For Wilcoxon signed-rank test with n paired observations:

$$\text{Power} = P(\text{reject } H_0 \mid H_1 \text{ true})$$

With $n = 10$ seeds, the minimum detectable Cliff's δ at α=0.05 is approximately:

$$|\delta_{\min}| \approx \frac{z_{\alpha/2}}{\sqrt{n}} \approx \frac{1.96}{\sqrt{10}} \approx 0.620$$

This means only **large** effects (|δ| > 0.474) are reliably detectable.
Medium effects require n ≥ 20-30 seeds; small effects n ≥ 50+.

---
## 6. Proposed Additional Experiments

### 6.1 Experiment A: Increased Seed Count

**Rationale**: Current $n=10$ seeds may lack power for medium/small effects.

**Proposal**: Increase to $n \geq 20$ seeds to detect medium effects (|δ| ≈ 0.33).

Required sample size for Wilcoxon signed-rank (approximation):
$$n \geq \left(\frac{z_{\alpha'/2} + z_\beta}{\delta}\right)^2$$

For Bonferroni-adjusted α' and power 0.80:

- To detect |δ| ≥ 0.33: $n \geq 174$ seeds
- To detect |δ| ≥ 0.474: $n \geq 84$ seeds

### 6.2 Experiment B: Ratio Sensitivity Analysis

**Rationale**: Current primary analysis fixes ratio=0.1. Different ratios may yield different rankings.

**Proposal**: Repeat full analysis with ratio=0.5 and compare rankings.

- If rankings differ: ratio is a significant moderator → report both

- If rankings agree: robust finding, ratio is secondary

### 6.3 Experiment C: Cross-Validation of Distance Metric Groupings

**Rationale**: Domain groups are defined by distance metric thresholds. 
Different threshold choices could change which subjects are in/out-domain.

**Proposal**: Vary the distance threshold (e.g., percentile-based: 25th, 50th, 75th) 
and check robustness of findings.

### 6.4 Experiment D: Permutation Test for Global Null

**Rationale**: Multiple testing correction is conservative. A permutation-based 
global test can provide an exact p-value.

**Proposal**:
1. For each seed, permute condition labels across all cells
2. Compute a global test statistic (e.g., sum of |Δ| across cells)
3. Repeat 10,000+ times
4. Compare observed statistic to null distribution

$$p_{\text{perm}} = \frac{1}{B}\sum_{b=1}^{B} \mathbb{1}[T^{(b)} \geq T_{\text{obs}}]$$

### 6.5 Experiment E: Bootstrap Confidence Intervals

**Rationale**: Provide interval estimates rather than just point estimates / p-values.

**Proposal**: For each (condition × mode × level) cell:

1. Bootstrap resample seeds B=10,000 times
2. Compute 95% BCa confidence interval for mean F2 and AUROC
3. Compare overlap of CIs between conditions

$$\text{CI}_{95\%} = \left[\hat{\theta}^*_{(\alpha/2)}, \hat{\theta}^*_{(1-\alpha/2)}\right]$$

### 6.6 Experiment F: Cross-Domain Degradation Ratio

**Rationale**: Absolute domain gap Δ depends on baseline performance level. 
A relative measure normalizes this.

**Proposal**: Compute degradation ratio:

$$\rho = \frac{Y_{\text{out-domain}}}{Y_{\text{in-domain}}}$$

$\rho < 1$ indicates degradation; $\rho = 1$ means no domain shift effect; $\rho > 1$ means improvement.

Compare $\rho$ across conditions to identify which method best preserves performance under domain shift.
