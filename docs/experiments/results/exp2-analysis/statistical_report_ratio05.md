# Experiment 2 — Domain Shift Statistical Analysis Report

**Generated**: auto  
**Records**: 719  
**Seeds**: [0, 1, 7, 13, 42, 123, 256, 512, 1337, 2024] (n=10)  
**Conditions**: ['baseline', 'rus', 'smote', 'sw_smote']  
**Ratio**: 0.5 (for rus/smote/sw_smote; n/a for baseline)  

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
| rus | 0.1576±0.0133 | 0.1538±0.0168 | -0.0038 |
| smote | 0.1082±0.0216 | 0.1182±0.0166 | +0.0099 |
| sw_smote | 0.0468±0.0109 | 0.0382±0.0122 | -0.0086 |

#### Mode: Within-domain

| Condition | In-domain (mean±SD) | Out-domain (mean±SD) | Δ (out−in) |
|-----------|--------------------:|---------------------:|-----------:|
| baseline | 0.2054±0.0345 | 0.2233±0.0528 | +0.0179 |
| rus | 0.1306±0.0504 | 0.1843±0.0401 | +0.0538 |
| smote | 0.5068±0.0977 | 0.5044±0.0913 | -0.0024 |
| sw_smote | 0.5006±0.1807 | 0.4794±0.1803 | -0.0212 |

#### Mode: Mixed

| Condition | In-domain (mean±SD) | Out-domain (mean±SD) | Δ (out−in) |
|-----------|--------------------:|---------------------:|-----------:|
| baseline | 0.2394±0.0221 | 0.3084±0.0299 | +0.0690 |
| rus | 0.1448±0.0221 | 0.1917±0.0212 | +0.0469 |
| smote | 0.4856±0.0939 | 0.5613±0.1197 | +0.0757 |
| sw_smote | 0.3828±0.1739 | 0.4636±0.1644 | +0.0808 |

### 2.2 Kruskal-Wallis H-test (Condition effect)

Tests whether the distribution of F2-score differs across the 4 conditions.

$$H = \frac{12}{N(N+1)} \sum_{i=1}^{k} \frac{R_i^2}{n_i} - 3(N+1)$$

where $R_i$ is the sum of ranks in group $i$, $n_i$ the group size, $N = \sum n_i$.

| Mode | Level | Distance | H | p-value | Sig (α=0.05) |
|------|-------|----------|--:|--------:|:------------:|
| Cross-domain | In-domain | MMD | 33.151 | 0.0000 | ✓ |
| Cross-domain | In-domain | DTW | 33.138 | 0.0000 | ✓ |
| Cross-domain | In-domain | WASSERSTEIN | 33.020 | 0.0000 | ✓ |
| Cross-domain | Out-domain | MMD | 32.718 | 0.0000 | ✓ |
| Cross-domain | Out-domain | DTW | 33.045 | 0.0000 | ✓ |
| Cross-domain | Out-domain | WASSERSTEIN | 32.564 | 0.0000 | ✓ |
| Within-domain | In-domain | MMD | 30.953 | 0.0000 | ✓ |
| Within-domain | In-domain | DTW | 29.167 | 0.0000 | ✓ |
| Within-domain | In-domain | WASSERSTEIN | 32.963 | 0.0000 | ✓ |
| Within-domain | Out-domain | MMD | 29.552 | 0.0000 | ✓ |
| Within-domain | Out-domain | DTW | 30.755 | 0.0000 | ✓ |
| Within-domain | Out-domain | WASSERSTEIN | 24.885 | 0.0000 | ✓ |
| Mixed | In-domain | MMD | 29.860 | 0.0000 | ✓ |
| Mixed | In-domain | DTW | 26.215 | 0.0000 | ✓ |
| Mixed | In-domain | WASSERSTEIN | 28.828 | 0.0000 | ✓ |
| Mixed | Out-domain | MMD | 31.453 | 0.0000 | ✓ |
| Mixed | Out-domain | DTW | 30.347 | 0.0000 | ✓ |
| Mixed | Out-domain | WASSERSTEIN | 30.474 | 0.0000 | ✓ |

**Summary**: 18/18 cells significant at α=0.05; 18/18 after Bonferroni correction (α'=0.0028).

### 2.3 Pairwise Comparisons (Mann-Whitney U)

Baseline vs each method, testing:

$$H_0: F_{\text{baseline}}(x) = F_{\text{method}}(x)$$
$$H_1: F_{\text{baseline}}(x) \neq F_{\text{method}}(x)$$

| Comparison | Mode | Level | Distance | U | p | Cliff's δ | Effect | Mean(method) | Mean(baseline) |
|------------|------|-------|----------|--:|--:|----------:|:------:|-------------:|---------------:|
| rus vs baseline | Cross-domain | In-domain | MMD | 79.0 | 0.0311 | -0.580 | large | 0.1548 | 0.1625 |
| smote vs baseline | Cross-domain | In-domain | MMD | 100.0 | 0.0002 * | -1.000 | large | 0.1240 | 0.1625 |
| sw_smote vs baseline | Cross-domain | In-domain | MMD | 100.0 | 0.0002 * | -1.000 | large | 0.0366 | 0.1625 |
| rus vs baseline | Cross-domain | In-domain | DTW | 62.0 | 0.3847 | -0.240 | small | 0.1654 | 0.1686 |
| smote vs baseline | Cross-domain | In-domain | DTW | 100.0 | 0.0002 * | -1.000 | large | 0.0939 | 0.1686 |
| sw_smote vs baseline | Cross-domain | In-domain | DTW | 100.0 | 0.0002 * | -1.000 | large | 0.0525 | 0.1686 |
| rus vs baseline | Cross-domain | In-domain | WASSERSTEIN | 58.0 | 0.5708 | -0.160 | small | 0.1526 | 0.1600 |
| smote vs baseline | Cross-domain | In-domain | WASSERSTEIN | 100.0 | 0.0002 * | -1.000 | large | 0.1067 | 0.1600 |
| sw_smote vs baseline | Cross-domain | In-domain | WASSERSTEIN | 100.0 | 0.0002 * | -1.000 | large | 0.0512 | 0.1600 |
| rus vs baseline | Cross-domain | Out-domain | MMD | 76.0 | 0.0539 | -0.520 | large | 0.1584 | 0.1670 |
| smote vs baseline | Cross-domain | Out-domain | MMD | 100.0 | 0.0002 * | -1.000 | large | 0.1228 | 0.1670 |
| sw_smote vs baseline | Cross-domain | Out-domain | MMD | 100.0 | 0.0002 * | -1.000 | large | 0.0367 | 0.1670 |
| rus vs baseline | Cross-domain | Out-domain | DTW | 59.0 | 0.5205 | -0.180 | small | 0.1424 | 0.1439 |
| smote vs baseline | Cross-domain | Out-domain | DTW | 100.0 | 0.0002 * | -1.000 | large | 0.1067 | 0.1439 |
| sw_smote vs baseline | Cross-domain | Out-domain | DTW | 100.0 | 0.0002 * | -1.000 | large | 0.0381 | 0.1439 |
| rus vs baseline | Cross-domain | Out-domain | WASSERSTEIN | 56.0 | 0.6776 | -0.120 | negligible | 0.1607 | 0.1607 |
| smote vs baseline | Cross-domain | Out-domain | WASSERSTEIN | 100.0 | 0.0002 * | -1.000 | large | 0.1250 | 0.1607 |
| sw_smote vs baseline | Cross-domain | Out-domain | WASSERSTEIN | 100.0 | 0.0002 * | -1.000 | large | 0.0396 | 0.1607 |
| rus vs baseline | Within-domain | In-domain | MMD | 100.0 | 0.0002 * | -1.000 | large | 0.1068 | 0.2272 |
| smote vs baseline | Within-domain | In-domain | MMD | 0.0 | 0.0002 * | +1.000 | large | 0.5177 | 0.2272 |
| sw_smote vs baseline | Within-domain | In-domain | MMD | 11.0 | 0.0036 | +0.780 | large | 0.4687 | 0.2272 |
| rus vs baseline | Within-domain | In-domain | DTW | 61.0 | 0.4274 | -0.220 | small | 0.1642 | 0.1720 |
| smote vs baseline | Within-domain | In-domain | DTW | 0.0 | 0.0002 * | +1.000 | large | 0.4980 | 0.1720 |
| sw_smote vs baseline | Within-domain | In-domain | DTW | 0.0 | 0.0002 * | +1.000 | large | 0.4931 | 0.1720 |
| rus vs baseline | Within-domain | In-domain | WASSERSTEIN | 100.0 | 0.0002 * | -1.000 | large | 0.1207 | 0.2169 |
| smote vs baseline | Within-domain | In-domain | WASSERSTEIN | 0.0 | 0.0002 * | +1.000 | large | 0.5048 | 0.2169 |
| sw_smote vs baseline | Within-domain | In-domain | WASSERSTEIN | 0.0 | 0.0002 * | +1.000 | large | 0.5402 | 0.2169 |
| rus vs baseline | Within-domain | Out-domain | MMD | 37.0 | 0.3447 | +0.260 | small | 0.2031 | 0.1909 |
| smote vs baseline | Within-domain | Out-domain | MMD | 0.0 | 0.0002 * | +1.000 | large | 0.5157 | 0.1909 |
| sw_smote vs baseline | Within-domain | Out-domain | MMD | 0.0 | 0.0002 * | +1.000 | large | 0.5117 | 0.1909 |
| rus vs baseline | Within-domain | Out-domain | DTW | 82.0 | 0.0173 | -0.640 | large | 0.1517 | 0.1939 |
| smote vs baseline | Within-domain | Out-domain | DTW | 0.0 | 0.0002 * | +1.000 | large | 0.5196 | 0.1939 |
| sw_smote vs baseline | Within-domain | Out-domain | DTW | 2.0 | 0.0003 * | +0.960 | large | 0.4312 | 0.1939 |
| rus vs baseline | Within-domain | Out-domain | WASSERSTEIN | 92.0 | 0.0017 | -0.840 | large | 0.1982 | 0.2851 |
| smote vs baseline | Within-domain | Out-domain | WASSERSTEIN | 0.0 | 0.0002 * | +1.000 | large | 0.4778 | 0.2851 |
| sw_smote vs baseline | Within-domain | Out-domain | WASSERSTEIN | 25.0 | 0.0640 | +0.500 | large | 0.4953 | 0.2851 |
| rus vs baseline | Mixed | In-domain | MMD | 100.0 | 0.0002 * | -1.000 | large | 0.1496 | 0.2264 |
| smote vs baseline | Mixed | In-domain | MMD | 0.0 | 0.0002 * | +1.000 | large | 0.5047 | 0.2264 |
| sw_smote vs baseline | Mixed | In-domain | MMD | 22.0 | 0.0376 | +0.560 | large | 0.3884 | 0.2264 |
| rus vs baseline | Mixed | In-domain | DTW | 100.0 | 0.0002 * | -1.000 | large | 0.1510 | 0.2506 |
| smote vs baseline | Mixed | In-domain | DTW | 0.0 | 0.0002 * | +1.000 | large | 0.4731 | 0.2506 |
| sw_smote vs baseline | Mixed | In-domain | DTW | 35.0 | 0.4379 | +0.222 | small | 0.3806 | 0.2506 |
| rus vs baseline | Mixed | In-domain | WASSERSTEIN | 100.0 | 0.0002 * | -1.000 | large | 0.1338 | 0.2412 |
| smote vs baseline | Mixed | In-domain | WASSERSTEIN | 0.0 | 0.0002 * | +1.000 | large | 0.4790 | 0.2412 |
| sw_smote vs baseline | Mixed | In-domain | WASSERSTEIN | 30.0 | 0.1405 | +0.400 | medium | 0.3793 | 0.2412 |
| rus vs baseline | Mixed | Out-domain | MMD | 100.0 | 0.0002 * | -1.000 | large | 0.1961 | 0.2938 |
| smote vs baseline | Mixed | Out-domain | MMD | 0.0 | 0.0002 * | +1.000 | large | 0.5336 | 0.2938 |
| sw_smote vs baseline | Mixed | Out-domain | MMD | 11.0 | 0.0036 | +0.780 | large | 0.4551 | 0.2938 |
| rus vs baseline | Mixed | Out-domain | DTW | 100.0 | 0.0002 * | -1.000 | large | 0.1861 | 0.3056 |
| smote vs baseline | Mixed | Out-domain | DTW | 0.0 | 0.0002 * | +1.000 | large | 0.5677 | 0.3056 |
| sw_smote vs baseline | Mixed | Out-domain | DTW | 21.0 | 0.0312 | +0.580 | large | 0.4581 | 0.3056 |
| rus vs baseline | Mixed | Out-domain | WASSERSTEIN | 100.0 | 0.0002 * | -1.000 | large | 0.1929 | 0.3258 |
| smote vs baseline | Mixed | Out-domain | WASSERSTEIN | 0.0 | 0.0002 * | +1.000 | large | 0.5825 | 0.3258 |
| sw_smote vs baseline | Mixed | Out-domain | WASSERSTEIN | 20.0 | 0.0257 | +0.600 | large | 0.4777 | 0.3258 |

**Bonferroni threshold**: α'=0.00093 (m=54). **36** comparisons significant after correction.

### 2.4 Paired Comparison (Wilcoxon Signed-Rank)

Paired by seed — more powerful than Mann-Whitney when seeds are shared.

$$W = \sum_{i=1}^{n} \text{sign}(d_i) \cdot R_i, \quad d_i = Y_{\text{method},i} - Y_{\text{baseline},i}$$

| Comparison | Mode | Level | Distance | W | p | Cliff's δ | Effect | n |
|------------|------|-------|----------|--:|--:|----------:|:------:|--:|
| rus vs baseline | Cross-domain | In-domain | MMD | 5.0 | 0.0195 | -0.580 | large | 10 |
| smote vs baseline | Cross-domain | In-domain | MMD | 0.0 | 0.0020 | -1.000 | large | 10 |
| sw_smote vs baseline | Cross-domain | In-domain | MMD | 0.0 | 0.0020 | -1.000 | large | 10 |
| rus vs baseline | Cross-domain | In-domain | DTW | 12.0 | 0.1309 | -0.240 | small | 10 |
| smote vs baseline | Cross-domain | In-domain | DTW | 0.0 | 0.0020 | -1.000 | large | 10 |
| sw_smote vs baseline | Cross-domain | In-domain | DTW | 0.0 | 0.0020 | -1.000 | large | 10 |
| rus vs baseline | Cross-domain | In-domain | WASSERSTEIN | 14.0 | 0.1934 | -0.160 | small | 10 |
| smote vs baseline | Cross-domain | In-domain | WASSERSTEIN | 0.0 | 0.0020 | -1.000 | large | 10 |
| sw_smote vs baseline | Cross-domain | In-domain | WASSERSTEIN | 0.0 | 0.0020 | -1.000 | large | 10 |
| rus vs baseline | Cross-domain | Out-domain | MMD | 12.0 | 0.1309 | -0.520 | large | 10 |
| smote vs baseline | Cross-domain | Out-domain | MMD | 0.0 | 0.0020 | -1.000 | large | 10 |
| sw_smote vs baseline | Cross-domain | Out-domain | MMD | 0.0 | 0.0020 | -1.000 | large | 10 |
| rus vs baseline | Cross-domain | Out-domain | DTW | 22.0 | 0.6250 | -0.180 | small | 10 |
| smote vs baseline | Cross-domain | Out-domain | DTW | 0.0 | 0.0020 | -1.000 | large | 10 |
| sw_smote vs baseline | Cross-domain | Out-domain | DTW | 0.0 | 0.0020 | -1.000 | large | 10 |
| rus vs baseline | Cross-domain | Out-domain | WASSERSTEIN | 27.0 | 1.0000 | -0.120 | negligible | 10 |
| smote vs baseline | Cross-domain | Out-domain | WASSERSTEIN | 0.0 | 0.0020 | -1.000 | large | 10 |
| sw_smote vs baseline | Cross-domain | Out-domain | WASSERSTEIN | 0.0 | 0.0020 | -1.000 | large | 10 |
| rus vs baseline | Within-domain | In-domain | MMD | 0.0 | 0.0020 | -1.000 | large | 10 |
| smote vs baseline | Within-domain | In-domain | MMD | 0.0 | 0.0020 | +1.000 | large | 10 |
| sw_smote vs baseline | Within-domain | In-domain | MMD | 2.0 | 0.0059 | +0.780 | large | 10 |
| rus vs baseline | Within-domain | In-domain | DTW | 23.0 | 0.6953 | -0.220 | small | 10 |
| smote vs baseline | Within-domain | In-domain | DTW | 0.0 | 0.0020 | +1.000 | large | 10 |
| sw_smote vs baseline | Within-domain | In-domain | DTW | 0.0 | 0.0020 | +1.000 | large | 10 |
| rus vs baseline | Within-domain | In-domain | WASSERSTEIN | 0.0 | 0.0020 | -1.000 | large | 10 |
| smote vs baseline | Within-domain | In-domain | WASSERSTEIN | 0.0 | 0.0020 | +1.000 | large | 10 |
| sw_smote vs baseline | Within-domain | In-domain | WASSERSTEIN | 0.0 | 0.0020 | +1.000 | large | 10 |
| rus vs baseline | Within-domain | Out-domain | MMD | 12.0 | 0.1309 | +0.260 | small | 10 |
| smote vs baseline | Within-domain | Out-domain | MMD | 0.0 | 0.0020 | +1.000 | large | 10 |
| sw_smote vs baseline | Within-domain | Out-domain | MMD | 0.0 | 0.0020 | +1.000 | large | 10 |
| rus vs baseline | Within-domain | Out-domain | DTW | 6.0 | 0.0273 | -0.640 | large | 10 |
| smote vs baseline | Within-domain | Out-domain | DTW | 0.0 | 0.0020 | +1.000 | large | 10 |
| sw_smote vs baseline | Within-domain | Out-domain | DTW | 0.0 | 0.0020 | +0.960 | large | 10 |
| rus vs baseline | Within-domain | Out-domain | WASSERSTEIN | 1.0 | 0.0039 | -0.840 | large | 10 |
| smote vs baseline | Within-domain | Out-domain | WASSERSTEIN | 0.0 | 0.0020 | +1.000 | large | 10 |
| sw_smote vs baseline | Within-domain | Out-domain | WASSERSTEIN | 6.0 | 0.0273 | +0.500 | large | 10 |
| rus vs baseline | Mixed | In-domain | MMD | 0.0 | 0.0020 | -1.000 | large | 10 |
| smote vs baseline | Mixed | In-domain | MMD | 0.0 | 0.0020 | +1.000 | large | 10 |
| sw_smote vs baseline | Mixed | In-domain | MMD | 5.0 | 0.0195 | +0.560 | large | 10 |
| rus vs baseline | Mixed | In-domain | DTW | 0.0 | 0.0020 | -1.000 | large | 10 |
| smote vs baseline | Mixed | In-domain | DTW | 0.0 | 0.0020 | +1.000 | large | 10 |
| sw_smote vs baseline | Mixed | In-domain | DTW | 11.0 | 0.2031 | +0.210 | small | 9 |
| rus vs baseline | Mixed | In-domain | WASSERSTEIN | 0.0 | 0.0020 | -1.000 | large | 10 |
| smote vs baseline | Mixed | In-domain | WASSERSTEIN | 0.0 | 0.0020 | +1.000 | large | 10 |
| sw_smote vs baseline | Mixed | In-domain | WASSERSTEIN | 11.0 | 0.1055 | +0.400 | medium | 10 |
| rus vs baseline | Mixed | Out-domain | MMD | 0.0 | 0.0020 | -1.000 | large | 10 |
| smote vs baseline | Mixed | Out-domain | MMD | 0.0 | 0.0020 | +1.000 | large | 10 |
| sw_smote vs baseline | Mixed | Out-domain | MMD | 3.0 | 0.0098 | +0.780 | large | 10 |
| rus vs baseline | Mixed | Out-domain | DTW | 0.0 | 0.0020 | -1.000 | large | 10 |
| smote vs baseline | Mixed | Out-domain | DTW | 0.0 | 0.0020 | +1.000 | large | 10 |
| sw_smote vs baseline | Mixed | Out-domain | DTW | 7.0 | 0.0371 | +0.580 | large | 10 |
| rus vs baseline | Mixed | Out-domain | WASSERSTEIN | 0.0 | 0.0020 | -1.000 | large | 10 |
| smote vs baseline | Mixed | Out-domain | WASSERSTEIN | 0.0 | 0.0020 | +1.000 | large | 10 |
| sw_smote vs baseline | Mixed | Out-domain | WASSERSTEIN | 4.0 | 0.0137 | +0.600 | large | 10 |

**Bonferroni threshold**: α'=0.00093 (m=54). **0** comparisons significant after correction.

**Note**: With n=10 paired observations, the minimum achievable p-value for Wilcoxon signed-rank is $p_{\min} = 1/2^{10-1} = 0.001953$. When $p_{\min} > \alpha'$, no comparison can reach Bonferroni significance regardless of effect magnitude. This is a **floor effect** of small sample size, not evidence of no difference.

### 2.5 Friedman Test (Repeated-Measures)

Seeds serve as blocks (subjects). Tests whether at least one condition differs.

$$\chi_F^2 = \frac{12}{bk(k+1)} \sum_{j=1}^{k} R_j^2 - 3b(k+1)$$

where $b$ = number of blocks (seeds), $k$ = number of conditions, $R_j$ = rank sum.

| Mode | Level | Distance | χ² | p-value | n_seeds | Sig |
|------|-------|----------|----|--------:|--------:|:---:|
| Cross-domain | In-domain | MMD | 27.720 | 0.0000 | 10 | ✓ |
| Cross-domain | In-domain | DTW | 27.480 | 0.0000 | 10 | ✓ |
| Cross-domain | In-domain | WASSERSTEIN | 27.120 | 0.0000 | 10 | ✓ |
| Cross-domain | Out-domain | MMD | 26.760 | 0.0000 | 10 | ✓ |
| Cross-domain | Out-domain | DTW | 27.120 | 0.0000 | 10 | ✓ |
| Cross-domain | Out-domain | WASSERSTEIN | 27.000 | 0.0000 | 10 | ✓ |
| Within-domain | In-domain | MMD | 25.560 | 0.0000 | 10 | ✓ |
| Within-domain | In-domain | DTW | 24.240 | 0.0000 | 10 | ✓ |
| Within-domain | In-domain | WASSERSTEIN | 27.120 | 0.0000 | 10 | ✓ |
| Within-domain | Out-domain | MMD | 24.600 | 0.0000 | 10 | ✓ |
| Within-domain | Out-domain | DTW | 27.000 | 0.0000 | 10 | ✓ |
| Within-domain | Out-domain | WASSERSTEIN | 15.960 | 0.0012 | 10 | ✓ |
| Mixed | In-domain | MMD | 25.680 | 0.0000 | 10 | ✓ |
| Mixed | In-domain | DTW | 18.200 | 0.0004 | 9 | ✓ |
| Mixed | In-domain | WASSERSTEIN | 23.160 | 0.0000 | 10 | ✓ |
| Mixed | Out-domain | MMD | 24.840 | 0.0000 | 10 | ✓ |
| Mixed | Out-domain | DTW | 24.840 | 0.0000 | 10 | ✓ |
| Mixed | Out-domain | WASSERSTEIN | 24.840 | 0.0000 | 10 | ✓ |

### 2.6 Domain Gap Analysis

Domain gap: $\Delta = Y_{\text{out-domain}} - Y_{\text{in-domain}}$
 (paired by seed).

Negative Δ indicates performance degrades in out-domain (expected for domain shift).

| Mode | Distance | Condition | Mean Δ | SD Δ | n | KW p (across conds) |
|------|----------|-----------|-------:|-----:|--:|--------------------:|
| Cross-domain | MMD | baseline | +0.0045 | 0.0075 | 10 | 0.5443 |
| Cross-domain | MMD | rus | +0.0035 | 0.0221 | 10 | 0.5443 |
| Cross-domain | MMD | smote | -0.0012 | 0.0156 | 10 | 0.5443 |
| Cross-domain | MMD | sw_smote | +0.0001 | 0.0135 | 10 | 0.5443 |
| Cross-domain | DTW | baseline | -0.0247 | 0.0085 | 10 | 0.0011 |
| Cross-domain | DTW | rus | -0.0230 | 0.0079 | 10 | 0.0011 |
| Cross-domain | DTW | smote | +0.0127 | 0.0201 | 10 | 0.0011 |
| Cross-domain | DTW | sw_smote | -0.0144 | 0.0204 | 10 | 0.0011 |
| Cross-domain | WASSERSTEIN | baseline | +0.0007 | 0.0056 | 10 | 0.0036 |
| Cross-domain | WASSERSTEIN | rus | +0.0081 | 0.0282 | 10 | 0.0036 |
| Cross-domain | WASSERSTEIN | smote | +0.0183 | 0.0159 | 10 | 0.0036 |
| Cross-domain | WASSERSTEIN | sw_smote | -0.0115 | 0.0119 | 10 | 0.0036 |
| Within-domain | MMD | baseline | -0.0363 | 0.0376 | 10 | 0.0032 |
| Within-domain | MMD | rus | +0.0963 | 0.0688 | 10 | 0.0032 |
| Within-domain | MMD | smote | -0.0019 | 0.0929 | 10 | 0.0032 |
| Within-domain | MMD | sw_smote | +0.0430 | 0.1332 | 10 | 0.0032 |
| Within-domain | DTW | baseline | +0.0219 | 0.0339 | 10 | 0.2279 |
| Within-domain | DTW | rus | -0.0124 | 0.0587 | 10 | 0.2279 |
| Within-domain | DTW | smote | +0.0216 | 0.0845 | 10 | 0.2279 |
| Within-domain | DTW | sw_smote | -0.0619 | 0.2177 | 10 | 0.2279 |
| Within-domain | WASSERSTEIN | baseline | +0.0681 | 0.0484 | 10 | 0.0293 |
| Within-domain | WASSERSTEIN | rus | +0.0775 | 0.0248 | 10 | 0.0293 |
| Within-domain | WASSERSTEIN | smote | -0.0271 | 0.0822 | 10 | 0.0293 |
| Within-domain | WASSERSTEIN | sw_smote | -0.0449 | 0.2841 | 10 | 0.0293 |
| Mixed | MMD | baseline | +0.0674 | 0.0371 | 10 | 0.2668 |
| Mixed | MMD | rus | +0.0464 | 0.0268 | 10 | 0.2668 |
| Mixed | MMD | smote | +0.0290 | 0.1242 | 10 | 0.2668 |
| Mixed | MMD | sw_smote | +0.0667 | 0.1082 | 10 | 0.2668 |
| Mixed | DTW | baseline | +0.0550 | 0.0396 | 10 | 0.0604 |
| Mixed | DTW | rus | +0.0351 | 0.0419 | 10 | 0.0604 |
| Mixed | DTW | smote | +0.0945 | 0.1140 | 10 | 0.0604 |
| Mixed | DTW | sw_smote | +0.0792 | 0.1325 | 9 | 0.0604 |
| Mixed | WASSERSTEIN | baseline | +0.0846 | 0.0360 | 10 | 0.1184 |
| Mixed | WASSERSTEIN | rus | +0.0590 | 0.0327 | 10 | 0.1184 |
| Mixed | WASSERSTEIN | smote | +0.1035 | 0.1168 | 10 | 0.1184 |
| Mixed | WASSERSTEIN | sw_smote | +0.0984 | 0.1114 | 10 | 0.1184 |

### 2.7 Condition × Distance Interaction

Does the best-performing condition depend on which distance metric is used?

| Mode | Level | Distance | Best Condition | Mean F2-score | Consistent? |
|------|-------|----------|----------------|---:|:-----------:|
| Cross-domain | In-domain | MMD | baseline | 0.1625 | ✓ |
| Cross-domain | In-domain | DTW | baseline | 0.1686 | ✓ |
| Cross-domain | In-domain | WASSERSTEIN | baseline | 0.1600 | ✓ |
| Cross-domain | Out-domain | MMD | baseline | 0.1670 | ✗ |
| Cross-domain | Out-domain | DTW | baseline | 0.1439 | ✗ |
| Cross-domain | Out-domain | WASSERSTEIN | rus | 0.1607 | ✗ |
| Within-domain | In-domain | MMD | smote | 0.5177 | ✗ |
| Within-domain | In-domain | DTW | smote | 0.4980 | ✗ |
| Within-domain | In-domain | WASSERSTEIN | sw_smote | 0.5402 | ✗ |
| Within-domain | Out-domain | MMD | smote | 0.5157 | ✗ |
| Within-domain | Out-domain | DTW | smote | 0.5196 | ✗ |
| Within-domain | Out-domain | WASSERSTEIN | sw_smote | 0.4953 | ✗ |
| Mixed | In-domain | MMD | smote | 0.5047 | ✓ |
| Mixed | In-domain | DTW | smote | 0.4731 | ✓ |
| Mixed | In-domain | WASSERSTEIN | smote | 0.4790 | ✓ |
| Mixed | Out-domain | MMD | smote | 0.5336 | ✓ |
| Mixed | Out-domain | DTW | smote | 0.5677 | ✓ |
| Mixed | Out-domain | WASSERSTEIN | smote | 0.5825 | ✓ |

### 2.8 Overall Condition Ranking

Mean rank across all 18 cells (3 modes × 2 levels × 3 distances). Rank 1 = best.

| Rank | Condition | Mean Rank |
|:----:|-----------|----------:|
| 1 | smote | 1.78 |
| 2 | baseline | 2.44 |
| 3 | sw_smote | 2.56 |
| 4 | rus | 3.22 |

---
## 3. Analysis: AUROC

### 3.1 Descriptive Statistics

#### Mode: Cross-domain

| Condition | In-domain (mean±SD) | Out-domain (mean±SD) | Δ (out−in) |
|-----------|--------------------:|---------------------:|-----------:|
| baseline | 0.5232±0.0053 | 0.5195±0.0131 | -0.0037 |
| rus | 0.5150±0.0169 | 0.5277±0.0228 | +0.0127 |
| smote | 0.5153±0.0068 | 0.5219±0.0069 | +0.0067 |
| sw_smote | 0.5257±0.0176 | 0.5151±0.0116 | -0.0106 |

#### Mode: Within-domain

| Condition | In-domain (mean±SD) | Out-domain (mean±SD) | Δ (out−in) |
|-----------|--------------------:|---------------------:|-----------:|
| baseline | 0.5933±0.0675 | 0.6680±0.1167 | +0.0747 |
| rus | 0.6047±0.0921 | 0.6021±0.0659 | -0.0026 |
| smote | 0.8805±0.0358 | 0.8889±0.0338 | +0.0084 |
| sw_smote | 0.8656±0.0635 | 0.8714±0.0594 | +0.0058 |

#### Mode: Mixed

| Condition | In-domain (mean±SD) | Out-domain (mean±SD) | Δ (out−in) |
|-----------|--------------------:|---------------------:|-----------:|
| baseline | 0.6600±0.0619 | 0.8400±0.0632 | +0.1800 |
| rus | 0.5473±0.0401 | 0.5935±0.0500 | +0.0462 |
| smote | 0.8444±0.0433 | 0.8969±0.0334 | +0.0525 |
| sw_smote | 0.8342±0.0505 | 0.8825±0.0417 | +0.0483 |

### 3.2 Kruskal-Wallis H-test (Condition effect)

Tests whether the distribution of AUROC differs across the 4 conditions.

$$H = \frac{12}{N(N+1)} \sum_{i=1}^{k} \frac{R_i^2}{n_i} - 3(N+1)$$

where $R_i$ is the sum of ranks in group $i$, $n_i$ the group size, $N = \sum n_i$.

| Mode | Level | Distance | H | p-value | Sig (α=0.05) |
|------|-------|----------|--:|--------:|:------------:|
| Cross-domain | In-domain | MMD | 22.856 | 0.0000 | ✓ |
| Cross-domain | In-domain | DTW | 20.542 | 0.0001 | ✓ |
| Cross-domain | In-domain | WASSERSTEIN | 19.924 | 0.0002 | ✓ |
| Cross-domain | Out-domain | MMD | 11.501 | 0.0093 | ✓ |
| Cross-domain | Out-domain | DTW | 7.763 | 0.0512 |  |
| Cross-domain | Out-domain | WASSERSTEIN | 17.264 | 0.0006 | ✓ |
| Within-domain | In-domain | MMD | 24.780 | 0.0000 | ✓ |
| Within-domain | In-domain | DTW | 29.624 | 0.0000 | ✓ |
| Within-domain | In-domain | WASSERSTEIN | 31.475 | 0.0000 | ✓ |
| Within-domain | Out-domain | MMD | 29.451 | 0.0000 | ✓ |
| Within-domain | Out-domain | DTW | 30.040 | 0.0000 | ✓ |
| Within-domain | Out-domain | WASSERSTEIN | 23.413 | 0.0000 | ✓ |
| Mixed | In-domain | MMD | 31.873 | 0.0000 | ✓ |
| Mixed | In-domain | DTW | 31.704 | 0.0000 | ✓ |
| Mixed | In-domain | WASSERSTEIN | 32.009 | 0.0000 | ✓ |
| Mixed | Out-domain | MMD | 23.994 | 0.0000 | ✓ |
| Mixed | Out-domain | DTW | 24.574 | 0.0000 | ✓ |
| Mixed | Out-domain | WASSERSTEIN | 24.691 | 0.0000 | ✓ |

**Summary**: 17/18 cells significant at α=0.05; 16/18 after Bonferroni correction (α'=0.0028).

### 3.3 Pairwise Comparisons (Mann-Whitney U)

Baseline vs each method, testing:

$$H_0: F_{\text{baseline}}(x) = F_{\text{method}}(x)$$
$$H_1: F_{\text{baseline}}(x) \neq F_{\text{method}}(x)$$

| Comparison | Mode | Level | Distance | U | p | Cliff's δ | Effect | Mean(method) | Mean(baseline) |
|------------|------|-------|----------|--:|--:|----------:|:------:|-------------:|---------------:|
| rus vs baseline | Cross-domain | In-domain | MMD | 90.0 | 0.0028 | -0.800 | large | 0.5084 | 0.5228 |
| smote vs baseline | Cross-domain | In-domain | MMD | 57.0 | 0.6232 | -0.140 | negligible | 0.5223 | 0.5228 |
| sw_smote vs baseline | Cross-domain | In-domain | MMD | 100.0 | 0.0002 * | -1.000 | large | 0.5095 | 0.5228 |
| rus vs baseline | Cross-domain | In-domain | DTW | 95.0 | 0.0008 * | -0.900 | large | 0.5076 | 0.5186 |
| smote vs baseline | Cross-domain | In-domain | DTW | 100.0 | 0.0002 * | -1.000 | large | 0.5088 | 0.5186 |
| sw_smote vs baseline | Cross-domain | In-domain | DTW | 43.0 | 0.6232 | +0.140 | negligible | 0.5202 | 0.5186 |
| rus vs baseline | Cross-domain | In-domain | WASSERSTEIN | 46.0 | 0.7913 | +0.080 | negligible | 0.5291 | 0.5282 |
| smote vs baseline | Cross-domain | In-domain | WASSERSTEIN | 100.0 | 0.0002 * | -1.000 | large | 0.5147 | 0.5282 |
| sw_smote vs baseline | Cross-domain | In-domain | WASSERSTEIN | 1.0 | 0.0002 * | +0.980 | large | 0.5473 | 0.5282 |
| rus vs baseline | Cross-domain | Out-domain | MMD | 59.0 | 0.5205 | -0.180 | small | 0.5418 | 0.5344 |
| smote vs baseline | Cross-domain | Out-domain | MMD | 94.0 | 0.0010 | -0.880 | large | 0.5255 | 0.5344 |
| sw_smote vs baseline | Cross-domain | Out-domain | MMD | 77.0 | 0.0452 | -0.540 | large | 0.5283 | 0.5344 |
| rus vs baseline | Cross-domain | Out-domain | DTW | 27.0 | 0.0890 | +0.460 | medium | 0.5201 | 0.5139 |
| smote vs baseline | Cross-domain | Out-domain | DTW | 38.0 | 0.3847 | +0.240 | small | 0.5157 | 0.5139 |
| sw_smote vs baseline | Cross-domain | Out-domain | DTW | 56.0 | 0.6776 | -0.120 | negligible | 0.5109 | 0.5139 |
| rus vs baseline | Cross-domain | Out-domain | WASSERSTEIN | 34.0 | 0.2413 | +0.320 | small | 0.5213 | 0.5101 |
| smote vs baseline | Cross-domain | Out-domain | WASSERSTEIN | 10.0 | 0.0028 | +0.800 | large | 0.5247 | 0.5101 |
| sw_smote vs baseline | Cross-domain | Out-domain | WASSERSTEIN | 64.0 | 0.3075 | -0.280 | small | 0.5060 | 0.5101 |
| rus vs baseline | Within-domain | In-domain | MMD | 52.0 | 0.9097 | -0.040 | negligible | 0.6547 | 0.6291 |
| smote vs baseline | Within-domain | In-domain | MMD | 4.0 | 0.0006 * | +0.920 | large | 0.8805 | 0.6291 |
| sw_smote vs baseline | Within-domain | In-domain | MMD | 3.0 | 0.0004 * | +0.940 | large | 0.8707 | 0.6291 |
| rus vs baseline | Within-domain | In-domain | DTW | 26.0 | 0.0757 | +0.480 | large | 0.5789 | 0.5329 |
| smote vs baseline | Within-domain | In-domain | DTW | 0.0 | 0.0002 * | +1.000 | large | 0.8759 | 0.5329 |
| sw_smote vs baseline | Within-domain | In-domain | DTW | 0.0 | 0.0002 * | +1.000 | large | 0.8567 | 0.5329 |
| rus vs baseline | Within-domain | In-domain | WASSERSTEIN | 88.0 | 0.0046 | -0.760 | large | 0.5806 | 0.6181 |
| smote vs baseline | Within-domain | In-domain | WASSERSTEIN | 0.0 | 0.0002 * | +1.000 | large | 0.8851 | 0.6181 |
| sw_smote vs baseline | Within-domain | In-domain | WASSERSTEIN | 0.0 | 0.0002 * | +1.000 | large | 0.8695 | 0.6181 |
| rus vs baseline | Within-domain | Out-domain | MMD | 40.0 | 0.4727 | +0.200 | small | 0.6295 | 0.5917 |
| smote vs baseline | Within-domain | Out-domain | MMD | 0.0 | 0.0002 * | +1.000 | large | 0.8921 | 0.5917 |
| sw_smote vs baseline | Within-domain | Out-domain | MMD | 0.0 | 0.0002 * | +1.000 | large | 0.8848 | 0.5917 |
| rus vs baseline | Within-domain | Out-domain | DTW | 71.0 | 0.1212 | -0.420 | medium | 0.5780 | 0.6111 |
| smote vs baseline | Within-domain | Out-domain | DTW | 0.0 | 0.0002 * | +1.000 | large | 0.8977 | 0.6111 |
| sw_smote vs baseline | Within-domain | Out-domain | DTW | 2.0 | 0.0003 * | +0.960 | large | 0.8582 | 0.6111 |
| rus vs baseline | Within-domain | Out-domain | WASSERSTEIN | 93.0 | 0.0013 | -0.860 | large | 0.5988 | 0.8012 |
| smote vs baseline | Within-domain | Out-domain | WASSERSTEIN | 23.0 | 0.0452 | +0.540 | large | 0.8769 | 0.8012 |
| sw_smote vs baseline | Within-domain | Out-domain | WASSERSTEIN | 26.0 | 0.0757 | +0.480 | large | 0.8711 | 0.8012 |
| rus vs baseline | Mixed | In-domain | MMD | 92.0 | 0.0017 | -0.840 | large | 0.5322 | 0.6155 |
| smote vs baseline | Mixed | In-domain | MMD | 0.0 | 0.0002 * | +1.000 | large | 0.8564 | 0.6155 |
| sw_smote vs baseline | Mixed | In-domain | MMD | 0.0 | 0.0002 * | +1.000 | large | 0.8395 | 0.6155 |
| rus vs baseline | Mixed | In-domain | DTW | 97.0 | 0.0004 * | -0.940 | large | 0.5448 | 0.6877 |
| smote vs baseline | Mixed | In-domain | DTW | 1.0 | 0.0002 * | +0.980 | large | 0.8332 | 0.6877 |
| sw_smote vs baseline | Mixed | In-domain | DTW | 0.0 | 0.0003 * | +1.000 | large | 0.8303 | 0.6877 |
| rus vs baseline | Mixed | In-domain | WASSERSTEIN | 95.0 | 0.0008 * | -0.900 | large | 0.5649 | 0.6769 |
| smote vs baseline | Mixed | In-domain | WASSERSTEIN | 1.0 | 0.0002 * | +0.980 | large | 0.8436 | 0.6769 |
| sw_smote vs baseline | Mixed | In-domain | WASSERSTEIN | 0.0 | 0.0002 * | +1.000 | large | 0.8323 | 0.6769 |
| rus vs baseline | Mixed | Out-domain | MMD | 100.0 | 0.0002 * | -1.000 | large | 0.6164 | 0.8371 |
| smote vs baseline | Mixed | Out-domain | MMD | 26.0 | 0.0757 | +0.480 | large | 0.8886 | 0.8371 |
| sw_smote vs baseline | Mixed | Out-domain | MMD | 34.0 | 0.2413 | +0.320 | small | 0.8739 | 0.8371 |
| rus vs baseline | Mixed | Out-domain | DTW | 100.0 | 0.0002 * | -1.000 | large | 0.5789 | 0.8383 |
| smote vs baseline | Mixed | Out-domain | DTW | 25.0 | 0.0640 | +0.500 | large | 0.8989 | 0.8383 |
| sw_smote vs baseline | Mixed | Out-domain | DTW | 27.0 | 0.0890 | +0.460 | medium | 0.8859 | 0.8383 |
| rus vs baseline | Mixed | Out-domain | WASSERSTEIN | 100.0 | 0.0002 * | -1.000 | large | 0.5852 | 0.8446 |
| smote vs baseline | Mixed | Out-domain | WASSERSTEIN | 24.0 | 0.0539 | +0.520 | large | 0.9033 | 0.8446 |
| sw_smote vs baseline | Mixed | Out-domain | WASSERSTEIN | 28.0 | 0.1041 | +0.440 | medium | 0.8878 | 0.8446 |

**Bonferroni threshold**: α'=0.00093 (m=54). **26** comparisons significant after correction.

### 3.4 Paired Comparison (Wilcoxon Signed-Rank)

Paired by seed — more powerful than Mann-Whitney when seeds are shared.

$$W = \sum_{i=1}^{n} \text{sign}(d_i) \cdot R_i, \quad d_i = Y_{\text{method},i} - Y_{\text{baseline},i}$$

| Comparison | Mode | Level | Distance | W | p | Cliff's δ | Effect | n |
|------------|------|-------|----------|--:|--:|----------:|:------:|--:|
| rus vs baseline | Cross-domain | In-domain | MMD | 3.0 | 0.0098 | -0.800 | large | 10 |
| smote vs baseline | Cross-domain | In-domain | MMD | 25.0 | 0.8457 | -0.140 | negligible | 10 |
| sw_smote vs baseline | Cross-domain | In-domain | MMD | 0.0 | 0.0020 | -1.000 | large | 10 |
| rus vs baseline | Cross-domain | In-domain | DTW | 0.0 | 0.0020 | -0.900 | large | 10 |
| smote vs baseline | Cross-domain | In-domain | DTW | 0.0 | 0.0020 | -1.000 | large | 10 |
| sw_smote vs baseline | Cross-domain | In-domain | DTW | 23.0 | 0.6953 | +0.140 | negligible | 10 |
| rus vs baseline | Cross-domain | In-domain | WASSERSTEIN | 25.0 | 0.8457 | +0.080 | negligible | 10 |
| smote vs baseline | Cross-domain | In-domain | WASSERSTEIN | 0.0 | 0.0020 | -1.000 | large | 10 |
| sw_smote vs baseline | Cross-domain | In-domain | WASSERSTEIN | 0.0 | 0.0020 | +0.980 | large | 10 |
| rus vs baseline | Cross-domain | Out-domain | MMD | 25.0 | 0.8457 | -0.180 | small | 10 |
| smote vs baseline | Cross-domain | Out-domain | MMD | 2.0 | 0.0059 | -0.880 | large | 10 |
| sw_smote vs baseline | Cross-domain | Out-domain | MMD | 7.0 | 0.0371 | -0.540 | large | 10 |
| rus vs baseline | Cross-domain | Out-domain | DTW | 8.0 | 0.0488 | +0.460 | medium | 10 |
| smote vs baseline | Cross-domain | Out-domain | DTW | 21.0 | 0.5566 | +0.240 | small | 10 |
| sw_smote vs baseline | Cross-domain | Out-domain | DTW | 21.0 | 0.5566 | -0.120 | negligible | 10 |
| rus vs baseline | Cross-domain | Out-domain | WASSERSTEIN | 19.0 | 0.4316 | +0.320 | small | 10 |
| smote vs baseline | Cross-domain | Out-domain | WASSERSTEIN | 3.0 | 0.0098 | +0.800 | large | 10 |
| sw_smote vs baseline | Cross-domain | Out-domain | WASSERSTEIN | 14.0 | 0.1934 | -0.280 | small | 10 |
| rus vs baseline | Within-domain | In-domain | MMD | 22.0 | 0.6250 | -0.040 | negligible | 10 |
| smote vs baseline | Within-domain | In-domain | MMD | 0.0 | 0.0020 | +0.920 | large | 10 |
| sw_smote vs baseline | Within-domain | In-domain | MMD | 0.0 | 0.0020 | +0.940 | large | 10 |
| rus vs baseline | Within-domain | In-domain | DTW | 15.0 | 0.2324 | +0.480 | large | 10 |
| smote vs baseline | Within-domain | In-domain | DTW | 0.0 | 0.0020 | +1.000 | large | 10 |
| sw_smote vs baseline | Within-domain | In-domain | DTW | 0.0 | 0.0020 | +1.000 | large | 10 |
| rus vs baseline | Within-domain | In-domain | WASSERSTEIN | 3.0 | 0.0098 | -0.760 | large | 10 |
| smote vs baseline | Within-domain | In-domain | WASSERSTEIN | 0.0 | 0.0020 | +1.000 | large | 10 |
| sw_smote vs baseline | Within-domain | In-domain | WASSERSTEIN | 0.0 | 0.0020 | +1.000 | large | 10 |
| rus vs baseline | Within-domain | Out-domain | MMD | 13.0 | 0.1602 | +0.200 | small | 10 |
| smote vs baseline | Within-domain | Out-domain | MMD | 0.0 | 0.0020 | +1.000 | large | 10 |
| sw_smote vs baseline | Within-domain | Out-domain | MMD | 0.0 | 0.0020 | +1.000 | large | 10 |
| rus vs baseline | Within-domain | Out-domain | DTW | 14.0 | 0.1934 | -0.420 | medium | 10 |
| smote vs baseline | Within-domain | Out-domain | DTW | 0.0 | 0.0020 | +1.000 | large | 10 |
| sw_smote vs baseline | Within-domain | Out-domain | DTW | 0.0 | 0.0020 | +0.960 | large | 10 |
| rus vs baseline | Within-domain | Out-domain | WASSERSTEIN | 1.0 | 0.0039 | -0.860 | large | 10 |
| smote vs baseline | Within-domain | Out-domain | WASSERSTEIN | 5.0 | 0.0195 | +0.540 | large | 10 |
| sw_smote vs baseline | Within-domain | Out-domain | WASSERSTEIN | 15.0 | 0.2324 | +0.480 | large | 10 |
| rus vs baseline | Mixed | In-domain | MMD | 0.0 | 0.0020 | -0.840 | large | 10 |
| smote vs baseline | Mixed | In-domain | MMD | 0.0 | 0.0020 | +1.000 | large | 10 |
| sw_smote vs baseline | Mixed | In-domain | MMD | 0.0 | 0.0020 | +1.000 | large | 10 |
| rus vs baseline | Mixed | In-domain | DTW | 0.0 | 0.0020 | -0.940 | large | 10 |
| smote vs baseline | Mixed | In-domain | DTW | 0.0 | 0.0020 | +0.980 | large | 10 |
| sw_smote vs baseline | Mixed | In-domain | DTW | 0.0 | 0.0039 | +1.000 | large | 9 |
| rus vs baseline | Mixed | In-domain | WASSERSTEIN | 0.0 | 0.0020 | -0.900 | large | 10 |
| smote vs baseline | Mixed | In-domain | WASSERSTEIN | 0.0 | 0.0020 | +0.980 | large | 10 |
| sw_smote vs baseline | Mixed | In-domain | WASSERSTEIN | 0.0 | 0.0020 | +1.000 | large | 10 |
| rus vs baseline | Mixed | Out-domain | MMD | 0.0 | 0.0020 | -1.000 | large | 10 |
| smote vs baseline | Mixed | Out-domain | MMD | 8.0 | 0.0488 | +0.480 | large | 10 |
| sw_smote vs baseline | Mixed | Out-domain | MMD | 17.0 | 0.3223 | +0.320 | small | 10 |
| rus vs baseline | Mixed | Out-domain | DTW | 0.0 | 0.0020 | -1.000 | large | 10 |
| smote vs baseline | Mixed | Out-domain | DTW | 7.0 | 0.0371 | +0.500 | large | 10 |
| sw_smote vs baseline | Mixed | Out-domain | DTW | 14.0 | 0.1934 | +0.460 | medium | 10 |
| rus vs baseline | Mixed | Out-domain | WASSERSTEIN | 0.0 | 0.0020 | -1.000 | large | 10 |
| smote vs baseline | Mixed | Out-domain | WASSERSTEIN | 6.0 | 0.0273 | +0.520 | large | 10 |
| sw_smote vs baseline | Mixed | Out-domain | WASSERSTEIN | 14.0 | 0.1934 | +0.440 | medium | 10 |

**Bonferroni threshold**: α'=0.00093 (m=54). **0** comparisons significant after correction.

**Note**: With n=10 paired observations, the minimum achievable p-value for Wilcoxon signed-rank is $p_{\min} = 1/2^{10-1} = 0.001953$. When $p_{\min} > \alpha'$, no comparison can reach Bonferroni significance regardless of effect magnitude. This is a **floor effect** of small sample size, not evidence of no difference.

### 3.5 Friedman Test (Repeated-Measures)

Seeds serve as blocks (subjects). Tests whether at least one condition differs.

$$\chi_F^2 = \frac{12}{bk(k+1)} \sum_{j=1}^{k} R_j^2 - 3b(k+1)$$

where $b$ = number of blocks (seeds), $k$ = number of conditions, $R_j$ = rank sum.

| Mode | Level | Distance | χ² | p-value | n_seeds | Sig |
|------|-------|----------|----|--------:|--------:|:---:|
| Cross-domain | In-domain | MMD | 17.640 | 0.0005 | 10 | ✓ |
| Cross-domain | In-domain | DTW | 15.600 | 0.0014 | 10 | ✓ |
| Cross-domain | In-domain | WASSERSTEIN | 17.280 | 0.0006 | 10 | ✓ |
| Cross-domain | Out-domain | MMD | 6.600 | 0.0858 | 10 |  |
| Cross-domain | Out-domain | DTW | 5.880 | 0.1176 | 10 |  |
| Cross-domain | Out-domain | WASSERSTEIN | 11.640 | 0.0087 | 10 | ✓ |
| Within-domain | In-domain | MMD | 24.240 | 0.0000 | 10 | ✓ |
| Within-domain | In-domain | DTW | 24.600 | 0.0000 | 10 | ✓ |
| Within-domain | In-domain | WASSERSTEIN | 25.200 | 0.0000 | 10 | ✓ |
| Within-domain | Out-domain | MMD | 24.240 | 0.0000 | 10 | ✓ |
| Within-domain | Out-domain | DTW | 26.160 | 0.0000 | 10 | ✓ |
| Within-domain | Out-domain | WASSERSTEIN | 19.800 | 0.0002 | 10 | ✓ |
| Mixed | In-domain | MMD | 27.000 | 0.0000 | 10 | ✓ |
| Mixed | In-domain | DTW | 24.600 | 0.0000 | 9 | ✓ |
| Mixed | In-domain | WASSERSTEIN | 27.120 | 0.0000 | 10 | ✓ |
| Mixed | Out-domain | MMD | 21.000 | 0.0001 | 10 | ✓ |
| Mixed | Out-domain | DTW | 19.560 | 0.0002 | 10 | ✓ |
| Mixed | Out-domain | WASSERSTEIN | 20.520 | 0.0001 | 10 | ✓ |

### 3.6 Domain Gap Analysis

Domain gap: $\Delta = Y_{\text{out-domain}} - Y_{\text{in-domain}}$
 (paired by seed).

Negative Δ indicates performance degrades in out-domain (expected for domain shift).

| Mode | Distance | Condition | Mean Δ | SD Δ | n | KW p (across conds) |
|------|----------|-----------|-------:|-----:|--:|--------------------:|
| Cross-domain | MMD | baseline | +0.0117 | 0.0042 | 10 | 0.0003 |
| Cross-domain | MMD | rus | +0.0334 | 0.0248 | 10 | 0.0003 |
| Cross-domain | MMD | smote | +0.0031 | 0.0096 | 10 | 0.0003 |
| Cross-domain | MMD | sw_smote | +0.0188 | 0.0094 | 10 | 0.0003 |
| Cross-domain | DTW | baseline | -0.0047 | 0.0095 | 10 | 0.0001 |
| Cross-domain | DTW | rus | +0.0125 | 0.0047 | 10 | 0.0001 |
| Cross-domain | DTW | smote | +0.0069 | 0.0061 | 10 | 0.0001 |
| Cross-domain | DTW | sw_smote | -0.0093 | 0.0112 | 10 | 0.0001 |
| Cross-domain | WASSERSTEIN | baseline | -0.0181 | 0.0086 | 10 | 0.0001 |
| Cross-domain | WASSERSTEIN | rus | -0.0077 | 0.0409 | 10 | 0.0001 |
| Cross-domain | WASSERSTEIN | smote | +0.0100 | 0.0066 | 10 | 0.0001 |
| Cross-domain | WASSERSTEIN | sw_smote | -0.0413 | 0.0093 | 10 | 0.0001 |
| Within-domain | MMD | baseline | -0.0374 | 0.0888 | 10 | 0.4842 |
| Within-domain | MMD | rus | -0.0252 | 0.1274 | 10 | 0.4842 |
| Within-domain | MMD | smote | +0.0116 | 0.0331 | 10 | 0.4842 |
| Within-domain | MMD | sw_smote | +0.0142 | 0.0570 | 10 | 0.4842 |
| Within-domain | DTW | baseline | +0.0782 | 0.0568 | 10 | 0.0086 |
| Within-domain | DTW | rus | -0.0009 | 0.0824 | 10 | 0.0086 |
| Within-domain | DTW | smote | +0.0218 | 0.0246 | 10 | 0.0086 |
| Within-domain | DTW | sw_smote | +0.0015 | 0.0883 | 10 | 0.0086 |
| Within-domain | WASSERSTEIN | baseline | +0.1832 | 0.1115 | 10 | 0.0049 |
| Within-domain | WASSERSTEIN | rus | +0.0182 | 0.0597 | 10 | 0.0049 |
| Within-domain | WASSERSTEIN | smote | -0.0082 | 0.0388 | 10 | 0.0049 |
| Within-domain | WASSERSTEIN | sw_smote | +0.0016 | 0.0806 | 10 | 0.0049 |
| Mixed | MMD | baseline | +0.2216 | 0.0894 | 10 | 0.0000 |
| Mixed | MMD | rus | +0.0842 | 0.0490 | 10 | 0.0000 |
| Mixed | MMD | smote | +0.0321 | 0.0478 | 10 | 0.0000 |
| Mixed | MMD | sw_smote | +0.0343 | 0.0456 | 10 | 0.0000 |
| Mixed | DTW | baseline | +0.1506 | 0.0948 | 10 | 0.0212 |
| Mixed | DTW | rus | +0.0341 | 0.0537 | 10 | 0.0212 |
| Mixed | DTW | smote | +0.0657 | 0.0414 | 10 | 0.0212 |
| Mixed | DTW | sw_smote | +0.0550 | 0.0449 | 9 | 0.0212 |
| Mixed | WASSERSTEIN | baseline | +0.1677 | 0.0939 | 10 | 0.0015 |
| Mixed | WASSERSTEIN | rus | +0.0204 | 0.0499 | 10 | 0.0015 |
| Mixed | WASSERSTEIN | smote | +0.0597 | 0.0443 | 10 | 0.0015 |
| Mixed | WASSERSTEIN | sw_smote | +0.0555 | 0.0452 | 10 | 0.0015 |

### 3.7 Condition × Distance Interaction

Does the best-performing condition depend on which distance metric is used?

| Mode | Level | Distance | Best Condition | Mean AUROC | Consistent? |
|------|-------|----------|----------------|---:|:-----------:|
| Cross-domain | In-domain | MMD | baseline | 0.5228 | ✗ |
| Cross-domain | In-domain | DTW | sw_smote | 0.5202 | ✗ |
| Cross-domain | In-domain | WASSERSTEIN | sw_smote | 0.5473 | ✗ |
| Cross-domain | Out-domain | MMD | rus | 0.5418 | ✗ |
| Cross-domain | Out-domain | DTW | rus | 0.5201 | ✗ |
| Cross-domain | Out-domain | WASSERSTEIN | smote | 0.5247 | ✗ |
| Within-domain | In-domain | MMD | smote | 0.8805 | ✓ |
| Within-domain | In-domain | DTW | smote | 0.8759 | ✓ |
| Within-domain | In-domain | WASSERSTEIN | smote | 0.8851 | ✓ |
| Within-domain | Out-domain | MMD | smote | 0.8921 | ✓ |
| Within-domain | Out-domain | DTW | smote | 0.8977 | ✓ |
| Within-domain | Out-domain | WASSERSTEIN | smote | 0.8769 | ✓ |
| Mixed | In-domain | MMD | smote | 0.8564 | ✓ |
| Mixed | In-domain | DTW | smote | 0.8332 | ✓ |
| Mixed | In-domain | WASSERSTEIN | smote | 0.8436 | ✓ |
| Mixed | Out-domain | MMD | smote | 0.8886 | ✓ |
| Mixed | Out-domain | DTW | smote | 0.8989 | ✓ |
| Mixed | Out-domain | WASSERSTEIN | smote | 0.9033 | ✓ |

### 3.8 Overall Condition Ranking

Mean rank across all 18 cells (3 modes × 2 levels × 3 distances). Rank 1 = best.

| Rank | Condition | Mean Rank |
|:----:|-----------|----------:|
| 1 | smote | 1.56 |
| 2 | sw_smote | 2.22 |
| 3 | baseline | 2.94 |
| 4 | rus | 3.28 |

---
## 4. Cross-Metric Synthesis

### 4.1 Overall Rankings Comparison

| Condition | Mean Rank (F2) | Mean Rank (AUROC) | Average |
|-----------|:--------------:|:-----------------:|:-------:|
| baseline | 2.44 | 2.94 | 2.69 |
| rus | 3.22 | 3.28 | 3.25 |
| smote | 1.78 | 1.56 | 1.67 |
| sw_smote | 2.56 | 2.22 | 2.39 |

### 4.2 Summary of Significant Findings

| Test | Metric | Sig/Total (raw α=0.05) | Sig/Total (Bonferroni) |
|------|--------|:----------------------:|:----------------------:|
| Kruskal-Wallis | F2 | 18/18 | 18/18 |
| Kruskal-Wallis | AUROC | 17/18 | 16/18 |
| Mann-Whitney U | F2 | 44/54 | 36/54 |
| Mann-Whitney U | AUROC | 34/54 | 26/54 |
| Wilcoxon SR | F2 | 45/54 | 0/54 |
| Wilcoxon SR | AUROC | 38/54 | 0/54 |

### 4.3 Effect Size Summary (Cliff's δ)

Distribution of effect sizes across all pairwise comparisons:

**F2**:
  - negligible: 1/54 (2%)
  - small: 6/54 (11%)
  - medium: 1/54 (2%)
  - large: 46/54 (85%)

**AUROC**:
  - negligible: 5/54 (9%)
  - small: 6/54 (11%)
  - medium: 4/54 (7%)
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
