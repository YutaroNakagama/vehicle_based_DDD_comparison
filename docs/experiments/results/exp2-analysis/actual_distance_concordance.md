# Domain Group Concordance: MMD vs DTW vs Wasserstein

**Subjects**: 87  
**GROUP_SIZE**: 29  
**Feature dimensions**: 145

## 1. Distance Matrix Summary

| Metric | Mean | Std | Min (off-diag) | Max |
|--------|:----:|:---:|:--------------:|:---:|
| MMD | nan | nan | nan | nan |
| WASSERSTEIN | 26.3796 | 24.8939 | 0.6544 | 178.2417 |
| DTW | 919.4495 | 562.4749 | 110.3643 | 4490.6172 |

## 2. Subject Rank Concordance (Spearman ρ)

Ranking = mean distance to all other subjects (descending = out_domain first).

| Metric Pair | Spearman ρ | p-value |
|-------------|:----------:|:-------:|
| MMD vs WASSERSTEIN | 0.7522 | 4.57e-17 |
| MMD vs DTW | 0.4768 | 3.03e-06 |
| WASSERSTEIN vs DTW | 0.8027 | 8.96e-21 |

## 3. Kendall τ Rank Correlation

| Metric Pair | Kendall τ | p-value |
|-------------|:---------:|:-------:|
| MMD vs WASSERSTEIN | 0.6220 | 1.45e-17 |
| MMD vs DTW | 0.3387 | 3.40e-06 |
| WASSERSTEIN vs DTW | 0.6172 | 2.56e-17 |

## 4. Domain Group Membership Overlap

Each group has 29 subjects.

### out_domain

| Pair | Overlap | Jaccard |
|------|:-------:|:-------:|
| MMD ∩ WASSERSTEIN | 21/29 (72%) | 0.568 |
| MMD ∩ DTW | 18/29 (62%) | 0.450 |
| WASSERSTEIN ∩ DTW | 25/29 (86%) | 0.758 |

**All three agree**: 18/29 (62%)

### mid_domain

| Pair | Overlap | Jaccard |
|------|:-------:|:-------:|
| MMD ∩ WASSERSTEIN | 14/29 (48%) | 0.318 |
| MMD ∩ DTW | 7/29 (24%) | 0.137 |
| WASSERSTEIN ∩ DTW | 16/29 (55%) | 0.381 |

**All three agree**: 5/29 (17%)

### in_domain

| Pair | Overlap | Jaccard |
|------|:-------:|:-------:|
| MMD ∩ WASSERSTEIN | 19/29 (66%) | 0.487 |
| MMD ∩ DTW | 12/29 (41%) | 0.261 |
| WASSERSTEIN ∩ DTW | 20/29 (69%) | 0.526 |

**All three agree**: 12/29 (41%)

## 5. Mean Distance by Domain Group

Do in_domain subjects have lower mean distance than out_domain subjects? (Sanity check)

| Metric | out_domain mean_d | mid_domain mean_d | in_domain mean_d | out/in ratio |
|--------|:-----------------:|:-----------------:|:----------------:|:------------:|
| MMD | 0.1544 | 0.0810 | nan | inf |
| WASSERSTEIN | 39.1668 | 21.1708 | 18.8013 | 2.08 |
| DTW | 1234.3123 | 802.2778 | 721.7582 | 1.71 |

## 6. Subjects That Switch Groups

Subjects assigned to different domain groups depending on distance metric.

**Total subjects that switch**: 52/87

| Subject | MMD | Wasserstein | DTW |
|---------|-----|-------------|-----|
| S0139_1 | in_domain | in_domain | mid_domain |
| S0116_2 | in_domain | out_domain | mid_domain |
| S0204_1 | mid_domain | mid_domain | in_domain |
| S0148_1 | in_domain | in_domain | mid_domain |
| S0171_1 | mid_domain | mid_domain | in_domain |
| S0197_1 | out_domain | mid_domain | mid_domain |
| S0134_1 | mid_domain | mid_domain | in_domain |
| S0196_1 | mid_domain | in_domain | in_domain |
| S0140_1 | mid_domain | out_domain | out_domain |
| S0181_1 | mid_domain | in_domain | mid_domain |
| S0174_2 | mid_domain | mid_domain | in_domain |
| S0113_1 | mid_domain | in_domain | in_domain |
| S0181_2 | mid_domain | in_domain | mid_domain |
| S0155_1 | mid_domain | in_domain | in_domain |
| S0171_2 | mid_domain | mid_domain | in_domain |
| S0172_1 | mid_domain | mid_domain | in_domain |
| S0202_1 | mid_domain | in_domain | in_domain |
| S0153_2 | out_domain | mid_domain | mid_domain |
| S0204_2 | out_domain | mid_domain | mid_domain |
| S0170_1 | out_domain | mid_domain | mid_domain |
| S0194_1 | out_domain | out_domain | mid_domain |
| S0207_1 | out_domain | mid_domain | mid_domain |
| S0207_2 | mid_domain | mid_domain | in_domain |
| S0198_2 | mid_domain | out_domain | out_domain |
| S0206_1 | in_domain | mid_domain | out_domain |
| S0199_1 | mid_domain | out_domain | out_domain |
| S0165_1 | out_domain | mid_domain | mid_domain |
| S0209_1 | mid_domain | in_domain | in_domain |
| S0172_2 | in_domain | mid_domain | out_domain |
| S0210_2 | in_domain | in_domain | mid_domain |
| S0200_1 | mid_domain | in_domain | in_domain |
| S0193_1 | in_domain | in_domain | mid_domain |
| S0210_1 | mid_domain | in_domain | in_domain |
| S0213_2 | out_domain | out_domain | mid_domain |
| S0205_2 | out_domain | mid_domain | mid_domain |
| S0199_2 | mid_domain | out_domain | out_domain |
| S0156_2 | mid_domain | out_domain | out_domain |
| S0206_2 | in_domain | in_domain | mid_domain |
| S0209_2 | mid_domain | mid_domain | in_domain |
| S0173_2 | out_domain | mid_domain | mid_domain |
| S0169_1 | in_domain | mid_domain | mid_domain |
| S0148_2 | in_domain | in_domain | mid_domain |
| S0200_2 | in_domain | out_domain | out_domain |
| S0194_2 | in_domain | out_domain | out_domain |
| S0147_2 | in_domain | mid_domain | mid_domain |
| S0185_2 | in_domain | mid_domain | out_domain |
| S0139_2 | in_domain | mid_domain | mid_domain |
| S0154_2 | in_domain | in_domain | mid_domain |
| S0169_2 | in_domain | mid_domain | out_domain |
| S0154_1 | mid_domain | mid_domain | in_domain |
| S0153_1 | out_domain | out_domain | mid_domain |
| S0190_1 | mid_domain | in_domain | in_domain |

## 7. Rank Comparison: Top 10 (most out_domain) and Bottom 10 (most in_domain)

### Top 10 (highest mean distance)

| Rank | MMD | Wasserstein | DTW |
|:----:|-----|-------------|-----|
| 1 | S0135_2 | S0135_2 | S0135_2 |
| 2 | S0120_2 | S0120_2 | S0198_1 |
| 3 | S0213_2 | S0198_1 | S0120_2 |
| 4 | S0167_1 | S0167_1 | S0198_2 |
| 5 | S0178_1 | S0116_2 | S0120_1 |
| 6 | S0178_2 | S0178_2 | S0186_2 |
| 7 | S0168_2 | S0178_1 | S0167_1 |
| 8 | S0168_1 | S0186_2 | S0178_2 |
| 9 | S0120_1 | S0120_1 | S0194_2 |
| 10 | S0188_1 | S0188_1 | S0201_2 |

### Bottom 10 (lowest mean distance)

| Rank | MMD | Wasserstein | DTW |
|:----:|-----|-------------|-----|
| 78 | S0134_2 | S0154_2 | S0189_2 |
| 79 | S0201_1 | S0196_2 | S0134_2 |
| 80 | S0206_1 | S0190_2 | S0185_1 |
| 81 | S0139_1 | S0189_2 | S0189_1 |
| 82 | S0185_2 | S0202_2 | S0156_1 |
| 83 | S0139_2 | S0156_1 | S0202_2 |
| 84 | S0156_1 | S0139_1 | S0136_1 |
| 85 | S0190_2 | S0134_2 | S0136_2 |
| 86 | S0196_2 | S0201_1 | S0210_1 |
| 87 | S0116_2 | S0193_2 | S0207_2 |

## 8. Summary

1. **Rank concordance**: Spearman ρ ranges from 0.4768 to 0.8027 across the three actual distance metrics (MMD, DTW, Wasserstein).

2. **out_domain overlap**: 18/29 (62%) subjects are classified as out_domain by all three metrics.

3. **in_domain overlap**: 12/29 (41%) subjects are classified as in_domain by all three metrics.

4. **Total group switchers**: 52/87 (59.8%) subjects change domain assignment depending on distance metric.
