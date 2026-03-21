# Normalised vs Unnormalised Distance Analysis

**Subjects**: 87  |  **Group size**: 29  |  **Features**: 135


### Feature Scale Summary (Raw)

- Dimensions: 135
- Std  — min: 0.0000, median: 0.1246, max: 7232.8800, ratio(max/min): 55599708928.9
- Range — min: 0.0000, median: 3.3333, max: 84653.5357, ratio(max/min): 3930718819.5


### Feature Scale Summary (Standardized)

- Dimensions: 135
- Std  — min: 1.0000, median: 1.0000, max: 1.0000, ratio(max/min): 1.0
- Range — min: 5.8683, median: 37.4807, max: 231.2605, ratio(max/min): 39.4

## 1. Distance Matrix Summary

| Normalisation | Metric | Mean | Std | Min | Max |
|:------------:|--------|:----:|:---:|:---:|:---:|
| raw | MMD | 0.1028 | 0.1142 | 0.000261 | 0.5343 |
| raw | WASSERSTEIN | 28.2098 | 26.5796 | 0.701924 | 178.2417 |
| raw | DTW | 986.6567 | 603.6783 | 118.381328 | 4816.4053 |
| standardized | MMD | 0.0782 | 0.0809 | 0.003749 | 0.5311 |
| standardized | WASSERSTEIN | 0.2528 | 0.4006 | 0.049030 | 2.9884 |
| standardized | DTW | 4.5930 | 4.4034 | 1.144920 | 23.3824 |

## 2. Rank Concordance Between Metrics (Spearman ρ)

| Normalisation | Pair | Spearman ρ | p-value |
|:------------:|------|:----------:|:-------:|
| raw | MMD vs WASSERSTEIN | 0.8188 | 3.42e-22 |
| raw | MMD vs DTW | 0.4779 | 2.86e-06 |
| raw | WASSERSTEIN vs DTW | 0.8047 | 5.98e-21 |
| standardized | MMD vs WASSERSTEIN | 0.7565 | 2.39e-17 |
| standardized | MMD vs DTW | 0.1155 | 2.87e-01 |
| standardized | WASSERSTEIN vs DTW | 0.4442 | 1.64e-05 |

## 3. Group Overlap: Standardized vs Raw (Same Metric)

How many subjects stay in the same domain group after normalisation?

| Metric | Domain | Overlap | Jaccard |
|--------|--------|:-------:|:-------:|
| MMD | out_domain | 18/29 (62%) | 0.450 |
| MMD | mid_domain | 9/29 (31%) | 0.184 |
| MMD | in_domain | 13/29 (45%) | 0.289 |
| WASSERSTEIN | out_domain | 14/29 (48%) | 0.318 |
| WASSERSTEIN | mid_domain | 12/29 (41%) | 0.261 |
| WASSERSTEIN | in_domain | 13/29 (45%) | 0.289 |
| DTW | out_domain | 10/29 (34%) | 0.208 |
| DTW | mid_domain | 9/29 (31%) | 0.184 |
| DTW | in_domain | 9/29 (31%) | 0.184 |

## 4. Cross-Metric Group Overlap (Standardized)

Do metrics DIFFERENTIATE after normalisation?

| Domain | Pair | Overlap | Jaccard |
|--------|------|:-------:|:-------:|
| out_domain | MMD vs WASSERSTEIN | 21/29 (72%) | 0.568 |
| out_domain | MMD vs DTW | 11/29 (38%) | 0.234 |
| out_domain | WASSERSTEIN vs DTW | 18/29 (62%) | 0.450 |
| in_domain | MMD vs WASSERSTEIN | 19/29 (66%) | 0.487 |
| in_domain | MMD vs DTW | 10/29 (34%) | 0.208 |
| in_domain | WASSERSTEIN vs DTW | 13/29 (45%) | 0.289 |

## 5. Subject Switching Summary

- **raw**: 50/87 subjects switch groups across metrics (57.5%)
- **standardized**: 60/87 subjects switch groups across metrics (69.0%)

## 6. Interpretation

- Raw ρ range: 0.478–0.819
- Standardized ρ range: 0.115–0.756
- Mean ρ change: -0.262

**Normalisation REDUCES cross-metric concordance** → metrics capture different distributional properties when lane offset dominance is removed. The H3 finding is partially explained by feature scale heterogeneity.
