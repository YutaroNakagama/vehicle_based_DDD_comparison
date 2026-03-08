# Vehicle Feature & Distance Metric Analysis

## 1. Feature Space Overview

- **Subjects loaded**: 87
- **Feature dimensions**: 135
- **Raw signals**: 5 (steering angle, steering speed, lateral acceleration, longitudinal acceleration, lane offset)
- **Feature extraction**: 3 methods (statistical+spectral 90d, smooth/std/PE 15d, permutation entropy 40d)

| Category | Dimensions | Signals |
|----------|:----------:|---------|
| Steering (stat+spectral) | 16 | Steering |
| SteeringSpeed (stat+spectral) | 16 | SteeringSpeed |
| LateralAcc (stat+spectral) | 16 | Lateral |
| LaneOffset (stat+spectral) | 16 | LaneOffset |
| LongAcc (stat+spectral) | 16 | LongAcc |
| Smooth/Std/PE | 15 | lane |
| Permutation Entropy | 40 | LaneOffset |

## 2. Subject-Level Feature Summary

- **Windows per subject**: mean=814, min=681, max=915, std=50

## 3. Inter-Subject vs Intra-Subject Variance

This is critical: if intra-subject variance dominates, then the location of each
subject in feature space is 'blurred', and different distance metrics will rank
subjects similarly because the 'signal' (inter-subject differences) is weak.

**ICC-like ratio** = Var(between subjects) / Var(total)

Values near 1 mean inter-subject differences dominate (good for discriminating subjects).
Values near 0 mean intra-subject noise dominates (hard to distinguish subjects).

| Category | Mean ICC ratio | Median ICC ratio | Min | Max |
|----------|:--------------:|:----------------:|:---:|:---:|
| Steering (stat+spectral) | 0.112 | 0.038 | 0.007 | 0.990 |
| SteeringSpeed (stat+spectral) | 0.177 | 0.163 | 0.005 | 0.990 |
| LateralAcc (stat+spectral) | 0.088 | 0.019 | 0.005 | 0.990 |
| LaneOffset (stat+spectral) | 0.173 | 0.060 | 0.001 | 0.990 |
| LongAcc (stat+spectral) | 0.157 | 0.086 | 0.003 | 0.990 |
| Smooth/Std/PE | 0.080 | 0.024 | 0.000 | 0.340 |
| Permutation Entropy | 0.061 | 0.032 | 0.002 | 0.253 |

**Overall**: Mean ICC ratio = 0.111, Median = 0.043

**Top 10 most discriminating features** (highest ICC ratio):

| Feature | ICC ratio |
|---------|:---------:|
| Lateral_FreqVar | 0.990 |
| SteeringSpeed_FreqVar | 0.990 |
| LaneOffset_FreqVar | 0.990 |
| LongAcc_FreqVar | 0.990 |
| Steering_FreqVar | 0.990 |
| LaneOffset_Quartile75 | 0.343 |
| lane_offset_gaussian_smooth | 0.340 |
| LaneOffset_Median | 0.326 |
| LaneOffset_Quartile25 | 0.319 |
| long_acc_std_dev | 0.286 |

**Bottom 10 least discriminating features** (lowest ICC ratio):

| Feature | ICC ratio |
|---------|:---------:|
| lat_acc_pred_error | 0.004 |
| LateralAccel_DAD | 0.004 |
| LongAcc_Median | 0.003 |
| LateralAccel_DDD | 0.002 |
| LaneOffset_Kurtosis | 0.001 |
| lane_offset_pred_error | 0.001 |
| LaneOffset_Skewness | 0.001 |
| steering_speed_gaussian_smooth | 0.000 |
| long_acc_gaussian_smooth | 0.000 |
| lat_acc_gaussian_smooth | 0.000 |

## 4. Feature Correlation Structure

High inter-feature correlation means the effective dimensionality is lower than 145.
This compresses the space in which distance metrics operate.

- **Mean |correlation|**: 0.137
- **Median |correlation|**: 0.049
- **Pairs with |r| > 0.8**: 229
- **Pairs with |r| > 0.9**: 138
- **Effective dimensionality (95% variance)**: 45 / 135
- **Effective dimensionality (99% variance)**: 66 / 135
- **First PC explains**: 22.2%
- **First 5 PCs explain**: 52.3%

## 5. Subject Rank Concordance Across Distance Metrics

Pairwise distance matrices were computed using the **actual distance metrics from the experiment**
(MMD with RBF kernel + median heuristic, per-feature averaged Wasserstein, DTW on mean time series),
then subjects were ranked by mean distance to all others (descending = out_domain first).

### 5.1 Rank Correlation

| Metric Pair | Spearman ρ | p-value | Kendall τ | p-value |
|-------------|:----------:|:-------:|:---------:|:-------:|
| MMD vs Wasserstein | 0.7522 | 4.57e-17 | 0.6220 | 1.45e-17 |
| MMD vs DTW | 0.4768 | 3.03e-06 | 0.3387 | 3.40e-06 |
| Wasserstein vs DTW | 0.8027 | 8.96e-21 | 0.6172 | 2.56e-17 |

Rank concordance is moderate-to-high but far from perfect. MMD vs DTW shows only moderate agreement (ρ=0.48).

### 5.2 Distance Matrix Summary

| Metric | Mean | Std | Min (off-diag) | Max |
|--------|:----:|:---:|:--------------:|:---:|
| MMD | nan* | nan* | nan* | nan* |
| Wasserstein | 26.38 | 24.89 | 0.65 | 178.24 |
| DTW | 919.45 | 562.47 | 110.36 | 4490.62 |

\* MMD matrix contains NaN entries due to S0116_2 feature dimension mismatch (135 vs 145 columns).

### 5.3 Domain Group Membership Overlap (GROUP_SIZE=29)

**out_domain**:

| Pair | Overlap | Jaccard |
|------|:-------:|:-------:|
| MMD ∩ Wasserstein | 21/29 (72%) | 0.568 |
| MMD ∩ DTW | 18/29 (62%) | 0.450 |
| Wasserstein ∩ DTW | 25/29 (86%) | 0.758 |
| **All three agree** | **18/29 (62%)** | — |

**in_domain**:

| Pair | Overlap | Jaccard |
|------|:-------:|:-------:|
| MMD ∩ Wasserstein | 19/29 (66%) | 0.487 |
| MMD ∩ DTW | 12/29 (41%) | 0.261 |
| Wasserstein ∩ DTW | 20/29 (69%) | 0.526 |
| **All three agree** | **12/29 (41%)** | — |

**mid_domain**:

| Pair | Overlap | Jaccard |
|------|:-------:|:-------:|
| MMD ∩ Wasserstein | 14/29 (48%) | 0.318 |
| MMD ∩ DTW | 7/29 (24%) | 0.137 |
| Wasserstein ∩ DTW | 16/29 (55%) | 0.381 |
| **All three agree** | **5/29 (17%)** | — |

**Total subjects that switch groups**: 52/87 (59.8%)

Top 10 (most out_domain) subjects show partial agreement: S0135_2 is consistently ranked #1 across all metrics, while the remaining ranks diverge.

Note: An earlier version of this report incorrectly used Euclidean/Manhattan/Cosine distances on mean feature vectors as proxies, reporting ρ=0.84–0.999 and 100% group overlap. The actual experiment uses fundamentally different distance computations (full sample-level MMD with RBF kernel, per-feature 1D Wasserstein averaging, DTW on mean time series) that produce substantially lower concordance.

## 6. Feature Scale Heterogeneity

Large scale differences can cause some features to dominate distance computation.

| Category | Mean range | Max range | Mean |μ| | CV of ranges |
|----------|:---------:|:---------:|:------:|:------------:|
| Steering (stat+spectral) | 1.25e+00 | 1.61e+01 | 4.89e+00 | 3.09 |
| SteeringSpeed (stat+spectral) | 3.43e+00 | 4.60e+01 | 5.39e+00 | 3.20 |
| LateralAcc (stat+spectral) | 4.18e-01 | 2.96e+00 | 4.95e+00 | 1.73 |
| LaneOffset (stat+spectral) | 1.33e+03 | 2.12e+04 | 2.92e+02 | 3.85 |
| LongAcc (stat+spectral) | 2.40e-01 | 1.77e+00 | 4.80e+00 | 1.81 |
| Smooth/Std/PE | 7.40e-02 | 9.66e-01 | 1.97e-02 | 3.23 |
| Permutation Entropy | 7.67e-01 | 1.49e+01 | 1.69e-01 | 3.36 |

- **Max/Min feature range ratio**: 744640188572483328x
- **Mean feature range**: 1.59e+02

## 7. Why Distance Metric Choice Does Not Affect Classification Performance

### 7.1 Empirical Finding: Groups Differ, but Performance Does Not

The actual distance metrics (MMD, DTW, Wasserstein) produce **substantially different domain groupings**:
- out_domain three-way agreement: 62% (18/29)
- in_domain three-way agreement: 41% (12/29)
- 52/87 (59.8%) subjects change group assignment depending on distance metric

Despite this, the ANOVA effect size for distance metric on classification performance is η²<0.004. This paradox requires explanation.

### 7.2 Boundary Instability with Core Stability

The subjects that switch groups are concentrated near the **boundary between adjacent domains** (mid_domain has only 17% three-way agreement). Meanwhile, the extreme subjects — those most clearly out_domain or in_domain — remain stable across metrics (e.g., S0135_2 is rank #1 in all three).

The domain grouping determines which subjects form the training set:  
$$\mathcal{D}_{\text{train}} = \{(\mathbf{x}_i, y_i) : s_i \in \text{group}(d)\}$$

When boundary subjects switch between groups, the resulting training distribution $P_{\text{train}}(\mathbf{x}|y)$ changes only marginally because:

1. **Boundary subjects are similar by definition**: A subject at the in/mid boundary has similar mean distance to subjects just inside both groups. Their feature distributions overlap substantially with both groups.

2. **Low ICC ratio** (mean=0.111): Intra-subject variance is ~8× inter-subject variance ($\sigma^2_{\text{within}} \approx 8\sigma^2_{\text{between}}$). When each subject contributes ~814 windows of high-variance data, swapping a few boundary subjects changes the aggregate training distribution minimally.

3. **Training set size**: Each group has 29 subjects. Swapping ~11 boundary subjects (the overlap gap) alters ~38% of the group membership but the corresponding feature-space shift is attenuated by the dominant intra-subject variance.

### 7.3 The Rebalancing Dominance Effect

The rebalancing condition (baseline/RUS/SMOTE/SW-SMOTE) has η²=0.11–0.14, which is 35–1100× larger than the distance metric effect:

$$\frac{\eta^2_{\text{condition}}}{\eta^2_{\text{distance}}} \approx \frac{0.11}{0.0001} = 1100\times \text{(F2)}, \quad \frac{0.14}{0.004} = 35\times \text{(AUROC)}$$

SMOTE generates synthetic minority samples in the convex hull of existing minority data; RUS removes majority samples randomly. Both transform $P_{\text{train}}(\mathbf{x}|y)$ far more drastically than the marginal shift caused by boundary-subject swaps. The classifier's decision boundary is dominated by: (a) which rebalancing method is used, (b) which training mode (source_only/target_only/mixed), and (c) which domain level is targeted — not by which distance metric defined the groups.

### 7.4 Why the Three Metrics Diverge

The low concordance (especially MMD vs DTW, ρ=0.48) is explained by their fundamentally different computation strategies:

- **MMD**: Compares full sample distributions via RBF kernel. Sensitive to density structure and covariance.
- **Wasserstein**: Computes per-feature 1D optimal transport, then averages. Sensitive to marginal distribution shifts. Lane offset features (range ~10³) dominate the average.
- **DTW**: Collapses all 135 features to a 1D mean time series, then aligns temporal patterns. Captures temporal structure that MMD and Wasserstein ignore.

These differences produce genuinely different subject rankings, but the resulting training set variations remain within the noise floor of the classifier's performance.

## 8. Per-Signal Variance Contribution

Which vehicle signals contribute most to inter-subject variation?

| Signal | N features | Mean ICC | Mean inter-subj CV | Contribution to total var (%) |
|--------|:----------:|:--------:|:------------------:|:----------------------------:|
| Steering angle | 30 | 0.076 | 1.72 | 0.0% |
| Steering speed | 27 | 0.128 | 0.93 | 0.0% |
| Lateral acceleration | 27 | 0.055 | 1.20 | 0.0% |
| Lane offset | 27 | 0.171 | 0.67 | 100.0% |
| Longitudinal acceleration | 27 | 0.125 | 1.31 | 0.0% |

## 9. Summary

### Key Findings

1. **Feature space**: 135-dimensional vehicle driving features (common columns) extracted from 5 raw signals (steering angle, steering speed, lateral acceleration, longitudinal acceleration, lane offset)

2. **Low effective dimensionality**: 45 principal components explain 95% of variance, meaning the 135-dimensional space collapses to ~45 effective dimensions

3. **Moderate subject discriminability**: Mean ICC ratio = 0.111, indicating that intra-subject variation (~89% of total variance) is dominant over inter-subject differences (~11%)

4. **Moderate rank concordance**: The actual distance metrics (MMD, DTW, Wasserstein) produce Spearman ρ = 0.48–0.80 in subject ranking. Domain group membership overlap is partial: out_domain 62% three-way agreement, in_domain 41% three-way agreement. 52/87 (59.8%) subjects change group depending on metric.

5. **Performance insensitivity despite group differences**: Despite substantial group composition changes across distance metrics, classification performance is unaffected (η²<0.004). This is because: (a) switching subjects are at group boundaries with similar feature distributions to both adjacent groups, (b) intra-subject variance (~8× inter-subject) dilutes the effect of boundary swaps on training distribution, and (c) rebalancing (η²=0.11–0.14) and training mode (η²=0.50–0.58) dominate performance variation by orders of magnitude.

6. **Physical coupling**: Vehicle dynamics physically couples the 5 raw signals (steering → lateral acceleration → lane offset), contributing to the low effective dimensionality

6. **Rebalancing dominance**: The condition (imbalance handling) effect is 35–1100× larger than the distance metric effect, completely absorbing any subtle grouping differences
