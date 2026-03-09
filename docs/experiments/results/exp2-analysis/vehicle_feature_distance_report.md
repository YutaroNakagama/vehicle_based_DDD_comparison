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

We compute pairwise distances between subjects using 3 approaches on the mean feature vectors,
then compare how similarly they rank subjects by 'mean distance to all others'.

| Metric pair | Spearman ρ |
|-------------|:----------:|
| Euclidean vs Manhattan | 0.9989 |
| Euclidean vs Cosine | 0.8425 |
| Manhattan vs Cosine | 0.8445 |

**Group membership overlap** (top 43 = 'out_domain'):

- Euclidean ∩ Manhattan: 43/43 (100%)
- Euclidean ∩ Cosine: 43/43 (100%)
- Manhattan ∩ Cosine: 43/43 (100%)
- All three agree: 43/43 (100%)

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

## 7. Why Distance Metrics Converge for Vehicle Features

### 7.1 Theoretical Argument

For three distance metrics $d_1, d_2, d_3$ (MMD, DTW, Wasserstein), the domain grouping is determined by the **rank ordering** of subjects by mean distance:

$$\text{group}(s_i) = \begin{cases} \text{out\_domain} & \text{if } \text{rank}(\bar{d}(s_i, \cdot)) \leq N/2 \\ \text{in\_domain} & \text{otherwise} \end{cases}$$

The groupings are identical when the rank orderings agree. This happens when:

1. **High effective correlation**: The feature space has low effective dimensionality (45 dims explain 95% variance from 135 features), so all distance metrics capture similar geometric structure.

2. **Moderate ICC ratio** (mean=0.111): Inter-subject differences are modest relative to intra-subject variation. This means subject 'positions' in feature space are noisy, and the **coarse binary split** (in/out) is robust to metric choice, even if fine-grained rankings differ.

3. **Vehicle signal redundancy**: Steering angle, steering speed, lateral acceleration, and lane offset are physically coupled through vehicle dynamics (lateral dynamics equation: $a_y = v \cdot \dot{\psi} = v \cdot \dot{\delta} \cdot L^{-1}$). This coupling means the 145 features reduce to a much smaller intrinsic manifold.

### 7.2 The Rebalancing Absorption Effect

When class imbalance handling is applied (SMOTE, RUS, SW-SMOTE), the classifier's decision boundary shifts substantially. The rebalancing effect has η²=0.11–0.14 (11–14% of variance), while the distance metric effect has η²<0.004 (<0.4%). The ratio is:

$$\frac{\eta^2_{\text{condition}}}{\eta^2_{\text{distance}}} \approx \frac{0.11}{0.0001} = 1100\times \text{(F2)}, \quad \frac{0.14}{0.004} = 35\times \text{(AUROC)}$$

Rebalancing changes the training distribution so dramatically that the minor difference in which subjects constitute the training set (due to distance metric choice) is overwhelmed.

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

1. **Feature space**: 145-dimensional vehicle driving features extracted from 5 raw signals (steering angle, steering speed, lateral acceleration, longitudinal acceleration, lane offset)

2. **Low effective dimensionality**: 45 principal components explain 95% of variance, meaning the 145-dimensional space collapses to ~45 effective dimensions

3. **Moderate subject discriminability**: Mean ICC ratio = 0.111, indicating that intra-subject variation is substantial relative to inter-subject differences

4. **High rank concordance**: Different distance metrics produce Spearman ρ = 0.999–0.843 in subject ranking, and domain group membership overlaps significantly

5. **Physical coupling**: Vehicle dynamics physically couples the 5 raw signals (steering → lateral acceleration → lane offset), further reducing the effective independent information available for distinguishing distance metrics

6. **Rebalancing dominance**: The condition (imbalance handling) effect is 35–1100× larger than the distance metric effect, completely absorbing any subtle grouping differences


> For detailed mathematical definitions of raw signals, feature extraction formulas, and quantitative vehicle dynamics discussion, see `vehicle_dynamics_formulation.md`.
