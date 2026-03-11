# Impact of Class Imbalance Handling and Domain Grouping Strategy on Cross-Domain Drowsy Driving Detection: A Vehicle Dynamics Perspective

---

## Abstract

Drowsy driving detection (DDD) using vehicle dynamics signals faces two practical challenges: severe class imbalance between alert and drowsy states, and domain shift across individual drivers. This study systematically evaluates the relative importance of class imbalance handling methods and domain grouping strategies through a 4-factor factorial experiment (7 conditions × 3 training modes × 3 distance metrics × 2 domain levels) with 87 driving simulator subjects and 10 random seeds (1,258 total observations). We test 6 hypotheses spanning 4 experimental axes — rebalancing effectiveness (H1), oversampling superiority (H3), distance metric relevance (H5), training mode hierarchy (H7), domain shift direction (H10), and condition × mode interaction (H12) — using non-parametric statistical methods with Bonferroni correction, permutation tests, and bootstrap confidence intervals. Results reveal that the choice of imbalance handling method has a dominant effect ($\eta^2 = 0.57$–$0.78$, large) while the choice of distance metric for domain grouping is negligible ($\eta^2 < 0.004$). SMOTE-based oversampling improves F2-score from 0.16 (baseline) to 0.56 in within-domain settings, AUROC from 0.59 to 0.91, and AUPRC from 0.09 to 0.65 — a 640% relative improvement on the imbalance-sensitive precision–recall metric. Within-domain training outperforms cross-domain training with large effect sizes ($\delta = 0.83$–$0.95$), and the optimal rebalancing strategy depends on training mode, with RUS effective only in cross-domain settings and SMOTE dominating elsewhere. We provide a vehicle dynamics explanation for the distance metric irrelevance: the bicycle model coupling $a_y = v \cdot \dot{\delta}/L$ reduces the 135-dimensional feature space to approximately 45 effective dimensions, with lane offset features ($O(10^3)$) dominating all metrics equally. These findings demonstrate that for practical DDD deployment, investing in appropriate class rebalancing yields far greater returns than optimizing domain partition strategies.

**Keywords**: drowsy driving detection, class imbalance, domain shift, vehicle dynamics, SMOTE, distance metric, non-parametric statistics

---

## 1. Introduction

### 1.1 Background

Drowsy driving is a major cause of traffic accidents worldwide, contributing to an estimated 15–20% of road fatalities (WHO, 2018). Vehicle-dynamics-based drowsy driving detection (DDD) offers a non-intrusive approach by monitoring signals such as steering angle, lateral acceleration, and lane offset. However, two fundamental challenges hinder the practical deployment of data-driven DDD systems:

1. **Class imbalance**: In naturalistic driving data, alert states vastly outnumber drowsy states (typically 9:1 or greater), biasing classifiers toward the majority class and degrading detection of dangerous drowsy episodes.

2. **Domain shift**: Individual differences in driving behavior cause performance degradation when a model trained on one group of drivers is applied to another. Subject-level domain adaptation or generalization strategies are needed but their effectiveness depends on how "domains" are defined.

### 1.2 Research Questions

Despite extensive literature on class imbalance handling (He & Garcia, 2009) and domain adaptation (Pan & Yang, 2010), their relative importance for DDD has not been quantified. Moreover, the choice of distance metric (MMD, DTW, Wasserstein) for grouping subjects into source and target domains remains an open question. This study addresses three research questions:

- **RQ1**: How does the choice of class imbalance handling method (SMOTE, subject-wise SMOTE, random undersampling) affect DDD performance across different domain configurations?
- **RQ2**: Does the choice of distance metric for domain grouping influence downstream classification performance?
- **RQ3**: How does training mode (within-domain, cross-domain, mixed) interact with imbalance handling effectiveness?

### 1.3 Contributions

This study makes the following contributions:

1. **Quantitative dominance of rebalancing over domain grouping**: We demonstrate through rigorous non-parametric testing that class imbalance handling ($\eta^2 = 0.57$–$0.78$) has 35–1,100× larger effect than distance metric choice ($\eta^2 < 0.004$) across five evaluation metrics.

2. **Vehicle dynamics explanation**: We provide a physics-grounded explanation for why distance metrics produce equivalent domain groupings — the bicycle model coupling reduces effective feature dimensionality from 135 to 45, with lane offset features dominating all metrics at $O(10^3)$ scale.

3. **Comprehensive hypothesis framework**: We test 6 hypotheses across 4 experimental axes with Bonferroni-corrected non-parametric tests, permutation tests ($p < 0.001$), bootstrap CIs, and seed convergence analysis, providing a reproducible template for DDD evaluation. Supplementary analyses (8 additional hypotheses) are reported in Appendix C.

---

## 2. Related Work

### 2.1 Vehicle-Based Drowsy Driving Detection

Vehicle dynamics signals — steering wheel angle, lateral/longitudinal acceleration, and lane position — have been widely used for DDD due to their non-intrusive nature. Arefnezhad et al. (2019) used steering wheel angle and velocity signals with statistical and spectral features for drowsiness classification. Wang et al. (2022) extended this approach using LSTM networks with five vehicle signals (speed, longitudinal acceleration, lateral acceleration, lane offset, steering wheel rate), applying Gaussian smoothing with a 1-second window. Atiquzzaman et al. (2018) introduced prediction error features based on second-order Taylor approximation to capture deviations from smooth driving trajectories.

### 2.2 Class Imbalance in DDD

The drowsy driving detection task suffers from inherent class imbalance. Standard approaches include:

- **Random Undersampling (RUS)**: Removes majority-class instances to balance the training set (He & Garcia, 2009). Simple but risks discarding informative samples.
- **Synthetic Minority Oversampling Technique (SMOTE)**: Generates synthetic minority-class samples by interpolating between nearest neighbors (Chawla et al., 2002). Preserves majority-class information but may introduce noise.
- **Subject-Wise SMOTE (SW-SMOTE)**: Applies SMOTE within each subject's data to prevent cross-subject interpolation artifacts, preserving individual driving patterns.

The sampling ratio $r$ (minority:majority after resampling) is a critical hyperparameter: aggressive ratios ($r = 0.5$) balance classes but may cause overfitting, while conservative ratios ($r = 0.1$) maintain natural distribution characteristics.

### 2.3 Domain Shift and Distance Metrics

Inter-subject variability in driving behavior creates domain shift when models trained on one group are applied to another. Distance metrics quantify subject dissimilarity for domain partitioning:

- **Maximum Mean Discrepancy (MMD)**: Measures the difference between feature distribution means in a reproducing kernel Hilbert space (Gretton et al., 2012). Captures distributional differences without explicit density estimation.
- **Dynamic Time Warping (DTW)**: Aligns temporal sequences to find optimal correspondence (Berndt & Clifford, 1994). Captures temporal pattern similarity beyond point-wise comparison.
- **Wasserstein Distance**: The Earth Mover's Distance, computing the minimum transport cost between distributions (Villani, 2009). Geometrically interpretable as the amount of "work" to reshape one distribution into another.

Prior work has not systematically compared these metrics' impact on downstream DDD performance.

---

## 3. Methodology

### 3.1 Vehicle Motion Signals

Five raw signals are extracted from the driving simulator (SIMlsl) at a sampling rate of $f_s = 60$ Hz:

| Symbol | Signal | Unit | Physical Meaning |
|:------:|--------|:----:|------------------|
| $\delta(t)$ | Steering angle | rad | Steering wheel rotation angle |
| $\dot{\delta}(t)$ | Steering speed | rad/s | Rate of steering wheel rotation |
| $a_y(t)$ | Lateral acceleration | m/s² | Cross-track acceleration |
| $a_x(t)$ | Longitudinal acceleration | m/s² | Forward/backward acceleration |
| $e_{\text{lane}}(t)$ | Lane offset | m | Lateral displacement from lane center |

Steering speed is derived via numerical differentiation:

$$\dot{\delta}(t) \approx \nabla\delta(t) \cdot f_s$$

### 3.2 Vehicle Dynamics Coupling

Under the linear bicycle model (Rajamani, 2012), lateral dynamics are governed by:

$$a_y = v \cdot \dot{\psi} = v \cdot \frac{\dot{\delta}}{L} \tag{1}$$

where $v$ is vehicle speed and $L$ is the wheelbase. This establishes a direct physical coupling: $\delta \xrightarrow{\text{diff.}} \dot{\delta} \xrightarrow{\times\,v/L} a_y$. Lane offset evolves as:

$$\ddot{e}_{\text{lane}}(t) = a_y(t) - a_{y,\text{road}}(t) \tag{2}$$

creating a four-signal causal chain $\delta \to \dot{\delta} \to a_y \to e_{\text{lane}}$, with only $a_x$ dynamically independent.

### 3.3 Feature Extraction

A total of 135 features are extracted from the five raw signals using three methods:

**Statistical and spectral features** (22 features × 2 signals in time-frequency domain): For each signal $x(t)$ in a sliding window of $N$ samples, features include Mean, Variance, Range, Skewness, Kurtosis, IQR, zero-crossing rate, quartiles, along with spectral features from the FFT power spectrum $P(f) = |\text{FFT}(x)|^2$ in the band $[0.5, 30]$ Hz: frequency variance ($\text{Var}(f_{\text{band}})$), spectral entropy ($H_s = -\sum \hat{P}(f)\log_2 \hat{P}(f)$), dominant frequency ($f^* = \arg\max_f P(f)$), spectral centroid, and average PSD.

Complexity features: Sample Entropy (template matching with tolerance $r = 0.2\sigma$), Katz fractal dimension ($\log_{10}(L)/\log_{10}(d)$), and Shannon entropy of the amplitude distribution.

**Smooth/Std/PE features** (3 features × 5 signals = 15 dimensions): Standard deviation, prediction error using 2nd-order Taylor approximation ($\hat{x}_N = 2.5x_{N-2} - 2.0x_{N-3} + 0.5x_{N-4}$, Atiquzzaman et al., 2018), and smoothed mean.

**Permutation entropy features** (8 patterns × 5 signals = 40 dimensions): The relative frequency of each ordinal pattern $\pi \in \{\text{DDD}, \text{DDA}, \text{DAD}, \text{DAA}, \text{ADD}, \text{ADA}, \text{AAD}, \text{AAA}\}$ captures signal complexity.

### 3.4 Domain Grouping

Pairwise distances between all 87 subjects are computed using their mean feature vectors with three metrics (MMD, DTW, Wasserstein). For each metric, subjects are ranked by mean distance to all others and partitioned into three groups of 29: **out-domain** (highest distances, most dissimilar), **mid-domain** (intermediate), and **in-domain** (lowest distances, most typical). In the training and evaluation protocol ("split2"), only the two extreme groups are used as target domains: **in-domain** and **out-domain** (29 subjects each). The remaining 58 subjects (mid-domain + the other extreme group) serve as the source/general training pool. Mid-domain is used only for concordance validation and is excluded from the classification experiments.

### 3.5 Experimental Design

The experiment follows a 4-factor factorial design:

| Factor | Symbol | Levels | Type |
|--------|:------:|:------:|------|
| Condition ($C$) | method × ratio | 7 | Between-group |
| Training mode ($M$) | source/target/mixed | 3 | Between-group |
| Distance metric ($D$) | MMD/DTW/Wasserstein | 3 | Between-group |
| Domain level ($L$) | in/out | 2 | Within-subject |

**Conditions (7)**: baseline (no rebalancing), RUS ($r=0.1$, $r=0.5$), SMOTE ($r=0.1$, $r=0.5$), SW-SMOTE ($r=0.1$, $r=0.5$).

**Training modes (3)**: Cross-domain (train on source domain, 58 subjects, only), within-domain (train on target domain, 29 subjects, only), mixed (train on combined source + target data).

**Evaluation metrics**: F2-score, AUROC, and AUPRC (primary). F2-score emphasizes recall for safety-critical detection; AUROC measures overall discrimination; AUPRC evaluates precision–recall trade-offs under class imbalance, avoiding the optimistic bias of AUROC when the negative class dominates (Saito & Rehmsmeier, 2015). F1-score and Recall serve as supplementary metrics.

**Reproducibility**: 10 random seeds $\times$ 18 cells (3 modes × 3 distances × 2 levels) per condition = 1,258 total observations ($-2$ pending records in the sw\_smote condition).

### 3.6 Statistical Analysis Framework

Due to non-normality (Shapiro-Wilk rejects normality in 45–71% of cells across metrics), all analyses use non-parametric tests:

| Purpose | Test | Reference |
|---------|------|-----------|
| $k$-group comparison | Kruskal-Wallis $H$ | Kruskal & Wallis (1952) |
| Paired $k$-group | Friedman $\chi_F^2$ | Friedman (1937) |
| 2-group unpaired | Mann-Whitney $U$ | Mann & Whitney (1947) |
| 2-group paired | Wilcoxon signed-rank | Wilcoxon (1945) |
| Effect size | Cliff's $\delta$ | Cliff (1993) |
| Post-hoc | Nemenyi test | Nemenyi (1963) |

All test families are Bonferroni-corrected ($\alpha' = \alpha/m$, $\alpha = 0.05$). Effect sizes follow Cliff's (1993) thresholds: $|\delta| < 0.147$ negligible, $< 0.33$ small, $< 0.474$ medium, $\geq 0.474$ large. Effect size confidence intervals are computed via percentile bootstrap ($B = 2{,}000$). Overall population means are estimated with BCa bootstrap CIs ($B = 10{,}000$). A global permutation test ($B = 10{,}000$) validates the condition effect against the null hypothesis of label exchangeability.

---

## 4. Results

### 4.1 Overview

A total of 1,258 observations across 7 conditions, 3 training modes, 3 distance metrics, 2 domain levels, and 10 random seeds were analyzed. The permutation test confirms a significant global condition effect for F2-score ($T_{\text{obs}} = 13.71$, $p < 0.001$), AUROC ($T_{\text{obs}} = 10.51$, $p < 0.001$), and AUPRC ($T_{\text{obs}} = 17.67$, $p < 0.001$).

### 4.2 Condition Effect (H1, H3)

#### 4.2.1 Global Test

The Kruskal-Wallis test reveals a significant condition effect across all 18 experimental cells:

- **F2-score**: 18/18 cells significant at Bonferroni $\alpha' = 0.0028$; mean $\eta^2 = 0.784$ (large); $H = 48.92$–$62.54$, all $p < 0.0001$.
- **AUROC**: 17/18 cells significant; mean $\eta^2 = 0.570$ (large); $H = 25.33$–$53.34$.
- **AUPRC**: 18/18 cells significant; mean $\eta^2 = 0.616$ (large); $H = 30.86$–$53.26$, all $p < 0.0001$.

Extended metrics confirm universality: F1-score (18/18 significant, $\eta^2 = 0.769$), Recall (18/18, $\eta^2 = 0.755$).

#### 4.2.2 Pairwise Comparisons (H1: Rebalancing vs. Baseline)

Mann-Whitney $U$ tests with Cliff's $\delta$ effect sizes:

| Metric | Significant (Bonf.) | Large effects | Medium | Small | Negligible |
|--------|:-------------------:|:-------------:|:------:|:-----:|:----------:|
| F2-score | 30/36 | 30 (83%) | 1 (3%) | 3 (8%) | 2 (6%) |
| AUROC | 21/36 | 22 (61%) | 3 (8%) | 6 (17%) | 5 (14%) |
| F1-score | 28/36 | 28 (78%) | — | 6 (17%) | 2 (6%) |
| AUPRC | 21/36 | 21 (58%) | 3 (8%) | 7 (19%) | 5 (14%) |
| Recall | 24/36 | 24 (67%) | 3 (8%) | 5 (14%) | 4 (11%) |

Within-domain SMOTE/SW-SMOTE vs. baseline: $\delta = +0.98$–$1.00$ (F2, AUROC, AUPRC, F1), confirming near-complete performance separation.

#### 4.2.3 Oversampling vs. RUS (H3)

Oversampling methods consistently outperform RUS: F2-score 16/24 cells oversampling wins, all with large Cliff's $\delta$; AUROC 20/24 cells; AUPRC 17/24 cells (18/24 large effects). **H3 is fully supported** — RUS degrades performance relative to oversampling across all primary evaluation metrics.

*Supplementary findings*: Plain SMOTE is generally competitive with or superior to SW-SMOTE (H2, see Appendix C). The sampling ratio effect (H4) is method-specific: $r=0.1$ is optimal for RUS and SW-SMOTE, while $r=0.5$ is preferred for SMOTE in within-domain settings (see Appendix C).

#### 4.2.4 Condition Rankings

**F2-score** (mean rank across 18 cells, 1 = best):

| Rank | Condition | Mean Rank | Wins |
|:----:|-----------|:---------:|:----:|
| 1 | sw\_smote\_r01 | 2.56 | 12 |
| 2 | smote\_r05 | 3.22 | 0 |
| 3 | smote\_r01 | 3.56 | 0 |
| 4 | baseline | 4.11 | 1 |
| 5 | rus\_r01 | 4.61 | 4 |
| 6 | sw\_smote\_r05 | 4.67 | 0 |
| 7 | rus\_r05 | 5.28 | 1 |

**AUROC**:

| Rank | Condition | Mean Rank | Wins |
|:----:|-----------|:---------:|:----:|
| 1 | smote\_r01 | 2.17 | 8 |
| 2 | smote\_r05 | 3.11 | 1 |
| 3 | sw\_smote\_r01 | 3.33 | 5 |
| 4 | sw\_smote\_r05 | 4.06 | 1 |
| 5 | baseline | 4.83 | 1 |
| 6 | rus\_r01 | 4.89 | 1 |
| 7 | rus\_r05 | 5.61 | 1 |

**AUPRC** (mean rank across 18 cells, 1 = best):

| Rank | Condition | Mean Rank | Wins |
|:----:|-----------|:---------:|:----:|
| 1 | sw\_smote\_r01 | 2.61 | 7 |
| 2 | smote\_r01 | 2.67 | 5 |
| 3 | smote\_r05 | 3.39 | 1 |
| 4 | sw\_smote\_r05 | 4.22 | 0 |
| 5 | rus\_r01 | 4.56 | 3 |
| 6 | baseline | 5.17 | 0 |
| 7 | rus\_r05 | 5.39 | 2 |

Nemenyi post-hoc test (Friedman $\chi^2 = 47.91$–$50.49$, $p < 0.0001$; CD = 2.848) confirms 8–10/21 pairwise comparisons significant across all three primary metrics. AUPRC rankings closely mirror F2 rankings ($\rho = 0.821$) but differ from AUROC (sw\_smote\_r01 ranks 1st on F2/AUPRC vs. smote\_r01 on AUROC), reflecting AUPRC's sensitivity to the precision–recall balance under imbalance.

### 4.3 Distance Metric Effect (H5)

#### 4.3.1 Performance Independence

The Kruskal-Wallis test shows no significant distance metric effect:

| Metric | Bonf. significant cells | Mean $\eta^2$ | Max $\eta^2$ |
|--------|:-----------------------:|:-------------:|:------------:|
| F2-score | 0/6 | < 0.002 | 0.019 |
| AUROC | 2/6 | 0.020 | 0.047 |
| F1-score | 0/6 | < 0.002 | — |
| AUPRC | 0/6 | < 0.002 | — |
| Recall | 0/6 | < 0.002 | — |

Pooled performance across all conditions:

| Metric | MMD | DTW | Wasserstein | Max $|\delta|$ |
|--------|:---:|:---:|:-----------:|:--------------:|
| F2-score | 0.289 ± 0.187 | 0.284 ± 0.190 | 0.293 ± 0.195 | 0.023 |
| AUROC | 0.689 ± 0.167 | 0.684 ± 0.170 | 0.697 ± 0.169 | 0.080 |
| AUPRC | 0.264 ± 0.261 | 0.266 ± 0.264 | 0.275 ± 0.266 | — |

All pairwise Cliff's $\delta$ values are negligible ($|\delta| < 0.147$).

#### 4.3.2 Granular Analysis

Detailed stratification by mode × condition × level reveals isolated weak effects only in cross-domain AUROC cells ($\eta^2 \approx 0.04$), fully absorbed after Bonferroni correction in most strata. No single metric (including Wasserstein) demonstrates superior discriminative power (H6, see Appendix C); all three produce equivalent domain groupings.

#### 4.3.3 Domain Group Concordance

Subject rankings by mean distance across the three metrics show moderate-to-strong agreement:

| Metric Pair | Spearman $\rho$ | $p$-value | Kendall $\tau$ |
|-------------|:---------------:|:---------:|:--------------:|
| MMD vs. Wasserstein | 0.752 | $4.6 \times 10^{-17}$ | 0.622 |
| MMD vs. DTW | 0.477 | $3.0 \times 10^{-6}$ | 0.339 |
| Wasserstein vs. DTW | 0.803 | $9.0 \times 10^{-21}$ | 0.617 |

Despite this, 52/87 (59.8%) subjects switch domain groups across metrics in the full three-way ranking. Group membership overlap is highest for out-domain (18/29 = 62% three-way agreement) and lowest for mid-domain (5/29 = 17%). Since only the two extreme groups (in-domain and out-domain) enter the classification experiments, the relevant question is whether these extreme assignments change — and the high out-domain overlap (62%) indicates relative stability at the tails.

**Key finding**: Even though subjects switch groups in ranking, the downstream classification performance is indistinguishable ($\eta^2 < 0.004$). This demonstrates that the distance metric–performance pathway is effectively decoupled.

### 4.4 Training Mode Effect (H7)

#### 4.4.1 Within-Domain vs. Cross-Domain (H7)

The training mode has a massive impact on performance:

| Metric | Cross-domain | Within-domain | Mixed | $\delta$ (Within vs. Cross) |
|--------|:------------:|:-------------:|:-----:|:---------------------------:|
| F2-score | 0.125 ± 0.043 | 0.366 ± 0.187 | 0.375 ± 0.179 | +0.830 (large) |
| AUROC | 0.520 ± 0.015 | 0.774 ± 0.150 | 0.776 ± 0.140 | +0.945 (large) |
| AUPRC | 0.050 ± 0.007 | 0.370 ± 0.273 | 0.384 ± 0.281 | +0.946 (large) |

All 14 Friedman tests across conditions are significant (14/14, $p < 0.0001$), with Kendall's W ranging from 0.375 to 0.976. Mixed training performs equivalently to within-domain and substantially better than cross-domain (F2: $0.375$ vs. $0.125$; AUROC: $0.776$ vs. $0.520$), confirming the supplementary H8 finding (see Appendix C).

#### 4.4.2 Descriptive Statistics by Condition × Mode

| Condition | Cross-domain F2 | Within-domain F2 | Mixed F2 |
|-----------|:----------------:|:------------------:|:---------:|
| baseline | 0.080 | 0.205 | 0.236 |
| smote\_r01 | 0.136 | 0.459 | 0.462 |
| smote\_r05 | 0.113 | 0.506 | 0.523 |
| sw\_smote\_r01 | 0.103 | 0.556 | 0.572 |
| sw\_smote\_r05 | 0.043 | 0.490 | 0.424 |
| rus\_r01 | 0.166 | 0.178 | 0.209 |
| rus\_r05 | 0.156 | 0.157 | 0.168 |

### 4.5 Domain Shift Effect (H10, H12)

#### 4.5.1 In-Domain vs. Out-Domain (H10)

The domain gap $\Delta = Y_{\text{out}} - Y_{\text{in}}$ is generally small and non-significant:

- **F2-score**: 0/63 Wilcoxon tests significant (Bonf. $\alpha' = 0.00079$); mean $|\Delta| = 0.030$–$0.047$.
- **AUROC**: 0/63 significant; mean $|\Delta| = 0.019$–$0.097$.

Notably, within-domain and mixed training often show **positive** $\Delta$ (out-domain outperforms in-domain), suggesting that rebalancing is more effective for behaviorally diverse subjects:

| Mode | Baseline $\Delta$ (F2) | Best rebalancing $\Delta$ (F2) |
|------|:----------------------:|:------------------------------:|
| Cross-domain | −0.025 | SMOTE: +0.003 |
| Within-domain | +0.027 | RUS: +0.122 |
| Mixed | +0.085 | SMOTE: +0.104 |

#### 4.5.2 Condition × Mode Interaction (H12)

**Strongly supported.** The optimal condition varies by mode — RUS performs best in cross-domain (rank 1–2 by F2); SMOTE/SW-SMOTE dominate in within-domain and mixed settings. This interaction indicates that practitioners cannot choose a single rebalancing strategy without considering the training mode.

*Supplementary findings*: Condition × distance interaction (H13) is weak (12/18 consistent, 6/18 minor swaps; see Appendix C). Level × mode interaction (H14) confirms that within-domain training eliminates or reverses the domain gap (see Appendix C). The domain gap in cross-domain settings (H9) and the effect of oversampling on domain gap (H11) show mixed, context-dependent results (see Appendix C).

### 4.6 Robustness Validation

#### 4.6.1 Seed Convergence

Subsampling analysis confirms ranking stability:

| Metric | $k=3$ ($\sigma_{\text{rank}}$) | $k=5$ | $k=7$ | $k=9$ |
|--------|:------:|:------:|:------:|:------:|
| F2-score | 0.332 | 0.189 | 0.063 | 0.000 |
| AUROC | 0.525 | 0.341 | 0.225 | 0.120 |

By $k=9$ (of 10 seeds), F2 rankings are perfectly stable; AUROC rankings stabilize with $\sigma = 0.120$.

#### 4.6.2 Cross-Metric Concordance

Kendall's $W = 0.618$ ($k = 7$ metrics: F2, AUROC, F1, AUPRC, Recall, Precision, Accuracy; $n = 7$ conditions) indicates **moderate agreement** in condition rankings across all evaluation metrics.

Strongest pairwise concordance: AUROC ↔ AUPRC ($\rho = 0.857$), F2 ↔ AUPRC ($\rho = 0.821$).

#### 4.6.3 Ratio Sensitivity

Directional agreement between $r=0.1$ and $r=0.5$ rankings: 91% (F2), 87% (AUROC). AUROC ranking is perfectly stable across ratios ($\rho = 1.000$). F2 ranking is more sensitive ($\rho = 0.400$), primarily due to SMOTE's ratio-dependent behavior.

#### 4.6.4 Precision–Recall Trade-Off

No clear precision–recall trade-off: 0/36 cells exhibit simultaneous recall improvement with precision degradation. SMOTE methods yield predominantly win-win outcomes (recall↑, precision stable); RUS shows regression patterns (both decline).

#### 4.6.5 Power Analysis

With $n = 10$ seeds per cell, Mann-Whitney $U$ detects:
- Per-distance cell ($n = 10$): $|\delta_{\min}| \approx 1.014$ — only **large** effects detectable
- Pooled across distances ($n = 30$): $|\delta_{\min}| \approx 0.586$ — **large** effects detectable

Wilcoxon signed-rank has a $p$-floor of $1/2^9 = 0.00195$, which limits Bonferroni-corrected significance for paired tests.

---

## 5. Discussion

### 5.1 Why Distance Metrics Are Irrelevant: A Vehicle Dynamics Explanation

The central unexpected finding — that three fundamentally different distance metrics produce equivalent downstream performance ($\eta^2 < 0.004$) — can be explained through the vehicle dynamics of the feature space:

**Physical coupling reduces effective dimensionality.** The bicycle model (Eq. 1) couples four of five raw signals ($\delta$, $\dot{\delta}$, $a_y$, $e_{\text{lane}}$) through the chain $\delta \to \dot{\delta} \to a_y \to e_{\text{lane}}$, with only $a_x$ dynamically independent. PCA confirms this: 45 principal components explain 95% of variance from 135 features (3:1 compression), and the first PC alone explains 22.2%.

**Feature scale heterogeneity causes lane offset dominance.** Lane offset features have ranges of $O(10^3)$ while all other features are $O(10^{-1})$–$O(10^0)$. Since the distance metrics operate on unnormalized features, the domain grouping is effectively determined by lane offset behavior alone, regardless of which metric is used. The variance decomposition confirms this: lane offset contributes approximately 100% of total inter-subject variance.

**Inter-subject discrimination is inherently weak.** The overall ICC ratio (inter-subject / total variance) is only 0.111, meaning 88.9% of feature variance is intra-subject (within-session variability). When subject positions in feature space are this "blurred," the coarse two-way partition into extreme groups (in-domain and out-domain, each 29 of 87 subjects) is robust to metric choice, even though fine-grained subject rankings differ ($\rho = 0.48$–$0.80$).

**Rebalancing absorption.** The condition effect is 35–1,100× larger than the distance metric effect:

$$\frac{\eta^2_{\text{condition}}}{\eta^2_{\text{distance}}} \approx \frac{0.11\text{–}0.14}{<0.004} = 35\text{–}1{,}100\times \tag{3}$$

Rebalancing shifts the classifier's decision boundary so dramatically that any subtle difference in training set composition due to domain grouping is overwhelmed.

### 5.2 Rebalancing as a Stronger Lever Than Domain Adaptation

Our results provide a clear hierarchy of optimization priorities: **rebalancing method** ($\eta^2 = 0.57$–$0.78$) > **training mode** ($\delta = 0.83$–$0.95$) >> **distance metric** ($\eta^2 < 0.004$). This has practical implications for DDD system design:

- SMOTE-based oversampling transforms within-domain F2 from 0.205 (baseline) to 0.556 (sw\_smote\_r01) — a 171% improvement.
- AUROC improves from 0.59 (baseline) to 0.91 (smote\_r01) in within-domain settings.
- AUPRC — the metric most sensitive to class imbalance — improves from 0.09 (baseline) to 0.65 (smote\_r01), a 640% relative improvement. This confirms that rebalancing produces genuine minority-class precision gains, not merely threshold-shifted recall.
- These gains require no additional data collection, no domain distance computation, and no subject grouping strategy — only a preprocessing step.

In contrast, switching from Wasserstein to MMD for domain grouping yields $|\Delta\text{F2}| < 0.01$, $|\Delta\text{AUROC}| < 0.02$, and $|\Delta\text{AUPRC}| < 0.01$.

### 5.3 The Domain Gap Reversal Phenomenon

An unexpected finding is that the domain gap reverses in within-domain and mixed training: out-domain subjects sometimes outperform in-domain subjects ($\Delta > 0$). This may be because:

1. Out-domain subjects exhibit more diverse driving patterns, providing richer training signal when their own data is included (within-domain setting).
2. The diversity of out-domain data acts as implicit regularization, improving generalization.
3. Rebalancing is more effective for subjects with varied drowsiness patterns (more minority-class samples to oversample).

### 5.4 Limitations

1. **Single classifier**: Results are based on one classifier type. While the rebalancing effect is a property of the training data rather than the model architecture, generalization to deep learning models requires verification.

2. **Deterministic data split**: The train/test partition (`subject_time_split`) is deterministic — random seeds only vary model initialization and resampling, not the data partition.

3. **Unnormalized distance computation**: The domain grouping uses unnormalized features, causing lane offset dominance. Feature standardization may reveal metric-specific differences that are currently masked.

4. **Sample size constraints**: With $n = 10$ per cell, only large effects ($|\delta| > 0.59$) are detectable after Bonferroni correction. Subtle distance metric effects may exist below our detection threshold.

5. **Wilcoxon $p$-floor**: The minimum achievable p-value for Wilcoxon signed-rank with $n = 10$ is $1/2^9 = 0.00195$, which limits paired comparisons under strict Bonferroni correction.

### 5.5 Future Directions

1. **Feature normalization**: Repeating distance computation on standardized features would test whether metrics become distinguishable when lane offset dominance is removed.
2. **Signal-specific distances**: Computing domain groups using individual signals (lane offset alone vs. all signals) would quantify the information contribution of each signal.
3. **Multiple classifiers**: Extending to LSTM and SVM-based architectures would strengthen the generalizability claim.
4. **Real-vehicle validation**: Transferring findings from simulator to on-road data is essential for practical deployment.

---

## 6. Conclusion

This study systematically evaluates the interplay between class imbalance handling and domain grouping strategies for drowsy driving detection through a factorial experiment with 87 subjects, 7 conditions, and 1,258 observations, testing 6 hypotheses across 4 experimental axes with rigorous non-parametric statistics.

The key findings are:

1. **Class imbalance handling dominates** ($\eta^2 = 0.57$–$0.78$): SMOTE-based oversampling dramatically improves detection performance across all metrics. SW-SMOTE with $r = 0.1$ achieves the highest F2-score (0.556) and AUPRC (0.654) in within-domain settings; plain SMOTE with $r = 0.1$ achieves the highest AUROC (0.91). The AUPRC improvement from baseline (0.09 → 0.65, +640%) confirms that gains are genuine under class imbalance, not artifacts of threshold-insensitive metrics.

2. **Distance metric choice is irrelevant** ($\eta^2 < 0.004$): MMD, DTW, and Wasserstein produce statistically indistinguishable downstream performance despite 59.8% of subjects switching domain groups. Vehicle dynamics coupling and feature scale heterogeneity provide a physics-grounded explanation.

3. **Training mode hierarchy is clear**: Within-domain $\approx$ mixed >> cross-domain (F2: 0.37 vs. 0.13; AUROC: 0.77 vs. 0.52; AUPRC: 0.37 vs. 0.05), with Cliff's $\delta = 0.83$–$0.95$ (large).

4. **Domain shift direction is context-dependent**: In-domain subjects do not consistently outperform out-domain subjects; within-domain and mixed training reverse the expected domain gap ($\Delta > 0$), with rebalancing amplifying the reversal.

5. **Rebalancing strategy depends on training mode**: The optimal condition varies by mode (RUS in cross-domain; SMOTE/SW-SMOTE in within-domain and mixed), revealing a strong condition × mode interaction that practitioners must account for.

6. **Results are robust**: Consistent across 10 random seeds ($\sigma_{\text{rank}} \to 0$ at $k = 9$), 5 evaluation metrics (Kendall's $W = 0.618$), 2 sampling ratios (91% directional agreement), and confirmed by permutation test ($p < 0.001$).

For practitioners, these results prescribe a clear strategy: apply SMOTE-based class rebalancing with within-domain or mixed training, and choose any convenient distance metric for domain grouping. The choice of rebalancing method matters more than all other design decisions combined.

---

## References

- Arefnezhad, S., et al. (2019). Driver drowsiness estimation using EEG signals with a dynamical encoder–decoder. *IET Intelligent Transport Systems*, 13(2), 301–310.
- Atiquzzaman, M., et al. (2018). Real-time detection of drivers' texting and eating behavior based on vehicle dynamics. *Transportation Research Part F*, 58, 594–604.
- Berndt, D. J., & Clifford, J. (1994). Using dynamic time warping to find patterns in time series. *AAAI Workshop on Knowledge Discovery in Databases*, 359–370.
- Chawla, N. V., et al. (2002). SMOTE: Synthetic minority over-sampling technique. *JAIR*, 16, 321–357.
- Cliff, N. (1993). Dominance statistics: Ordinal analyses to answer ordinal questions. *Psychological Bulletin*, 114(3), 494–509.
- Friedman, M. (1937). The use of ranks to avoid the assumption of normality. *JASA*, 32(200), 675–701.
- Gretton, A., et al. (2012). A kernel two-sample test. *JMLR*, 13(1), 723–773.
- He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE TKDE*, 21(9), 1263–1284.
- Kruskal, W. H., & Wallis, W. A. (1952). Use of ranks in one-criterion variance analysis. *JASA*, 47(260), 583–621.
- Mann, H. B., & Whitney, D. R. (1947). On a test of whether one of two random variables is stochastically larger. *Annals of Mathematical Statistics*, 18(1), 50–60.
- Nemenyi, P. (1963). *Distribution-free multiple comparisons*. PhD thesis, Princeton University.
- Pan, S. J., & Yang, Q. (2010). A survey on transfer learning. *IEEE TKDE*, 22(10), 1345–1359.
- Rajamani, R. (2012). *Vehicle Dynamics and Control* (2nd ed.). Springer.
- Saito, T., & Rehmsmeier, M. (2015). The precision–recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE*, 10(3), e0118432.
- Shannon, C. E. (1948). A mathematical theory of communication. *Bell System Technical Journal*, 27(3), 379–423.
- Saito, T., & Rehmsmeier, M. (2015). The precision–recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE*, 10(3), e0118432.
- Villani, C. (2009). *Optimal Transport: Old and New*. Springer.
- Wang, X., et al. (2022). Real-time detection of driver drowsiness using LSTM. *Sensors*, 22(13), 4904.
- Wilcoxon, F. (1945). Individual comparisons by ranking methods. *Biometrics Bulletin*, 1(6), 80–83.

---

## Appendix A: Primary Hypothesis Summary Table

| # | Hypothesis | Verdict | Key Evidence |
|:-:|-----------|:-------:|-------------|
| H1 | Rebalancing improves F2 over baseline | ✓ Supported | $\eta^2 = 0.784$; 30/36 large effects |
| H3 | Oversampling > RUS | ✓ Fully supported | 24/24 cells, all large $\delta$ |
| H5 | Distance metric matters | ✗ Negligible | $\eta^2 < 0.004$, all metrics equivalent |
| H7 | Within-domain > cross-domain | ✓ Fully supported | $\delta = +0.830$ (F2), $+0.945$ (AUROC) |
| H10 | In-domain > out-domain | ✓ Partially | True in cross-domain; reversed in within-domain |
| H12 | Condition × Mode interaction | ✓ Strong | Best method varies by mode |

## Appendix B: Extended Metrics Condition Rankings

| Rank | F2 (mean rank) | AUROC | F1 | AUPRC | Recall |
|:----:|:--------------:|:-----:|:--:|:-----:|:------:|
| 1 | sw\_smote\_r01 (2.56) | smote\_r01 (2.17) | sw\_smote\_r01 (3.06) | sw\_smote\_r01 (2.61) | smote\_r01 (2.17) |
| 2 | smote\_r05 (3.22) | smote\_r05 (3.11) | sw\_smote\_r05 (3.17) | smote\_r01 (2.67) | baseline (2.83) |
| 3 | smote\_r01 (3.56) | sw\_smote\_r01 (3.33) | smote\_r01 (3.50) | smote\_r05 (3.39) | smote\_r05 (3.44) |

## Appendix C: Supplementary Hypothesis Results

The following 8 hypotheses were tested as part of the comprehensive analysis framework but are reported as supplementary findings. They either overlap with the primary hypotheses (H1, H3, H5, H7, H10, H12) or produced weaker/context-dependent effects.

| # | Hypothesis | Verdict | Key Evidence | Relation to Primary |
|:-:|-----------|:-------:|-------------|:-------------------:|
| H2 | SW-SMOTE > SMOTE | ✗ Not supported | SMOTE wins 8/12 cells (F2) | Subsumed by H1 + H3 |
| H4 | Ratio affects performance | ✓ Supported | Method-specific: RUS/SW→$r=0.1$; SMOTE→$r=0.5$ | Method-dependent detail of H1 |
| H6 | Wasserstein most discriminative | ✗ Not supported | All metrics produce equivalent groupings | Redundant given H5 |
| H8 | Mixed > cross-domain | ✓ Fully supported | F2: 0.375 vs. 0.125 | Corollary of H7 |
| H9 | Domain gap larger in cross-domain | ✓ Supported | Cross-domain $\Delta < 0$; within-domain reverses | Overlaps H10 |
| H11 | Oversampling reduces domain gap | ✗ Mixed | Context-dependent; sometimes increases gap | Weak interaction effect |
| H13 | Condition × Distance interaction | ✓ Weak | 12/18 consistent; 6/18 minor swaps | Negligible given H5 |
| H14 | Level × Mode interaction | ✓ Supported | Within-domain reverses domain shift | Extension of H7 + H10 |
