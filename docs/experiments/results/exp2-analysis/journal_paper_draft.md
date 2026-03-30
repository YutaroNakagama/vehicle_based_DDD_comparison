# Impact of Class Imbalance Handling and Domain Grouping Strategy on Cross-Domain Drowsy Driving Detection: A Vehicle Dynamics Perspective

---

## Abstract

Drowsy driving detection (DDD) using vehicle dynamics signals faces two practical challenges: severe class imbalance between alert and drowsy states, and domain shift across individual drivers. This study systematically evaluates the relative importance of class imbalance handling methods and domain grouping strategies through a 4-factor factorial experiment (7 rebalancing strategies × 3 training modes × 3 distance metrics × 2 domain membership groups) with 87 driving simulator subjects and 12 random seeds (1,512 total observations). We address three research questions — effectiveness of class imbalance handling (RQ1: H1), influence of domain grouping decisions (RQ2: H2, H3, H4), and interaction between rebalancing and domain configuration (RQ3: H5) — using non-parametric statistical methods with Bonferroni correction, permutation tests, and bootstrap confidence intervals. A variance-based sensitivity analysis (functional ANOVA decomposition) reveals that **Mode** ($S_M = 0.31$–$0.50$) and **Rebalancing** ($S_R = 0.24$–$0.29$) together account for over 60% of total variance as main effects, with a substantial $R \times M$ interaction ($S_{R \times M} = 0.16$–$0.21$), while Distance ($S_D < 0.001$) and Membership ($S_G < 0.01$) are negligible. SMOTE-based oversampling improves F2-score from 0.22 (baseline) to 0.56 in within-domain settings, AUROC from 0.63 to 0.90, and AUPRC from 0.12 to 0.65 — a 460% relative improvement on the imbalance-sensitive precision–recall metric. Within-domain training outperforms cross-domain training with large effect sizes ($\delta = 0.83$–$0.95$), and the optimal rebalancing strategy depends on training mode, with RUS effective only in cross-domain settings and SMOTE dominating elsewhere. We provide a vehicle dynamics explanation for the distance metric irrelevance: the bicycle model coupling $a_y = v \cdot \dot{\delta}/L$ reduces the 135-dimensional feature space to approximately 45 effective dimensions, and despite global z-score normalisation, the low inter-subject discrimination (ICC = 0.111) and dominant rebalancing effect ($S_{TR}/S_{TD} > 27\times$) render metric choice immaterial. These findings demonstrate that for practical DDD deployment, selecting an appropriate training mode and applying class rebalancing jointly yield far greater returns than optimizing domain partition strategies.

**Keywords**: drowsy driving detection, class imbalance, domain shift, vehicle dynamics, SMOTE, distance metric, non-parametric statistics

---

## 1. Introduction

### 1.1 Background

Drowsy driving is a major cause of traffic accidents worldwide, contributing to an estimated 15–20% of road fatalities (WHO, 2018). Vehicle-dynamics-based drowsy driving detection (DDD) offers a non-intrusive approach by monitoring signals such as steering angle, lateral acceleration, and lane offset. However, two fundamental challenges hinder the practical deployment of data-driven DDD systems:

1. **Class imbalance**: In naturalistic driving data, alert states vastly outnumber drowsy states (typically 9:1 or greater), biasing classifiers toward the majority class and degrading detection of dangerous drowsy episodes.

2. **Domain shift**: Individual differences in driving behavior cause performance degradation when a model trained on one group of drivers is applied to another. Subject-level domain adaptation or generalization strategies are needed but their effectiveness depends on how "domains" are defined.

### 1.2 Research Questions


Despite extensive literature on class imbalance handling (He & Garcia, 2009) and domain adaptation (Pan & Yang, 2010), their relative importance for DDD has not been quantified. Moreover, the interplay between rebalancing strategy and domain configuration — distance metric selection, domain membership, and training mode — remains unexplored. This study addresses three research questions:

- **RQ1 (Class imbalance handling)**: How does the choice of rebalancing method (SMOTE, subject-wise SMOTE, random undersampling) and sampling ratio affect DDD performance? (H1)
- **RQ2 (Domain analysis)**: How do domain grouping decisions — distance metric, domain membership (in-domain vs. out-domain), and training mode (within-domain, cross-domain, mixed) — influence classification outcomes? (H2, H3, H4)
- **RQ3 (Interaction)**: How does the effectiveness of imbalance handling interact with domain configuration? (H5)

### 1.3 Contributions

This study makes the following contributions:

1. **Quantitative dominance of training mode and rebalancing over domain grouping** (RQ1 vs. RQ2): A variance-based sensitivity analysis (functional ANOVA decomposition) shows that training mode ($S_{TM} = 0.48$–$0.66$) and rebalancing ($S_{TR} = 0.40$–$0.46$), including their interaction ($S_{R \times M} = 0.12$–$0.21$), together account for $>80$% of systematic variance, while distance metric ($S_{TD} < 0.015$) and domain membership ($S_{TG} < 0.031$) are negligible.

2. **Vehicle dynamics explanation** (RQ2): We provide a physics-grounded explanation for why distance metrics produce equivalent downstream performance despite generating genuinely different subject rankings — the bicycle model coupling reduces effective feature dimensionality from 135 to 45, inter-subject discrimination is inherently weak (ICC = 0.111), and rebalancing absorbs any grouping differences ($S_{TR}/S_{TD} > 27\times$).

3. **Mode-dependent rebalancing strategy** (RQ3): We reveal a strong rebalancing × mode interaction ($S_{R \times M} = 0.12$–$0.21$, the third-largest effect) — RUS is effective only in cross-domain settings while SMOTE dominates elsewhere — demonstrating that practitioners must jointly consider imbalance handling and domain configuration.

4. **Comprehensive hypothesis framework**: We test 6 hypotheses with Bonferroni-corrected non-parametric tests, permutation tests ($p < 0.001$), bootstrap CIs, and seed convergence analysis, providing a reproducible template for DDD evaluation. Supplementary analyses (7 additional hypotheses) are reported in Appendix C.

---

## 2. Related Work

### 2.1 Vehicle-Based Drowsy Driving Detection

Vehicle dynamics signals — steering wheel angle, lateral/longitudinal acceleration, and lane position — have been widely used for DDD due to their non-intrusive nature. Arefnezhad et al. (2019) used steering wheel angle and velocity signals with statistical and spectral features for drowsiness classification. Wang et al. (2022) extended this approach using LSTM networks with five vehicle signals (speed, longitudinal acceleration, lateral acceleration, lane offset, steering wheel rate), applying Gaussian smoothing with a 1-second window. Zhao et al. (2012) applied continuous wavelet transform to vehicle dynamics signals for drowsiness detection, demonstrating that time–frequency representations can capture transient changes in driving behaviour that fixed-window spectral methods may miss.

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

Pairwise distances between all 87 subjects are computed using their mean feature vectors with three metrics (MMD, DTW, Wasserstein). For each metric, a $K$-nearest-neighbour (KNN) score is computed per subject as the mean distance to the $K = 5$ closest neighbours, and subjects are ranked by this score and split at the median into two groups: **in-domain** (44 subjects with the lowest KNN scores, most typical) and **out-domain** (43 subjects with the highest KNN scores, most dissimilar). This binary partition ensures that every subject is assigned to exactly one domain group with no intermediate category, maximizing statistical power for the in-domain vs. out-domain contrast.

### 3.5 Experimental Design

The experiment follows a 4-factor factorial design:

| Factor | Symbol | Levels | Type |
|--------|:------:|:------:|------|
| Rebalancing strategy ($R$) | method × ratio | 7 | Between-group |
| Distance metric ($D$) | MMD/DTW/Wasserstein | 3 | Between-group |
| Domain membership ($G$) | in/out | 2 | Within-subject |
| Training mode ($M$) | cross/within/mixed | 3 | Between-group |

**Rebalancing strategies (7)**: baseline (no rebalancing), RUS ($r=0.1$, $r=0.5$), SMOTE ($r=0.1$, $r=0.5$), SW-SMOTE ($r=0.1$, $r=0.5$).

**Distance metrics (3)**: MMD, DTW, and Wasserstein, used to compute subject-level KNN scores for domain grouping (§3.4).

**Domain membership (2)**: In-domain (44 subjects) and out-domain (43 subjects), determined by median split of KNN scores.

**Training modes (3)**: Cross-domain (train on the opposite domain group only, 43 or 44 subjects), within-domain (train on the target domain group only, 44 or 43 subjects), mixed (train on all 87 subjects).

**Evaluation metrics**: F2-score, AUROC, and AUPRC (primary). F2-score emphasizes recall for safety-critical detection; AUROC measures overall discrimination; AUPRC evaluates precision–recall trade-offs under class imbalance, avoiding the optimistic bias of AUROC when the negative class dominates (Saito & Rehmsmeier, 2015). F1-score and Recall serve as supplementary metrics.

**Reproducibility**: 12 random seeds $\times$ 18 cells (3 modes × 3 distances × 2 membership groups) per strategy = 1,512 total observations.

### 3.6 Statistical Analysis Framework

Due to non-normality (Shapiro-Wilk rejects normality in 45–71% of cells across metrics), all analyses use non-parametric tests:

| Purpose | Test | Reference |
|---------|------|-----------|
| $k$-group comparison | Kruskal-Wallis $H$ | Kruskal & Wallis (1952) |
| Paired $k$-group | Friedman $\chi_F^2$ | Friedman (1937) |
| 2-group unpaired | Mann-Whitney $U$ | Mann & Whitney (1947) |
| 2-group paired | Wilcoxon signed-rank | Wilcoxon (1945) |
| Effect size | Cliff's $\delta$ | Cliff (1993) |
| Sensitivity analysis | Sobol indices ($S_i$, $S_{Ti}$) | Saltelli et al. (2008) |
| Post-hoc | Nemenyi test | Nemenyi (1963) |
| Bayesian null support | $BF_{01}$ (BIC approximation) | Wagenmakers (2007); Masson (2011) |

All test families are Bonferroni-corrected ($\alpha' = \alpha/m$, $\alpha = 0.05$). Effect sizes follow Cliff's (1993) thresholds: $|\delta| < 0.147$ negligible, $< 0.33$ small, $< 0.474$ medium, $\geq 0.474$ large. Effect size confidence intervals are computed via percentile bootstrap ($B = 2{,}000$). Overall population means are estimated with BCa bootstrap CIs ($B = 10{,}000$). A global permutation test ($B = 10{,}000$) validates the rebalancing effect against the null hypothesis of label exchangeability. For null claims (H2: distance metric equivalence), we supplement frequentist tests with Bayes factors using the BIC approximation (Wagenmakers, 2007), interpreted on the Jeffreys (1961) scale.

#### 3.6.1 Kruskal–Wallis effect size ($\eta^2$)

Kruskal–Wallis $H$ is computed as:

$$
H = \frac{12}{N(N+1)} \sum_{i=1}^{k} \frac{R_i^2}{n_i} - 3(N+1)
$$

where:
- $k$ is the number of groups (e.g., $k=7$ for Rebalancing strategy, $k=3$ for Mode/Distance, $k=2$ for Domain membership),
- $N$ is the total number of observations,
- $R_i$ is the rank sum for group $i$,
- $n_i$ is the sample size for group $i$.

The nonparametric effect size $\eta^2$ is approximated by:

$$
\eta^2 = \frac{H}{N - 1}.
$$

Higher $\eta^2$ values indicate a larger proportion of variance explained by the factor.

#### 3.6.2 Variance-based sensitivity analysis (Sobol indices)

To quantify the relative importance of each factor and their interactions, we adopt the Sobol--Hoeffding functional ANOVA decomposition (Sobol, 1993; Saltelli et al., 2008). For a model output $Y = f(X_1, \ldots, X_k)$ with $k$ input factors, the total variance admits a unique orthogonal decomposition:

$$V(Y) = \sum_{i=1}^{k} V_i + \sum_{i<j} V_{ij} + \cdots + V_{1,2,\ldots,k} \tag{3}$$

where $V_i = \mathrm{Var}_{X_i}\bigl[\mathbb{E}_{X_{\sim i}}(Y \mid X_i)\bigr]$ is the variance of the conditional expectation over factor $i$ alone, $V_{ij}$ captures the joint effect of $(X_i, X_j)$ not accounted for by their individual contributions, and higher-order terms follow analogously. The first-order Sobol index normalises each contribution by the total variance:

$$S_i = \frac{V_i}{V(Y)} \tag{4}$$

and the total-order index aggregates the main effect of factor $i$ with all interactions in which it participates:

$$S_{Ti} = S_i + \sum_{j \neq i} S_{ij} + \sum_{j<k,\; i \in \lbrace j,k\rbrace} S_{ijk} + \cdots = 1 - \frac{V_{\sim i}}{V(Y)} \tag{5}$$

where $V_{\sim i} = \mathrm{Var}_{X_{\sim i}}\bigl[\mathbb{E}_{X_i}(Y \mid X_{\sim i})\bigr]$. The difference $S_{Ti} - S_i$ quantifies the fraction of variance attributable to interactions involving factor $i$.

**Factorial design.** Our experiment constitutes a balanced full-factorial design with $k = 4$ categorical factors:

| Factor | Symbol | Levels | Values |
|--------|:------:|:------:|--------|
| Rebalancing | $R$ | 7 | Baseline, RUS-0.1, RUS-0.5, SMOTE-0.1, SMOTE-0.5, SW-SMOTE-0.1, SW-SMOTE-0.5 |
| Training mode | $M$ | 3 | Cross-domain (source-only), Within-domain (target-only), Mixed |
| Distance metric | $D$ | 3 | MMD, DTW, Wasserstein |
| Domain membership | $G$ | 2 | In-domain, Out-domain |

Each of the $7 \times 3 \times 3 \times 2 = 126$ factor combinations is evaluated over 12 fixed random seeds $\mathcal{S} = \{0, 1, 3, 7, 13, 42, 123, 256, 512, 999, 1337, 2024\}$, yielding $N = 1{,}512$ total observations. Because the design is balanced, all sum-of-squares terms ($\text{SS}_i$, $\text{SS}_{ij}$, ...) are computed exactly via the marginal means without Monte Carlo sampling; hence $S_i = \text{SS}_i / \text{SS}_{\text{total}}$ with no estimation error beyond seed-to-seed variability. Interactions up to fourth order ($R \times M \times D \times G$) are resolved. Confidence intervals (95%) are obtained by percentile bootstrap ($B = 2{,}000$) resampling over the 12 seeds, preserving the factorial structure within each resample.

### 3.7 Hypotheses

Based on the literature and factorial design, we formulate 5 primary hypotheses and 6 supplementary hypotheses.

**Primary hypotheses** (tested with full statistical rigour in §4):

| # | Hypothesis | Factor | RQ |
|:-:|-----------|:------:|:--:|
| H1 | The choice of rebalancing method (Baseline, SMOTE, SW-SMOTE, RUS) and sampling ratio significantly affect classification performance, with method-dependent ratio sensitivity | Rebalancing | RQ1 |
| H2 | The choice of distance metric affects downstream performance | Distance | RQ2 |
| H3 | Domain membership (in-domain vs. out-domain) significantly affects classification performance | Membership | RQ2 |
| H4 | Within-domain training outperforms cross-domain training | Mode | RQ2 |
| H5 | The effect of rebalancing depends on training mode and domain membership ($R \times M \times G$ interaction) | Rebalancing × Mode (× Membership) | RQ3 |

**Supplementary hypotheses** (H6–H11, reported in Appendix C) refine the primary findings with finer-grained comparisons: Wasserstein superiority (H6), mixed vs. cross-domain (H7), domain gap in cross-domain (H8), oversampling and domain gap (H9), rebalancing × distance interaction (H10), and membership × mode interaction (H11).

---

## 4. Results

### 4.1 Overview

A total of 1,512 observations across 7 rebalancing strategies, 3 training modes, 3 distance metrics, 2 domain membership groups, and 12 random seeds were analyzed. The permutation test confirms a significant global rebalancing effect for F2-score ($T_{\text{obs}} = 13.41$, $p < 0.001$), AUROC ($T_{\text{obs}} = 10.35$, $p < 0.001$), and AUPRC ($p < 0.001$).

Fig. 2 presents a variance-based sensitivity analysis (functional ANOVA decomposition) that decomposes total variance into main-effect and interaction contributions for each factor. **Mode** contributes the largest main effect ($S_M = 0.31$–$0.50$), followed by **Rebalancing** ($S_R = 0.24$–$0.29$). Both factors also participate in a substantial $R \times M$ interaction (21.2% of F2-score variance, 13.8% of AUROC variance). The total-order indices — which include all interactions — show that Mode accounts for $S_{TM} = 0.48$–$0.66$ and Rebalancing for $S_{TR} = 0.40$–$0.46$ of total variance. In contrast, **Distance** ($S_{TD} < 0.015$) and **Membership** ($S_{TG} < 0.031$) are negligible even when interactions are included. Residual variance (seed-to-seed variation) accounts for 7.9%–21.9%. This decomposition establishes the central narrative: rebalancing and training mode, including their interaction, account for $> 80$% of systematic variance; domain grouping strategy does not.

**$R \times M$ interaction concentration.** The coupling between Rebalancing and Mode is not only the largest interaction but effectively the *only* one. Defining the interaction concentration ratio $C_{ij}^{(i)} = S_{ij}/(S_{Ti} - S_i)$ — the share of factor $i$'s total interaction attributable to the pairwise term $(i,j)$ — yields:

| Metric | $S_{TM} - S_M$ | $C_{R \times M}^{(M)}$ | $S_{TR} - S_R$ | $C_{R \times M}^{(R)}$ |
|--------|:---:|:---:|:---:|:---:|
| F2-score | 0.226 | 94% | 0.221 | 96% |
| AUROC | 0.159 | 87% | 0.157 | 88% |
| AUPRC | 0.174 | 88% | 0.171 | 89% |

Mode's remaining interactions ($M \times D$, $M \times G$, higher-order terms) collectively account for only 6%–13% of its interaction budget. This concentration indicates that the factorial design's $2^k - k - 1 = 11$ interaction terms collapse into a single dominant pairwise effect.

**Membership ($G$) interaction decomposition.** Although Membership's total-order index ($S_{TG} = 0.018$–$0.031$) is 2–4$\times$ its main effect ($S_G = 0.002$–$0.009$), the absolute interaction contribution $S_{TG} - S_G = 0.014$–$0.024$ remains an order of magnitude below $S_{R \times M}$ (ratio $S_{R \times M}/S_{TG} = 4.5$–$12\times$). The largest $G$-related two-way interaction is $G \times M$ ($S_{G \times M} = 0.006$–$0.009$, 0.6%–0.9% of total variance), followed by $G \times R$ ($S_{G \times R} < 0.005$). The three-way $R \times G \times M$ term ($0.003$–$0.005$) is individually $< 0.5$% of total variance. Notably, 76%–82% of Membership's interaction budget involves Mode, consistent with the intuition that the relevance of domain membership depends on whether target-subject data is included in training. Yet even this $G \times M$ coupling is too small to alter any practical design decision.

Appendix F provides the confirmatory hypothesis tests and Bayesian evidence. A complementary one-factor-at-a-time (OFAT) analysis (Appendix E, Figs. S1–S4) visualises the per-condition effect of each factor, confirming the Sobol ranking and revealing that the $R$ effect is strongly mode-dependent.

![Sensitivity Analysis](../../../../results/analysis/exp2_domain_shift/figures/svg/split2/journal_v2/fig2_effect_hierarchy.svg)
*Fig. 2. Variance-based sensitivity analysis (Sobol indices) for the four experimental factors across F2-score, AUROC, and AUPRC. Solid bars show first-order indices ($S_i$, main effect); hatched bars show interaction contributions ($S_{Ti} - S_i$). Error bars indicate 95% bootstrap CIs on total-order indices ($S_{Ti}$). Mode and Rebalancing dominate; Distance and Membership are negligible.*

### 4.2 Rebalancing × Mode Interaction (RQ1, RQ3: H1, H4, H5)

The sensitivity analysis (§4.1) identified Rebalancing ($S_{TR} = 0.40$–$0.46$) and Mode ($S_{TM} = 0.48$–$0.66$) as the only practically important factors, with a substantial interaction ($S_{R \times M} = 0.12$–$0.21$) — together accounting for $> 80$% of systematic variance. This section analyses how these two factors interact and quantifies individual strategy performance.

**Mode dichotomy.** Training mode produces a binary performance split (Table 2). Within-domain and Mixed are statistically equivalent ($\delta < 0.05$), while Cross-domain collapses to near-chance levels ($\delta > 0.83$ vs. Within).

**Table 2.** Mean ± SD performance by rebalancing strategy and training mode.

| Strategy | F2 (overall) | AUROC (overall) | AUPRC (overall) | F2 (Cross) | F2 (Within) | F2 (Mixed) |
|----------|:------------:|:---------------:|:---------------:|:----------:|:-----------:|:----------:|
| Baseline | 0.215 ± 0.057 | 0.631 ± 0.123 | 0.135 ± 0.168 | 0.160 | 0.215 | 0.268 |
| RUS $r{=}0.1$ | 0.184 ± 0.049 | 0.594 ± 0.094 | 0.108 ± 0.122 | 0.165 | 0.178 | 0.208 |
| RUS $r{=}0.5$ | 0.162 ± 0.036 | 0.564 ± 0.064 | 0.072 ± 0.052 | 0.156 | 0.159 | 0.170 |
| SMOTE $r{=}0.1$ | 0.346 ± 0.158 | 0.765 ± 0.176 | 0.407 ± 0.286 | 0.138 | 0.448 | 0.452 |
| SMOTE $r{=}0.5$ | 0.376 ± 0.205 | 0.755 ± 0.172 | 0.394 ± 0.276 | 0.114 | 0.499 | 0.514 |
| SW-SMOTE $r{=}0.1$ | **0.416** ± 0.243 | **0.765** ± 0.183 | **0.447** ± 0.337 | 0.101 | 0.558 | **0.587** |
| SW-SMOTE $r{=}0.5$ | 0.304 ± 0.233 | 0.745 ± 0.166 | 0.274 ± 0.223 | 0.042 | 0.468 | 0.402 |

The best overall strategy is SW-SMOTE $r{=}0.1$ (within-domain F2 = 0.558, AUROC = 0.903, AUPRC = 0.648; +159%/+43%/+457% over Baseline). RUS degrades performance relative to Baseline ($\delta = -0.84$). Only in cross-domain — where absolute performance is near chance — does RUS show a marginal advantage.

**Ranking reversal.** The *identity* of the best strategy reverses between modes. Per-mode Friedman tests (all 9 tests $p < 10^{-5}$) show RUS $r{=}0.1$ ranks first in cross-domain while SW-SMOTE $r{=}0.1$ ranks first in within-domain and mixed (consistently across all three metrics). Cross-mode Spearman $\rho$ on the 7-strategy ranking vectors confirms the reversal: cross- vs. within-domain $\rho = -0.79$ to $-0.86$ ($p < 0.04$), while within- vs. mixed $\rho \geq +0.96$ ($p < 0.001$). Kendall's $W$ across the three modes is $0.17$–$0.22$ (non-significant), meaning the modes do *not* agree on strategy rankings — the defining signature of the $R \times M$ interaction.

Fig. 3 visualises this interaction. In the Cross-domain column, all 7 strategies collapse to a narrow low-performance band. In Within-domain and Mixed, SMOTE-based strategies separate sharply from Baseline and RUS.

![Strategy Comparison](../../../../results/analysis/exp2_domain_shift/figures/svg/split2/journal_v2/fig4_strategy_comparison.svg)
*Fig. 3. Performance distributions of the 7 rebalancing strategies by training mode (rows: F2, AUROC, AUPRC; columns: Cross, Within, Mixed). The Cross-domain column is uniformly compressed near chance; Within-domain and Mixed reveal large inter-strategy separation — the $R \times M$ interaction.*

**Strategy clusters.** Nemenyi post-hoc tests (CD = 2.600) identify two statistically separated clusters: oversampling methods (mean ranks 2.76–3.83) vs. Baseline/RUS (ranks 4.12–5.64). The top-ranked methods (SW-SMOTE $r{=}0.1$ for F2; SMOTE $r{=}0.1$ for AUROC/AUPRC) are statistically interchangeable (rank difference < CD). Sampling ratio $r=0.1$ is the more robust default across methods.

**Verdict — H1, H4, H5 strongly supported.** The 7 strategies differ significantly (53/54 omnibus tests, Friedman $p < 0.0001$). Mode creates a binary split ($\delta > 0.83$), and rebalancing effectiveness reverses across modes ($\rho = -0.79$ to $-0.86$). SMOTE-based oversampling transforms within-domain F2 from 0.215 to 0.558 (+159%), while RUS degrades it.

### 4.3 Robustness Validation

Distance ($S_{TD} < 0.015$, $BF_{01} = 71$–$767$) and Membership ($S_{TG} < 0.031$) are confirmed negligible by both frequentist and Bayesian tests; full statistical details are in Appendix F. Subsampling analysis shows $\sigma_{\text{rank}}$ decreases monotonically, reaching 0 (F2, AUPRC) or 0.147 (AUROC) by $k = 11$ of 12 seeds (Fig. 4), confirming sufficient seed count. Cross-metric concordance is moderate (Kendall’s $W = 0.643$; AUROC–AUPRC $\rho = 0.929$). Post-hoc power analysis indicates the pooled design ($n = 36$) detects $|\delta| \geq 0.53$, well below all reported large effects ($|\delta| > 0.8$).

![Seed Convergence](../../../../results/analysis/exp2_domain_shift/figures/svg/split2/journal_v2/fig8_seed_convergence.svg)
*Fig. 4. Ranking stability ($\sigma_{\text{rank}}$) as a function of seed subset size $k$. Monotonic convergence confirms that $n = 12$ seeds is sufficient.*

---

## 5. Discussion

### 5.1 Why Distance Metrics Are Irrelevant: A Vehicle Dynamics Explanation

The central unexpected finding — that three fundamentally different distance metrics produce equivalent downstream performance ($S_{TD} < 0.015$, $\eta^2 < 0.004$) — is particularly notable because the production pipeline applies global z-score normalisation before distance computation, ensuring that no single feature dimension dominates. Under normalisation, the three metrics generate genuinely different subject rankings:

| Pair | Normalised $\rho$ |
|------|:--------------------:|
| MMD vs Wasserstein | 0.757 |
| MMD vs DTW | 0.116 (n.s.) |
| Wasserstein vs DTW | 0.444 |

DTW diverges substantially from MMD and Wasserstein ($\rho = 0.12$, $p = 0.29$), meaning DTW — which operates on temporal trajectory shape — captures distributional properties genuinely distinct from the sample-level statistics used by MMD and Wasserstein. Consequently, 42.5% of subjects switch domain groups depending on the metric used. Yet this produces no measurable effect on classification ($\eta^2 < 0.004$, $BF_{01} = 71$–$767$; Appendix F). The Bayesian analysis is particularly important here: because H2 is a null claim ("distance does not matter"), the frequentist framework can only fail to reject — it cannot affirm the null. The Bayes factors provide this affirmation, placing the evidence firmly in the "very strong" to "extreme" range for all three metrics (Jeffreys, 1961). Three mechanisms explain this decoupling:

**Physical coupling reduces effective dimensionality.** The bicycle model (Eq. 1) couples four of five raw signals ($\delta$, $\dot{\delta}$, $a_y$, $e_{\text{lane}}$) through the chain $\delta \to \dot{\delta} \to a_y \to e_{\text{lane}}$, with only $a_x$ dynamically independent. PCA confirms this: 45 principal components explain 95% of variance from 135 features (3:1 compression), and the first PC alone explains 22.2%.

**Inter-subject discrimination is inherently weak.** The overall ICC ratio (inter-subject / total variance) is only 0.111, meaning 88.9% of feature variance is intra-subject (within-session variability). When subject positions in feature space are this "blurred," the coarse binary partition at the median (in-domain: 44 subjects, out-domain: 43 subjects) absorbs ranking differences without altering the downstream training set composition substantially.

**Rebalancing absorption.** The sensitivity analysis confirms rebalancing's dominance — the total-order Sobol index ratio is:

$$\frac{S_{TR}}{S_{TD}} \approx \frac{0.40\text{–}0.46}{<0.015} = 27\text{–}31\times \tag{6}$$

Rebalancing shifts the classifier's decision boundary so dramatically that any difference in training set composition due to domain grouping is overwhelmed.

**Supplementary unnormalised analysis.** To assess whether metric equivalence depends on normalisation, we also computed distance matrices on raw (unnormalised) features. Without normalisation, lane offset features ($O(10^3)$) dominate all other features ($O(10^{-1})$–$O(10^0)$), and all three metrics converge to higher cross-metric correlations ($\rho = 0.48$–$0.82$). This confirms that unnormalised distances would also produce equivalent downstream performance — but for a trivial reason: lane offset dominance forces all metrics toward the same ranking. The normalised production pipeline reveals the stronger result: even when metrics capture genuinely different distributional properties, the coarse binary partition and rebalancing absorption render the choice immaterial.

### 5.2 Rebalancing and Training Mode as the Dominant Design Factors

The functional ANOVA decomposition (Fig. 2) decomposes total variance into additive contributions. Let $V_{\mathrm{sys}} = V - V_{\varepsilon}$ denote the systematic (non-residual) variance. For F2-score the decomposition is:

$$\frac{S_M + S_R + S_{R \times M}}{1 - S_{\varepsilon}} = \frac{0.368 + 0.243 + 0.212}{1 - 0.157} = \frac{0.823}{0.843} = 97.5\% \tag{7}$$

Analogous ratios are 95.8\% (AUROC) and 96.1\% (AUPRC), confirming that Mode, Rebalancing, and their interaction jointly account for $>95$\% of systematic variance across all three primary metrics. In contrast, Distance and Membership together contribute $< 5$\% ($S_{TD} + S_{TG} < 0.046$).

The interaction term $S_{R \times M}$ is not merely statistically present but practically decisive: it produces a full ranking reversal between modes. The Spearman rank correlation between the 7-strategy ranking vectors in cross-domain vs. within-domain is:

$$\rho_{\text{cross,within}} = -0.74 \;\text{to}\; -0.89 \tag{8}$$

while within-domain vs. mixed yields $\rho \geq +0.99$ ($p < 0.001$). This sign reversal maps directly onto the Sobol interaction: the 21.2\% of F2 variance attributed to $R \times M$ manifests as a qualitative change in which strategy is optimal, not merely a quantitative shift in effect magnitude. The practical consequence is stark: a practitioner who selects a rebalancing strategy based on cross-domain results would deploy the *worst* strategy for within-domain operation, and vice versa (Table 2).

![Ranking Reversal](../../../../results/analysis/exp2_domain_shift/figures/svg/split2/journal_v2/fig_ranking_reversal.svg)
*Fig. 5. Strategy ranking reversal across training modes. Lines trace Friedman mean ranks (lower = better) for each of the 7 strategies. In Cross-domain, RUS variants rank highest; in Within-domain and Mixed, SMOTE-based strategies dominate — a complete inversion consistent with $\rho_{C,W} = -0.74$ to $-0.89$ (Eq. 8). The near-perfect overlap of Within and Mixed columns ($\rho \geq 0.99$) confirms mode equivalence.*


**Mechanistic interpretation.** The mode-dependent reversal has a data-geometric explanation. In within-domain and mixed modes, each subject’s own data is present in the training set, providing a subject-specific decision boundary. Here the bottleneck is class imbalance: minority-class (drowsy) epochs are scarce, and oversampling via SMOTE generates synthetic examples along the minority-class manifold, enriching the decision boundary without discarding majority-class information. In quantitative terms:

$$\Delta\text{F2}_{\text{SMOTE}} = \text{F2}_{\text{SW-SMOTE}} - \text{F2}_{\text{BL}} = 0.558 - 0.215 = +0.343 \;(+159\%) \tag{9}$$

RUS, by contrast, discards majority-class samples ($\delta_{\text{RUS vs BL}} = -0.84$), destroying the very information that calibrates the boundary for subject-specific patterns.

In cross-domain mode, target-subject data is entirely absent from training. The bottleneck shifts from class imbalance to domain mismatch: the classifier must generalise across subjects with different driving dynamics. Here, oversampling amplifies source-domain patterns that may not transfer, while RUS’s information loss is less damaging because the majority-class signal was already misaligned. All strategies collapse to near-chance levels (F2 $= 0.04$–$0.17$), and the inter-strategy variance shrinks dramatically — consistent with the $R \times M$ interaction absorbing 21\% of total F2 variance.

AUPRC improvements confirm that SMOTE’s gains are genuine minority-class precision improvements, not threshold-shifted recall: 0/36 strategy–mode cells exhibit a precision–recall trade-off; the dominant pattern is win–win (recall↑, precision stable). For comparison, switching distance metric yields $|\Delta\text{F2}| < 0.01$ — two orders of magnitude below the rebalancing effect (Eq. 9), consistent with the Sobol ratio $S_{TR}/S_{TD} > 27\times$ (Eq. 6).

### 5.3 The Domain Gap Reversal Phenomenon

An unexpected finding is that the domain gap reverses in within-domain and mixed training: out-domain subjects sometimes outperform in-domain subjects ($\Delta > 0$). Fig. 6 visualizes this pattern through diverging horizontal bars for each Rebalancing × Mode × Membership cell. Green bars (positive $\Delta$) indicate that out-domain performance exceeds in-domain. Across all three metrics (F2-score, AUROC, AUPRC), the majority of bars point green — especially in the Mixed mode panel — demonstrating that domain shift does not systematically degrade performance. This visual is consistent with the Wilcoxon test results (H8: 0/63 to 12/63 significant) and provides direct evidence that the domain split does not introduce a meaningful performance penalty.

![Domain Shift Direction](../../../../results/analysis/exp2_domain_shift/figures/svg/split2/journal_v2/fig7_domain_shift_reversal.svg)
*Fig. 6. Domain gap direction ($\Delta = \text{out} - \text{in}$) by Rebalancing × Mode. Green = out-domain outperforms in-domain (gap reversal). The prevalence of green bars, especially in Mixed mode, demonstrates that domain shift does not cause systematic performance degradation.*

The sensitivity analysis quantifies this phenomenon: the $G \times M$ interaction accounts for only 0.6%–0.9% of total variance ($S_{G \times M} = 0.006$–$0.009$), confirming that while the direction reversal is qualitatively notable, its magnitude is small relative to the dominant $R \times M$ interaction ($S_{R \times M} = 0.14$–$0.21$). As reported in §4.1, even Membership's total-order index ($S_{TG} = 0.018$–$0.031$) is 4.5–12$\times$ smaller than $S_{R \times M}$ alone. The reversal may be because:

1. Out-domain subjects exhibit more diverse driving patterns, providing richer training signal when their own data is included (within-domain setting).
2. The diversity of out-domain data acts as implicit regularization, improving generalization.
3. Rebalancing is more effective for subjects with varied drowsiness patterns (more minority-class samples to oversample).

### 5.4 Limitations

1. **Single classifier**: Results are based on one classifier type. While the rebalancing effect is a property of the training data rather than the model architecture, generalization to deep learning models requires verification.

2. **Deterministic data split**: The train/test partition (`subject_time_split`) is deterministic — random seeds only vary model initialization and resampling, not the data partition.

3. **Sample size constraints**: With $n = 12$ per cell, only large effects ($|\delta| > 0.53$) are detectable after Bonferroni correction. Subtle distance metric effects may exist below our detection threshold.

4. **Wilcoxon $p$-floor**: The minimum achievable p-value for Wilcoxon signed-rank with $n = 12$ is $1/2^{11} = 0.000488$, which is below the Bonferroni-corrected threshold, enabling paired tests to reach significance.

### 5.5 Future Directions

1. **Signal-specific distances**: Computing domain groups using individual signals (lane offset alone vs. all signals) would quantify the information contribution of each signal.
2. **Multiple classifiers**: Extending to LSTM and SVM-based architectures would strengthen the generalizability claim.
3. **Real-vehicle validation**: Transferring findings from simulator to on-road data is essential for practical deployment.

---

## 6. Conclusion

This study systematically evaluates the interplay between class imbalance handling and domain grouping strategies for drowsy driving detection through a factorial experiment with 87 subjects, 7 rebalancing strategies, and 1,512 observations, addressing three research questions with rigorous non-parametric statistics.

The key findings are:

**RQ1 — Class imbalance handling:**

1. **Training mode and rebalancing jointly dominate** ($S_{TM} = 0.48$–$0.66$; $S_{TR} = 0.40$–$0.46$): These two factors, together with their interaction ($S_{R \times M} = 0.12$–$0.21$), account for $>80$% of systematic variance. Among the 7 strategies evaluated, SW-SMOTE $r{=}0.1$ achieves the best within-domain performance (F2 = 0.558, AUROC = 0.903, AUPRC = 0.648), improving over Baseline by +159% (F2), +43% (AUROC), and +457% (AUPRC). The four SMOTE-based strategies form a top cluster (mean ranks 2.76–3.83), statistically separated from Baseline (4.12–4.73) and RUS (4.65–5.64). RUS consistently degrades performance relative to Baseline.

**RQ2 — Domain analysis:**

2. **Distance metric choice is irrelevant** ($S_{TD} < 0.015$, $BF_{01} = 71$–$767$): MMD, DTW, and Wasserstein produce statistically indistinguishable downstream performance despite generating genuinely different subject rankings under normalised features (DTW–MMD $\rho = 0.12$, n.s.) and 42.5% of subjects switching domain groups across metrics. Bayesian analysis provides affirmative evidence for the null hypothesis (\"very strong\" to \"extreme\" on the Jeffreys scale). Vehicle dynamics coupling, weak inter-subject discrimination (ICC = 0.111), and rebalancing absorption ($S_{TR}/S_{TD} > 27\times$) provide a physics-grounded explanation.

3. **Domain shift direction is context-dependent**: In-domain subjects do not consistently outperform out-domain subjects; within-domain and mixed training reverse the expected domain gap ($\Delta > 0$), with rebalancing amplifying the reversal. The $G \times M$ interaction accounts for only 0.6%–0.7% of total variance.

4. **Training mode hierarchy is clear**: Within-domain $\approx$ mixed $\gg$ cross-domain (F2: 0.36 vs. 0.13; AUROC: 0.77 vs. 0.52), with Cliff's $\delta = 0.83$–$0.95$ (large). Mode is the single largest source of variance ($S_M = 0.31$–$0.50$).

**RQ3 — Interaction:**

5. **Rebalancing strategy depends on training mode**: The optimal strategy varies by mode (RUS in cross-domain; SMOTE/SW-SMOTE in within-domain and mixed), revealing a strong $R \times M$ interaction ($S_{R \times M} = 0.12$–$0.21$, the third-largest effect). Domain membership ($G$) modulates the magnitude of performance differences but does not alter strategy selection ($S_{TG} < 0.031$; §4.2).

6. **Results are robust**: Consistent across 12 random seeds ($\sigma_{\text{rank}} \to 0$ at $k = 9$), 5 evaluation metrics (Kendall's $W = 0.643$), 2 sampling ratios (91% directional agreement), and confirmed by permutation test ($p < 0.001$).

For practitioners, these results prescribe a clear strategy: use within-domain or mixed training (the largest single factor), apply SMOTE-based class rebalancing (the largest controllable preprocessing step), and choose any convenient distance metric for domain grouping. The combined effect of training mode and rebalancing — including their interaction — accounts for over 80% of systematic variance, dwarfing all domain-related design choices.

---

## References

- Arefnezhad, S., et al. (2019). Driver drowsiness estimation using EEG signals with a dynamical encoder–decoder. *IET Intelligent Transport Systems*, 13(2), 301–310.
- Atiquzzaman, M., et al. (2018). Real-time detection of drivers' texting and eating behavior based on vehicle dynamics. *Transportation Research Part F*, 58, 594–604.
- Zhao, C., et al. (2012). Driver drowsiness detection using continuous wavelet transform. *Proceedings of the IEEE International Conference on Signal Processing*, 1–4.
- Berndt, D. J., & Clifford, J. (1994). Using dynamic time warping to find patterns in time series. *AAAI Workshop on Knowledge Discovery in Databases*, 359–370.
- Chawla, N. V., et al. (2002). SMOTE: Synthetic minority over-sampling technique. *JAIR*, 16, 321–357.
- Cliff, N. (1993). Dominance statistics: Ordinal analyses to answer ordinal questions. *Psychological Bulletin*, 114(3), 494–509.
- Friedman, M. (1937). The use of ranks to avoid the assumption of normality. *JASA*, 32(200), 675–701.
- Gretton, A., et al. (2012). A kernel two-sample test. *JMLR*, 13(1), 723–773.
- He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE TKDE*, 21(9), 1263–1284.
- Jeffreys, H. (1961). *Theory of Probability* (3rd ed.). Oxford University Press.
- Kruskal, W. H., & Wallis, W. A. (1952). Use of ranks in one-criterion variance analysis. *JASA*, 47(260), 583–621.
- Mann, H. B., & Whitney, D. R. (1947). On a test of whether one of two random variables is stochastically larger. *Annals of Mathematical Statistics*, 18(1), 50–60.
- Masson, M. E. J. (2011). A tutorial on a practical Bayesian alternative to null-hypothesis significance testing. *Behavior Research Methods*, 43(3), 679–690.
- Nemenyi, P. (1963). *Distribution-free multiple comparisons*. PhD thesis, Princeton University.
- Pan, S. J., & Yang, Q. (2010). A survey on transfer learning. *IEEE TKDE*, 22(10), 1345–1359.
- Rajamani, R. (2012). *Vehicle Dynamics and Control* (2nd ed.). Springer.
- Saltelli, A., Ratto, M., Andres, T., Campolongo, F., Cariboni, J., Gatelli, D., Saisana, M., & Tarantola, S. (2008). *Global Sensitivity Analysis: The Primer*. Wiley.
- Saito, T., & Rehmsmeier, M. (2015). The precision–recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE*, 10(3), e0118432.
- Shannon, C. E. (1948). A mathematical theory of communication. *Bell System Technical Journal*, 27(3), 379–423.
- Villani, C. (2009). *Optimal Transport: Old and New*. Springer.
- Wagenmakers, E.-J. (2007). A practical solution to the pervasive problems of $p$ values. *Psychonomic Bulletin & Review*, 14(5), 779–804.
- Wang, X., et al. (2022). Real-time detection of driver drowsiness using LSTM. *Sensors*, 22(13), 4904.
- Wilcoxon, F. (1945). Individual comparisons by ranking methods. *Biometrics Bulletin*, 1(6), 80–83.

---

## Appendix A: Primary Hypothesis Summary Table

| # | Hypothesis | Verdict | Key Evidence |
|:-:|-----------|:-------:|-------------|
| H1 | Rebalancing methods and sampling ratio affect performance | ✓ Strongly supported | 4-group Kruskal-Wallis significant in 53/54 cells ($\eta^2 = 0.06$–$0.80$). Post-hoc ordering is mode-dependent: OS > BL > RUS (within/mixed); RUS ≈ OS ≈ BL (cross-domain). SMOTE ≈ SW-SMOTE. Ratio: $r=0.1$ preferred overall; SMOTE prefers $r=0.5$ on F2 only |
| H2 | Distance metric matters | ✗ Negligible | $\eta^2 < 0.004$, all metrics equivalent |
| H3 | Domain membership affects performance | ✓ Strongly supported (direction is mode-dependent) | Significant in 11–16/63 cells; cross-domain: in-domain advantage ($\Delta < 0$); within-domain/mixed: out-domain advantage ($\Delta > 0$) |
| H4 | Within-domain > cross-domain | ✓ Fully supported | $\delta = +0.833$ (F2), $+0.945$ (AUROC) |
| H5 | Rebalancing × Mode × Membership interaction | ✓ Strong ($R \times M$); secondary ($G$) | Best method varies by mode; membership modulates magnitude but not strategy selection |

## Appendix B: Extended Metrics Rebalancing Strategy Rankings

| Rank | F2 (mean rank) | AUROC | F1 | AUPRC | Recall |
|:----:|:--------------:|:-----:|:--:|:-----:|:------:|
| 1 | sw\_smote\_r01 (2.67) | smote\_r01 (2.22) | sw\_smote\_r01 (2.94) | sw\_smote\_r01 (2.44) | smote\_r01 (2.11) |
| 2 | smote\_r05 (3.11) | sw\_smote\_r01 (2.89) | sw\_smote\_r05 (3.33) | smote\_r01 (2.89) | baseline (2.89) |
| 3 | smote\_r01 (3.56) | smote\_r05 (3.39) | smote\_r01 (3.39) | smote\_r05 (3.28) | sw\_smote\_r01 (3.61) |

## Appendix C: Supplementary Hypothesis Results

The following 6 hypotheses were tested as part of the comprehensive analysis framework but are reported as supplementary findings. They either overlap with the primary hypotheses (H1–H5) or produced weaker/context-dependent effects.

| # | Hypothesis | Verdict | Key Evidence | Relation to Primary |
|:-:|-----------|:-------:|-------------|:-------------------:|
| H6 | Wasserstein most discriminative | ✗ Not supported | All metrics produce equivalent groupings | Redundant given H2 |
| H7 | Mixed > cross-domain | ✓ Fully supported | F2: 0.372 vs. 0.125 | Corollary of H4 |
| H8 | Domain gap larger in cross-domain | ✓ Supported | Cross-domain $\Delta < 0$; within-domain reverses | Overlaps H3 |
| H9 | Oversampling reduces domain gap | ✗ Mixed | Context-dependent; sometimes increases gap | Weak interaction effect |
| H10 | Rebalancing × Distance interaction | ✓ Weak | 12/18 consistent; 6/18 minor swaps | Negligible given H2 |
| H11 | Membership × Mode interaction | ✓ Supported | Within-domain reverses domain shift | Extension of H3 + H4 |

## Appendix D: Statistical Methods–Hypothesis Correspondence Table

### D.1 Methods Applied per Hypothesis

| Hypothesis | Omnibus test | Post-hoc / pairwise | Effect size | Correction | Bootstrap | Section |
|:----------:|:------------|:--------------------|:-----------|:-----------|:----------|:--------|
| H1 | Kruskal-Wallis $H$ (4 groups, 18 cells) | Mann-Whitney $U$: OS vs RUS (48 pairs), RUS vs BL (12 pairs), SMOTE vs SW-SMOTE (6 pairs); $r=0.1$ vs $r=0.5$ (18 pairs) | Cliff's $\delta$, $\eta^2$ | Bonf. $\alpha'=0.00104$ (OS–RUS), $0.00417$ (RUS–BL), $0.00278$ (SM–SW, ratio) | Percentile $B=2{,}000$ | §4.2 |

*Note: H1 is analysed jointly with H4 and H5 in §4.2.*
| H2 | Kruskal-Wallis $H$ (6 cells) | Mann-Whitney $U$ (pooled) | Cliff's $\delta$, $\eta^2$ | Bonf. $\alpha'=0.0028$ | — | App. F |
| H3 | — | Wilcoxon signed-rank (63 pairs) | Mean $\lvert\Delta\rvert$, Cliff's $\delta$ | Bonf. $\alpha'=0.00079$ | — | App. F |
| H4 | Friedman $\chi_F^2$ (14 strategies) | Nemenyi post-hoc (CD = 2.600) | Cliff's $\delta$, Kendall's $W$ | — | — | §4.2 |
| H5 | Friedman $\chi_F^2$ (per mode, 9 tests) | Spearman $\rho$ (cross-mode ranking concordance) | Kendall's $W$ (3-mode) | — | — | §4.2 |

### D.2 Methods Applied per Purpose

| Purpose | Method | Parameters | Hypotheses |
|---------|--------|-----------|:----------:|
| Global rebalancing effect | Permutation test | $B=10{,}000$; $H_0$: label exchangeability | H1 |
| $k$-group comparison | Kruskal-Wallis $H$ | $\alpha'=\alpha/m$ | H1, H2 |
| Paired $k$-group ranking | Friedman $\chi_F^2$ | 7 strategies × 18 cells | H4, H5 |
| Pairwise unpaired | Mann-Whitney $U$ | Per-cell ($n=12$) or pooled ($n=36$) | H1, H2 |
| Pairwise paired | Wilcoxon signed-rank | In-domain vs. out-domain pairs | H3, H8, H9 |
| Post-hoc ranking | Nemenyi test | $\text{CD} = q_\alpha \sqrt{k(k+1)/6n}$ | H4 |
| Subject ranking concordance | Spearman $\rho$, Kendall $\tau$ | 3 metric pairs | H2 (App. F) |
| Cross-metric agreement | Kendall's $W$ | $k=7$ metrics, $n=7$ strategies | Robustness (§4.3.2) |
| Effect size (rank-based) | Cliff's $\delta$ | Thresholds: negligible/small/medium/large | H1, H2, H4 |
| Effect size (variance) | $\eta^2 = H / (N-1)$ | Proportion of variance explained | H1, H2 |
| Sensitivity analysis | Sobol indices ($S_i$, $S_{Ti}$) | Functional ANOVA decomposition; bootstrap $B=2{,}000$ | §4.1 |
| Bayesian null support | $BF_{01}$ (BIC approx.) | Jeffreys scale; $BF_{01} > 10$: strong $H_0$ support | H2 (App. F) |
| Effect size CI | Percentile bootstrap | $B=2{,}000$ | H1 |
| Population mean CI | BCa bootstrap | $B=10{,}000$ | All |
| Multiple testing | Bonferroni | $\alpha'=\alpha/m$, $\alpha=0.05$ | All |
| Post-hoc power | Mann-Whitney detectable $\lvert\delta_{\min}\rvert$ | Per-cell: 0.923; pooled: 0.533 | §4.3.3 |

## Appendix E: One-Factor-at-a-Time (OFAT) Analysis

The Sobol decomposition in §4.1 quantifies each factor's contribution to total variance but, as a global summary, does not reveal how the effect varies across individual experimental conditions. To provide this complementary view, we perform a one-factor-at-a-time (OFAT) analysis: for each target factor, we fix every other factor at each of its levels, sweep the target factor, and plot the resulting performance trajectory. Each thin line in Figs. S1–S4 represents one such fixed condition (averaged over 12 seeds); the bold black line is the grand OFAT mean and the grey band indicates ±1 SD across conditions.

**Rebalancing ($R$; Fig. S1).** SMOTE and SW-SMOTE variants consistently outperform Baseline and RUS across all 18 fixed conditions ($D \times G \times M$). The spread across conditions (Δ = 0.08–0.55 for F2-score) reflects the $R \times M$ interaction: cross-domain conditions cluster at the bottom with small rebalancing gains, while within-domain and mixed conditions fan upward with large SMOTE-driven improvements.

**Distance ($D$; Fig. S2).** The OFAT mean is flat across MMD, DTW, and Wasserstein, and individual condition lines remain nearly parallel. This confirms that $D$ is negligible regardless of which other factors are held fixed ($S_{TD} < 0.015$).

**Mode ($M$; Fig. S3).** Every condition shows a monotonic rise from cross-domain to within-domain/mixed. At the "Cross" level, all lines converge near floor performance (F2 ≈ 0.05–0.20); at "Within" and "Mixed", the lines fan out according to the rebalancing strategy, again evidencing the $R \times M$ interaction.

**Membership ($G$; Fig. S4).** The OFAT mean shows a slight upward slope from In to Out, but individual lines diverge in both directions. The effect is condition-dependent and averages to near zero, consistent with $S_{TG} < 0.031$.

Taken together, the OFAT analysis corroborates the Sobol ranking ($M > R \gg D \approx G$) while providing condition-level granularity that the variance decomposition summarises away.

![OFAT Rebalancing](../../../../results/analysis/exp2_domain_shift/figures/svg/split2/journal_v2/fig_s_ofat_condition.svg)
*Fig. S1. OFAT analysis for the Rebalancing factor ($R$). Each line represents one of 18 fixed conditions ($D \times G \times M$), averaged over 12 seeds. Bold line: grand OFAT mean; grey band: ±1 SD.*

![OFAT Distance](../../../../results/analysis/exp2_domain_shift/figures/svg/split2/journal_v2/fig_s_ofat_distance.svg)
*Fig. S2. OFAT analysis for the Distance factor ($D$). Each line represents one of 42 fixed conditions ($R \times G \times M$).*

![OFAT Mode](../../../../results/analysis/exp2_domain_shift/figures/svg/split2/journal_v2/fig_s_ofat_mode.svg)
*Fig. S3. OFAT analysis for the Mode factor ($M$). Each line represents one of 42 fixed conditions ($R \times D \times G$).*

![OFAT Membership](../../../../results/analysis/exp2_domain_shift/figures/svg/split2/journal_v2/fig_s_ofat_level.svg)
*Fig. S4. OFAT analysis for the Membership factor ($G$). Each line represents one of 63 fixed conditions ($R \times D \times M$).*

## Appendix F: Distance Metric Equivalence — Full Statistical Analysis (RQ2: H2, H3)

This appendix provides the complete statistical tests for the distance metric and domain membership effects summarised in §4.3. The sensitivity analysis (§4.1) established that Distance ($S_{TD} < 0.015$) and Membership ($S_{TG} < 0.031$) are negligible factors. The following subsections provide confirmatory evidence.

### F.1 Distance Metric Effect (H2)

Consistent with the near-zero Sobol index ($S_{TD} < 0.015$), the Kruskal–Wallis test (6 cells = mode × membership, pooling strategies) confirms negligible distance metric effects, with isolated statistical significance confined to cross-domain AUROC cells:

| Metric | Bonf. significant cells | Mean $\eta^2$ | Max $\eta^2$ |
|--------|:-----------------------:|:-------------:|:------------:|
| F2-score | 0/6 | 0.004 | 0.023 |
| AUROC | 2/6 | 0.089 | 0.322 |
| AUPRC | 2/6 | 0.038 | 0.117 |

The elevated AUROC $\eta^2$ is driven entirely by the two cross-domain cells, where all strategies perform near chance (AUROC $\approx$ 0.51–0.53) and low within-group variance inflates the statistic despite absolute mean differences of < 2 percentage points. In the fully pooled analysis, $\eta^2 < 0.003$ across all metrics.

Pooled performance across all strategies:

| Metric | MMD | DTW | Wasserstein | Max $\lvert\delta\rvert$ |
|--------|:---:|:---:|:-----------:|:--------------------------:|
| F2-score | 0.286 ± 0.185 | 0.281 ± 0.187 | 0.290 ± 0.192 | 0.044 |
| AUROC | 0.687 ± 0.166 | 0.683 ± 0.169 | 0.696 ± 0.167 | 0.086 |
| AUPRC | 0.258 ± 0.269 | 0.260 ± 0.270 | 0.270 ± 0.275 | 0.055 |

All pooled pairwise Cliff's $\delta$ values are negligible ($|\delta| < 0.09$).

**Bayesian evidence for equivalence.** Because the frequentist tests can only fail to reject $H_0$ — not affirm it — we supplement with a Bayesian analysis using the BIC approximation to the Bayes factor (Masson, 2011; Wagenmakers, 2007). The omnibus $BF_{01}$ (evidence favouring $H_0$: no distance effect) is:

| Metric | $H$ | $p$ | $BF_{01}$ | Interpretation |
|--------|:---:|:---:|:---------:|:--------------:|
| F2-score | 1.36 | 0.507 | 767 | Extreme evidence for $H_0$ |
| AUROC | 6.12 | 0.047 | 71 | Very strong evidence for $H_0$ |
| AUPRC | 2.76 | 0.252 | 381 | Extreme evidence for $H_0$ |

All three Bayes factors exceed $BF_{01} > 70$, providing "very strong" to "extreme" evidence on the Jeffreys (1961) scale that the distance metric has no effect on downstream classification. Even for AUROC — where the frequentist $p = 0.047$ narrowly crosses the uncorrected $\alpha = 0.05$ threshold — the Bayesian analysis firmly favours $H_0$, illustrating a classic large-$N$ divergence between $p$-values and Bayes factors.

![Distance Metric Violin](../../../../results/analysis/exp2_domain_shift/figures/svg/split2/journal_v2/fig5_distance_violin.svg)
*Fig. F1. Performance distributions by distance metric (Within-domain and Mixed modes). F2-score, AUROC, and AUPRC distributions are virtually indistinguishable across MMD, DTW, and Wasserstein, consistent with $|\delta| < 0.15$ for all pairwise comparisons.*

### F.2 Domain Membership Effect (H3)

Consistent with the negligible Membership Sobol index ($S_{TG} < 0.031$), the domain gap $\Delta = Y_{\text{out}} - Y_{\text{in}}$ is generally small and non-significant:

| Metric | Significant / 63 | Mean $\lvert\Delta\rvert$ range |
|--------|:-----------------:|:-------------------------------:|
| F2-score | 11 | 0.032–0.043 |
| AUROC | 12 | 0.018–0.092 |
| AUPRC | 16 | 0.016–0.114 |

All 63 Wilcoxon signed-rank tests per metric (7 strategies × 3 modes × 3 distances, paired by seed) use Bonferroni $\alpha' = 0.00079$. Within-domain and mixed training often show **positive** $\Delta$ (out-domain outperforms in-domain), a reversal discussed in §5.3.

**Verdict — H2 strongly supported; H3 not supported.** Distance metric choice does not influence classification performance ($BF_{01} > 70$, all $|\delta| < 0.09$). Domain membership effects are negligible per Sobol and do not alter strategy rankings.
