# Experiment 2 — Vehicle Dynamics Formulation and Quantitative Feature Analysis

This document provides the mathematical definitions of the vehicle motion signals and extracted features used in the domain shift analysis, and discusses the domain analysis results quantitatively through the lens of vehicle dynamics.

---

## 1. Raw Vehicle Motion Signals

Five raw signals are extracted from the driving simulator (SIMlsl, $f_s = 60\,\text{Hz}$):

| Symbol | Signal | MAT Index | Unit | Physical Meaning |
|:------:|--------|:---------:|:----:|------------------|
| $\delta(t)$ | Steering angle | 29 | rad | Steering wheel rotation angle |
| $\dot{\delta}(t)$ | Steering speed | — (derived) | rad/s | Rate of steering wheel rotation |
| $a_y(t)$ | Lateral acceleration | 19 | m/s² | Vehicle acceleration in the lateral (cross-track) direction |
| $a_x(t)$ | Longitudinal acceleration | 18 | m/s² | Vehicle acceleration in the forward direction |
| $e_{\text{lane}}(t)$ | Lane offset | 27 | m | Lateral displacement from lane center |

**Steering speed derivation**: The steering speed is derived from the steering angle via numerical differentiation:

$$\dot{\delta}(t) \approx \nabla\delta(t) \cdot f_s = \frac{\delta(t + \Delta t) - \delta(t)}{\Delta t}$$

where $\Delta t = 1/f_s = 1/60\,\text{s}$ and `np.gradient` is used for central differences.

---

## 2. Vehicle Dynamics Coupling

### 2.1 Bicycle Model

The five raw signals are not independent. Under the linear bicycle model (Rajamani, 2012), the lateral dynamics of a vehicle are governed by:

$$a_y = v \cdot \dot{\psi} = v \cdot \frac{\dot{\delta}}{L}$$

where:
- $v$ is the vehicle speed [m/s]
- $\dot{\psi}$ is the yaw rate [rad/s]
- $L$ is the wheelbase [m]

This equation establishes a **direct physical coupling** among three of the five raw signals: $\delta$, $\dot{\delta}$, and $a_y$.

### 2.2 Lane Offset Dynamics

The lane offset evolves as the double integral of the lateral acceleration deviation from the road curvature:

$$\ddot{e}_{\text{lane}}(t) = a_y(t) - a_{y,\text{road}}(t)$$

On straight or constant-curvature sections, $a_{y,\text{road}}$ is approximately constant, so lane offset is directly coupled to lateral acceleration. This creates a causal chain:

$$\delta(t) \;\xrightarrow{\text{diff.}}\; \dot{\delta}(t) \;\xrightarrow{\times\, v/L}\; a_y(t) \;\xrightarrow{\text{double int.}}\; e_{\text{lane}}(t)$$

### 2.3 Implications for Feature Independence

Of the 5 raw signals, 4 ($\delta$, $\dot{\delta}$, $a_y$, $e_{\text{lane}}$) are physically coupled through the bicycle model. Only $a_x$ is dynamically independent (governed by throttle/brake input). This coupling means the 135 extracted features contain substantial redundancy — the effective dimensionality should be much lower than 135.

**Empirical confirmation**: PCA analysis shows that **45 principal components** explain 95% of variance from the 135-dimensional feature space, and the first PC alone explains 22.2%. This is consistent with the 4:1 coupling ratio (4 coupled signals out of 5).

---

## 3. Feature Extraction Formulations

### 3.1 Statistical Features (22 features × 2 signals = 44 dimensions; time-frequency domain)

For each signal $x(t)$ in a sliding window $[t_0, t_0 + W]$ with $N = W \cdot f_s$ samples:

| Feature | Formula | Physical Interpretation |
|---------|---------|------------------------|
| Mean | $\bar{x} = \frac{1}{N}\sum_{i=1}^{N} x_i$ | DC component of the signal |
| Variance | $\sigma^2 = \frac{1}{N}\sum_{i=1}^{N}(x_i - \bar{x})^2$ | Signal power (dispersion) |
| Range | $R = x_{\max} - x_{\min}$ | Peak-to-peak amplitude |
| Skewness | $\gamma_1 = \frac{1}{N}\sum\left(\frac{x_i - \bar{x}}{\sigma}\right)^3$ | Asymmetry of the amplitude distribution |
| Kurtosis | $\gamma_2 = \frac{1}{N}\sum\left(\frac{x_i - \bar{x}}{\sigma}\right)^4 - 3$ | Tail heaviness (impulsiveness) |
| IQR | $Q_{75} - Q_{25}$ | Robust measure of dispersion |
| CV | $\sigma / |\bar{x}|$ | Relative variability |
| ZCR | $\frac{1}{N-1}\sum_{i=1}^{N-1} \mathbb{1}[\text{sgn}(x_i) \neq \text{sgn}(x_{i+1})]$ | Oscillation frequency proxy |

### 3.2 Spectral Features (per signal; time-frequency domain)

From the FFT of the windowed signal, $X(f) = \text{FFT}(x)$, with power spectrum $P(f) = |X(f)|^2$ in the band $[0.5, 30]\,\text{Hz}$:

| Feature | Formula |
|---------|---------|
| FreqVar | $\text{Var}(f_{\text{band}})$ — Variance of the frequency axis within the band |
| SpectralEntropy | $H_s = -\sum_f \hat{P}(f)\log_2 \hat{P}(f)$, $\hat{P}(f) = P(f)/\sum P$ |
| DominantFreq | $f^* = \arg\max_f P(f)$ |
| FreqCOG | $\bar{f} = \frac{\sum f \cdot P(f)}{\sum P(f)}$ — Spectral centroid |
| AvgPSD | $\frac{1}{|B|}\sum_{f \in B} P(f)$ — Mean power spectral density |

**Note on FreqVar**: FreqVar is the variance of the frequency *axis values* within the band, not of the power. It is identical for all subjects and signals when the window length is the same (since the FFT frequency grid is deterministic). This explains the ICC = 0.990 observed — it does not reflect driver-specific spectral signatures but rather a computational artifact. Features where ICC approaches 0.990 uniformly (all 5 FreqVar features) should be interpreted with this caveat.

### 3.3 Smooth/Std/PE Features (3 features × 5 signals = 15 dimensions)

For each signal $x(t)$, three features are extracted per sliding window:

| Feature | Formula | Reference |
|---------|---------|-----------|
| Std. deviation | $\sigma_w = \text{std}(x_{[t_0:t_0+W]})$ | — |
| Mean | $\bar{x}_w = \text{mean}(x_{[t_0:t_0+W]})$ | — |
| Prediction error | $e_p = |x_N - \hat{x}_N|$ | Atiquzzaman et al. (2018) |

The prediction error uses a 2nd-order Taylor approximation:

$$\hat{x}_N = 2.5\,x_{N-2} - 2.0\,x_{N-3} + 0.5\,x_{N-4}$$

For LSTM models, Gaussian smoothing is applied prior to feature extraction (Wang et al. 2022, Section 3.2):

$$x_{\text{smooth}}(t) = (x * g_\sigma)(t), \quad \sigma = \frac{N_{\text{gauss}}}{6}, \quad N_{\text{gauss}} = 1.0 \cdot f_s = 60$$

### 3.4 Permutation Entropy Features (8 patterns × 5 signals = 40 dimensions)

Permutation entropy (PE) captures the complexity of ordinal pattern sequences. For each signal, a 3-character sequence of the operations D (Decrease), A (Ascend) is computed at each time step, then the relative frequency of each of the $2^3 = 8$ patterns is used as a feature:

$$\text{PE}_{\pi}(x) = \frac{\#\{t : \text{pattern}(x_{t-2}, x_{t-1}, x_t) = \pi\}}{N - 2}$$

where $\pi \in \{\text{DDD, DDA, DAD, DAA, ADD, ADA, AAD, AAA}\}$.

---

## 4. Quantitative Discussion of Domain Analysis Results

### 4.1 Lane Offset Dominance in Variance Decomposition

The per-signal variance contribution analysis reveals:

| Signal | N features | Contribution to total inter-subject variance |
|--------|:----------:|:---------------------------------------------:|
| Lane offset ($e_{\text{lane}}$) | ~27 | **~100%** |
| All others combined | ~108 | <1% |

This dominance is explained by the **feature scale heterogeneity**:

$$\frac{\text{Range}(e_{\text{lane}}\text{ features})}{\text{Range}(\text{other features})} \approx \frac{10^3}{10^{-1} \sim 10^0} \approx 10^{3} \sim 10^{4}$$

Since distance metrics (MMD, DTW, Wasserstein) operate on unnormalized features, the inter-subject variance is dominated by lane offset features whose absolute magnitudes are $O(10^3)$ while other signals produce features at $O(10^{-1})$–$O(10^0)$.

**Physical interpretation**: Lane offset has a characteristically large range because:
1. It is measured in meters with values spanning $\pm 1$–$2\,\text{m}$
2. Its derived features (Mean, Range, Energy) scale as $O(e_{\text{lane}})$ or $O(e_{\text{lane}}^2)$
3. In contrast, steering angle (rad with fractional values), accelerations ($<1\,\text{m/s}^2$ for normal driving) all have much smaller absolute values

### 4.2 Physical Coupling and Effective Dimensionality

The bicycle model coupling $\delta \to \dot{\delta} \to a_y \to e_{\text{lane}}$ creates a hierarchical correlation structure:

| Feature Pair | Expected Correlation | Mechanism |
|-------------|:--------------------:|-----------|
| $\delta$ ↔ $\dot{\delta}$ | High | Derivative relationship |
| $\dot{\delta}$ ↔ $a_y$ | High | $a_y \propto \dot{\delta} \cdot v/L$ |
| $a_y$ ↔ $e_{\text{lane}}$ | Moderate–High | Double integration with noise |
| $\delta$ ↔ $e_{\text{lane}}$ | Moderate | Indirect (2 layers of coupling) |
| $a_x$ ↔ others | Low | Dynamically independent |

This correlation structure is confirmed by the PCA analysis:
- **229 feature pairs** have $|r| > 0.8$ (out of $\binom{135}{2} = 9045$ possible pairs)
- **138 feature pairs** have $|r| > 0.9$
- Effective dimensionality = **45** (95% variance) from 135 original dimensions

The compression ratio of $135 \to 45$ (3:1) is consistent with having $\approx 4$ coupled signals contributing correlated features plus 1 independent signal ($a_x$).

### 4.3 Why Distance Metrics Produce Equivalent Domain Groupings

The actual distance concordance analysis using MMD, DTW, and Wasserstein metrics shows Spearman rank correlations $\rho = 0.48$–$0.80$ between metrics, with 59.8% of subjects switching domain groups across metrics. Despite this non-trivial group switching, the **downstream performance effect** is negligible:

$$\eta^2_{\text{distance}} < 0.004 \quad (\text{for all metrics: F2, AUROC, F1, AUPRC, Recall})$$

This can be quantitatively explained through the vehicle dynamics:

1. **Lane offset dominance**: Because $e_{\text{lane}}$ features have ranges $O(10^3)$, they dominate all three distance metrics equally. The subject ranking is effectively determined by lane offset behavior, regardless of which distance metric is used.

2. **Signal coupling reduces information gain**: Adding $\delta$, $\dot{\delta}$, and $a_y$ features to the distance computation provides limited additional information beyond $e_{\text{lane}}$ because they are physically coupled:

$$I(\delta, \dot{\delta}, a_y ; e_{\text{lane}}) \approx I(\delta, \dot{\delta}, a_y) \quad \text{(near-complete redundancy)}$$

3. **Rebalancing absorption**: The condition (imbalance handling) effect is orders of magnitude larger:

$$\frac{\eta^2_{\text{condition}}}{\eta^2_{\text{distance}}} \approx 1100\times \text{(F2)}, \quad 35\times \text{(AUROC)}$$

   Even when distance metrics produce different domain groupings, the classifier's decision boundary shift from SMOTE/RUS/SW-SMOTE overwhelms any subtle grouping difference.

### 4.4 ICC Ratio Interpretation Through Vehicle Dynamics

The mean ICC ratio (inter-subject / total variance) is only 0.111, meaning 88.9% of feature variance is intra-subject (within-session variability). Through the lens of vehicle dynamics:

- **Low ICC for $a_y$ and $a_x$ features** (mean 0.088, 0.157): Accelerations are highly variable within a single driving session due to traffic, road geometry, and momentary adjustments. The standard deviation of acceleration varies greatly from window to window but is relatively similar across drivers in comparable road conditions.

- **Moderate ICC for $e_{\text{lane}}$ features** (0.173): Lane offset captures habitual positioning (some drivers consistently drive closer to the center, others to the edge), giving slightly higher inter-subject discrimination.

- **Highest ICC for FreqVar** (0.990): As discussed in §3.2, this is a computational artifact — FreqVar of the frequency axis is constant for fixed window length, so all variation is due to the number of valid windows per subject. This feature does not carry meaningful physical information about driving behavior.

- **Low ICC for permutation entropy** (mean 0.061): Ordinal pattern distributions ($\pi \in \{\text{DDD}, \ldots, \text{AAA}\}$) are dominated by the local fluctuation structure, which is more influenced by the immediate driving context (curves, intersections) than by stable driver traits.

### 4.5 Prediction Error as a Fatigue Indicator

The prediction error feature $e_p = |x_N - \hat{x}_N|$ based on the Taylor approximation represents how well a smooth (2nd-order polynomial) trajectory predicts the next sample. Under vehicle dynamics:

- For **steering** and **lateral acceleration**: Drowsy drivers exhibit more erratic corrections, increasing prediction error because the smooth polynomial model fails to capture sudden micro-corrections
- For **lane offset**: Drowsy drivers show characteristic low-frequency drift followed by abrupt corrections (the "drift-and-jerk" pattern), which also increases $e_p$

The low ICC of prediction error features (e.g., `lat_acc_pred_error` ICC = 0.004, `lane_offset_pred_error` ICC = 0.001) indicates that this fatigue indicator varies more *within* a session (as drowsiness develops) than *between* subjects, which is exactly the desired property for a drowsiness detection feature — it should capture state changes rather than trait differences.

---

## 5. Implications for Experiment Design

### 5.1 Feature Normalization

The extreme scale heterogeneity ($e_{\text{lane}}$ features dominating by $O(10^3)$) suggests that **standardized features** should be used when computing domain distances. Currently, the raw (unnormalized) feature space is used for MMD/DTW/Wasserstein computation, which means the domain grouping is effectively a function of lane offset behavior alone. Future experiments should compare:

$$d_{\text{normalized}}(s_i, s_j) = d\!\left(\frac{\mathbf{x}_i - \boldsymbol{\mu}}{\boldsymbol{\sigma}}, \frac{\mathbf{x}_j - \boldsymbol{\mu}}{\boldsymbol{\sigma}}\right)$$

### 5.2 Feature Selection for Distance Computation

Given the physical coupling, computing distances on all 135 features introduces noise without adding information. A more principled approach would use:
1. The 45 principal components (capturing 95% variance), or
2. One representative feature per physical signal (e.g., Range or StdDev), giving 5 dimensions

### 5.3 Interpretation of Domain Shift

The domain shift effect (in-domain vs out-domain) observed in the experimental results should be interpreted as primarily reflecting **lane offset behavioral similarity** rather than overall driving style similarity. Subjects in the "out-domain" group are those with the most extreme lane offset patterns, which may reflect different lane-keeping strategies, road familiarity, or drowsiness susceptibility.

---

## 6. References

- Rajamani, R. (2012). *Vehicle Dynamics and Control* (2nd ed.). Springer.
- Atiquzzaman, M. et al. (2018). Real-time detection of drivers' texting and eating behavior based on vehicle dynamics. *Transportation Research Part F*, 58, 594–604.
- Wang, X. et al. (2022). Real-time detection of driver drowsiness using LSTM. *Sensors*, 22(13), 4904.
- Katz, M. J. (1988). Fractals and the analysis of waveform complexity. *Computers and Biomedical Research*, 21(2), 150–166.
- Shannon, C. E. (1948). A mathematical theory of communication. *Bell System Technical Journal*, 27(3), 379–423.

---

*Generated from the feature extraction pipeline in `src/data_pipeline/features/simlsl.py` and the domain analysis results in this directory.*
