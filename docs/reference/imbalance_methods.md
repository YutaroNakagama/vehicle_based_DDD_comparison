# Imbalanced Data Handling Methods

This document summarizes the imbalanced data handling methods implemented and evaluated in this project.

## Overview

In drowsy driving detection tasks, there is a severe class imbalance where **positive cases (drowsy) account for only about 3.9%**.
To address this problem, we implemented and compared methods in the following three categories.

| Category | Methods | Data Size Change |
|----------|---------|------------------|
| **Oversampling** | SMOTE, SMOTE+Tomek, SMOTE+ENN, SMOTE+RUS | Increase |
| **Undersampling** | RUS, Tomek Links | Decrease |
| **Data Augmentation** | Jittering + Scaling | Increase |
| **Ensemble** | BalancedRF, EasyEnsemble | No change |

> For evaluation metrics details, see [Evaluation Metrics](evaluation_metrics.md).

---

## 1. Oversampling Methods

### 1.1 SMOTE (Synthetic Minority Over-sampling Technique)

A method that generates synthetic samples by linear interpolation between minority class samples.

**Algorithm:**
1. Select a minority class sample $x_i$
2. Select a minority class sample $x_{nn}$ from its k-nearest neighbors
3. Synthetic sample: $x_{new} = x_i + \lambda (x_{nn} - x_i)$, where $\lambda \in [0, 1]$

**Reference:**
> Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002).
> **SMOTE: Synthetic Minority Over-sampling Technique.**
> *Journal of Artificial Intelligence Research*, 16, 321-357.
> https://doi.org/10.1613/jair.953

**Implementation:** `imblearn.over_sampling.SMOTE`

---

### 1.5 Subject-wise SMOTE (SW-SMOTE)

A variant of SMOTE that applies oversampling **per subject** rather than globally across the entire dataset.

**Algorithm:**
1. For each subject's data independently:
   - Calculate the minority/majority class ratio
   - Apply SMOTE to the subject's data to reach the target ratio
2. Concatenate all (original + augmented) per-subject data

**Motivation:**
In multi-subject physiological datasets, feature distributions differ across subjects.
Global SMOTE generates synthetic samples that may blend characteristics of different subjects,
potentially losing subject-specific patterns. SW-SMOTE preserves per-subject data distributions
while addressing within-subject class imbalance.

**Settings in this project:**
- Applied via `--subject_wise_oversampling` flag combined with `--oversample_method smote`
- `target_ratio`: 0.1 or 0.5 (minority-to-majority ratio after oversampling)

**Implementation:** `src/data_pipeline/augmentation.py` with `--subject_wise_oversampling` flag

> **Naming convention:** In launcher scripts, this method is specified as `CONDITION=smote`.
> The resulting evaluation files use the tag prefix `imbalv3_*` (legacy naming).
> This is distinct from `CONDITION=smote_plain` which applies standard global SMOTE.

---

### 1.2 SMOTE + Tomek Links

A hybrid method that removes boundary noise with Tomek Links after oversampling with SMOTE.

**Definition of Tomek Links:**
A sample pair $(x_i, x_j)$ is a Tomek Link if $x_i$ and $x_j$ belong to different classes,
and there exists no $x_k$ such that $d(x_i, x_j) < d(x_i, x_k)$ and $d(x_i, x_j) < d(x_j, x_k)$.

**Reference:**
> Batista, G. E., Prati, R. C., & Monard, M. C. (2004).
> **A Study of the Behavior of Several Methods for Balancing Machine Learning Training Data.**
> *ACM SIGKDD Explorations Newsletter*, 6(1), 20-29.
> https://doi.org/10.1145/1007730.1007735

**Implementation:** `imblearn.combine.SMOTETomek`

---

### 1.3 SMOTE + ENN (Edited Nearest Neighbours)

More aggressive noise sample removal with ENN after oversampling with SMOTE.

**ENN Algorithm:**
For each sample, determine the class by majority vote of k-nearest neighbors, and remove if different from the actual label.

**Reference:**
> Wilson, D. L. (1972).
> **Asymptotic Properties of Nearest Neighbor Rules Using Edited Data.**
> *IEEE Transactions on Systems, Man, and Cybernetics*, SMC-2(3), 408-421.
> https://doi.org/10.1109/TSMC.1972.4309137

> Batista, G. E., Prati, R. C., & Monard, M. C. (2004).
> **A Study of the Behavior of Several Methods for Balancing Machine Learning Training Data.**
> *ACM SIGKDD Explorations Newsletter*, 6(1), 20-29.

**Implementation:** `imblearn.combine.SMOTEENN`

---

### 1.4 SMOTE + RUS (Random Under-Sampling)

A hybrid method that increases the minority class with SMOTE and then reduces the majority class with RUS.

**Settings in this implementation:**
- SMOTE: Increase minority class to 50% of majority class
- RUS: Final ratio minority:majority = 0.8:1

**Reference:**
> Chawla, N. V., et al. (2002). SMOTE. *JAIR*, 16, 321-357.

**Implementation:** `imblearn.pipeline.Pipeline` with SMOTE + RandomUnderSampler

---

## 2. Undersampling Methods

### 2.1 Random Under-Sampling (RUS)

Randomly removes samples from the majority class to adjust class balance.

**Advantages:**
- Simple and fast
- Uses only real data (no synthetic data)

**Disadvantages:**
- Potential information loss
- Data size may decrease dramatically

**Reference:**
> He, H., & Garcia, E. A. (2009).
> **Learning from Imbalanced Data.**
> *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263-1284.
> https://doi.org/10.1109/TKDE.2008.239

**Implementation:** `imblearn.under_sampling.RandomUnderSampler`

---

### 2.2 Tomek Links (Standalone)

Removes only majority class samples that are noise near the decision boundary.

**Characteristics:**
- Does not perform aggressive downsampling (removes only boundary samples)
- Effect of cleaning the decision boundary

**Reference:**
> Tomek, I. (1976).
> **Two Modifications of CNN.**
> *IEEE Transactions on Systems, Man, and Cybernetics*, SMC-6(11), 769-772.
> https://doi.org/10.1109/TSMC.1976.4309452

**Implementation:** `imblearn.under_sampling.TomekLinks`

---

## 3. Time-Series Data Augmentation Methods

### 3.1 Jittering + Scaling

Data augmentation method considering time-series data characteristics. Instead of interpolation in feature space like SMOTE,
simulates realistic noise and variations.

**Jittering:**
Adds Gaussian noise to features.
$$x_{aug} = x + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$

**Scaling:**
Simulates amplitude variations.
$$x_{aug} = x \cdot s, \quad s \sim \mathcal{N}(1, \sigma_s^2)$$

**Settings in this implementation:**
- `jitter_sigma = 0.03` (ratio to feature standard deviation)
- `scale_sigma = 0.1` (standard deviation of scaling coefficient)

**Reference:**
> Um, T. T., Pfister, F. M., Pichler, D., Endo, S., Lang, M., Hirche, S., ... & Kulić, D. (2017).
> **Data Augmentation of Wearable Sensor Data for Parkinson's Disease Monitoring using Convolutional Neural Networks.**
> *Proceedings of the 19th ACM International Conference on Multimodal Interaction (ICMI)*, 216-220.
> https://doi.org/10.1145/3136755.3136817

**Implementation:** `src/data_pipeline/augmentation.py`

---

## 4. Ensemble Methods

### 4.1 Balanced Random Forest

Adjusts class balance by bootstrap sampling during each decision tree training.

**Algorithm:**
For each tree, select all minority class samples and randomly select the same number of majority class samples for training.

**Reference:**
> Chen, C., Liaw, A., & Breiman, L. (2004).
> **Using Random Forest to Learn Imbalanced Data.**
> *University of California, Berkeley*, Technical Report 666.
> https://statistics.berkeley.edu/sites/default/files/tech-reports/666.pdf

**Implementation:** `imblearn.ensemble.BalancedRandomForestClassifier`

---

### 4.2 EasyEnsemble

Ensembles multiple AdaBoost classifiers. Each classifier is trained on a differently undersampled dataset.

**Algorithm:**
1. Randomly divide the majority class into $T$ subsets
2. Train an AdaBoost classifier with each subset and all minority class samples
3. Aggregate predictions from $T$ classifiers

**Reference:**
> Liu, X. Y., Wu, J., & Zhou, Z. H. (2008).
> **Exploratory Undersampling for Class-Imbalance Learning.**
> *IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics)*, 39(2), 539-550.
> https://doi.org/10.1109/TSMCB.2008.2007853

**Implementation:** `imblearn.ensemble.EasyEnsembleClassifier`

---

## 5. Recommended Reading

### Survey Papers

> He, H., & Garcia, E. A. (2009).
> **Learning from Imbalanced Data.**
> *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263-1284.

> Krawczyk, B. (2016).
> **Learning from Imbalanced Data: Open Challenges and Future Directions.**
> *Progress in Artificial Intelligence*, 5(4), 221-232.
> https://doi.org/10.1007/s13748-016-0094-0

### Time-Series Data Augmentation

> Iwana, B. K., & Uchida, S. (2021).
> **An Empirical Survey of Data Augmentation for Time Series Classification with Neural Networks.**
> *PLOS ONE*, 16(7), e0254841.
> https://doi.org/10.1371/journal.pone.0254841

### Deep Learning with Imbalanced Data

> Johnson, J. M., & Khoshgoftaar, T. M. (2019).
> **Survey on Deep Learning with Class Imbalance.**
> *Journal of Big Data*, 6(1), 1-54.
> https://doi.org/10.1186/s40537-019-0192-5

---

## 6. Related Files

- **Augmentation implementation:** `src/data_pipeline/augmentation.py`
- **Training Pipeline:** `src/models/architectures/common.py`
- **CLI Helper:** `src/utils/cli/train_cli_helpers.py`
- **Visualization:** `src/analysis/imbalance_analysis.py`
- **HPC scripts:** `scripts/hpc/jobs/imbalance/`
