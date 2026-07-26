# exp3 c1 — Statistical Characterisation of the Recorded Evaluation Metrics

**Scope.** A model-characteristic analysis of the exp3 "c1" recorded AUROC metrics: 5 methods
× 6 evaluation modes (Cross excluded). The goal is to characterise the *model-dependent
structure* of the recorded metrics — how the methods differ, and how each method responds to
imbalance handling and to domain restriction — using standard inferential statistics. This is a
descriptive/methodological summary of the recorded metrics under the within-domain / pooled
evaluation protocol; it is a cross-method comparison, not a claim about absolute out-of-sample
performance.

- **Methods:** RF (feature-selection on, top-10), RF-nofs (feature-selection off, all 165
  vehicle-dynamics features), SvmW (Zhao steering-wavelet SVM), SvmA (Arefnezhad ANFIS/PSO SVM),
  Lstm (Wang Bi-LSTM; target = DRT `event_label`, a different construct from the KSS label
  used by the other four).
- **Modes:** Pooled-base (no imbalance handling), Pooled-SW-SMOTE, Within-in, Within-out,
  Mixed-in, Mixed-out.
- **Metric:** AUROC per seed (with AUPRC, confusion-matrix quantities, and predicted-probability
  spread collected for the degeneracy analysis).
- **Reproducibility:** `scripts/python/analysis/exp3_c1_recorded_value_analysis.py`. Tools:
  scipy / numpy / pandas / matplotlib (Dunn, Cliff's δ, Scheirer–Ray–Hare implemented in-script).

> Note: Pooled-SW-SMOTE cells for SvmW/SvmA/RF-nofs have small provisional n (a seed
> augmentation is in progress) and are marked with an asterisk.

---

## A. Descriptive statistics and interval estimates (AUROC)

Mean ± SD (n); 95% t-CI and percentile bootstrap CI are computed in the script.

| Method | Pooled-base | Pooled-SW-SMOTE | Within-in | Within-out | Mixed-in | Mixed-out |
|---|---|---|---|---|---|---|
| **RF (fs)** | 0.738 ± 0.090 (15) | 0.795 ± 0.052 (15) | 0.746 ± 0.089 (24) | 0.778 ± 0.108 (24) | 0.719 ± 0.085 (24) | 0.749 ± 0.104 (24) |
| **RF (nofs)** | — | 0.855 ± 0.032 (2)* | 0.874 ± 0.087 (5) | 0.916 ± 0.065 (5) | 0.829 ± 0.112 (5) | 0.891 ± 0.125 (5) |
| **SvmW** | 0.519 ± 0.011 (6) | 0.684 ± 0.020 (3)* | 0.800 ± 0.012 (8) | 0.759 ± 0.013 (8) | 0.742 ± 0.012 (8) | 0.771 ± 0.016 (8) |
| **SvmA** | 0.481 ± 0.008 (6) | 0.569 ± 0.031 (2)* | 0.576 ± 0.029 (8) | 0.574 ± 0.074 (8) | 0.532 ± 0.024 (8) | 0.597 ± 0.025 (8) |
| **Lstm** | 0.512 ± 0.011 (6) | 0.513 ± 0.006 (6) | 0.779 ± 0.007 (15) | 0.763 ± 0.012 (15) | 0.782 ± 0.009 (15) | 0.779 ± 0.009 (15) |

RF-nofs has no Pooled-base arm by design.

![Recorded AUROC by method and mode](figures/c1_recorded/fig1_auroc_method_mode.png)

*Figure 1. Recorded AUROC (mean ± SD, clipped to [0,1]) for each method across the five
SW-SMOTE evaluation modes (Pooled-base omitted), with the 0.5 reference line. The error bars use
the same SD as the "±" column of the table above; they are clipped at 1.0 because a symmetric
bar on a bounded metric can otherwise overshoot for small-n, near-ceiling cells (e.g. RF-nofs
Mixed-out, n=5, mean 0.891, SD 0.125). RF-nofs is the highest but with the widest dispersion;
SvmW and Lstm are tightly grouped; SvmA is the lowest throughout.*

---

## B. Between-method differences within each mode (Kruskal–Wallis)

Kruskal–Wallis across the methods present in each mode, with η²_H, Dunn/Holm post-hoc and
Cliff's δ (details in the script). **The method effect is highly significant in every mode.**

| Mode | H | p | η²_H | Rank (mean AUROC) |
|---|---|---|---|---|
| Pooled-base | 27.4 | 4.8e-06 | 0.84 | RF-fs (0.738) ≫ SvmW ≈ Lstm ≈ SvmA (0.48–0.52) |
| Pooled-SW-SMOTE* | 21.7 | 2.3e-04 | 0.77 | RF-nofs > RF-fs > SvmW > SvmA > Lstm |
| Within-in | 28.2 | 1.1e-05 | 0.44 | RF-nofs > SvmW > Lstm > RF-fs > SvmA |
| Within-out | 26.9 | 2.1e-05 | 0.42 | RF-nofs > RF-fs > Lstm ≈ SvmW > SvmA |
| Mixed-in | 33.2 | 1.1e-06 | 0.53 | RF-nofs > Lstm > SvmW > RF-fs > SvmA |
| Mixed-out | 20.9 | 3.2e-04 | 0.31 | RF-nofs > Lstm > SvmW > RF-fs > SvmA |

- **Under Pooled-base only RF-fs separates from the 0.5 level** (RF-fs vs each of SvmW / SvmA /
  Lstm: p_holm ≤ 0.03, **Cliff's δ = 1.0**); the other three are statistically indistinguishable.
- **SvmA is the consistent lowest rank** under every domain-restricted mode (p_holm < 0.02,
  |δ| ≥ 0.75–1.0).
- **RF-nofs is the consistent top rank** across Within/Mixed (mechanism in D1).

---

## C. Within-method mode contrasts (seed-paired Wilcoxon signed-rank)

Paired across shared seeds; median Δ and two-sided p (Pooled contrasts for
SvmW/SvmA/RF-nofs under-powered pending the seed augmentation).

| Method | Imbalance (base→SW-SMOTE) | Domain restriction (Pooled→Within) | Domain shift (in→out) |
|---|---|---|---|
| **RF-fs** | Δ=+0.060, **p=0.035** | Δ=+0.005, p=0.60 (n.s.) | Within Δ=+0.024 **p=0.005** |
| **RF-nofs** | (n.a.) | (n.a.) | Within Δ=+0.040 p=0.063 |
| **SvmW** | Δ=+0.176 (n=3, p=0.25) | Δ=+0.111 (n=3, p=0.25) | Within Δ=−0.042 **p=0.008** |
| **SvmA** | (n.a.) | (n.a.) | Within Δ=+0.023 p=0.95 |
| **Lstm** | Δ=+0.005, **p=1.0 (inactive)** | Δ=+0.267, **p=0.031** | Within Δ=−0.025 **p=0.003** |

- **Lstm is imbalance-inactive** (base→SW-SMOTE Δ≈0, p=1.0) but shows the **largest
  domain-restriction change** (Pooled→Within Δ=+0.27): its recorded AUROC is governed by the
  evaluation regime, not by rebalancing — consistent with its near-balanced DRT target.
- **RF-fs has a small but significant imbalance change** and significant, consistent
  domain-shift sensitivity.

---

## D. Model-characteristic quantities

### D1. RF feature-count effect (fs top-10 vs nofs all-165) — Mann–Whitney + Cliff's δ

| Mode | RF-fs | RF-nofs | Δ | p | Cliff's δ |
|---|---|---|---|---|---|
| Within-in | 0.746 | 0.874 | +0.128 | **0.019** | +0.67 (large) |
| Within-out | 0.778 | 0.916 | +0.139 | **0.005** | +0.77 (large) |
| Mixed-in | 0.719 | 0.829 | +0.110 | **0.032** | +0.62 (large) |
| Mixed-out | 0.749 | 0.891 | +0.142 | **0.027** | +0.63 (large) |
| Pooled-SW-SMOTE* | 0.795 | 0.855 | +0.060 | 0.13 | +0.73 (large) |

**Using all 165 features instead of the top-10 raises RF's recorded AUROC by ≈0.11–0.14 (large
effect) across every domain-restricted mode** — but a controlled dose-response (D1b) shows this
dependence **saturates by k≈20**, so it is not an open-ended benefit of ever-more features.

![RF feature-count effect](figures/c1_recorded/fig2_rf_feature_count.png)

*Figure 2. RF feature-count effect: all 165 features (RF-nofs) vs the top-10 (RF-fs) per mode.*

### D1b. Feature-count dose-response — where does the gain saturate?

D1 uses only the two endpoints actually run (k=10, k=165). To locate the *shape* of the
dependence, a controlled within-in-domain probe (plain RF, RF-importance top-k, recorded
evaluation, 3 seeds) sweeps intermediate k, comparing two selection orders: choosing the top-k
*after* SW-SMOTE (the c1 pipeline order) vs on the natural training data.

| k | SW-SMOTE→select (c1 order) | select on natural data |
|---:|:---:|:---:|
| 5 | 0.725 | 0.858 |
| 10 | 0.863 | 0.892 |
| **20** | **0.887** | 0.891 |
| 40 | 0.896 | 0.895 |
| 80 | 0.899 | 0.885 |
| 120 | 0.901 | 0.886 |
| 165 | 0.895 | 0.882 |

**The dependence is not monotone to 165 — it saturates by k≈20.** Almost all of the gain falls
between k=5 and k≈20 (+0.16 for the c1-order curve); from k=20 to k=165 the recorded AUROC is flat
within noise (0.887→0.895). The D1 endpoint gap (+0.13, k=10→165) therefore overstates a
"feature-count" law: the same plateau (≈0.89) is already reached at k≈20, and the low RF-fs anchor
(0.746) reflects that the c1 selection stage — which ranks features on SW-SMOTE-oversampled data,
*before* the model is fit — picks a suboptimal top-10. Selecting on the natural training data
instead reaches the plateau by k=5–10.

Practical reading: **≈20 vehicle-dynamics features suffice to reach RF's recorded ceiling; the
top-10 arm underperforms mainly because of the SMOTE-before-selection order, not because 10
features are intrinsically too few.**

![RF feature-count dose-response](figures/c1_recorded/fig5_feature_dose_response.png)

*Figure 5. Recorded within-in-domain AUROC vs feature count k. Both selection orders plateau by
k≈20; the recorded c1 anchors (RF-fs k=10 = 0.746 ★, RF-nofs k=165 = 0.874 ◆) bracket the curve.
Probe = plain RF with fixed hyperparameters and a simplified evaluation split, so absolute levels
are not directly comparable to the tuned c1 pipeline; the saturation shape is what transfers.*

### D2. Decision spread / degeneracy (specificity, predicted-positive rate)

| Method · mode | proba SD | specificity | pred-positive rate |
|---|---|---|---|
| SvmW · Pooled-base | 0.001 | 0.004 | 0.997 |
| SvmW · Pooled-SW-SMOTE | 0.199 | 0.504 | 0.506 |
| Lstm · Pooled-base | — | 0.003 | 0.999 |
| Lstm · Pooled-SW-SMOTE | — | 0.005 | 0.997 |
| RF-fs · Pooled-base | 0.013 | 0.569 | 0.444 |

Under Pooled-base **SvmW is an all-positive constant classifier** (probability spread ≈ 0,
specificity ≈ 0). **SW-SMOTE restores a usable probability ranking for SvmW** (spread 0.20,
specificity 0.50) — a de-degeneration. Lstm's Pooled collapse is a majority-class artefact of a
near-balanced target and is unaffected by SMOTE; RF is never degenerate.

![Decision spread (specificity) under Pooled](figures/c1_recorded/fig3_specificity.png)

*Figure 3. Specificity under Pooled (≈0 = all-positive). SW-SMOTE lifts SvmW's specificity from
≈0.004 to ≈0.50; Lstm stays collapsed; RF is non-degenerate.*

### D3. Across-seed stability (Brown–Forsythe equal-variance test per mode)

| Mode | RF-fs | RF-nofs | SvmW | SvmA | Lstm | Brown–Forsythe |
|---|---|---|---|---|---|---|
| Within-in | 0.089 | 0.087 | 0.012 | 0.029 | 0.007 | W=4.80, **p=0.002** |
| Within-out | 0.108 | 0.065 | 0.013 | 0.074 | 0.012 | W=6.30, **p=3e-04** |
| Mixed-in | 0.085 | 0.112 | 0.012 | 0.024 | 0.009 | W=5.10, **p=0.001** |
| Mixed-out | 0.104 | 0.125 | 0.016 | 0.025 | 0.009 | W=4.01, **p=0.006** |
| Pooled-base | 0.090 | — | 0.011 | 0.008 | 0.011 | W=5.77, **p=0.003** |

**RF (both variants) is markedly the least seed-stable method** — its across-seed SD (0.09–0.13)
is an order of magnitude larger than SvmW's / Lstm's (≈0.01), with significant variance
heterogeneity in every mode. This reflects the seed sensitivity of the RF ensemble + Optuna
pipeline relative to the near-deterministic SVM and LSTM pipelines.

![Across-seed variability by method](figures/c1_recorded/fig4_seed_variability.png)

*Figure 4. Mean across-seed SD of AUROC over the Within/Mixed modes. RF ≫ SvmW / SvmA / Lstm.*

### D4. SvmA under-performance: a learner effect, not a feature-set effect

SvmA records the lowest AUROC of all methods (A: 0.53–0.60 within/mixed; 0.481 Pooled-base). On a
recorded-value basis the cause is the *learner*, not the steering feature set and not class
imbalance.

**Same features, different learner (recorded within-in-domain).** SvmA's own 36 steering features
fed to RF reach 0.884 — essentially identical to the full 165-feature vehicle set (0.873) — whereas
an RBF-SVM (SvmA's learner) on the same 36 features gives only 0.597, matching SvmA's recorded
within-in AUROC (0.576, A). The steering feature set is therefore not the bottleneck under the
recorded protocol: it yields as much recorded separability as the full set when a tree ensemble
reads it; the RBF-SVM cannot.

| Feature set → learner | Recorded AUROC |
|---|---|
| SvmA 36 steering → RF | 0.884 |
| SvmA 36 steering → RBF-SVM | 0.597 |
| all 165 vehicle → RF | 0.873 |

**Imbalance treatment does not lift the SVM.** Under the recorded split, neither SW-SMOTE nor
class weighting moves the RBF-SVM off ~0.58 (0.597 → 0.582 → 0.572), while RF stays ~0.88 across the
same treatments — so class imbalance is not the cause of SvmA's low score either. (Domain shift is
likewise not the driver: within-domain SvmA already sits at 0.53–0.60, A.)

| Imbalance treatment (36 steering feats) | RF | RBF-SVM |
|---|---|---|
| raw | 0.878 | 0.597 |
| SW-SMOTE | 0.883 | 0.582 |
| class-weight | 0.877 | 0.572 |

**SvmA's bottom rank is thus a learner limitation** (RBF-SVM on steering features), not a deficient
feature set and not an imbalance artefact — unlike SvmW, whose Pooled degeneracy *is* a
decision-function artefact that SW-SMOTE repairs (D2).

![SvmA learner vs feature effect](figures/c1_recorded/fig6_svma_learner_vs_feature.png)

*Figure 6. Recorded within-in-domain AUROC. (A) The same 36 steering features reach the full-set
ceiling under RF (0.884 vs 0.873) but only 0.597 under the RBF-SVM that SvmA uses (≈ its recorded
c1 value 0.576). (B) Neither SW-SMOTE nor class weighting lifts the RBF-SVM off ~0.58, while RF
stays ~0.88. Probe = plain learners on a recorded-style split; the learner contrast is the point.*

---

## E. Two-way structure (method × mode), Scheirer–Ray–Hare

Non-parametric two-way test on a balanced subset (5 methods × 4 Within/Mixed × in/out modes × 5
common seeds = 100 observations):

| Effect | H | df | p |
|---|---|---|---|
| **Method** | 62.1 | 4 | **1.0e-12** |
| Mode | 3.2 | 3 | 0.37 (n.s.) |
| Method × Mode | 10.1 | 12 | 0.61 (n.s.) |

**Within the Within/Mixed regime the classifier is the overwhelmingly dominant factor; the four
domain sub-modes do not differ significantly and there is no significant method × mode
interaction.** The method ranking (RF-nofs > SvmW ≈ Lstm > RF-fs > SvmA) is essentially stable
across Within/Mixed and in/out.

---

## Synthesis — model-dependent structure of the recorded metrics

- **RF (fs):** the only method that separates from 0.5 under Pooled-base (B), with a small
  significant imbalance change (C); but the **least seed-stable** method (D3), and, with all
  features (RF-nofs), a **feature-count dependence** that saturates by k≈20 (D1, D1b).
- **RF-nofs:** the highest recorded AUROC throughout; the feature-count gain saturates by k≈20
  (D1b).
- **SvmW:** all-positive **degenerate** without rebalancing; SW-SMOTE **de-degenerates** it (D2),
  restoring a usable ranking.
- **SvmA:** bottom rank is a **learner limitation** (RBF-SVM), not a feature-set or imbalance
  effect — the same steering features under RF reach the full-set ceiling (D4).
- **Lstm:** **imbalance-inactive** (C, D2) but strongly **regime-driven** (largest Pooled→Within
  change); its within/mixed AUROC reflects the near-balanced DRT target, a different construct
  from the KSS label used by the other methods, so its absolute level is not directly comparable.
- **Across methods (E):** the classifier dominates; mode and interaction are non-significant.
