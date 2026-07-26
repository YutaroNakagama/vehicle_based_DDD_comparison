# Statistical Analysis of the exp3 c1 Recorded Results

**Scope.** A model-characteristic-driven statistical analysis of the exp3 "c1" recorded
evaluation metrics, deliberately **not** reproducing the exp2/TIV2026 Sobol
variance-decomposition. The design is 5 methods × 6 experiment modes (Cross excluded), and
the goal is to characterise, at an academic level, what can be inferred about **each
model's behaviour** across **each experiment mode** from the recorded per-seed metrics.

- **Methods:** RF (feature-selection on, top-10), RF-nofs (feature-selection off, all 165
  non-EEG features), SvmW (Zhao steering-wavelet SVM), SvmA (Arefnezhad ANFIS/PSO SVM),
  Lstm (Wang Bi-LSTM; note: predicts the DRT-distraction `event_label`, **not** EEG-drowsiness).
- **Modes:** Pooled-base (no imbalance handling), Pooled-SW-SMOTE, Within-in, Within-out,
  Mixed-in, Mixed-out. Cross excluded by request.
- **Metric:** primary AUROC per seed (AUPRC, confusion-matrix quantities, and predicted-
  probability arrays collected for the degeneracy analysis).
- **Reproducibility:** `scripts/python/analysis/exp3_c1_statistical_analysis.py` →
  `results/analysis/exp3_verification/c1_statistical_analysis.json`. Tools: scipy/numpy/pandas
  (Dunn, Cliff's δ, Scheirer–Ray–Hare implemented in-script; statsmodels unavailable).

> **Validity note (read first).** These are **recorded** AUROCs computed under the paper's
> within-domain / pooled protocols, which share a documented train/eval **row-overlap leak**
> (~60 % pooled, ~69 % within). Every inferential result below therefore describes the
> **leaked** metric surface. Section F contrasts it with the leakage-free (honest) estimates,
> which are the values that carry scientific weight. The statistics below are valid as a
> characterisation of *how each model behaves under this protocol*, not as evidence that any
> vehicle method detects drowsiness.
>
> Pooled-SW-SMOTE is still being regenerated (subject-wise SMOTE); its cells have provisional
> n (SvmW 3, SvmA 2, RF-nofs 2) and are marked accordingly.

---

## A. Descriptive statistics and interval estimates (AUROC)

Mean ± SD (n), with 95 % t-CI and percentile bootstrap CI (B=2000). Full CIs in the JSON.

| Method | Pooled-base | Pooled-SW-SMOTE | Within-in | Within-out | Mixed-in | Mixed-out |
|---|---|---|---|---|---|---|
| **RF (fs)** | 0.738 ± 0.090 (15) | 0.795 ± 0.052 (15) | 0.746 ± 0.089 (24) | 0.778 ± 0.108 (24) | 0.719 ± 0.085 (24) | 0.749 ± 0.104 (24) |
| **RF (nofs)** | — | 0.855 ± 0.032 (2)* | 0.874 ± 0.087 (5) | 0.916 ± 0.065 (5) | 0.829 ± 0.112 (5) | 0.891 ± 0.125 (5) |
| **SvmW** | 0.519 ± 0.011 (6) | 0.684 ± 0.020 (3)* | 0.800 ± 0.012 (8) | 0.759 ± 0.013 (8) | 0.742 ± 0.012 (8) | 0.771 ± 0.016 (8) |
| **SvmA** | 0.481 ± 0.008 (6) | 0.569 ± 0.031 (2)* | 0.576 ± 0.029 (8) | 0.574 ± 0.074 (8) | 0.532 ± 0.024 (8) | 0.597 ± 0.025 (8) |
| **Lstm** | 0.512 ± 0.011 (6) | 0.513 ± 0.006 (6) | 0.779 ± 0.007 (15) | 0.763 ± 0.012 (15) | 0.782 ± 0.009 (15) | 0.779 ± 0.009 (15) |

\* provisional n (Pooled-SW-SMOTE regeneration in progress). RF-nofs has no Pooled-base arm by design.

---

## B. Between-method differences within each mode (Kruskal–Wallis)

Kruskal–Wallis across the methods present in each mode, ε²/η²_H effect size, Dunn post-hoc
with Holm correction, and Cliff's δ for the largest contrasts. **Every mode shows a highly
significant method effect.**

| Mode | H | p | η²_H | Rank (mean AUROC) |
|---|---|---|---|---|
| Pooled-base | 27.4 | 4.8e-06 | 0.84 | RF-fs (0.738) ≫ SvmW ≈ Lstm ≈ SvmA (0.48–0.52) |
| Pooled-SW-SMOTE\* | 21.7 | 2.3e-04 | 0.77 | RF-nofs > RF-fs > SvmW > SvmA > Lstm |
| Within-in | 28.2 | 1.1e-05 | 0.44 | RF-nofs > SvmW > Lstm > RF-fs > SvmA |
| Within-out | 26.9 | 2.1e-05 | 0.42 | RF-nofs > RF-fs > Lstm ≈ SvmW > SvmA |
| Mixed-in | 33.2 | 1.1e-06 | 0.53 | RF-nofs > Lstm > SvmW > RF-fs > SvmA |
| Mixed-out | 20.9 | 3.2e-04 | 0.31 | RF-nofs > Lstm > SvmW > RF-fs > SvmA |

Key post-hoc results (Holm-adjusted, Cliff's δ):
- **In Pooled-base, only RF-fs separates from chance** — RF-fs vs each of SvmW / SvmA / Lstm:
  p_holm ≤ 0.03, **δ = 1.0 (large)** in every case; the other three are statistically
  indistinguishable from one another and from chance.
- **SvmA is the consistent low outlier** in every domain-restricted mode (RF-nofs/SvmW/Lstm
  vs SvmA: p_holm < 0.02, |δ| ≥ 0.75–1.0).
- **RF-nofs is the consistent top rank** across Within/Mixed (see D1 for the mechanism).

---

## C. Within-method mode contrasts (seed-paired Wilcoxon signed-rank)

Paired across shared seeds; median Δ and two-sided p. (Pooled contrasts for SvmW/SvmA/RF-nofs
are under-powered pending regeneration — reported as available.)

| Method | Imbalance (base→SW-SMOTE) | Domain restriction (Pooled→Within) | Domain shift (in→out) |
|---|---|---|---|
| **RF-fs** | Δ=+0.060, **p=0.035** | Δ=+0.005, p=0.60 (n.s.) | Within Δ=+0.024 **p=0.005**; Mixed Δ=+0.032 **p<1e-4** |
| **RF-nofs** | (n.a.) | (n.a.) | Within Δ=+0.040 p=0.063 |
| **SvmW** | Δ=+0.176 (n=3, p=0.25) | Δ=+0.111 (n=3, p=0.25) | Within Δ=−0.042 **p=0.008** |
| **SvmA** | (n.a.) | (n.a.) | Mixed Δ=+0.065 **p=0.008** |
| **Lstm** | Δ=+0.005, **p=1.0 (inactive)** | Δ=+0.267, **p=0.031** | Within Δ=−0.025 **p=0.003** |

Reading:
- **Lstm is imbalance-inactive** (base→SW-SMOTE Δ≈0, p=1.0) yet shows the **largest
  domain-restriction jump** (Pooled→Within Δ=+0.27): its behaviour is driven entirely by the
  evaluation regime, not by rebalancing — consistent with its near-balanced DRT target.
- **RF-fs has a small but significant imbalance gain** and significant, directionally-consistent
  domain-shift sensitivity.
- **SvmW's apparent imbalance/domain-restriction gains are large in point estimate but
  under-powered** (n=3 paired) pending regeneration; its domain-shift contrasts are significant.

---

## D. Model-characteristic quantities

### D1. RF feature-count effect (fs top-10 vs nofs all-165) — Mann–Whitney + Cliff's δ

| Mode | RF-fs | RF-nofs | Δ | p | Cliff's δ |
|---|---|---|---|---|---|
| Within-in | 0.746 | 0.874 | +0.128 | **0.019** | +0.67 (large) |
| Within-out | 0.778 | 0.916 | +0.139 | **0.005** | +0.77 (large) |
| Mixed-in | 0.719 | 0.829 | +0.110 | **0.032** | +0.62 (large) |
| Mixed-out | 0.749 | 0.891 | +0.142 | **0.027** | +0.63 (large) |
| Pooled-SW-SMOTE\* | 0.795 | 0.855 | +0.060 | 0.13 | +0.73 (large) |

**Using all 165 features instead of the top-10 raises RF's recorded AUROC by ≈0.11–0.14
(large effect) across every domain-restricted mode.** Because more capacity cannot add
*generalising* signal but can add *memorising* capacity, this monotone "more features → higher
recorded AUROC" gradient is the signature of leak-row memorisation, not of better drowsiness
discrimination. (This is exactly the "feature bias toward RF" that IV2025 flagged as a threat.)

### D2. Decision degeneracy (predicted-probability spread / specificity / predicted-positive rate)

| Method · mode | proba SD | specificity | pred-positive rate | interpretation |
|---|---|---|---|---|
| SvmW · Pooled-base | 0.001 | 0.004 | 0.997 | **all-positive collapse** (degenerate) |
| SvmW · Pooled-SW-SMOTE | 0.199 | 0.504 | 0.506 | **spread restored → non-degenerate** |
| Lstm · Pooled-base | — | 0.003 | 0.999 | majority (all-positive) collapse |
| Lstm · Pooled-SW-SMOTE | — | 0.005 | 0.997 | still collapsed (SMOTE inactive for Lstm) |
| RF-fs · Pooled-base | 0.013 | 0.569 | 0.444 | non-degenerate throughout |

**SW-SMOTE's effect on SvmW is de-degeneration, not signal injection:** it converts an
all-positive constant classifier (proba SD ≈ 0, specificity ≈ 0) into one with a usable
probability ranking (proba SD 0.20, specificity 0.50). The AUROC "recovery" is the mechanical
consequence of restoring rank information on a leaked test set — Lstm, whose collapse is a
majority-class artefact of a near-balanced target, is unmoved by SMOTE (D2 rows + C).

### D3. Seed-variance / stability (Brown–Forsythe equal-variance test per mode)

Per-method AUROC SD and the Brown–Forsythe (median-centred Levene) test that method variances
are equal within each mode:

| Mode | RF-fs | RF-nofs | SvmW | SvmA | Lstm | Brown–Forsythe |
|---|---|---|---|---|---|---|
| Within-in | 0.089 | 0.087 | 0.012 | 0.029 | 0.007 | W=4.80, **p=0.002** |
| Within-out | 0.108 | 0.065 | 0.013 | 0.074 | 0.012 | W=6.30, **p=3e-04** |
| Mixed-in | 0.085 | 0.112 | 0.012 | 0.024 | 0.009 | W=5.10, **p=0.001** |
| Mixed-out | 0.104 | 0.125 | 0.016 | 0.025 | 0.009 | W=4.01, **p=0.006** |
| Pooled-base | 0.090 | — | 0.011 | 0.008 | 0.011 | W=5.77, **p=0.003** |

**RF (both variants) is markedly the least seed-stable method** — its across-seed SD (0.09–0.13)
is an order of magnitude larger than SvmW's and Lstm's (≈0.01), and the variance heterogeneity
is significant in every mode. High seed-variance is itself a model characteristic: RF's
tree-ensemble + Optuna pipeline is sensitive to the seed-dependent split/subsample, whereas
the SVM and LSTM pipelines are near-deterministic given the data.

### D4. SvmA feature-signal probe (established)

SvmA's 18 steering statistics carry no drowsiness signal even under a stronger learner:
univariate max AUROC 0.515, multivariate RBF 0.509, and the same features fed to RF give 0.496.
This is why SvmA sits at the bottom of every ranking (B) and never de-degenerates into signal
(unlike SvmW): rebalancing cannot recover a signal that the features do not contain.

---

## E. Two-way structure (method × mode), Scheirer–Ray–Hare

Non-parametric two-way test on a **balanced** subset — 5 methods × 4 (Within/Mixed × in/out)
modes × 5 common seeds = 100 observations (the modes and seeds shared by all methods; Pooled
excluded because RF-nofs has no Pooled-base and Pooled-SW-SMOTE is incomplete):

| Effect | H | df | p |
|---|---|---|---|
| **Method** | 62.1 | 4 | **1.0e-12** |
| Mode | 3.2 | 3 | 0.37 (n.s.) |
| Method × Mode | 10.1 | 12 | 0.61 (n.s.) |

**Within the leaked within/mixed regime, method identity is the overwhelmingly dominant
factor; the four domain modes do not differ significantly and there is no significant
method×mode interaction.** In plain terms: the method ranking (RF-nofs > SvmW ≈ Lstm > RF-fs
> SvmA) is essentially the same whether one evaluates Within or Mixed, in-domain or
out-domain. This is the model-characteristic analogue of, and is consistent with, the exp2
Sobol finding that *distance/grouping* is negligible — but here it is obtained directly from
the c1 results without the Sobol machinery, and it identifies **the classifier, not the
protocol sub-mode, as the driver**.

---

## F. Validity contrast: recorded (leaked) vs honest (leakage-free)

The single most important result. Recorded Within-in means vs the leakage-free (subject-disjoint,
0 % row overlap) re-evaluation point estimates:

| Method | Recorded (Within-in) | Honest (leakage-free) |
|---|---|---|
| RF-fs | 0.746 | **0.517** |
| RF-nofs | 0.874 | **0.534** |
| SvmW | 0.800 | **0.520** |
| SvmA | 0.576 | **0.500** |
| Lstm (KSS target) | 0.779 | **0.510** |
| EEG band-power **positive control** | — | **0.61** |
| Lstm on its native **DRT** target | — | **0.74** (cross-subject) |

**Every vehicle→EEG-drowsiness method collapses to chance once the row overlap is removed**,
while the EEG positive control retains signal (0.61) on the same honest harness — so the
collapse is a real null, not a broken evaluator. The only genuine signal is Lstm on the
distraction (DRT) target, i.e. a different question from drowsiness detection.

---

## Synthesis — what the statistics say about each model

- **RF (fs):** the only method non-degenerate under Pooled (B: separates from chance,
  δ=1.0), with a small significant imbalance gain (C); but the **least seed-stable** method
  (D3) and, with all features (RF-nofs), the clearest **memorisation gradient** (D1). Its
  apparent superiority is capacity to exploit the leak, not drowsiness signal (F).
- **RF-nofs:** highest recorded AUROC everywhere, driven monotonically by feature count
  (D1) — the memorisation signature; honest value is chance (F).
- **SvmW:** all-positive **degenerate** without rebalancing; SW-SMOTE **de-degenerates** it
  (D2) so it can rank the leaked test rows, producing the recorded "recovery" — which does not
  survive honest evaluation (F).
- **SvmA:** **no feature signal** (D4), hence the consistent bottom rank (B) and inability to
  recover under rebalancing; recorded values only marginally above chance, honest exactly chance.
- **Lstm:** **imbalance-inactive** (C, D2) but strongly **regime-driven** (largest
  Pooled→Within jump); its within/mixed AUROC reflects the near-balanced **DRT** target, a
  different construct from EEG-drowsiness.
- **Across methods (E):** method dominates, mode and interaction are non-significant — the
  ranking is a property of the classifiers, stable across domain sub-modes.

**Bottom line.** The recorded-value statistics are internally strong and coherent, but they
characterise how each model *interacts with a leaked evaluation protocol* (memorisation,
degeneracy, feature-signal, seed-stability), not drowsiness-detection ability. Under
leakage-free evaluation all vehicle methods are at chance; the only real signal is Lstm on a
different (distraction) target.
