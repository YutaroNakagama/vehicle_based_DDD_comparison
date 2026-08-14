# exp3 c1 — Statistical Characterisation of the Recorded Evaluation Metrics

**Scope.** A model-characteristic analysis of the exp3 "c1" recorded AUROC metrics: 5 methods
× 4 evaluation modes — the deployable **Pooled** and **Mixed** regimes (Within and Cross retired,
see [within_retirement_plan.md](within_retirement_plan.md)). The goal is to characterise the
*model-dependent structure* of the recorded metrics — how the methods differ, and how each method
responds to imbalance handling and to domain restriction — using standard inferential statistics.
This is a descriptive/methodological summary of the recorded metrics under the pooled / mixed
evaluation protocol; it is a cross-method comparison, not a claim about absolute out-of-sample
performance.

- **Methods:** RF (feature-selection on, top-10), RF-nofs (feature-selection off, all 165
  vehicle-dynamics features), SvmW (Zhao steering-wavelet SVM), SvmA (Arefnezhad ANFIS/PSO SVM),
  Lstm (Wang Bi-LSTM; target = DRT `event_label`, a different construct from the KSS label
  used by the other four).
- **Modes:** Pooled-base (no imbalance handling), Pooled-SW-SMOTE, Mixed-in, Mixed-out. (The
  Within modes are retired; `mixed` trains on all 87 recordings and evaluates on a domain subgroup,
  a deployable operating point, whereas `within` trains on the target group alone.)
- **Cross (`source_only`) is excluded from this write-up by a recorded decision** (2026-07-11,
  `c1_domain_launcher.py:68` — cross-domain transfer collapses to ~0.51 for every method), **reaffirmed
  2026-08-13** after a full-grid audit. It is not part of this cross-method comparison, but note that it
  is the *leakage-free* arm the manuscript's central claim uses, and its seed coverage is uneven:
  RF-fs n=24, Lstm n=15, SvmA n=8, **SvmW n=2, RF-nofs n=0**. The last two are carried as explicit
  manuscript limitations rather than closed with additional compute.
- **Metric:** AUROC per seed (with AUPRC, confusion-matrix quantities, and predicted-probability
  spread collected for the degeneracy analysis).
- **Reproducibility:** `scripts/python/analysis/exp3_c1_recorded_value_analysis.py`. Tools:
  scipy / numpy / pandas / matplotlib (Dunn, Cliff's δ, Scheirer–Ray–Hare implemented in-script).

> **Status (2026-08-03).** The recorded-value campaign is **complete** — all methods × the retained
> Pooled/Mixed modes are final (RF-nofs reached its full 15-seed extension on 2026-08-02). The Part B
> mechanism probes (Figures 5–7) report **pooled** (the deployment regime), with **mixed-in retained
> as a diagnostic**. Sections A–E below are the finalized numbers.

> **Migration in progress — see [within_retirement_plan.md](within_retirement_plan.md).** The two
> Within modes are being retired because `within` trains only on the target group, discarding the
> other group's data. The deployable regimes retained for reporting are Pooled (primary) and Mixed
> (diagnostic). The Part B mechanism probes (§D1b, §D4, §D5) now report `pooled`; Part A will
> rebuild Sections A–E without the Within columns once the RF-nofs seed extension is complete.

---

## A. Descriptive statistics and interval estimates (AUROC)

Mean ± SD (n); 95% t-CI and percentile bootstrap CI are computed in the script.

| Method | Pooled-base | Pooled-SW-SMOTE | Mixed-in | Mixed-out |
|---|---|---|---|---|
| **RF (fs)** | 0.724 ± 0.085 (24) | 0.795 ± 0.052 (15) | 0.719 ± 0.085 (24) | 0.749 ± 0.104 (24) |
| **RF (nofs)** | **0.667 ± 0.093 (24)†** | 0.875 ± 0.026 (6) | 0.846 ± 0.077 (15) | 0.912 ± 0.081 (15) |
| **SvmW** | 0.519 ± 0.011 (6) | 0.694 ± 0.018 (6) | 0.742 ± 0.012 (8) | 0.771 ± 0.016 (8) |
| **SvmA** | 0.481 ± 0.008 (6) | 0.538 ± 0.042 (6) | 0.530 ± 0.026 (11) | 0.597 ± 0.022 (11) |
| **Lstm** | 0.512 ± 0.011 (6) | 0.513 ± 0.006 (6) | 0.782 ± 0.009 (15) | 0.779 ± 0.009 (15) |

† **Filled 2026-08-15 (closed at n=24).** This cell had **never been run** — the earlier
"RF-nofs has no Pooled-base arm by design" wording described the hole rather than a recorded design
decision, and no operations entry ever scoped the full-feature ablation to the SW-SMOTE arm. It was
launched 2026-08-12 and closed 2026-08-15 at **n=24** — the full pre-registered `SEED_MASTER` set
(`c1_domain_launcher.py:56`), which is also the canonical RF seed count used elsewhere in this grid, so
the seed set involved no discretion at any point. CI half-width **0.039** with a flat running mean.
**RF-fs Pooled-base was extended on the same seeds** (15 → 24) because §D1's contrast and the
interaction test need both arms; that retired its long-standing "borderline" CI (0.0499 → **0.036**).
**With this, all 20 cells in the grid are seed-adequate (§F).** 36 new cells, zero failures.

> **Retraction (2026-08-14).** Interim commits at n=9/n=10 reported this cell as
> 0.647/0.670 with a *significant* "reversal" of the feature-count effect (paired Wilcoxon
> p=0.049 then p=0.042). **Those significance claims are withdrawn, not merely superseded.** Seeds
> were processed in fixed launcher order and the n=10 boundary coincided exactly with the end of the
> first launch wave, so the interim p-values were a stopping-rule artefact. The claim that AUPRC shows
> the extra features "dilute split quality for the rare class" is withdrawn as well (seed-paired
> AUPRC difference p=0.36). What survives is stated in the box below.

The two Within modes are retired (see
[within_retirement_plan.md](within_retirement_plan.md)); Pooled and Mixed are the deployable
regimes. RF-nofs reached its full 15-seed extension (2026-08-02); all Mixed cells are final.

> ### The filled cell changes §D1 from "a feature-count effect" to "a feature-count effect *conditional on rebalancing*"
>
> In the three rebalanced/domain modes RF-nofs beats RF-fs by +0.079 to +0.163 (all large, p ≤ 0.002).
> **Under Pooled-base — the only mode with no imbalance handling — that advantage is not present:**
>
> | arm | AUROC | AUPRC | proba SD | pred-pos @0.5 |
> |---|---|---|---|---|
> | RF-fs Pooled-base (n=24) | 0.724 ± 0.085 | 0.202 | 0.013 | 0.446 |
> | **RF-nofs Pooled-base (n=24)** | **0.667 ± 0.093** | **0.141** | 0.014 | 0.485 |
> | RF-fs Pooled-SW-SMOTE (n=15) | 0.795 ± 0.052 | 0.306 | 0.121 | 0.035 |
> | RF-nofs Pooled-SW-SMOTE (n=6) | 0.875 ± 0.026 | 0.561 | 0.141 | 0.049 |
>
> **The robust claim is the interaction, not the single-cell contrast.** Testing the Pooled-base
> fs-vs-nofs difference against the same difference in the mixed modes (difference-of-differences over
> the shared seeds) gives **+0.222, p = 0.0084** (vs Mixed-in) and **+0.224, p = 0.0020** (vs
> Mixed-out). This was stable across every seed count from n=15 to n=24. So:
> **the feature-count advantage is significantly smaller under Pooled-base than in the rebalanced
> modes** — a statement about an interaction, which the data support.
>
> The single-cell contrast is weaker and test-dependent, and must be reported as such:
> Mann–Whitney (the test §D1 uses throughout) gives **p = 0.015, Cliff's δ = −0.41 (medium)**;
> a seed-paired Wilcoxon gives p = 0.053 (16/24 seeds favour fs). **The unpaired test is the
> appropriate one here** — the two arms' seed-level outcomes are uncorrelated (Pearson r = −0.09,
> p = 0.67), so the SD of the paired difference (0.132) *exceeds* the independent-sampling value
> (0.126); pairing adds noise instead of removing it. Seeds share only the RNG integer: the data split is seed-independent, so a seed fixes
> the RF `random_state` and the TPE trajectory, nothing the two arms hold in common.
>
> **Do not claim** "the full feature set is worse without rebalancing" (the single-cell effect is
> medium at best and fails a paired test), nor "the advantage is absent" as an established negative —
> even at n=24 with SD ≈ 0.09 this cell has only ~60 % power to detect an effect the size of the
> Pooled-SW-SMOTE one (+0.079), so absence of evidence is thin here. The interaction is the finding.
>
> Worth noting for the record: unlike the retracted interim "reversal", this contrast **strengthened
> monotonically as seeds were added** (Mann–Whitney p = 0.159 → 0.039 → 0.021 → 0.015 at n = 15/21/23/24;
> Cliff's δ −0.31 → −0.41), which is what a real effect looks like under a pre-registered seed set.
>
> **What the extra features cost, mechanistically.** AUROC in *both* Pooled-base arms is almost
> entirely determined by one hyperparameter of the winning Optuna trial — the minimum weight fraction
> per leaf (Spearman ρ = −0.97 in each arm independently) — while the Optuna objective itself is flat
> across seeds (best CV-F2 spans ~1 % relative). Optuna is breaking a statistical tie, and whichever
> seed lands on a weakly-regularised configuration scores highest. Since this protocol's train/eval
> splits overlap by ~60 % of rows by construction (the deliberate IV2025 reproduction), weaker
> regularisation means more memorisation of the overlapping rows. This is a *direct, independent
> replication of the paper's memorisation account* — and it explains the large seed dispersion of both
> RF arms (§D3) as tie-breaking rather than noise.

![Recorded AUROC by method and mode](figures/c1_recorded/fig1_auroc_method_mode.png)

*Figure 1. Recorded AUROC (mean ± SD, clipped to [0,1]) for each method across the SW-SMOTE
evaluation modes retained after the Within retirement (Pooled-SW-SMOTE, Mixed-in, Mixed-out;
Pooled-base omitted), with the 0.5 reference line. The error bars use the same SD as the "±" column
of the table above; they are clipped at 1.0 because a symmetric bar on a bounded metric can
otherwise overshoot for small-n, near-ceiling cells (e.g. RF-nofs Mixed-out, n=15, mean 0.912, SD
0.081). RF-nofs is the highest but with the widest dispersion; in the Mixed modes the top-10 RF-fs
is 4th of 5 (SvmW and Lstm both above it); SvmA is the lowest throughout.*

---

## B. Between-method differences within each mode (Kruskal–Wallis)

Kruskal–Wallis across the methods present in each mode, with η²_H, Dunn/Holm post-hoc and
Cliff's δ (details in the script). **The method effect is highly significant in every mode.**

| Mode | H | p | η²_H | Rank (mean AUROC) |
|---|---|---|---|---|
| Pooled-base | 42.8 | 1.1e-08 | 0.64 | RF-fs (0.724) > RF-nofs (0.667) ≫ SvmW ≈ Lstm ≈ SvmA (0.48–0.52) |
| Pooled-SW-SMOTE | 33.4 | 9.9e-07 | 0.86 | RF-nofs (0.875) > RF-fs (0.795) > SvmW > SvmA > Lstm |
| Mixed-in | 47.1 | 1.5e-09 | 0.63 | RF-nofs > **Lstm > SvmW > RF-fs** > SvmA |
| Mixed-out | 40.7 | 3.0e-08 | 0.54 | RF-nofs > **Lstm > SvmW > RF-fs** > SvmA |

- **Under Pooled-base both RF variants separate from the 0.5 level, and only they do** (each vs
  SvmW / SvmA / Lstm: RF-fs p_holm ≤ 0.0008, RF-nofs p_holm ≤ 0.03); the other three are
  statistically indistinguishable from one another (all p_holm = 1.0). **RF-fs vs RF-nofs is itself not
  separable in this mode** (Dunn/Holm p = 0.29) — the family-wise correction across ten pairs is
  conservative, and §D1's direct two-sample test of the same contrast reaches p = 0.015; report the
  contrast as medium and test-dependent either way. Filling the RF-nofs cell lowered η²_H here from
  0.84 to 0.64, because the mode now contains two discriminating methods instead of one.
- **RF (top-10) leads pooled only.** In both Mixed modes the top-10 RF-fs is **4th of 5**: setting
  Lstm aside as non-commensurable (DRT target), **SvmW beats RF-fs in both** (0.742 vs 0.719;
  0.771 vs 0.749). Only RF-nofs — this study's own full-feature ablation, and its most dispersed
  variant — leads Mixed. The defensible regime-scoped claim is that RF is the only method that
  works pooled and the only one domain restriction does not move (C), not that it has a higher
  Mixed ceiling.
- **SvmA is the consistent lowest rank** in every mode (p_holm < 0.02, |δ| ≥ 0.75–1.0).
- **RF-nofs is the consistent top rank** across the SW-SMOTE modes (mechanism in D1).

---

## C. Within-method mode contrasts (seed-paired Wilcoxon signed-rank)

Paired across shared seeds; median Δ and two-sided p. Contrasts are anchored on the retained
regimes: domain restriction is Pooled→Mixed and domain shift is Mixed in→out.

| Method | Imbalance (base→SW-SMOTE) | Domain restriction (Pooled→Mixed) | Domain shift (Mixed in→out) |
|---|---|---|---|
| **RF-fs** | Δ=+0.060, **p=0.035** | Δ=−0.039, **p=0.005** | Δ=+0.032 **p=8e-06** |
| **RF-nofs** | Δ=+0.210, **p=0.031** | Δ=−0.009, p=0.56 (n.s.) | Δ=+0.066 **p=6e-05** |
| **SvmW** | Δ=+0.179, **p=0.031** | Δ=+0.052, **p=0.031** | Δ=+0.030 **p=0.008** |
| **SvmA** | Δ=+0.068, p=0.063 | Δ=−0.017, p=0.69 (n.s.) | Δ=+0.060 **p=0.001** |
| **Lstm** | Δ=+0.005, **p=1.0 (inactive)** | Δ=+0.268, **p=0.031** | Δ=−0.006, p=0.25 (n.s.) |

**RF-nofs's imbalance response (+0.210, 6/6 seeds positive) is the largest of any method in the study**
— larger than SvmW's +0.179, and 3.5× RF-fs's +0.060. The contrast is only testable at all because a
6th Pooled-SW-SMOTE seed was added on 2026-08-14 (a 5-pair Wilcoxon caps at p=0.0625). Reading: the
**imbalance-robustness that this study attributes to "RF" is a property of the top-10 variant**; the
full-feature variant depends on rebalancing about as much as SvmW does. Note also that RF-nofs's
domain-restriction contrast moved from +0.008 (p=1.0) to −0.009 (p=0.56) when that 6th seed landed —
still firmly non-significant, so the "domain restriction does not move RF-nofs" reading is unchanged.

- **Lstm is imbalance-inactive** (base→SW-SMOTE Δ≈0, p=1.0) but shows the **largest
  domain-restriction change** (Pooled→Mixed Δ=+0.27): its recorded AUROC is governed by the
  evaluation regime, not by rebalancing — consistent with its near-balanced DRT target. Notably its
  Mixed in→out shift is *not* significant (Δ=−0.006, p=0.25), so the swing is regime-driven, not a
  sensitivity to the in/out domain split (D6).
- **RF-fs has a small but significant imbalance change**, a significant Pooled→Mixed drop
  (Δ=−0.039), and significant, consistent in→out sensitivity.

---

## D. Model-characteristic quantities

### D1. RF feature-count effect (fs top-10 vs nofs all-165) — Mann–Whitney + Cliff's δ

| Mode | RF-fs | RF-nofs | Δ | p | Cliff's δ |
|---|---|---|---|---|---|
| **Pooled-base** | **0.724** | **0.667** | **−0.056** | **0.015** | **−0.41 (medium)** |
| Pooled-SW-SMOTE | 0.795 | 0.875 | +0.079 | **0.002** | +0.82 (large) |
| Mixed-in | 0.719 | 0.846 | +0.127 | **1e-04** | +0.74 (large) |
| Mixed-out | 0.749 | 0.912 | +0.163 | **4e-05** | +0.79 (large) |

**Using all 165 features instead of the top-10 raises RF's recorded AUROC by ≈0.08–0.16 (large effect)
in every rebalanced mode — but that advantage is conditional on the rebalancing.** The Pooled-base row
(added 2026-08-15 when the never-run cell was filled) is the only one where it does not hold: the point
estimate goes the other way (−0.056) at a *medium* effect size, p = 0.015. All rows use the same
unpaired Mann–Whitney test, which is the appropriate one — the arms' per-seed outcomes are uncorrelated
(r = −0.09), so seed-pairing would add noise rather than remove it (see the §A box).

**The defensible claim is the interaction**: the Pooled-base contrast differs significantly from the
Mixed contrasts (difference-of-differences +0.222, p = 0.008 vs Mixed-in; +0.224, p = 0.002 vs
Mixed-out). The single Pooled-base cell on its own is medium-sized and only borderline under a paired
test (p = 0.053), and the cell is underpowered (~60 %) for an effect the size of the Pooled-SW-SMOTE
one — so it should not be reported as an established null, nor as a "reversal". A controlled dose-response (D1b) further
shows the dependence **saturates by k≈20**, so it is not an open-ended benefit of ever-more features. A controlled dose-response (D1b) further
shows the dependence **saturates by k≈20**, so it is not an open-ended benefit of ever-more features.
(RF-nofs Mixed cells are final at n=15.)

![RF feature-count effect](figures/c1_recorded/fig2_rf_feature_count.png)

*Figure 2. RF feature-count effect: all 165 features (RF-nofs) vs the top-10 (RF-fs) per retained
mode (Pooled-SW-SMOTE, Mixed-in, Mixed-out).*

### D1b. Feature-count dose-response — where does the gain saturate?

The controlled probe is complete under **pooled** evaluation — the deployment regime that scores
the model on the whole 87-recording cohort (plain RF, RF-importance top-k, recorded evaluation, 3
seeds). It compares selecting top-k after SW-SMOTE (the c1 pipeline order) with selecting on the
natural training data.

**The dependence is not monotone to 165 — it saturates by k≈20 in both selection orders.** The
c1-order curve rises sharply before k=20 (0.69 at k=5 → 0.90 at k=20) and stays around 0.91
thereafter; natural-data selection is already near that plateau by k=5–10. The recorded pooled
anchors are RF-fs (Pooled-SW-SMOTE) = 0.795 and RF-nofs = 0.866, which are not directly comparable
to the simplified probe's absolute levels. The transferable result is the saturation shape, not the
raw level. The same k≈20 knee **reproduces under mixed-in** (the diagnostic regime; anchors 0.719 /
0.829).

Practical reading: **the probe supports a compact feature set (approximately 20 vehicle-dynamics
features); the top-10 arm's low recorded value is primarily consistent with the
SMOTE-before-selection order rather than with ten features being intrinsically insufficient.**

![RF feature-count dose-response](figures/c1_recorded/fig5_feature_dose_response.png)

*Figure 5. Recorded pooled AUROC vs feature count k. Both selection orders plateau by k≈20; the
recorded c1 anchors (RF-fs k=10 = 0.795 ★, RF-nofs k=165 = 0.866 ◆) sit below the curve. Probe =
plain RF with fixed hyperparameters and a simplified evaluation split, so absolute levels are not
directly comparable to the tuned c1 pipeline; the saturation shape is what transfers (and it
reproduces under mixed-in, the diagnostic regime).*

### D2. Decision spread / degeneracy (specificity, predicted-positive rate)

| Method · mode | proba SD | specificity | pred-positive rate |
|---|---|---|---|
| SvmW · Pooled-base | 0.001 | 0.004 | 0.997 |
| SvmW · Pooled-SW-SMOTE | 0.199 | 0.540 | 0.472 |
| Lstm · Pooled-base | — | 0.003 | 0.999 |
| Lstm · Pooled-SW-SMOTE | — | 0.005 | 0.997 |
| RF-fs · Pooled-base | 0.013 | 0.566 | 0.446 |
| RF-nofs · Pooled-base | 0.014 | 0.523 | 0.485 |
| RF-nofs · Pooled-SW-SMOTE | 0.141 | 0.974 | 0.049 |

Under Pooled-base **SvmW is an all-positive constant classifier** (probability spread ≈ 0,
specificity ≈ 0). **SW-SMOTE restores a usable probability ranking for SvmW** (spread 0.20,
specificity 0.50) — a de-degeneration. Lstm's Pooled collapse is a majority-class artefact of a
near-balanced target and is unaffected by SMOTE; **neither RF variant is ever degenerate** — both keep
a usable ranking under Pooled-base (specificity 0.52–0.57, spread ≈ 0.013), which is what distinguishes
them from SvmW. The RF-nofs rows added 2026-08-14 confirm this: the full-feature variant's weaker
Pooled-base score (§D1) is **not** a degeneracy — its probability spread matches RF-fs's almost exactly.
Both RF arms' spreads rise ~10× under SW-SMOTE (0.013 → 0.121/0.141) as the decision threshold moves off
the majority class (predicted-positive rate 0.45/0.48 → 0.035/0.049).

![Decision spread (specificity) under Pooled](figures/c1_recorded/fig3_specificity.png)

*Figure 3. Specificity under Pooled (≈0 = all-positive). SW-SMOTE lifts SvmW's specificity from
≈0.004 to ≈0.50; Lstm stays collapsed; RF is non-degenerate.*

### D3. Across-seed stability (Brown–Forsythe equal-variance test per mode)

| Mode | RF-fs | RF-nofs | SvmW | SvmA | Lstm | Brown–Forsythe |
|---|---|---|---|---|---|---|
| Pooled-base | 0.085 | 0.093 | 0.011 | 0.008 | 0.011 | W=4.83, **p=0.002** |
| Pooled-SW-SMOTE | 0.052 | 0.026 | 0.018 | 0.042 | 0.006 | W=1.94, p=0.13 (n.s.) |
| Mixed-in | 0.085 | 0.077 | 0.012 | 0.026 | 0.009 | W=5.19, **p=0.001** |
| Mixed-out | 0.104 | 0.081 | 0.016 | 0.022 | 0.009 | W=4.41, **p=0.003** |

**RF (both variants) is markedly the least seed-stable method** — its across-seed SD (0.05–0.10)
is up to an order of magnitude larger than SvmW's / Lstm's (≈0.01), with significant variance
heterogeneity in every mode except Pooled-SW-SMOTE. This reflects the seed sensitivity of the RF
ensemble + Optuna pipeline relative to the near-deterministic SVM and LSTM pipelines.

**The mechanism is now identified (2026-08-14), and it is tie-breaking rather than noise.** In both
Pooled-base arms the recorded AUROC is almost perfectly predicted by a single hyperparameter of the
winning Optuna trial — the minimum weight fraction per leaf (Spearman ρ = −0.97 in each arm,
independently) — while the Optuna objective is flat across seeds (best CV-F2 varies by ~1 % relative).
The search is therefore choosing between near-equivalent configurations, and the seed decides which;
because this protocol's train and eval splits overlap by ~60 % of rows by construction, the
weakly-regularised winners memorise more of the overlap and score higher. **RF's seed instability in
this study is a symptom of the evaluation protocol, not of the classifier alone** — the same reading
the paper's memorisation account gives, arrived at independently.

![Across-seed variability by method](figures/c1_recorded/fig4_seed_variability.png)

*Figure 4. Mean across-seed SD of AUROC over the Mixed modes. RF ≫ SvmW / SvmA / Lstm.*

### D4. SvmA under-performance: a learner effect, not a feature-set effect

SvmA records the lowest AUROC of all methods (A: 0.53–0.60 within/mixed; 0.481 Pooled-base). On a
recorded-value basis the cause is the *learner*, not the steering feature set and not class
imbalance.

**Same features, different learner (recorded pooled).** SvmA's own 36 steering features fed to RF
reach 0.893, close to the full 165-feature vehicle set (0.904), whereas an RBF-SVM (SvmA's learner)
on the same 36 features gives only 0.544, matching the recorded SvmA Pooled-SW-SMOTE band (0.569,
A). The steering feature set is therefore not the bottleneck under the recorded protocol: it yields
as much recorded separability as the full set when a tree ensemble reads it; the RBF-SVM cannot.

| Feature set → learner | Recorded AUROC (pooled) |
|---|---|
| SvmA 36 steering → RF | 0.893 |
| SvmA 36 steering → RBF-SVM | 0.544 |
| all 165 vehicle → RF | 0.904 |

**Imbalance treatment does not lift the SVM.** Under the pooled recorded split, neither SW-SMOTE
nor class weighting moves the RBF-SVM off approximately 0.55 (0.558 / 0.557 / 0.544), while RF stays
around 0.88–0.90 across the same treatments. Class imbalance is therefore not the cause of SvmA's
low score. (The learner gap reproduces as a diagnostic under mixed-in: RF 0.877 vs RBF-SVM 0.544.)

**SvmA's bottom rank is thus a learner limitation** (RBF-SVM on steering features), not a deficient
feature set and not an imbalance artefact — unlike SvmW, whose Pooled degeneracy *is* a
decision-function artefact that SW-SMOTE repairs (D2).

![SvmA learner vs feature effect](figures/c1_recorded/fig6_svma_learner_vs_feature.png)

*Figure 6. Recorded pooled AUROC. (A) The same 36 steering features reach the full-set ceiling
under RF (0.893 vs 0.904) but only 0.544 under the RBF-SVM that SvmA uses (≈ its recorded c1
Pooled-SW-SMOTE value 0.569). (B) Neither SW-SMOTE nor class weighting lifts the RBF-SVM off ~0.55,
while RF stays ~0.88–0.90. Probe = plain learners on a recorded-style split; the learner contrast is
the point (and it reproduces under mixed-in, the diagnostic regime).*

### D5. SvmW under imbalance: a recoverable learner degeneracy, not a feature deficit

SvmW behaves oppositely to SvmA. Its wavelet features carry recorded drowsiness signal, but its
SVM cannot absorb severe class imbalance unaided.

**Collapse under imbalance, recovery under SW-SMOTE.** Under Pooled-base (no resampling, natural
3.9% minority) SvmW degenerates to an all-positive constant classifier — AUROC 0.519, specificity
0.004, predicted-positive rate 0.997, probability SD 0.001 (D2). SW-SMOTE de-degenerates it: the
Pooled AUROC rises to 0.684 (+0.165), specificity to 0.540 and probability SD to 0.199 — a usable
ranking is restored. With SW-SMOTE in place, SvmW then delivers 0.74–0.77 across the Mixed modes
(A).

| SvmW (Pooled) | AUROC | specificity | pred-pos rate | proba SD |
|---|---|---|---|---|
| base (no SMOTE) | 0.519 | 0.004 | 0.997 | 0.001 |
| +SW-SMOTE | 0.684 | 0.540 | 0.472 | 0.199 |

**The features carry the signal.** The failure is the SVM's, not the feature set's: feeding SvmW's
8 steering-wheel wavelet features to RF (recorded-style, pooled) gives ~0.87 under *every*
imbalance treatment — raw 0.874, SW-SMOTE 0.878, class-weight 0.871 (Fig 7B) — so a robust learner
reads the recorded signal from them without any rebalancing, and once imbalance is handled the SVM
itself reaches 0.74–0.80. SvmW's low Pooled score is therefore a *recoverable* decision-function
degeneracy under imbalance — the mirror image of SvmA, whose low score is a learner ceiling that
rebalancing does not lift (D4). (The ~0.87 RF read reproduces as a diagnostic under mixed-in:
0.877 / 0.858 / 0.871.)

![SvmW imbalance collapse and SW-SMOTE recovery](figures/c1_recorded/fig7_svmw_imbalance_recovery.png)

*Figure 7. (A) Recorded SvmW AUROC by regime: Pooled-base collapses to ~0.52 (all-positive),
SW-SMOTE recovers it to 0.684 (mixed shown as diagnostic). (B) The same 8 steering-wheel wavelet
features fed to RF reach ~0.87 under every pooled imbalance treatment (raw 0.874, SW-SMOTE 0.878,
class-weight 0.871) — a robust learner reads the recorded signal without rebalancing, so SvmW's
collapse is a learner–imbalance interaction, not a feature deficit. (The de-degeneration metrics —
specificity 0.004→0.540, predicted-positive rate 0.997→0.472, probability SD 0.001→0.199 — are
tabulated above.)*

### D6. Lstm: insensitive to imbalance, governed by the evaluation regime

Lstm's recorded AUROC is decoupled from class-imbalance treatment but strongly tied to the
evaluation regime. Seed-paired (C): the imbalance contrast (Pooled-base→Pooled-SW-SMOTE) is
Δ=+0.005 (p=1.0, inactive), whereas the domain-restriction contrast (Pooled→Mixed) is Δ=+0.268
(p=0.031) — the largest of any method. Its recorded AUROC swings from ~0.51 under Pooled to ~0.78
across the Mixed modes, while SW-SMOTE moves it by essentially zero.

| Lstm axis (seed-paired) | Δ AUROC | p |
|---|---|---|
| imbalance (base→SW-SMOTE) | +0.005 | 1.0 (inactive) |
| domain restriction (Pooled→Mixed) | +0.268 | 0.031 |
| domain shift (Mixed in→out) | −0.006 | 0.25 (n.s.) |

**Caution for domain-shift robustness.** Because Lstm's strong Mixed numbers are so
regime-dependent — collapsing to ~0.51 under Pooled — they are protocol-specific and should not be
read as evidence of robust generalisation. Two points sharpen this: (i) Lstm predicts the DRT
`event_label` (a near-balanced construct, different from the KSS label the other methods use), so
its Pooled behaviour and absolute levels are not directly comparable, and the large Pooled→Mixed
swing partly reflects that target rather than domain per se; (ii) the *pure* in→out domain-shift
effect is **not significant under Mixed** (Δ=−0.006, p=0.25), so the swing is a regime effect rather
than a sensitivity to the in/out domain split. The defensible reading is therefore: **Lstm's
recorded performance is governed by the evaluation protocol, not by class imbalance — so its high
domain-aware numbers warrant caution as an indicator of domain-shift robustness.**

![Lstm regime sensitivity](figures/c1_recorded/fig8_lstm_regime_sensitivity.png)

*Figure 8. (A) Recorded Lstm AUROC across modes: the imbalance pair (Pooled-base vs Pooled-SW-SMOTE)
is flat (Δ+0.005), while the jump into the Mixed regime is large (Δ+0.268); the Mixed in→out shift
is not significant. (B) Seed-paired |ΔAUROC| by method: only Lstm is flat on the imbalance axis yet
largest on the domain-restriction (Pooled→Mixed) axis (RF-fs and SvmW both respond to imbalance).*

---

## E. Two-way structure (method × mode), Scheirer–Ray–Hare

With the Within modes retired, the balanced two-way design reduces to 5 methods × **2 Mixed modes
(in/out)** × 8 common seeds = 80 observations. The "mode" factor is therefore just the in/out domain
contrast (df=1) — a much weaker design than the previous four-sub-mode one.

| Effect | H | df | p |
|---|---|---|---|
| **Method** | 51.1 | 4 | **2.1e-10** |
| Mode (Mixed in/out) | — | 1 | 0.13 (n.s.) |
| Method × Mode | — | 4 | 0.83 (n.s.) |

**The classifier is the overwhelmingly dominant factor; the Mixed in/out contrast does not differ
significantly and there is no significant method × mode interaction.** With only two modes the
mode-effect test is low-powered, so this null should be read as "no *detectable* in/out difference
in the Mixed regime" rather than a strong claim of equivalence. The method ranking
(RF-nofs > Lstm > SvmW > RF-fs > SvmA) is stable across Mixed in/out.

---

## F. Seed-count convergence (per paper-target case)

Following the TIV2026 / exp2 seed-validity framework (exp2 fig8): for **every reported case** the
seeds are added one at a time (fixed augmentation order) and the running mean AUROC and its 95 %
t-CI half-width are tracked. A case is **adequate** when

- *discriminating* (running mean > 0.55): the 95 % CI half-width ≤ 0.05 **and** the running mean has
  flattened (last-3 span ≤ 0.01); or
- *near-0.5* (running mean ≤ 0.55): the percentile-bootstrap 95 % CI upper bound < 0.60 (excludes any
  weak signal).

**All 20 cells are adequate** — including the 20th (RF-nofs × Pooled-base) that had never been run,
which closed 2026-08-15 at n=24 with CI half-width 0.039 and a flat running mean. **RF-fs × Pooled-base
is adequate for the first time on both criteria** (half-width 0.036, last-3 span 0.0096): it had carried
a 0.0499 "borderline" CI at n=15, and at the intermediate n=21 it met the interval (0.041) but failed
flatness (0.0115). Worth stating plainly: extending that arm moved its running mean 0.738 → 0.727 →
0.724, so **the n=15 value was mildly optimistic**. Both arms are now at the full 24-seed
`SEED_MASTER` set, the same count RF uses in the Mixed/Within cells. Final
95 % CI half-width per cell (✓ = adequate; "near-0.5" = judged by the bootstrap-upper < 0.60 rule):

| Method | Pooled-base | Pooled-SW-SMOTE | Mixed-in | Mixed-out |
|---|---|---|---|---|
| **RF (fs)** | n=24, hw 0.036 ✓ | n=15, hw 0.029 ✓ | n=24, hw 0.036 ✓ | n=24, hw 0.044 ✓ |
| **RF (nofs)** | **n=24, hw 0.039 ✓** | n=6, hw 0.027 ✓ | n=15, hw 0.043 ✓ | n=15, hw 0.045 ✓ |
| **SvmW** | n=6, hw 0.011 ✓(near-0.5) | n=6, hw 0.019 ✓ | n=8, hw 0.010 ✓ | n=8, hw 0.013 ✓ |
| **SvmA** | n=6, hw 0.009 ✓(near-0.5) | n=6, hw 0.044 ✓(near-0.5) | n=11, hw 0.017 ✓(near-0.5) | n=11, hw 0.015 ✓ |
| **Lstm** | n=6, hw 0.011 ✓(near-0.5) | n=6, hw 0.007 ✓(near-0.5) | n=15, hw 0.005 ✓ | n=15, hw 0.005 ✓ |

- **RF-nofs Pooled-SW-SMOTE is adequate at n=6** (CI half-width 0.027): the small seed count is
  offset by a low across-seed SD (0.026), so the running mean is converged — **no extra seeds are
  required** for this cell despite its low n. (It was n=5 until 2026-08-14; the 6th seed was added so
  the §C imbalance contrast could clear p<0.05, since a 5-pair Wilcoxon caps at p=0.0625.)
- **RF-fs Pooled-base is adequate at n=24** on both criteria — see the paragraph above. Its flatness
  margin is thin (0.0096 against the 0.01 threshold) and its running mean is still drifting slightly
  downward, so the point estimate should be read as ±0.01 rather than exact. `SEED_MASTER` is exhausted
  at 24, so any further extension would mean inventing seeds — a discretion this arm has deliberately
  avoided throughout.
- **Seed counts here are not tuned per cell.** Both Pooled-base arms use the pre-registered
  `SEED_MASTER` order, extended in whole waves, and the extension was decided on the CI criterion
  before the values were inspected. This matters because interim readings of the RF-nofs cell at
  n=9/n=10 looked significant and did not survive (see the retraction in §A): with a metric this
  seed-noisy, stopping when a result looks good is exactly the failure mode to avoid.
- Every other reported cell converges with margin (CI bands narrow and means flatten with k).

![Seed-count convergence](figures/c1_recorded/fig_seed_convergence.png)

*Figure. Running mean AUROC ± 95 % CI vs number of seeds for every method × mode reported. Blue =
discriminating (target CI half-width ≤ 0.05); red = near-0.5 baseline (bootstrap CI upper < 0.60). Each panel
annotates the final n, CI half-width and verdict. Reproducibility:
`scripts/python/analysis/exp3_c1_seed_convergence.py` (output also in
`results/analysis/exp3_verification/c1_seed_convergence.json`).*

---

## Synthesis — model-dependent structure of the recorded metrics

- **RF (fs):** separates from 0.5 under Pooled-base (B), where it is the highest-scoring method
  (0.724, though not separably so from RF-nofs under the family-wise correction: Dunn/Holm p=0.29),
  with a small
  significant imbalance change (C); but the **least seed-stable** method (D3), and, with all
  features (RF-nofs), a **feature-count dependence** that saturates by k≈20 (D1, D1b).
- **RF-nofs:** the highest recorded AUROC **in the rebalanced modes** (and only there); the
  feature-count gain saturates by k≈20 (D1b) and is **significantly smaller under Pooled-base than in
  the Mixed modes** (interaction p=0.008/0.002), so its lead is conditional on imbalance handling rather
  than a property of the feature set. Its own imbalance response (+0.210, 6/6 seeds) is the **largest in
  the study**, which relocates the "RF is imbalance-robust" finding to the top-10 variant specifically
  (§A box, §C, §D1). Its Pooled-base deficit is **not** a degeneracy — the probability spread matches
  RF-fs's (§D2).
- **SvmW:** all-positive **degenerate** without rebalancing; SW-SMOTE **de-degenerates** it (D2,
  D5) and its wavelet features then deliver 0.74–0.77 in Mixed — a *recoverable* learner degeneracy
  under imbalance, not a feature deficit (mirror image of SvmA). But in the deployable regimes it
  still **beats the top-10 RF in Mixed** (B), so it is not dominated once Within is set aside.
- **SvmA:** bottom rank is a **learner limitation** (RBF-SVM), not a feature-set or imbalance
  effect — the same steering features under RF reach the full-set ceiling (D4).
- **Lstm:** **imbalance-inactive** (C, D2) but strongly **regime-driven** (largest Pooled→Mixed
  change, D6); its Mixed AUROC reflects the near-balanced DRT target, a different construct
  from the KSS label used by the other methods, so its absolute level is not directly comparable.
  Its performance is governed by the evaluation protocol rather than rebalancing — **so its high
  domain-aware numbers warrant caution as an indicator of domain-shift robustness** (D6).
- **Across methods (E):** the classifier dominates; mode and interaction are non-significant.
