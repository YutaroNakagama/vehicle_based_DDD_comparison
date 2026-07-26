# Experiment 3 — c1 Domain Comparison Grid + IV2025 Baseline Reproduction (Results & Progress)

> Canonical record of the design, results, and verification status of the local execution campaign started 2026-06-27.
> For the old HPC campaign see [operations_log.md](operations_log.md), for the 2026-05 local
> Phase 1–5 see [local_execution.md](local_execution.md), and for the detailed methodological
> decisions/verification see [verification_log.md](verification_log.md).
> **Last updated: 2026-07-24** (**c1 grid fully complete**: RF/SvmW/SvmA/Lstm, iv25 baseline, RF-nofs all completed, SvmW 32/32, all evals clean = verification_log 2026-07-24 section). **Policy decision (user, 07-20)**: complete exp3 **with the same methodology as IV2025/TIV2026** and
> prioritise cross-paper comparability above all (following the judgement in verification_log 2026-07-01). The numbers in §2 are the values under this **same protocol
> (including the known limitation of the train/eval split mismatch)**. **§3.6 is retained as a de-leak / robustness appendix**
> (under an honest split, vehicle→EEG is expected to be chance), and **the implementation review / de-leak re-test is kept "on standby" non-destructively** (see
> §3.7 plan below). Only the previous day's claim that "RF with all features = feature selection underrates RF" is withdrawn as a factual error (memorisation leak).)
>
> **Key point**: The consistent overall finding is that "**RF is the only method that works under pooled (the non-domain-restricted evaluation corresponding to real operation)**". RF is immovable at ~0.74 across pooled↔mixed, whereas the other methods collapse to chance under pooled
> and recover to ~0.75 only under domain-restricted evaluation (within/mixed/cross). For the detailed mechanism see §3 (the difference in evaluation protocol between Pooled and
> Mixed).
> **[2026-07-19 correction]** The "honest pooled=0.86 with all 160 features" written here the previous day is **withdrawn** (it was a memorisation leak from the train/eval
> split mismatch, §3.6). **An even more serious implication**: the absolute AUROC of this pooled column itself (including RF ~0.74)
> is contaminated by leakage, so the central claim "RF works under pooled" requires re-verification under an honest split (when correctly separated,
> vehicle→EEG is expected to be chance ~0.52 for all methods, consistent with [[project_exp2_rf_087_unreproducible]]).

## 1. Purpose and Design

Based on the supervisor's revised plan (2026-06-27/28), we compare the 4 methods (RF / SvmW / SvmA / Lstm)
under **domain shift × class imbalance**:

- **c1 grid (6 conditions)**: {Within=`target_only`, Cross=`source_only`, Mixed=`mixed`}
  × {in_domain, out_domain}. All evaluated on the target domain. SW-SMOTE(ratio 0.5),
  wasserstein/KNN domain split (split2, in=44 / out=43 subjects).
- **IV2025 baseline (Pooled, no treatment)**: pooled, no imbalance handling — a local reproduction of the
  IV2025 published setting (tag `iv25base_`). The reference for the before/after comparison against c1 (after).
- **Pooled + SW-SMOTE arm (Pooled, with treatment)** (added 2026-07-04, tag `iv25smote_`): the pooled
  training/evaluation keeps exactly the IV2025 protocol, and adds **only the same SMOTE (ratio 0.5) as c1**.
  This lets us cleanly separate "**the effect of imbalance treatment**" (Pooled no-treatment→with-treatment) from "**the effect of
  domain restriction**" (Pooled with-treatment→Mixed/Within). Launched with `iv2025_baseline_launcher.py
  --smote`, eval uses fixed `--jobid` (race-free).
  > **[2026-07-21 · important / deliberate deviation from the intended methodology]** This pooled arm, even when passed `--subject_wise_oversampling`,
  > was **in fact running with pooled SMOTE for all 4 methods** (`falling back to pooled oversampling`:
  > RF 20/20 · SvmW 5/6 · SvmA 6/6 · **Lstm 6/6**, `Applying subject-wise`=0). **In c1, Within/Mixed have subject-wise
  > working correctly** (SvmW target 16/16 · mixed 16/16). **Root cause**: the iv25 launcher does not pass `--time_stratify_labels` under pooled
  > → routes via `data_time_split_by_subject` (split.py L468 drops subject_id unconditionally, no keep_subject_id support)
  > → subject-wise impossible. **Because this violates the design of "all imbalance treatment is SW-SMOTE", a fix (Option A: add keep_subject_id to `data_time_split_by_subject`)
  > + regeneration of the iv25smote pooled SW-SMOTE is required** (verification_log 2026-07-21 addendum 2, needs user judgement).

**Rationale for the distance metric and ratio selection (derived from exp2/TIV2026, fixed in this experiment)**:
- Distance = **wasserstein**, fixed. In the Sobol sensitivity analysis (`results/analysis/.../sobol_indices.csv`), the
  contribution of distance is S1≈0.0009 · ST≈2% (negligible against Mode's S1≈0.58). Across the 3 distances the AUROC differences are Within
  0.771/0.761/0.790 (no significant difference, p=0.128). We adopt wasserstein, which is numerically the best and the cheapest (the original
  exp3 distance-selection experiment was cancelled on the basis of the Sobol results).
- Ratio = **0.5**, fixed. In the TIV2026 ratio sensitivity analysis the AUROC ranking is almost invariant to the ratio (Spearman ρ=1.00).
  Only threshold-dependent metrics such as F2 are ratio-sensitive. ratio 0.1 is infeasible for Lstm (event_label natural minority rate
  ~27% > target 10% causes an imblearn error) → we adopt 0.3/0.5 as the feasible set common to all methods, with c1 using 0.5.
- Labels are **faithful to each method's original paper** (RF/SvmW/SvmA=KSS, Lstm=event_label ~74% positive). No unification
  is performed (a design decision). For SvmA, the T1 probe already confirmed no signal in the features, so even changing the label
  stays chance (feature-bound) and unification has no practical benefit. Since Lstm's event_label is a different task, we avoid
  cross-comparing absolute AUROC values and instead compare via the collapse/recovery pattern.

Each method is faithful to its original paper (RF=top-10 features + KSS, SvmW=Zhao2009 GHM wavelet 8 bands
+ KSS + Optuna 50 trials, SvmA=Arefnezhad2019 steering statistics 18 features + PSO/ANFIS/SVM,
Lstm=Wang2022 + event_label). The only operational change is SvmA's `SVMA_PSO_MAXITER=30`
(pso_history confirmed convergence at iter~2, the method itself is unchanged — see verification_log).

### Seed Design (proportional to variance, judged with the TIV2026 adequacy framework)

| Method | # seeds | Rationale |
|---|---|---|
| RF | 24 | Within seed variance is large (std ~0.09–0.11) → n=21 needed for 95%CI half-width ≤0.05 |
| Lstm | 15 | Low variance (req_n=3) but maintaining the conventional level |
| SvmW / SvmA | 8 | Low variance (req_n≤7) + computational constraint of ~15h per cell. Statistical sufficiency is satisfied |
| IV25 RF | 15 / others 6 | RF is high-variance, chance methods satisfy CI upper bound <0.60 at 6 |

### Split methodology (an important known limitation)

**We adopt the same methodology as TIV2026/IV2025** (train: label-stratified time-series split via
`--time_stratify_labels`, eval: target_timewise stratify=False). This combination has the property that the
train–eval split boundaries do not coincide under within/mixed, a temporal-split characteristic
(in the de-leaked version, within goes 0.78→0.526). **Prioritising cross-paper consistency, we unify under the same framework**
and note it explicitly as a known limitation (details/measurements in verification_log.md 2026-07-01 section).
Cross is not affected by this property because it trains on a different domain.

## 2. Results (as of 2026-07-05, AUROC mean ± std (n))

> **Note (2026-07-20 policy)**: The Pooled/Within/Mixed numbers below are the values under **the same evaluation protocol as IV2025/TIV2026
> (train with `--time_stratify_labels`, eval with stratify=False)**, and we complete/report under this framework for cross-paper comparability
> (a known limitation, verification_log 2026-07-01). **This framework contains row duplication from the train/eval split mismatch, and the absolute
> AUROC is affected by it** (the de-leak behaviour and honest values are in the **§3.6 robustness appendix**;
> relative comparisons = method-vs-method / before-after are interpreted within the same framework). **The de-leak re-test is prepared in §3.7** (on standby).

### Pooled 2 conditions (train/eval on all subjects) — "the effect of treatment"

| Method | Pooled no treatment (`iv25base`) | Pooled + SW-SMOTE (`iv25smote`) | Δ(SMOTE) |
|---|---|---|---|
| **RF** | **0.738 ± 0.090 (15)** | **0.748 ± 0.061 (15)** | +0.010 |
| SvmW | 0.519 ± 0.011 (6, degenerate) | **0.717 ± 0.016 (5)** | **+0.198** |
| Lstm | 0.512 ± 0.011 (6) | 0.508 ± 0.004 (6) | −0.004 |
| SvmA | 0.481 ± 0.008 (6) | 0.533 ± 0.066 (6) | +0.052 (within chance band) |

→ **[2026-07-12 update · important] The old conclusion "only RF works under pooled" needs revision due to the SvmW result**.
**SvmW recovers substantially with SW-SMOTE from 0.519 (all-positive degeneracy) → 0.717 (non-degenerate, predicting both classes, recall_pos ~0.72)**
(**n=5 completed 07-16**, seeds{0,1,7,42,123}=0.689–0.728). That is, **the methods that work under pooled(+SMOTE) are the two methods
RF and SvmW**; Lstm(0.508) and SvmA(0.533) stay chance even with SMOTE added. → "the effect of treatment" splits into two groups by
method: **RF=robust regardless of treatment / SvmW=under pooled, SMOTE releases the degeneracy and it works / Lstm·SvmA=feature-/
task-bound and do not recover**. The central claim of §3–4 that "RF is the only one that works under pooled" has been **revised to include SvmW**
(§3.5: only SvmW rivals RF under pooled, but seed-paired RF leads by +0.039). RF's Δ is
small because it is internally `class_weight`+calibrated.

### c1 grid (with SW-SMOTE, evaluated on the target domain)

| Condition | RF | Lstm | SvmW | SvmA |
|---|---|---|---|---|
| Within-in | 0.746 ± 0.089 (24) | 0.779 ± 0.007 (15) | 0.800 ± 0.012 (8) | 0.576 ± 0.029 (8) |
| Within-out | **0.778 ± 0.108 (24)** | 0.763 ± 0.012 (15) | 0.759 ± 0.012 (8) | 0.574 ± 0.074 (8) |
| Mixed-in | 0.719 ± 0.085 (24) | 0.782 ± 0.009 (15) | 0.742 ± 0.011 (8) | 0.532 ± 0.024 (8) |
| Mixed-out | 0.749 ± 0.104 (24) | 0.779 ± 0.009 (15) | 0.771 ± 0.015 (8) | 0.597 ± 0.025 (8) |
| Cross-in | 0.519 ± 0.006 (24) | **0.733 ± 0.015 (15)** | 0.506 ± 0.005 (2) | 0.512 ± 0.021 (8) |
| Cross-out | 0.507 ± 0.004 (24) | **0.747 ± 0.012 (15)** | 0.514 ± 0.004 (2) | 0.504 ± 0.019 (8) |

(Within/Mixed are independent cells per in/out. Mixed = train on all 87 subjects, evaluate on target domain. SvmA
**completed** 8/8 seeds (all conditions confirmed in the chance band). **SvmW also completed 32/32** (Within 8/8 · Mixed 8/8, Cross 2/8 demoted).
**[2026-07-24 complete] c1 grid completed for all methods, all iv25 baselines, and RF-nofs** (the watchdog also self-terminated at 07-23 12:18 with "ALL COMPLETE"). Zero abnormal terminations, all values within expected bands.)

- **RF**: 24/24 seeds completed. Within-out 0.778 is consistent with the TIV2026 within reference value.
  Seed variance is large (Within-out range 0.568–0.954) → the rationale for 24 seeds.
  Cross fully collapses (~0.51, confirmed via the prediction distribution to be true chance = minority-tracking).
- **Lstm**: 15/15 completed. **The only domain-invariant method** (all 6 conditions 0.73–0.78). Note, however,
  that its label is event_label (~74% positive) so its detection target differs from the other methods (KSS).
  The threshold sticks to the majority side, but AUROC is threshold-independent, and a constant predictor
  can only reach 0.5, so the ranking performance is genuine (verification_log 2026-07-04 section).
- **SvmW**: Within 8/8 completed · Mixed 10/16 (remaining 6 running) · Cross 2/8. Within 0.76–0.81 is real discrimination
  (predicting both classes), Cross ~0.51 collapse. Same type as RF: "within effective × cross collapse". Mixed at
  n=5 gives in 0.738 / out 0.766 (within band), directionally consistent (from a tentative n=1 of 0.74–0.75 toward confirmed).
- **SvmA**: **completed** 8/8 seeds. **All conditions in the chance band (0.50–0.60)**. Consistent with the T1 probe (no signal
  in the features via either univariate or RF) — the 18 steering-statistics features carry no
  drowsiness signal in this dataset (true no-signal, not degeneracy, verification_log T1 section). All 48 cells clean
  (no NaN, no out-of-range AUROC, no degeneracy).

Pooled no-treatment locally reproduces the IV2025 published values (RF dominant · SvmW 51% · Lstm 0.52 · SvmA 0.53).
Prediction forms: RF=real discrimination (both classes predicted), SvmW=all-positive degeneracy, Lstm=majority collapse, SvmA=chance.

### Per-method summary (across the 4 condition families)

| Method | Pooled(none/with) | Within | Mixed | Cross | One-liner |
|---|---|---|---|---|---|
| **RF** | 0.738 / 0.753 | 0.75–0.78 | 0.72–0.75 | **0.51 collapse** | pooled↔mixed **immovable ~0.74**. Robust to imbalance, fragile only to domain transfer |
| SvmW | 0.519 / **0.717** | 0.76–0.80 | 0.74–0.75 | 0.51 collapse | pooled recovers with SMOTE and closes on RF (RF +0.039), cross transfer impossible |
| Lstm | 0.512 / 0.508 | 0.76–0.78 | 0.78 | **0.73–0.75** | pooled **collapse**, ~0.75 under domain-restricted evaluation (including cross). See §3 |
| SvmA | 0.481 / 0.520 | chance | chance | chance | chance in all conditions. No signal in the features (T1), unrecoverable |

**Central finding**: RF is immovable at nearly the same value (~0.74) under pooled and mixed. The other methods **collapse to chance under pooled,
and recover to ~0.75 only when the evaluation becomes domain-restricted (within/mixed/cross)**. → §3 analyses the mechanism.

## 3. The Pooled–Mixed asymmetry — "Why is RF dominant under pooled and merely on par under mixed"

**Question**: Pooled(+SMOTE) and Mixed(+SMOTE) are similar in that both "train on all subjects", yet under pooled
RF runs away alone while under mixed Lstm draws level with RF. Can this be explained?

**Conclusion**: What changes is not RF but **the other methods**. RF is immovable at ~0.74 in both conditions. It looks like "RF dominance"
under Pooled because the other methods collapse to 0.51, and it looks "on par" under Mixed because the other methods (especially Lstm)
recover to 0.78. **The two conditions are effectively different evaluation protocols, and that difference helps only the non-RF methods.**

### Facts confirmed by verification (measured from train/eval logs)

1. **RF is flat**: Pooled 0.738/0.753 ≈ Mixed 0.719–0.749. Lstm goes Pooled 0.508 → Mixed 0.780.
2. **The training label distribution is identical for both Pooled and Mixed (~73% positive, balanced)** → the theory that "pooled collapses because
   the training data is label-imbalanced" is **rejected** (both use subject_time_split, train/val/test all ~0.73).
3. **The evaluation protocol differs in 2 points**:
   - **Evaluation scope**: Pooled = evaluated on all 87 subjects (a heterogeneous mixed pool) via a random split.
     Mixed = evaluated on the target-domain 44/43 subjects (a homogeneous subset) via a within-subject time-series split.
   - **train/eval window overlap**: c1 (Within/Mixed/Cross) selects overlapping windows due to the mismatch of train `time_stratify=True`
     and eval `time_stratify=False` (a known characteristic shared with TIV2026; the de-leaked version measures
     within-RF at 0.78→0.526, verification_log 2026-07-01 section). Pooled's eval is a random split.
4. **The decisive fingerprint**: Lstm switches on "whether the evaluation is domain-restricted or not" —
   full-pool evaluation (Pooled) → **0.51**, domain-restricted evaluation (Within 0.77 / Mixed 0.78 / **Cross 0.73–0.75**)
   → **~0.75**. Regardless of the training domain (even cross), it recovers as long as the evaluation is domain-restricted.

### Interpretation of the mechanism

- **RF is insensitive to both protocol differences**: it outputs, as-is, the raw signal ~0.74 learned from all 87 subjects, via a
  regularised ensemble + class_weight + calibration. Unaffected by window overlap or evaluation heterogeneity → Mixed 0.734 ≈
  Pooled 0.738 are both genuine.
- **Lstm is sensitive to the evaluation protocol**: due to its high capacity it can exploit (a) memorisation of overlapping windows and (b) the
  evaluation's domain homogeneity, and is lifted to Mixed 0.78. On the honest evaluation of the fully heterogeneous pool (Pooled) it cannot
  separate event_label and collapses to 0.51. **The Within→Cross invariance on the same evaluation set (RF −0.246 vs Lstm −0.046)
  demonstrates this** (§3.5). ※ SvmW's pooled collapse is a **different mechanism** (not high-capacity overlapping-window exploitation, but
  kernel+Platt degeneracy under imbalance. The old wording "high-capacity kernel SVM is the same type" is wrong, corrected in §3.5).

### Implications for the paper (recommended form of the claim)

**"Pooled is the strictest and most method-agnostically fair evaluation, and RF is the only method that works there. The other methods'
'recovery' under Mixed/Within is a condition-dependent lift supported by the evaluation being domain-restricted and by the
time-series overlap characteristic shared with TIV2026, and it does not relativise RF's robustness"**. RF's dominance is most defensible
if written with **the Pooled column as the primary evidence**. RF should be claimed not as "highest accuracy in all conditions" but as a
"**unique advantage in robustness under the real-operation condition (pooled)**" (in Within it is statistically hard to distinguish from
SvmW/Lstm, and in Cross RF collapses).

### Limitations (honest statement)

- Pooled's random eval can also overlap the train by roughly ~60% (a random test falling in the time-series train region). Therefore
  "Pooled = completely overlap-free" is not the case. **The main reason Mixed is easier than Pooled may be more the contribution of
  "evaluation domain homogeneity" than "overlap amount"**, and a strict separation of the two contributions has not been done.
  However, the conclusion "both differences help only the non-RF methods and do nothing for RF" is unchanged.
- We plan a final confirmation, via SvmW Mixed (running), of whether the same structure holds for KSS-based methods too. Currently SvmA is on the same trend
  (Pooled 0.52 → Mixed 0.59), directionally consistent.

## 3.5 The mechanism of each method's divergence from RF (prediction-level analysis, 2026-07-12)

**The 3 methods diverge from RF "in different conditions, for different reasons"** (confirmed by tracing back to
confusion matrices, probability distributions, and the T1 probe. Each method basically diverges from RF in only 1 condition, only SvmA in all conditions). RF itself also falls to
chance under Cross (transfer failure of different-domain training), so RF is no exception either.

| Method | Condition where it diverges from RF | The nature of the divergence | Collapse form |
|---|---|---|---|
| **Lstm** | **Pooled only** (Within/Mixed/Cross are on par or better) | **Evaluation-protocol dependent** (invariant to what was learned) | Ranking collapse (AUROC 0.78→0.51, CM always majority all-positive) |
| **SvmW** | **Pooled-without-SMOTE only** (others on par, Cross both collapse) | **Imbalance robustness** (kernel+Platt degeneracy) | All-positive degeneracy → released by SMOTE |
| **SvmA** | **All 8 conditions** (0.576 even in Within) | **No signal in the features** (method-independent) | True random (ROC diagonal, non-degenerate) |

### Lstm — mirror-image collapse (RF=training failure Cross / Lstm=evaluation failure Pooled)
- **Decisive evidence**: Within/Mixed/Cross use the **same evaluation set** (only the training set changes). At the same seed, going
  Within→Cross gives **RF −0.246** (0.773→0.527, removing the target domain from training gives base-rate chance
  = genuine training signal) versus **Lstm −0.046** (0.789→0.743, almost unchanged = **does not depend on what was learned**).
- Therefore Lstm's domain-restricted score is not "learned signal" but a **boost from the evaluation protocol (within-subject time-series split
  × homogeneous domain)**. Corroboration: genuine generalisation should rise under a random split (pooled), but instead it **drops** 0.78→0.51
  (PR-lift +0.18→0.00). Because event_label (74% positive), it **predicts majority all-positive in all conditions**, so the
  pooled collapse is a "ranking collapse", not a new degeneracy (even the 0.78 cell has an all-positive CM). SMOTE is ineffective
  (0.512→0.508, 3/6 seeds fully collapse to all-positive).

### SvmW — the only divergence is the pooled SMOTE dependence + **a mechanism correction**
- Within/Mixed/Pooled+SMOTE agree with RF, Cross both collapse → **zero difference in domain capability**. It diverges only in **Pooled-
  without-SMOTE** (RF 0.738 vs SvmW 0.519, Δ+0.219).
- **Correction**: the old explanation "SvmW degenerates because it has no internal re-weighting" is **wrong at the code level**. In fact it is
  **doubly balanced** with `SVC(kernel="rbf", class_weight="balanced")` + balanced `sample_weight` ([classifiers.py:86-91](../../../../src/models/training/classifiers.py#L86)).
  The true mechanism is **kernel-margin + Platt degeneracy**: on the 87-subject heterogeneous pool with 3.9% positive and 8 wavelet features,
  the RBF decision function is nearly constant → Platt maps it to a nearly constant probability (proba pstd **0.001**) → all-positive at threshold.
  **SW-SMOTE fixes it not by "weight" but by "density (geometry)"** (synthetically densifies the minority → RBF support vectors wrap the minority
  cluster → proba spread revives, pstd **0.229**). RF intrinsically preserves ranking via leaf purity of the trees, so it does not
  collapse even under pooled (0.738→0.748 with/without SMOTE). ※ [domain_imbalance_factor_analysis.md](domain_imbalance_factor_analysis.md) §2.2 already documents this correct mechanism.
- **Parity vs RF (n=5 confirmed, 07-16)**: SvmW 0.717 vs RF 0.748 (marginal). **Seed-paired (the same 5 seeds
  {0,1,7,42,123}) gives RF 0.756 vs SvmW 0.717 = RF leads by +0.039** (RF is higher in 3/5 seeds, notably s0 +0.13,
  s42 +0.07; SvmW is marginally higher for s1/s123). Conclusion: SvmW **closes on RF under pooled+SMOTE, but RF is still slightly
  ahead**. RF's unique advantage lies in that it **works even without SMOTE** (0.738 vs SvmW 0.519 degenerate) = its independence from imbalance treatment.

### SvmA — the only "unconditional" divergence (feature-bound, negative control)
- Chance in all 8 conditions (0.48–0.60). In Within-in, SvmW(0.803)/Lstm(0.779)/RF(0.752) all align, yet SvmA is
  only 0.576. **The only one that catches up to RF in no condition**.
- **Root cause = no feature signal (confirmed by the T1 probe)**: **feeding the same RF the SvmA 18 steering statistics features gives
  0.4955** (0.75 with its own top-10). A 2×2 dissociation shows **features are decisive, method is irrelevant**. SvmA's CM is **on the ROC diagonal**
  (TPR≈FPR, precision≈prevalence) = **true random, not degeneracy** (qualitatively different from the single-class degeneracy of SvmW/Lstm).
  SMOTE is ineffective (0.481→0.533, chance band) because the features carry no threshold-independent ranking signal.
- **Recommendation**: SvmA should be presented not as a "weak competitor" but as a **negative control / lower bound of feature representation**
  (a faithful Arefnezhad2019 reproduction being chance in all conditions = verification that the pipeline is not doing a trivial leak + attributing the
  other methods' performance to the **richness of the feature representation**).

### Confirmed / speculative
- **Confirmed** (from prediction data): the above divergence conditions, mechanisms, the 2×2 dissociation, the same-eval-set invariance, and the distinction of collapse forms.
- **Speculative / unseparated**: (a) the contribution split of "heterogeneity vs split method" in Lstm's pooled collapse is not quantified, (b) the parity of SvmW pooled+
  SMOTE vs RF is RF-lead at n=2 · seed-matched, (c) Within/Mixed are comparisons on a protocol with a leakage tendency shared by all methods,
  (d) the raw `y_pred_proba` arrays were not saved, so we rely on CM-derived quantities.

## 3.6 Robustness appendix: de-leak sensitivity analysis (2026-07-19–20, 4-agent adversarial audit + real-model de-leak)

> **Positioning (2026-07-20 policy)**: This section is a **de-leak / robustness appendix / sensitivity analysis** against the main results of §2 (same protocol as IV2025/TIV2026).
> exp3 is completed/reported under the same framework, and this section quantifies "the impact of the train/eval split
> mismatch that the same framework carries" (a deep-dive into the known limitation, verification_log 2026-07-01). **Key point**: under an honest split
> (train/eval consistent, subject holdout), vehicle→EEG(KSS) can drop to chance ~0.52 for all methods and all feature counts.
> ※ Only this section's original derivative claim "RF with all features = feature selection underrates RF" is withdrawn as a factual error (derived from a memorisation leak).
> The formal de-leak re-test is prepared in **§3.7**.

### The mechanism confirmed by the audit (code + real model + real logs, passed adversarial refutation)
- **pooled**: train=within-subject time-series (`data_time_split_by_subject` first **60%**, `TRAIN_RATIO=0.6`) / eval=
  **random 20%** (`eval_pipeline.py:152`, does not pass `--subject_wise_split` to eval) → with independent splits,
  **~60% of the eval test rows are contained in the training set**.
- **within(target_only)/mixed**: train=`time_stratified_three_way_split`(70% of the whole) / eval=
  `data_time_split_by_subject`(per-subject last 20%), a **function mismatch** → overlap **69% / 61–78%**.
- **honest (0% overlap) is only Cross(source_only, subjects disjoint) and domain_train(same split)**.
  The matched-split counterfactual is 0% overlap = proof that the mechanism derives from the split mismatch.

### Decomposition on the real model (reproducing the recorded JSON to 3 decimals)
| Model | recorded(leaked) | SEEN(trained) | UNSEEN(honest) | subject holdout | cross-subj |
|---|---|---|---|---|---|
| RF top-10 | 0.765 | 0.848 | 0.650 | **0.517** | 0.495 |
| RF top-10+SMOTE | 0.758 | 0.830 | 0.658 | **0.525** | — |
| RF nofs(160) | 0.870 | **0.974** | 0.714 | **0.534** | 0.514 |

**When correctly separated, all are chance (0.49–0.53), and the feature count is irrelevant**. nofs looked best because its larger memorisation capacity
memorises the leaked rows more completely. Positive control: EEG band power is 0.62 under the same honest split (> vehicle 0.53) =
the harness passes genuine signal (the vehicle chance is not a verification artifact).

### The honest conclusion (direction of the paper)
1. **Vehicle dynamics → EEG drowsiness(KSS) is chance for all methods under honest evaluation (~0.50–0.53)** (RF with/without selection, SvmW, SvmA).
   All the pooled/within/mixed numbers in §2 are inflated by the split mismatch. Consistent with [[project_exp2_rf_087_unreproducible]].
2. **The only real signal is Lstm (event_label=DRT event, a different task)**. Under honest cross-subject, Within/Cross
   ≈ 0.72–0.75 survives domain-invariantly (DRT detection, not EEG drowsiness). pooled Lstm 0.51 is the no-treatment setting.
3. **The high AUROCs of prior work (IV2025/TIV2026/exp2 0.89) are possibly artifacts of the same train/eval split mismatch**
   (verification_log 2026-07-01 already recorded this for within/mixed, a mechanism deliberately retained for comparability).
4. **To confirm**: the positive control of the shipped KSS labels stops at 0.56–0.62 (`kss.py` consistency issue, spearman 0.157).
   The vehicle=chance conclusion is unchanged, but a strong contrast for "the harness detects signal" requires confirming label consistency.

### [Direct measurement] within-domain de-leak re-evaluation — the paper's showpiece "SvmW recovery" is also leak (2026-07-19, task2)
Taking the saved c1 within(target_only,in_domain) models, (A) reproduce with leaked eval → matches the recorded values, confirming faithfulness,
(B) measure honestly on the **held-out test (disjoint) of the same `time_stratified_three_way_split` as the model's own train**:

| Method | LEAKED reproduction (=recorded value) | **HONEST-matched (de-leaked)** |
|---|---|---|
| RF within-in | 0.77 (recorded 0.746) | **0.47 (chance)** |
| SvmW within-in | 0.77 (recorded 0.805) | **0.52 (chance)** |
| SvmA within-in | — (cannot load, no cuml env) | Cross 0.51 + T1 no-signal confirms chance |

→ **The central finding of the paper (TIV2026_exp3) "SW-SMOTE recovers SvmW in within from 0.52→0.80 (signal present) / SvmA is
unchanged (no signal)" is a leak artifact**. Honestly, **both SvmW and RF are within=chance**, and both the "recovery" and
the "SvmW vs SvmA dissociation" vanish (both chance). The only honest signal is Lstm (a different label = DRT event, Cross 0.73–0.75).
(Note: honest cross-subject leaks even if we hold out, because the saved model has already trained on all 44 subjects, so subject separation
**requires retraining** = the audit's GroupShuffleSplit 0.51 / Cross column 0.51 are the correct honest cross-subject values.)

### [Honest all-conditions · confirmed table] de-leaked re-evaluation (2026-07-20, task2 complete)
Clean honest values after bug removal (see note below). Position-indexed, standalone retraining removes index contamination, and matches audit agent3:

| Evaluation protocol | leaked(recorded) | **HONEST(clean)** | Verdict |
|---|---|---|---|
| within(target_only) RF/SvmW | 0.77 / 0.80 | **0.47–0.52** | chance |
| pooled RF/SvmW | 0.76 / 0.71 | **0.51 / ~0.51** | chance |
| within-subject-time-series (personalized, same driver latter half+gap) 160/10 features | — | **0.496 / 0.515** | chance |
| cross-subject (new driver) 160/10 features | — | **0.518 / 0.507** | chance |
| (honest positive control) EEG band power cross-subject | — | **0.62** | signal present = harness valid |

→ **Vehicle dynamics→EEG drowsiness(KSS) is chance (~0.50–0.52) under every honest split (within-subject-time-series · between-subject) and feature count
(10/160)**. Even personalized (same driver) has no signal. The high leaked values (0.72–0.86) are all train/eval row overlap.
The only real signal is Lstm (a different label = DRT event, Cross honest 0.73–0.75).

> **⚠️ Methodological note (a pitfall we hit this time, worth recording)**: `data_time_split_by_subject` **resets the index and
> returns it**. Using the returned `X_test.index` with `df.loc[]` on the original df **erroneously fetches leading (training-region) rows**, so a supposed honest test
> evaluates training rows and inflates to 0.75. Honest re-evaluation must use **position (iloc/numpy) indexing**, or have the split function retain the
> original index. The audit's 0.52 is correct; the intermediate 0.75 is a product of this bug (not adopted).

### De-leak policy (next action, needs user judgement)
Use **the same split** for train and eval (make `evaluate.py`'s eval split coincide with the train's held-out / reuse the saved split).
Re-measure all cells (4 methods × all conditions) honestly and rebuild §2. **Paper implication**: reconstruct RQ1/the central claim into "under leakage-free
evaluation, vehicle dynamics does not predict EEG drowsiness above chance (negative result), and the prior high values / 'recovery' / 'dissociation' are
artifacts of the train/eval split mismatch. The only real signal is DRT-event detection (Lstm, a different label)".
The record of the old, withdrawn claims follows below.

---
(Below, the original withdrawn memo discovered on 07-19 — detailed and expanded by the audit above)

**The mechanism of the leak (confirmed by measurement)**: pooled is **train=within-subject time-series split (first 70% of each subject) / eval=random
15%** ([eval_pipeline.py:152](../../../../src/evaluation/eval_pipeline.py#L152), `iv2025_baseline_launcher` does not pass
`--subject_wise_split` to eval). Because the two splits are independent, **69% of the eval test rows are contained in the train
training set**. Reproduction with standard RF (common data · KSS · same 160 features):

| Evaluation | 160 features | top-10 |
|---|---|---|
| pipeline reproduction (train=time-series / eval=random) **overall** | **0.925** | 0.907 |
| ┗ **SEEN rows (trained)** | **1.000 (complete memorisation)** | 1.000 |
| ┗ **UNSEEN rows (untrained=honest)** | **0.505 (chance)** | 0.448 |
| **correctly separated** (single random split, train∩test=∅) | **0.524** | 0.507 |
| **subject holdout (cross-subject)** | **0.529** | 0.506 |

**Conclusions**:
1. **When train/test are correctly separated, RF is chance (~0.52) for both all features and top-10** = agrees with the campaign's central finding
   "vehicle dynamics→EEG drowsiness ≈ chance" ([[project_exp2_rf_087_unreproducible]]).
2. **160 features coming out higher than top-10 is because the larger memorisation capacity memorises the leaked rows more completely**, not a difference in real signal
   (both are chance in UNSEEN). "Feature selection underrates RF" does not hold.
3. **This split-mismatch leak is not specific to nofs but extends to the entire pooled column**. The iv25base/iv25smote pooled values
   (RF 0.738/0.748, SvmW 0.519/0.717, etc.) are also likely inflated by the same mechanism, so **the §2 pooled table and
   the §3 central claim "RF works under pooled" require re-verification** (all methods expected to be chance under an honest split). within/mixed
   also have the time-series overlap leak described in §3/§3.5, hence the same.
4. **Lesson**: treating pooled as an "honest column" was the mistake. Unless the train and eval split methods coincide, absolute AUROC is
   leakage-contaminated. It can be used for the **relative** comparison (before/after) against IV2025/TIV2026, but **absolute values must not be interpreted
   as "real signal"**. An honest absolute evaluation requires the same subject-holdout (cross-subject) split for train/eval.

## 3.7 The "standby" for the implementation review / de-leak re-test (2026-07-20, prepared to execute after completion)

### ★ Prior demonstration (Method 1) — complete, double-verified by 2 independent agents (2026-07-20, exp3 keeps running, non-destructive)
Take the saved models and (a) reproduce with leaked eval → **matches the recorded JSON to 4 decimals (faithfulness gate passed)**, (b) re-score on an honest split
(physical row = subject_id+Timestamp, **confirmed 0% overlap with training rows**):

| model | cell | recorded(=leaked) | **honest(disjoint)** | training row overlap |
|---|---|---|---|---|
| RF | within-in | 0.773–0.800 | **0.472–0.475** | leaked 69% → honest **0%** |
| RF | **pooled** | 0.721–0.765 | **0.517–0.525** | leaked **59.6%** → honest 0% |
| SvmW | within-in | 0.683–0.795 | **0.516–0.526** | leaked 69% → honest 0% |
| SvmW | pooled+SMOTE | 0.689–0.726 | **0.489–0.492** | leaked 59.6% → honest 0% |
| (positive control) EEG band power | same honest split | — | **0.56–0.58 (>vehicle 0.52)** | — |

**Confirmed**: the high leaked values are memorisation from the 59–69% training-row reuse, and **making the physical rows 0% overlap makes everything chance (0.47–0.53)**.
Faithfulness exactly matches + the positive control passes = it is not merely "the split broke the signal". **"RF is the only method that works under pooled"
does not hold under honest evaluation** (0.76→0.517=chance). Scripts: scratchpad `deleak_demo_A.py` / `adv_verify.py`.

**Refinements (found during verification, worth recording)**:
- **Recorded values depend on the code version**: some RF within JSON (generated 2026-06-22, `split_data_domain_train`=subject holdout
  commit `466e5b1`) had train/eval consistent at the time and were **already honest (0% overlap, ~0.52)**. Regenerating with the current code (from 06-27, within-subject
  commit `3e67282`) raises the overlap to 68.6% · ~0.58. → **The §2 numbers can be a mix of leaked/honest depending on the "code version at generation time"**.
  In the de-leak re-evaluation, all cells need to be **re-generated in bulk** under the current honest split for uniformity.
- **SvmW pooled (no-treatment iv25base) degenerates without SMOTE (all-positive, 0.52) = chance to begin with**. What is inflated by memorisation is
  the SMOTE version (0.72). The SvmW baseline is chance in both leaked/honest.

### Plan (main execution after completion)
Policy: **completing exp3 with the current protocol is the top priority** (§2 main results). The de-leak re-test is kept **non-destructive to running experiments and shared
code**, and executed after completion. Below are the preparations and procedure (this section is a checklist, execution is on hold).

**A. Implementation review (read-only, no code changes)**
1. Tabulate the split mismatch between `eval_pipeline.py:151-165` (pooled→random) / `138-149` (within/mixed→per-subject last 20%) and the train side
   (`model_pipeline.py:108-191`), by mode, as the "row overlap % of train partition vs eval-test partition"
   (formalise the audit `overlap_audit.py` into an official script: pooled ~60% · within ~69% · mixed 61–78% ·
   Cross/domain_train 0%). → Confirm numerically "which column leaks by how much".
2. Separately confirm the positive control (EEG→KSS) and label consistency (`kss.py`, the spearman 0.157 concern) to guarantee "the harness passes signal"
   (currently cross-subject 0.62).

**B. De-leak re-test (execute after completion, existing artifacts unchanged)**
- Method 1 (minimal, safe): a standalone that **re-scores the saved models on the honest split's held-out** (formalise `honest_*.py`). ⚠️ `data_time_split_by_subject` resets the
  index → **position (iloc) indexing is mandatory** (the pitfall in §3.6).
  Generate the honest table for 4 methods × all conditions. SvmA needs the cuml env (substitutable with Cross+T1).
- Method 2 (full): add a **`--honest_split` flag (use the held-out of the same split as train for eval) to `evaluate.py`**, and
  **run a separate eval under a new tag `honest_`** (existing leaked JSON preserved, listed alongside for comparison). The shared code is additive only, with
  default behaviour unchanged, so it does not interfere with running experiments.
- Expected result (from the audit): vehicle→EEG(KSS) is chance ~0.52 for all methods under honest, only Lstm(DRT) has Cross honest 0.73–0.75.

**C. Reflection in the paper (draft TIV2026_exp3, judgement on hold)**: present the main results per IV2025/TIV2026, and **note the de-leak sensitivity analysis
(§3.6/§3.7-B) in a Limitation/Robustness section** — this composition is the leading candidate that reconciles comparability and honesty.

**Next action (needs user judgement)**: execute B after waiting for exp3 completion (c1 SvmW remaining · RF-nofs remaining). If in a hurry, B Method 1
can be demonstrated on 1–2 cells right now (non-destructive).

## 4. Verification status (details: verification_log.md)

- **2026-07-04 adversarial re-verification (11 agents)**: independently re-scanned all completed cells
  + byte-compared under a skeptical pass. **Found and fixed 1 real bug** — the 2 c1 RF Cross cells
  (in/s42, out/s123) byte-matched the within prediction vectors (recurrence of the known Bug#4 =
  the eval model resolution race via `latest_job.txt`).
  - Fix: fixed `--jobid` in the eval of `c1_domain_launcher.py` (commit `e0923fa`)
    → race-free regardless of the number of workers. The 2 contaminated cells were deleted and re-run, and recovery to the honest values
    (0.527 / 0.508) and proba independence were confirmed.
  - Impact: RF Cross mean 0.529→0.519 (in), 0.517→0.507 (out). **Conclusion unchanged**.
  - An exhaustive comparison over all 4 models × (domain,seed) × mode found no other collisions (clean).
- Everything else is clean across all targets: no missing/stale cells, no NaN or out-of-range AUROC,
  no degeneracy in discrimination conditions, CM totals consistent with the split sizes, all cells generated after the revert (902ce96).
- **2026-07-11 re-verification** (full scan of the 356 completed cells): zero NaN, zero out-of-range AUROC, no unexpected degeneracy.
  Degeneracy (single-class prediction) is only the 6 iv25base SvmW pooled cells = the **expected** reproduction of the IV2025 "SvmW all-positive degeneracy".
  The 2 duplicate RF Cross files (in/s42, out/s123; leftovers of an old re-run, both honest ~0.51 · AUROC diff <0.001) were
  moved to `results/_archived_duplicates_20260711/` (with MANIFEST, reversible) → RF c1 is exactly 144 (24/cell).
  The `TRAIN FAILED` groups in the SvmW logs are all from restart churn due to `STATUS_CONTROL_C_EXIT`/`DLL_INIT_FAILED`/forced-terminate
  (not a code bug, resume-safe) → permanently addressed by the §6 watchdog fix.
- Standing operation: at every status check we run (a) a scan of the logs for TRAIN FAILED/Traceback,
  (b) an AUROC-range / degeneracy check, (c) a cross-mode collision scan.

## 5. Execution infrastructure

| Item | Content |
|---|---|
| Launcher | [`c1_domain_launcher.py`](../../../../scripts/python/train/c1_domain_launcher.py) (c1) / [`iv2025_baseline_launcher.py`](../../../../scripts/python/train/iv2025_baseline_launcher.py) (both Pooled arms, with-treatment via `--smote`) |
| Execution order | **table-priority** (2026-07-04, commit `696bb60`): Within+Mixed all seeds first → Cross last. Only the order changes; the cell set, tags, and methods are unchanged |
| Placement | RF/SvmW=Windows CPU (**SvmW 8 workers** scaled up, 20 cores, single-thread pinned), Lstm/SvmA=WSL2 GPU (TF2.21 CUDA / cuML) |
| Monitoring | `c1_watchdog.sh` (schtasks every 10 min, liveness monitoring + auto-restart + self-delete on full completion) + ntfy.sh notification |
| Tags | c1: `imbalv3_knn_wasserstein_<dom>_<mode>_split2_subjectwise_ratio0.5_s<seed>` / Pooled: `iv25base_<M>_pooled_baseline_s<seed>` (no treatment) · `iv25smote_<M>_pooled_swsmote_s<seed>` (with treatment) |
| eval model resolution | fixed `--jobid` references the cell's own trained model (eradicates the Bug#4 race, §4) |

## 6. Progress and completion forecast (as of 2026-07-11 13:40)

| Case | Complete | Forecast |
|---|---|---|
| c1 RF / c1 Lstm / c1 SvmA | **144/144 ✅ / 90/90 ✅ / 48/48 ✅** | done (SvmA completed 07-10 18:16) |
| Pooled no treatment (RF/SvmW/Lstm/SvmA) | **15 / 6 / 6 / 6 ✅** | done (IV2025 reproduction concluded) |
| Pooled + SW-SMOTE (RF/SvmA/Lstm/SvmW) | **15 ✅ / 6 ✅ / 6 ✅** / SvmW 0/6 | only SvmW remaining (currently s0/s1/s7 running) → ~07-12–13 |
| c1 SvmW (Within+Mixed only, Cross abolished) | **~12/32** (Within 4+5 · Mixed 1+1) | **Mixed ~4 days/cell is the bottleneck** → ~07-16–20 |
| **RF no-feature-selection version (new, seed 24, 160 features)** | 0/120 (Pooled+SMOTE 24 + Within/Mixed×in/out 24 each) | RF is also ~1h/cell with 50-trial Optuna → **~1–2 days** (BelowNormal, running alongside SvmW on free cores, SvmW timeline unchanged) |

**Measured SvmW time per cell**: Within ~15–21h · Cross ~14–17h · **Mixed ~100h (≈4 days, the final bottleneck)** ·
iv25smote pooled ~17h (50 Optuna trials × ~20min). SvmW's heaviness is the intrinsic cost of the somewhat non-converging RBF-SVM + Optuna
50 trials (N_TRIALS=50 unchanged to maintain fidelity). Mixed is heaviest because its training set is the largest
(all 87 subjects + SMOTE).

- **2026-07-11 watchdog churn fix**: added a guard (`pool_healthy`) to `c1_watchdog.sh`. Previously, every time the launcher
  died on console-close (`STATUS_CONTROL_C_EXIT`), the watchdog would reap the orphaned workers → restart, redoing the multi-day Mixed cells
  from Optuna trial 0 (4 churns on the night of 07-10). After the fix, "if non-pooled workers exist and have progressed within 6h, adopt the orphans
  and skip the restart". The current run is protected without stalling. **iv25smote SvmW (0/6, unfinished for 7 days) is outside watchdog management and still fragile** —
  it keeps being killed before the final evaluation by console-close (policy: let it complete without interference).
- **Cross abolished (2026-07-11, user judgement)**: since cross-domain transfer collapses to chance (~0.51) for all methods and has
  low informational value, **Cross (source_only) is removed from all of exp3** (`c1_domain_launcher.py`'s
  `build_cells` generates only Within+Mixed). The running SvmW cross (4 cells remaining) is also aborted. The existing
  Cross eval JSON (RF24/Lstm15/SvmA8/SvmW4) is **not deleted but demoted to "reference"**. → This replaces the previous "halve Cross seeds"
  proposal. SvmW remaining is within+mixed only (total 48→32).
- **RF no-feature-selection version added (2026-07-11, user judgement)**: to see the contribution of RF's top-10 importance selection,
  **add an RF version using all 165 features (excluding EEG/meta)**. Implementation: `train.py --feature_selection
  none` (new flag, default `rf`), issued from `c1_domain_launcher.py --no-fs` / `iv2025_baseline_launcher.py
  --smote --no-fs`. Tags get a `_nofs` suffix to separate them from the top-10 version. The target is **the with-treatment columns only**
  (Pooled+SMOTE · Mixed in/out · Within in/out, no Cross). The eval side reads the saved
  `selected_features`, so no change is needed (method · SW-SMOTE · split identical to the top-10 version, only the selection differs).
  **Seeds later reduced to 5 (user judgement)**. **⚠️ The interpretation of the result is withdrawn (2026-07-19)**: pooled 5/5=0.864 is
  not real signal but **a memorisation leak of the train/eval split mismatch** (chance ~0.52 when correctly separated, §3.6). The substantive outcome is that this verification
  revealed **the leak across the entire pooled column**.
- At 04:30 (07-05) the old 4-parallel launcher was auto-switched to the table-priority + 8-parallel version (`_svmw_table_priority_switchover.sh`).
  The Bug#4 contamination check at switchover was also run automatically, confirming no contamination.

## 7. Remaining tasks

- [ ] Completion of c1 SvmW / SvmA · Pooled+SW-SMOTE (SvmW/RF remaining) (watchdog auto-operation)
- [ ] After completion: final aggregation table + figures for 4 methods × {Pooled none/with, Within in/out, Mixed in/out, Cross in/out}
- [ ] Significance tests of Pooled RF vs each method + equivalence notation for Within's RF≈SvmW/Lstm (seed-paired)
- [ ] Final run of `exp3_seed_adequacy.py` (confirm/record the ADEQUATE verdict for all conditions)
- [ ] Adversarial re-verification on the completed data (final version)
- [ ] Reflection into the TIV2026_exp3 manuscript ([TIV2026_exp3/outline.md](TIV2026_exp3/outline.md))

## 8. References

- Detailed decision/verification log: [verification_log.md](verification_log.md)
- Factor analysis (the mechanism of RF robustness, the Sobol contribution of distance): [domain_imbalance_factor_analysis.md](domain_imbalance_factor_analysis.md)
- Distance/ratio sensitivity (exp2): `../exp2-analysis/distance_granular_report.md`, `../exp2-analysis/ratio_sensitivity_report.md`, `results/analysis/exp2_domain_shift/figures/csv/split2/sensitivity/sobol_indices.csv`
- Seed adequacy: [`exp3_seed_adequacy.py`](../../../../scripts/python/analysis/exp3_seed_adequacy.py) → `results/analysis/exp3_verification/seed_adequacy.json`
- Feature-signal probe (basis for SvmA no-signal): `results/analysis/exp3_verification/t1_feature_signal_probe.json`
- Raw data: `results/outputs/evaluation/<Model>/**/eval_results_*imbalv3_knn_wasserstein*ratio0.5*.json` (c1) / `*iv25base*.json` (Pooled no treatment) / `*iv25smote*.json` (Pooled with treatment)
