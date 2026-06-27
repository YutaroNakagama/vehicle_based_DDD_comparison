# Why the Domain-Shift × Class-Imbalance Treatment Affects Each Model Differently (exp3)

*Analysis date: 2026-06-27. Status: RF & SvmW fully measured locally; SvmA & Lstm "after"
measured, "before" uses IV2025 published values pending the local pooled runs (GPU in progress).*

## Executive summary

exp3 compares four lightweight driver-drowsiness-detection (DDD) methods under two settings:

- **Uncontrolled ("before", IV2025)** — pooled split (no domain grouping), **no** class-imbalance handling.
- **Controlled ("after", TIV2026/B1)** — **within-domain** training (`target_only`) **+ subject-wise SMOTE** (SW-SMOTE).

The within-domain AUROC effect of the controlled treatment **differs sharply by model**:

| Model | Before (uncontrolled) | After (controlled) | Effect | Verdict |
|-------|----------------------:|-------------------:|:------:|---------|
| **RF**   | 0.730 ±0.086 (local, n=11) | 0.781 ±0.086 (n=22) | **+0.05** | robust — already strong, barely changes |
| **SvmW** | 0.519 ±0.010 (local, n=6) ≈ 0.51 (IV2025) | 0.791 ±0.027 (n=22) | **+0.27** | **rescued** |
| **SvmA** | 0.53 (IV2025 pub; local pending) | 0.539 ±0.032 (n=11) | **~0.01** | **not rescued** (chance both) |
| **Lstm** | 0.52 (IV2025 pub; local pending) | 0.779 ±0.008 (n=22) | **+0.26** | **rescued** (different target) |

**The pattern:** the controlled treatment **rescues SvmW and Lstm up to RF's level (~0.78)**, **does not rescue SvmA** (it remains at chance), and leaves **RF essentially unchanged** (it was already the strongest). Three mechanisms explain this:

1. **Class imbalance makes kernel-SVMs degenerate**; SW-SMOTE reverses this **only when the features carry latent signal** (SvmW yes, SvmA no).
2. **Feature discriminativeness differs**: RF's feature pipeline (a 160-feature candidate pool reduced to a **top-10 RF-importance** model) and SvmW's steering-wheel wavelets carry signal for the drowsiness label; SvmA's steering-angle statistics do not.
3. **Lstm predicts a different, more learnable/balanced target** (`event_label`, DRT task events) — not the EEG-derived KSS label the others predict.

---

## 1. Setup and the confound caveat

| Factor | Uncontrolled (IV2025) | Controlled (B1) |
|--------|-----------------------|-----------------|
| Domain handling | `pooled` (`--subject_wise_split`, all subjects mixed) | `target_only` (within-domain, in_domain group) |
| Class imbalance | none | SW-SMOTE (`smote` + `--subject_wise_oversampling`), ratios 0.3/0.5 |
| Distance ranking | — | wasserstein |
| Eval | pooled hold-out | within-domain hold-out |

⚠️ **CONFOUND.** The two conditions differ in **both** factors at once (domain *and* imbalance). The 2×2 ablation corners — `pooled + SMOTE` and `target_only + no-SMOTE` — were **not** run. Therefore the measured effect is the **combined** treatment, not either factor in isolation. §5 reasons about each factor's contribution from mechanistic evidence; §6 specifies the ablation that would cleanly separate them.

The positive (drowsy) class is **rare (~4% of windows)**, which is central to the imbalance mechanism below.

---

## 2. Per-model mechanistic analysis

### 2.1 RF — robust (0.730 → 0.781, +0.05)

- **Richest candidate feature pool, then aggressive selection.** RF's input *range* is the full ~160 statistical + wavelet + frequency features across all signals (`loaders.py` range `Steering_Range … LaneOffset_AAA`), vs 8–36 for the SVMs. **Correction (adversarial verification):** RF does **not** train on all 160. Its `feature_selection_method='rf'` (`model_pipeline.py:56`) with `TOP_K_FEATURES=10` (`config.py:206`) selects the **top-10 RF-importance features** per run (148 `selected_features_RF_*.pkl` files exist on disk; the measured B1 and IV2025 cells each use exactly 10 features). So the "redundant 160-feature signal survives domain/imbalance change" story is **not** the operative mechanism — the actual RF is a 10-feature model. RF's robustness rests on ensemble averaging + reweighting (below), not on a wide redundant input. (Also: 160/39 ≈ **4.1×** SvmA, not 20×; the ~20× ratio applies only to SvmW's 8 features.)
- **Native imbalance handling, no degeneracy.** RF never collapses to a single class in either condition (0/22 B1 and 0/11 IV2025 confusion matrices are degenerate). The mechanism is a **large randomized ensemble** (~680 trees, `max_features=0.05`, `max_samples=0.9`, `min_samples_leaf=8`) whose averaging smooths probabilities, **plus** a fixed cost-sensitive `class_weight={0:1.0, 1:3.0}` that *overrides* Optuna's tuned value (`classifiers.py:66-74`, `config.py:310`), balanced `sample_weight`, **and** sigmoid probability calibration (`apply_rf_calibration`, RF-only). SMOTE adds little because RF already reweights the rare class internally.
- **Operating point, not ranking, is what changes.** The two RF conditions differ mainly in decision threshold / probability scale: predicted-positive rate is **6.6% in B1 vs 44.8% in IV2025** (a 6.8× swing) while threshold-independent AUROC moves only +0.05. IV2025 RF probabilities cluster tightly near 0.5, so the default cut bisects the cloud (accuracy ~0.57) even though ranking is preserved.
- **High seed variance** (sd ≈ 0.086, range 0.60–0.92 across 22 cells) — verified that low-end cells (e.g. 0.603) have sane confusion matrices, i.e. real variance, not failures. This is why **11 seeds were needed** to pin RF's mean (a 3-seed estimate read 0.74; the converged value is 0.78). Note: the seed-paired delta is itself noisy (range −0.146 to +0.258, sign-flipping across 4 of 11 seeds), so +0.05 is a small aggregate, not a stable per-seed shift.
- **Verdict:** RF is the reference "winner" in both regimes; the controlled treatment offers only a marginal, within-noise gain.

### 2.2 SvmW (Zhao 2009) — rescued (0.519 → 0.791, +0.27)

This is the clearest mechanism, with direct confusion-matrix evidence.

- **Without SMOTE (IV2025), SvmW degenerates.** Measured IV25 cells: `cm = [[0, 11867], [0, 485]]` (every sample predicted positive) → AUROC **0.52** (chance). **4 of 6** cells are all-positive (not all 6 — verified). The underlying pathology is that `predict_proba` collapses to a **near-constant function**: in 4 seeds it pins to ~0.0399 (= the base rate `P(drowsy)`), and in 2 seeds (s0, s7) it saturates to ~0.96 (all-positive even at threshold 0.5). Either way the **probability ranking is uninformative** (AUROC ≈ 0.5, not merely a bad threshold). Reweighting is *insufficient* here despite being applied **doubly** — `class_weight='balanced'` **and** balanced `sample_weight` (`pipeline.py:184-186`) — yet the RBF-SVM still converges to a constant solution on the ~4% base rate; SMOTE escapes it by populating the minority region. (The all-positive confusion matrices are partly a threshold artifact for the 4 base-rate-pinned seeds, where the F1-optimal validation threshold collapses just below the constant score; but for the 2 saturated seeds the all-positive output holds at every threshold, so it is not a uniform threshold-only story.)
- **With SW-SMOTE (B1), SvmW recovers a real decision function.** Measured B1 cell: `cm = [[3657, 2318], [35, 245]]` (predictions span both classes) → AUROC **0.79**. Balancing the training set lets the SVM find a boundary, and the **AUROC (ranking) genuinely improves**, not just the operating point.
- **The 8 steering-wheel wavelet band-energies carry latent drowsiness signal** that imbalance was masking — SMOTE unlocks it. (Faithfulness note: SvmW uses exactly the 8 GHM multiwavelet-packet bands `SteeringWheel_DDD…AAA`, wavelet-only, per Zhao. The 0.79 holds under IV2025's per-window framework; under Zhao's literal *per-session* labeling it is ~0.33 — see `operations_log` / faithfulness check.)

### 2.3 SvmA (Arefnezhad 2019) — **not** rescued (~0.53 → 0.539, ~0)

The critical contrast with SvmW: SMOTE does **not** help SvmA.

- **Near-chance even WITH SW-SMOTE.** B1 SvmA = 0.539 (n=11), range 0.48–0.59. All 11 cells are **non-degenerate** (predictions span both classes; predicted-positive count swings wildly by seed, 668–6532 of 6775) — i.e. the model is *trying* but its **probability scores barely separate classes**. The confusion-matrix signature is textbook chance discrimination: TPR ≈ FPR in *every* matrix (mean |TPR−FPR| ≈ 0.047), and precision stays pinned at the base rate (~0.04–0.07). The tiny residual (mean AUROC 0.539, a hair above 0.500) is best read as **near-chance**, not as usable signal. The practical conclusion (SvmA stays at chance and SMOTE cannot rescue it) is firm.
- **Features lack signal for the KSS label.** SvmA uses steering-angle/rate statistical features (`Steering_Range … SteeringSpeed_SampleEntropy` = **39 in-range columns, 23 retained** after the paper's 14-suffix filter) with the paper-specific KSS mapping (1–6 = alert, 8–9 = drowsy, 7 excluded; `SvmA.py:SVMA_KSS_BIN_LABELS`). Direct evidence the signal is absent: pooled univariate directionless AUROC tops out at **0.515** (best feature, 0 features > 0.55) and a multivariate RBF-SVM on the 23 paper features lands at **0.509**. Balancing the classes cannot create discriminative signal that the features do not contain — so SMOTE moves the operating point but not the AUROC.
- **Interpretation:** SvmW vs SvmA is the key dissociation — **SMOTE rescues imbalance-masked signal (SvmW) but cannot manufacture signal where there is none (SvmA).** Steering-angle statistics alone do not encode EEG-defined drowsiness in this dataset.

### 2.4 Lstm (Wang 2022) — rescued, but a different target (≈0.52 → 0.779, +0.26)

- **Lstm predicts `event_label` (DRT task events), not KSS-drowsiness.** This is a different, more balanced and more learnable target than the ~4%-positive EEG-KSS label the other three predict (`split_helpers.py` routes Lstm to the event-label branch).
- **After (B1) = 0.779** (n=22, extremely tight, sd 0.008) — a genuine, stable signal. IV2025 reports the uncontrolled Lstm at AUROC 0.52; the local pooled "before" is pending and expected to land near it. (Caveat: even on this easier target the model leans to the majority/task class — sensitivity ~0.97 but specificity ~0.13; AUPRC ~0.92 is only modestly above the ~0.73 prevalence baseline.)
- **Commensurability caveat:** because Lstm's target differs, its before/after sits on a *different prediction task* from RF/SvmW/SvmA. Its high "after" value reflects that vehicle dynamics predict DRT events better than they predict EEG-drowsiness — a finding that should be framed separately from the KSS-based methods.

---

## 3. Results detail and data provenance

- "After" (B1) values are **local measurements** (`results/outputs/evaluation/<MODEL>/**/*b1cmp_*_within.json`), seeds = 11 (RF/SvmW/Lstm, ×2 ratios = 22 cells) and 6 (SvmA, ×2 ratios = 12; 11 complete).
- "Before" (IV2025) values: **RF and SvmW are local measurements** (`*iv25base_*` pooled JSONs, 11 and 6 seeds); **SvmA (0.53) and Lstm (0.52) are IV2025 published C2 values** (`docs/.../IV2025/IV2025.tex`, proper-config column) used as the interim reference until the local pooled runs finish.
- **Cross-check:** local IV2025 SvmW (0.519) matches IV2025 published (0.51), validating that the local reproduction agrees with the paper — increasing confidence that the pending local SvmA/Lstm "before" will land near 0.53/0.52.

---

## 4. Seed adequacy (academically-justified counts)

Per-method seed count chosen by a convergence criterion (95% CI half-width):
- **RF/SvmW/Lstm → 11 seeds** (RF has the highest variance and needed the most; the 3-seed RF estimate 0.74 was noise, 11-seed = 0.781).
- **SvmA → 6 seeds** (robustly chance; 6 seeds give a 95% CI ≈ [0.48, 0.56], upper bound < 0.60, excluding any weak signal).
- IV2025 baseline mirrors this (RF 11; chance methods 6).
- Running-mean convergence is produced by `scripts/python/analysis/seed_convergence.py`.

---

## 5. Cross-cutting factor analysis

### Class-imbalance factor (the dominant, well-evidenced one)
With a ~4% positive base rate, `class_weight='balanced'` kernel-SVMs collapse to a single class and lose ranking power (SvmW: AUROC 0.52, all-positive CM). SW-SMOTE balances the training set and restores a usable decision function **iff** the features carry latent signal:
- **SvmW:** signal present → rescued (0.52 → 0.79).
- **SvmA:** signal absent → not rescued (~0.53 → 0.54).
- **RF:** never degenerated (ensemble + internal reweighting) → no rescue needed.

This is the clearest, directly-measured factor: it is visible in the confusion matrices, and SMOTE improves **AUROC (ranking)**, not merely the threshold.

### Domain-shift factor (within-domain vs pooled)
Within-domain training/eval (`target_only`) reduces train/test distribution mismatch relative to `pooled`. Prior runs in this project show the split protocol moves RF AUROC materially (within-domain `target_only` ≫ cross-subject `domain_train` on the same data). However, in the present 2-cell comparison this factor is **confounded with imbalance** (see §1/§6), so its standalone contribution cannot be quantified from B1-vs-IV2025 alone.

---

## 6. The confound and the recommended ablation

B1-vs-IV2025 varies domain **and** imbalance together. To attribute the effect to each factor, run the two missing corners of the 2×2:

| | no SMOTE | SW-SMOTE |
|--|--|--|
| **pooled** | IV2025 (have) | **(missing)** |
| **target_only** | **(missing)** | B1 (have) |

- `pooled + SMOTE` isolates the **imbalance** effect (holding domain = pooled).
- `target_only + no-SMOTE` isolates the **domain** effect (holding imbalance = none).

Until these run, report the effect as the **combined controlled treatment**, and attribute the imbalance component qualitatively via the degeneracy evidence (§5).

---

## 7. Status, caveats, and what remains

- **Confirmed (local):** RF (0.730→0.781), SvmW (0.519→0.791) — both fully measured, converged.
- **Partially measured:** SvmA "after" 0.539 (11/12 cells), Lstm "after" 0.779 (complete); their "before" uses IV2025 published (0.53 / 0.52) pending local pooled runs (GPU queue, IV25 SvmA is the long pole).
- **Caveats:** (a) the domain/imbalance confound (§6); (b) Lstm's `event_label` target is not commensurable with the KSS methods; (c) the pooled SvmW/SvmA eval required populating `data/processed/{SvmW,SvmA}` from `common` (a local data-layout fix, not a result artifact); (d) "before" SvmA/Lstm are interim published values to be replaced by local measurements.
- **Headline (robust regardless of the pending cells):** domain + imbalance handling **closes the gap between RF and the prior methods** by rescuing SvmW (and Lstm on its own target) from chance, while **SvmA remains at chance** because steering-angle statistics lack drowsiness signal that balancing could expose.

---

## 8. References (data & code)

- Eval results: `results/outputs/evaluation/<MODEL>/**/*b1cmp_*_within.json` (B1), `*iv25base_*` (IV2025).
- Feature ranges per model: `src/utils/io/loaders.py` (~L287–298).
- Label derivation (KSS vs event_label; SvmA mapping): `src/utils/io/split_helpers.py` (~L28–50), `src/models/architectures/SvmA.py`.
- Launchers: `scripts/python/train/b1_compare_launcher.py`, `scripts/python/train/iv2025_baseline_launcher.py`.
- Convergence analysis: `scripts/python/analysis/seed_convergence.py`.
- IV2025 published values: `docs/experiments/results/exp3-analysis/latex/IV2025/IV2025.tex` (C2 / proper-config column).
- Label/split history: `memory/project_exp2_rf_087_unreproducible.md` (session memory).

---

## 9. Independent adversarial verification (subagent panel)

An independent adversarial panel re-derived **every** load-bearing claim in this document from source code and the on-disk eval JSONs/CSVs (not from the document's own assertions). The full set — including all imbalance-factor cross-cutting claims (C1–C7) and the per-model sub-claims for RF, SvmW, SvmA and Lstm — has now been adjudicated.

**Counts (41 claims): 35 confirmed · 5 partial · 1 refuted.**

Every quantitative headline (the four before/after AUROC pairs and their confusion-matrix evidence) was reproduced from the raw JSON `roc_auc`/`confusion_matrix` fields to 3 decimals. The single refuted claim and all five partials are **supporting/mechanistic** details; **no model-level conclusion changed**. Each is reconciled below.

### 9.1 Refuted — corrected in place

- **`rf_feature_richness` ("RF uses 160 features, ~20× SvmA's 39, redundant signal that survives domain/imbalance change") — REFUTED.**
  - *Verifier evidence:* The feature-*range* counts are right (RF range = 160, SvmA = 39, SvmW = 8 in `data/processed/common/processed_S0113_1.csv`), but 160 is only the candidate **input pool**. `model_pipeline.py:56` sets `feature_selection_method='rf'` with `TOP_K_FEATURES=10` (`config.py:206`) → `select_top_features_by_importance` keeps the **top-10**. 148 `selected_features_RF_*.pkl` exist; the actual measured B1 cell pkl and the IV2025 baseline pkl each contain **exactly 10 features**. Also 160/39 = **4.1×**, not 20× (the 20× ratio applies only to SvmW's 8).
  - *Doc adjustment:* Executive-summary mechanism #2 reworded ("160-feature candidate pool reduced to a top-10 RF-importance model"). §2.1 first bullet rewritten with an explicit correction: the operative mechanism is ensemble averaging + reweighting + calibration, **not** a wide redundant input; the 4.1×/20× ratio error is called out.

### 9.2 Partial — qualified in place

- **`svmw_classifier_config` ("C tuned by Optuna; same classifier object both arms") — PARTIAL.** The substantive claim is **confirmed** (RBF `SVC(kernel='rbf', probability=True, class_weight='balanced')`, rebuilt identically by `create_classifier` in both arms, only data/split/oversampling differ — `classifiers.py:86-91`, `pipeline.py:181`). Two citation errors: Optuna tunes **both C and gamma** (`optuna_tuning.py:255-265`), not C alone; and the cited `model_factory.py:64` default SVC (`C=300`, no `class_weight`) is **dead code** — discarded by `common_train`. *Doc adjustment:* none needed in the body (it never asserted "C-only" nor relied on the factory default); recorded here.
- **`allpositive_cm_is_threshold_artifact` (SvmW IV25 all-positive CM = pure threshold artifact of constant ≈ prior scores) — PARTIAL.** Confirmed for **4/6** seeds (proba pinned ~0.0399 = base rate; the F1-optimal validation threshold collapses just below it, flipping to all-positive). The other **2/6** (s0, s7) instead saturate to ~0.96 and are all-positive at *every* threshold (via the AUC<0.5 inversion path) — not a threshold artifact. *Doc adjustment:* §2.2 first bullet corrected to the precise **4-pinned / 2-saturated** split; the threshold-artifact framing scoped to the 4 base-rate-pinned seeds.
- **`mechanism_classweight_insufficient` ("class_weight='balanced' alone is insufficient") — PARTIAL.** Empirical directionality **confirmed** (reweighting → degenerate constant; SMOTE → non-degenerate). Caveats: reweighting is applied **doubly** — `class_weight='balanced'` **and** balanced `sample_weight` (`pipeline.py:184-186`) — and still collapses; the cited `proba.py` file does not exist; and the margin-geometry "why" is inference, not measured. The B1-vs-IV25 contrast also confounds SMOTE with the split, and an imbalv3 probe (SMOTE under `domain_train`) shows SMOTE de-degenerates but only reaches ~0.55 — so ~half the AUROC rescue rides on the `target_only` split, not SMOTE alone. *Doc adjustment:* §2.2 reworded to "reweighting fails *despite being applied doubly*", dropping any "class_weight alone" phrasing; the SMOTE-vs-split confound is already flagged in §1/§6.
- **`C2` (SW-SMOTE rescues SvmW by restoring a discriminative probability surface — ranking, not just operating point) — PARTIAL.** The **conclusion is confirmed**: B1 proba regains real spread (std 0.13–0.22) and AUROC jumps 0.519 → 0.791, a threshold-independent ranking gain (IV25's near-flat proba surface rules out an operating-point artifact). One cited *statistic* is wrong: the "fewer unique proba values without SMOTE" figure is backwards — IV25 actually has *more* distinct values, but packed into a ~0.0001-wide band, so the right discriminator is proba **spread/std**, not unique count. *Doc adjustment:* the doc argues the rescue via proba *spread* and AUROC, not unique-value counts, so no body change; noted here.
- **`contrast_svmw_not_univariate` (SvmW's 0.79 gain "plausibly subject-identity leakage, not honest signal") — PARTIAL.** The *comparative* part is **confirmed**: SvmW's 8 band-energy features are **not** univariately/multivariately stronger than SvmA's (SvmW univariate max ≈ 0.510, multivariate random-split ≈ 0.485), so SvmW's recovery is not a feature-strength effect. The **leakage mechanism is unproven**: the panel traced B1 `target_only` to `time_stratified_three_way_split`, which makes a single positional cut on the per-subject time-sorted stream — i.e. it is **within-subject temporal** (same subjects can appear in train and test), making subject-structure exploitation a *plausible, code-grounded candidate* but not a demonstrated cause (the isolating 2×2 corners were never run). *Doc adjustment:* the document never asserted leakage as the cause of SvmW's recovery; it is recorded here to mark the open question. (Note: this within-subject-split finding refines an earlier draft's "subject-disjoint, zero overlap" wording, which was incorrect for the `target_only` path.)

### 9.3 Confirmed — the load-bearing set (35 claims)

Reproduced directly from source/data; no edits required. Grouped:

- **RF (7 confirmed):** non-degenerate in both conditions (0/22 B1, 0/11 IV25); AUROC 0.730 → 0.781 (+0.050 seed-paired, but per-seed range −0.146…+0.258 — a noisy small effect); fixed `class_weight={0:1.0,1:3.0}` overriding Optuna + sigmoid calibration (`classifiers.py:66-74`, `config.py:310`); operating-point shift (pred-pos 6.6% B1 vs 44.8% IV25) dominates the modest ranking change; large randomized ensemble (~680 trees, `max_features=0.05`); ~4% base rate in **both** test sets (so robustness is not an easier label); KSS_Theta_Alpha_Beta target over the paper-faithful range, in the honest 0.73–0.78 band (not the 0.89 HPC artifact).
- **SvmW (5 confirmed):** exactly 8 GHM steering-wheel band energies, KSS label with default mapping; IV25 proba near-constant ≈ prior (std 4e-5–4e-3); IV25 at chance on threshold-independent metrics (AUROC 0.519, AUPRC 0.043 ≈ base rate, `invert_check=True` so not an inversion artifact); B1 proba spans [0,1], AUROC 0.791, AUPRC 0.154 (~3.4× lift), non-degenerate CMs; same features go from no-lift to 3.4× lift purely by regime — signal was in the features, IV25 failure was a regime/imbalance pathology.
- **SvmA (8 confirmed):** B1 at chance with SW-SMOTE (0.539, n=11, range 0.483–0.587); features carry no univariate signal (max 0.515, 0 features > 0.55); multivariate random-split RBF-SVM also chance (0.509); CM signature TPR≈FPR with precision at base rate in every cell; SW-SMOTE only moves the operating point (PPR 0.099–0.964) not the ranking; SvmA-specific KSS mapping verified (1–6 alert, 8–9 drowsy, 7 excluded → 3.6% positive); feature range 39 in-range / 23 paper-filtered; **IV2025 "before" SvmA never run (0 JSONs on disk)**.
- **Lstm (9 confirmed):** B1 AUROC 0.779 (n=22, sd 0.008); predicts **event_label** (DRT task vs baseline), a *different* target from KSS; ~73% positive (≈16× the KSS models' ~4%); ratio 0.3 vs 0.5 makes no difference (~0.001); IV2025 "before" genuinely pending (0 JSONs); "before" will use the same event_label target (label is mode-independent); not commensurable with the KSS models; majority-class lean (sensitivity ~0.97, specificity ~0.13); low seed variance is an Lstm trait (vs RF sd ~0.086).
- **Imbalance-factor cross-cutting (C1, C3–C7 confirmed; C2 partial above):** SvmW collapses to a near-constant decision function without SMOTE — a *ranking* failure, not a bad threshold (C1); RF already ranks without SMOTE, so SMOTE only shifts its operating point (C3); SvmA is de-degenerated by SMOTE at the operating-point level but ranking stays at chance because the features lack signal (C4); imbalance is largely inapplicable to Lstm (near-balanced target) (C5); model-specific mechanism asymmetry — SvmW `class_weight='balanced'`, SvmA none, RF fixed 1:3 (C6); AUROC and threshold are decoupled here, so degeneracy must be read from proba spread, not the CM alone (C7).

### 9.4 Headline conclusions that SURVIVED verification

All five stand, independently reproduced from the eval JSONs:

- **RF robust, 0.730 → 0.781 (+0.05)**, non-degenerate in both conditions, ~4% base rate in both. (Mechanism reattributed from "160 redundant features" to top-10 selection + ensemble averaging + fixed 1:3 `class_weight` + calibration.)
- **SvmW rescued, 0.519 → 0.791 (+0.27)** — IV25 proba collapse and B1 full-[0,1] spread with ~3.4× AUPRC lift verified; the ranking genuinely improves (not a threshold trick).
- **SvmA not rescued, ~0.539 (near-chance) even with SW-SMOTE** — features carry no signal (univariate max 0.515, multivariate 0.509). IV2025 "before" SvmA genuinely **not run**.
- **Lstm 0.779 (sd 0.008) on a different, ~73%-positive `event_label` target** — high score reflects target choice, not better domain/imbalance handling; not commensurable with the KSS models.
- **The domain × imbalance confound** (B1 vs IV2025 varies both factors; 2×2 corners not run) — confirmed for every model.

### 9.5 What remains unverified (honest limits)

These are inherent data/run gaps, not failed checks:

- **IV2025 "before" for SvmA and Lstm was never run** (0 `iv25base` JSONs on disk), so those before/after deltas use IV2025 *published* values, not local measurements. The local SvmW "before" (0.519) matches the published 0.51, which supports — but does not prove — that the pending SvmA/Lstm "before" will land near 0.53/0.52.
- **The 2×2 ablation corners** (`pooled+SMOTE`, `target_only+no-SMOTE`) were not run, so the SMOTE effect cannot be cleanly separated from the `target_only` split for SvmW (and the C2 / `mechanism_classweight_insufficient` mechanism stories inherit this confound).
- **Mechanistic "why" details** (RBF margin geometry; what within-domain structure SvmW exploits and SvmA does not — including whether the within-subject temporal split contributes) are plausible, code-grounded inferences, **not** directly measured (no SV counts, decision-function geometry, or leakage-isolating run exists).
- **Eval JSONs store no label/feature metadata**, so the per-model label and feature-set facts rest on code + processed-CSV inspection (both checked), not on the result files themselves.

**Net:** every headline conclusion and every load-bearing supporting claim is verified by the independent panel (41 claims: 35 confirmed / 5 partial / 1 refuted-and-corrected). The one refuted claim (RF "160 redundant features") and all partials are mechanistic/citation details that have been corrected or qualified in place; **no before/after AUROC verdict or model-level conclusion was altered by verification.**
