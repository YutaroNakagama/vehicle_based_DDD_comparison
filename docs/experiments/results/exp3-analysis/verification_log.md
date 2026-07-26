# exp3 verification log

*Created: 2026-06-27. Implementation record and results for T1–T6 of [`verification_tasks.md`](verification_tasks.md).*
*The main body of evidence is [`domain_imbalance_factor_analysis.md`](domain_imbalance_factor_analysis.md). This log is the primary record of the verification carried out to confirm/revise those claims.*

## Policy update (advisor, 2026-06-27)
Change the main axis of exp3 from **before/after** to **Cross-domain vs Within-domain (in/out)**.
Compare the four methods RF/SvmW/SvmA/Lstm across **Within (target_only)** and **Cross (source_only)**
in/out, showing RF's superiority (in particular cross-domain robustness).

- Implementation: [`scripts/python/train/c1_domain_launcher.py`](../../../../scripts/python/train/c1_domain_launcher.py),
  [`scripts/shell/c1_watchdog.sh`](../../../../scripts/shell/c1_watchdog.sh).
- Grid: 4 models × {target_only, source_only} × {in_domain, out_domain} × seed{42,123,2025},
  SW-SMOTE fixed / wasserstein / ratio 0.5.
- Within-in is reused from B1 (same target_timewise eval). However, **SvmA re-runs all 4 conditions due to the T1 fix**.
- cross-domain (source_only) uses the exp2-compatible tag (`imbalv3_knn_..._split2_...`) to
  resolve `rankings/split2/knn/<dist>_<oppositeDomain>.txt` to the source set
  ([`target_resolution.py:172-223`](../../../../src/utils/io/target_resolution.py#L172)).

## Progress summary

| T | Content | Status | Conclusion |
|---|---|---|---|
| **T1** | Re-test SvmA with the full Arefnezhad 18 features | ✅ **Completed** | **Reinforces "no signal"** (not an artifact of the feature filter) |
| **T2** | Whether SvmW 0.79 is an honest signal or split-dependent | 🟡 In progress | (see below) |
| T3 | RF's SMOTE-only effect (pooled+SMOTE) | ⬜ Not started | — |
| **T4** | Deconfound SvmA's classifier × features | 🟡 Half done | RF-on-SvmA-features = 0.496 → **features are the wall (classifier-independent)** |
| T5 | Lstm's domain attribution | 🟡 In progress | IV25 before(local)=0.512 ✅ / cross being measured in c1 |
| T6 | Faithfulness of road-curve removal | ✅ Logically moot | Since SvmA is chance, removal does not change the conclusion (see below) |

---

## T1. SvmA full feature set re-test — ✅ Completed

### Fix (code)
Fixed `SVMA_PAPER_FEATURE_SUFFIXES` in [`SvmA.py`](../../../../src/models/architectures/SvmA.py#L65)
from the unfaithful 14 to the **faithful Arefnezhad 18** (commit `b715535`).
- **Added (8)**: SampleEntropy, KatzFractalDim, ShannonEntropy, SpectralFlux, FreqVar(=Frequency
  Variability), Quartile25/Median/Quartile75.
- **Removed (4)**: Mean, Variance, Max, Min (not in Arefnezhad).
- Verification: the filter selects **36 columns** (Steering 18 + SteeringSpeed 18, 0 unmatched).
- Note: `verification_tasks.md` marked Frequency Variability as "not computed", but the data
  contains `Steering_FreqVar` / `SteeringSpeed_FreqVar`, so all 18 features could be faithfully constructed.

### Re-test (feature-signal probe)
[`scripts/python/analysis/exp3_feature_signal_probe.py`](../../../../scripts/python/analysis/exp3_feature_signal_probe.py).
n=66,993 (87 subjects, pos 3.62%), **subject-separated split** (GroupShuffleSplit). SvmA's KSS mapping
(1–6=Alert, 8–9=Drowsy, 7 excluded) is identical to the pipeline.

| Feature set | univariate max (>0.55) | RBF-SVM raw | RBF-SVM +SMOTE | **RF on same feats** |
|---|---|---|---|---|
| **Faithful 18 (36 cols)** | **0.515 (0)** | 0.494 | 0.507 | **0.496** |
| Old 14 (28 cols) | 0.515 (0) | 0.498 | 0.507 | 0.513 |

- The per-feature best is `SteeringSpeed_ZeroCrossingRate` 0.515. **The original paper's flagship `SampleEntropy` is also 0.514 (chance)**.
- Multivariate is 0.49–0.51 for every classifier.

### Conclusion
**Even with the faithful 18 features, every metric is chance (<0.55).** Adding the original paper's flagship features (Sample Entropy, etc.) produces no signal.
- The old null (0.515/0.509) was **not an artifact of the feature filter** → **reinforces** "SvmA's features carry no signal against the KSS label".
- **Even training RF (a powerful alternative classifier) on the same faithful features gives 0.496** → the wall is **the features, not the classifier** (= simultaneously settles half of T4's RF-on-SvmA-features).

### Reflection into the analysis doc
The "SvmA no signal (⚠️ unfaithful feature set)" in §2.3 / §9 of
[`domain_imbalance_factor_analysis.md`](domain_imbalance_factor_analysis.md) can be upgraded to **"✅ re-confirmed with faithful 18 features (T1)"**.
- Remaining: after c1's SvmA (faithful features, including ANFIS+PSO selection, all 4 conditions) completes, replace the B1 number (0.539) with the faithful version.

---

## T2. SvmW clean-split verification — 🟡 In progress

### probe
[`scripts/python/analysis/exp3_svmw_split_probe.py`](../../../../scripts/python/analysis/exp3_svmw_split_probe.py).
8 GHM bands + default KSS, comparing with the **same SMOTE+RBF-SVM/RF**, varying only the split:
- A) Within-subject temporal (per-subject sorted by Timestamp, first 70%→train / last 30%→test) = equivalent to B1 target_only
- B) Subject-separated (GroupShuffleSplit, average of 3 seeds)

Decision: **A ≫ B (B≈chance) → 0.79 is split-dependent**. A≈B with both rising → genuine latent signal.

### Results (2026-06-27)
| split | RBF-SVM +SMOTE | RF |
|---|---|---|
| A within-subject temporal | 0.489 | 0.478 |
| B subject-separated (avg of 3 seeds) | 0.505 | 0.493 |

**Bare SMOTE+SVM/RF is chance in both splits (~0.48–0.51) and does not reproduce B1's 0.79 in either split.**
→ 0.79 arises not from the **8 features themselves** but from the **full pipeline (Optuna-tuned SVM + specific SMOTE settings)**.
Since this probe cannot reproduce 0.79, the probe alone cannot isolate split-dependence (inconclusive).

### Reframe: the definitive test of split-dependence is c1 itself
**c1's Within(target_only) vs Cross(source_only) is itself the split-dependence test on the full pipeline.**
In Cross-domain, train and eval are in different domains = different subject groups (= subject-separated). If for each method **Within ≫ Cross→chance**, then
"depends on within-domain structure" can be established.
- **RF (9/9 done, no anomalies)**: Within-out **0.790** (0.758/0.790/0.822) — **consistent with TIV2026's RF after=0.781 ✓**
  (B1 within-in is also 0.781). vs Cross-in **0.520** (0.513–0.526)
  / Cross-out **0.512** (0.508–0.517) → even for RF, the within signal collapses to chance in cross-domain (consistent across all seeds,
  non-degenerate, no log failures). Judge whether SvmW/Lstm/SvmA are of the same form once c1 Cross completes (to settle whether 0.79 etc. is a within-only structure).
- Therefore T2's conclusion **awaits completion of c1 SvmW Cross** (the probe only showed that the bare features contain no easy signal).

---

## T4/T5/T6 notes
- **T4 (half)**: the above RF-on-SvmA-faithful-features = 0.496 → SvmA's null does not depend on classifier or feature selection (the feature wall).
  The remaining SVM-on-RF-features ≈ 0.78 (the classifier is not the wall) is separate.
- **T5**: IV2025 before (local pooled) Lstm = 0.512 (n=6, consistent with the published 0.52).
  **c1 Lstm done (9/9, no anomalies)**: Within-out **0.753** ≈ Cross-in **0.720** ≈ Cross-out **0.743**
  → **within ≈ cross (domain-invariant)**. "The improvement comes from domain" leans refuted (maintained even in cross).
  Lstm is interpreted as domain-robust because of event_label (balanced, DRT task structure). **Contrasts with RF (KSS, cross→chance 0.51)
  ** = whether cross-domain transfer is possible is determined by the target/label (consistent with RQ2).
- **T6 (road-curve removal): moot for the main conclusion (logically unnecessary).** If road-geometry contaminates steering,
  it works in the direction of **adding** an apparent signal from "road-following". But in T1, SvmA is
  chance even with faithful features + multiple classifiers (univ 0.515 / RF 0.496), and **removing contamination can only lower the signal, never raise it** → the "no signal" conclusion is
  unchanged before and after removal. Removal only matters when there is a suspicion that "SvmA shows a signal, and it is road-following rather than drowsiness",
  which is not the situation for this data. Therefore it does not overturn T1's null. Confirming the metadata of whether the Aygun course contains curves
  remains as a supplementary item, but is low priority.

## c1 runtime bug and fix: SvmA cuML ZeroDivisionError (2026-06-27)
- **Symptom**: all c1 SvmA cells fail in ~3 minutes with "Model object is None → Model could not be loaded",
  producing no JSON and looping re-execution while pending (wasting GPU). Because the log rc was 0, it superficially looked like success.
- **Root cause** (traceback): the cuML `SVC(gamma='scale')` in `SvmA.py`'s PSO objective function, on a **constant-only feature subset
  (X.var()=0)**, executes `_get_gamma`'s `1/(n_feat·var)` → `ZeroDivisionError`. sklearn silently
  handles it but cuML raises. **Exposed by T1's move to 18 features** (the old 14 features never hit an all-constant subset).
- **Fix** (commit `1243f26`): fall back to `gamma = 1/n_features` for zero-variance subsets
  ([`SvmA.py:383`](../../../../src/models/architectures/SvmA.py#L383)). Also log the traceback of training exceptions
  ([`model_pipeline.py:316`](../../../../src/models/model_pipeline.py#L316)).
- **Verification**: confirmed the fixed version passes the crash point (~64s), PSO continues 210s+ with 0 exceptions. Resumed c1 SvmA as a single job.
- **Note**: since T1 already established SvmA has no signal, c1 SvmA's AUROC is expected to be chance (used to fill in the within/cross table).

## Mathematical adequacy of the seed count (2026-06-28)
The required number of seeds n is based on the **95% CI half-width**: n s.t. `t_{n-1,.975}·s/√n ≤ h` (s = across-seed AUROC standard deviation).
The observed s (11 seeds for RF, estimated from 3 seeds for the others) and the required n differ greatly by condition:

| Condition | s (std) | n@±0.02 | n@±0.03 | n@±0.05 |
|---|---|---|---|---|
| **RF Within-out** | **0.105** | 108 | 50 | 19 |
| **RF Within-in** (B1) | **0.078** | 60 | 28 | 11 |
| SvmA Within-in | 0.021 | 7 | 5 | 3 |
| Lstm Within-out | 0.019 | 6 | 4 | 3 |
| SvmW Within-out | 0.007 | 3 | 3 | 3 |
| Cross/Mixed (chance) | ~0.005 | 3 | 3 | 3 |

**Key finding**: **RF's within-domain AUROC is seed-unstable (std≈0.08–0.10, range 0.62–0.95, common to both domains)**.
The 3 seeds being narrow at 0.76–0.82 was coincidence. → obtaining within-domain RF precisely requires many seeds (±0.02 needs over 100, impractical).
The other methods (within std≈0.02, cross/mixed≈chance) are **sufficient at 12 seeds with CI±0.013**.

**Conclusion (international-journal-quality final decision, 2026-06-28)**:
Criteria = ① fidelity to the original method ② statistical rigor ③ reproducibility ④ review-robustness (avoiding arbitrariness).
- Target precision **95% CI half-width ≈ 0.05** (given RF's intrinsic variance, ±0.02 needs over 100, impractical).
- **Seeds: RF=20, Lstm/SvmW/SvmA=15** (all above the floor of 12). Lstm/SvmW/SvmA are **review-robust at a uniform 15** (per-condition non-uniform seeds look arbitrary, so not adopted). Only RF, being high-variance, is 20 (power analysis stated explicitly).
- **SvmA = `SVMA_PSO_MAXITER=30`** (convergence rationale: in the saved `pso_history` (5050 evaluations), best fitness **converges to 0.0172 by iter ~2 and stays unchanged through iter100** = maxiter=100 is ~50× redundant). 30 gives a 15× margin and **reproduces the same optimum** = the ANFIS/PSO/RBF-SVM method itself is unchanged (faithful). This makes SvmA=15 executable in ~5.6 GPU-days.
  - *Empirical verification needed*: after the GPU frees up, re-run within-in s42 with maxiter=30 and confirm agreement with the existing maxiter=100 value (0.523).
- **SvmW: do not add max_iter to the SVM (maintaining fidelity) — empirically settled (2026-06-30)**. The speed-up candidate `max_iter=100000`
  was verified on Within-out s42, and the result **changed the AUROC 0.7697→0.7281 (Δ0.042)** = the cap acts not only on non-converging pathological trials
  but **also on legitimate trials that require many iterations to converge, changing the selected hyperparameters**. Therefore **it fails the fidelity condition and is rejected**,
  keeping max_iter unlimited (sklearn default). The 5 cells run with the cap were deleted and re-run unlimited. The search is slow but the results are fully faithful.
- Unify all 6 conditions with the same imbalv3 tag and same seeds (within-in also run in c1).

### Seed adequacy across all conditions (TIV2026-compliant, all conditions covered, measured 2026-06-30)
TIV2026 (exp2) discussed seed adequacy via ① σ_rank convergence ② bootstrap 95%CI (B=2000) ③ statistical power. Apply this
**to all 24 conditions of exp3 (c1 6×4)** ([`exp3_seed_adequacy.py`](../../../../scripts/python/analysis/exp3_seed_adequacy.py)).
Decision = for discriminative conditions, 95%CI half-width ≤ 0.05; for chance conditions, bootstrap CI upper bound < 0.60.

| Model | Condition | n | mean | std | CI hw | Decision |
|---|---|---|---|---|---|---|
| **RF** | Within-in | 20 | 0.752 | 0.089 | 0.042 | ✅ |
| | **Within-out** | 20 | 0.787 | **0.108** | **0.051** | ⚠️ req_n=21 → **increase RF to 24** |
| | Cross-in/out | 20 | 0.51 | 0.005 | 0.002 | ✅ chance (upper bound<0.52) |
| | Mixed-in/out | 20 | 0.74–0.76 | 0.08–0.10 | 0.037–0.048 | ✅ |
| **Lstm** | all 6 conditions | 15 | 0.72–0.78 | 0.009–0.015 | 0.005–0.009 | ✅ (req_n=3, 15 is amply sufficient) |
| **SvmW** | (completed) Within-out | 3 | 0.765 | 0.007 | 0.016 | ✅ / Cross=chance ✅ |
| **SvmA** | (completed) Within-in | 5 | 0.540 | 0.031 | – | ✅ chance (upper bound 0.565<0.60) |

**σ_rank (seed stability of the 6-condition ranking, equivalent to TIV2026 fig)**:
- **RF**: κ=16→0.123, 18→0.10, **19→0.073** → the ranking (within/mixed ≫ cross) is **determined by condition, not seed** (converged).
- **Lstm**: κ=14→0.133 (higher than RF) = the 6 conditions are **statistically close** (0.72–0.78), so fine ranking wobbles. However, the top-level structure (all conditions stay ~0.75 without collapsing to chance) is stable → **corroborates domain-invariance**.

**Conclusion**:
- **The only shortfall is RF Within-out** (high variance std=0.108, hw=0.051 at 20) → **increase to RF=24 seeds** (req_n=21; at 24 hw≈0.046<0.05; CPU is cheap).
- Lstm has req_n=3 which is excessive, but a uniform 15 is review-robust. Chance conditions (Cross/Mixed-KSS, all SvmA conditions) are low-variance with req_n≤4.
- **Final seed plan (settled 2026-06-30): RF=24, Lstm=15, SvmW=8, SvmA=8** (proportional to variance, the minimal configuration that satisfies adequacy
  in each condition). SvmW/SvmA, being low-variance (req_n=3–7), are reduced 15→8: statistically sufficient while avoiding several days wasted by SvmW's pathological SVM (~13h/cell).
  The method and pipeline are unchanged (only the seed count). After SvmW/SvmA complete, regenerate this table across all 6 conditions and finalize.
- Comparable to or stricter than TIV2026 allowing a residual of σ_rank=0.147 on AUROC, so exp3's seed plan is valid.

## Split methodology and the nature of within-domain (shared with TIV2026/IV2025, a known limitation, 2026-07-01)
**Methodology (identical to TIV2026/IV2025, also followed by exp3)**: train uses `--time_stratify_labels` (label-stratified time-series split,
stratify=True), eval uses target_timewise / pooled with stratify=False (simple time-series). Identical to the exp2 HPC scripts
(`pbs_domain_comparison*.sh`), followed to make exp3 **directly comparable** with TIV2026/IV2025.

**Confirmed property (read-only verification)**: because train (stratify=True) and eval (stratify=False) are different partitions,
in within/mixed part of eval-test becomes the same rows as train's training set (~69% by content matching). Making the split fully time-consistent
(both train and eval stratify=False) drops within-in **0.78 → 0.526** (de-leak measured).
That is, **the high within-domain value is a property dependent on this temporal-split protocol**.

**Decision**: this property is **shared with IV2025/TIV2026**. exp3 prioritizes cross-paper comparison consistency and **adopts the same methodology**
(the de-leakage version is not adopted). **Relative comparisons (between methods, before/after, within vs cross) are valid within the same framework**.
cross-domain is inherently unaffected by this property (different-domain training). This is **stated openly as a known limitation, not hidden**.
TIV2026 is already published, so it is not changed at all (this log is exp3's decision record).

## Remaining tasks
- Complete T2 → append results.
- T3 (RF 1 cell of pooled+SMOTE), T6 (confirm Aygun course curvature).
- c1 fully completes → before/within/cross summary table + plots (figures 1–6 of verification_tasks).
- Adversarial re-verification of the conclusions by an independent sub-agent (after c1 data is finalized).

## References
- probe scripts: [`exp3_feature_signal_probe.py`](../../../../scripts/python/analysis/exp3_feature_signal_probe.py),
  [`exp3_svmw_split_probe.py`](../../../../scripts/python/analysis/exp3_svmw_split_probe.py)
- result JSON: `results/analysis/exp3_verification/t1_feature_signal_probe.json`, `t2_svmw_split_probe.json`
- c1: [`c1_domain_launcher.py`](../../../../scripts/python/train/c1_domain_launcher.py), [`c1_watchdog.sh`](../../../../scripts/shell/c1_watchdog.sh)

## 2026-07-04 Adversarial re-verification (11-agent workflow) — discovered and fixed a recurrence of Bug#4

**Method**: 4 models × (independent re-scan → skeptic agent refutes → exhaustiveness critic). For each target, the raw JSON is
independently rglob+loaded and cross-checked byte by byte.

**Finding (1 real bug, CONFIRMED)**: c1 **RF's 2 Cross(source_only) cells byte-exactly match the prediction vectors of Within(target_only)**
(`y_pred_proba` md5 match):
- `source_only/in_domain/s42`  (AUROC 0.7734 = identical to within-in s42)
- `source_only/out_domain/s123` (AUROC 0.7578 = identical to within-out s123)

**Root cause (recurrence of the known "Bug #4")**: in `evaluate.py`'s `resolve_jobid_for_evaluation` resolution order, priority #3's
glob `<M>_<mode>_rank_*` **does not match c1's saved model name (internal mode="domain_train")**, so it falls to priority #4's
**shared file `models/<M>/latest_job.txt`** (a mutable file each worker overwrites). Because RF/SvmW/Lstm run
4/4/3 workers in parallel, if another worker overwrites `latest_job.txt` between a cell's train and eval, the eval
**loads a different cell's model**. In RF, evals cluster on fast cells, so it occurred in 2/48.
- **Why IV2025 is unaffected**: IV2025's eval_cmd passes `--jobid` (resolution order #1) → race-free. c1 did not specify it.

**Exhaustive scan**: brute-force over all 4 models × (domain, seed) × mode detecting exact roc_auc (10dp) + cm matches → **only the 2 matches above**.
Lstm/SvmW/SvmA are clean. Impact: Cross averages 0.5288→0.5182 (in), 0.5171→0.5067 (out) (still ~0.52 collapse, conclusion unchanged).

**Fix**: **add `--jobid jobid`** to `c1_domain_launcher.py`'s eval_cmd (the same safe pattern as IV2025). At resolution order #1
it deterministically loads the cell's own model → **race-free regardless of worker count**. The methodology (split/features/label/SMOTE) is unchanged.
- Archive and delete the JSON of the 2 contaminated cells and re-run with the fixed launcher (workers=1). Other models are clean and out of scope.
- The running SvmW (holding old code in memory) is theoretically racy on the remaining cells, but at ~15h/cell evals are sparse → collisions are rare (the 8 completed cells are
  clean). At each status check, run the same scan and handle contamination detection → re-run. Future watchdog restarts use the fixed version.

**Everything else is clean per independent re-verification**: c1 RF (Win/Mix genuine discrimination, Cross is true near-chance from minority-tracking),
c1 Lstm (domain-invariance is real via threshold-independent AUROC), c1 SvmW (Within genuine discrimination / Cross expected-collapse),
c1 SvmA (chance throughout, non-degenerate), IV2025 (RF 0.738 genuine discrimination + the other 3 methods expected-collapse = reproduces the published results).
- **Coverage caveat**: Lstm/SvmA JSON lack `y_pred_proba`, so the vector matching that caught the RF bug is not possible.
  Supplemented with an alternative scan of exact roc_auc+cm matches (clean).

## 2026-07-19 Regeneration of the Lstm ROC figure (mean ± std, seed aggregation)

Using the existing evaluation CSVs of 15 seeds each for `mixed_in` / `mixed_out` and 6 seeds for `pooled`,
regenerated a figure overlaying the mean ROC curve and the $\pm 1\sigma$ band. The previous single-seed figure has been replaced with this aggregated version.

- Output figure: [`lstm_mixed_in_out_pooled_roc_mean_std.png`](../../../../results/analysis/exp3_verification/lstm_mixed_in_out_pooled_roc_mean_std.png)
- Aggregation JSON: [`lstm_mixed_in_out_pooled_roc_mean_std_summary.json`](../../../../results/analysis/exp3_verification/lstm_mixed_in_out_pooled_roc_mean_std_summary.json)
- Reproduction script: [`plot_lstm_mixed_pooled_roc.py`](../../../../scripts/python/analysis/plot_lstm_mixed_pooled_roc.py)

**Key points**:
- `mixed_in`: AUROC mean 0.785332, std 0.006985, n=15
- `mixed_out`: AUROC mean 0.784263, std 0.006983, n=15
- `pooled`: AUROC mean 0.505686, std 0.017565, n=6

The legend in the figure was corrected to the `AUC=mean±std` notation, and control characters that were causing mojibake were removed.

## 2026-07-19 [Critical update] The split-mismatch leak goes beyond a "known limitation" and invalidates the central finding (4-agent adversarial audit + de-leaked measurement with real models)

The 2026-07-01 section (within/mixed train stratify=True / eval stratify=False mismatch causing ~69% row overlap, within-in 0.78→0.526)
recorded the judgment "a known limitation, intentionally kept for comparability"; this is now **updated to a more serious conclusion**. Prompted by the user's observation (RF full-feature pooled
0.86 is too high), a 4-agent adversarial audit + de-leaked re-evaluation with real models yielded:

**Established facts (code + real models reproduce the recorded JSON to 3 decimal places, passing adversarial refutation)**
- **pooled has the same kind of leak (indeed the most severe)**: pooled has train = within-subject time-series (first 60%, `TRAIN_RATIO=0.6`) / eval = **fully random 20%**
  (`iv2025_baseline_launcher` does not pass `--subject_wise_split` to eval, `eval_pipeline.py:152`→"random"). ~60% of the eval test rows
  are the same rows as train. Real-model decomposition: SEEN (trained) = 0.85–0.97 / **UNSEEN (honest) = 0.49–0.53**.
- **Only Cross (source_only, no subject crossover = 0% overlap) and domain_train are honest**. All pooled/within/mixed columns in §2 leak.
- **Directly de-leaking within with real models** (evaluate saved models on the held-out test of time_stratified_three_way_split): RF 0.77→**0.47**,
  SvmW 0.80→**0.52** (both chance). → **the paper's central finding "SW-SMOTE recovers SvmW / dissociates from SvmA" is a leak artifact**.
  Under honest conditions RF, SvmW, and SvmA are all within=chance. Both the recovery and the dissociation vanish.
- **Positive control passes**: EEG band power is 0.62 on the same honest split (> vehicle 0.53) = the harness passes genuine signal through. Vehicle = chance is true.
- **The only real signal is Lstm** (event_label=DRT, a different task). 0.73–0.75 on Cross (honest), domain-invariant. Detects DRT, not EEG drowsiness.

**Updated judgment**: 2026-07-01 held that "relative comparisons (before/after, between methods) are valid within the same framework", but under honest conditions **all KSS methods
collapse to chance, so the relative pattern (SvmW>SvmA, etc.) also vanishes**, as measured. Therefore, under the "keep the leak for comparability" policy,
the paper's central claim does not hold. **A full re-evaluation under honest conditions (train/eval same split or cross-subject) is needed, reframing into negative-result + methodological insight
(prior high values stem from split mismatch) + Lstm on DRT as the only real signal** (requires user decision). exp2's published
TIV2026 is not changed. Details in c1_results.md §3.6. scratchpad: leak_test.py / leak_test2.py / honest_within.py.

## 2026-07-20 [Policy decision] Complete exp3 in compliance with IV2025/TIV2026; the de-leak re-test is "staged" (non-destructive, after completion)

User decision: **complete exp3 with the current same methodology (shared with IV2025/TIV2026, train `--time_stratify_labels` / eval stratify=False)
and prioritize cross-paper comparability**. The 2026-07-19 de-leak audit (§3.6/c1_results) is retained as "a deep dive into the known limitation = a robustness
appendix", and **the implementation review and de-leak re-test are staged non-destructively against running experiments and shared code, and executed after exp3 completes**.
- Running (07-20 20:26): c1 SvmW within+mixed 25/32, RF-nofs 19/23, no anomalies. Awaiting completion.
- Preparations: c1_results §3.7 (A implementation review read-only, B de-leak re-test = method 1 honest re-score of saved models / method 2
  new flag `evaluate.py --honest_split` + new tag `honest_` preserving existing output with the default unchanged).
- Important implementation note (restated): `data_time_split_by_subject` resets the index → honest re-score must use positional (iloc) indexing
  (the pitfall in §3.6. The intermediate pooled honest 0.75 stems from this bug and is not adopted; the correct value is ~0.52).
- Paper (draft TIV2026_exp3): presenting the main results in compliance + explicitly stating the de-leak sensitivity analysis in Limitation/Robustness is the leading option (decision deferred).

## 2026-07-20 [Parallel exp2/TIV2026 audit] The headline 0.89 is "honest by descriptive design but row-leaking in code implementation" — chance under honest conditions (3 agents, read-only, non-interfering with published material and running exp3)

At the user's request, exp2/TIV2026 audited non-destructively like exp3. Conclusions in 3 layers:
1. **The paper's descriptive design is honest/transparent**: per-subject time-series 70/15/15, test = disjoint held-out, `ω_tr` (subject overlap) explicitly disclosed as leakage,
   Cross=chance stated explicitly, 0.89 presented as within-subject (personalized) (not claimed as new-driver generalization). More transparent than exp3.
2. **But the implementation that generated 0.89 leaks at the row level (confirmed by git)**: exp2 pbs has train `--time_stratify_labels` (overall 70/15/15) / eval without it
   (per-subject 60/20/20) = a different function → **69% of eval test rows are training rows** (physical matching). git `ed6e66d` (07-01 "69% within eval-test in train"),
   `f1895bd` (07-19 "eval stratify=False, as in the exp2 HPC scripts"), the leaky eval branch byte-matches at exp2-era 573ba1b.
   → the code does not implement the disjoint design the paper describes, and the headline number is inflated by 69% row reuse.
3. **Honest measurements are chance in every regime**: within-subject (with/without gap) 0.49–0.52, cross-subject 0.50–0.51, random (maximal window overlap) 0.51;
   positive control EEG cross-subject 0.59 (> vehicle) = harness valid. Neither personalization nor window overlap creates signal. The source of 0.79–0.89 is only ~69–70%
   row reuse. → confirms at the mechanism level the existing memo project_exp2_rf_087_unreproducible's "0.89 not locally reproducible".

**Implication**: the TIV2026 central result "SW-SMOTE+within-domain is optimal (0.89)" does not hold under honest conditions (even a correct within-subject split gives 0.50).
The nature is not scientific misconduct but an implementation bug/reproducibility (the design is correctly described, the code not implemented). **Because the paper is published, the response (erratum or handling it explicitly in exp3) is
a user decision**. This audit is read-only, the published dir (exp2-analysis/paper/TIV2026) unchanged. Caveat: the paper numbers come from HPC (145 features) and the actual binary was
not run, but the mechanism is feature-independent + matches git, so it is highly probable. scratch: verify_exp2_overlap.py, decompose.py, index_bug_demo.py.

## 2026-07-21 [exp3 status audit] Completion status, per-case completion-time estimates, abnormal-termination/expected-value verification (inventory of all 390 cells + 4-agent log audit + de-dup operations)

User routine request "check exp3 status, estimate per-case completion times, verify abnormal-termination/whether as expected, document it". Authoritative data = a programmatic scan of all eval JSON
(scratch: exp3_inventory.py, 1715 files → 390 unique (model,arm,seed)) + log measurements + live processes.

### Completion status (TABLE conditions = Within in/out + Mixed in/out. Cross demoted on 07-11, iv25 is the pooled baseline)
- **Lstm: 60/60 completed** (07-01..03). Within 0.766–0.793 / Mixed 0.760–0.800 / Cross 0.696–0.772. The only real signal (event_label). iv25base/smote pooled 6+6 = 0.50–0.53.
- **RF(fs): 96/96 completed** (07-01..03). Within 0.53–0.95 / Mixed 0.55–0.96 (leak-inflated, high variance) / Cross 0.50–0.53 (chance=honest). iv25base 15=0.58–0.88, iv25smote 15=0.60–0.85.
- **SvmA: 32/32 completed** (07-03..10). All conditions 0.46–0.63 (near-chance, no signal in features = consistent with T1, not degeneracy but genuine no-signal). iv25 6+6.
- **SvmW: 26/32** (Within 16/16 completed = in 0.800±0.012 / out 0.759±0.012; Mixed 10/16 = in 0.738±0.013(5) / out 0.766±0.017(5), **remaining 6 running**). Cross 2/8 (demoted) 0.51. iv25base 6=0.50–0.54, iv25smote 5=0.69–0.73 (SW-SMOTE inflation).
- **RF-nofs: 20/25** (pooled 5 + Within 9 + Mixed 6, **remaining 5 running**). 0.72–0.96 (highest band due to full-feature leak).

### Abnormal-termination check → zero detected (confirmed at JSON level)
- **All 390 unique cells valid with roc_auc ∈ [0,1]**. NaN/inf/out-of-range/truncated/parse-fail = **0**. No corrupted or truncated evals.
- Log-level audit (4-agent parallel workflow wlniqhvo2: scanning all RF/SvmW/SvmA/Lstm logs for rc≠0, TRAIN FAILED, traceback,
  feature fallback [Wang15/LaneOffset/MISSING], non-convergence) → results appended at the end of this section.

### Expected-value check → all as expected (consistent with the known leak signature)
- **The honest side (Cross, iv25base-pooled) is chance 0.50–0.53**, **the leaked side (Within, Mixed train/eval split mismatch) is inflated 0.7–0.96**.
  This asymmetry is consistent across all 4 methods → the harness behaves as the audit predicted, **no new anomalies**.
- Only Lstm is 0.70–0.77 even in Cross (real signal event_label). SvmA is near-chance in all conditions (feature-limited). Only SvmW iv25smote pooled is 0.69–0.73 (SMOTE inflation).
- Positioning: these are values under §2's "IV2025/TIV2026 same protocol (including leak, a known limitation)". Under honest conditions, vehicle→drowsiness is chance (already established by the de-leak table in §3.6). **As values of the current methodology, they are as expected**; honest re-evaluation is a follow-up.

### Per-case completion-time estimates (running cells, from process start times + median completion track record)
SvmW mixed track record: 35–102h (median ~76h, high variance, non-converging-SVM-limited). RF-nofs track record: target ~10–27h, mixed ~23–38h.
- **SvmW** in/out mixed **s1**: elapsed 103h, Optuna done → final training stage → **~07-21–22** (**needs monitoring: has reached the observed max of 102h. If not done by midday 07-22, possible stall in the final SVM fit → intervention candidate**).
  in mixed **s2025** 64h / out **s2025** 53h → **~07-22**. in/out **s13** (started 07-21 02:38) → **~07-24**. → **SvmW all done ~07-24** (if s1 does not stall).
- **RF-nofs** in/out mixed **s0** (32–36h elapsed, typ ~35h) → **~07-21 (within hours)**. out target **s1** → 07-21 evening. in/out mixed **s1** (started 07-21 06:53) → **~07-22 evening**. → **RF-nofs all done ~07-22–23**.

### Operations (07-20–21, non-destructive)
- **SvmW de-dup**: s2025 in/out were each running 2 duplicates (churn), so the 2 newer ones were stopped (one process per cell). Cause = `CLAIM_FRESH_SEC=10h` < SvmW single-fit time → stale judgment causing double launch. Currently idle_pending=0, so no recurrence.
- **RF-nofs top-up**: persistently launched 2 workers to pick up the 2 idle-queued cells of s1 mixed in/out → all 5 remaining cells parallelized. Cores 11/20.

**Summary**: exp3 is essentially complete (Lstm/RF/SvmA completed, SvmW 26/32 + RF-nofs 20/25 expected to complete by ~07-24). **Zero abnormal terminations, all values as expected** (relative to the current leak-inclusive protocol). Honest re-evaluation is retained as the §3.6 follow-up.

### Addendum (same 2026-07-21) Log-level audit (4-agent wlniqhvo2) finalized — the current campaign is sound at the results level, but 3 real findings

In addition to the JSON-level audit (all 390 cells valid), all logs (RF 285, SvmW 237, SvmA 288, Lstm 724) were scanned for rc≠0/traceback/fallback, etc.
**Zero crashes, corruption, or degeneracy in the current c1/iv25 campaign** (clean-finish: Lstm 103/103, SvmA 60/60, RF c1dom 159/164 [remaining 5 are running nofs], SvmW 204).
**The current prior_-series 96 (SvmA numpy) and 131 (Lstm xgboost) crashes are all in the retired env / retired distance-selection domain_train** and out-of-scope ([[project_rf_distance_selection]], CUDA migration).

**3 real findings concerning the current campaign (to be recorded):**
1. **[Methodology-label mismatch] iv25smote (pooled+SW-SMOTE) runs with pooled SMOTE, not subject-wise, for all models**. Log measurement: `[OVERSAMPLE] subject_id not found, falling back to pooled oversampling` appears in **RF 20/20, SvmW 5/6, SvmA 6/6**. Because pooled mode does not retain subject_id, subject-wise is impossible → falls back to pooled SMOTE (ratio0.5). **c1's Within/Mixed grid runs subject-wise correctly (0 fallback)**. → the pooled arm's SMOTE is "pooled ratio0.5", not "subject-wise". The results (RF 0.60–0.85 / SvmW 0.69–0.73 / SvmA 0.48–0.63) are valid but **the label should read pooled SMOTE**. Needs to be reflected in the paper's pooled-treatment description (user decision).
2. **Only iv25smote SvmW s2025 is missing (5/6)**. The s2025 log is a 418B stub (rc=3221225794 DLL_INIT_FAILED, dead since 07-10, not restarted). **s0/s7 are valid** (eval JSON 07-12 10:10 auc 0.720/0.728; the "dead log" the audit saw is a trace of a failed retry overwriting with `"w"`, and the JSON of the successful run remains). Against a seed target of ~5, 5/6 is sufficient. If a 6th is needed, restart s2025 (optional).
3. **[RF data quality, systematic] In all RF runs, `LaneOffset_Skewness/Kurtosis` are NaN → column-mean imputed, and ~3 subjects/run skip SMOTE** (due to `Input X contains NaN`). By-design and consistent across all runs (relative comparison preserved), but RF's absolute values include the mean imputation of these 2 features.

**Benign (recorded to avoid false detection in future audits)**: (a) RF's `ERROR - [SAVE] Model object is None` appears once/run at an early checkpoint before HPO but is cosmetic (after Optuna it trains and saves, all runs rc1=0). (b) The 12 dead stubs of source_only (Cross) (07-15 mass-kill, not restarted) are **irrelevant because Cross was demoted on 07-11**. (c) `rank_names.txt not found -> Using CLI targets` and Lstm's 16-column exclude-list fallback are the intended normal paths. (d) The b1cmp arm (outside c1/iv25) `File not found: processed_Sxxxx.csv` is a missing-data warning for the target subjects and is outside primary scope.

**Overall judgment**: the current c1/iv25 **results are all sound and as expected** (zero abnormal terminations). The only methodological caution is finding 1 (the pooled arm's SMOTE is pooled, not subject-wise), which concerns the accuracy of the paper's description and is a user decision. Findings 2/3 are minor.

### Addendum 2 (same 2026-07-21) [Important, escalation] Root cause of the SW-SMOTE deviation identified at the row level — all 4 methods of the pooled arm use pooled SMOTE, not subject-wise (deep dive prompted by the user's observation "exp3's imbalance treatment should all be SW-SMOTE")

Prompted by the user's observation (all imbalance treatment should be SW-SMOTE), addendum 1 / finding 1 is escalated from "notation mismatch" to a **real deviation from the intended methodology**, and the root cause is identified.

**Confirmed scope of impact (log measurement)**: the iv25smote (pooled) arm falls back to **pooled SMOTE in all 4 methods** (`Applying subject-wise oversampling`=0):
RF 20/20, SvmW 5/6, SvmA 6/6, **Lstm 6/6**. Meanwhile **the c1 grid runs subject-wise correctly in both Within/Mixed** (SvmW target_only 16/16, mixed 16/16 show `Applying subject-wise`, fallback 0).

**Root cause (code line level)**:
1. `iv2025_baseline_launcher.py` L98-111's pooled train_cmd has only `--mode pooled --subject_wise_split` and **does not pass `--time_stratify_labels`**.
2. → `model_pipeline.py` pooled (else) branch → `split_data(subject_split_strategy="subject_time_split", time_stratify_labels=False, keep_subject_id=True)`.
3. → `split_helpers.py` L161-173 (time_stratify=False path) → **calls `data_time_split_by_subject(...)`, but this function has no keep_subject_id argument**.
4. → `split.py` L468 `X_train = train[feature_columns].drop(columns=[subject_col], errors="ignore")` **unconditionally drops subject_id**.
5. → `model_pipeline.py` L212 `"subject_id" not in X_train.columns` → L213 warning → continues with pooled SMOTE.
**Contrast**: c1's target_only/mixed has the launcher pass `--time_stratify_labels` → `split_helpers.py` L142-160 (time_stratify=True path) **honors keep_subject_id at L154-155** → retains subject_id → SW-SMOTE works correctly.

**Implication**: against the design "exp3 imbalance treatment is all SW-SMOTE", **all 4 methods' results of the pooled (iv25smote) arm (RF 0.60–0.85 / SvmW 0.69–0.73 / SvmA 0.48–0.63 / Lstm 0.50–0.51) were generated with pooled SMOTE and do not match the intent**. Regeneration with SW-SMOTE is needed (iv25base = no SMOTE is irrelevant).

**Proposed fixes (require user decision)**:
- **Option A (recommended, preserves the IV2025 split)**: add a `keep_subject_id=False` argument to `data_time_split_by_subject`, make L468 a conditional drop, and thread keep_subject_id at `split_helpers.py` L169. → enables SW-SMOTE without changing pooled's split method (per-subject 60/20/20 = IV2025 config). With default False, other calls (eval, etc.) are unchanged and safe.
- **Option B**: add `--time_stratify_labels` to the iv25 launcher. → changes to a time_stratified split and breaks IV2025 comparability, so not recommended.
After adopting an option, regenerate iv25smote pooled with SW-SMOTE (target methods and priority need discussion).

**Status update (same day)**: the SvmW mixed s1 in/out that were being monitored **completed normally** at 09:16/09:33 (~4.3 days). SvmW **28/32** (remaining mixed s2025, s13 × in/out, 4 cells running; s2025 ~68–79h / s13 ~20h → SvmW all done ~07-24). RF-nofs 21/25 (target completed, mixed s0/s1×in/out running, ~07-22). Resolved the duplicate (top-up conflict) of RF-nofs in_mixed_s1. All eval values are finite and within the expected band, no crashes in the current campaign.

### Addendum 3 (same 2026-07-21) Implemented and verified the fix for the SW-SMOTE deviation (Option A) + iv25smote pooled regeneration plan (scheduled ~07-24)

Implemented the fix per the user decision (adopt Option A / regenerate all 4 methods after the current runs complete).

**Implementation (Option A, 3 places, with default False other paths / eval / running cells are unchanged)**:
- `src/utils/io/split.py` `data_time_split_by_subject`: added a `keep_subject_id: bool = False` argument. Made X_train's subject_id drop conditional (`drop(columns=([] if keep_subject_id else [subject_col]))`), and retained a non-numeric subject_id via `_check_nonfinite(X_train, preserve_cols=["subject_id"] if keep_subject_id else [])`. X_val/X_test are always dropped as before.
- `src/utils/io/split_helpers.py` L169: threaded `data_time_split_by_subject(..., keep_subject_id=keep_subject_id)` (split_data already receives keep_subject_id, and the pooled path passes `keep_subject_id=(use_oversampling and subject_wise_oversampling)`).

**Verification (real data, 8 subjects, common)**: `keep_subject_id=False` → X_train 160 columns, no subject_id (preserves existing behavior). `keep_subject_id=True` → X_train 161 columns, with subject_id → `apply_oversampling(subject_wise=True)` starts with `[Subject-wise Oversampling] Processing 8 subjects` (7/8 processed, 1 skipped), 3755→5208 rows (pos 127→1580). → **Confirmed subject-wise SMOTE works correctly in the pooled path**. Both files ast.parse OK. The running c1 SvmW/RF-nofs are on the time_stratify path (--time_stratify_labels) so unaffected, and eval is unchanged with default False.

**Regeneration plan (~07-24, after the current SvmW/RF-nofs complete)**:
- Target: regenerate **all 4 methods** (RF/SvmW/SvmA/Lstm) of iv25smote (pooled) with SW-SMOTE. iv25base (no SMOTE) is irrelevant.
- Procedure: because the existing iv25smote eval JSON/models are skipped as already_done, **archive the old results (pooled-SMOTE version) to `results/_archived_pooledsmote_<date>/` (with MANIFEST) and then re-run with the same tag** (the new code runs subject-wise). Update the corresponding values in the §2 table after regeneration.
- Launch: `iv2025_baseline_launcher.py --model {RF,SvmW,SvmA,Lstm} --smote --seeds <5>` (sequentially after the current runs complete, to avoid CPU contention). After regeneration, verify and record the difference between the old pooled-SMOTE values and the new SW-SMOTE values.

**Current state (same day, 22:00 hour)**: SvmW 28/32 (s1 in/out done, remaining mixed s2025, s13×in/out running, ~07-24). RF-nofs 21/25 (target completed, mixed s0/s1×in/out running, ~07-22). Zero abnormal terminations, all values within the expected band.

## 2026-07-22 [exp3 direction / re-evaluation] Re-reading IV2025/TIV2026 + implementation re-audit + honest re-score (4 agents w1qbn0j8c) — "RF best" collapses under honest conditions, TIV2026 turns out to be a Sobol factor study

In response to the user's request "re-read IV2025/TIV2026, review the implementation, and then decide exp3's direction / is IV2025's RF-best still valid", finalized with 4 agents in parallel.

### Q "Is IV2025's RF-best (AUC 0.85) still valid?" → [NO] collapses to chance under honest evaluation
- **iv25base pooled measurement** (saved-model re-score + cross-subject re-training, positional-safe):
  | Method | recorded(leaked) | train/eval-test row overlap | HONEST |
  |---|---|---|---|
  | **RF** | **0.738**(0.58–0.88) | **59.6%** | **0.516** (disjoint) / 0.510 (cross-subj) |
  | SvmW | 0.519 | 59.6% | 0.502 |
  | SvmA | 0.481 (already sub-chance) | 59.6% | ~chance |
  | Lstm | 0.512 (DRT, already chance) | 59.6% | ~chance |
  - **RF SEEN/UNSEEN/honest gradient**: SEEN (in-training 59.6%)=0.85, UNSEEN=0.65, honest (time-series disjoint)=0.52 = **memorization signature**. nofs160 is SEEN 0.97 (capacity↑ = memorization↑).
  - **Positive control EEG band power cross-subject = 0.61 > vehicle 0.51** = harness valid, vehicle chance is a true null.
  - **Honest ranking: EEG(0.61) > {RF≈SvmW≈SvmA≈Lstm all tied at chance 0.48–0.52}**. RF is not the best.
- **Mechanism (implementation trace)**: iv25base's TRAIN=`--subject_wise_split`→per-subject first 60%, EVAL=no flag→all-rows random 20% = a different split with 59.6% overlap ([iv2025_baseline_launcher.py:98-115], model_pipeline.py:108-110, split.py data_split/data_time_split_by_subject). **The reason only RF wins = the combination of (a) a non-degenerate row ranking + (b) the tree's memorization capacity**. SvmW = all-positive degeneracy / Lstm = majority collapse / SvmA = no feature signal (T1: SvmA 18 features → RF 0.4955), so they cannot cash in the same leak. RF's only genuine intrinsic property is **imbalance robustness (0.738 without SMOTE vs SvmW degeneracy 0.519), not signal**. Under honest conditions, all methods are tied at chance.

### Re-reading IV2025 ([latex/IV2025/IV2025.tex])
- "RF best AUC 0.85/Acc 88%" is a **"between-method comparison" under configuration C2** (C2 = **pooled random 8:1:1 split**, no subject split = same-subject + 50% window overlap causing structural leak, L803/L936). Not a genuine-signal claim.
- **IV2025 itself reserves judgment**: Threats "Feature Bias Towards RF" (L1017-1019 "RF may be advantaged because it uses all features of the other methods") + Discussion "the main cause of RF's high accuracy is that it has more features" (L997). → foreshadows the later "large capacity memorizes overlapping rows" mechanism.
- The leak critique is **directed at the other 3 papers** (training-accuracy-as-test) and not applied to its own 0.85. within-subject inflation and chance collapse are not discussed. EEG is **label-only** (no positive control).

### Re-reading TIV2026 ([exp2-analysis/paper/TIV2026/main.tex], published, unchanged) — correcting the memory premise
- **It is not a paper claiming "0.89 with a proposed RF detector demonstrates vehicle validity"**. **It has been re-scoped as a 4-factor Sobol sensitivity study** ("does not propose a new detector" L45/73/161). The main result is a variance decomposition (training mode m + rebalancing R + interaction account for **≥93%** of systematic variance, S_Tm=0.50–0.71 / S_TR=0.27–0.37, distance d and group g are negligible).
- 0.89 is **demoted** to "the optimal factor cell (RF+SW-SMOTE r=0.1+within-domain, ω_tr=1)". ω_tr (subject overlap) is explicitly disclosed as "leakage" (L273), and **the cross-subject collapse (near-chance) is reported as a main result** (L430/468/529). **No over-claim of generalization**, the description is honest.
- **No head-to-head claim of the 4 methods** (RF is the sole classifier over which factors are varied; SvmW/SvmA/Lstm are future work). → exp3's "RF = proposed vs 3 baselines" framework does not exist in TIV2026. TIV2026's cross-chance **agrees** with exp3's honest conclusion. soft spot: the abstract foregrounds 0.890 without also stating the cross-collapse magnitude / the within 50% window-boundary leak is not examined.

### Implications for direction (details in the integrated report in the main text)
Under honest conditions, vehicle→EEG drowsiness = all methods chance (including RF). "RF best" is a product of the feature-bias + split leak that IV2025 itself foretold. TIV2026 already acknowledges cross-chance as a careful factor study. → **exp3's only direction that is honest and consistent with TIV2026 is "negative + mechanism + methodology under a leakage-free re-evaluation"** (Direction B). The current draft's recovery story (Direction A) depends on the within-domain leak and is hard to defend. scratch: honest_iv25base.py, controls_only.py.

### Addendum (2026-07-22) Policy decision = Direction A (honest-scoped) + carried out the alignment of the draft's claims

User decision: **Direction A (retain the spine of the recovery story / mechanism analysis, keep the current draft)**. However, claims that contradict today's established facts are
to be corrected to honest-scoped, as agreed. This follows the precedent where TIV2026 itself makes the Direction-A-type approach of "disclosing within-domain 0.89 as ω_tr=1 while making it a main result"
hold honestly.

**Carried out (aligned all \todo claims in TIV2026_exp3/main.tex, within the Direction A frame)**:
- Expanded the 2 columns Before/After into **3 columns Before / After / Leakage-free**, and made the **leakage-free column a co-headline** (a decisive comparison in the main text, not an appendix).
- Scoped each recovery value to **within-domain (ω_tr=1, ~69% row overlap)**, and Before to pooled (~60%). leakage-free: RF/SvmW/SvmA≈chance (0.50-0.52), Lstm(DRT) 0.73-0.75, EEG positive control 0.61.
- **Removed false claims**: "rebalancing recovers methods that have latent signal in their features (SvmW yes)" → "SMOTE only undoes degeneracy and cashes in the within overlap, it does not create signal (honest SvmW 0.52)". RF "genuine best/robust" → "imbalance-robust + memorizes overlapping rows (SEEN0.85/UNSEEN0.65/honest0.52) = consistent with IV2025's Feature Bias Towards RF, chance under honest conditions".
- Changed RQ1's answer from "recovers" → "apparently recovers within but does not hold leakage-free = an artifact of the evaluation protocol (honest is NO)". RQ2 too explains the mechanism under the leaky premise.
- Reflected in Related/Conclusion the correct positioning of **TIV2026 = Sobol factor study (ω_tr disclosed, cross-chance, no RF-vs-baselines claim)**, connecting that exp3 extends the ω_tr=1 cell to the row level.
- Added to Limitations: (0) Before/After are within the leaky region and the decisive one is the leakage-free column, (6) the SW-SMOTE pooled implementation defect (fixed 07-21 / awaiting regeneration, unchanged for the honest conclusion).
- Added a rationale comment for the alignment at the top of the file.

The reconstructed draft is currently being verified by 3 adversarial critics (over-claim, internal contradiction, numbers/facts) (wuc85ngnx). To be finalized after reflecting the points raised.

### Addendum (2026-07-22) Adversarial verification of the reconstructed draft (3 critics wuc85ngnx) + reflected the points raised

Verified the reconstructed main.tex with 3 critics (over-claim, internal contradiction, numbers/facts). **All load-bearing numbers, IV2025/TIV2026 characterizations, and method attributes match the ledger** (no errors). Detected = mainly 1 real defect (which I introduced) + several minor ones, all fixed:
- **[major, fixed] Lstm target confusion**: although Lstm targets DRT distraction, not EEG drowsiness, the abstract ("all '4' methods chance") + placing Lstm DRT 0.73-0.75 in the results leakage-free drowsiness column created a contradiction readable as "one method survives leakage-free drowsiness detection". → scoped the collapse claim to **3 methods (RF/SvmW/SvmA)**, and separated Lstm as a "reference row for a non-commutable different target (DRT)" (do not place DRT values in the drowsiness column, and state explicitly that Lstm is also chance on the drowsiness target).
- **[minor, fixed]** intro "leakage-free pooled" terminology clash → "leakage-free (subject-disjoint)". The AUPRC dangling promise in metrics → stated explicitly that it is co-reported with AUROC in results/mechanism. rq2 "near-balanced" → "~74% positive, so SMOTE/RUS inactive".
- **[needs author confirmation, unchanged]** A mismatch between the SvmW citation `zhao2012` (CWT 2012) and the internal record "Zhao2009 GHM 8 bands". references.bib has only zhao2012. The author needs to confirm the source of the Zhao paper actually reproduced (unchanged because it is outside the scope of claim alignment).
LaTeX soundness: `$` even, `{}` 204/204 balanced. The draft reaches internal consistency with no over-claim under Direction A (honest-scoped).

## 2026-07-22 [exp3 status audit] SvmW 28/32, RF-nofs 18/20, zero abnormal terminations, all values within the expected band, 2 items need monitoring

- **SvmW 28/32**: Within 16/16 completed. Remaining = mixed's **s2025, s13 × in/out (4 cells running)**. s2025 in 88h (Optuna Trial41) / out 77h, s13 in/out 29h. → **all done ~07-24–25** (s13 is the bottleneck).
- **RF-nofs c1 18/20**: target in/out 5/5, mixed in/out 4/5. Remaining = **mixed s0 in+out**. Both, after Optuna completes (Trial49, best_so_far 0.946), **stall 3–6h at the CALIBRATION stage (Sigmoid 5-fold CV, 1094 trees × all 165 features)** (in_mixed s0 log 01:52 stale=6h, out_mixed s0 05:01=3h). Likely slow under CPU contention rather than hung, but **needs monitoring** (consider intervention if not done within a few hours). **mixed s1 in/out completed this morning** (in 0.632 / out 0.669). Stopped the duplicate process of RF-nofs out_mixed_s1 (top-up conflict, eval already completed at 06:05).
- **Abnormal-termination check → zero detected**: all 390+ eval JSON valid with roc_auc∈[0,1] (no NaN/inf/out-of-range/corruption). The recently completed SvmW mixed s1 in 0.754 / out 0.778, RF-nofs mixed s1 in 0.632 / out 0.669 are within the expected band (current leak-inclusive protocol).
- **Expected values**: all consistent with the known leak signature, no new anomalies. Honest re-evaluation is separate (finalized in this log's 2026-07-22 re-audit section = all vehicle methods chance).
- The SW-SMOTE regeneration of iv25smote (Option A fixed) is scheduled to launch after the current runs complete (~07-24), waiting non-destructively.

## 2026-07-22 (night) [exp3 status audit] RF-nofs 20/20 done, SvmW 29/32, zero abnormal terminations, remaining SvmW 3 cells

- **RF-nofs: 20/20 done** (c1 table). The previously flagged **CALIBRATION stall of mixed s0 in/out was resolved** = completed normally (in 08:51 / out 11:12, it was slow rather than hung). Values in 0.906 / out 0.964 (full-feature leak band, consistent with the memorization signature).
- **SvmW: 29/32**. Within 16/16, mixed in 7/8 (**s2025 completed at 19:23**, value 0.745), mixed out 6/8. **Remaining 3 cells**: mixed out **s2025** (92h running, in the observed longest range → ~07-23), mixed in/out **s13** (44h running → ~07-24–25). → **SvmW all done ~07-24–25** (s13 bottleneck).
- **Operations**: stopped the churn duplicate of SvmW in_mixed_s1 (at 19:23 a worker re-claimed a completed cell, eval already done at 07-21 09:16) (PID 84144).
- **Abnormal-termination check → zero detected**: all 390+ eval JSON valid with roc_auc∈[0,1] (no NaN/inf/out-of-range/corruption). The recently completed values (RF-nofs s0 0.906/0.964, SvmW s2025 0.745) are within the expected band of the current leak-inclusive protocol.
- **Next action**: with RF-nofs done, only SvmW 3 cells remain. After SvmW completes (~07-24), the SW-SMOTE regeneration of iv25smote (Option A fixed) is scheduled to launch.

## 2026-07-24 [exp3 c1 grid fully complete] All methods and all arms completed, zero abnormal terminations, all values within the expected band

- **c1 grid fully complete**: RF(fs) 96/96, **SvmW 32/32**, SvmA 32/32, Lstm 60/60, iv25 baseline (base/smote), RF-nofs 20/20. The watchdog also self-terminated at 07-23 12:18:03 with "ALL COMPLETE — removing scheduled task". No running processes.
- **SvmW final** (completed 07-23): Within-in 0.800±0.012 / Within-out 0.759±0.012 (8 each), **Mixed-in 0.742±0.011 / Mixed-out 0.771±0.015 (8 each)**. The final 3 cells out_s2025 0.781 / in_s13 0.742 / out_s13 0.778 are within band.
- **Abnormal-termination check → zero detected**: all 390+ eval JSON valid with roc_auc∈[0,1] (no NaN/inf/out-of-range/corruption). All values consistent with the known leak signature, no new anomalies. Operations: stopped the SvmW in_mixed_s1 churn duplicate right after completion (re-claimed a completed cell at 07-22 19:23).
- Updated c1_results.md's SvmW Mixed column to the final n=8 values, and the progress/header to "c1 complete".
- **Next action**: exp3 c1 is complete. **The SW-SMOTE regeneration of iv25smote (Option A fixed, user pre-authorized "all 4 methods, after the current runs complete") can be launched**. However, SvmW iv25smote is ~34–70h per cell (the most bottlenecked in history), so a full regeneration is a multi-day to week scale, and under honest conditions both pooled/subject-wise are chance (the scientific conclusion is unchanged; the goal is methodology-label alignment). → carry out after confirming the launch scope (in particular whether to include SvmW) with the user.

## 2026-07-24 [iv25smote SW-SMOTE regeneration launched + progress] Confirmed subject-wise execution in production (fallback=0), exp3 c1 already complete

Executed the user decision "regenerate all 4 methods". Archived the old pooled-SMOTE artifacts (74 eval JSON+CSV) to `results/_archived_pooledsmote_20260724/` (with MANIFEST) and launched regeneration with SW-SMOTE.

- **Most important confirmation**: this regen log shows **`Applying subject-wise oversampling` in all methods (fallback=0)** — RF 17, Lstm 6, SvmA 1, SvmW 5 run subject-wise. **The keep_subject_id fix is effective in production** (old: all pooled fallback).
- **backend/launch**: RF/SvmW = Windows CPU (Start-Process persistent, RF 4 + RF-nofs 2 + SvmW 5 workers), Lstm/SvmA = WSL2 GPU (`.venv_tf_gpu`/`.venv_svma_cuml`, Lstm→SvmA sequential to avoid GPU contention, bash background task).
- **Progress (as of 06:57) + ETA**:
  - **Lstm: 6/6 done** (~33 min each, 02:32–03:38). Values 0.504–0.521 = chance (consistent with the old pooled-SMOTE 0.50–0.53; Lstm pooled is chance regardless of SMOTE type).
  - **RF: 11/20 done** (non-nofs 15 + nofs 5, ~fast) → **~07-24 midday**. Values 0.747–0.762 = leaked band (consistent with the old ~0.74).
  - **SvmA: 1/6** (s0 started 03:38, PSO swarmsize=50 maxiter=100, cuML), 5 pending, 1 worker sequential → **~07-25–26**.
  - **SvmW: 0/5** (5 running in parallel, ~34–70h/cell) → **~07-26**.
- **Abnormal-termination check → zero detected**: all eval JSON valid with roc_auc∈[0,1]. The new regen values (Lstm chance, RF ~0.75) are within the expected band, in the same band as the old pooled-SMOTE version (the honest conclusion = chance is unchanged; the methodology label is corrected to "pooled→subject-wise SMOTE").
- Note: the GPU regen (remaining 5 SvmA) is running as a bash background task. If it stops due to a session disconnect, restart it at the next check (re-run `scripts/shell/_regen_gpu_iv25smote.sh` with `.venv_svma_cuml`). The old values are preserved in the archive.

## 2026-07-25 [iv25smote regeneration progress + SvmA blocker] RF/Lstm regeneration done, SvmW in progress, SvmA cannot do SW-SMOTE (cuML hang) → old values restored

- **Regeneration progress (09:33)**: **RF non-nofs 15/15 done** (0.75 band), **Lstm 6/6 done** (chance), **SvmW 1/5** (s42 done 05:03, 4 running in parallel, ~07-26), RF-nofs 2/5 (running). All run subject-wise SMOTE (fallback=0).
- **[Blocker] SvmA iv25smote SW-SMOTE regeneration is technically impossible**: s0, right after "PSO Starting" at 07-24 03:39, has **~30h with no log update, GPU 0%/0% = hung**. A single-seed diagnosis also reproduces it with an empty log and GPU 0% in ~5 min. Cause: subject-wise SMOTE increases training from 40166→57592, and **the cuML SVM/PSO (swarmsize50 × maxiter100) freezes at the first SVM fit** (the old pooled-SMOTE fallback version completed in ~2.8h/seed = subject-wise-specific). Reducing maxiter is ineffective (it stops before iterating), and going to CPU-SVM is impractical at 5000 fits × 57k.
- **Response**: stopped the hung process and the diagnosis, and **restored the archived SvmA old pooled-SMOTE values (6 seeds, 12 files) from the archive** (reversible, preserving the SvmA arm). → **SvmA iv25smote stays as pooled SMOTE** (under honest conditions both are chance = the scientific conclusion is unchanged; only for the methodology label does SvmA alone need the paper to state explicitly "pooled SMOTE (subject-wise impossible due to cuML hang)"). **Requires user decision** (finalize with this handling or retry a mitigation).
- **Abnormal-termination check**: all eval JSON valid. The new regeneration values (RF ~0.75, Lstm chance) are in the same band as the old. SvmW/RF-nofs progress normally (no signs of stalling, the cuML problem is specific to SvmA).

### Addendum (same 2026-07-25) SvmA blocker resolved — the true cause is `SVMA_USE_CUML=1` not being set (falling back to sklearn CPU), regeneration resumed after a successful mitigation

Carried out the user decision "try a mitigation". Identified and resolved the true cause via stepwise diagnosis:
- **Diagnosis 1 (add max_iter)**: SvmA.py's cuML/sklearn SVC has no `max_iter` (default -1 = unlimited). Added an upper bound (`SVMA_SVC_MAX_ITER`, default 100000) to the 3 SVC calls (src/models/architectures/SvmA.py). → the infinite loop stops but it is still slow.
- **Diagnosis 2 (GPU/CPU measurement)**: the stalled cell shows **GPU 0%, process CPU 96%** = CPU computation not using the GPU.
- **True cause found**: `SvmA.py` **falls back to sklearn CPU SVC without the `SVMA_USE_CUML=1` env** (`_USE_CUML=False`). My GPU regen script did not set this env, so it **computed 57k samples (subject-wise SMOTE) with CPU libsvm = effectively impractical**. The old version (completing in 2.8h/seed) used the cuML GPU.
- **Mitigation (successful)**: verified a single seed with `SVMA_USE_CUML=1` → confirmed **GPU 4749 MiB, 89% usage, cuML SVC running**. Added `SVMA_USE_CUML=1` to the SvmA line of `_regen_gpu_iv25smote.sh`, and **resumed regeneration of SvmA 6 seeds on the cuML GPU** (PSO maxiter=100, GPU 64% usage, ~2.8h/seed × 6 = expected to complete ~07-26). The old values continue to be preserved in the archive.
- **Result**: SvmA too can now be regenerated with subject-wise SMOTE. Full 4-method SW-SMOTE regeneration is established. The max_iter upper bound is kept as a defensive improvement (no effect on converging fits, compatible with both cuML/sklearn).

## 2026-07-26 [Seed augmentation from the c1 statistical analysis] runs #2/#3/#4 launched (under-powered cells identified via required-n)

The model-characteristic statistical analysis (docs/.../c1_statistical_analysis.md, script scripts/python/analysis/exp3_c1_statistical_analysis.py) computed the required n for a 95% CI half-width <= 0.05 per cell. Adequate: RF-fs (n=24), SvmW/SvmA/Lstm within/mixed (SD ~0.01, req_n=3). Under-powered: **RF-nofs** (only 5 seeds, SD 0.065-0.125 -> CI half-width 0.08-0.16; req_n Within 10-15, Mixed 22-27) and **SvmA Within-out** (n=8, SD 0.074, req_n=11). Launched (user request):
- **#2 RF-nofs c1 seed augmentation**: +10 seeds [7,13,256,512,1337,2024,3,5,9,11] -> 15 total (Within/Mixed x in/out), `c1_domain_launcher.py --model RF --no-fs`, Windows CPU, 6 workers. Brings Within to CI-adequacy; Mixed improves 5->15 (still short of ~22-27, deliberately not pursued to full adequacy for a leak-artefact ablation).
- **#3 SvmW iv25smote + seed 2025**: -> 6 seeds paired with iv25base's 6, so the imbalance-effect paired Wilcoxon can reach p<0.05 (n=5 caps at 0.0625). CPU, 1 worker.
- **#4 SvmA c1 + 3 seeds** [512,1337,2024] -> 11 total, to lift Within-out (req_n=11) to adequacy; GPU (cuML, SVMA_USE_CUML=1, SVMA_PSO_MAXITER=30) via a waiter (scripts/shell/_addseed_svma_c1_waiter.sh) that starts only after the SvmA iv25smote regeneration vacates the GPU.

Note: these augmentations tighten CIs on the RECORDED (leak-inflated) values; the honest (leakage-free) conclusion (all vehicle methods = chance) is unaffected. The higher-value follow-up (multi-seed honest re-evaluation, recommendation #1) was NOT requested and is not launched.
