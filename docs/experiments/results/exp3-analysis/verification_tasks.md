# exp3 Verification / Analysis Tasks (run on the compute PC)

*Created: 2026-06-27. Intended to be run on the analysis machine (WSL2/HPC, with `data/processed` + eval JSON available).*
*Background: the verifications and figures still to be done, to confirm that the discussion of each method (rescued / not-rescued / robust / different-target) is "correct as we expect".
The rationale is in [`domain_imbalance_factor_analysis.md`](domain_imbalance_factor_analysis.md).*

---

## P0 (top priority — determines the validity of the main conclusions)

### T1. Re-test SvmA with the full Arefnezhad feature set (resolving the faithfulness bug)
**Problem (confirmed):** SvmA's feature filter does not match the 18 features of Arefnezhad (2019).
- Filtering is done via `SVMA_PAPER_FEATURE_SUFFIXES` (14 types, [`SvmA.py:65-70`](../../../../src/models/architectures/SvmA.py#L65)) ([`SvmA.py:664-684`](../../../../src/models/architectures/SvmA.py#L664)).
- **Missing (8 types)**: Sample Entropy, Katz Fractal Dimension, Shannon Entropy, Spectral Flux, Frequency Variability, Q1/Q2/Q3.
- **Extra (4 types)**: Mean, Variance, Max, Min (not in Arefnezhad).
- **Critical**: Sample Entropy is 2 of the 5 final selected features in the original paper (I11ᵃ, I11ᵛ). Katz FD and Shannon Entropy are also mainstays.
- These are **already computed in `simlsl.py`** (SampleEntropy `:215`, KatzFractal `:207`, ShannonEntropy `:208`, SpectralFlux `:211`, Quartile `:202/204`), yet **SvmA discards them via the filter**. Only Frequency Variability is uncomputed (0 hits).

**The current "no signal" (univariate 0.515 / multivariate 0.509) is a value obtained with a feature set that excludes the original paper's mainstay features**
→ this is not a faithful reproduction of Arefnezhad. As it stands, this null is not airtight.

**What to do:**
1. Fix `SVMA_PAPER_FEATURE_SUFFIXES` to the 18 types of Arefnezhad
   (add: SampleEntropy, KatzFractal, ShannonEntropy, SpectralFlux, Q1/Q2/Q3 / remove: Mean, Variance, Max, Min).
   - Confirm the exact column names against the processed CSV header. Frequency Variability is uncomputed, so add it to `simlsl.py` if needed.
2. Re-run:
   - **univariate directionless AUROC** (all features, especially Sample Entropy / Katz FD)
   - **multivariate RBF-SVM** (full feature set, subject-disjoint split)
   - **SvmA (including ANFIS+PSO selection)** under the B1 condition
3. **Expectation / decision**:
   - Still chance (<0.55) → "no signal" is **strengthened** (chance even with the original paper's features) → discussion confirmed.
   - Signal appears → the original null was a **product of the feature filter** → revise the discussion.

### T2. SvmW clean-split verification (is 0.79 an honest signal or split-dependent?)
**Problem:** SvmW's 8 wavelet bands give univariate 0.510 / **multivariate random-split 0.485 (chance)**.
Nevertheless, under B1 (target_only split) it is 0.79. → **suspicion that 0.79 depends on the B1 split structure (time/subject)**.
"The 8 bands carry a latent drowsiness signal" is currently unproven and leans toward refuted.

**What to do:**
- Under the same conditions as B1 (SvmW, in_domain, SW-SMOTE 0.3/0.5, same seed), change `split_data`'s
  `subject_split_strategy` to **subject holdout (subject-disjoint random split)** and obtain the AUROC.
  (Not the current `subject_time_split` = a single cut on subject-sorted order, but separating train/test by subject.)
- **Expectation / decision**:
  - 0.79 remains → the features carry a genuine (multivariate) signal → "carries a latent signal" OK.
  - Drops to chance → rewrite as "SMOTE recovers the decision function but merely rides structure specific to the within-domain regime".
- This also simultaneously decides the **explicit within-domain self-consistency** (consistency with the leakage critique).

---

## P1 (reinforces the main conclusions)

### T3. RF's SMOTE-only effect (1 cell of the 2×2)
- Run `pooled + SMOTE` and compare with `pooled + baseline` (= IV2025) → confirm the "small SMOTE effect" for RF without confounding.

### T4. SvmA classifier × feature deconfound
- **RF-on-SvmA-features** (train the SvmA 23/full features with RF) and **RBF-SVM-on-RF-features** on the same split.
- Expectation: RF-on-SvmA-features ≈ chance (the features are the wall) / SVM-on-RF-features ≈ 0.78 (the classifier is not the wall) → confirm that SvmA's null does not depend on the classifier/selection.

### T5. Pin down Lstm's domain attribution
- **Complete local before (IV2025 pooled)** → replace the published value 0.52 with a measured one (SvmA before likewise, expected ~7/1).
- Measure **Lstm cross-domain (ω=0)** → if within ≫ cross, confirm that "the improvement is domain-derived".

### T6. Faithfulness of road-curve removal (SvmA auxiliary)
- Arefnezhad Eq.1-2 (subtracting the sliding-window mean of steering to remove road geometry) is **not implemented**.
- If Aygun's course contains curves, steering may be contaminated by road shape. Either add the removal and re-confirm T1, or
  confirm that Aygun's course is straight / not applicable and note "not needed".

---

## Plots to generate (output on the data machine → for paper/slides)

| # | Plot | What it shows | Priority |
|---|---|---|---|
| 1 | SvmA per-feature univariate AUROC (bars): full 18 vs current 14 | whether the missing features (Sample Entropy etc.) carry signal → T1 | P0 |
| 2 | SvmW: target_only-split vs clean-split AUROC (seed box plot) | whether 0.79 is honest or split-dependent → T2 | P0 |
| 3 | predict_proba histogram: SvmW IV25 (constant spike) / B1 (spread) / SvmA B1 (both classes overlapping) | visual evidence of degeneracy → recovery | P1 |
| 4 | before/within/cross AUROC (grouped bars by method) | explicit within-domain (within recovery / cross collapse) | P1 |
| 5 | seed convergence: running-mean AUROC ±95%CI vs k (by method) | justification of the number of seeds (not saved in exp3) | P1 |
| 6 | confusion-matrix heatmap (by condition) | at-a-glance evidence of collapse / non-collapse | P2 |

---

## Discussion to be confirmed after completion (decision table)

| Discussion | Current status | Confirmed by T |
|---|---|---|
| RF: does not collapse under imbalance | ✅ confirmed | — |
| RF: small SMOTE effect | 🟡 confounded | T3 |
| SvmA: no signal in the features | ⚠️ **feature set unfaithful** | **T1** (+T4, T6) |
| SvmW: SMOTE recovers the decision boundary | ✅ confirmed | — |
| SvmW: 8 bands carry a latent signal | ❌ leans refuted | **T2** |
| Lstm: SMOTE ineffective because balanced | ✅ confirmed | — |
| Lstm: improvement is domain-derived | 🟡 direction only | T5 |

---

## References (code)
- SvmA feature filter: [`src/models/architectures/SvmA.py:65-70, 664-684`](../../../../src/models/architectures/SvmA.py#L65)
- Feature computation (missing features already computed): [`src/data_pipeline/features/simlsl.py:194-215`](../../../../src/data_pipeline/features/simlsl.py#L194)
- split switching: [`src/utils/io/split_helpers.py`](../../../../src/utils/io/split_helpers.py) (`subject_time_split` ↔ subject holdout)
- Mechanistic rationale: [`domain_imbalance_factor_analysis.md`](domain_imbalance_factor_analysis.md) §2, §9
