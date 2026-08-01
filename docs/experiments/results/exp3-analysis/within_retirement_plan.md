# exp3 — Migration of the Recorded-Value Analysis to the Pooled/Mixed Regimes

**Status: OPEN.** Part A is a re-analysis of existing results and can be done now. Part B needs
new probe runs. The TIV2026_exp3 manuscript is blocked on Part B and will be revised in one pass
once the new values land.

## 1. Decision and rationale

The recorded-value analysis currently reports six modes: `pooled_base`, `pooled_smote`,
`within_in`, `within_out`, `mixed_in`, `mixed_out`. **The two `within_*` modes are being retired
from the analysis and from the paper**, joining `cross`, which was already out of scope.

Rationale: `within` trains on the target group only, i.e. it deliberately discards the other
group's training data. That is not an operating point anyone would deploy. The regimes that
correspond to real use are

- **pooled** — train on everything available, evaluate on the population, no domain grouping; and
- **mixed** — train on all 87 recordings, evaluate on a specific domain subgroup.

Everything the paper claims should therefore be measured in those two regimes.

### 1.1 A consequence that must not be lost

Dropping `within` does **not** make the headline "RF is best in the realistic regimes" true as
stated. In the recorded values, RF (top-10) leads **pooled only**:

| Mode | 1st | 2nd | 3rd | 4th | 5th |
|---|---|---|---|---|---|
| Pooled-base | **RF 0.738** | the other three are indistinguishable in the chance band | | | |
| Pooled-SW-SMOTE | RF-nofs 0.855 | **RF 0.795** | SvmW 0.684 | SvmA 0.569 | Lstm 0.513 |
| Mixed-in | RF-nofs 0.829 | Lstm 0.782 | SvmW 0.742 | **RF 0.719** | SvmA 0.532 |
| Mixed-out | RF-nofs 0.891 | Lstm 0.779 | SvmW 0.771 | **RF 0.749** | SvmA 0.597 |

In both mixed modes the top-10 RF is **4th of 5**. Setting Lstm aside as non-commensurable (DRT
target), **SvmW still beats RF in both** (0.742 vs 0.719; 0.771 vs 0.749). Only RF-nofs leads
mixed, and that is this study's own full-feature ablation, not a prior method — and it is the
most dispersed variant in the study (SD 0.112–0.125 at n=5).

So after the migration the defensible claim is still regime-scoped: **RF is the only method that
works pooled, and it is the only method that domain restriction does not move.** Its advantage is
insensitivity to the treatments, not a higher ceiling. Any draft that claims RF is best in mixed
is refuted by the SvmW row above.

## 2. Part A — re-analysis only (no new training)

All of these come from the recorded eval JSONs already on disk, via
`scripts/python/analysis/exp3_c1_recorded_value_analysis.py`. No models are retrained.

| Section | Change | Script location |
|---|---|---|
| §A table | drop the `Within-in` / `Within-out` columns | `MODES` (L33) |
| §B Kruskal–Wallis | drop the two within rows; rankings recomputed per remaining mode | `MODES`, `core` (L199) |
| §C contrasts | **replace** `domain_restrict(pooled->within)` with `pooled_smote -> mixed_in`, and `domain_shift(in->out) within` with `mixed_in -> mixed_out` | contrast list (L153–154) |
| §D1 feature count | keep the `mixed_in` / `mixed_out` rows and `Pooled-SW-SMOTE`; drop the two within rows | `core` |
| §D3 seed stability | drop the two within rows | `core`, fig4 mode list (L251) |
| §E Scheirer–Ray–Hare | design becomes 5 methods × 2 modes (`mixed_in`, `mixed_out`) × common seeds; Mode df 3→1, interaction df 12→4 | `core`, `scheirer_ray_hare` |
| fig1, fig4 | regenerate without the within modes | L215, L234, L251 |

Note on §E: with only two modes left the "the domain sub-modes do not differ" result rests on a
much weaker design than the current 4-mode one. Report the new df and p honestly; if the mode
effect is no longer estimable in a useful way, say so rather than restating the old conclusion.

## 3. Part B — new probe runs required

Three probes are currently measured **within-in-domain** and must be re-measured under
**mixed-in** (mixed-out optional but preferred, so that both target groups are covered). Keep the
probe harness otherwise identical — same plain learners, same fixed hyperparameters, same
simplified split, same seed count — so that only the mode changes.

### B1. Feature-count dose–response (report §D1b, Fig. 5)

- **Current**: within-in-domain, plain RF with fixed hyperparameters, 3 seeds, recorded
  evaluation. Sweep k ∈ {5, 10, 20, 40, 80, 120, 165}, two selection orders — top-k chosen
  *after* SW-SMOTE (the c1 pipeline order) and top-k chosen on the natural training data.
- **Needed**: identical sweep under `mixed_in`.
- **Recorded anchors change**: the current figure anchors on RF-fs k=10 = 0.746 and RF-nofs
  k=165 = 0.874 (within-in). Under mixed-in they become **0.719** and **0.829**; under mixed-out,
  **0.749** and **0.891**.
- **Load-bearing claim**: the gain saturates by k ≈ 20, and the low top-10 arm is mostly an
  artifact of selecting features on SW-SMOTE-oversampled data rather than of ten being too few.

### B2. SvmA learner-versus-feature-set probe (report §D4, Fig. 6)

- **Current** (within-in-domain, recorded evaluation):

  | Feature set → learner | AUROC |
  |---|---|
  | SvmA 36 steering → RF | 0.884 |
  | SvmA 36 steering → RBF-SVM | 0.597 |
  | all 165 vehicle → RF | 0.873 |

  and, on the 36 steering features, imbalance treatments raw / SW-SMOTE / class-weight giving
  RF 0.878 / 0.883 / 0.877 against RBF-SVM 0.597 / 0.582 / 0.572.
- **Needed**: both panels under `mixed_in`.
- **Load-bearing claim**: SvmA's bottom rank is a learner ceiling, not a feature-set deficit and
  not an imbalance artifact. The RF-versus-RBF-SVM gap on identical features is the whole result.

### B3. SvmW wavelet-feature probe (report §D5, Fig. 7 panel B)

- **Current** (within-in-domain, recorded evaluation): SvmW's own 8 GHM steering-wheel wavelet
  band energies fed to RF give ≈0.87 under every imbalance treatment — raw 0.871, SW-SMOTE 0.870,
  class-weight 0.873.
- **Needed**: same three bars under `mixed_in`.
- **Load-bearing claim**: a learner that does not degenerate reads the recorded signal from these
  features with no rebalancing, so SvmW's pooled collapse is a learner–imbalance interaction
  rather than a feature deficit.

## 4. What is unaffected

- **The SvmW degeneracy evidence (§D2, §D5 first table) is measured under Pooled**, not within:
  probability SD 0.001 → 0.199, specificity 0.004 → 0.504, predicted-positive rate 0.997 → 0.506,
  AUROC 0.519 → 0.684. It survives the migration unchanged and needs no re-run.
- The Lstm characterization is pooled-versus-domain-restricted; the imbalance leg
  (Δ = +0.005, p = 1.0) is pooled-only. Only the domain leg is re-anchored to mixed by Part A.
- `cross` stays out of scope.

## 5. Acceptance criteria

The probes are contrasts between learners and between feature counts, so they are expected to
reproduce under mixed. They are being re-run to remove the within regime, not because the result
is in doubt — but the following would change the paper's claims and must be reported if seen:

- **B1**: if the saturation point moves materially away from k ≈ 20, or the two selection-order
  curves no longer converge, the RF feature-count claim is rewritten rather than rescoped.
- **B2**: if the RF-versus-RBF-SVM gap on SvmA's 36 features narrows substantially, the "learner
  ceiling" conclusion weakens and the SvmW/SvmA dissociation — the sharpest result in the paper —
  has to be softened.
- **B3**: if RF no longer reads ≈0.87 from SvmW's 8 wavelet features, the "not a feature deficit"
  half of the SvmW conclusion is lost.

## 6. Deliverables when Part B completes

1. Re-run Part A and regenerate `c1_recorded_value_analysis.md` §A–§E and figures 1–4 without the
   within modes.
2. Replace the §D1b, §D4, §D5 tables and figures 5–7 with the mixed-in measurements.
3. Update the manuscript in `TIV2026_exp3/` in a single pass: drop the within columns from
   Table III, re-anchor the domain-restriction contrast to pooled → mixed, and re-point the
   mechanism subsections at the new probe values. The regime-scoped headline of §1.1 above is
   what the manuscript should end up claiming.
