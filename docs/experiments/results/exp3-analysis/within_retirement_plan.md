# exp3 — Migration of the Recorded-Value Analysis to the Pooled/Mixed Regimes

**Status: IN PROGRESS (2026-08-01).** Part B is complete: its three mechanism probes have been
re-run under `mixed_in` and are reflected in Figures 5–7. Part A remains pending until the active
RF-nofs seed extension completes and the recorded-value aggregation can be regenerated. The
TIV2026_exp3 manuscript will be revised in one pass from that finalized public artifact set.

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

## 3. Part B — mixed-in probe runs completed

The three probes were re-run under **mixed-in** using the same plain-learner harness, fixed
hyperparameters, simplified split, and seed count. Figures 5–7 now contain these mixed-in values.

### B1. Feature-count dose–response (report §D1b, Fig. 5)

- **Result**: both selection orders plateau by k≈20 under mixed-in. The recorded anchors are
  RF-fs k=10 = 0.719 and RF-nofs k=165 = 0.829. The mechanism claim is retained: the low top-10
  arm is primarily consistent with selecting features after SW-SMOTE rather than with ten
  features being intrinsically insufficient.

### B2. SvmA learner-versus-feature-set probe (report §D4, Fig. 6)

- **Result**: on the same 36 steering features, RF = 0.877 and RBF-SVM = 0.544; RF on all 165
  vehicle features = 0.889. Across raw, SW-SMOTE, and class-weight treatments, RF remains
  ~0.87–0.88 while RBF-SVM remains ~0.54.
- **Conclusion retained**: SvmA's bottom rank is a learner ceiling, not a feature-set deficit or
  imbalance artifact.

### B3. SvmW wavelet-feature probe (report §D5, Fig. 7 panel B)

- **Result**: SvmW's 8 GHM steering-wavelet features fed to RF give raw 0.877, SW-SMOTE 0.858,
  and class-weight 0.871 under mixed-in.
- **Conclusion retained**: a learner that does not degenerate reads the recorded signal from
  these features without rebalancing, so SvmW's pooled collapse is a learner–imbalance
  interaction rather than a feature deficit.

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

## 6. Remaining deliverables

1. After RF-nofs reaches 15 seeds in every c1 cell, re-run Part A and regenerate
  `c1_recorded_value_analysis.md` §A–§E and figures 1–4 without the within modes.
2. Retain the completed mixed-in measurements in §D1b, §D4, §D5 and Figures 5–7.
3. Update the manuscript in `TIV2026_exp3/` in a single pass: drop the within columns from
   Table III, re-anchor the domain-restriction contrast to pooled → mixed, and re-point the
   mechanism subsections at the new probe values. The regime-scoped headline of §1.1 above is
   what the manuscript should end up claiming.
