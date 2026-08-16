# exp3 — Results of the Verification Items for the Paired → Unpaired Switch

**Status: RUN (2026-08-16) on the experiment PC.** Answers every item of
`unpaired_contrast_verification.md`. No models were retrained; this is a re-analysis of the
evaluation JSONs already on disk.

**Headline: A2 passes, A1 fails.** The blanket switch is **not** licensed by the diagnostic the
document itself specified. Six of fifteen contrasts show |rho| ≥ 0.3, and on the
`target_group(in→out)` axis the correlation is near-perfect for the three highest-n methods
(+0.96, +1.00, +0.81). A per-axis decision is now needed instead of a blanket one — see §6.

## 1. Environment

| | |
|---|---|
| Interpreter | Python 3.13.2, numpy 2.3.5, scipy 1.16.3 |
| Cross-check | Python 3.11.9, numpy 1.26.4, scipy 1.16.3 — **§C output byte-identical** |
| Script | `scripts/python/analysis/exp3_c1_recorded_value_analysis.py`, §C replaced per §3.3 |
| Figures | regenerated on 3.11; **byte-identical to the tracked PNGs** (no figure churn) |

The repo's `.venv_tf210_cuda` has no numpy/scipy/matplotlib (it is the TF-only training venv);
the analysis runs on the system interpreters. matplotlib 3.10.0 exists only on 3.11, so the full
script including figures must be run there — on 3.13 it completes §A–§E and then raises
`ModuleNotFoundError: matplotlib` at the figure section. That is pre-existing and unrelated to
the §C change.

## 2. Acceptance criteria

### A2 — seed counts — **PASS**

Every cell is between 6 and 24 seeds; `RF_fs` and `RF_nofs` are at n = 24 in `pooled_base`;
`RF_nofs` is at n = 6 in `pooled_smote`. Matches what the manuscript states.

### A1 — no substantial positive seed correlation — **FAIL**

A1 required |rho| < 0.3 with no consistent sign **in any contrast**. Measured: **6 of 15
contrasts have |rho| ≥ 0.3**, and the exceedances are not scattered — they concentrate on one
axis with a consistent positive sign:

| axis | RF_fs | RF_nofs | SvmW | SvmA | Lstm | verdict |
|---|---|---|---|---|---|---|
| imbalance(base→SWSMOTE) | +0.11 | +0.20 | **+0.54** | +0.14 | −0.03 | mostly small, one exceedance |
| population(pooled→mixed) | **+0.52** | +0.26 | +0.20 | −0.20 | **−0.49** | mixed signs, two exceedances |
| target_group(in→out) | **+0.96** | **+1.00** | **+0.81** | +0.03 | −0.08 | **strong, systematic, positive** |

The `target_group` axis is exactly the case pairing is designed for: the two cells differ only in
which group of the target population is evaluated, at the same seed and the same training
configuration. For `RF_fs`, `RF_nofs` and `SvmW` the seed evidently carries almost all of the
between-cell variation. (`SvmA` +0.03 and `Lstm` −0.08 do not follow this pattern, so it is not a
universal property of the axis — the two methods have separate `in_domain`/`out_domain` training
runs on disk, as do the others, so the mechanism was not confirmed further and should not be
asserted without checking.)

This is the opposite of the r = −0.09 that retired the pairing for RF-fs vs RF-nofs. The premise
of §1 reason 1 — "the equivalent correlation has never been measured for the mode-to-mode
contrasts" — has now been measured, and for one axis it does not support the switch.

## 3. Item 3.0 — seed inventory

n per method × mode:

| method | pooled_base | pooled_smote | mixed_in | mixed_out |
|---|---|---|---|---|
| RF_fs | 24 | 15 | 24 | 24 |
| RF_nofs | 24 | 6 | 15 | 15 |
| SvmW | 6 | 6 | 8 | 8 |
| SvmA | 6 | 6 | 11 | 11 |
| Lstm | 6 | 6 | 15 | 15 |

Seed IDs (the 24-seed set is
`0,1,3,5,7,9,11,13,17,23,31,42,47,99,101,123,256,333,512,777,1337,2024,2025,2718`; the 15-seed
set is its first 8 plus `42,123,256,512,1337,2024,2025`; the 6-seed set is `0,1,7,42,123,2025`;
`SvmW` mixed adds `13,256` to the 6-set and `SvmA` mixed adds `13,256,512,1337,2024`).

Seeds discarded by pairing, per contrast:

| method | contrast | n1 | n2 | n_common | discarded |
|---|---|---|---|---|---|
| RF_fs | imbalance | 24 | 15 | 15 | 9 |
| RF_fs | population | 15 | 24 | 15 | 9 |
| RF_fs | target_group | 24 | 24 | 24 | 0 |
| RF_nofs | imbalance | 24 | 6 | 6 | **18** |
| RF_nofs | population | 6 | 15 | 6 | 9 |
| RF_nofs | target_group | 15 | 15 | 15 | 0 |
| SvmW | imbalance | 6 | 6 | 6 | 0 |
| SvmW | population | 6 | 8 | 6 | 2 |
| SvmW | target_group | 8 | 8 | 8 | 0 |
| SvmA | imbalance | 6 | 6 | 6 | 0 |
| SvmA | population | 6 | 11 | 6 | 5 |
| SvmA | target_group | 11 | 11 | 11 | 0 |
| Lstm | imbalance | 6 | 6 | 6 | 0 |
| Lstm | population | 6 | 15 | 6 | 9 |
| Lstm | target_group | 15 | 15 | 15 | 0 |

Both `target_group` contrasts and three `imbalance` contrasts discard nothing — pairing costs no
seeds there. The cost is concentrated in `RF_nofs imbalance` (18 of 30 seeds discarded).

## 4. Items 3.1 + 3.2 — the §C table

Verbatim stdout of the modified §C:

```
(C) WITHIN-METHOD mode contrasts (unpaired Mann-Whitney + paired diagnostic)
  method   contrast                    n1  n2      HL  p_unpair  delta |  nc    rho    dmed   p_pair
  RF_fs    imbalance(base->SWSMOTE)    24  15  +0.080     0.039  +0.40 |  15  +0.11  +0.060   0.0353
  RF_fs    population(pooled->mixed)   15  24  -0.060   0.00256  -0.58 |  15  +0.52  -0.039  0.00537
  RF_fs    target_group(in->out)       24  24  +0.034     0.146  +0.25 |  24  +0.96  +0.032 8.34e-06
  RF_nofs  imbalance(base->SWSMOTE)    24   6  +0.240   6.4e-05  +0.93 |   6  +0.20  +0.210   0.0312
  RF_nofs  population(pooled->mixed)    6  15  -0.007     0.677  -0.13 |   6  +0.26  -0.009    0.562
  RF_nofs  target_group(in->out)       15  15  +0.068   0.00323  +0.64 |  15  +1.00  +0.066  6.1e-05
  SvmW     imbalance(base->SWSMOTE)     6   6  +0.178   0.00216  +1.00 |   6  +0.54  +0.179   0.0312
  SvmW     population(pooled->mixed)    6   8  +0.046  0.000666  +1.00 |   6  +0.20  +0.052   0.0312
  SvmW     target_group(in->out)        8   8  +0.032   0.00466  +0.81 |   8  +0.81  +0.030  0.00781
  SvmA     imbalance(base->SWSMOTE)     6   6  +0.068     0.026  +0.78 |   6  +0.14  +0.068   0.0625
  SvmA     population(pooled->mixed)    6  11  -0.013     0.884  -0.06 |   6  -0.20  -0.017    0.688
  SvmA     target_group(in->out)       11  11  +0.067  8.15e-05  +1.00 |  11  +0.03  +0.060 0.000977
  Lstm     imbalance(base->SWSMOTE)     6   6  +0.003     0.699  +0.17 |   6  -0.03  +0.005        1
  Lstm     population(pooled->mixed)    6  15  +0.268  3.69e-05  +1.00 |   6  -0.49  +0.268   0.0312
  Lstm     target_group(in->out)       15  15  -0.002     0.507  -0.15 |  15  -0.08  -0.006    0.252
```

Cliff's δ magnitude labels: RF_fs medium / large / small; RF_nofs large / negligible / large;
SvmW large / large / large; SvmA large / negligible / large; Lstm small / large / negligible.

### 4.1 Ties and the exact-vs-asymptotic choice

**No scipy warnings of any kind were raised** during the block. Exactly one cell contains a tie —
`Lstm mixed_out` (n = 15, 14 unique values); every other cell is untied. That tie touches only
`Lstm target_group`, which is null under both tests.

`method='auto'` resolved to **exact for 9 of 15 rows** (those with min(n1,n2) ≤ 8) and
**asymptotic for the other 6**. Two rows differ materially between the two:

| row | auto (used) | exact | asymptotic |
|---|---|---|---|
| SvmA target_group | 8.15e-05 *(asymptotic)* | **2.84e-06** | 8.15e-05 |
| RF_nofs imbalance | 6.4e-05 *(exact)* | 6.4e-05 | 5.65e-04 |

`SvmA target_group` has δ = +1.00 (complete separation at 11 vs 11), so the exact null gives a
p ~29× smaller than the one `auto` reports. If the manuscript quotes exact p-values anywhere it
should say which rows are exact, because `auto` is not uniform across the table.

The document's §1 claim that unpaired at 24 vs 6 "can reach ~3×10⁻⁶" is correct in principle but
is not attained here: `RF_nofs imbalance` reaches 6.4e-05, not the 3e-06 floor, because δ = +0.93
rather than 1.00.

## 5. The four flagged items

**1. The population axis / RF-fs — claim SURVIVES, and strengthens.**
Paired −0.039, p = 0.005 → unpaired **HL −0.060, p = 0.00256, δ = −0.58 (large)**. It does not
lose significance. It also remains *the only method that decreases* on this axis: RF_nofs −0.007
(p = 0.68, ns) and SvmA −0.013 (p = 0.88, ns) are null, while SvmW +0.046 (p = 0.00067) and Lstm
+0.268 (p = 3.7e-05) *increase*. The §IV-A / RQ / table-note claim holds under either test, and
§V-C does not need rewriting on this account.

**2. The imbalance axis for SvmA — CROSSES 0.05. This is the one claim-changing result.**
Paired p = 0.0625 ("inconclusive") → unpaired **p = 0.026, HL +0.068, δ = +0.78 (large)**.
n1 = n2 = 6 with zero seeds discarded, so *nothing was gained from extra data* — the entire change
comes from replacing the signed-rank test with Mann–Whitney. This contrast's rho is +0.14, i.e.
it is one of the contrasts where §3.1 does license the switch, which makes the result hard to set
aside. Taken at face value it makes SvmA a responder to SW-SMOTE and contradicts the "SvmA
responds to neither treatment" line in the RQ2 answer and the §VI-A threats wording.
**But see §6 — Holm correction within method returns it to p = 0.052.**

**3. Lstm on the imbalance axis — stays null, as required.**
+0.005, p = 1.0 → **+0.003, p = 0.699, δ = +0.17 (small)**. The "SW-SMOTE has nothing to act on
when the target is already balanced" mechanism claim is unaffected.

**4. Sign flips — NONE.** All 15 contrasts agree in sign between the paired median Δ and the
Hodges–Lehmann estimate, and the magnitudes are close (largest divergence: RF_fs imbalance,
+0.060 → +0.080).

### 5.1 The saturation argument is confirmed

Exactly four paired p-values sit on the 2^(1−6) = 0.031 floor, precisely the four the document
named (RF_nofs imbalance +0.210, SvmW imbalance +0.179, SvmW population +0.052, Lstm population
+0.268). Unpaired, those four separate into 6.4e-05, 0.00216, 0.000666 and 3.69e-05 — a spread of
~2 orders of magnitude that the signed-rank test collapsed onto a single value. This part of the
rationale (§1 reason 3) is fully borne out, and it is the strongest argument for switching the
`imbalance` and `population` axes even though A1 blocks a blanket switch.

## 6. §5 open decision — Holm within method changes the answer

Because the SvmA result in §5 item 2 is the only claim-changing one, whether the 15 contrasts stay
uncorrected is now decision-relevant rather than cosmetic. Holm over the 3 contrasts of each
method:

| method | contrast | p_raw | p_holm | sig raw → holm |
|---|---|---|---|---|
| RF_fs | imbalance | 0.039 | 0.078 | **Y → n** |
| RF_fs | population | 0.00256 | 0.00767 | Y → Y |
| RF_fs | target_group | 0.146 | 0.146 | n → n |
| RF_nofs | imbalance | 6.4e-05 | 0.000192 | Y → Y |
| RF_nofs | population | 0.677 | 0.677 | n → n |
| RF_nofs | target_group | 0.00323 | 0.00646 | Y → Y |
| SvmW | imbalance | 0.00216 | 0.00433 | Y → Y |
| SvmW | population | 0.000666 | 0.002 | Y → Y |
| SvmW | target_group | 0.00466 | 0.00466 | Y → Y |
| SvmA | imbalance | 0.026 | **0.0519** | **Y → n** |
| SvmA | population | 0.884 | 0.884 | n → n |
| SvmA | target_group | 8.15e-05 | 0.000245 | Y → Y |
| Lstm | imbalance | 0.699 | 1 | n → n |
| Lstm | population | 3.69e-05 | 0.000111 | Y → Y |
| Lstm | target_group | 0.507 | 1 | n → n |

Under Holm, **SvmA imbalance returns to 0.052 and the "responds to neither treatment" claim
survives unchanged**; RF_fs imbalance also drops out at 0.078. So the two schemes disagree on
exactly the flagged claim. Choosing "uncorrected" and choosing "Holm" are therefore no longer
equivalent presentations of the same evidence, and the choice should be made and stated
explicitly rather than inherited.

## 7. What was changed in this repo

- `scripts/python/analysis/exp3_c1_recorded_value_analysis.py` — §C replaced with the §3.3
  drop-in **verbatim**, plus the module docstring line for §C updated. The block prints the
  unpaired numbers, the paired numbers and the rho diagnostic side by side, so the manuscript's
  current paired values remain reproducible from it whichever way the decision goes. §A, §B, §D,
  §E and all figures are untouched and verified unchanged.
- This results file.

## 8. Left for the manuscript side (not done here)

The document's §6 substitution table assumes a blanket switch, which A1 does not support. The
manuscript revision in `TIV2026_exp3/` still needs a decision on:

1. **Per-axis vs blanket.** The evidence supports switching `imbalance` and `population` to
   unpaired (small, unsystematic rho; saturation demonstrably harmful) and **keeping
   `target_group` paired** for at least RF_fs / RF_nofs / SvmW (rho +0.96 / +1.00 / +0.81). Note
   that `target_group` discards no seeds for any method, so pairing costs nothing there — the two
   reasons for switching (seeds discarded, saturation) both have their weakest case on exactly
   the axis where the correlation is strongest.
2. **Correction scheme** (§6 above) — it decides the SvmA claim.
3. Whether `main.tex:632` ("Comparisons are of two kinds and no others are made") becomes one
   kind or three; under a per-axis scheme it stays two kinds but the description changes.

Both decisions are judgement calls about how the paper presents the evidence, so they are left to
the manuscript pass rather than settled here.
