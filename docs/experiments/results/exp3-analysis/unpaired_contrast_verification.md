# exp3 — Verification Items for Replacing the Seed-Paired Mode Contrasts with Unpaired Tests

**Status: RUN on the experiment PC, 2026-08-16.** Results are in
[`unpaired_contrast_verification_results.md`](unpaired_contrast_verification_results.md).
**A2 passed, A1 FAILED** — the `target_group(in→out)` axis shows rho = +0.96 / +1.00 / +0.81 for
RF-fs / RF-nofs / SvmW, so the blanket switch specified below is not licensed by its own
diagnostic; a per-axis decision is required. The §C block of the analysis script has been
replaced with the §3.3 drop-in (which reports paired and unpaired side by side). The manuscript
pass described in §6 has **not** been done.

Original brief follows.

**Status when written: PENDING (2026-08-16). To be run on the experiment PC**, where
`results/outputs/evaluation/` holds the per-seed evaluation JSONs. No models are retrained:
every item below is a re-analysis of files already on disk.

This document specifies exactly what to compute and what to hand back. The manuscript in
`TIV2026_exp3/` will be revised in one pass from the returned numbers.

## 1. Decision and rationale

The within-method mode contrasts (Table VI right half, `§C` of
`scripts/python/analysis/exp3_c1_recorded_value_analysis.py`) are currently **seed-paired**
Wilcoxon signed-rank tests over the seeds common to the two compared cells. **The intention is
to replace them with unpaired Mann–Whitney tests over all seeds of each cell**, subject to the
diagnostic in §3.1 below.

Three reasons:

1. **The pairing is not justified by the design, and the analysis already says so elsewhere.**
   Pairing helps only when the seed acts as a nuisance factor shared by both cells, so that the
   two per-seed samples are positively correlated. The split is fixed and common to every cell
   (chronological 70:15:15), so the seed does not move the split; it moves the model's random
   state and the trajectory of the hyperparameter search, and that search is repeated per
   (method, mode) cell. Under SW-SMOTE the training data itself differs, so the trajectories
   diverge. The same argument was already accepted for the RF-fs vs RF-nofs contrast, which was
   switched from paired to Mann–Whitney on 2026-08-15 after the per-seed outcomes measured
   **r = −0.09** (see the DATA UPDATE entry in `TIV2026_exp3/HISTORY.md`). The equivalent
   correlation has never been measured for the mode-to-mode contrasts.

2. **Pairing discards seeds.** The RF-nofs imbalance contrast has 24 seeds in `pooled_base` and
   6 in `pooled_smote`, and is currently tested on the 6 common ones.

3. **The signed-rank p-value is saturated.** Its floor is 2^(1−n), i.e. 0.031 at n = 6, and four
   of the reported p-values sit exactly there (RF-nofs +0.210, SvmW +0.179, SvmW +0.052, Lstm
   +0.268 all read p = 0.031). The test cannot separate those effects at all. Unpaired at
   24 vs 6 the exact null has C(30,6) = 593,775 arrangements, so p can reach ~3×10⁻⁶.

## 2. Where the data and the code are

| What | Path |
|---|---|
| Per-seed evaluation JSONs | `results/outputs/evaluation/{model}/**/eval_results_{model}_*_s{seed}.json` |
| Analysis script | `scripts/python/analysis/exp3_c1_recorded_value_analysis.py` |
| Repo root constant to check | `REPO` (L26) — currently `c:/git/work/vehicle_ddd_eval/vehicle_based_DDD_comparison` |
| Eval dir constant | `EVAL` (L28) = `results/outputs/evaluation` |
| Cell classifier (method/mode/seed from filename) | `classify()` (L40–52) |
| **Block to replace** | `§C WITHIN-METHOD mode contrasts (paired Wilcoxon)`, **L151–165** |
| Helpers to reuse | `cliffs_delta()` (L82), `cliff_mag()` (L88), `arr()` (L78) |

The four modes are `pooled_base`, `pooled_smote`, `mixed_in`, `mixed_out` (L33); the two
`within_*` modes stay retired (`within_retirement_plan.md`).

## 3. Verification items

### 3.0 Seed inventory (prerequisite)

Print, for each of the 5 methods × 4 modes, the number of seeds actually found and the seed IDs.
This is needed to confirm that the values the manuscript reports still match what is on disk, and
to see how many seeds each contrast currently throws away.

**Hand back**: a 5 × 4 table of `n` and, per contrast, `n_cell1 / n_cell2 / n_common`.

### 3.1 Does the seed pair anything? (the diagnostic that decides the switch)

For each method × contrast, over the **common** seeds only, compute the Spearman correlation
between the two cells' per-seed AUROC. This is the direct analogue of the r = −0.09 measurement
that retired the paired test for RF-fs vs RF-nofs.

**Hand back**: 5 methods × 3 contrasts of `n_common` and `rho`.

**Reading**: if |rho| is small (say < 0.3) and unsystematic in sign, the pairing buys nothing and
the switch is justified. If some cell shows a substantial positive rho, that contrast keeps its
paired test and the manuscript says which ones are paired and why — see §5.

### 3.2 Unpaired recomputation (the numbers the paper will use)

For each method × contrast, using **all** seeds of each cell:

- two-sided Mann–Whitney U with the exact null where scipy can use it,
- the Hodges–Lehmann point estimate (median of all pairwise differences), which is the estimator
  consistent with Mann–Whitney,
- Cliff's δ and its magnitude label.

**Hand back**: 5 × 3 of `n1`, `n2`, `HL`, `p`, `delta`, alongside the current paired `dmed` and
`p` so the two can be compared line by line.

### 3.3 Drop-in code

Replace L151–165 with:

```python
print("\n(C) WITHIN-METHOD mode contrasts (unpaired Mann-Whitney + paired diagnostic)")
CONTR = [("imbalance(base->SWSMOTE)", "pooled_base", "pooled_smote"),
         ("population(pooled->mixed)", "pooled_smote", "mixed_in"),
         ("target_group(in->out)",     "mixed_in",    "mixed_out")]

def hodges_lehmann(x, y):
    """Median of all pairwise differences y_j - x_i: the Mann-Whitney-consistent estimate."""
    return float(np.median([b - a for a in x for b in y]))

print(f"  {'method':8} {'contrast':26} {'n1':>3}{'n2':>4} {'HL':>7} {'p_unpair':>9} "
      f"{'delta':>6} | {'nc':>3} {'rho':>6} {'dmed':>7} {'p_pair':>8}")
for method in METHODS:
    for name, m1, m2 in CONTR:
        d1, d2 = arr(method, m1), arr(method, m2)
        x = np.array(list(d1.values()), float)
        y = np.array(list(d2.values()), float)
        if len(x) < 2 or len(y) < 2:
            print(f"  {method:8} {name:26}: NA (n1={len(x)}, n2={len(y)})"); continue
        # --- unpaired: what the paper will report ---
        U, p = stats.mannwhitneyu(y, x, alternative="two-sided")   # 'auto' -> exact when it can
        hl = hodges_lehmann(x, y); dl = cliffs_delta(y, x)
        # --- diagnostic: the pairing the switch is retiring ---
        common = sorted(set(d1) & set(d2))
        xc = np.array([d1[s] for s in common], float)
        yc = np.array([d2[s] for s in common], float)
        rho = stats.spearmanr(xc, yc)[0] if len(common) >= 4 else float('nan')
        if len(common) >= 3:
            try: _, p_pair = stats.wilcoxon(xc, yc)
            except ValueError: p_pair = float('nan')
            d_pair = float(np.median(yc - xc))
        else:
            p_pair = d_pair = float('nan')
        print(f"  {method:8} {name:26} {len(x):3d}{len(y):4d} {hl:+7.3f} {p:9.3g} "
              f"{dl:+6.2f} | {len(common):3d} {rho:+6.2f} {d_pair:+7.3f} {p_pair:8.3g}")
```

Notes for whoever runs it:

- `stats.mannwhitneyu` is left on `method='auto'` on purpose: it uses the exact null when the
  samples are small and untied, and the tie-corrected normal approximation otherwise. If it warns
  about ties, report that — with AUROC floats ties should be rare, and their presence is itself
  worth knowing.
- `stats.spearmanr(xc, yc)[0]` is used rather than `.statistic` for scipy-version portability.
- Everything else in the script (§A, §B Dunn/Holm, §D1 fs-vs-nofs, §D2, §D3, §E) is untouched —
  those are already unpaired and stay as they are.
- The block was dry-run against synthetic cells on numpy 2.5.2 / scipy 1.18 before being written
  down here. Two edge cases behave as follows and are expected rather than a bug: a cell whose
  values are all identical makes both `spearmanr` (`ConstantInputWarning`) and `mannwhitneyu`
  return `nan`, and a cell with fewer than four common seeds leaves `rho` as `nan`. Report those
  rows as undefined rather than dropping them.

## 4. Acceptance criteria

The switch goes through if:

- **A1** — §3.1 shows no substantial positive seed correlation (|rho| < 0.3, no consistent sign)
  in any contrast. This is the finding that licenses dropping the pairing.
- **A2** — §3.0 reproduces the seed counts the manuscript states: six to twenty-four per cell,
  RF-fs and RF-nofs at n = 24 in `pooled_base`, RF-nofs at n = 6 in `pooled_smote`.

**These would change the paper's claims and must be reported if seen:**

- **The population axis.** RF-fs currently decreases with `−0.039, p = 0.005` (paired) and the
  manuscript calls it "the only method that decreases significantly from pooled to mixed
  evaluation" — a claim carried in the table note, in §IV-A, and in the RQ answers. It is the
  smallest effect that the paper reports as significant, so **it is the one most likely to lose
  significance when unpaired.** If it does, say so; the RF-fs/RF-nofs dissociation then rests on
  the imbalance axis alone and §V-C is rewritten rather than renumbered.
- **The imbalance axis for SvmA.** Currently `p = 0.063`, described as inconclusive rather than as
  evidence of no effect. Unpaired at full n it may cross 0.05, which would make SvmA a responder
  and change the "SvmA responds to neither treatment" line in the RQ2 answer.
- **Lstm on the imbalance axis.** Currently `+0.005, p = 1.0` ("does not move at all"). It should
  stay null; if it does not, the "SW-SMOTE has nothing to act on when the target is already
  balanced" mechanism claim is affected.
- **Any sign flip** between the paired median and the Hodges–Lehmann estimate, in any cell.

## 5. Open decision to record with the results

The 15 contrasts (5 methods × 3 axes) are currently reported **uncorrected**, each as a
pre-specified within-method comparison, in contrast to the between-method Dunn tests which are
Holm-corrected over the ten pairs of a mode. That asymmetry is deliberate and the manuscript's
Appendix C states it. Keeping it under the unpaired scheme is the default; if the returned
numbers are to be Holm-corrected within each method (3 tests per method) instead, note that with
the results so the appendix can be written to match.

## 6. Manuscript-side substitutions this feeds (for reference; done in `TIV2026_exp3/`)

| Location | Current content that depends on the paired test |
|---|---|
| `main.tex:805` | Table VI header, "Seed-paired contrast, median Δ (p)" |
| `main.tex:815–826` | the 15 contrast cells |
| `main.tex:831` | table note "*Right.*" — the "taken over the seeds common to the two compared cells / not the differences of the columns" caveat |
| `main.tex:836` | note § — "six seeds are common … 0.031 is the floor of the two-sided test" (deleted under the switch) |
| `main.tex:856–858` | §IV-A, all three axis paragraphs |
| `main.tex:876` | §V-B, Lstm in-to-out `Δ = −0.006, p = 0.25` |
| `main.tex:885` | §V-C, RF-nofs "+0.210 over six common seeds" |
| `main.tex:906` | §VI-A threats, SvmA "p = 0.063 … inconclusive" |
| `main.tex:632` | §III-G, "Comparisons are of two kinds and no others are made" — becomes one kind |
| `main.tex:1046–1063` | Appendix C, the *Between modes, within a method* block: the pairing justification, eq. (diff), and the three Wilcoxon equations are removed and folded into the existing Mann–Whitney block |

## 7. Deliverable

The stdout of the modified §C (the table in §3.3's format) is sufficient — it carries the unpaired
numbers, the paired numbers, and the correlation diagnostic side by side. Paste it back, together
with the §3.0 seed inventory and any scipy tie warnings.
