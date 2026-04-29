# Prior Research Replication Results (Experiment 3)

> **Note:** For pipeline / model details, see [Prior Research Models](../../architecture/prior_research.md).
> For condition definitions, see [Prior Research Conditions](../conditions/03-prior-research-conditions.md).
> For HPC operational history (re-submissions, known issues), see [Operations Log](exp3-analysis/operations_log.md).

---

## Experiment 3: Prior Research Replication (domain_train unified, 15-seed scope)

**Last updated:** 2026-04-29

### Overview

Experiment 3 replicates three prior-research baseline models — **SvmW**, **SvmA**, **Lstm** —
on the same vehicle-based subject split (split2) as Experiment 2. All three models are
trained in the **`domain_train` unified mode** (split each domain's data 70/15/15, train once,
evaluate twice: within-domain test 15% and cross-domain test 15%).

The current target is the **15-seed expanded scope** (see Job Status below) which matches
Experiment 2's seed list. The earlier 2-seed canonical scope (252 jobs) is retained as
a historical reference point in [Operations Log](exp3-analysis/operations_log.md).

### Experiment Matrix

| Parameter | Values | Count |
|-----------|--------|-------|
| Models | SvmW, SvmA, Lstm | 3 |
| Distance metrics | mmd, dtw, wasserstein | 3 |
| Domain groups | in_domain (44), out_domain (43) | 2 |
| Training mode | domain_train (within + cross eval per job) | 1 |
| Seeds | 0, 1, 3, 7, 13, 42, 99, 123, 256, 512, 777, 999, 1234, 1337, 2024 | **15** |
| Conditions | baseline, smote_plain, smote (sw_smote), undersample | 4 |
| Target ratios | 0.1, 0.5 (ratio-based methods only) | 2 |

**Total configurations:** 84 unique tags / model / 2 seeds × (15 / 2) seed expansion
→ **630 tags / model × 3 models = 1890 jobs** (each job produces both `_within.json` and `_cross.json`).

### Experiment Status (2026-04-29, 15-seed scope)

Coverage measured by **unique tags with both `_within.json` and `_cross.json`**.

| Model | Within | Cross | Target | Coverage | Status |
|---|---|---|---|---|---|
| SvmW | 630 | 630 | 630 | **100%** | Complete ✅ |
| SvmA | 592 | 592 | 630 | 94% | 38 tags pending |
| Lstm | 190 | 187 | 630 | 30% | Major expansion in flight |
| **Total** | **1412** | **1409** | **1890** | **75%** | Re-submission running |

> Use `qstat -u $USER` for live job status. The 24 canonical-seed re-submissions
> (PBS 20199–20236) and ongoing seed-expansion submissions are documented in
> [Operations Log](exp3-analysis/operations_log.md).

### Per-Condition Tag Counts (within JSON)

| Model | baseline | smote (sw_smote / imbalv3) | smote_plain | undersample_rus | Sum |
|---|---|---|---|---|---|
| SvmW | 90 | 180 | 180 | 180 | 630 |
| SvmA | 90 | 180 | 167 | 155 | 592 |
| Lstm | 53 | 65 | 38 | 34 | 190 |

Expected per-condition (15 seed): baseline 90, others 180 each (= 630).

### Artifact Completeness Audit (canonical 2-seed sub-set, 2026-04-23)

| Model | Tag/JSON (canon) | Final model | Scaler | Selected features | Extra |
|---|---|---|---|---|---|
| SvmW | 84 / 84 | 84 / 84 | 84 / 84 | 84 / 84 | Optuna study **84 / 84** ✅ |
| SvmA | 83 / 84 | 83 / 83 | 83 / 83 | 83 / 83 | — |
| Lstm | 63 / 84 | 63 / 63 | 63 / 63 | 63 / 63 | training_history **63 / 63**, 5-fold models **63 / 63** ✅ |

> SvmW domain_train average execution time: ~15 min (significantly reduced from ~5 hours in the old split2 version).

---

## Key Results Summary

> Aggregated 2026-04-29 from
> `results/analysis/exp3_prior_research/figures/csv/split2/{Model}/{condition}/*.csv`
> via `collect_split2_prior_research_metrics.py --seeds all`.
> N = number of records (= unique tag × eval_type, so a fully covered condition has N=180 = 90 tags × 2 evals).
> Cells with very low N (e.g., Lstm smote_plain r=0.1 N=2) are unreliable and pending more jobs.

### Overall Mean Across All Distances / Domains / Eval Types

#### SvmW

| Condition | N | F2 | AUROC | AUPRC | Recall | Precision |
|---|---|---|---|---|---|---|
| baseline | 180 | 0.328±0.059 | 0.545±0.041 | 0.116±0.025 | 0.861±0.251 | 0.100±0.018 |
| smote_plain (r=0.1) | 180 | 0.280±0.074 | 0.543±0.026 | 0.116±0.019 | 0.532±0.257 | 0.109±0.018 |
| smote_plain (r=0.5) | 180 | 0.212±0.044 | 0.529±0.021 | 0.114±0.019 | 0.282±0.067 | 0.108±0.022 |
| sw_smote (r=0.1) | 180 | 0.313±0.069 | 0.538±0.025 | 0.112±0.017 | 0.665±0.237 | 0.106±0.015 |
| sw_smote (r=0.5) | 180 | 0.253±0.050 | 0.526±0.019 | 0.113±0.020 | 0.393±0.095 | 0.106±0.021 |
| undersample_rus (r=0.1) | 180 | 0.334±0.054 | 0.548±0.040 | 0.116±0.021 | 0.876±0.221 | 0.101±0.018 |
| undersample_rus (r=0.5) | 180 | **0.347±0.047** | **0.556±0.044** | **0.119±0.021** | **0.955±0.116** | 0.100±0.018 |

#### SvmA

| Condition | N | F2 | AUROC | AUPRC | Recall | Precision |
|---|---|---|---|---|---|---|
| baseline | 180 | 0.159±0.005 | 0.499±0.018 | 0.037±0.004 | 0.931±0.084 | 0.037±0.002 |
| smote_plain (r=0.1) | 180 | 0.159±0.005 | 0.500±0.021 | 0.037±0.003 | 0.911±0.117 | 0.037±0.002 |
| smote_plain (r=0.5) | 154 | 0.159±0.007 | 0.501±0.024 | 0.038±0.005 | 0.885±0.116 | 0.037±0.003 |
| sw_smote (r=0.1) | 180 | 0.159±0.005 | 0.499±0.019 | 0.037±0.004 | 0.896±0.118 | 0.037±0.002 |
| sw_smote (r=0.5) | 180 | 0.159±0.006 | 0.500±0.019 | 0.038±0.004 | 0.911±0.091 | 0.037±0.002 |
| undersample_rus (r=0.1) | 160 | 0.159±0.005 | 0.499±0.018 | 0.037±0.004 | 0.906±0.106 | 0.037±0.002 |
| undersample_rus (r=0.5) | 150 | 0.158±0.005 | 0.500±0.020 | 0.039±0.005 | 0.905±0.100 | 0.037±0.002 |

> ⚠️ SvmA collapses to a near-constant predictor (AUROC ≈ 0.50, F2 ≈ 0.16) across **all** conditions.
> See Key Findings #2.

#### Lstm

| Condition | N | F2 | AUROC | AUPRC | Recall | Precision |
|---|---|---|---|---|---|---|
| baseline | 105 | 0.932±0.001 | 0.779±0.040 | 0.913±0.019 | 0.999±0.001 | 0.735±0.006 |
| smote_plain (r=0.1) | 2 | 0.932±0.000 | 0.774±0.113 | 0.910±0.052 | 1.000±0.000 | 0.734±0.001 |
| smote_plain (r=0.5) | 72 | 0.933±0.002 | 0.773±0.057 | 0.910±0.026 | 0.999±0.002 | 0.738±0.008 |
| sw_smote (r=0.1) | 68 | 0.933±0.001 | 0.782±0.046 | 0.915±0.022 | 0.999±0.001 | 0.737±0.006 |
| sw_smote (r=0.5) | 62 | 0.932±0.001 | 0.754±0.064 | 0.904±0.029 | 1.000±0.001 | 0.734±0.004 |
| undersample_rus (r=0.1) | 2 | 0.933±0.000 | 0.776±0.084 | 0.911±0.042 | 0.999±0.002 | 0.737±0.004 |
| undersample_rus (r=0.5) | 66 | 0.933±0.001 | **0.783±0.041** | 0.915±0.019 | 0.999±0.002 | 0.736±0.006 |

> ⚠️ Lstm always predicts the positive class (Recall ≈ 1.0, F2 ≈ 0.93 driven by class prior).
> AUROC ≈ 0.77–0.78 indicates real (but modest) discrimination. See Key Findings #3.

### Out-Domain F2 by Eval Type (Domain Shift Robustness)

`within` = trained on the same out-domain train split, tested on its held-out 15%.
`cross`  = trained on the *opposite* domain (in_domain), tested on out_domain 15%.

#### SvmW

| Condition | within | cross |
|---|---|---|
| baseline | 0.334±0.067 | 0.276±0.038 |
| smote_plain (r=0.1) | 0.256±0.051 | 0.204±0.067 |
| smote_plain (r=0.5) | 0.219±0.035 | 0.174±0.044 |
| sw_smote (r=0.1) | 0.314±0.064 | 0.270±0.092 |
| sw_smote (r=0.5) | 0.263±0.036 | 0.211±0.047 |
| undersample_rus (r=0.1) | 0.357±0.060 | 0.293±0.031 |
| undersample_rus (r=0.5) | **0.377±0.029** | **0.321±0.046** |

#### SvmA

| Condition | within | cross |
|---|---|---|
| baseline | 0.155±0.004 | 0.161±0.004 |
| smote_plain (r=0.1) | 0.156±0.005 | 0.162±0.005 |
| smote_plain (r=0.5) | 0.155±0.004 | 0.163±0.005 |
| sw_smote (r=0.1) | 0.156±0.005 | 0.162±0.004 |
| sw_smote (r=0.5) | 0.156±0.004 | 0.163±0.005 |
| undersample_rus (r=0.1) | 0.157±0.004 | 0.161±0.004 |
| undersample_rus (r=0.5) | 0.154±0.004 | 0.162±0.004 |

#### Lstm

| Condition | within | cross |
|---|---|---|
| baseline | 0.932±0.001 | 0.933±0.001 |
| smote_plain (r=0.5) | 0.932±0.001 | 0.934±0.002 |
| sw_smote (r=0.1) | 0.932±0.001 | 0.933±0.001 |
| sw_smote (r=0.5) | 0.931±0.001 | 0.933±0.001 |
| undersample_rus (r=0.5) | 0.932±0.001 | 0.933±0.002 |

(rows with N≤2 omitted)

### Out-Domain AUROC by Eval Type

#### SvmW

| Condition | within | cross |
|---|---|---|
| baseline | 0.521±0.015 | 0.577±0.041 |
| smote_plain (r=0.1) | 0.526±0.016 | 0.541±0.014 |
| smote_plain (r=0.5) | 0.518±0.014 | 0.531±0.014 |
| sw_smote (r=0.1) | 0.525±0.021 | 0.537±0.015 |
| sw_smote (r=0.5) | 0.521±0.016 | 0.521±0.015 |
| undersample_rus (r=0.1) | 0.522±0.012 | 0.576±0.045 |
| undersample_rus (r=0.5) | 0.531±0.014 | **0.597±0.053** |

#### SvmA

All conditions: within ≈ 0.50, cross ≈ 0.50 (chance-level). See Key Findings #2.

#### Lstm

| Condition | within | cross |
|---|---|---|
| baseline | 0.756±0.030 | 0.797±0.034 |
| smote_plain (r=0.5) | 0.721±0.021 | **0.812±0.045** |
| sw_smote (r=0.1) | 0.746±0.024 | 0.809±0.036 |
| sw_smote (r=0.5) | 0.687±0.041 | 0.782±0.046 |
| undersample_rus (r=0.5) | 0.754±0.028 | 0.812±0.030 |

(rows with N≤2 omitted)

### Key Findings (Preliminary)

> Findings are based on **current 75% coverage**; conclusions may shift after Lstm/SvmA fully complete.
> A formal hypothesis-test report (mirroring `exp2-analysis/hypothesis_test_report.md`)
> is deferred until 100% coverage.

1. **SvmW is the only prior baseline that responds meaningfully to imbalance handling.**
   `undersample_rus (r=0.5)` is consistently best on overall F2 (0.347), out-domain within F2 (0.377),
   and out-domain cross AUROC (0.597). SMOTE variants slightly underperform baseline for SvmW.

2. **SvmA degenerates to a constant-class predictor under domain_train.**
   AUROC ≈ 0.50 and F2 ≈ 0.16 (= class prior) across every condition, distance, and ratio.
   PSO optimization is converging to a trivial solution. This warrants investigation
   before publishing comparisons (likely a feature scaling / kernel / ν-bound issue).

3. **Lstm achieves high F2 via Recall-saturation.**
   Recall ≈ 0.999 and Precision ≈ 0.74 (≈ class prior) → F2 is dominated by recall, not by
   discrimination. AUROC 0.77–0.81 shows genuine but modest separation. Class-balanced thresholds
   were not tuned per fold, which inflates the apparent F2.

4. **Cross-domain generalization is non-trivial in this set:**
   for SvmW and Lstm, cross-domain AUROC is *higher* than within for several conditions
   (e.g., SvmW undersample_rus r=0.5: cross 0.597 vs within 0.531; Lstm sw_smote r=0.1:
   cross 0.809 vs within 0.746). This is opposite to exp2 (RF) and likely reflects the
   small held-out within-fold (15% = ~6–7 subjects) being noisier than the full 43-subject
   cross-domain test pool.

### Visualizations

Per-seed bar plots (235 total, generated by the collector):

```
results/analysis/exp3_prior_research/figures/png/split2/
├── SvmW/
│   ├── baseline/baseline_s{seed}.png            # 15 files
│   ├── smote_plain/smote_r{01,05}_s{seed}.png   # ~30 files
│   ├── sw_smote/smote_r{01,05}_s{seed}.png      # ~30 files
│   └── undersample_rus/rus_r{01,05}_s{seed}.png # ~30 files
├── SvmA/  (same layout)
└── Lstm/  (same layout, partial)
```

> **Missing (TODO):** seed-aggregated `*_summary.png` plots like exp2.
> The collector currently emits per-seed plots only; an exp2-style summary plotter
> (analogue of `plot_condition_seed_summaries.py`) is not yet implemented for prior-research models.

### Condition Naming Convention

Mapping between the `CONDITION` parameter passed to launchers and the actual tag names in eval files:

| CONDITION (launcher / env var) | Tag in eval files | Processing |
|---|---|---|
| `baseline` | `baseline_*` | No imbalance handling (class_weight only) |
| `smote_plain` | `smote_plain_*` | Global SMOTE (oversampling) |
| `smote` | `imbalv3_*` (a.k.a. `sw_smote`, subjectwise SMOTE) | Per-subject SMOTE |
| `undersample` | `undersample_rus_*` | Random Under-Sampling |

> Multiple naming aliases exist for the subjectwise variant: `swsmote_*` → `smote_subjectwise_*` → `imbalv3_*`.
> All refer to the same processing. The collector / docs label this `sw_smote` for readability.

### Analysis Scripts

| Script | Purpose |
|--------|---------|
| `scripts/python/analysis/domain/collect_split2_prior_research_metrics.py` | Eval JSON → per-condition CSV + per-seed bar plots (handles old split2 *and* new domain_train naming) |
| `scripts/python/visualization/visualize_prior_research.py` | Per-job model/training visualization |
| `scripts/python/visualization/visualize_prior_research_optuna.py` | Optuna study convergence/importance plots (SvmW) |
| `scripts/hpc/launchers/submit_exp3_seed_expansion.sh` | One-shot submitter for the 15-seed expansion (reads `/tmp/exp3_missing_jobs.txt`) |
| `scripts/hpc/launchers/submit_exp3_missing_24_v2.sh` | Targeted re-submission for the 24 canonical-seed jobs lost on 2026-04-23 |

### Output Structure

```
results/outputs/evaluation/{SvmW,SvmA,Lstm}/{JOB_ID}/{JOB_ID}[1]/
└── eval_results_{MODEL}_domain_train_prior_{MODEL}_{cond}_knn_{dist}_{dom}_..._s{seed}_{within|cross}.json

models/{SvmW,SvmA,Lstm}/{JOB_ID}/{JOB_ID}[1]/
├── {MODEL}_..._s{seed}_*.{pkl,keras}        # final model + scaler + features
├── (Lstm only) fold{1-5} models + history
└── (SvmW only) optuna_*.{pkl,csv,json}

results/analysis/exp3_prior_research/
├── figures/csv/split2/{Model}/{condition}/{prefix}_split2_metrics.csv
├── figures/png/split2/{Model}/{condition}/{prefix}_s{seed}.png
└── docs/optimization_methods.md
```

### Statistical Analysis Reports

Currently available:

- [Optimization Methods](exp3-analysis/optimization_methods.md) — SvmW Optuna / SvmA PSO / Lstm K-Fold CV summary
- [Operations Log](exp3-analysis/operations_log.md) — HPC re-submission history & known issues

Planned (deferred until 100% coverage):

- `exp3-analysis/hypothesis_test_report.md` — analogue of exp2's hypothesis-driven analysis
- `exp3-analysis/statistical_report.md` — across-condition statistical comparison
- `exp3-analysis/journal_paper_draft.md` — comparison-with-exp2 narrative

### Related Documents

- [Experiment Conditions](../conditions/03-prior-research-conditions.md) — Condition matrix and HPC resource settings
- [Experiment 2 Results](02-domain-results.md) — RF baseline for direct comparison
- [Reproducibility Guide](../reproducibility.md) — How to reproduce experiments
- [Prior Research Models](../../architecture/prior_research.md) — Model architecture details
- [Operations Log](exp3-analysis/operations_log.md) — Job re-submissions and known issues (Issues 1–6)
