# Data Dictionary

**Driver Drowsiness Detection Comparison Dataset**

This document describes every file in the published bundle: what it contains, how
its columns and keys are defined, and — most importantly — how to decode the
experimental condition that is encoded in each artefact's **filename**.

| | |
|---|---|
| Bundle version | 1.0.0 |
| Source repository | https://github.com/YutaroNakagama/vehicle_based_DDD_comparison |
| Source commit | recorded in `BUNDLE_INFO.json` (`git.commit`) |
| Upstream dataset | Aygun et al. (2024), Harvard Dataverse, [doi:10.7910/DVN/HMZ5RG](https://doi.org/10.7910/DVN/HMZ5RG) |
| Licence | Data: CC BY 4.0 · Code: MIT — see `LICENSE` |
| Integrity | `MANIFEST.sha256` (SHA-256 per file) |

> **Read this first.** The artefact filenames in `results/outputs/` are not
> self-describing. Their meaning is a *convention* enforced by the launcher
> scripts in the source repository, not a schema validated by any parser. This
> bundle is therefore only fully interpretable **together with the matching code
> release**. Section 3 documents that convention so the bundle can be read
> without reverse-engineering the launchers.

---

## 1. Scope — what is and is not included

### Included

| Path | Contents | Approx. size |
|---|---|---|
| `data/processed/common/` | 87 per-subject-session feature tables (210 columns) | 266 MB |
| `data/processed/Lstm/` | 87 per-subject-session tables, LSTM windowing | 251 MB |
| `results/outputs/training/` | 1,946 training records (JSON + flattened CSV twin) | 3.6 GB |
| `results/outputs/evaluation/` | 4,204 evaluation records (JSON + CSV) | 587 MB |
| `results/analysis/` | Aggregated tables, Sobol indices, journal figures | 14 MB |

### Deliberately excluded

| Excluded | Reason |
|---|---|
| `models/` (~115 GB: 22,770 `.pkl`, 9,177 `.keras`) | Pickled estimators require an exact dependency-version match to load, execute arbitrary code on unpickling, and add nothing the recorded metrics do not already establish. Retrain from the code release instead (Section 8). |
| `results/outputs/hpc/` (1.4 GB archives) | HPC-cluster artefacts are reference-only under project policy and are **not** the reported results. |
| `data/processed/SvmA/`, `data/processed/SvmW/` | Identical to `data/processed/common/` by design — these models are differentiated by column range, not by a separate table (Section 5.2). |
| `data/processed/_hpc_common_pull/` | HPC-derived; see above. |
| `data/processed/common_fixedlabels_bak/` | Label-correction backup of undetermined canonicity (Section 5.2). |
| `data/interim/` | Intermediate preprocessing stages, regenerable from the code. |
| `logs/`, `docs/experiments/results/**` | Execution logs and unpublished/internal write-ups. |

---

## 2. Bundle layout

```
<bundle-root>/
├── BUNDLE_INFO.json        provenance: source commit, counts, sizes, allow/deny lists
├── MANIFEST.sha256         SHA-256 of every file in the bundle
├── DATADICTIONARY.md       this document
├── LICENSE
├── CITATION.cff
├── data/
│   └── processed/
│       ├── common/         processed_S0XXX_Y.csv   (87 files)
│       └── Lstm/           processed_S0XXX_Y.csv   (87 files)
└── results/
    ├── analysis/           aggregated tables + journal figures
    └── outputs/
        ├── training/<Model>/<jobID>/<jobID>[<idx>]/train_results_*.json|.csv
        └── evaluation/<Model>/<jobID>/<jobID>[<idx>]/eval_results_*.json|.csv
```

---

## 3. Filename tag grammar — how to decode a result file

### 3.1 The pattern

Artefact basenames are built in
[`src/utils/io/savers.py`](https://github.com/YutaroNakagama/vehicle_based_DDD_comparison/blob/main/src/utils/io/savers.py)
(lines 313 and 423) as:

```
train_results_{model}_{mode}_{tag}.json
 eval_results_{model}_{mode}_{tag}_{eval_type}.json
```

Only `{model}` and `{mode}` are validated by the CLI. **`{tag}` is a free-form
string** supplied via `--tag` by the launcher script that submitted the run. Its
internal structure is the convention documented below.

The observed tag layout is:

```
{experiment}_{model}_{variant}_{ranking}_{distance}_{domain}_{mode}_{split}_[{granularity}]_[ratio{r}]_s{seed}
```

### 3.2 Field reference

| # | Field | Values observed on disk | Meaning |
|---|---|---|---|
| 1 | `model` | `RF`, `Lstm`, `SvmA`, `SvmW` (CLI allows 16 total) | Model family. `SvmA` = ANFIS+PSO-weighted SVM; `SvmW` = wavelet-feature SVM; `Lstm` = BiLSTM with attention. |
| 2 | `mode` | `pooled`, `target_only`, `source_only`, `joint_train`, `mixed`, `domain_train` | Training/evaluation regime. Defined in [`train_cli_helpers.py:171`](https://github.com/YutaroNakagama/vehicle_based_DDD_comparison/blob/main/src/utils/cli/train_cli_helpers.py#L171). |
| 3 | `experiment` | `baseline`, `imbalv3`, `prior`, `iv25smote` | Experiment campaign. `prior` = prior-research replication (Exp3); `imbalv3` = imbalance campaign v3; `baseline` = no resampling (class weights only); `iv25smote` = IV2025 SMOTE variant. |
| 4 | `ranking` | `knn` | Subject-ranking method used to partition subjects into domain groups. (The codebase also implements LOF, Isolation Forest, UMAP-centroid and mean/median distance; only `knn` appears in this bundle.) |
| 5 | `distance` | `wasserstein`, `mmd`, `dtw` | Distance metric used to rank subjects by domain dissimilarity. |
| 6 | `domain` | `in_domain`, `out_domain` | Which ranked subject list supplied the **training** target subjects. |
| 7 | `split` | `split2` | Subject-grouping scheme (Section 4.3). |
| 8 | `granularity` | `subjectwise` (or absent) | Present when resampling was applied **per subject** (`--subject_wise_oversampling`) rather than pooled. |
| 9 | `ratio` | `ratio0.3`, `ratio0.5` | Target minority/majority ratio for resampling (`--target_ratio`). |
| 10 | `seed` | `s0 s1 s3 s5 s7 s9 s11 s13 s17 s23 s31 s42 s47 s99 s101 s123 s256 s333 s512 s777 s1000 s1337 s2024 s2025 s2718 s3407 s9999 s31415` | Random seed. |
| 11 | `eval_type` | `within`, `cross` | **Evaluation files only.** Which subject list forms the *test* set. |

Resampling method vocabulary accepted by the CLI ([`train_cli_helpers.py:230`](https://github.com/YutaroNakagama/vehicle_based_DDD_comparison/blob/main/src/utils/cli/train_cli_helpers.py#L230)):
`smote`, `adasyn`, `borderline`, `smote_tomek`, `smote_enn`, `smote_rus`, `jitter`,
`scale`, `jitter_scale`, `undersample_rus`, `undersample_tomek`.

### 3.3 `within` vs `cross` — the critical distinction

`--eval_type` ([`evaluate.py:77`](https://github.com/YutaroNakagama/vehicle_based_DDD_comparison/blob/main/scripts/python/evaluation/evaluate.py#L77))
selects which ranked subject list becomes the test set:

* **`within`** — tested on the **same** domain group the model trained on
  (e.g. trained on `wasserstein_in_domain.txt`, tested on `wasserstein_in_domain.txt`).
* **`cross`** — tested on the **opposite** domain group
  (`wasserstein_out_domain.txt`). This is the domain-shift measurement.

The subject lists live in
`results/analysis/exp2_domain_shift/distance/rankings/split2/{ranking}/{distance}_{in|out}_domain.txt`
and are shipped in this bundle, so the exact test population of any run is recoverable.

### 3.4 Worked decodings

**(a)** `results/outputs/training/Lstm/177889225781328428/177889225781328428[1]/`
`train_results_Lstm_domain_train_prior_Lstm_imbalv3_knn_mmd_in_domain_domain_train_split2_subjectwise_ratio0.3_s42.json`

| Token | Value |
|---|---|
| artefact | training record |
| model | `Lstm` |
| mode | `domain_train` |
| tag → experiment | `prior` (Exp3, prior-research replication) |
| tag → model echo | `Lstm` |
| tag → variant | `imbalv3` |
| tag → ranking | `knn` |
| tag → distance | `mmd` |
| tag → domain | `in_domain` (trained on the in-domain subject list) |
| tag → mode echo | `domain_train` |
| tag → split | `split2` |
| tag → granularity | `subjectwise` (per-subject SMOTE) |
| tag → ratio | `0.3` |
| tag → seed | `42` |

**(b)** `eval_results_Lstm_domain_train_prior_Lstm_imbalv3_knn_mmd_in_domain_domain_train_split2_subjectwise_ratio0.3_s42_cross.json`
— the evaluation of run (a), **tested on the out-domain subject list**.

**(c)** `train_results_RF_mixed_imbalv3_knn_wasserstein_in_domain_mixed_split2_subjectwise_ratio0.5_s1337.json`
— Random Forest, `mixed` mode, imbalance-v3 campaign, kNN ranking on Wasserstein
distance, trained on in-domain subjects, split2, per-subject resampling at
ratio 0.5, seed 1337.

**(d)** `models/RF/<jobID>/<jobID>[<idx>]/RF_source_only_imbalance_knn_mmd_out_domain_smote_14572594_1_1.pkl`
— the model-artefact naming convention (models are **not** in this bundle):
`{model}_{mode}_{experiment}_{ranking}_{distance}_{domain}_{sampling}_{jobID}_{arrayIdx}_{seed}`.

### 3.5 Directory layers above the filename

```
results/outputs/{training|evaluation}/{Model}/{jobID}/{jobID}[{arrayIdx}]/
```

`jobID` is the PBS job identifier for cluster runs and a synthetic numeric id for
local runs; `arrayIdx` is the PBS array-job index. These identify *when* a run was
executed, not *what* it was — the condition lives entirely in the filename tag.

### 3.6 Realized experimental grid

Counts over the 1,946 training records in this bundle:

| Dimension | Distribution |
|---|---|
| Model | `Lstm` 833 · `RF` 628 · `SvmW` 257 · `SvmA` 228 |
| Experiment | `imbalv3` 1,382 · `prior` 826 · `baseline` 199 · `iv25smote` 76 |
| Distance | `wasserstein` 1,272 · `mmd` 262 · `dtw` 223 |
| Ratio | `ratio0.5` 1,347 · `ratio0.3` 313 |
| Seeds | 27 distinct values (Section 3.2) |

*(Experiment and distance tokens are counted by substring occurrence and may
co-occur across the model split; treat them as marginal distributions.)*

> **Why ratios are 0.3/0.5 and not 0.1.** A minority ratio of 0.1 is infeasible for
> the LSTM configuration (its natural minority share already exceeds that level,
> which makes the resampler reject the target). 0.3/0.5 is the deliberate common
> feasible set across all four model families.

---

## 4. `data/processed/` — feature tables

### 4.1 Files

`processed_S0XXX_Y.csv` — one file per **subject-session**.

| Property | Value |
|---|---|
| Files | 87 |
| Unique subjects | 47 (`S0XXX`) |
| Session index `Y` | `1` or `2` |
| Columns | 210 |
| Rows | ≈855 per file (varies with recording length) |
| Source sampling rate | Vehicle (SIMlsl): 60 Hz · EEG: 500 Hz |
| Windowing (`common`) | 3 s window, 1.5 s step (50 % overlap) |
| Windowing (`Lstm`) | 1 s window, 0.1 s step |

Each row is one feature window. Subject identifiers are the pseudonymous codes
used by the upstream deposit; no direct identifiers are present.

### 4.2 Column groups (210 total)

**(A) Index — 1 column**

| Column | Description |
|---|---|
| `Timestamp` | Window start time |

**(B) Statistical / spectral window features — 118 columns**

Pattern: `{Signal}_{Statistic}`

*Signals (5):* `Steering`, `SteeringSpeed`, `LaneOffset`, `LongAcc`, `Lateral`

*Statistics (22, in column order):*
`Mean`, `Variance`, `StdDev`, `Max`, `Min`, `Range`, `Energy`,
`ZeroCrossingRate`, `Quartile25`, `Median`, `Quartile75`, `Skewness`,
`Kurtosis`, `KatzFractalDim`, `ShannonEntropy`, `FreqVar`, `SpectralEntropy`,
`SpectralFlux`, `FreqCOG`, `DominantFreq`, `AvgPSD`, `SampleEntropy`

The first 13 are time-domain moments and order statistics; `KatzFractalDim` and
`ShannonEntropy` are complexity measures; the remaining seven are computed from
the window's power spectrum. `SampleEntropy` is a regularity measure.

**(C) Wavelet-packet energies — 40 columns**

Pattern: `{Signal}_{Band}`

*Signals (5):* `SteeringSpeed`, `LaneOffset`, `SteeringWheel`,
`LongitudinalAccel`, `LateralAccel`

*Bands (8):* `AAA`, `AAD`, `ADA`, `ADD`, `DAA`, `DAD`, `DDA`, `DDD`

Three-level wavelet packet decomposition; `A` = approximation, `D` = detail, read
left to right by decomposition level. `AAA` is the lowest-frequency sub-band,
`DDD` the highest.

> Note the two naming families for the same physical channels: the statistical
> block uses `LongAcc`/`Lateral`, the wavelet block uses
> `LongitudinalAccel`/`LateralAccel`/`SteeringWheel`. This is a historical
> artefact of two feature extractors, not two different signals.

**(D) Smoothing / prediction-error features — 15 columns**

Pattern: `{signal}_{measure}` where *signal* ∈ {`steering`, `steering_speed`,
`lat_acc`, `long_acc`, `lane_offset`} (lower-case) and *measure* ∈
{`_mean`, `_std_dev`, `_pred_error`}.

`_pred_error` is the residual of a short-horizon predictor over the window — a
proxy for behavioural irregularity.

**(E) EEG band powers — 40 columns**

Pattern: `Channel_{N}_{Band} ({range})` for `N` = 1…8, e.g.
`Channel_3_Alpha (8-13 Hz)`.

*Bands (5):* `Delta (0.5-4 Hz)`, `Theta (4-8 Hz)`, `Alpha (8-13 Hz)`,
`Beta (13-30 Hz)`, `Gamma (30-100 Hz)`

**The electrode montage (which scalp position each `Channel_N` corresponds to) is
not recorded in this repository** — consult the upstream deposit's documentation.

EEG is present as the **label source**, not as a model input: the compared models
are vehicle-dynamics-only. Do not treat these columns as available features when
reproducing the benchmark.

**(F) Labels and drowsiness indices — 4 columns**

| Column | Description |
|---|---|
| `KSS_Theta_Alpha_Beta` | Karolinska Sleepiness Scale value (1–9) estimated from the EEG θ/α/β power ratio |
| `KSS_Theta_Alpha_Beta_percent` | The same index expressed as a percentile |
| `theta_alpha_over_beta` | Raw (θ+α)/β ratio |
| `theta_alpha_over_beta_label` | **Binary target** derived by thresholding the above |

Positive (drowsy) prevalence is ≈3.9 % — the class imbalance this work studies.

> **Label caveat.** A label-correction pass was applied during the project's
> history (see `logs/fix_lstm_event_labels.log` in the source repository, not
> shipped here). The `data/processed/common_fixedlabels_bak/` directory in the
> source tree holds an alternative label set whose canonicity **has not been
> established**; it is excluded from this bundle. If you need to audit label
> provenance, raise it with the depositor rather than assuming the shipped labels
> are the pre- or post-correction set.

### 4.3 `split2` — the unit of domain partitioning

`split2` partitions the **87 records** into two domain groups by kNN outlier
score (k = 5) over normalised feature vectors, cut at the median:

| Group | Records |
|---|---|
| `in_domain` | 44 (lowest kNN scores — most typical) |
| `out_domain` | 43 (highest kNN scores — most dissimilar) |

**The unit of partitioning is the subject-session (`S0XXX_Y`), not the subject.**
The 87 records comprise 47 distinct subjects contributing one or two sessions
each, and each session is scored and assigned independently. Consequently nine
subjects have one session in `in_domain` and the other in `out_domain`:

```
S0116  S0167  S0181  S0185  S0190  S0196  S0202  S0204  S0205
```

This is a documented property of the released grouping, and the per-session
assignments are reproduced in the analysis notes
(`docs/experiments/results/exp2-analysis/actual_distance_concordance.md` in the
source repository lists them record by record).

**What it means for interpreting `_cross` results.** For these nine subjects the
cross-domain evaluation is between-session rather than strictly
between-subject, so `_cross` measures transfer across the *domain grouping*
rather than across *individuals* for that portion of the data. Readers computing
strictly subject-independent generalisation figures should account for the
overlap. Everything needed to do so is shipped: the per-group record lists live
in `results/analysis/exp2_domain_shift/distance/rankings/split2/knn/`, and the
subject of any record is its identifier with the `_Y` session suffix removed, so
the overlap can be quantified, or the affected subjects excluded, directly from
the bundle.

> **Note for readers comparing against the manuscript.** Draft manuscript text in
> the source repository describes the cohort as "87 subjects" and states that
> "every subject is assigned to exactly one domain group". Relative to the data
> released here those phrasings should be read as referring to the 87 *records*;
> the record-level assignment above is authoritative for this bundle.

---

## 5. Dataset variants

### 5.1 Variant equivalence

| Source directory | Status | In bundle |
|---|---|---|
| `data/processed/common/` | Canonical shared feature table | ✅ shipped |
| `data/processed/SvmA/` | **Byte-identical** to `common/` (all 87 files) | ❌ deduplicated |
| `data/processed/SvmW/` | **Byte-identical** to `common/` (all 87 files) | ❌ deduplicated |
| `data/processed/Lstm/` | Distinct — LSTM windowing (1 s / 0.1 s), 17 columns, ≈9,463 rows | ✅ shipped |
| `data/processed/common_fixedlabels_bak/` | Distinct labels, canonicity undetermined | ❌ excluded |
| `data/processed/_hpc_common_pull/` | 44 files, HPC-derived | ❌ excluded |

Equivalence was verified by SHA-256 over all 87 files in each directory, and the
packaging script re-verifies it at build time rather than trusting this table.

### 5.2 How the classical models are differentiated — by column range, not by window

`common/`, `SvmA/` and `SvmW/` being identical is **by design, not an omission**.
The RF, SvmA and SvmW pipelines all read the same `data/processed/common/`
table; each model is then given a distinct **contiguous column range** of it at
load time (`src/utils/io/loaders.py`):

| Model | Column range | Feature count | Character |
|---|---|---|---|
| `RF` | `Steering_Range` … `LaneOffset_AAA` | 160 | Full vehicle-dynamics pool |
| `SvmA` | `Steering_Range` … `SteeringSpeed_SampleEntropy` | 39 | Steering + steering-speed statistics (Arefnezhad et al., 2019) |
| `SvmW` | `SteeringWheel_DDD` … `SteeringWheel_AAA` | 8 | Steering-wheel wavelet-packet band energies (Zhao et al.) |
| `Lstm` | `speed_std_dev` … `steering_speed_mean` | 15 | Smoothing / prediction-error features, own table |

Model differentiation therefore happens at **feature selection**, downstream of
feature generation. Shipping one physical copy of the shared table loses no
information; a re-user reconstructs any model's input by slicing the documented
range.

**Consequence for the window configuration.** The repository's
`MODEL_WINDOW_CONFIG` also declares a model-specific window (`common`/`SvmA`
3 s / 1.5 s, `SvmW` 10 s / 5 s, `Lstm` 1 s / 0.1 s). That table governs the
*model-specific preprocessing path*, which is **not** the path that produced the
released tables: every column consumed by RF, SvmA and SvmW in this bundle was
computed on the shared `common` grid of **3 s windows with a 1.5 s step**, and
`Lstm` on its own 1 s / 0.1 s grid. Read the `SvmW` 10 s / 5 s entry as a
configuration option of the codebase rather than a description of the data
released here.

---

## 6. `results/outputs/` — per-run records

Every run emits a `.json` and a `.csv` twin with the same basename. **The CSV is a
flattened rendering of the JSON** (header row + one data row; nested keys joined
with `.`, e.g. `class_weight.0`, `train.accuracy`). Large CSVs (up to ~5 MB) are
runs whose JSON contains per-sample probability vectors and ROC/PR curve arrays.

### 6.1 `training/` — `train_results_*.json`

Common to all model families:

| Key | Description |
|---|---|
| `train`, `val`, `test` | Metric blocks per split: `accuracy`, `precision`, `recall`, `f1`, `confusion_matrix`, `roc_auc`, `auc_pr` |
| `model_name` | Model family |
| `mode` | Training regime |
| `tag` | The full condition tag (Section 3) |
| `suffix` | Filename suffix actually used |

`Lstm` additionally records:

| Key | Description |
|---|---|
| `cv_f1_scores` | Per-fold F1 across cross-validation folds |
| `cv_mean_f1` | Mean of the above |
| `best_fold` | Index of the selected fold |
| `class_weight` | Per-class weights applied (`{"0": …, "1": …}`) |

### 6.2 `evaluation/` — `eval_results_*.json`

Shared across families:

| Key | Description |
|---|---|
| `accuracy`, `precision`/`recall`/`f1` | Headline metrics at the default threshold |
| `confusion_matrix` | 2×2, ordered `[[TN, FP], [FN, TP]]` |
| `roc_auc`, `auc_pr` | AUROC and AUPRC. **AUPRC is the primary metric** for this work; AUROC is retained for comparability but is known to be optimistic under ≈4 % prevalence. |
| `classification_report` | scikit-learn report: per-class precision/recall/F1/support plus `macro avg` and `weighted avg` |
| `precision_pos`, `recall_pos`, `f1_pos`, `support_pos` | Positive (drowsy) class, broken out |
| `precision_neg`, `recall_neg`, `specificity` | Negative class |
| `custom_threshold`, `threshold_source`, `threshold_beta` | Decision threshold actually applied, how it was chosen, and the β used for F-β threshold selection |
| `thr`, `acc_thr`, `prec_thr`, `recall_thr`, `f1_thr`, `f2_thr`, `specificity_thr` | Metrics recomputed at the tuned threshold |
| `subject_list` | The subject-sessions that formed this test set — the definitive record of the evaluation population |
| `mode`, `tag`, `original_tag`, `eval_type` | Condition provenance |
| `distance`, `level`, `ranking_method`, `distance_metric` | Domain-partition provenance |
| `timestamp`, `jobid_idx` | Execution provenance |

Family-specific additions:

| Family | Extra keys |
|---|---|
| `RF`, `SvmW` | `roc_curve`, `pr_curve`, `y_pred_proba`, `mse`, `decision_threshold_default`, `pred_pos_rate_at_0p5` |
| `SvmA` | `f1_score` (instead of `f1`), threshold sweep block |
| `Lstm` | `loss` (Keras validation loss) |

---

## 7. `results/analysis/` — derived artefacts

| Subtree | Contents |
|---|---|
| `exp1/IV2025_preprint/` | Exp1 (imbalance-only) preprint artefacts |
| `exp2_domain_shift/distance/` | Subject-wise and group-wise distance matrices; `rankings/split2/knn/{mmd,wasserstein,dtw}_{in,out}_domain.txt` — the subject lists referenced by every tag |
| `exp2_domain_shift/figures/csv/` | Aggregated per-strategy metric tables, e.g. `paper_table_NEW_may2026.csv` with columns `strategy, strategy_short, ratio, mode, n, f2_mean, auc_mean, auc_pr_mean, f2_std, auc_std, auc_pr_std`; `sensitivity/sobol_indices*.csv` with `S1, S1_lo, S1_hi, ST, ST_lo, ST_hi` |
| `exp2_domain_shift/figures/{png,svg}/split2/journal_v2/` | Journal figures `fig1_pipeline_overview` … `fig8_seed_convergence` plus OFAT supplements |
| `exp3_prior_research/figures/` | Exp3 replication figures |
| `exp3_verification/` | Pooled ROC verification outputs |

### 7.1 Derivation chain

```
data/processed/               (feature tables, this bundle)
        │  scripts/python/train/…            ← code release
        ▼
results/outputs/training/     (per-run records, this bundle)
        │  scripts/python/evaluation/evaluate.py
        ▼
results/outputs/evaluation/   (per-run records, this bundle)
        │  src/analysis/**, scripts/python/analysis/**
        ▼
results/analysis/             (aggregated tables, Sobol indices, this bundle)
        │  scripts/python/visualization/**
        ▼
journal figures               (this bundle)
```

Each arrow is a script in the code release. Because the tag grammar is a launcher
convention rather than a schema, **the aggregation step is the only authoritative
decoder of the filenames** — which is why the data and the code must be cited at
the same version.

---

## 8. Reproduction

1. Obtain the code release at the commit recorded in `BUNDLE_INFO.json`.
2. Create the environment from `requirements.txt` (Python 3.11 reference).
3. Extract this bundle over the repository root so that `data/processed/` and
   `results/outputs/` land in place.
4. Re-run the analysis and figure scripts (Section 7.1) to regenerate
   `results/analysis/` and verify it matches the shipped copy.
5. To reproduce *training* rather than analysis, re-run the launchers with the
   condition decoded from a target filename (Section 3). Model binaries are not
   shipped and must be retrained; seeds are recorded in the tag, but exact
   bit-level reproduction also depends on library versions and, for `Lstm` and
   the cuML `SvmA` runs, on GPU non-determinism.

Verify integrity first:

```bash
sha256sum -c MANIFEST.sha256
```

---

## 9. Known limitations

1. **The domain split is at record (subject-session) level.** Nine subjects have
   one session in each group (Section 4.3), so `_cross` results measure transfer
   across the domain grouping rather than strictly across individuals for that
   portion of the data. The information needed to quantify or exclude the overlap
   ships with the bundle.
2. **Label provenance.** A label-correction pass occurred during the project's
   history; the canonicity of the alternative label set is unresolved (Section 4.2).
3. **One shared feature grid for the classical models.** RF, SvmA and SvmW are
   differentiated by column range over a single 3 s / 1.5 s table, not by
   model-specific windowing (Section 5.2). The `SvmW` 10 s / 5 s entry in
   `MODEL_WINDOW_CONFIG` describes a codebase option, not this data.
4. **Evaluation-protocol caveat carried over from the source project.** Under the
   `within` and `mixed` modes the training split (label-stratified temporal) and
   the evaluation split are not drawn on coinciding boundaries. The source
   project adopted this deliberately to keep results comparable across its
   earlier papers and records it as a known limitation; de-leaked re-tests are
   kept as a robustness appendix rather than as the headline protocol. `cross` is
   not affected, since it trains on a different domain group. Treat `within` and
   `mixed` figures as protocol-consistent with the associated manuscripts rather
   than as clean held-out estimates.
5. **Cite the upstream deposit.** The source deposit doi:10.7910/DVN/HMZ5RG is
   released under CC0 1.0, so it places no licensing restriction on this
   release. Per Harvard Dataverse community norms and ordinary scholarly
   practice, cite it alongside this bundle (see `LICENSE` §3.1 and
   `CITATION.cff`).
6. **Electrode montage unavailable.** `Channel_1`…`Channel_8` cannot be mapped to
   scalp positions from this bundle alone.
7. **Marginal grid counts.** Section 3.6 reports substring frequencies, not a
   fully cross-tabulated design matrix; not every cell of the nominal grid is
   populated with the same number of seeds.
8. **HPC artefacts excluded** as reference-only; any figure that would require
   them is not reproducible from this bundle alone.

---

## 10. Contact

Yutaro Nakagama — School of Information Science, Japan Advanced Institute of
Science and Technology (JAIST). See `CITATION.cff`.
