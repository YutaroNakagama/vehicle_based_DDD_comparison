# Data Dictionary

**Driver Drowsiness Detection Comparison Dataset**

This document describes the files in the published bundle and how they relate to the
paper *"Class Imbalance and Domain Shift in Vehicle Dynamics-Based Driver Drowsiness
Detection"* (Nakagama, Ishii, Yoshizoe). The experiment is a full factorial design over
four design factors and the random seed, with a random forest (RF) classifier:

| Factor | Levels | Token in file names and tables |
|---|---|---|
| Distance metric `D` | MMD, DTW, W1 | `mmd`, `dtw`, `wasserstein` |
| Subject group `G` | In, Out | `in_domain`, `out_domain` |
| Training mode `M` | Cross, Within, Mixed | `source_only`, `target_only`, `mixed` |
| Rebalancing `R` | Baseline, RUS 0.1/0.5, SMOTE 0.1/0.5, SW-SMOTE 0.1/0.5 | `baseline_domain`, `undersample_rus`, `smote_plain`, `imbalv3` (+ `subjectwise`), with `ratio0.1`/`ratio0.5` |
| Seed `S` | 0, 1, 3, 7, 13, 42, 123, 256, 512, 999, 1337, 2024 | `s<seed>` |

3 × 2 × 3 × 7 = 126 factor cells × 12 seeds = **1,512 evaluations**. The runs were
executed on the JAIST HAKUSAN cluster in May 2026.

| | |
|---|---|
| Source repository | https://github.com/YutaroNakagama/vehicle_based_DDD_comparison |
| Source commit | recorded in `BUNDLE_INFO.json` (`git.commit`) |
| Upstream dataset | Aygun et al. (2024), Harvard Dataverse, [doi:10.7910/DVN/HMZ5RG](https://doi.org/10.7910/DVN/HMZ5RG) |
| Licence | Data: CC BY 4.0 · Code: MIT — see `LICENSE` |
| Integrity | `MANIFEST.sha256` (SHA-256 per file) |

---

## 1. Bundle layout

```
<bundle-root>/
├── BUNDLE_INFO.json        provenance: source commit, counts, sizes, allow/deny lists
├── MANIFEST.sha256         SHA-256 of every file in the bundle
├── DATADICTIONARY.md       this document
├── LICENSE
├── CITATION.cff
├── data/processed/common/                     feature tables, 87 files (Section 2)
├── results/outputs/evaluation/RF/             evaluation records, 1,512 files (Section 3)
└── results/analysis/exp2_domain_shift/
    ├── figures/csv/split2/                    metric tables (Section 4)
    ├── figures/svg/split2/journal_v2/         figures of the paper (3 PDF files)
    └── distance/                              distance matrices, In/Out lists (Section 5)
```

Trained models and per-run training records are not included: the pickled estimators
require an exact dependency-version match to load and execute arbitrary code on
unpickling, and the evaluation records hold the reported metrics.

---

## 2. `data/processed/common/` — feature tables

`processed_S0XXX_Y.csv` — one file per **subject-session** (87 files; 47 distinct
drivers, some of whom contributed two sessions). As in the paper, each subject-session is
treated as a distinct subject. Each row is one window of 3 s with a 1.5 s step (50 %
overlap); 70,841 windows in total. Subject identifiers are the pseudonymous codes of the
upstream deposit.

| Column group | Pattern | Columns |
|---|---|---|
| Index | `Timestamp` | 1 |
| Statistical / spectral window features | `{Signal}_{Statistic}`, signals `Steering`, `SteeringSpeed`, `Lateral`, `LongAcc`, `LaneOffset` | 90 |
| Smoothing / prediction-error features | `{signal}_{gaussian_smooth,std_dev,pred_error}` | 15 |
| Wavelet-packet energies (3 levels, 8 bands `AAA` … `DDD`) | `{Signal}_{Band}` | 40 |
| EEG band powers (8 channels × 5 bands) | `Channel_{N}_{Band} ({range})` | 40 |
| Labels and drowsiness indices | `KSS_Theta_Alpha_Beta`, `KSS_Theta_Alpha_Beta_percent`, `theta_alpha_over_beta`, `theta_alpha_over_beta_label` | 4 |

The RF reads the 145 columns `Steering_Range` … `LaneOffset_AAA`
(`src/utils/io/loaders.py`). EEG is used only to derive the labels. The binary target is
derived from `KSS_Theta_Alpha_Beta`: 1–5 → alert (0), 8–9 → drowsy (1); levels 6 and 7
and rows without a label are excluded (`src/config.py`, `KSS_LABEL_MAP`). With rows that
have a missing feature value removed, 61,750 windows remain (59,326 alert, 2,424
drowsy).

The electrode montage of `Channel_1` … `Channel_8` is not recorded in this repository;
consult the upstream deposit.

---

## 3. `results/outputs/evaluation/RF/` — evaluation records

```
results/outputs/evaluation/RF/<job_id>/<job_id>[1]/eval_results_RF_<mode>_<tag>.json
```

One JSON file per evaluation of the paper, 1,512 in total. `<job_id>` (`manual_YYYYMMDD_HHMMSS`)
identifies the cluster job; the condition is encoded in the file name, e.g.
`eval_results_RF_target_only_imbalv3_knn_mmd_in_domain_target_only_split2_subjectwise_ratio0.1_s42.json`
= Within, SW-SMOTE 0.1, MMD, In, seed 42.

**Selection.** The bundle contains exactly the records referenced by the per-evaluation
metric tables (Section 4.1). Six factor cells were run twice on the cluster; for these,
the tables keep the run with the later job id (`regenerate_all_conditions_may2026.py`),
and the bundle contains only that run.

| Key | Description |
|---|---|
| `accuracy`, `precision`, `recall`, `f1`, `confusion_matrix` | Computed from `clf.predict` on the test set, i.e. the calibrated classifier's decision at probability 0.5; `confusion_matrix` is `[[TN, FP], [FN, TP]]` |
| `roc_auc`, `auc_pr`, `roc_curve`, `pr_curve` | AUROC and AUPRC on the test set. If the AUROC of the predicted probabilities is below 0.5, the evaluation code inverts the probabilities (`1 − p`) before computing these values and `y_pred_proba` (`src/evaluation/models/common.py`) |
| `y_pred_proba` | Predicted probability of the drowsy class for each test window (after the inversion above, where it applies) |
| `thr`, `acc_thr`, `prec_thr`, `recall_thr`, `f1_thr`, `f2_thr`, `specificity_thr` | Metrics at the threshold `thr` chosen on the validation set (`threshold_source`) with the F-β criterion of `threshold_beta` |
| `precision_pos`, `recall_pos`, `f1_pos`, `support_pos`, `precision_neg`, `recall_neg`, `specificity`, `classification_report` | Per-class metrics at the same decision as `precision`/`recall` |
| `mode`, `tag`, `original_tag`, `timestamp`, `jobid_idx` | Provenance |
| `subject_list` | The 87 subject-sessions of the dataset (not the test group) |

---

## 4. Metric tables — `figures/csv/split2/`

### 4.1 Per-evaluation tables

One row per evaluation (factor cell × seed), 1,512 rows in total:

| File | Rebalancing `R` |
|---|---|
| `baseline_NEW_may2026/baseline_domain_split2_metrics_NEW_may2026.csv` | Baseline |
| `undersample_rus_NEW_may2026/undersample_rus_split2_metrics_NEW_may2026.csv` | RUS 0.1, 0.5 |
| `smote_plain_NEW_may2026/smote_plain_split2_metrics_NEW_may2026.csv` | SMOTE 0.1, 0.5 |
| `sw_smote_NEW_may2026/sw_smote_split2_metrics_NEW_may2026.csv` | SW-SMOTE 0.1, 0.5 |

| Column | Description |
|---|---|
| `job_id` | Cluster job of the evaluation record |
| `mode` | `source_only` = Cross, `target_only` = Within, `mixed` = Mixed |
| `distance` | `mmd`, `dtw`, `wasserstein` (W1) |
| `level` | `in_domain` = In, `out_domain` = Out (group of the test subjects) |
| `seed` | Random seed |
| `ratio` | Target ratio `r` of the rebalancing method (empty for Baseline) |
| `accuracy`, `precision`, `recall`, `f1` | Copied from the evaluation record |
| `f2` | F2 score computed from `precision` and `recall` of the record |
| `auc`, `auc_pr` | `roc_auc` and `auc_pr` of the record |

The `*_split2_metrics_v2.csv` files in `baseline/`, `undersample_rus/`, `smote_plain/` and
`sw_smote/` hold the same evaluations together with evaluations of additional seeds; the
analysis scripts restrict them to the 12 seeds above, which yields the same 1,512 rows.

### 4.2 Derived tables

| File | Contents |
|---|---|
| `paper_table_NEW_may2026.csv` | Mean and standard deviation per (R, M) over D, G and S (n = 72): the performance table of the paper |
| `sensitivity/sobol_indices.csv` | First-order (`S1`) and total (`ST`) Sobol indices per factor and metric, with 95 % bootstrap intervals (`*_lo`, `*_hi`; B = 2,000 resamples of the seeds) |
| `sensitivity/variance_decomposition.csv` | The underlying variance components |

---

## 5. `distance/` — subject grouping

`rankings/split2/knn/{mmd,dtw,wasserstein}_{in,out}_domain.txt` list the subject-sessions
of the In (44) and Out (43) groups used by the runs: each subject-session is scored by its
mean distance to its k = 5 nearest neighbours under the metric, and the 43 with the
highest scores form the Out group. `ranks43_summary.txt` summarises the split.
`matrices_normalized/{metric}_matrix.npy` are the 87 × 87 distance matrices (computed on
globally standardised features) from which the lists were derived, with the subject
order in `{metric}_subjects.json` (`regenerate_split2_normalized.py`).

The grouping is at subject-session level, so a driver with two sessions can have one
session in each group.

---

## 6. Reproduction

1. Obtain the code release at the commit recorded in `BUNDLE_INFO.json` and install it as
   described in its `README.md` (`requirements-lock.txt` pins the versions of the cluster
   environment).
2. Extract the bundle over the repository root.
3. Run the analysis scripts listed in the README ("Scripts for reproducing the paper"):
   `regenerate_all_conditions_may2026.py` rebuilds the per-evaluation tables from the
   evaluation records, and the remaining scripts rebuild the derived tables and figures.

Verify integrity first:

```bash
sha256sum -c MANIFEST.sha256
```

---

## 7. Notes

1. **Cite the upstream deposit.** doi:10.7910/DVN/HMZ5RG is released under CC0 1.0, so it
   places no licensing restriction on this release; per Harvard Dataverse community norms,
   cite it alongside this bundle (see `LICENSE` and `CITATION.cff`).
2. **Electrode montage unavailable** (Section 2).

---

## 8. Contact

Yutaro Nakagama — Japan Advanced Institute of Science and Technology (JAIST). See
`CITATION.cff`.
