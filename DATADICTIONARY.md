# Data Dictionary

**Driver Drowsiness Detection Comparison Dataset**

This document describes the files in the published bundle and how they relate to the
paper *"Class Imbalance and Domain Shift in Vehicle Dynamics-Based Driver Drowsiness
Detection"* (Nakagama, Ishii, Yoshizoe). The experiment is a full factorial design over
four design factors and the random seed, with a random forest (RF) classifier:

| Factor | Levels |
|---|---|
| Distance metric `D` | MMD, DTW, W1 (`mmd`, `dtw`, `wasserstein`) |
| Subject group `G` | In, Out (`in_domain`, `out_domain`) |
| Training mode `M` | Cross, Within, Mixed (`source_only`, `target_only`, `mixed`) |
| Rebalancing `R` | Baseline, RUS 0.1/0.5, SMOTE 0.1/0.5, SW-SMOTE 0.1/0.5 |
| Seed `S` | 0, 1, 3, 7, 13, 42, 123, 256, 512, 999, 1337, 2024 |

3 × 2 × 3 × 7 = 126 factor cells × 12 seeds = **1,512 evaluations**.

| | |
|---|---|
| Source repository | https://github.com/YutaroNakagama/vehicle_based_DDD_comparison |
| Source commit | recorded in `BUNDLE_INFO.json` (`git.commit`) |
| Upstream dataset | Aygun et al. (2024), Harvard Dataverse, [doi:10.7910/DVN/HMZ5RG](https://doi.org/10.7910/DVN/HMZ5RG) |
| Licence | Data: CC BY 4.0 · Code: MIT — see `LICENSE` |
| Integrity | `MANIFEST.sha256` (SHA-256 per file) |

---

## 1. Scope

| Path | Contents |
|---|---|
| `data/processed/common/` | 87 per-subject-session feature tables (Section 3) |
| `results/analysis/exp2_domain_shift/figures/csv/split2/` | Per-evaluation metric tables of the 1,512 evaluations and the tables derived from them (Section 4) |
| `results/analysis/exp2_domain_shift/distance/` | Inter-subject distance matrices and the In/Out subject lists (Section 5) |
| `results/analysis/exp2_domain_shift/figures/{png,svg}/split2/journal_v2/` | Figures of the paper |

Trained model binaries are not included: pickled estimators require an exact
dependency-version match to load and execute arbitrary code on unpickling. Retrain them
with the code release if needed.

---

## 2. Bundle layout

```
<bundle-root>/
├── BUNDLE_INFO.json        provenance: source commit, counts, sizes, allow/deny lists
├── MANIFEST.sha256         SHA-256 of every file in the bundle
├── DATADICTIONARY.md       this document
├── LICENSE
├── CITATION.cff
├── data/processed/common/  processed_S0XXX_Y.csv   (87 files)
└── results/analysis/exp2_domain_shift/
    ├── distance/           distance matrices, In/Out subject lists
    └── figures/            metric tables (csv/), figures (png/, svg/)
```

---

## 3. `data/processed/common/` — feature tables

`processed_S0XXX_Y.csv` — one file per **subject-session** (87 files; 47 distinct
drivers, some of whom contributed two sessions). As in the paper, each subject-session is
treated as a distinct subject. Each row is one window of 3 s with a 1.5 s step (50 %
overlap). Subject identifiers are the pseudonymous codes of the upstream deposit.

| Column group | Pattern |
|---|---|
| Index | `Timestamp` |
| Statistical / spectral window features | `{Signal}_{Statistic}` for the signals `Steering`, `SteeringSpeed`, `LaneOffset`, `LongAcc`, `Lateral` |
| Wavelet-packet energies (3 levels, 8 bands `AAA` … `DDD`) | `{Signal}_{Band}` |
| Smoothing / prediction-error features | `{signal}_{mean,std_dev,pred_error,…}` |
| EEG band powers (8 channels × 5 bands) | `Channel_{N}_{Band} ({range})` |
| Labels and drowsiness indices | `KSS_Theta_Alpha_Beta`, `KSS_Theta_Alpha_Beta_percent`, `theta_alpha_over_beta`, `theta_alpha_over_beta_label` |

The number of columns of each group is recorded in `BUNDLE_INFO.json`.

The RF pipeline reads the contiguous column range `Steering_Range` … `LaneOffset_AAA`
(`src/utils/io/loaders.py`). EEG is used only to derive the labels, not as a model input.
The binary target is derived from `KSS_Theta_Alpha_Beta`: 1–5 → alert (0), 8–9 → drowsy
(1); levels 6 and 7 are excluded (`src/config.py`, `KSS_LABEL_MAP`).

The electrode montage of `Channel_1` … `Channel_8` is not recorded in this repository;
consult the upstream deposit.

---

## 4. Metric tables — `figures/csv/split2/`

### 4.1 Per-evaluation tables

One row per evaluation (factor cell × seed):

| File | Rebalancing `R` |
|---|---|
| `baseline_NEW_may2026/baseline_domain_split2_metrics_NEW_may2026.csv` | Baseline |
| `undersample_rus_NEW_may2026/undersample_rus_split2_metrics_NEW_may2026.csv` | RUS 0.1, 0.5 |
| `smote_plain_NEW_may2026/smote_plain_split2_metrics_NEW_may2026.csv` | SMOTE 0.1, 0.5 |
| `sw_smote_NEW_may2026/sw_smote_split2_metrics_NEW_may2026.csv` | SW-SMOTE 0.1, 0.5 |

| Column | Description |
|---|---|
| `job_id` | Identifier of the run that produced the evaluation |
| `mode` | `source_only` = Cross, `target_only` = Within, `mixed` = Mixed |
| `distance` | `mmd`, `dtw`, `wasserstein` (W1) |
| `level` | `in_domain` = In, `out_domain` = Out (group of the test subjects) |
| `seed` | Random seed |
| `ratio` | Target ratio `r` of the rebalancing method (empty for Baseline) |
| `accuracy`, `precision`, `recall`, `f1`, `f2` | Threshold-dependent metrics at the F2-tuned threshold |
| `auc`, `auc_pr` | AUROC and AUPRC |

The `*_split2_metrics_v2.csv` files in `baseline/`, `undersample_rus/`, `smote_plain/` and
`sw_smote/` hold the same evaluations together with runs of additional seeds; the analysis
scripts restrict them to the 12 seeds above, which yields the same 1,512 rows.

### 4.2 Derived tables

| File | Contents |
|---|---|
| `paper_table_NEW_may2026.csv` | Mean and standard deviation per (R, M) over D, G and S (n = 72): the performance table of the paper |
| `sensitivity/sobol_indices.csv` | First-order (`S1`) and total (`ST`) Sobol indices per factor and metric, with 95 % bootstrap intervals (`*_lo`, `*_hi`; B = 2,000 over seeds) |
| `sensitivity/variance_decomposition.csv` | The underlying variance components |

---

## 5. `distance/` — subject grouping

`rankings/split2/knn/{mmd,dtw,wasserstein}_{in,out}_domain.txt` list the subject-sessions
of the In (44) and Out (43) groups for each distance metric: each subject-session is scored
by its mean distance to its k = 5 nearest neighbours and the 87 are split at the median.
`matrices_normalized/` holds the pairwise distance matrices used for the ranking.

The grouping is at subject-session level, so a driver with two sessions can have one
session in each group.

---

## 6. Reproduction

1. Obtain the code release at the commit recorded in `BUNDLE_INFO.json` and install it as
   described in its `README.md`.
2. Extract the bundle over the repository root.
3. Run the analysis scripts listed in the README (section "Scripts for reproducing the
   paper") to regenerate the tables and figures and compare them with the shipped copies.

Verify integrity first:

```bash
sha256sum -c MANIFEST.sha256
```

---

## 7. Notes

1. **Cite the upstream deposit.** doi:10.7910/DVN/HMZ5RG is released under CC0 1.0, so it
   places no licensing restriction on this release; per Harvard Dataverse community norms,
   cite it alongside this bundle (see `LICENSE` and `CITATION.cff`).
2. **Electrode montage unavailable** (Section 3).

---

## 8. Contact

Yutaro Nakagama — Japan Advanced Institute of Science and Technology (JAIST). See
`CITATION.cff`.
