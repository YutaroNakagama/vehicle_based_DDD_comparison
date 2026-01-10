# Domain Generalization Analysis Pipeline

## Overview

This document explains the **Domain Generalization (DG)** analysis workflow implemented in the *vehicle_based_DDD_comparison* repository.

The goal of this pipeline is to **quantify domain differences** among subject groups (e.g., pretrain, general, target domains) and relate them to model performance degradation across domains.

The DG pipeline consists of three sequential stages:

1. **Distance computation**  
   Compute subject- and group-level distance matrices (MMD, Wasserstein, DTW) from processed features.  
   → `scripts/python/analysis/distance/`

2. **Subject ranking**  
   Rank subjects by inter-domain distances (mean/std) to identify *general* and *target* domains.  
   → `scripts/python/analysis/domain/`

3. **Domain-specific training / evaluation**  
   Fine-tune or evaluate models using ranked subject groups (e.g., "only general", "only target", "finetune").  
   → `scripts/hpc/jobs/domain_analysis/`

---

## Pipeline Summary (High-Level)

```{mermaid}
graph TD
  A["pbs_compute_distances.sh"] --> B["domain_analysis/run_analysis.py comp-dist"]
  B --> C["distance matrices (.npy)"]
  C --> D["domain_analysis/run_analysis.py rank-export"]
  D --> E["ranked subject lists (.txt)"]
  E --> F["pbs_rank_only_general.sh / pbs_rank_only_target.sh"]
  F --> G["train.py (fine-tune or evaluate)"]
```

---

## 1. Distance Computation (`domain_analysis/run_analysis.py comp-dist` → `run_comp_dist()`)

### Purpose

To compute pairwise subject and group distances using **three complementary metrics**:

* **MMD (Maximum Mean Discrepancy)** — feature distribution similarity
* **Wasserstein distance** — marginal feature distance
* **DTW (Dynamic Time Warping)** — temporal sequence alignment distance

### Entry Point

```bash
$ python scripts/python/run_analysis.py comp-dist \
    --subject_list dataset/mdapbe/subject_list.txt \
    --data_root data/processed/common \
    --groups_file config/target_groups.txt
```

### Core Function

```python
from src.analysis.domain.distance import run_comp_dist
```

#### Data Flow

| Step | Function                                          | Input                                   | Output                                | Description                                         |
| ---- | ------------------------------------------------- | --------------------------------------- | ------------------------------------- | --------------------------------------------------- |
| 1    | `_load_subject_list()`                            | `subject_list.txt`                      | List[str]                             | Read all subject IDs                                |
| 2    | `_extract_features_with_cache()`                  | `data/processed/common/processed_*.csv` | Dict[subj → np.ndarray]               | Extract numeric feature arrays per subject (cached) |
| 3    | `compute_mmd()`                                   | Feature arrays                          | `mmd_matrix.npy`                      | Kernel-based domain similarity                      |
| 4    | `wasserstein_distance()`                          | Feature arrays                          | `wasserstein_matrix.npy`              | Marginal feature distance                           |
| 5    | `dtw()`                                           | Mean feature sequences                  | `dtw_matrix.npy`                      | Temporal alignment distance                         |
| 6    | `_compute_group_dist_matrix()`                    | Distance matrix + groups                | `group_matrix.npy`                    | Mean distances between groups                       |
| 7    | `_compute_group_centroids_from_distance_matrix()` | Distance matrix + groups                | `group_centroid_distance_heatmap.png` | 2D embedding via MDS                                |
| 8    | `_compute_intra_inter_stats()`                    | Distance matrix + groups                | `intra_inter_comparison.png`          | Intra/inter domain dispersion                       |

#### Output Structure

```
results/analysis/domain/
├── distance/
│   ├── subject-wise/
│   │   ├── mmd/
│   │   ├── wasserstein/
│   │   └── dtw/
│   └── group-wise/
│       ├── mmd/
│       ├── wasserstein/
│       └── dtw/
├── rankings/
│   ├── centroid_umap/
│   ├── lof/
│   ├── mean_distance/
│   └── ranking_comparison/
└── summary/
    ├── csv/
    └── png/
```

#### Notes

* Cached feature arrays under `results/.cache/features_*.npz` accelerate repeated runs.
* Normalization: global z-score across all subjects before computing distances.
* MDS-based 2D projections visualize domain separability.

---

## 2. Subject Ranking (`domain_analysis/run_analysis.py rank-export` → `run_rank_export()`)

### Purpose

To categorize subjects into **general / neutral / target domains** based on their distance statistics.

### Entry Point

```bash
$ python scripts/python/domain_analysis/run_analysis.py rank-export \
    --outdir results/ranks10 \
    --k 10
```

### Core Function

```python
from src.analysis.domain.ranking import run_rank_export
```

#### Algorithm Summary

For each distance metric (MMD, Wasserstein, DTW):

1. Load the corresponding distance matrix and subject list.
2. Compute per-subject **mean** and **std** (excluding diagonal).
3. Sort and export:

   * *Low* (smallest k)
   * *Middle* (closest to median)
   * *High* (largest k)
4. Additionally, export *uniform* samples.

#### Output Files

```
results/ranks10/
├── mmd_mean_in_domain.txt
├── mmd_mean_mid_domain.txt
├── mmd_mean_out_domain.txt
├── wasserstein_mean_out_domain.txt
├── dtw_mean_in_domain.txt
└── ...
```

Each file contains subject IDs separated by spaces (one line per list).

#### Interpretation

| Category        | Meaning                                    | Typical Usage                      |
| --------------- | ------------------------------------------ | ---------------------------------- |
| `*_mean_in_domain`    | Most *typical* (domain-similar) subjects   | Used for "only general" training   |
| `*_mean_mid_domain`   | Intermediate group                         | Optional                           |
| `*_mean_out_domain`   | Most *atypical* (domain-dissimilar) subjects | Used for "only target" fine-tuning |

---

## 3. Domain-Specific Training (`pbs_rank_only_general.sh` / `pbs_rank_only_target.sh`)

### Purpose

To evaluate generalization ability by **training/fine-tuning models** on different domain subsets determined by ranking results.

### Typical PBS Flow

#### (1) Distance Computation

```bash
qsub scripts/hpc/domain_gen/pbs_compute_distances.sh
```

#### (2) Rank Generation

```bash
python scripts/python/domain_analysis/run_analysis.py rank-export --outdir results/ranks10 --k 10
```

#### (3) Group-Specific Fine-Tuning

```bash
qsub scripts/hpc/domain_gen/pbs_rank_only_general.sh
qsub scripts/hpc/domain_gen/pbs_rank_only_target.sh
```

Each job array (`-J 1-9`) reads a line from rank list files and executes:

```bash
python bin/train.py \
    --mode finetune \
    --group <group_name> \
    --tag rank10_wasserstein_mean_out_domain
```

### Output Artifacts

```
models/
├── RF/
├── SvmA/
├── SvmW/
└── Lstm/
results/
├── evaluation/
│   ├── metrics_RF_only_general.csv
│   ├── metrics_RF_finetune_target.csv
│   └── ...
└── domain_generalization/
    ├── correlation/
    └── summary/
```

---

## 4. Correlation Analysis (`domain_analysis/run_analysis.py corr`)

### Purpose

To relate *domain distance metrics* (e.g., MMD mean) to *performance gaps* (Δ metrics).

```bash
python scripts/python/domain_analysis/run_analysis.py corr \
  --summary_csv model/common/summary_6groups_only10_vs_finetune_wide.csv \
  --distance results/mmd/mmd_matrix.npy \
  --subjects_json results/mmd/mmd_subjects.json \
  --groups_dir misc/pretrain_groups \
  --group_names_file misc/pretrain_groups/group_names.txt \
  --outdir model/common/dist_corr_mmd
```

**Outputs:**

* `correlations_dUG_vs_deltas.csv`
* `correlation_heatmap_all.png`

This enables interpretation of how domain distance relates to model degradation.

---

## 5. Overall Data Flow Summary

| Stage                | Command                                                | Input                         | Output                                           |
| -------------------- | ------------------------------------------------------ | ----------------------------- | ------------------------------------------------ |
| Distance computation | `run_analysis.py comp-dist`                                 | `data/processed/common/*.csv` | `results/domain_generalization/{mmd,distances}/` |
| Subject ranking      | `run_analysis.py rank-export`                               | Distance matrices             | `results/ranks10/*.txt`                          |
| Domain finetuning    | `pbs_rank_only_general.sh` / `pbs_rank_only_target.sh` | Ranked lists                  | `models/`, `results/outputs/evaluation/`                 |
| Correlation          | `run_analysis.py corr`                                      | `summary_*.csv` + distances   | `correlation_heatmap_all.png`                    |

---

## 6. Extensibility

| Extension                       | How to Add                                                       | Affected Modules           |
| ------------------------------- | ---------------------------------------------------------------- | -------------------------- |
| **New distance metric**         | Implement in `src/analysis/distances.py` (following MMD pattern) | `domain_analysis/run_analysis.py comp-dist`     |
| **New ranking logic**           | Extend `_write_rank_with_middle()`                               | `rank_export.py`           |
| **Alternative visualization**   | Add in `src/analysis/radar.py` or `summary_groups.py`            | Optional                   |
| **New domain group definition** | Update `config/target_groups.txt`                                | Reused across all analyses |

---

## 7. Key Directories

```
results/domain_generalization/   # Core DG artifacts
results/ranks10/                 # Ranked subject lists
models/                          # Domain-specific fine-tuned models
scripts/hpc/domain_gen/          # PBS job scripts for DG experiments
```

---

## 8. Experiment Results (2025-01-10)

### 8.1 Experiment Status

| Category | Count |
|----------|-------|
| **Running (R)** | 39 jobs |
| **Queued (Q)** | 77 jobs |
| **Total Submitted** | ~116 jobs |
| **Completed Results** | 20 files (domain experiments) |

### 8.2 Job Distribution

| Job Type | Running | Queued | Description |
|----------|---------|--------|-------------|
| `dom_smot_*` | 34 | 43 | SMOTE + BalancedRF |
| `dom_unde_*` | 2 | 15 | Undersample RUS |
| `dom_bala_*` | 1 | 12 | BalancedRF (no resampling) |
| `dom_base_*` | 2 | 4 | Baseline |

### 8.3 Preliminary Results

#### Best Performing Configurations (by test_F1)

| Experiment | val_F1 | test_F1 | Notes |
|------------|--------|---------|-------|
| `source_only_balanced_rf_knn_dtw_mid_domain_s123` | 0.077 | 0.083 | Mid-domain source |
| `target_only_balanced_rf_knn_dtw_mid_domain_s42` | 0.078 | 0.081 | Mid-domain target |
| `target_only_balanced_rf_knn_dtw_in_domain_s123` | 0.064 | 0.073 | In-domain target |
| `source_only_balanced_rf_knn_dtw_in_domain_s123` | 0.086 | 0.072 | In-domain source |
| `source_only_balanced_rf_knn_mmd_in_domain_s42` | 0.071 | 0.071 | MMD-based in-domain |
| `source_only_balanced_rf_knn_dtw_out_domain_s123` | 0.068 | 0.055 | Out-domain source |

#### Observations

1. **SMOTE + Ranking experiments failed**: All `rank_knn_mmd` and `rank_lof_mmd` with SMOTE produced test_F1 = 0.0
2. **Baseline (balanced_rf) relatively stable**: F1 scores around 0.05-0.08
3. **Validation-Test Gap**: Some experiments show extreme gaps (e.g., val_F1=0.52 → test_F1=0.0)
4. **Distance metric comparison**: DTW-based ranking appears more stable than MMD

### 8.4 Experiment Design Matrix

| Dimension | Values |
|-----------|--------|
| **Data Split** | source_only, target_only |
| **Distance Metric** | knn_dtw, knn_mmd, knn_wasserstein, lof_mmd |
| **Domain Group** | in_domain, mid_domain, out_domain |
| **Imbalance Method** | baseline, balanced_rf, undersample_rus, smote |
| **Seeds** | 42, 123 |

**Total Combinations**: 2 × 4 × 3 × 4 × 2 = 192 experiments

### 8.5 Next Steps

1. **Wait for all jobs to complete** (estimated: several hours)
2. **Aggregate results** into unified comparison table
3. **Visualize domain-distance vs performance correlation**
4. **Identify best distance metric and domain selection strategy**

---

## References

* **Distance metrics:** Gretton et al., *Kernel Two-Sample Tests and MMD*, JMLR (2012).
* **Wasserstein:** Villani, *Optimal Transport: Old and New*, Springer (2009).
* **DTW:** Berndt & Clifford, *Using Dynamic Time Warping to Find Patterns in Time Series*, AAAI (1994).

---
