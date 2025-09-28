---

# `bin/` Directory Overview

This directory contains command-line entry point scripts for different stages of the **Driver Drowsiness Detection (DDD) pipeline**.
Each script orchestrates a specific stage (preprocessing, training, evaluation, analysis, and visualization), while delegating detailed implementations to modules under `src/`.

---

## `preprocess.py`

**Purpose**
Preprocess data for training or evaluation in the DDD pipeline.
It selects and executes the model-specific preprocessing routine (single-process or multi-process) and applies optional data augmentation.

**Key Features**

* Chooses preprocessing pipeline from:

  * `src.data_pipeline.processing_pipeline`
  * `src.data_pipeline.processing_pipeline_mp` (multi-process variant)
* Supports augmentation with jittering.
* Designed to be run as a CLI tool.

**Example Usage**

```bash
python preprocess.py --model Lstm --jittering
python preprocess.py --model RF
```

---

## `train.py`

**Purpose**
Train a machine learning model for driver drowsiness detection.
Provides flexible training with support for augmentation, domain generalization, feature selection, cross-validation, and subject-specific splitting strategies.

**Key Features**

* Supports multiple architectures (`src.config.MODEL_CHOICES`).
* Augmentation options: Domain Mixup, CORAL, VAE.
* Cross-validation (`--n_folds`, `--fold`).
* Subject-wise and custom split strategies.
* Feature selection: `rf`, `mi`, `anova`.
* Experiment modes: `only_target`, `only_general`, `eval_only`, `finetune`, `train_only`.
* Label balancing and time-stratified splits.
* Supports pretraining/finetuning workflows.

**Example Usage**

```bash
python train.py --model Lstm --domain_mixup --coral --vae
python train.py --model RF
python train.py --model Lstm --n_folds 5 --subject_wise_split
```

---

## `evaluate.py`

**Purpose**
Evaluate a trained model for driver drowsiness detection.
Loads the trained model and test data, computes evaluation metrics (accuracy, precision, recall, confusion matrix), and prints results.

**Key Features**

* Works with all supported architectures.
* Fold-based evaluation for cross-validation.
* Optional subject-wise split to prevent leakage.
* Compatible with `train.py` experiment modes.
* Delegates execution to `src.evaluation.eval_pipeline`.

**Example Usage**

```bash
python evaluate.py --model Lstm
python evaluate.py --model RF --subject_wise_split --tag coral
python evaluate.py --model Lstm --fold 3
```

---

## `analyze.py`

**Purpose**
Unified CLI for post-hoc analysis of results and domain distances.
Provides subcommands to compute distances, correlate metrics, summarize experiments, and export rankings.

**Key Subcommands**

* `comp-dist`: Compute distance matrices (MMD, Wasserstein, DTW).
* `corr`: Correlate group distances with performance deltas.
* `summarize`: Compare *only10* vs *finetune* results (with optional radar plots).
* `summarize-metrics`: Aggregate `metrics_*.csv` into long-form summaries.
* `make-table`: Create wide-format comparison tables.
* `report-pretrain-groups`: Compute intra/inter/NN stats for pretrain groups.
* `corr-collect`: Collect correlation CSVs and generate a heatmap.
* `rank-export`: Export top/bottom-k subject lists from rankings.

**Example Usage**

```bash
python analyze.py comp-dist --subject_list misc/subject_list.txt --data_root data/processed/common
python analyze.py corr --summary_csv results/summary.csv --distance results/mmd/mmd_matrix.npy --groups_dir misc/pretrain_groups
python analyze.py summarize --make_radar
```

---

## `plot.py`

**Purpose**
Plot ROC curves from the latest evaluation results across models.
Automatically loads `metrics_*.json` files in a results directory and generates comparative plots.

**Key Features**

* Reads evaluation metrics from JSON.
* Produces ROC curve comparisons across models.
* Uses `src.utils.visualization.visualization.plot_roc_curves_from_latest_json`.

**Example Usage**

```bash
python plot.py --results_dir results/common --title "ROC Comparison"
```

---

## `make_pretrain_groups.py`

**Purpose**
Generate a **wide-format comparison table** for *only10* vs *finetune* experiments.
Ensures that summary CSVs exist (creating them if missing), then produces the final wide comparison table.

**Key Features**

* Builds summaries via `src.analysis.metrics_tables.summarize_metrics`.
* Produces wide-format comparison tables with `make_comparison_table`.
* Output saved to `table_only10_vs_finetune_wide.csv` by default.

**Example Usage**

```bash
python make_pretrain_groups.py --model_dir model/common --model_tag RF
```

---

## `__init__.py`

Empty file to mark the directory as a Python package.

---

