# Vehicle-Based DDD Comparison

## Overview

This repository benchmarks **lightweight Driver Drowsiness Detection (DDD) models** using **vehicle-based features**. It aims to compare the effectiveness and computational efficiency of multiple classical and neural ML models in detecting driver drowsiness.

## Dataset

<!-- need to explain how to download complete dataset via API -->
We use the **open dataset** from *Multi-modal Data Acquisition Platform for Behavioral Evaluation* (Aygun et al., 2024), which includes:

* Vehicle-based features (e.g., steering angle, velocity)
    * EEG signals
    * Physiological signals (e.g., GSR, heart rate, EOG)
* Annotated drowsiness labels (KSS, vigilance state)

    **DOI:** [10.7910/DVN/HMZ5RG](https://doi.org/10.7910/DVN/HMZ5RG)
```sh
curl -L -O -J "https://dataverse.harvard.edu/api/access/dataset/:persistentId/?persistentId=doi:10.7910/DVN/HMZ5RG"
```

## Compared Approaches

    Our experiments evaluate:

    * **Classic ML models**: Random Forest, Logistic Regression, SVM, etc.
    * **Shallow neural networks**
    * **SvmA/SvmW**: Adaptive feature selection using ANFIS or wavelet decomposition
    * **LSTM**: Deep learning using temporal EEG and vehicle data

    These are compared against:

1. **Zhao et al. (2009)** - Multiwavelet packet energy spectrum [DOI](http://dx.doi.org/10.1109/CISP.2009.5301253)
2. **Arefnezhad et al. (2019)** - Adaptive neuro-fuzzy selection on steering [DOI](https://doi.org/10.3390/s19040943)
3. **Wang et al. (2022)** - Vehicle dynamics + naturalistic driving [DOI](http://dx.doi.org/10.1016/j.trc.2022.103561)

    ---

## Directory Structure

```
├── dataset
│   └── mdapbe
│       ├── subject_list.txt
│       └── physio
│           ├── S0101
│           │   ├── EEG_S0101_1.mat
│           │   ├── SIMlsl_S0101_1.mat
│           │   └── ...
│           └── ...
└── vehicle_based_DDD_comparison
└── project
├── bin
├── data
│   ├── interim
│   └── processed
├── model
└── src
```

## Installation

```bash
cd project
pip install -r misc/requirements.txt
```

Python 3.10 is recommended.

---

## 1. Data Preprocessing

```bash
python bin/preprocess.py --model [common|SvmA|SvmW|Lstm] [--jittering]
```

* `--model`: Required. Chooses the preprocessing logic per model type.
* `--jittering`: Optional. Applies noise-based data augmentation.

---

## 2. Model Training

```bash
python bin/train.py \
    --model [RF|SvmA|SvmW|Lstm|BalancedRF|LightGBM|XGBoost|CatBoost|LogisticRegression|SVM|DecisionTree|AdaBoost|GradientBoosting|K-Nearest\ Neighbors|MLP] \
    [--domain_mixup] [--coral] [--vae] \
    [--sample_size N] [--seed N] [--n_folds N | --fold N] \
    [--tag TAG] [--subject_wise_split] \
    [--feature_selection rf|mi|anova] [--data_leak] \
    [--subject_split_strategy random|leave-one-out|custom|isolate_target_subjects|finetune_target_subjects|single_subject_data_split] \
    [--target_subjects S0101_1 S0101_2 ...] \
    [--general_subjects S0101_1 S0101_2 ...]
```

### Options:

* `--domain_mixup`: Mixes source/target samples (for domain generalization)
* `--coral`: CORAL-based alignment across subject domains
* `--vae`: VAE-based augmentation for latent data variability
* `--feature_selection`: Selects features via RandomForest (rf), MutualInfo (mi), or ANOVA (anova)
* `--data_leak`: Forces feature selection to access validation set (for ablation)
* `--n_folds` / `--fold`: Cross-validation control
* `--subject_split_strategy`: Defines how subjects are split.
    * `isolate_target_subjects`: Trains and evaluates only on `--target_subjects`. 80% of the data is used for training, 10% for validation, and 10% for testing.
    * `finetune_target_subjects`: Trains on `--general_subjects` (if provided, otherwise all other subjects) plus 80% of `--target_subjects`, and evaluates on the remaining 10% for validation and 10% for testing from `--target_subjects`.
    * `single_subject_data_split`: Performs a within-subject split (80% train, 10% validation, 10% test) on a single subject specified by `--target_subjects`.
* `--target_subjects`: A list of subject IDs to be used with specific split strategies.
* `--general_subjects`: A list of subject IDs to be used as general training data, typically with `finetune_target_subjects` strategy.

### Usage Examples:

**Isolate 10 subjects for training and evaluation:**
```bash
python bin/train.py \
    --model RF \
    --subject_split_strategy isolate_target_subjects \
    --target_subjects S0101_1 S0101_2 S0201_1 S0201_2 S0301_1 S0301_2 S0401_1 S0401_2 S0501_1 S0501_2
```

**Perform within-subject split for a single subject:**
```bash
python bin/train.py \
    --model RF \
    --subject_split_strategy single_subject_data_split \
    --target_subjects S0101_1
```

**Use 70 subjects to pre-train and 10 subjects to fine-tune/evaluate:**
```bash
python bin/train.py \
    --model RF \
    --subject_split_strategy finetune_target_subjects \
    --target_subjects S0101_1 S0101_2 S0201_1 S0201_2 S0301_1 S0301_2 S0401_1 S0401_2 S0501_1 S0501_2 \
    --general_subjects S0601_1 S0601_2 ... # List of 70 other subjects
```

Trained models are saved to:

```
model/
└── [model_type]/
├── model.pkl
├── scaler.pkl
├── selected_features_train.pkl
└── feature_meta.json
```

---

## 3. Model Evaluation

```bash
python bin/evaluate.py \
    --model [RF|SvmA|SvmW|Lstm] \
    [--tag TAG] [--sample_size N] [--seed N] [--subject_wise_split]
```

---

## 4. Analysis

After running training, you can run distance computation and correlation analysis with the unified CLI `bin/analyze.py`:

**(A) Distance matrices (MMD / Wasserstein / DTW)**
```bash
python bin/analyze.py comp-dist \
  --subject_list ../../dataset/mdapbe/subject_list.txt \
  --data_root data/processed/common \
  --groups_file misc/target_groups.txt
```

**(B) Correlation: d(U,G) / disp(G) vs (finetune − only10) deltas**
```bash
python bin/analyze.py corr \
  --summary_csv model/common/summary_6groups_only10_vs_finetune_wide.csv \
  --distance results/distances/wasserstein_matrix.npy \
  --subjects_json results/distances/subjects.json \
  --groups_dir misc/pretrain_groups \
  --group_names_file misc/pretrain_groups/group_names.txt \
  --outdir model/common/dist_corr
```

---

## Notes

* All training uses `optuna` for hyperparameter tuning
* `selected_features` are saved per model variant for reproducibility
* Evaluation includes ROC AUC, precision/recall, and threshold-optimized F1-score
