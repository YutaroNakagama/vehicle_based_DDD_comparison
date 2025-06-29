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
pip install -r requirements.txt
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
    [--feature_selection rf|mi|anova] [--data_leak]
```

### Options:

* `--domain_mixup`: Mixes source/target samples (for domain generalization)
* `--coral`: CORAL-based alignment across subject domains
* `--vae`: VAE-based augmentation for latent data variability
* `--feature_selection`: Selects features via RandomForest (rf), MutualInfo (mi), or ANOVA (anova)
* `--data_leak`: Forces feature selection to access validation set (for ablation)
* `--n_folds` / `--fold`: Cross-validation control

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

## Notes

* All training uses `optuna` for hyperparameter tuning
* `selected_features` are saved per model variant for reproducibility
* Evaluation includes ROC AUC, precision/recall, and threshold-optimized F1-score

---

## License

MIT License

## Citation

If you use this code or dataset, please cite:

* Aygun et al. (2024) for the dataset

