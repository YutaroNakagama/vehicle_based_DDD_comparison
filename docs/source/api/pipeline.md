# Pipeline API Documentation

## Pipeline Overview

```mermaid
graph TD
  train_pipeline --> load_data
  train_pipeline --> split_data
  train_pipeline --> feature_selection
  train_pipeline --> train_model
  train_model --> save_model
  train_model --> save_metrics

  evaluate_pipeline --> load_model
  evaluate_pipeline --> load_test_data
  evaluate_pipeline --> evaluate_model
  evaluate_pipeline --> optimise_threshold
  evaluate_pipeline --> save_results

  distance_pipeline --> load_features
  distance_pipeline --> compute_mmd
  distance_pipeline --> compute_wasserstein
  distance_pipeline --> compute_dtw
  compute_mmd --> save_mmd
  compute_wasserstein --> save_was
  compute_dtw --> save_dtw
````

---

### 1. `train_pipeline`

| Function            | Input                                             | Output                                        | Notes                           |
| ------------------- | ------------------------------------------------- | --------------------------------------------- | ------------------------------- |
| `train_pipeline`    | dataset path(s), config params                    | trained model(s), metrics files               | Orchestrates training workflow  |
| `load_data`         | `data/processed/*.csv`                            | `pandas.DataFrame`                            | Reads preprocessed subject data |
| `split_data`        | DataFrame, split strategy (subject-wise / k-fold) | Train/val/test DataFrames                     | Uses `sklearn.model_selection`  |
| `feature_selection` | Training DataFrame, feature config                | Reduced feature DataFrame                     | e.g. ANOVA, MI, RF importance   |
| `train_model`       | Reduced training set, model params                | fitted model object                           | RF / SVM-A / SVM-W / LSTM       |
| `save_model`        | fitted model object, features                     | `models/<model>.pkl`, `selected_features.pkl` | Stored with joblib              |
| `save_metrics`      | training logs, metrics (loss, F1, AUC)            | `results/trainmetrics_*.{csv,json}`           | Used later in evaluation        |

---

### 2. `evaluate_pipeline`

| Function             | Input                               | Output                                                  | Notes                                 |
| -------------------- | ----------------------------------- | ------------------------------------------------------- | ------------------------------------- |
| `evaluate_pipeline`  | trained model path, dataset path(s) | evaluation metrics, threshold files                     | Orchestrates evaluation workflow      |
| `load_model`         | `models/<model>.pkl`                | fitted model object                                     | joblib load                           |
| `load_test_data`     | `data/processed/*.csv`              | Test DataFrame                                          | Same preprocessing as training        |
| `evaluate_model`     | model object, test DataFrame        | metrics (accuracy, F1, ROC AUC, AP)                     | Outputs raw scores & confusion matrix |
| `optimise_threshold` | predicted probabilities, labels     | threshold value                                         | Search for max F1 (Optuna-based)      |
| `save_results`       | metrics, threshold                  | `results/evalmetrics_*.csv`, `results/threshold_*.json` | Consistent naming per model           |

---

### 3. `distance_pipeline`

| Function              | Input                                    | Output                                       | Notes                                 |
| --------------------- | ---------------------------------------- | -------------------------------------------- | ------------------------------------- |
| `distance_pipeline`   | subject list, data root                  | distance matrices                            | Orchestrates domain distance analysis |
| `load_features`       | Cached features (`.pkl`, `.npy`)         | Feature arrays                               | Reuses cache if available             |
| `compute_mmd`         | Feature arrays (source vs target groups) | MMD distance matrix (`.npy`)                 | Kernel-based                          |
| `compute_wasserstein` | Feature arrays                           | Wasserstein distance matrix (`.npy`)         | Uses POT/OT library                   |
| `compute_dtw`         | Time-series signals                      | DTW distance matrix (`.npy`)                 | Uses fastdtw/scipy                    |
| `save_mmd`            | MMD results                              | `results/mmd/mmd_matrix.npy`                 | Numpy array                           |
| `save_was`            | Wasserstein results                      | `results/wasserstein/wasserstein_matrix.npy` | Numpy array                           |
| `save_dtw`            | DTW results                              | `results/dtw/dtw_matrix.npy`                 | Numpy array                           |

---

```

