# Configuration Reference

This document describes all configurable parameters in the `src/config.py` file.

## Path Configuration

### Base Paths

| Variable | Default | Description |
|----------|---------|-------------|
| `DATASET_PATH` | `data/raw/` | Path to raw dataset files |
| `INTERIM_PATH` | `data/interim/` | Intermediate processed data |
| `PROCESSED_PATH` | `data/processed/` | Final feature matrices |
| `MODEL_PKL_PATH` | `models/` | Trained model artifacts |
| `RESULT_PATH` | `results/` | Evaluation outputs |

### File Naming

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_FILE_EXTENSION` | `.pkl` | Model artifact extension |
| `METRICS_FILE_EXTENSION_JSON` | `.json` | JSON metrics extension |
| `METRICS_FILE_EXTENSION_CSV` | `.csv` | CSV metrics extension |

## Signal Processing

### Sampling Rates

| Variable | Value | Description |
|----------|-------|-------------|
| `SAMPLE_RATE_SIMLSL` | 100 Hz | Vehicle dynamics signals |
| `SAMPLE_RATE_EEG` | 256 Hz | EEG signals |

### Window Configuration

```python
MODEL_WINDOW_CONFIG = {
    "common": {"window_sec": 60, "stride_sec": 30},
    "SvmA": {"window_sec": 30, "stride_sec": 15},
    "SvmW": {"window_sec": 60, "stride_sec": 30},
    "Lstm": {"window_sec": 60, "stride_sec": 30},
}
```

## Training Parameters

### Optuna Hyperparameter Tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `N_TRIALS` | 100 | Number of Optuna optimization trials |
| `OPTUNA_N_STARTUP_TRIALS` | 10 | Random sampling trials before TPE |
| `OPTUNA_N_WARMUP_STEPS` | 5 | Warmup steps for pruner |
| `OPTUNA_INTERVAL_STEPS` | 1 | Pruning check interval |

### Cross-Validation

| Variable | Default | Description |
|----------|---------|-------------|
| `CV_FOLDS_OPTUNA` | 3 | CV folds during optimization |
| `CV_FOLDS_OPTUNA_DATA_LEAK` | 2 | CV folds for data leak experiments |
| `CV_FOLDS_CALIBRATION` | 5 | CV folds for probability calibration |

### Model Hyperparameters

| Variable | Default | Description |
|----------|---------|-------------|
| `RF_CLASS_WEIGHT` | `{0: 1.0, 1: 3.0}` | Random Forest class weights |
| `LIGHTGBM_N_ESTIMATORS_RANGE` | `(100, 300)` | LightGBM estimator range |
| `LIGHTGBM_LEARNING_RATE_RANGE` | `(1e-3, 0.3)` | LightGBM learning rate range |

## Evaluation Thresholds

| Variable | Default | Description |
|----------|---------|-------------|
| `MIN_RECALL_THRESHOLD` | 0.70 | Minimum recall constraint for precision optimization |
| `FBETA_SCORE_BETA` | 2.0 | Beta parameter for F-beta score (emphasizes recall) |
| `BASELINE_AUPRC` | 0.039 | Baseline AU-PRC for imbalanced comparison |

## Sample Count Thresholds

| Variable | Default | Description |
|----------|---------|-------------|
| `SAMPLE_COUNT_MIN_TRAIN` | 100 | Minimum training samples |
| `SAMPLE_COUNT_MIN_VAL` | 30 | Minimum validation samples |
| `SAMPLE_COUNT_MIN_TEST` | 30 | Minimum test samples |

## Labeling Scheme

### KSS to Binary Conversion

```python
TARGET_SCHEME = {
    "alert": [1, 2, 3, 4, 5, 6],  # Label 0: Alert
    "drowsy": [7, 8, 9],          # Label 1: Drowsy
}
TARGET_SCHEME_NAME = "only10"
```

### Alternative Schemes

```python
# Balanced scheme
KSS_TO_BINARY_BALANCED = {
    "alert": [1, 2, 3, 4],
    "drowsy": [5, 6, 7, 8, 9],
}

# Conservative scheme (high confidence drowsy)
KSS_TO_BINARY_CONSERVATIVE = {
    "alert": [1, 2, 3, 4, 5],
    "drowsy": [8, 9],
}
```

## Domain Generalization

### Subject Groups

```python
SUBJECT_GROUPS = {
    "all": "config/subjects/subject_list.txt",
    "young": "config/subjects/young_subjects.txt",
    "elderly": "config/subjects/elderly_subjects.txt",
    "male": "config/subjects/male_subjects.txt",
    "female": "config/subjects/female_subjects.txt",
}
```

## Example: Custom Configuration

```python
# config/custom_config.py
from src.config import *

# Override for quick experiments
N_TRIALS = 20
CV_FOLDS_OPTUNA = 2

# Higher recall emphasis for safety
FBETA_SCORE_BETA = 3.0
MIN_RECALL_THRESHOLD = 0.80

# Aggressive class weighting
RF_CLASS_WEIGHT = {0: 1.0, 1: 5.0}
```

Usage:

```bash
PYTHONPATH=config python scripts/python/train.py --model RF
```
