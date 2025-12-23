# Quickstart Guide

Get up and running with your first drowsiness detection experiment in 5 minutes.

## Overview

The typical workflow consists of three stages:

```
1. Preprocess → 2. Train → 3. Evaluate
```

## Step 1: Preprocess Data

Extract features from raw sensor data:

```bash
python scripts/python/preprocess.py --model common
```

This extracts:
- **Vehicle dynamics features**: steering angle, acceleration, lane offset
- **Time-frequency features**: spectral entropy, wavelet coefficients
- **Statistical features**: mean, std, skewness, kurtosis

Output: `data/processed/common/{subject}_features.csv`

## Step 2: Train a Model

Train a Random Forest classifier with default settings:

```bash
python scripts/python/train.py \
    --model RF \
    --model_name common \
    --mode only10 \
    --group all
```

### Key Parameters

| Parameter | Description | Options |
|-----------|-------------|---------|
| `--model` | Classifier type | RF, LightGBM, XGBoost, SVM, LogisticRegression, etc. |
| `--model_name` | Feature set | common, SvmA, SvmW, Lstm |
| `--mode` | Training mode | only10, finetune, transfer |
| `--group` | Subject group | all, young, elderly, male, female |

### Training with Oversampling

Handle class imbalance with SMOTE:

```bash
python scripts/python/train.py \
    --model RF \
    --model_name common \
    --mode only10 \
    --use_oversampling \
    --oversample_method smote
```

Available methods: `smote`, `adasyn`, `borderline`, `smote_tomek`, `smote_enn`

## Step 3: Evaluate the Model

```bash
python scripts/python/evaluate.py \
    --model RF \
    --model_name common \
    --mode only10
```

Output metrics include:
- **Accuracy, Precision, Recall, F1-score**
- **ROC-AUC, PR-AUC**
- **F2-score** (recall-weighted for safety)
- **Confusion matrix**

Results saved to: `results/evaluation/`

## Complete Pipeline Example

```bash
# 1. Preprocess all subjects
python scripts/python/preprocess.py --model common

# 2. Train with hyperparameter optimization
python scripts/python/train.py \
    --model LightGBM \
    --model_name common \
    --mode only10 \
    --use_oversampling \
    --oversample_method smote_tomek

# 3. Evaluate on test set
python scripts/python/evaluate.py \
    --model LightGBM \
    --model_name common \
    --mode only10
```

## Experiment Configurations

### Quick Test (Development)

```python
# In src/config.py
N_TRIALS = 10           # Optuna trials (default: 100)
CV_FOLDS_OPTUNA = 2     # CV folds (default: 3)
```

### Full Experiment (Production)

```python
N_TRIALS = 100
CV_FOLDS_OPTUNA = 5
```

## Viewing Results

### Metrics CSV

```bash
cat results/evaluation/metrics_RF_common_only10.csv
```

### Model Artifacts

```bash
ls models/common/RF/
# RF_only10.pkl          # Trained model
# scaler_only10.pkl      # StandardScaler
# features_only10.json   # Selected features
```

## HPC Batch Execution

For large-scale experiments on PBS/Slurm clusters:

```bash
# Submit training job
qsub scripts/hpc/train_rf.pbs

# Monitor jobs
qstat -u $USER
```

See the Developer Guide for HPC details.

## Common Issues

### Low Recall

Drowsiness detection prioritizes recall (safety). If recall is low:

1. Adjust class weights: `RF_CLASS_WEIGHT = {0: 1.0, 1: 5.0}`
2. Use oversampling: `--oversample_method smote`
3. Lower decision threshold in evaluation

### Overfitting

If validation metrics are much lower than training:

1. Reduce model complexity
2. Increase regularization
3. Use early stopping with Optuna pruning

## Next Steps

- [Developer Guide](developer_guide.md) - Understand the codebase architecture
- [Imbalance Methods](imbalance_methods.md) - Detailed oversampling strategies
- [Evaluation Metrics](evaluation_metrics.md) - Understanding metrics for DDD
