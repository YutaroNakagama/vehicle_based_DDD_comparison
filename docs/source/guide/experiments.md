# Experiments and Reproducibility

This guide explains how to reproduce the experiments from our research.

## Experiment Overview

The repository supports several types of experiments:

1. **Model Comparison** - Compare ML classifiers on drowsiness detection
2. **Imbalance Analysis** - Evaluate oversampling strategies
3. **Domain Generalization** - Test cross-subject transfer
4. **Feature Analysis** - Analyze feature importance

## Reproducing Main Experiments

### Experiment 1: Model Comparison

Compare all supported classifiers:

```bash
# Train all models
for model in RF LightGBM XGBoost CatBoost SVM LogisticRegression; do
    python scripts/python/train.py \
        --model $model \
        --model_name common \
        --mode only10 \
        --seed 42
done

# Aggregate results
python scripts/python/aggregate_results.py \
    --experiment model_comparison \
    --output results/evaluation/model_comparison.csv
```

### Experiment 2: Imbalance Handling

Compare oversampling strategies:

```bash
methods=("none" "smote" "adasyn" "borderline" "smote_tomek" "smote_enn" "rus")

for method in "${methods[@]}"; do
    python scripts/python/train.py \
        --model RF \
        --model_name common \
        --mode only10 \
        --use_oversampling \
        --oversample_method $method \
        --seed 42
done
```

### Experiment 3: Multi-Seed Evaluation

Run experiments with multiple random seeds:

```bash
for seed in 42 123 456 789 1000; do
    python scripts/python/train.py \
        --model RF \
        --model_name common \
        --mode only10 \
        --use_oversampling \
        --oversample_method smote \
        --seed $seed
done

# Compute statistics
python scripts/results/aggregate_multiseed.py \
    --seeds 42,123,456,789,1000 \
    --output results/imbalance_analysis/multiseed/
```

### Experiment 4: Domain Generalization

Cross-subject transfer learning:

```bash
# Leave-one-subject-out evaluation
python scripts/python/loso_evaluation.py \
    --model RF \
    --model_name common

# Fine-tuning experiment
python scripts/python/train.py \
    --model RF \
    --model_name common \
    --mode finetune \
    --target_group young
```

## HPC Batch Execution

For large-scale experiments on PBS clusters:

### Submit Jobs

```bash
# Single model training
qsub scripts/hpc/train_single.pbs

# Array job for multiple configurations
qsub -t 1-10 scripts/hpc/train_array.pbs
```

### Job Template

```bash
#!/bin/bash
#PBS -N ddd_train
#PBS -l nodes=1:ppn=4
#PBS -l walltime=24:00:00
#PBS -q default

cd $PBS_O_WORKDIR
source activate ddd

python scripts/python/train.py \
    --model ${MODEL} \
    --seed ${SEED}
```

### Monitor Jobs

```bash
# Check job status
qstat -u $USER

# View job output
tail -f logs/train_${PBS_JOBID}.out
```

## Results Structure

```
results/
├── evaluation/
│   ├── metrics_RF_common_only10.csv
│   ├── metrics_LightGBM_common_only10.csv
│   └── ...
├── imbalance_analysis/
│   ├── multiseed/
│   │   ├── confusion_matrix_summary.csv
│   │   └── metrics_summary.csv
│   └── comparison/
│       └── imbalance_comparison.csv
└── domain_analysis/
    ├── distances/
    └── ranks/
```

## Saved Artifacts

Each experiment saves:

| File | Description |
|------|-------------|
| `{model}_{mode}.pkl` | Trained model |
| `scaler_{mode}.pkl` | Feature scaler |
| `features_{mode}.json` | Selected features |
| `metrics_{model}_{mode}.csv` | Evaluation metrics |
| `optuna_study_{model}_{mode}.pkl` | Optuna study object |

## Visualization

Generate plots for analysis:

```bash
# Confusion matrix
python scripts/results/plot_confusion_matrix.py \
    --input results/evaluation/metrics_RF_common_only10.csv

# ROC curves comparison
python scripts/results/plot_roc_comparison.py \
    --models RF,LightGBM,XGBoost

# Feature importance
python scripts/results/plot_feature_importance.py \
    --model RF \
    --top_n 20
```

## Reproducibility Checklist

✅ **Fixed random seeds** - Set in config and CLI

✅ **Versioned dependencies** - `requirements.txt` with pinned versions

✅ **Documented preprocessing** - Feature extraction parameters in config

✅ **Saved hyperparameters** - Optuna study objects preserved

✅ **Git versioning** - All code changes tracked

## Citation

If you use this code, please cite:

```bibtex
@misc{nakagama2025ddd,
  author = {Nakagama, Yutaro},
  title = {Vehicle-Based Driver Drowsiness Detection Comparison},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/YutaroNakagama/vehicle_based_DDD_comparison}
}
```
