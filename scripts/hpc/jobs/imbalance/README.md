# Imbalance Comparison Job Scripts

PBS job scripts for class imbalance experiments on KAGAYAKI HPC.

## Scripts

| Script | Purpose |
|--------|---------|
| `pbs_train.sh` | Unified training for all imbalance methods (pooled mode) |
| `pbs_train_ranking.sh` | Training with ranking-based subject selection (source_only/target_only) |
| `pbs_eval.sh` | Model evaluation with optional threshold tuning |
| `pbs_visualize_pooled_comparison.sh` | Generate comparison figures for pooled experiments |

## Usage Examples

### Training (Pooled Mode)

```bash
# Baseline (no oversampling)
qsub -N baseline -l select=1:ncpus=4:mem=8gb -l walltime=08:00:00 -q SINGLE \
     -v METHOD=baseline,SEED=42 pbs_train.sh

# SMOTE
qsub -N smote -l select=1:ncpus=4:mem=8gb -l walltime=10:00:00 -q SINGLE \
     -v METHOD=smote,SEED=42,RATIO=0.33 pbs_train.sh

# Subject-wise SMOTE
qsub -N sw_smote -l select=1:ncpus=4:mem=6gb -l walltime=12:00:00 -q SINGLE \
     -v METHOD=smote_subjectwise,SEED=42 pbs_train.sh

# BalancedRandomForest
qsub -N balanced_rf -l select=1:ncpus=8:mem=8gb -l walltime=12:00:00 -q DEFAULT \
     -v METHOD=balanced_rf,SEED=42 pbs_train.sh

# SMOTE + BalancedRF
qsub -N smote_brf -l select=1:ncpus=8:mem=8gb -l walltime=12:00:00 -q DEFAULT \
     -v METHOD=smote_balanced_rf,SEED=42,RATIO=0.33 pbs_train.sh

# SMOTE + Tomek Links
qsub -N smote_tomek -l select=1:ncpus=4:mem=8gb -l walltime=12:00:00 -q SINGLE \
     -v METHOD=smote_tomek,SEED=42 pbs_train.sh
```

### Training (Ranking-based)

```bash
# Simple SMOTE with knn ranking
qsub -N smote_knn -l select=1:ncpus=4:mem=8gb -l walltime=10:00:00 -q SINGLE \
     -v METHOD=smote,MODE=source_only,SUBJECT_FILE=/path/to/subjects.txt,TAG=smote_knn_top8 \
     pbs_train_ranking.sh

# Subject-wise SMOTE with lof ranking
qsub -N sw_smote_lof -l select=1:ncpus=4:mem=6gb -l walltime=12:00:00 -q SINGLE \
     -v METHOD=smote_subjectwise,MODE=source_only,SUBJECT_FILE=/path/to/subjects.txt,TAG=sw_smote_lof_top8 \
     pbs_train_ranking.sh
```

### Evaluation

```bash
# Standard evaluation
qsub -N eval -l select=1:ncpus=2:mem=4gb -l walltime=02:00:00 -q SINGLE \
     -v MODEL=RF,TAG=imbal_v2_smote_seed42,TRAIN_JOBID=12345678 pbs_eval.sh

# With custom threshold
qsub -N eval_thr -l select=1:ncpus=2:mem=4gb -l walltime=02:00:00 -q SINGLE \
     -v MODEL=RF,TAG=imbal_v2_smote_seed42,TRAIN_JOBID=12345678,THRESHOLD=0.35 pbs_eval.sh
```

## Supported Methods

| METHOD | Description | Model |
|--------|-------------|-------|
| `baseline` | No oversampling | RF |
| `smote` | Standard SMOTE | RF |
| `smote_subjectwise` | Subject-wise SMOTE | RF |
| `smote_tomek` | SMOTE + Tomek Links | RF |
| `smote_enn` | SMOTE + ENN | RF |
| `smote_rus` | SMOTE + Random Undersampling | RF |
| `balanced_rf` | BalancedRandomForest | BalancedRF |
| `easy_ensemble` | EasyEnsemble | EasyEnsemble |
| `smote_balanced_rf` | SMOTE + BalancedRF | BalancedRF |
| `jitter_scale` | Jittering + Scaling | RF |
| `undersample_enn` | ENN only | RF |
| `undersample_rus` | Random Undersampling | RF |
| `undersample_tomek` | Tomek Links only | RF |

## Environment Variables

### pbs_train.sh
- `METHOD`: Sampling method (required)
- `SEED`: Random seed (default: 42)
- `RATIO`: Target ratio (default: 0.33)
- `TAG`: Experiment tag (auto-generated if not set)
- `N_TRIALS`: Optuna trials (default: 50)

### pbs_train_ranking.sh
- `METHOD`: smote, smote_subjectwise, smote_balanced_rf
- `MODE`: source_only, target_only
- `SUBJECT_FILE`: Path to subject list file
- `TAG`: Experiment tag (required)

### pbs_eval.sh
- `MODEL`: RF, BalancedRF, EasyEnsemble
- `TAG`: Experiment tag (required)
- `TRAIN_JOBID`: Training job ID (required)
- `THRESHOLD`: Custom prediction threshold (optional)
