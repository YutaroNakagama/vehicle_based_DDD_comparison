# HPC Job Launchers

Scripts to submit PBS jobs to KAGAYAKI HPC.

## Scripts

| Script | Purpose |
|--------|---------|
| `launch_imbalance.sh` | Submit imbalance comparison training jobs |
| `launch_ranking.sh` | Submit ranking-based training jobs |
| `launch_preprocess_all.sh` | Submit preprocessing jobs for all models |
| `launch_train_all.sh` | Submit basic training jobs |
| `run_smote_analysis.sh` | Run SMOTE comparison analysis pipeline |

## Usage Examples

### Imbalance Training

```bash
# All methods with seed 42 (default)
./launch_imbalance.sh

# Multiple seeds
./launch_imbalance.sh --seeds "42 123 456"

# Specific methods
./launch_imbalance.sh --methods "smote smote_subjectwise balanced_rf"

# Preview without submitting
./launch_imbalance.sh --dry-run
```

### Ranking-based Training

```bash
# All combinations
./launch_ranking.sh

# Specific ranking method
./launch_ranking.sh --rankings "knn"

# Specific training methods
./launch_ranking.sh --methods "smote smote_subjectwise"

# Preview without submitting
./launch_ranking.sh --dry-run
```

### Analysis

```bash
# Run full SMOTE analysis pipeline
./run_smote_analysis.sh

# Filter to specific job IDs
./run_smote_analysis.sh 14653722 14653746
```

## Supported Methods

### Imbalance Methods
- `baseline` - No oversampling
- `smote` - Standard SMOTE
- `smote_subjectwise` - Subject-wise SMOTE
- `smote_tomek` - SMOTE + Tomek Links
- `smote_enn` - SMOTE + ENN
- `smote_rus` - SMOTE + Random Undersampling
- `balanced_rf` - BalancedRandomForest
- `easy_ensemble` - EasyEnsemble
- `smote_balanced_rf` - SMOTE + BalancedRF

### Ranking Methods
- `smote` - Simple SMOTE with ranking
- `smote_subjectwise` - Subject-wise SMOTE with ranking
- `smote_balanced_rf` - SMOTE + BalancedRF with ranking

## Resource Allocation

Resources are automatically selected based on method:

| Method | CPUs | Memory | Walltime | Queue |
|--------|------|--------|----------|-------|
| balanced_rf, easy_ensemble, smote_balanced_rf, smote_enn | 8 | 8GB | 12h | DEFAULT |
| smote_subjectwise | 4 | 6GB | 12h | SINGLE |
| baseline, smote_rus | 4 | 8GB | 8h | SINGLE |
| Others | 4 | 8GB | 10h | SINGLE |
