# Domain Analysis HPC Job Scripts

PBS job scripts for domain-based driver drowsiness detection experiments.

## Scripts Overview

| Script | Description |
|--------|-------------|
| `pbs_train.sh` | Unified training script for domain analysis |
| `eval_domain_ranking.sh` | Evaluate trained models on test data |

## Usage

### Training - Ranking Comparison

Compare which ranking method (knn, lof, etc.) produces the best subject selection:

```bash
qsub -N rank_knn -l select=1:ncpus=4:mem=16gb -l walltime=12:00:00 -q SINGLE \
     -v EXPERIMENT=ranking_comparison,RANKING_METHOD=knn,DISTANCE_METRIC=mmd,DOMAIN_LEVEL=out_domain,MODE=source_only \
     pbs_train.sh
```

**Environment Variables:**
- `EXPERIMENT`: `ranking_comparison`
- `RANKING_METHOD`: knn, lof, mean_distance, median_distance, isolation_forest, centroid_umap
- `DISTANCE_METRIC`: mmd, dtw, wasserstein
- `DOMAIN_LEVEL`: out_domain, in_domain, cross_domain
- `MODE`: source_only, target_only
- `SEED`: Random seed (default: 42)

### Training - SMOTE Comparison

Compare SMOTE methods using ranking-based subject selection:

```bash
qsub -N smote_knn -l select=1:ncpus=4:mem=8gb -l walltime=10:00:00 -q SINGLE \
     -v EXPERIMENT=smote_comparison,METHOD=smote_subjectwise,SUBJECT_FILE=/path/to/subjects.txt,MODE=source_only \
     pbs_train.sh
```

**Environment Variables:**
- `EXPERIMENT`: `smote_comparison`
- `METHOD`: smote, smote_subjectwise, smote_balanced_rf
- `SUBJECT_FILE`: Path to subject list file
- `RATIO`: Target ratio for oversampling (default: 0.33)
- `TAG`: Experiment tag (auto-generated if not set)
- `MODE`: source_only, target_only
- `SEED`: Random seed (default: 42)

### Evaluation

```bash
qsub -v TRAIN_JOBID=12345678,TAG=rank_knn_mmd_out eval_domain_ranking.sh
```

## Resource Requirements

| Experiment | CPUs | Memory | Walltime | Queue |
|------------|------|--------|----------|-------|
| ranking_comparison | 4 | 16GB | 12h | SINGLE |
| smote | 4 | 8GB | 10h | SINGLE |
| smote_subjectwise | 4 | 6GB | 12h | SINGLE |
| smote_balanced_rf | 8 | 8GB | 12h | DEFAULT |
| eval | 4 | 16GB | 2h | SINGLE |

## Launcher

Use the launcher for batch submissions:

```bash
# Ranking comparison experiments
./scripts/hpc/launchers/launch_domain.sh ranking --dry-run
./scripts/hpc/launchers/launch_domain.sh ranking

# SMOTE comparison experiments
./scripts/hpc/launchers/launch_domain.sh smote --methods "smote smote_subjectwise"
```

## Output Locations

- **Models:** `models/<ModelName>/<JobID>/`
- **Results:** `results/outputs/training/<ModelName>/<JobID>/`
- **Logs:** `scripts/hpc/logs/train/`
