# results/

This directory stores all experiment results.  
The structure separates **analysis outputs** (derived/computed) from **job run outputs** (raw job data).

## Directory Structure

```
results/
├── analysis/                  # Analysis results (derived from job outputs)
│   ├── domain/                # Domain analysis
│   │   ├── distance/          # Distance matrices
│   │   │   ├── subject-wise/  # Subject-wise distances (MMD, Wasserstein, DTW)
│   │   │   │   └── ranks/     # Distance-based ranking groups
│   │   │   └── group-wise/    # Group-wise distances
│   │   ├── rankings/          # Ranking method results
│   │   │   ├── centroid_umap/
│   │   │   ├── lof/
│   │   │   ├── mean_distance/
│   │   │   └── ranking_comparison/
│   │   ├── hyperparam/        # Hyperparameter convergence analysis
│   │   └── summary/           # Summary tables and visualizations
│   │       ├── csv/
│   │       └── png/
│   └── imbalance/             # Imbalance experiment analysis
│       ├── hyperparam/        # Hyperparameter convergence analysis
│       ├── plots/             # Metric visualizations
│       ├── sampling/          # Sampling distribution analysis
│       └── multiseed/         # Multi-seed experiment analysis
│
└── outputs/                   # Job outputs (raw data from HPC/local)
    ├── evaluation/            # General evaluation outputs (by model type)
    │   ├── RF/                # Random Forest evaluations
    │   │   └── {job_id}/      # Per-job results
    │   ├── BalancedRF/        # Balanced Random Forest
    │   └── EasyEnsemble/      # EasyEnsemble
    └── imbalance/             # Imbalance experiment outputs
        ├── training/          # Training outputs by method
        │   ├── baseline/
        │   ├── smote_0.5/
        │   └── ...
        └── evaluation/        # Evaluation outputs by method
            ├── baseline/
            ├── smote_0.5/
            └── ...
```

## Design Philosophy

### analysis/ (Derived Results)
- **Purpose**: Store computed/analyzed results derived from raw job outputs
- **Examples**: Distance matrices, ranking comparisons, summary plots, hyperparameter convergence
- **Structure**: Organized by research category (domain, imbalance)
- **Lifecycle**: Can be regenerated from `outputs/` data

### outputs/ (Raw Job Outputs)
- **Purpose**: Store direct outputs from training and evaluation jobs
- **Examples**: Model evaluation JSON files, training logs, job metadata
- **Structure**: Organized by job ID for traceability
- **Lifecycle**: Primary data, should not be overwritten

## Naming Conventions

### Job Outputs (outputs/)
- `{job_id}/{job_id}[{array_idx}]/` - Array job outputs
- `eval_results_{model}_{tag}.json` - Evaluation metrics

### Analysis Outputs (analysis/)
- `summary_{category}_{date}.csv` - Summary tables
- `{metric}_{comparison_type}.png` - Visualization plots

## Config Paths

The following paths are defined in `src/config.py`:

| Variable | Path |
|----------|------|
| `RESULTS_ANALYSIS_PATH` | `./results/analysis` |
| `RESULTS_ANALYSIS_DOMAIN_PATH` | `./results/analysis/domain` |
| `RESULTS_ANALYSIS_IMBALANCE_PATH` | `./results/analysis/imbalance` |
| `RESULTS_OUTPUTS_PATH` | `./results/outputs` |
| `RESULTS_OUTPUTS_EVALUATION_PATH` | `./results/outputs/evaluation` |
| `RESULTS_OUTPUTS_IMBALANCE_PATH` | `./results/outputs/imbalance` |

## Policy

- **CSV** for numerical data, **PNG** for visualizations
- Job results are never overwritten: each run is placed in its own job ID folder
- Analysis results can be regenerated and may be overwritten

