# results/

This directory stores all experiment results.  
The structure follows a consistent policy to ensure reproducibility, clarity, and reusability.

## Directory Structure

```
results/
├── domain/                    # Domain analysis results
│   ├── distance/              # Distance matrices and visualizations
│   │   ├── subject-wise/      # Subject-wise distances
│   │   │   ├── mmd/           # MMD distance
│   │   │   ├── wasserstein/   # Wasserstein distance
│   │   │   ├── dtw/           # DTW distance
│   │   │   └── ranks/         # Distance-based rankings
│   │   └── group-wise/        # Group-wise distances
│   │       ├── mmd/
│   │       ├── wasserstein/
│   │       ├── dtw/
│   │       ├── intergroup_analysis/
│   │       └── random10/
│   ├── rankings/              # Ranking results
│   │   ├── centroid_umap/
│   │   ├── lof/
│   │   ├── mean_distance/
│   │   └── ranking_comparison/
│   └── summary/               # Summary tables and visualizations
│       ├── csv/
│       └── png/
├── evaluation/                # Model evaluation results (Job ID based)
│   ├── RF/                    # Random Forest
│   ├── BalancedRF/            # Balanced Random Forest
│   └── EasyEnsemble/          # EasyEnsemble
├── hyperparam/                # Hyperparameter analysis results
│   ├── *.csv                  # Raw hyperparameter data
│   └── *.png                  # Hyperparameter visualizations
├── imbalance/                 # Imbalance experiments (category-based)
│   ├── training/              # Training results by imbalance method
│   ├── evaluation/            # Evaluation results by imbalance method
│   └── analysis/              # Analysis and visualizations
│       ├── plots/
│       ├── sampling/
│       └── multiseed/
└── train/                     # General training results (currently empty)
```

## Naming Conventions

### Per-job Results (evaluation/)

Evaluation results are saved per Job ID directory:
- `<jobID>/<jobID>[<idx>]/` - Organized by array job index

### Summary Files

- **Single-job summary:** `summary_<target>_<jobID>.csv`
- **Multi-job comparison:** `compare_<target>_<analysisType>_<date>.csv`
- **Global summary:** `summary_all_<target>_<date>.csv`

## Policy

- **CSV** for numerical data, **PNG** for visualizations
- Job results are never overwritten: each run is placed in its own job ID folder
- No PDF/SVG — PNG is the single standard format

