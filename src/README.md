# src/ directory

This directory contains the **core source code** for the Vehicle-Based DDD Comparison project.  
It is organized into modular components to handle data preprocessing, feature extraction, model training, evaluation, and analysis.

---

## Structure

```
src/
├── analysis/              # Domain distance & correlation analysis
│   ├── imbalance/         # Imbalance analysis utilities
│   │   ├── performance_results.py
│   │   └── sample_distribution.py
│   ├── clustering_projection.py        # Clustering and projection analysis
│   ├── clustering_projection_ranked.py # Ranked clustering projection
│   ├── confusion_matrix.py             # Confusion matrix analysis
│   ├── distance_computation.py         # Distance computation (MMD, Wasserstein, DTW)
│   ├── distance_correlation.py         # Correlation between distances and performance
│   ├── group_comparison.py             # Group-wise comparison utilities
│   ├── group_distance_report.py        # Distance report generation
│   ├── imbalance_analysis.py           # Imbalance metrics analysis
│   ├── metrics_tables.py               # Table generation for metrics
│   ├── sampling.py                     # Sampling utilities
│   ├── subject_group_generator.py      # Subject group generation
│   └── subject_ranking.py              # Subject ranking by domain distance
│
├── data_pipeline/         # Data preprocessing & feature extraction
│   ├── features/          # Feature definitions
│   │   ├── eeg.py         # EEG feature extraction
│   │   ├── kss.py         # Karolinska Sleepiness Scale label handling
│   │   ├── physio.py      # Physiological features (GSR, HR, etc.)
│   │   ├── simlsl.py      # Vehicle-based simulator features
│   │   └── wavelet.py     # Wavelet decomposition features
│   ├── augmentation.py            # Data augmentation utilities
│   ├── processing_pipeline.py     # Single-process preprocessing pipeline
│   └── processing_pipeline_mp.py  # Multi-process preprocessing pipeline
│
├── evaluation/            # Evaluation pipelines
│   ├── eval_pipeline.py   # Unified evaluation entry point
│   ├── eval_stages.py     # Evaluation stage utilities
│   └── models/            # Evaluation model wrappers
│       ├── common.py
│       ├── lstm.py
│       └── SvmA.py
│
├── models/                # Model definitions & training pipelines
│   ├── architectures/     # Classical & neural model architectures
│   │   ├── common.py              # RF, BalancedRF, EasyEnsemble, etc.
│   │   ├── common_backup.py       # Backup of common architectures
│   │   ├── common_evaluation.py   # Evaluation utilities for common models
│   │   ├── common_models.py       # Model definitions
│   │   ├── common_optuna.py       # Optuna hyperparameter optimization
│   │   ├── common_oversampling.py # Oversampling utilities
│   │   ├── helpers.py             # Training helper functions
│   │   ├── lstm.py                # LSTM architecture
│   │   ├── SvmA.py                # SVM architecture
│   │   └── train_helpers.py       # Training utilities
│   ├── feature_selection/ # Feature selection methods
│   │   ├── anfis.py       # ANFIS-based selection
│   │   ├── feature_helpers.py
│   │   ├── index.py       # Feature selection index
│   │   └── rf_importance.py # Random Forest importance
│   ├── model_pipeline.py  # End-to-end training pipeline
│   └── train_stages.py    # Training stage utilities
│
├── utils/                 # Utility functions
│   ├── analysis/          # Analysis utilities
│   │   ├── distance_utils.py     # Distance calculation utilities
│   │   ├── projection_utils.py   # Projection utilities
│   │   └── statistical_utils.py  # Statistical utilities
│   ├── cli/               # CLI helper utilities
│   │   └── train_cli_helpers.py
│   ├── domain_generalization/  # Domain generalization (CORAL, Mixup, VAE, Jitter)
│   │   ├── coral.py
│   │   ├── domain_mixup.py
│   │   ├── jitter.py
│   │   └── vae_augment.py
│   ├── evaluation/        # Evaluation metrics and threshold optimization
│   │   ├── metrics.py
│   │   └── threshold.py
│   ├── io/                # Data loading/saving utilities
│   │   ├── data_io.py
│   │   ├── feature_utils.py
│   │   ├── loaders.py
│   │   ├── merge.py
│   │   ├── model_artifacts.py
│   │   ├── preprocessing.py
│   │   ├── savers.py
│   │   ├── split.py
│   │   ├── split_helpers.py
│   │   └── target_resolution.py
│   ├── visualization/     # Visualization tools (ROC, radar charts)
│   │   ├── color_palettes.py
│   │   ├── plot_roc_cli.py
│   │   ├── radar.py
│   │   ├── setup.py
│   │   └── visualization.py
│   └── artifact_loader.py # Artifact loading utilities
│
└── config.py              # Centralized configuration settings
```

---

## Notes
- **analysis/**: post-training evaluations, domain distance calculations, and imbalance analysis
- **data_pipeline/**: converts raw data (EEG, vehicle, physio) into processed features  
- **evaluation/**: evaluation framework using trained models  
- **models/**: classical and neural architectures + training pipelines  
- **utils/**: shared helpers for CLI, domain generalization, evaluation, I/O, visualization, and analysis

> **Note**: Data preparation scripts (subject grouping, feature checks) are located in `scripts/python/setup/`.  

---

## Future Work
- Add comprehensive docstrings (Google/NumPy style)  
- Consider packaging `src/` as a Python module (`setup.py` or `pyproject.toml`)  
- Expand unit tests under a dedicated `tests/` directory

