# src/ directory

This directory contains the **core source code** for the Vehicle-Based DDD Comparison project.  
It is organized into modular components to handle data preprocessing, feature extraction, model training, evaluation, and analysis.

---

## Structure

```
src/
├── analysis/              # Domain distance & correlation analysis
│   ├── imbalance/         # Imbalance analysis utilities
│   ├── clustering_projection.py    # Clustering and projection analysis
│   ├── distance_computation.py     # Distance computation (MMD, Wasserstein, DTW)
│   ├── distance_correlation.py     # Correlation between distances and performance
│   ├── group_comparison.py         # Group-wise comparison utilities
│   ├── group_distance_report.py    # Distance report generation
│   ├── imbalance_analysis.py       # Imbalance metrics analysis
│   ├── metrics_tables.py           # Table generation for metrics
│   ├── subject_group_generator.py  # Subject group generation
│   └── subject_ranking.py          # Subject ranking by domain distance
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
│   │   ├── common.py      # RF, BalancedRF, EasyEnsemble, etc.
│   │   ├── helpers.py     # Training helper functions
│   │   ├── lstm.py        # LSTM architecture
│   │   ├── SvmA.py        # SVM architecture
│   │   └── train_helpers.py # Training utilities
│   ├── feature_selection/ # Feature selection methods
│   │   ├── anfis.py       # ANFIS-based selection
│   │   ├── feature_helpers.py
│   │   ├── index.py       # Feature selection index
│   │   └── rf_importance.py # Random Forest importance
│   ├── model_pipeline.py  # End-to-end training pipeline
│   └── train_stages.py    # Training stage utilities
│
├── utils/                 # Utility functions
│   ├── cli/               # CLI helper utilities
│   ├── domain_generalization/  # Domain generalization (CORAL, Mixup, VAE, Jitter)
│   ├── evaluation/        # Evaluation metrics and threshold optimization
│   ├── io/                # Data loading/saving utilities
│   └── visualization/     # Visualization tools (ROC, radar charts)
│
└── config.py              # Centralized configuration settings
```

---

## Notes
- **analysis/**: post-training evaluations, domain distance calculations, and imbalance analysis
- **data_pipeline/**: converts raw data (EEG, vehicle, physio) into processed features  
- **evaluation/**: evaluation framework using trained models  
- **models/**: classical and neural architectures + training pipelines  
- **utils/**: shared helpers for CLI, domain generalization, evaluation, I/O, and visualization

> **Note**: Data preparation scripts (subject grouping, feature checks) are located in `scripts/python/setup/`.  

---

## Future Work
- Add comprehensive docstrings (Google/NumPy style)  
- Consider packaging `src/` as a Python module (`setup.py` or `pyproject.toml`)  
- Expand unit tests under a dedicated `tests/` directory  
```

