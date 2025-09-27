# src/ directory

This directory contains the **core source code** for the Vehicle-Based DDD Comparison project.  
It is organized into modular components to handle data preprocessing, feature extraction, model training, evaluation, and analysis.

---

## Structure

```

src/
├── analysis/              # Correlation & distance analysis
│   ├── correlation.py     # Correlation between domain metrics and model performance
│   ├── distances.py       # Distance computation (MMD, Wasserstein, DTW)
│   ├── metrics_tables.py  # Table generation for metrics
│   ├── pretrain_groups_report.py # Report generation for pretrain/target groups
│   ├── radar.py           # Radar chart visualization
│   ├── rank_export.py     # Export ranked feature/model results
│   └── summary_groups.py  # Group-based summaries
│
├── data/                  # Data-related utilities (checks, subject grouping)
│   ├── check_feature_columns.py
│   ├── check_only_modes.py
│   ├── make_pretrain_group.py
│   └── make_target_groups.py
│
├── data_pipeline/         # Data preprocessing & feature extraction
│   ├── features/          # Feature definitions
│   │   ├── eeg.py         # EEG feature extraction
│   │   ├── kss.py         # Karolinska Sleepiness Scale label handling
│   │   ├── physio.py      # Physiological features (GSR, HR, etc.)
│   │   ├── simlsl.py      # Vehicle-based simulator features
│   │   └── wavelet.py     # Wavelet decomposition features
│   ├── processing_pipeline.py     # Single-process preprocessing pipeline
│   └── processing_pipeline_mp.py  # Multi-process preprocessing pipeline
│
├── evaluation/            # Evaluation pipelines
│   ├── eval_pipeline.py   # Unified evaluation entry point
│   └── models/            # Evaluation models
│       ├── common.py
│       ├── lstm.py
│       └── SvmA.py
│
├── models/                # Model definitions & training pipelines
│   ├── architectures/     # Classical & neural model architectures
│   │   ├── common.py
│   │   ├── lstm.py
│   │   └── SvmA.py
│   ├── feature_selection/ # Feature selection methods
│   │   ├── anfis.py
│   │   ├── rf_importance.py
│   │   └── index.py
│   └── model_pipeline.py  # End-to-end training pipeline
│
└── utils/                 # Utility functions
├── domain_generalization/ # Domain generalization methods (CORAL, Mixup, VAE, etc.)
├── io/                   # Data loading/saving utilities
└── visualization/        # Visualization tools

```

---

## Notes
- **analysis/**: post-training evaluations and domain distance calculations  
- **data/**: helper scripts for subject grouping and dataset checks  
- **data_pipeline/**: converts raw data (EEG, vehicle, physio) into processed features  
- **evaluation/**: evaluation framework using trained models  
- **models/**: classical and neural architectures + training pipelines  
- **utils/**: shared helpers for domain generalization, I/O, and visualization  

---

## Future Work
- Add comprehensive docstrings (Google/NumPy style)  
- Consider packaging `src/` as a Python module (`setup.py` or `pyproject.toml`)  
- Expand unit tests under a dedicated `tests/` directory  
```

