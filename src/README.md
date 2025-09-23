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
│   ├── metrics\_tables.py  # Table generation for metrics
│   ├── pretrain\_groups\_report.py # Report generation for pretrain/target groups
│   ├── radar.py           # Radar chart visualization
│   ├── rank\_export.py     # Export ranked feature/model results
│   └── summary\_groups.py  # Group-based summaries
│
├── data\_pipeline/         # Data preprocessing & feature extraction
│   ├── features/          # Feature definitions
│   │   ├── eeg.py         # EEG feature extraction
│   │   ├── kss.py         # Karolinska Sleepiness Scale label handling
│   │   ├── physio.py      # Physiological features (GSR, HR, etc.)
│   │   ├── simlsl.py      # Vehicle-based simulator features
│   │   └── wavelet.py     # Wavelet decomposition features
│   ├── processing\_pipeline.py     # Single-process preprocessing pipeline
│   └── processing\_pipeline\_mp.py  # Multi-process preprocessing pipeline
│
├── evaluation/            # Evaluation pipelines
│   ├── eval\_pipeline.py   # Unified evaluation entry point
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
│   ├── feature\_selection/ # Feature selection methods
│   │   ├── anfis.py
│   │   ├── rf\_importance.py
│   │   └── index.py
│   └── model\_pipeline.py  # End-to-end training pipeline
│
└── utils/                 # Utility functions
├── domain\_generalization/ # Domain generalization methods
│   ├── coral.py
│   ├── domain\_mixup.py
│   ├── jitter.py
│   └── vae\_augment.py
├── io/                # Data loading/saving utilities
│   ├── loaders.py
│   ├── merge.py
│   └── split.py
└── visualization/     # Visualization tools
└── visualization.py

```

---

## Notes
- **analysis/**: for post-training evaluations and domain distance calculations  
- **data_pipeline/**: converts raw data (EEG, vehicle, physio) into processed features  
- **evaluation/**: runs evaluation using trained models  
- **models/**: contains architectures and training pipelines  
- **utils/**: shared helpers for domain generalization, I/O, and plotting  

---

## Future Work
- Add docstrings for all modules (Google/NumPy style)
- Consider packaging `src/` as a Python module (`setup.py` or `pyproject.toml`)
```

