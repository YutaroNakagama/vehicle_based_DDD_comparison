# Documentation

Project documentation for the Vehicle-Based Driver Drowsiness Detection comparison study.

## 📖 Structure

```
docs/
├── README.md                          ← This file (navigation index)
├── getting-started/                   ← For new users
│   ├── installation.md                  Setup and dependencies
│   └── quickstart.md                    First experiment in 5 minutes
├── reference/                         ← Technical references
│   ├── configuration.md                 All configurable parameters
│   ├── evaluation_metrics.md            Metric definitions and guidelines
│   ├── imbalance_methods.md             Oversampling/undersampling methods
│   └── ranking_methods.md               Subject ranking algorithms
├── architecture/                      ← System design and pipelines
│   ├── developer_guide.md               Repository architecture and data flow
│   ├── domain_generalization.md         Domain analysis pipeline
│   └── prior_research.md               SvmA, SvmW, Lstm baselines
└── experiments/                       ← Reproducibility and results
    ├── reproducibility.md               How to reproduce all experiments
    └── results/                         Historical experiment logs
        ├── imbalance_results.md           Imbalance experiment (2026-01-10)
        ├── domain_results.md              Domain shift experiment (2025-01-10)
        └── prior_research_results.md      Prior research experiment (2025-01-10)
```

## 🚀 Getting Started

1. **[Installation](getting-started/installation.md)** — Clone, set up Python environment, download dataset
2. **[Quickstart](getting-started/quickstart.md)** — Preprocess → Train → Evaluate in 5 minutes

## 📚 Reference

- **[Configuration](reference/configuration.md)** — Paths, sampling rates, Optuna parameters, labeling schemes
- **[Evaluation Metrics](reference/evaluation_metrics.md)** — AUPRC, F2, Recall, Precision, threshold selection
- **[Imbalance Methods](reference/imbalance_methods.md)** — SMOTE, RUS, Tomek, BalancedRF, EasyEnsemble
- **[Ranking Methods](reference/ranking_methods.md)** — KNN, LOF, Mean Distance, Isolation Forest, etc.

## 🏗️ Architecture

- **[Developer Guide](architecture/developer_guide.md)** — Repository structure, preprocessing/training/evaluation pipelines, HPC integration
- **[Domain Generalization](architecture/domain_generalization.md)** — Distance computation, subject ranking, cross-domain training
- **[Prior Research](architecture/prior_research.md)** — SvmA (ANFIS+SVM), SvmW (Wavelet+SVM), Lstm (BiLSTM+Attention)

## 🧪 Experiments

- **[Reproducibility Guide](experiments/reproducibility.md)** — Experiment conditions, HPC submission, result checking
- **[Experiment Results](experiments/results/)** — Historical logs separated from reference documentation
