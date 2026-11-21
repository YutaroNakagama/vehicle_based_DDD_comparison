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
│   └── domain_generalization.md         Subject grouping and domain-shift pipeline
└── experiments/                       ← Reproducibility
    ├── reproducibility.md               How the experiments were run
    ├── conditions/                      Experiment conditions
    │   ├── 01-imbalance-conditions.md     Experiment 1 (class imbalance, pooled training)
    │   └── 02-domain-conditions.md        Experiment 2 (domain shift): the paper
    └── results/                         Working analysis notes (see below)
        ├── 01-imbalance-results.md
        ├── 02-domain-results.md
        └── exp2-analysis/                 Intermediate analyses of Experiment 2
```

## 🚀 Getting Started

1. **[Installation](getting-started/installation.md)** — Clone, set up Python environment, download dataset
2. **[Quickstart](getting-started/quickstart.md)** — Preprocess → Train → Evaluate in 5 minutes

## 📚 Reference

- **[Configuration](reference/configuration.md)** — Paths, sampling rates, Optuna parameters, labeling schemes
- **[Evaluation Metrics](reference/evaluation_metrics.md)** — AUPRC, F2, Recall, Precision, threshold selection
- **[Imbalance Methods](reference/imbalance_methods.md)** — SMOTE, RUS, Tomek, BalancedRF, EasyEnsemble
- **[Ranking Methods](reference/ranking_methods.md)** — kNN (used in the paper), LOF, Mean Distance, Isolation Forest, etc.

## 🏗️ Architecture

- **[Developer Guide](architecture/developer_guide.md)** — Repository structure, preprocessing/training/evaluation pipelines, HPC integration
- **[Domain Generalization](architecture/domain_generalization.md)** — Distance computation, In/Out grouping, Cross/Within/Mixed training

## 🧪 Experiments

- **[Reproducibility Guide](experiments/reproducibility.md)** — Experiment conditions, HPC submission, result checking
- **[Experiment 1 conditions](experiments/conditions/01-imbalance-conditions.md)** — Class imbalance, pooled training
- **[Experiment 2 conditions](experiments/conditions/02-domain-conditions.md)** — Domain shift: the design of the paper (126 cells × 12 seeds)
- **Working notes** — `experiments/results/` holds analysis notes written while the
  experiments were run. They predate the final runs, and their numbers can differ from the
  paper; the paper's numbers are reproduced by the scripts listed in the top-level
  `README.md`.
