"""
Analysis Module
================

Core analysis functionality for the DDD comparison project.

Subpackages
-----------
domain : Domain analysis (distances, correlations, rankings, projections)
imbalance : Imbalance experiment analysis and visualization
metrics : Evaluation metrics analysis and visualization

Directory Structure
-------------------
src/analysis/
├── domain/           # Domain analysis
│   ├── distance.py       # Distance matrix computation (DTW, MMD, Wasserstein)
│   ├── correlation.py    # Distance-performance correlation
│   ├── projection.py     # Clustering visualization (t-SNE, UMAP)
│   ├── ranking.py        # Subject ranking
│   ├── group_comparison.py
│   ├── distance_report.py
│   └── group_generator.py
├── imbalance/        # Imbalance experiment analysis
│   ├── analysis.py       # Results analysis
│   ├── sampling.py       # Sampling distribution analysis
│   ├── performance_results.py
│   └── sample_distribution.py
└── metrics/          # Evaluation metrics
    ├── confusion_matrix.py
    └── tables.py
"""

# Re-export subpackages for convenience
from . import domain
from . import imbalance
from . import metrics

# Re-export commonly used functions from subpackages
from .metrics.confusion_matrix import (
    # Data loading
    load_confusion_matrix,
    parse_filename,
    collect_eval_data,
    # Visualization
    plot_confusion_matrix,
    generate_distance_plot,
    generate_overview_plot,
    # Tables
    generate_summary_table,
    print_detailed_tables,
    # Multi-seed aggregation
    extract_multiseed_results,
    aggregate_multiseed_results,
    generate_rates_visualization,
    create_aggregate_summary,
    # Constants
    DISTANCES,
    LEVELS,
    MODES,
)

from .imbalance.sampling import (
    extract_sampling_distribution,
    calculate_sampling_distribution,
    calculate_batch_distributions,
    compare_actual_vs_theoretical,
    DEFAULT_TRAIN_ALERT,
    DEFAULT_TRAIN_DROWSY,
)

__all__ = [
    # Subpackages
    "domain",
    "imbalance",
    "metrics",
    # Confusion matrix (from metrics)
    "load_confusion_matrix",
    "parse_filename",
    "collect_eval_data",
    "plot_confusion_matrix",
    "generate_distance_plot",
    "generate_overview_plot",
    "generate_summary_table",
    "print_detailed_tables",
    "extract_multiseed_results",
    "aggregate_multiseed_results",
    "generate_rates_visualization",
    "create_aggregate_summary",
    "DISTANCES",
    "LEVELS",
    "MODES",
    # Sampling (from imbalance)
    "extract_sampling_distribution",
    "calculate_sampling_distribution",
    "calculate_batch_distributions",
    "compare_actual_vs_theoretical",
    "DEFAULT_TRAIN_ALERT",
    "DEFAULT_TRAIN_DROWSY",
]
