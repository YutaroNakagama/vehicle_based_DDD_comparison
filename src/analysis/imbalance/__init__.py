"""Imbalance Analysis Module.

This subpackage provides analysis and visualization tools for
imbalanced data experiments in drowsy driving detection.

Modules
-------
analysis
    Core analysis functions for imbalance experiments
sampling
    Sampling distribution extraction and theoretical comparison
sample_distribution
    Visualizations for sample count changes across methods
performance_results
    Visualizations for performance metrics comparison
"""

from .sample_distribution import (
    create_sample_distribution_data,
    create_split_distribution_data,
    create_train_after_sampling_data,
    plot_sample_counts_bar,
    plot_class_ratio_comparison,
    plot_sample_change_waterfall,
    plot_method_summary_dashboard,
    plot_train_val_test_split,
    generate_all_visualizations,
)

from .performance_results import (
    create_performance_results_data,
    plot_recall_comparison,
    plot_precision_recall_scatter,
    plot_metrics_radar,
    plot_performance_summary_dashboard,
    plot_method_ranking,
    generate_performance_visualizations,
)

from .sampling import (
    extract_sampling_distribution,
    calculate_sampling_distribution,
    calculate_batch_distributions,
    compare_actual_vs_theoretical,
    DEFAULT_TRAIN_ALERT,
    DEFAULT_TRAIN_DROWSY,
)

from .analysis import (
    load_imbalance_results,
)

__all__ = [
    # Analysis
    "load_imbalance_results",
    # Sampling
    "extract_sampling_distribution",
    "calculate_sampling_distribution",
    "calculate_batch_distributions",
    "compare_actual_vs_theoretical",
    "DEFAULT_TRAIN_ALERT",
    "DEFAULT_TRAIN_DROWSY",
    # Sample distribution
    "create_sample_distribution_data",
    "create_split_distribution_data",
    "create_train_after_sampling_data",
    "plot_sample_counts_bar",
    "plot_class_ratio_comparison",
    "plot_sample_change_waterfall",
    "plot_method_summary_dashboard",
    "plot_train_val_test_split",
    "generate_all_visualizations",
    # Performance results
    "create_performance_results_data",
    "plot_recall_comparison",
    "plot_precision_recall_scatter",
    "plot_metrics_radar",
    "plot_performance_summary_dashboard",
    "plot_method_ranking",
    "generate_performance_visualizations",
]
