"""
Analysis Module
================

Core analysis functionality for the DDD comparison project.
"""

from .confusion_matrix import (
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

from .sampling import (
    extract_sampling_distribution,
    calculate_sampling_distribution,
    calculate_batch_distributions,
    compare_actual_vs_theoretical,
    DEFAULT_TRAIN_ALERT,
    DEFAULT_TRAIN_DROWSY,
)

__all__ = [
    # Confusion matrix
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
    # Sampling
    "extract_sampling_distribution",
    "calculate_sampling_distribution",
    "calculate_batch_distributions",
    "compare_actual_vs_theoretical",
    "DEFAULT_TRAIN_ALERT",
    "DEFAULT_TRAIN_DROWSY",
]
