# src/analysis/metrics/__init__.py
"""Metrics and evaluation analysis module.

This module provides tools for analyzing model evaluation metrics,
including confusion matrix visualization and metrics table generation.

Submodules
----------
confusion_matrix : Confusion matrix analysis and visualization
tables : Metrics aggregation and table generation
"""

from .confusion_matrix import (
    collect_eval_data,
    generate_distance_plot,
    generate_overview_plot,
    generate_summary_table,
    print_detailed_tables,
    extract_multiseed_results,
    aggregate_multiseed_results,
)
from .tables import (
    load_metrics_df,
    pivot_metrics_wide,
)

__all__ = [
    # confusion_matrix
    "collect_eval_data",
    "generate_distance_plot",
    "generate_overview_plot",
    "generate_summary_table",
    "print_detailed_tables",
    "extract_multiseed_results",
    "aggregate_multiseed_results",
    # tables
    "load_metrics_df",
    "pivot_metrics_wide",
]
