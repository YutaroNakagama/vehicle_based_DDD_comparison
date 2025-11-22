"""Evaluation utilities for model performance assessment.

This module provides reusable functions for calculating classification metrics,
extracting metrics from evaluation results, and generating performance reports.
"""

from .metrics import (
    calculate_classification_metrics,
    calculate_roc_metrics,
    calculate_pr_metrics,
    calculate_extended_metrics,
    find_optimal_threshold,
    apply_threshold,
    calculate_class_specific_metrics,
    get_positive_class_block,
    get_metric_from_positive_class,
    estimate_positive_rate,
    compute_f2_score_from_pr,
    extract_metrics_from_eval_json,
)

from .threshold import (
    optimize_threshold_f2,
    load_or_optimize_threshold,
    extract_jobid_components,
)

__all__ = [
    "calculate_classification_metrics",
    "calculate_roc_metrics",
    "calculate_pr_metrics",
    "calculate_extended_metrics",
    "find_optimal_threshold",
    "apply_threshold",
    "calculate_class_specific_metrics",
    "get_positive_class_block",
    "get_metric_from_positive_class",
    "estimate_positive_rate",
    "compute_f2_score_from_pr",
    "extract_metrics_from_eval_json",
    "optimize_threshold_f2",
    "load_or_optimize_threshold",
    "extract_jobid_components",
]
