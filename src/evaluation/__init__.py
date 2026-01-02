"""Evaluation module for model assessment and metrics calculation.

This module provides:
- metrics: Classification metrics, ROC/PR curves, threshold optimization
- threshold: Threshold optimization and loading utilities
- eval_pipeline: Main evaluation pipeline
- eval_stages: Evaluation pipeline stages
"""

from src.evaluation.metrics import (
    calculate_classification_metrics,
    calculate_extended_metrics,
    calculate_roc_metrics,
    calculate_pr_metrics,
    find_optimal_threshold,
    apply_threshold,
)
from src.evaluation.threshold import (
    optimize_threshold_f2,
    load_or_optimize_threshold,
)
