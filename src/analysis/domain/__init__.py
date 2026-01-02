# src/analysis/domain/__init__.py
"""Domain analysis module.

This module provides tools for analyzing domain characteristics of subjects,
including distance computation, correlation analysis, subject ranking, and
clustering/projection visualization.

Submodules
----------
distance : Distance matrix computation (DTW, MMD, Wasserstein)
correlation : Distance-performance correlation analysis
projection : Clustering and projection visualization (t-SNE, UMAP)
ranking : Subject ranking and group generation
group_comparison : Metrics comparison across subject groups
distance_report : Distance statistics reporting
group_generator : Generate ranked subject group files
"""

from .distance import (
    compute_distance_matrix,
    compute_distance_matrix_parallel,
)
from .correlation import (
    compute_group_distances,
    compute_correlations,
)
from .ranking import (
    rank_subjects_by_mean_distance,
    rank_subjects_by_std_distance,
)

__all__ = [
    # distance
    "compute_distance_matrix",
    "compute_distance_matrix_parallel",
    # correlation
    "compute_group_distances",
    "compute_correlations",
    # ranking
    "rank_subjects_by_mean_distance",
    "rank_subjects_by_std_distance",
]
