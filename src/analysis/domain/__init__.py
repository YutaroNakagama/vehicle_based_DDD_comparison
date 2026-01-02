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

Utility Submodules
------------------
distance_utils : Low-level distance computation utilities
projection_utils : Dimensionality reduction and clustering helpers
statistical_utils : Statistical test helpers (Cohen's d, Wilcoxon, etc.)
"""

# Import submodules for easy access (lazy loading)
# Users can access via: from src.analysis.domain import correlation
# Or directly: from src.analysis.domain.correlation import run_distance_vs_delta

__all__ = [
    # Main analysis modules
    "distance",
    "correlation",
    "projection",
    "ranking",
    "group_comparison",
    "distance_report",
    "group_generator",
    # Utility modules
    "distance_utils",
    "projection_utils",
    "statistical_utils",
]
