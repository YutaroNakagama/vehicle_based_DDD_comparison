"""Analysis utilities package.

This package provides shared utilities for analysis modules.
"""

from src.utils.analysis.projection_utils import (
    get_projection_coords,
    get_available_projectors,
    cluster_kmeans,
    cluster_hierarchical,
    cluster_dbscan,
    cluster_spectral,
    compute_cluster_centroids,
    compute_mean_distances,
    CLUSTER_COLORS,
    DEFAULT_RANDOM_STATE,
)

from src.utils.analysis.statistical_utils import (
    cohens_d,
    interpret_cohens_d,
    wilcoxon_test,
    paired_ttest,
    bootstrap_ci,
    format_p_value,
)

from src.utils.analysis.distance_utils import (
    load_distance_matrix,
    load_group_subjects,
    load_all_subjects,
    get_group_indices,
    load_all_group_indices,
    compute_intergroup_distances,
    compute_intragroup_distances,
    compute_embedding,
    compute_centroids,
    METRICS,
    METRIC_DIRS,
    LEVELS,
    DISTANCE_DIR,
)

__all__ = [
    # Projection utilities
    "get_projection_coords",
    "get_available_projectors",
    "cluster_kmeans",
    "cluster_hierarchical",
    "cluster_dbscan",
    "cluster_spectral",
    "compute_cluster_centroids",
    "compute_mean_distances",
    "CLUSTER_COLORS",
    "DEFAULT_RANDOM_STATE",
    # Statistical utilities
    "cohens_d",
    "interpret_cohens_d",
    "wilcoxon_test",
    "paired_ttest",
    "bootstrap_ci",
    "format_p_value",
    # Distance utilities
    "load_distance_matrix",
    "load_group_subjects",
    "load_all_subjects",
    "get_group_indices",
    "load_all_group_indices",
    "compute_intergroup_distances",
    "compute_intragroup_distances",
    "compute_embedding",
    "compute_centroids",
    "METRICS",
    "METRIC_DIRS",
    "LEVELS",
    "DISTANCE_DIR",
]
