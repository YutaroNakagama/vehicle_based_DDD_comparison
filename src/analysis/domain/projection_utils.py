"""Shared projection and clustering utilities for distance-based analysis.

This module provides unified projection (MDS, t-SNE, UMAP) and clustering
(K-Means, Hierarchical, DBSCAN, Spectral) functions used across multiple
analysis modules.

Functions
---------
get_projection_coords(matrix, method) -> np.ndarray
    Compute 2D projection coordinates from distance matrix.
cluster_kmeans(coords, n_clusters) -> np.ndarray
    K-Means clustering on projection coordinates.
cluster_hierarchical(matrix, n_clusters) -> np.ndarray
    Hierarchical clustering on distance matrix.
cluster_dbscan(coords, eps, min_samples) -> np.ndarray
    DBSCAN clustering on projection coordinates.
cluster_spectral(matrix, n_clusters) -> np.ndarray
    Spectral clustering on affinity matrix.
"""

import logging
from typing import Optional

import numpy as np
from sklearn.manifold import MDS, TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.neighbors import NearestNeighbors

try:
    import umap
except ImportError:
    umap = None

# Default random state for reproducibility
DEFAULT_RANDOM_STATE = 42

# Default cluster colors for visualization
CLUSTER_COLORS = [
    "#e41a1c",  # red
    "#377eb8",  # blue
    "#4daf4a",  # green
    "#984ea3",  # purple
    "#ff7f00",  # orange
    "#ffff33",  # yellow
    "#a65628",  # brown
    "#f781bf",  # pink
    "#999999",  # gray
    "#00ced1",  # dark cyan
]


def get_projection_coords(
    matrix: np.ndarray,
    method: str = "MDS",
    random_state: int = DEFAULT_RANDOM_STATE,
) -> np.ndarray:
    """Compute 2D projection coordinates from distance matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Pairwise distance matrix (n x n).
    method : str, default="MDS"
        Projection method: "MDS", "tSNE", or "UMAP".
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        2D coordinates (n x 2).

    Raises
    ------
    ImportError
        If UMAP is requested but not installed.
    ValueError
        If unknown projection method is specified.
    """
    matrix_filled = np.nan_to_num(matrix)
    n_samples = len(matrix)

    if method == "MDS":
        projector = MDS(
            n_components=2,
            dissimilarity="precomputed",
            random_state=random_state,
        )
    elif method == "tSNE":
        # t-SNE must use init='random' when metric='precomputed'
        perplexity = min(5, n_samples - 1)
        projector = TSNE(
            n_components=2,
            metric="precomputed",
            init="random",
            random_state=random_state,
            perplexity=perplexity,
        )
    elif method == "UMAP":
        if umap is None:
            raise ImportError("UMAP not installed. Use: pip install umap-learn")
        projector = umap.UMAP(
            n_components=2,
            metric="precomputed",
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown projection method: {method}")

    return projector.fit_transform(matrix_filled)


def get_available_projectors(matrix: np.ndarray, random_state: int = DEFAULT_RANDOM_STATE) -> dict:
    """Get dictionary of available projection methods.

    Parameters
    ----------
    matrix : np.ndarray
        Pairwise distance matrix (for determining perplexity).
    random_state : int, default=42
        Random seed.

    Returns
    -------
    dict
        Dictionary mapping method names to projector instances.
    """
    n_samples = len(matrix)
    perplexity = min(5, n_samples - 1)

    methods = {
        "MDS": MDS(n_components=2, dissimilarity="precomputed", random_state=random_state),
        "tSNE": TSNE(
            n_components=2,
            metric="precomputed",
            init="random",
            random_state=random_state,
            perplexity=perplexity,
        ),
    }

    if umap is not None:
        methods["UMAP"] = umap.UMAP(
            n_components=2,
            metric="precomputed",
            random_state=random_state,
        )

    return methods


def cluster_kmeans(
    coords: np.ndarray,
    n_clusters: int = 3,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> np.ndarray:
    """K-Means clustering on projection coordinates.

    Parameters
    ----------
    coords : np.ndarray
        2D coordinates (n x 2).
    n_clusters : int, default=3
        Number of clusters.
    random_state : int, default=42
        Random seed.

    Returns
    -------
    np.ndarray
        Cluster labels (n,).
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    return kmeans.fit_predict(coords)


def cluster_hierarchical(
    matrix: np.ndarray,
    n_clusters: int = 3,
    linkage: str = "average",
) -> np.ndarray:
    """Hierarchical (Agglomerative) clustering on distance matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Pairwise distance matrix (n x n).
    n_clusters : int, default=3
        Number of clusters.
    linkage : str, default="average"
        Linkage criterion: "ward", "complete", "average", "single".

    Returns
    -------
    np.ndarray
        Cluster labels (n,).
    """
    agg = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage=linkage,
    )
    return agg.fit_predict(matrix)


def cluster_dbscan(
    coords: np.ndarray,
    eps: Optional[float] = None,
    min_samples: int = 3,
) -> np.ndarray:
    """DBSCAN clustering on projection coordinates.

    Parameters
    ----------
    coords : np.ndarray
        2D coordinates (n x 2).
    eps : float, optional
        Maximum distance between samples. If None, estimated automatically.
    min_samples : int, default=3
        Minimum samples in a neighborhood.

    Returns
    -------
    np.ndarray
        Cluster labels (n,). -1 indicates noise points.
    """
    if eps is None:
        # Estimate eps using k-distance graph
        nn = NearestNeighbors(n_neighbors=min_samples)
        nn.fit(coords)
        distances, _ = nn.kneighbors(coords)
        eps = np.mean(distances[:, -1]) * 1.5
        logging.info(f"DBSCAN auto-estimated eps: {eps:.4f}")

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    return dbscan.fit_predict(coords)


def cluster_spectral(
    matrix: np.ndarray,
    n_clusters: int = 3,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> np.ndarray:
    """Spectral clustering on affinity matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Pairwise distance matrix (n x n).
    n_clusters : int, default=3
        Number of clusters.
    random_state : int, default=42
        Random seed.

    Returns
    -------
    np.ndarray
        Cluster labels (n,).
    """
    # Convert distance matrix to affinity matrix using RBF kernel
    gamma = 1.0 / (2 * np.mean(matrix) ** 2)
    affinity = np.exp(-gamma * matrix ** 2)
    np.fill_diagonal(affinity, 1.0)

    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        random_state=random_state,
    )
    return spectral.fit_predict(affinity)


def compute_cluster_centroids(
    coords: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """Compute centroid coordinates for each cluster.

    Parameters
    ----------
    coords : np.ndarray
        2D coordinates (n x 2).
    labels : np.ndarray
        Cluster labels (n,).

    Returns
    -------
    dict
        Dictionary mapping cluster label to centroid coordinates.
    """
    unique_labels = np.unique(labels[labels >= 0])  # Exclude noise (-1)
    centroids = {}
    for label in unique_labels:
        mask = labels == label
        centroids[int(label)] = coords[mask].mean(axis=0)
    return centroids


def compute_mean_distances(matrix: np.ndarray) -> np.ndarray:
    """Compute mean distance to all other subjects for each subject.

    Parameters
    ----------
    matrix : np.ndarray
        Pairwise distance matrix (n x n).

    Returns
    -------
    np.ndarray
        Mean distances (n,).
    """
    matrix_masked = matrix.copy()
    np.fill_diagonal(matrix_masked, np.nan)
    return np.nanmean(matrix_masked, axis=1)
