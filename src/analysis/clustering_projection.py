#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clustering_projection.py
========================

Generate projection plots with various clustering methods for domain analysis.

Clustering methods:
1. K-Means on projection coordinates (t-SNE/UMAP space)
2. Hierarchical (Agglomerative) clustering on distance matrix
3. DBSCAN on projection coordinates
4. Spectral Clustering on distance matrix

Each method generates separate projection plots for comparison.
"""

from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS, TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

from src import config as cfg
from src.utils.io.data_io import load_numpy, load_json, save_json

try:
    import umap
except ImportError:
    umap = None

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Color palettes for different number of clusters
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


def _get_projection_coords(matrix: np.ndarray, method: str = "MDS") -> np.ndarray:
    """Compute 2D projection coordinates from distance matrix.
    
    Parameters
    ----------
    matrix : np.ndarray
        Pairwise distance matrix (n x n).
    method : str
        Projection method: "MDS", "tSNE", or "UMAP".
    
    Returns
    -------
    np.ndarray
        2D coordinates (n x 2).
    """
    matrix_filled = np.nan_to_num(matrix)
    
    if method == "MDS":
        projector = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    elif method == "tSNE":
        projector = TSNE(
            n_components=2,
            metric="precomputed",
            init="random",
            random_state=42,
            perplexity=min(5, len(matrix) - 1)
        )
    elif method == "UMAP":
        if umap is None:
            raise ImportError("UMAP not installed. Use: pip install umap-learn")
        projector = umap.UMAP(n_components=2, metric="precomputed", random_state=42)
    else:
        raise ValueError(f"Unknown projection method: {method}")
    
    return projector.fit_transform(matrix_filled)


def cluster_kmeans(coords: np.ndarray, n_clusters: int = 3) -> np.ndarray:
    """K-Means clustering on projection coordinates.
    
    Parameters
    ----------
    coords : np.ndarray
        2D coordinates (n x 2).
    n_clusters : int
        Number of clusters.
    
    Returns
    -------
    np.ndarray
        Cluster labels (n,).
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return kmeans.fit_predict(coords)


def cluster_hierarchical(matrix: np.ndarray, n_clusters: int = 3) -> np.ndarray:
    """Hierarchical (Agglomerative) clustering on distance matrix.
    
    Parameters
    ----------
    matrix : np.ndarray
        Pairwise distance matrix (n x n).
    n_clusters : int
        Number of clusters.
    
    Returns
    -------
    np.ndarray
        Cluster labels (n,).
    """
    # Use precomputed distance matrix
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage="average"
    )
    return clustering.fit_predict(matrix)


def cluster_dbscan(coords: np.ndarray, eps: float = None, min_samples: int = 3) -> np.ndarray:
    """DBSCAN clustering on projection coordinates.
    
    Parameters
    ----------
    coords : np.ndarray
        2D coordinates (n x 2).
    eps : float, optional
        Maximum distance for neighborhood. If None, auto-estimate.
    min_samples : int
        Minimum samples in neighborhood.
    
    Returns
    -------
    np.ndarray
        Cluster labels (n,). -1 indicates noise/outlier.
    """
    if eps is None:
        # Auto-estimate eps using k-distance heuristic
        from sklearn.neighbors import NearestNeighbors
        k = min_samples
        neigh = NearestNeighbors(n_neighbors=k)
        neigh.fit(coords)
        distances, _ = neigh.kneighbors(coords)
        k_distances = np.sort(distances[:, k-1])
        # Use knee point (simple heuristic: 90th percentile)
        eps = np.percentile(k_distances, 90)
        logger.info(f"DBSCAN auto eps: {eps:.4f}")
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    return dbscan.fit_predict(coords)


def cluster_spectral(matrix: np.ndarray, n_clusters: int = 3) -> np.ndarray:
    """Spectral clustering on distance matrix.
    
    Parameters
    ----------
    matrix : np.ndarray
        Pairwise distance matrix (n x n).
    n_clusters : int
        Number of clusters.
    
    Returns
    -------
    np.ndarray
        Cluster labels (n,).
    """
    # Convert distance to affinity (similarity)
    # Using RBF kernel: affinity = exp(-gamma * distance^2)
    gamma = 1.0 / (2.0 * np.median(matrix[matrix > 0]) ** 2)
    affinity = np.exp(-gamma * matrix ** 2)
    np.fill_diagonal(affinity, 1.0)
    
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        random_state=42
    )
    return spectral.fit_predict(affinity)


def plot_projection_with_clusters(
    coords: np.ndarray,
    subjects: list[str],
    labels: np.ndarray,
    metric: str,
    proj_method: str,
    cluster_method: str,
    outdir: Path,
    show_labels: bool = True
) -> None:
    """Plot projection with cluster coloring.
    
    Parameters
    ----------
    coords : np.ndarray
        2D coordinates (n x 2).
    subjects : list[str]
        Subject IDs.
    labels : np.ndarray
        Cluster labels.
    metric : str
        Distance metric name (mmd, wasserstein, dtw).
    proj_method : str
        Projection method name (MDS, tSNE, UMAP).
    cluster_method : str
        Clustering method name.
    outdir : Path
        Output directory for plots.
    show_labels : bool
        Whether to show subject ID labels.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        if label == -1:
            # DBSCAN noise points
            color = "black"
            marker = "x"
            label_name = "Noise"
        else:
            color = CLUSTER_COLORS[label % len(CLUSTER_COLORS)]
            marker = "o"
            label_name = f"Cluster {label + 1}"
        
        ax.scatter(
            coords[mask, 0], 
            coords[mask, 1], 
            c=color, 
            marker=marker,
            s=50, 
            label=label_name,
            alpha=0.7
        )
        
        if show_labels:
            for idx in np.where(mask)[0]:
                ax.annotate(
                    subjects[idx],
                    (coords[idx, 0], coords[idx, 1]),
                    fontsize=5,
                    alpha=0.7
                )
    
    ax.set_title(f"{metric.upper()} - {proj_method} + {cluster_method}\n({n_clusters} clusters)")
    ax.set_xlabel(f"{proj_method} Dim 1")
    ax.set_ylabel(f"{proj_method} Dim 2")
    ax.legend(loc="best", fontsize=8)
    ax.set_aspect('equal', adjustable='datalim')
    
    fig.tight_layout()
    
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"{metric}_{proj_method.lower()}_{cluster_method.lower()}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out_path}")


def plot_dendrogram(
    matrix: np.ndarray,
    subjects: list[str],
    metric: str,
    outdir: Path,
    n_clusters: int = 3
) -> None:
    """Plot hierarchical clustering dendrogram.
    
    Parameters
    ----------
    matrix : np.ndarray
        Pairwise distance matrix.
    subjects : list[str]
        Subject IDs.
    metric : str
        Distance metric name.
    outdir : Path
        Output directory.
    n_clusters : int
        Number of clusters for color threshold.
    """
    # Convert square distance matrix to condensed form
    from scipy.spatial.distance import squareform
    condensed = squareform(matrix)
    
    # Compute linkage
    Z = linkage(condensed, method='average')
    
    # Plot
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Compute color threshold for n_clusters
    # Get the distances at which clusters are merged
    distances = Z[:, 2]
    if len(distances) >= n_clusters:
        threshold = distances[-n_clusters + 1]
    else:
        threshold = 0
    
    dendrogram(
        Z,
        labels=subjects,
        leaf_rotation=90,
        leaf_font_size=6,
        ax=ax,
        color_threshold=threshold
    )
    
    ax.set_title(f"{metric.upper()} - Hierarchical Clustering Dendrogram")
    ax.set_xlabel("Subject")
    ax.set_ylabel("Distance")
    
    fig.tight_layout()
    
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"{metric}_dendrogram.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out_path}")


def run_clustering_analysis(
    metric: str = "mmd",
    n_clusters: int = 3,
    output_subdir: str = "clustering"
) -> dict:
    """Run all clustering methods and generate projection plots.
    
    Parameters
    ----------
    metric : str
        Distance metric (mmd, wasserstein, dtw).
    n_clusters : int
        Number of clusters for K-Means, Hierarchical, Spectral.
    output_subdir : str
        Subdirectory name for outputs.
    
    Returns
    -------
    dict
        Clustering results with labels for each method.
    """
    # Load distance matrix and subjects
    base_dir = Path(cfg.RESULTS_DOMAIN_ANALYSIS_PATH) / "distance" / "subject-wise" / metric
    matrix_path = base_dir / f"{metric}_matrix.npy"
    subjects_path = base_dir / f"{metric}_subjects.json"
    
    if not matrix_path.exists() or not subjects_path.exists():
        raise FileNotFoundError(f"Distance matrix or subjects file not found for {metric}")
    
    matrix = load_numpy(matrix_path)
    subjects = load_json(subjects_path)
    
    logger.info(f"Loaded {metric.upper()} matrix: {matrix.shape}, {len(subjects)} subjects")
    
    # Output directory
    outdir = base_dir / "png" / output_subdir
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Projection methods
    proj_methods = ["MDS", "tSNE"]
    if umap is not None:
        proj_methods.append("UMAP")
    
    # Compute projections
    projections = {}
    for pm in proj_methods:
        logger.info(f"Computing {pm} projection...")
        projections[pm] = _get_projection_coords(matrix, pm)
    
    # Clustering results
    results = {"metric": metric, "n_subjects": len(subjects), "clusters": {}}
    
    # 1. K-Means on each projection
    logger.info("Running K-Means clustering...")
    for pm, coords in projections.items():
        labels = cluster_kmeans(coords, n_clusters)
        results["clusters"][f"kmeans_{pm.lower()}"] = labels.tolist()
        plot_projection_with_clusters(
            coords, subjects, labels, metric, pm, f"KMeans(k={n_clusters})", outdir
        )
    
    # 2. Hierarchical clustering on distance matrix
    logger.info("Running Hierarchical clustering...")
    labels_hier = cluster_hierarchical(matrix, n_clusters)
    results["clusters"]["hierarchical"] = labels_hier.tolist()
    for pm, coords in projections.items():
        plot_projection_with_clusters(
            coords, subjects, labels_hier, metric, pm, f"Hierarchical(k={n_clusters})", outdir
        )
    
    # Plot dendrogram
    plot_dendrogram(matrix, subjects, metric, outdir, n_clusters)
    
    # 3. DBSCAN on each projection
    logger.info("Running DBSCAN clustering...")
    for pm, coords in projections.items():
        labels = cluster_dbscan(coords, min_samples=3)
        n_found = len(set(labels)) - (1 if -1 in labels else 0)
        results["clusters"][f"dbscan_{pm.lower()}"] = labels.tolist()
        plot_projection_with_clusters(
            coords, subjects, labels, metric, pm, f"DBSCAN(n={n_found})", outdir
        )
    
    # 4. Spectral clustering on distance matrix
    logger.info("Running Spectral clustering...")
    labels_spectral = cluster_spectral(matrix, n_clusters)
    results["clusters"]["spectral"] = labels_spectral.tolist()
    for pm, coords in projections.items():
        plot_projection_with_clusters(
            coords, subjects, labels_spectral, metric, pm, f"Spectral(k={n_clusters})", outdir
        )
    
    # Save clustering results
    results_path = outdir / f"{metric}_clustering_labels.json"
    save_json(results, results_path)
    logger.info(f"Saved clustering labels: {results_path}")
    
    return results


def main():
    """Run clustering analysis for all metrics."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Clustering-based projection analysis")
    parser.add_argument("--metric", choices=["mmd", "wasserstein", "dtw", "all"], default="all",
                        help="Distance metric to analyze")
    parser.add_argument("--n_clusters", type=int, default=3,
                        help="Number of clusters for K-Means, Hierarchical, Spectral")
    parser.add_argument("--output_subdir", default="clustering",
                        help="Subdirectory name for outputs")
    args = parser.parse_args()
    
    metrics = cfg.DISTANCE_METRICS if args.metric == "all" else [args.metric]
    
    for metric in metrics:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing {metric.upper()}")
        logger.info(f"{'='*50}")
        try:
            run_clustering_analysis(
                metric=metric,
                n_clusters=args.n_clusters,
                output_subdir=args.output_subdir
            )
        except Exception as e:
            logger.error(f"Failed for {metric}: {e}")
            raise


if __name__ == "__main__":
    main()
