#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clustering_projection_ranked.py
===============================

Generate projection plots with clustering methods, labeling clusters as
High/Middle/Low based on their distance from the center (like the original
mean-distance ranking approach).

Clustering methods:
1. K-Means on projection coordinates
2. Hierarchical (Agglomerative) clustering on distance matrix
3. DBSCAN on projection coordinates (clusters ranked by density/outlier score)
4. Spectral Clustering on distance matrix

Each cluster is ranked by:
- Mean distance of cluster members to the global centroid
- High = furthest from center (outliers)
- Low = closest to center (typical)
- Middle = in between
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS, TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from scipy.cluster.hierarchy import dendrogram, linkage

from src import config as cfg
from src.utils.io.data_io import load_numpy, load_json, save_json

try:
    import umap
except ImportError:
    umap = None

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Fixed colors for High/Middle/Low (consistent with original)
RANK_COLORS = {
    "High": "#e41a1c",    # red - outliers
    "Middle": "#999999",  # gray - intermediate
    "Low": "#377eb8",     # blue - typical/central
    "Noise": "#000000",   # black - DBSCAN noise
}

# Fixed group size (29 subjects each)
GROUP_SIZE = 29


# Available ranking methods
RANKING_METHODS = [
    "mean_distance",        # 現在の方法: 他被験者との平均距離 (baseline)
    "centroid_umap",        # UMAP空間でのクラスタ中心からの距離 (cluster-aware)
    "lof",                  # Local Outlier Factor (density-based outlier detection)
]

# Legacy methods (not used by default, but available)
RANKING_METHODS_LEGACY = [
    "centroid_mds",         # MDS空間でのクラスタ中心からの距離
    "medoid",               # Medoid（実データ点の代表）からの距離
]


def _get_projection_coords(matrix: np.ndarray, method: str = "MDS") -> np.ndarray:
    """Compute 2D projection coordinates from distance matrix."""
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
            raise ImportError("UMAP not installed")
        projector = umap.UMAP(n_components=2, metric="precomputed", random_state=42)
    else:
        raise ValueError(f"Unknown projection method: {method}")
    
    return projector.fit_transform(matrix_filled)


def _rank_clusters_by_distance(
    coords: np.ndarray,
    labels: np.ndarray,
    matrix: np.ndarray
) -> Tuple[Dict[int, str], np.ndarray]:
    """Rank subjects as High/Middle/Low based on mean distance, fixed 29 each.
    
    Instead of ranking clusters, we rank individual subjects by their mean
    distance to all other subjects, then assign top 29 to High, middle 29 to
    Middle, and bottom 29 to Low.
    
    Parameters
    ----------
    coords : np.ndarray
        2D projection coordinates (n x 2). Not used in current implementation.
    labels : np.ndarray
        Cluster labels. Not used in current implementation.
    matrix : np.ndarray
        Original distance matrix.
    
    Returns
    -------
    Tuple[Dict[int, str], np.ndarray]
        - rank_labels: array of rank names for each subject ("High", "Middle", "Low")
        - subject_ranks: indices sorted by mean distance (descending)
    """
    n_subjects = matrix.shape[0]
    
    # Calculate mean distance for each subject (excluding diagonal)
    matrix_masked = matrix.copy()
    np.fill_diagonal(matrix_masked, np.nan)
    mean_distances = np.nanmean(matrix_masked, axis=1)
    
    # Sort subjects by mean distance (descending: highest = High)
    sorted_indices = np.argsort(-mean_distances)  # descending order
    
    # Assign ranks: top GROUP_SIZE -> High, middle GROUP_SIZE -> Middle, bottom GROUP_SIZE -> Low
    rank_labels = np.array(["Other"] * n_subjects, dtype=object)
    
    # High: top GROUP_SIZE subjects (highest mean distance = outliers)
    high_indices = sorted_indices[:GROUP_SIZE]
    rank_labels[high_indices] = "High"
    
    # Middle: centered GROUP_SIZE subjects
    mid_start = n_subjects // 2 - GROUP_SIZE // 2
    mid_end = mid_start + GROUP_SIZE
    middle_indices = sorted_indices[mid_start:mid_end]
    rank_labels[middle_indices] = "Middle"
    
    # Low: bottom GROUP_SIZE subjects (lowest mean distance = central)
    low_indices = sorted_indices[-GROUP_SIZE:]
    rank_labels[low_indices] = "Low"
    
    return rank_labels, sorted_indices


def _rank_by_centroid_distance(
    coords: np.ndarray,
    group_size: int = GROUP_SIZE
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rank subjects by distance from centroid in projection space.
    
    This method computes the centroid (mean point) of all subjects in the
    projection space, then ranks subjects by their Euclidean distance from
    this centroid.
    
    Parameters
    ----------
    coords : np.ndarray
        2D projection coordinates (n x 2).
    group_size : int
        Number of subjects per group.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        - rank_labels: array of rank names ("High", "Middle", "Low")
        - distances: distances from centroid for each subject
        - centroid: the centroid coordinates
    """
    n_subjects = coords.shape[0]
    
    # Compute centroid (center of mass)
    centroid = np.mean(coords, axis=0)
    
    # Calculate Euclidean distance from centroid for each subject
    distances = np.sqrt(np.sum((coords - centroid) ** 2, axis=1))
    
    # Sort by distance (ascending: closest = Low, furthest = High)
    sorted_indices = np.argsort(distances)  # ascending order
    
    # Assign ranks
    rank_labels = np.array(["Other"] * n_subjects, dtype=object)
    
    # Low: closest GROUP_SIZE subjects (near centroid = typical)
    low_indices = sorted_indices[:group_size]
    rank_labels[low_indices] = "Low"
    
    # Middle: centered GROUP_SIZE subjects
    mid_start = n_subjects // 2 - group_size // 2
    mid_end = mid_start + group_size
    middle_indices = sorted_indices[mid_start:mid_end]
    rank_labels[middle_indices] = "Middle"
    
    # High: furthest GROUP_SIZE subjects (outliers)
    high_indices = sorted_indices[-group_size:]
    rank_labels[high_indices] = "High"
    
    return rank_labels, distances, centroid


def _rank_by_medoid_distance(
    matrix: np.ndarray,
    group_size: int = GROUP_SIZE
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Rank subjects by distance from medoid (most central real data point).
    
    Unlike centroid, medoid is an actual data point that minimizes the sum of
    distances to all other points. This is more robust to outliers.
    
    Parameters
    ----------
    matrix : np.ndarray
        Distance matrix (n x n).
    group_size : int
        Number of subjects per group.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, int]
        - rank_labels: array of rank names ("High", "Middle", "Low")
        - distances: distances from medoid for each subject
        - medoid_idx: index of the medoid subject
    """
    n_subjects = matrix.shape[0]
    
    # Find medoid: point with minimum sum of distances to all others
    sum_distances = np.nansum(matrix, axis=1)
    medoid_idx = np.argmin(sum_distances)
    
    # Get distances from medoid
    distances = matrix[medoid_idx, :].copy()
    
    # Sort by distance (ascending: closest = Low, furthest = High)
    sorted_indices = np.argsort(distances)
    
    # Assign ranks
    rank_labels = np.array(["Other"] * n_subjects, dtype=object)
    
    # Low: closest to medoid (most typical)
    low_indices = sorted_indices[:group_size]
    rank_labels[low_indices] = "Low"
    
    # Middle: intermediate distance
    mid_start = n_subjects // 2 - group_size // 2
    mid_end = mid_start + group_size
    middle_indices = sorted_indices[mid_start:mid_end]
    rank_labels[middle_indices] = "Middle"
    
    # High: furthest from medoid (outliers)
    high_indices = sorted_indices[-group_size:]
    rank_labels[high_indices] = "High"
    
    return rank_labels, distances, medoid_idx


def _rank_by_lof(
    coords: np.ndarray,
    group_size: int = GROUP_SIZE,
    n_neighbors: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """Rank subjects by Local Outlier Factor (LOF) score.
    
    LOF measures the local density deviation of a data point compared to its
    neighbors. Points with higher LOF scores are more likely to be outliers.
    
    This is a common technique in domain adaptation research for identifying
    samples that are far from the distribution center.
    
    Parameters
    ----------
    coords : np.ndarray
        2D projection coordinates (n x 2).
    group_size : int
        Number of subjects per group.
    n_neighbors : int
        Number of neighbors for LOF computation.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - rank_labels: array of rank names ("High", "Middle", "Low")
        - lof_scores: LOF scores (negative, higher = more outlier-like)
    """
    from sklearn.neighbors import LocalOutlierFactor
    
    n_subjects = coords.shape[0]
    
    # Compute LOF scores
    lof = LocalOutlierFactor(n_neighbors=min(n_neighbors, n_subjects - 1), novelty=False)
    lof.fit(coords)
    
    # negative_outlier_factor_: more negative = more outlier-like
    # We negate to get: higher value = more outlier-like
    lof_scores = -lof.negative_outlier_factor_
    
    # Sort by LOF score (descending: highest = most outlier = High)
    sorted_indices = np.argsort(-lof_scores)
    
    # Assign ranks
    rank_labels = np.array(["Other"] * n_subjects, dtype=object)
    
    # High: highest LOF scores (outliers)
    high_indices = sorted_indices[:group_size]
    rank_labels[high_indices] = "High"
    
    # Middle: intermediate LOF scores
    mid_start = n_subjects // 2 - group_size // 2
    mid_end = mid_start + group_size
    middle_indices = sorted_indices[mid_start:mid_end]
    rank_labels[middle_indices] = "Middle"
    
    # Low: lowest LOF scores (typical, inliers)
    low_indices = sorted_indices[-group_size:]
    rank_labels[low_indices] = "Low"
    
    return rank_labels, lof_scores


def _compute_ranking(
    matrix: np.ndarray,
    coords: np.ndarray,
    method: str = "centroid_mds"
) -> Tuple[np.ndarray, dict]:
    """Compute subject ranking using the specified method.
    
    Parameters
    ----------
    matrix : np.ndarray
        Distance matrix.
    coords : np.ndarray
        Projection coordinates.
    method : str
        Ranking method name.
    
    Returns
    -------
    Tuple[np.ndarray, dict]
        - rank_labels: array of rank names for each subject
        - info: dictionary with method-specific information
    """
    if method == "mean_distance":
        # Original method: average distance to all others
        rank_labels, sorted_indices = _rank_clusters_by_distance(None, None, matrix)
        return rank_labels, {"method": method, "sorted_indices": sorted_indices.tolist()}
    
    elif method.startswith("centroid"):
        # Centroid-based ranking in projection space
        rank_labels, distances, centroid = _rank_by_centroid_distance(coords)
        return rank_labels, {
            "method": method,
            "centroid": centroid.tolist(),
            "distances": distances.tolist()
        }
    
    elif method == "medoid":
        # Medoid-based ranking (actual data point as center)
        rank_labels, distances, medoid_idx = _rank_by_medoid_distance(matrix)
        return rank_labels, {
            "method": method,
            "medoid_index": int(medoid_idx),
            "distances": distances.tolist()
        }
    
    elif method == "lof":
        # Local Outlier Factor ranking
        rank_labels, lof_scores = _rank_by_lof(coords)
        return rank_labels, {
            "method": method,
            "lof_scores": lof_scores.tolist()
        }
    
    else:
        raise ValueError(f"Unknown ranking method: {method}")


def cluster_kmeans(coords: np.ndarray, n_clusters: int = 3) -> np.ndarray:
    """K-Means clustering on projection coordinates."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return kmeans.fit_predict(coords)


def cluster_hierarchical(matrix: np.ndarray, n_clusters: int = 3) -> np.ndarray:
    """Hierarchical clustering on distance matrix."""
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage="average"
    )
    return clustering.fit_predict(matrix)


def cluster_dbscan(coords: np.ndarray, eps: float = None, min_samples: int = 3) -> np.ndarray:
    """DBSCAN clustering on projection coordinates."""
    if eps is None:
        from sklearn.neighbors import NearestNeighbors
        k = min_samples
        neigh = NearestNeighbors(n_neighbors=k)
        neigh.fit(coords)
        distances, _ = neigh.kneighbors(coords)
        k_distances = np.sort(distances[:, k-1])
        eps = np.percentile(k_distances, 85)  # Slightly lower for more clusters
        logger.info(f"DBSCAN auto eps: {eps:.4f}")
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    return dbscan.fit_predict(coords)


def cluster_spectral(matrix: np.ndarray, n_clusters: int = 3) -> np.ndarray:
    """Spectral clustering on distance matrix."""
    gamma = 1.0 / (2.0 * np.median(matrix[matrix > 0]) ** 2)
    affinity = np.exp(-gamma * matrix ** 2)
    np.fill_diagonal(affinity, 1.0)
    
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        random_state=42
    )
    return spectral.fit_predict(affinity)


def plot_projection_ranked(
    coords: np.ndarray,
    subjects: list[str],
    rank_labels: np.ndarray,
    metric: str,
    proj_method: str,
    cluster_method: str,
    outdir: Path,
    show_labels: bool = True
) -> None:
    """Plot projection with High/Middle/Low coloring (fixed 29 each).
    
    Parameters
    ----------
    coords : np.ndarray
        2D coordinates.
    subjects : list[str]
        Subject IDs.
    rank_labels : np.ndarray
        Array of rank names for each subject ("High", "Middle", "Low", "Other").
    metric : str
        Distance metric name.
    proj_method : str
        Projection method name.
    cluster_method : str
        Clustering method name.
    outdir : Path
        Output directory.
    show_labels : bool
        Whether to show subject IDs.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Count subjects in each rank
    rank_counts = {"High": 0, "Middle": 0, "Low": 0, "Other": 0}
    
    # Plot each rank group
    for rank_name in ["High", "Middle", "Low", "Other"]:
        # Find all subjects in this rank
        mask = rank_labels == rank_name
        if not mask.any():
            continue
        
        rank_counts[rank_name] = mask.sum()
        color = RANK_COLORS.get(rank_name, "#000000")
        marker = "o"
        
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=color,
            marker=marker,
            s=50,
            label=f"{rank_name} (n={mask.sum()})",
            alpha=0.7
        )
        
        if show_labels:
            for idx in np.where(mask)[0]:
                ax.annotate(
                    subjects[idx],
                    (coords[idx, 0], coords[idx, 1]),
                    fontsize=5,
                    alpha=0.7,
                    color=color
                )
    
    ax.set_title(f"{metric.upper()} - {proj_method} + {cluster_method}\n"
                 f"(High={rank_counts['High']}, Middle={rank_counts['Middle']}, Low={rank_counts['Low']})")
    ax.set_xlabel(f"{proj_method} Dim 1")
    ax.set_ylabel(f"{proj_method} Dim 2")
    ax.legend(loc="best", fontsize=8)
    ax.set_aspect('equal', adjustable='datalim')
    
    fig.tight_layout()
    
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"{metric}_{proj_method.lower()}_{cluster_method.lower()}_ranked.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out_path}")


def plot_dendrogram_ranked(
    matrix: np.ndarray,
    subjects: list[str],
    rank_labels: np.ndarray,
    metric: str,
    outdir: Path
) -> None:
    """Plot dendrogram with High/Middle/Low color coding (fixed 29 each)."""
    from scipy.spatial.distance import squareform
    
    condensed = squareform(matrix)
    Z = linkage(condensed, method='average')
    
    # Create color mapping for leaves based on rank_labels
    leaf_colors = {}
    for i, subj in enumerate(subjects):
        rank = rank_labels[i]
        leaf_colors[subj] = RANK_COLORS.get(rank, "#000000")
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    dend = dendrogram(
        Z,
        labels=subjects,
        leaf_rotation=90,
        leaf_font_size=6,
        ax=ax,
        color_threshold=0,  # All black links
        above_threshold_color='gray'
    )
    
    # Color the labels
    xlabels = ax.get_xticklabels()
    for lbl in xlabels:
        subj = lbl.get_text()
        if subj in leaf_colors:
            lbl.set_color(leaf_colors[subj])
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=RANK_COLORS["High"], 
               markersize=10, label=f'High (n={GROUP_SIZE}, outliers)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=RANK_COLORS["Middle"], 
               markersize=10, label=f'Middle (n={GROUP_SIZE})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=RANK_COLORS["Low"], 
               markersize=10, label=f'Low (n={GROUP_SIZE}, central)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.set_title(f"{metric.upper()} - Hierarchical Clustering Dendrogram (High/Middle/Low, {GROUP_SIZE} each)")
    ax.set_xlabel("Subject")
    ax.set_ylabel("Distance")
    
    fig.tight_layout()
    
    out_path = outdir / f"{metric}_dendrogram_ranked.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out_path}")


def save_ranked_groups(
    subjects: list[str],
    rank_labels: np.ndarray,
    metric: str,
    cluster_method: str,
    outdir: Path
) -> Dict[str, List[str]]:
    """Save ranked groups to text files (fixed 29 subjects each).
    
    Returns
    -------
    Dict[str, List[str]]
        Mapping of rank name to list of subjects.
    """
    groups = {"High": [], "Middle": [], "Low": []}
    
    for i, subj in enumerate(subjects):
        rank = rank_labels[i]
        if rank in groups:
            groups[rank].append(subj)
    
    # Save to files
    for rank_name, members in groups.items():
        if members:
            out_path = outdir / f"{metric}_{cluster_method.lower()}_{rank_name.lower()}.txt"
            out_path.write_text("\n".join(members) + "\n")
            logger.info(f"Saved group: {out_path} ({len(members)} subjects)")
    
    return groups


def run_ranked_clustering_analysis(
    metric: str = "mmd",
    n_clusters: int = 3,
    output_subdir: str = "clustering_ranked",
    ranking_methods: List[str] = None
) -> dict:
    """Run clustering visualization with multiple ranking methods.
    
    Parameters
    ----------
    metric : str
        Distance metric (mmd, wasserstein, dtw).
    n_clusters : int
        Number of clusters for K-Means, Hierarchical, Spectral.
    output_subdir : str
        Subdirectory name for outputs.
    ranking_methods : List[str]
        List of ranking methods to use. Default: mean_distance, centroid_umap, lof
        Options: mean_distance, centroid_mds, centroid_umap, medoid, lof
    
    Returns
    -------
    dict
        Results with ranked groups for each method.
    """
    if ranking_methods is None:
        # Default: 3 selected cluster-aware methods
        ranking_methods = ["mean_distance", "centroid_umap", "lof"]
    
    # Load data
    base_dir = Path(cfg.RESULTS_DOMAIN_ANALYSIS_PATH) / "distance" / "subject-wise" / metric
    matrix_path = base_dir / f"{metric}_matrix.npy"
    subjects_path = base_dir / f"{metric}_subjects.json"
    
    if not matrix_path.exists() or not subjects_path.exists():
        raise FileNotFoundError(f"Files not found for {metric}")
    
    matrix = load_numpy(matrix_path)
    subjects = load_json(subjects_path)
    
    logger.info(f"Loaded {metric.upper()} matrix: {matrix.shape}, {len(subjects)} subjects")
    
    # Output directories
    png_dir = base_dir / "png" / output_subdir
    groups_dir = base_dir / "groups" / output_subdir
    png_dir.mkdir(parents=True, exist_ok=True)
    groups_dir.mkdir(parents=True, exist_ok=True)
    
    # Projection methods
    proj_methods = ["MDS", "tSNE"]
    if umap is not None:
        proj_methods.append("UMAP")
    
    # Compute projections
    projections = {}
    for pm in proj_methods:
        logger.info(f"Computing {pm} projection...")
        projections[pm] = _get_projection_coords(matrix, pm)
    
    results = {
        "metric": metric,
        "n_subjects": len(subjects),
        "group_size": GROUP_SIZE,
        "ranking_methods": {}
    }
    
    # Process each ranking method
    for rank_method in ranking_methods:
        logger.info(f"\n{'='*60}")
        logger.info(f"Ranking Method: {rank_method}")
        logger.info(f"{'='*60}")
        
        # Select appropriate coordinates for centroid-based methods
        if rank_method == "centroid_mds":
            coords_for_ranking = projections["MDS"]
        elif rank_method == "centroid_umap" and "UMAP" in projections:
            coords_for_ranking = projections["UMAP"]
        elif rank_method == "lof":
            # Use MDS coordinates for LOF
            coords_for_ranking = projections["MDS"]
        else:
            coords_for_ranking = projections["MDS"]  # Default
        
        # Compute ranking
        rank_labels, rank_info = _compute_ranking(matrix, coords_for_ranking, rank_method)
        
        # Count groups
        high_count = np.sum(rank_labels == "High")
        mid_count = np.sum(rank_labels == "Middle")
        low_count = np.sum(rank_labels == "Low")
        
        logger.info(f"  Groups: High={high_count}, Middle={mid_count}, Low={low_count}")
        
        # Log sample subjects
        high_subjects = [subjects[i] for i in range(len(subjects)) if rank_labels[i] == "High"][:5]
        low_subjects = [subjects[i] for i in range(len(subjects)) if rank_labels[i] == "Low"][:5]
        logger.info(f"  High (outliers, sample): {high_subjects}...")
        logger.info(f"  Low (central, sample): {low_subjects}...")
        
        # Save groups
        groups = save_ranked_groups(subjects, rank_labels, metric, rank_method, groups_dir)
        
        results["ranking_methods"][rank_method] = {
            "groups": {k: len(v) for k, v in groups.items()},
            "info": {k: v for k, v in rank_info.items() if k not in ["distances", "lof_scores", "sorted_indices"]}
        }
        
        # Create output subdirectory for this ranking method
        rank_png_dir = png_dir / rank_method
        rank_png_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate plots for all projection methods
        for pm, coords in projections.items():
            plot_projection_ranked(
                coords, subjects, rank_labels,
                metric, pm, rank_method, rank_png_dir
            )
        
        # Plot dendrogram for each ranking method
        plot_dendrogram_ranked(matrix, subjects, rank_labels, metric, rank_png_dir)
        
        # For centroid-based methods, plot with centroid marker
        if rank_method.startswith("centroid"):
            _plot_with_centroid(
                coords_for_ranking, subjects, rank_labels, rank_info,
                metric, rank_method, rank_png_dir
            )
    
    # Save summary
    summary_path = png_dir / f"{metric}_ranking_summary.json"
    save_json(results, summary_path)
    logger.info(f"Saved summary: {summary_path}")
    
    return results


def _plot_with_centroid(
    coords: np.ndarray,
    subjects: List[str],
    rank_labels: np.ndarray,
    rank_info: dict,
    metric: str,
    rank_method: str,
    outdir: Path
) -> None:
    """Plot projection with centroid marker highlighted."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot each rank group
    for rank_name in ["High", "Middle", "Low"]:
        mask = rank_labels == rank_name
        if not mask.any():
            continue
        
        color = RANK_COLORS.get(rank_name, "#000000")
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=color,
            s=50,
            label=f"{rank_name} (n={mask.sum()})",
            alpha=0.7
        )
        
        for idx in np.where(mask)[0]:
            ax.annotate(
                subjects[idx],
                (coords[idx, 0], coords[idx, 1]),
                fontsize=5,
                alpha=0.7,
                color=color
            )
    
    # Plot centroid
    if "centroid" in rank_info:
        centroid = rank_info["centroid"]
        ax.scatter(
            centroid[0], centroid[1],
            c="gold", s=300, marker="*", edgecolors="black", linewidths=2,
            label="Centroid", zorder=10
        )
    
    # Draw distance circles from centroid
    if "centroid" in rank_info and "distances" in rank_info:
        centroid = rank_info["centroid"]
        distances = np.array(rank_info["distances"])
        
        # Draw circles at 25%, 50%, 75% percentile distances
        for pct, ls in [(25, ':'), (50, '--'), (75, '-.')]:
            radius = np.percentile(distances, pct)
            circle = plt.Circle(centroid, radius, fill=False, color='gray', 
                              linestyle=ls, alpha=0.5, label=f'{pct}% distance')
            ax.add_patch(circle)
    
    ax.set_title(f"{metric.upper()} - {rank_method}\n"
                 f"Centroid-based ranking (Low=central, High=outlier)")
    ax.set_xlabel("Projection Dim 1")
    ax.set_ylabel("Projection Dim 2")
    ax.legend(loc="best", fontsize=8)
    ax.set_aspect('equal', adjustable='datalim')
    
    fig.tight_layout()
    
    out_path = outdir / f"{metric}_{rank_method}_centroid_view.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out_path}")


def main():
    """Run ranked clustering analysis for all metrics."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Clustering with High/Middle/Low ranking")
    parser.add_argument("--metric", choices=["mmd", "wasserstein", "dtw", "all"], default="all")
    parser.add_argument("--metrics", nargs="+", choices=["mmd", "wasserstein", "dtw"], 
                       help="Multiple metrics to process")
    parser.add_argument("--n_clusters", type=int, default=3)
    parser.add_argument("--output_subdir", default="clustering_ranked")
    parser.add_argument("--ranking", nargs="+", 
                       choices=["mean_distance", "centroid_mds", "centroid_umap", "medoid", "lof", "all"],
                       default=["all"],
                       help="Ranking methods to use. Options: mean_distance (平均距離), "
                            "centroid_mds (MDS空間のcentroid), centroid_umap (UMAP空間のcentroid), "
                            "medoid (medoidからの距離), lof (Local Outlier Factor)")
    args = parser.parse_args()
    
    # Determine metrics to process
    if args.metrics:
        metrics = args.metrics
    elif args.metric == "all":
        metrics = cfg.DISTANCE_METRICS
    else:
        metrics = [args.metric]
    
    # Determine ranking methods
    if "all" in args.ranking:
        ranking_methods = None  # Use default (all available)
    else:
        ranking_methods = args.ranking
    
    for metric in metrics:
        logger.info(f"\n{'#'*60}")
        logger.info(f"# Processing {metric.upper()}")
        logger.info(f"{'#'*60}")
        try:
            run_ranked_clustering_analysis(
                metric=metric,
                n_clusters=args.n_clusters,
                output_subdir=args.output_subdir,
                ranking_methods=ranking_methods
            )
        except Exception as e:
            logger.error(f"Failed for {metric}: {e}")
            raise


if __name__ == "__main__":
    main()
