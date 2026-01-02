#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared utilities for distance-based analysis scripts.

This module provides common functions used across distance analysis scripts:
- compute_intergroup_distances.py
- compute_umap_domain_distances.py  
- compute_tsne_umap_projections.py

Functions:
    load_distance_matrix: Load distance matrix from .npy file
    load_group_subjects: Load group subject list from text file
    load_all_subjects: Load all subjects from JSON file
    get_group_indices: Get indices of group members in all_subjects list
"""

import json
from pathlib import Path
from typing import List

import numpy as np

# =============================================================================
# Path Configuration
# =============================================================================
# Note: These can be overridden by the calling script if needed
BASE_DIR = Path(__file__).resolve().parents[2]
DISTANCE_DIR = BASE_DIR / "results" / "analysis" / "domain" / "distance"

# =============================================================================
# Constants
# =============================================================================
METRICS = ["dtw_mean", "mmd_mean", "wasserstein_mean"]
METRIC_DIRS = {
    "dtw_mean": "dtw",
    "mmd_mean": "mmd",
    "wasserstein_mean": "wasserstein",
}
LEVELS = ["out_domain", "mid_domain", "in_domain"]


# =============================================================================
# Data Loading Functions
# =============================================================================
def load_distance_matrix(
    metric: str,
    distance_dir: Path = None,
) -> np.ndarray:
    """Load distance matrix from .npy file.
    
    Parameters
    ----------
    metric : str
        Distance metric (dtw_mean, mmd_mean, wasserstein_mean)
    distance_dir : Path, optional
        Base distance directory. Defaults to DISTANCE_DIR.
        
    Returns
    -------
    np.ndarray
        Distance matrix (n_subjects x n_subjects)
    """
    if distance_dir is None:
        distance_dir = DISTANCE_DIR
        
    metric_dir = METRIC_DIRS[metric]
    matrix_path = distance_dir / "subject-wise" / metric_dir / f"{metric_dir}_matrix.npy"
    return np.load(matrix_path)


def load_group_subjects(
    metric: str,
    level: str,
    ranking_method: str = "mean_distance",
    distance_dir: Path = None,
) -> List[str]:
    """Load group subject list from text file.
    
    Parameters
    ----------
    metric : str
        Distance metric (dtw_mean, mmd_mean, wasserstein_mean)
    level : str
        Group level (out_domain, mid_domain, in_domain)
    ranking_method : str
        Ranking method (mean_distance, centroid_mds, centroid_umap, medoid, lof)
    distance_dir : Path, optional
        Base distance directory. Defaults to DISTANCE_DIR.
        
    Returns
    -------
    list
        List of subject IDs in the group
    """
    if distance_dir is None:
        distance_dir = DISTANCE_DIR
        
    # New folder structure: ranks29/{ranking_method}/{metric}_{level}.txt
    # Remove "_mean" suffix from metric
    metric_base = metric.replace("_mean", "")
    group_path = (
        distance_dir / "subject-wise" / "ranks" / "ranks29" 
        / ranking_method / f"{metric_base}_{level}.txt"
    )
    
    # Fallback: legacy format (mean_distance_legacy)
    if not group_path.exists():
        group_path = (
            distance_dir / "subject-wise" / "ranks" / "ranks29"
            / "mean_distance_legacy" / f"{metric}_{level}.txt"
        )
    
    with open(group_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def load_all_subjects(
    metric: str,
    distance_dir: Path = None,
) -> List[str]:
    """Load all subjects list from JSON file.
    
    Parameters
    ----------
    metric : str
        Distance metric (dtw_mean, mmd_mean, wasserstein_mean)
    distance_dir : Path, optional
        Base distance directory. Defaults to DISTANCE_DIR.
        
    Returns
    -------
    list
        List of all subject IDs
    """
    if distance_dir is None:
        distance_dir = DISTANCE_DIR
        
    metric_dir = METRIC_DIRS[metric]
    subject_path = distance_dir / "subject-wise" / metric_dir / f"{metric_dir}_subjects.json"
    with open(subject_path, "r") as f:
        return json.load(f)


def get_group_indices(
    all_subjects: List[str],
    group_subjects: List[str],
    verbose: bool = True,
) -> np.ndarray:
    """Get indices of group members in the all_subjects list.
    
    Parameters
    ----------
    all_subjects : list
        List of all subject IDs
    group_subjects : list
        List of subject IDs in the group
    verbose : bool
        Whether to print warnings for missing subjects
        
    Returns
    -------
    np.ndarray
        Array of indices
    """
    indices = []
    for subj in group_subjects:
        try:
            idx = all_subjects.index(subj)
            indices.append(idx)
        except ValueError:
            if verbose:
                print(f"Warning: Subject {subj} not found in all_subjects")
    return np.array(indices)


def load_all_group_indices(
    metric: str,
    ranking_method: str = "mean_distance",
    distance_dir: Path = None,
) -> dict:
    """Load indices for all domain levels at once.
    
    Parameters
    ----------
    metric : str
        Distance metric (dtw_mean, mmd_mean, wasserstein_mean)
    ranking_method : str
        Ranking method (mean_distance, centroid_mds, etc.)
    distance_dir : Path, optional
        Base distance directory
        
    Returns
    -------
    dict
        Dictionary mapping level names to index arrays
    """
    all_subjects = load_all_subjects(metric, distance_dir)
    
    group_indices = {}
    for level in LEVELS:
        group_subjects = load_group_subjects(metric, level, ranking_method, distance_dir)
        group_indices[level] = get_group_indices(all_subjects, group_subjects, verbose=False)
    
    return group_indices


# =============================================================================
# Distance Computation Functions
# =============================================================================
def compute_intergroup_distances(
    dist_matrix: np.ndarray,
    indices_A: np.ndarray,
    indices_B: np.ndarray
) -> dict:
    """Compute distance statistics between two groups.
    
    Parameters
    ----------
    dist_matrix : np.ndarray
        Precomputed distance matrix
    indices_A : np.ndarray
        Indices of group A
    indices_B : np.ndarray
        Indices of group B
        
    Returns
    -------
    dict
        Distance statistics (mean, std, min, max, median)
    """
    inter_distances = dist_matrix[np.ix_(indices_A, indices_B)]
    return {
        "mean": float(np.mean(inter_distances)),
        "std": float(np.std(inter_distances)),
        "min": float(np.min(inter_distances)),
        "max": float(np.max(inter_distances)),
        "median": float(np.median(inter_distances))
    }


def compute_intragroup_distances(
    dist_matrix: np.ndarray,
    indices: np.ndarray
) -> dict:
    """Compute within-group distance statistics.
    
    Parameters
    ----------
    dist_matrix : np.ndarray
        Precomputed distance matrix
    indices : np.ndarray
        Indices of group members
        
    Returns
    -------
    dict
        Distance statistics (mean, std, min, max, median)
    """
    intra_distances = dist_matrix[np.ix_(indices, indices)]
    mask = ~np.eye(len(indices), dtype=bool)
    intra_distances = intra_distances[mask]
    return {
        "mean": float(np.mean(intra_distances)),
        "std": float(np.std(intra_distances)),
        "min": float(np.min(intra_distances)),
        "max": float(np.max(intra_distances)),
        "median": float(np.median(intra_distances))
    }


# =============================================================================
# Embedding Functions
# =============================================================================
def compute_embedding(
    dist_matrix: np.ndarray,
    method: str = "mds",
    n_components: int = 2,
    random_state: int = 42
) -> tuple:
    """Compute dimensionality reduction embedding.
    
    Parameters
    ----------
    dist_matrix : np.ndarray
        Precomputed distance matrix
    method : str
        One of "mds", "tsne", "umap"
    n_components : int
        Number of output dimensions
    random_state : int
        Random seed for reproducibility
    
    Returns
    -------
    coords : np.ndarray
        Embedded coordinates (n_samples, n_components)
    meta : dict
        Metadata about the embedding (stress, kl_divergence, etc.)
    """
    from sklearn.manifold import MDS, TSNE
    
    method = method.lower()
    meta = {"method": method, "n_components": n_components}
    
    if method == "mds":
        print(f"  Running MDS embedding...")
        reducer = MDS(
            n_components=n_components,
            dissimilarity="precomputed",
            random_state=random_state,
            n_init=4,
            max_iter=300
        )
        coords = reducer.fit_transform(dist_matrix)
        meta["stress"] = float(reducer.stress_)
        print(f"  ✓ MDS completed (stress={reducer.stress_:.4f})")
        
    elif method == "tsne":
        print(f"  Running t-SNE embedding...")
        reducer = TSNE(
            n_components=n_components,
            metric="precomputed",
            init="random",
            random_state=random_state,
            perplexity=min(30, len(dist_matrix) - 1),
            max_iter=1000
        )
        coords = reducer.fit_transform(dist_matrix)
        meta["kl_divergence"] = float(reducer.kl_divergence_)
        print(f"  ✓ t-SNE completed (KL={reducer.kl_divergence_:.4f})")
        
    elif method == "umap":
        try:
            import umap
        except ImportError:
            raise ImportError("UMAP not available. Install with: pip install umap-learn")
        print(f"  Running UMAP embedding...")
        reducer = umap.UMAP(
            n_components=n_components,
            metric="precomputed",
            random_state=random_state,
            n_neighbors=min(15, len(dist_matrix) - 1)
        )
        coords = reducer.fit_transform(dist_matrix)
        print(f"  ✓ UMAP completed")
    else:
        raise ValueError(f"Unknown method: {method}. Use 'mds', 'tsne', or 'umap'")
    
    return coords, meta


def compute_centroids(
    coords: np.ndarray,
    group_indices_dict: dict
) -> dict:
    """Compute group centroids from embedded coordinates.
    
    Parameters
    ----------
    coords : np.ndarray
        Embedded coordinates (n_samples, n_components)
    group_indices_dict : dict
        Dictionary mapping level names to index arrays
        
    Returns
    -------
    dict
        Centroid information including coordinates, spread, and distances
    """
    global_centroid = np.mean(coords, axis=0)
    
    centroids = {}
    for level, indices in group_indices_dict.items():
        group_coords = coords[indices]
        centroid = np.mean(group_coords, axis=0)
        
        # Distance from group centroid to global centroid
        dist_to_global = float(np.linalg.norm(centroid - global_centroid))
        
        # Spread: mean distance of group members to group centroid
        spread = float(np.mean(np.linalg.norm(group_coords - centroid, axis=1)))
        
        centroids[level] = {
            "coordinates": centroid.tolist(),
            "spread": spread,
            "distance_to_global": dist_to_global,
            "n_samples": len(indices)
        }
    
    # Compute pairwise centroid distances
    centroid_distances = {}
    for i, level1 in enumerate(LEVELS):
        for level2 in LEVELS[i + 1:]:
            if level1 in centroids and level2 in centroids:
                c1 = np.array(centroids[level1]["coordinates"])
                c2 = np.array(centroids[level2]["coordinates"])
                dist = float(np.linalg.norm(c1 - c2))
                centroid_distances[f"{level1}_to_{level2}"] = dist
    
    return {
        "global_centroid": global_centroid.tolist(),
        "group_centroids": centroids,
        "centroid_distances": centroid_distances
    }