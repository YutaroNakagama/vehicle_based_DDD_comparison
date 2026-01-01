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
DISTANCE_DIR = BASE_DIR / "results" / "domain_analysis" / "distance"

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
