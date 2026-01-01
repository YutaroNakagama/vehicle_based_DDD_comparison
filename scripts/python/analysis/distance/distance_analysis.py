#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
distance_analysis.py
====================

Unified tool for distance-based domain analysis.

This script provides:
1. intergroup  - Compute inter/intra-group distances and statistics
2. projections - Compute MDS/t-SNE/UMAP projections and centroids
3. domain      - Compute domain center distances for each group
4. all         - Run full analysis pipeline

Consolidates functionality from:
- compute_intergroup_distances.py
- compute_tsne_umap_projections.py
- compute_umap_domain_distances.py

Usage:
    python distance_analysis.py intergroup --metric dtw
    python distance_analysis.py projections --method umap --metric dtw
    python distance_analysis.py domain --method mds --metric dtw
    python distance_analysis.py all --metric dtw
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import MDS, TSNE

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# Import shared utilities
from distance_utils import (
    load_distance_matrix,
    load_group_subjects,
    load_all_subjects,
    get_group_indices,
    METRICS,
    METRIC_DIRS,
    LEVELS,
    DISTANCE_DIR,
)

# Add project root for src imports
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from src.utils.visualization.color_palettes import DOMAIN_LEVEL_COLORS

# Output directory
OUTPUT_DIR = DISTANCE_DIR / "group-wise" / "intergroup_analysis"


# ============================================================
# Core Computation Functions
# ============================================================
def compute_intergroup_distances(dist_matrix: np.ndarray,
                                  indices_A: np.ndarray,
                                  indices_B: np.ndarray) -> dict:
    """Compute distance statistics between two groups."""
    inter_distances = dist_matrix[np.ix_(indices_A, indices_B)]
    return {
        "mean": float(np.mean(inter_distances)),
        "std": float(np.std(inter_distances)),
        "min": float(np.min(inter_distances)),
        "max": float(np.max(inter_distances)),
        "median": float(np.median(inter_distances))
    }


def compute_intragroup_distances(dist_matrix: np.ndarray,
                                  indices: np.ndarray) -> dict:
    """Compute within-group distance statistics."""
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


def compute_embedding(dist_matrix: np.ndarray,
                       method: str = "mds",
                       n_components: int = 2,
                       random_state: int = 42) -> Tuple[np.ndarray, dict]:
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
        if not UMAP_AVAILABLE:
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


def compute_centroids(coords: np.ndarray,
                       group_indices_dict: Dict[str, np.ndarray]) -> dict:
    """Compute group centroids from embedded coordinates."""
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


# ============================================================
# Visualization Functions
# ============================================================
def plot_projection(coords: np.ndarray,
                     group_indices_dict: Dict[str, np.ndarray],
                     centroids: dict,
                     title: str,
                     output_path: Path) -> None:
    """Create scatter plot of projection with group colors and centroids."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot each group
    for level, indices in group_indices_dict.items():
        color = DOMAIN_LEVEL_COLORS.get(level, "gray")
        ax.scatter(
            coords[indices, 0], coords[indices, 1],
            c=color, label=level, alpha=0.6, s=50, edgecolors="white"
        )
    
    # Plot centroids
    for level, data in centroids["group_centroids"].items():
        c = data["coordinates"]
        color = DOMAIN_LEVEL_COLORS.get(level, "gray")
        ax.scatter(c[0], c[1], c=color, s=200, marker="*", edgecolors="black", linewidth=2)
    
    # Plot global centroid
    gc = centroids["global_centroid"]
    ax.scatter(gc[0], gc[1], c="black", s=300, marker="X", label="Global Center")
    
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_intergroup_heatmap(results: dict, metric: str, output_path: Path) -> None:
    """Create heatmap of inter-group distances."""
    # Build matrix
    levels = LEVELS
    n = len(levels)
    matrix = np.zeros((n, n))
    
    for i, l1 in enumerate(levels):
        for j, l2 in enumerate(levels):
            if i == j:
                matrix[i, j] = results["intragroup"][l1]["mean"]
            else:
                key = f"{l1}_to_{l2}" if f"{l1}_to_{l2}" in results["intergroup"] else f"{l2}_to_{l1}"
                matrix[i, j] = results["intergroup"].get(key, {}).get("mean", 0)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="YlOrRd",
                xticklabels=levels, yticklabels=levels, ax=ax)
    ax.set_title(f"Inter/Intra-group Distances ({metric})")
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ============================================================
# CLI Commands
# ============================================================
def cmd_intergroup(args) -> int:
    """Compute inter/intra-group distance statistics."""
    print("=" * 60)
    print(f"Computing inter-group distances for metric: {args.metric}")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    dist_matrix = load_distance_matrix(args.metric)
    all_subjects = load_all_subjects()
    group_indices = {
        level: get_group_indices(load_group_subjects(level), all_subjects)
        for level in LEVELS
    }
    
    results = {
        "metric": args.metric,
        "intergroup": {},
        "intragroup": {}
    }
    
    # Compute intra-group distances
    print("\nIntra-group distances:")
    for level in LEVELS:
        stats = compute_intragroup_distances(dist_matrix, group_indices[level])
        results["intragroup"][level] = stats
        print(f"  {level}: mean={stats['mean']:.4f} ± {stats['std']:.4f}")
    
    # Compute inter-group distances
    print("\nInter-group distances:")
    for i, l1 in enumerate(LEVELS):
        for l2 in LEVELS[i + 1:]:
            stats = compute_intergroup_distances(dist_matrix, group_indices[l1], group_indices[l2])
            key = f"{l1}_to_{l2}"
            results["intergroup"][key] = stats
            print(f"  {l1} ↔ {l2}: mean={stats['mean']:.4f} ± {stats['std']:.4f}")
    
    # Save results
    out_file = OUTPUT_DIR / f"intergroup_stats_{args.metric}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_file}")
    
    # Plot heatmap
    plot_intergroup_heatmap(results, args.metric, OUTPUT_DIR / f"intergroup_heatmap_{args.metric}.png")
    
    return 0


def cmd_projections(args) -> int:
    """Compute projections and centroids."""
    print("=" * 60)
    print(f"Computing {args.method.upper()} projection for metric: {args.metric}")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    dist_matrix = load_distance_matrix(args.metric)
    all_subjects = load_all_subjects()
    group_indices = {
        level: get_group_indices(load_group_subjects(level), all_subjects)
        for level in LEVELS
    }
    
    # Compute embedding
    coords, meta = compute_embedding(dist_matrix, method=args.method, random_state=args.seed)
    
    # Compute centroids
    centroids = compute_centroids(coords, group_indices)
    
    results = {
        "metric": args.metric,
        "embedding": meta,
        **centroids
    }
    
    # Save results
    out_file = OUTPUT_DIR / f"projection_{args.method}_{args.metric}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_file}")
    
    # Plot
    plot_projection(
        coords, group_indices, centroids,
        title=f"{args.method.upper()} Projection ({args.metric})",
        output_path=OUTPUT_DIR / f"projection_{args.method}_{args.metric}.png"
    )
    
    return 0


def cmd_domain(args) -> int:
    """Compute domain center distances."""
    print("=" * 60)
    print(f"Computing domain distances ({args.method.upper()}) for metric: {args.metric}")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    dist_matrix = load_distance_matrix(args.metric)
    all_subjects = load_all_subjects()
    group_indices = {
        level: get_group_indices(load_group_subjects(level), all_subjects)
        for level in LEVELS
    }
    
    # Compute embedding
    coords, meta = compute_embedding(dist_matrix, method=args.method, random_state=args.seed)
    
    # Compute centroids
    centroids = compute_centroids(coords, group_indices)
    
    # Compute domain-specific metrics
    print("\nDomain center distances:")
    for level, data in centroids["group_centroids"].items():
        print(f"  {level}:")
        print(f"    Spread (mean dist to centroid): {data['spread']:.4f}")
        print(f"    Distance to global center: {data['distance_to_global']:.4f}")
    
    print("\nCentroid-to-centroid distances:")
    for pair, dist in centroids["centroid_distances"].items():
        print(f"  {pair}: {dist:.4f}")
    
    results = {
        "metric": args.metric,
        "method": args.method,
        "embedding": meta,
        **centroids
    }
    
    out_file = OUTPUT_DIR / f"domain_distances_{args.method}_{args.metric}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_file}")
    
    return 0


def cmd_all(args) -> int:
    """Run full analysis pipeline."""
    print("=" * 60)
    print(f"Running full distance analysis for metric: {args.metric}")
    print("=" * 60)
    
    # Run intergroup analysis
    print("\n--- Inter-group Analysis ---")
    cmd_intergroup(args)
    
    # Run projections for all methods
    for method in ["mds", "tsne", "umap"] if UMAP_AVAILABLE else ["mds", "tsne"]:
        print(f"\n--- {method.upper()} Projection ---")
        args.method = method
        cmd_projections(args)
    
    print("\n" + "=" * 60)
    print("✓ Full analysis completed")
    print("=" * 60)
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Unified distance-based domain analysis tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python distance_analysis.py intergroup --metric dtw
    python distance_analysis.py projections --method umap --metric dtw
    python distance_analysis.py domain --method mds --metric dtw
    python distance_analysis.py all --metric dtw
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Common arguments
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--metric", default="dtw", choices=METRICS,
                        help="Distance metric to use")
    common.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # intergroup
    p_inter = subparsers.add_parser("intergroup", parents=[common],
                                     help="Compute inter/intra-group distance statistics")
    p_inter.set_defaults(func=cmd_intergroup)
    
    # projections
    p_proj = subparsers.add_parser("projections", parents=[common],
                                    help="Compute MDS/t-SNE/UMAP projections")
    p_proj.add_argument("--method", default="mds", choices=["mds", "tsne", "umap"],
                        help="Dimensionality reduction method")
    p_proj.set_defaults(func=cmd_projections)
    
    # domain
    p_domain = subparsers.add_parser("domain", parents=[common],
                                      help="Compute domain center distances")
    p_domain.add_argument("--method", default="mds", choices=["mds", "tsne", "umap"],
                          help="Dimensionality reduction method")
    p_domain.set_defaults(func=cmd_domain)
    
    # all
    p_all = subparsers.add_parser("all", parents=[common],
                                   help="Run full analysis pipeline")
    p_all.set_defaults(func=cmd_all)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
