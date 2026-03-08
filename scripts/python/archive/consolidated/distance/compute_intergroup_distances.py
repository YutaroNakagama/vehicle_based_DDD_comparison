#!/usr/bin/env python3
"""
Script to compute inter-group distances and group centroids

Computations:
1. Inter-group distances
   - Mean distance and standard deviation for High ↔ Middle, High ↔ Low, Middle ↔ Low
2. Group centroids
   - Centroid coordinates of each group in MDS projection space
   - Euclidean distances between centroids
3. Intra/Inter ratio computation
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.manifold import MDS, TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import json
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not available. Install with: pip install umap-learn")

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
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from src.utils.visualization.color_palettes import DOMAIN_LEVEL_COLORS

# Path settings
BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = DISTANCE_DIR / "group-wise" / "intergroup_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_intergroup_distances(dist_matrix: np.ndarray, 
                                  indices_A: np.ndarray, 
                                  indices_B: np.ndarray) -> dict:
    """Compute distance statistics between two groups"""
    # Extract distances between each element of group A and group B
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
    """Compute intra-group distance statistics"""
    intra_distances = dist_matrix[np.ix_(indices, indices)]
    # Exclude diagonal elements (self-distance=0)
    mask = ~np.eye(len(indices), dtype=bool)
    intra_distances = intra_distances[mask]
    
    return {
        "mean": float(np.mean(intra_distances)),
        "std": float(np.std(intra_distances)),
        "min": float(np.min(intra_distances)),
        "max": float(np.max(intra_distances)),
        "median": float(np.median(intra_distances))
    }


def compute_mds_centroids(dist_matrix: np.ndarray, 
                          group_indices_dict: dict,
                          n_components: int = 2,
                          random_state: int = 42) -> dict:
    """Compute group centroids via MDS projection"""
    # MDS projection
    mds = MDS(n_components=n_components, 
              dissimilarity="precomputed", 
              random_state=random_state,
              n_init=10,
              max_iter=1000)
    coords = mds.fit_transform(dist_matrix)
    
    # Compute centroid of each group
    centroids = {}
    for level, indices in group_indices_dict.items():
        group_coords = coords[indices]
        centroid = np.mean(group_coords, axis=0)
        centroids[level] = {
            "coordinates": centroid.tolist(),
            "spread": float(np.mean(np.linalg.norm(group_coords - centroid, axis=1)))
        }
    
    # Compute distances between centroids
    centroid_distances = {}
    for i, level1 in enumerate(LEVELS):
        for level2 in LEVELS[i+1:]:
            c1 = np.array(centroids[level1]["coordinates"])
            c2 = np.array(centroids[level2]["coordinates"])
            dist = float(np.linalg.norm(c1 - c2))
            centroid_distances[f"{level1}_vs_{level2}"] = dist
    
    return {
        "centroids": centroids,
        "centroid_distances": centroid_distances,
        "mds_coords": coords,
        "stress": float(mds.stress_)
    }


def compute_projection_centroids(dist_matrix: np.ndarray, 
                                   group_indices_dict: dict,
                                   method: str = "mds",
                                   n_components: int = 2,
                                   random_state: int = 42) -> dict:
    """Compute group centroids using a specified dimensionality reduction method
    
    Parameters
    ----------
    dist_matrix : np.ndarray
        Distance matrix
    group_indices_dict : dict
        Dictionary of group name -> index list
    method : str
        Projection method ("mds", "tsne", "umap")
    n_components : int
        Number of dimensions
    random_state : int
        Random seed
    
    Returns
    -------
    dict
        Centroid information, coordinates, and metrics
    """
    # Perform dimensionality reduction
    if method == "mds":
        reducer = MDS(n_components=n_components, 
                     dissimilarity="precomputed", 
                     random_state=random_state,
                     n_init=10,
                     max_iter=1000)
        coords = reducer.fit_transform(dist_matrix)
        metric_value = float(reducer.stress_)
        metric_name = "stress"
    
    elif method == "tsne":
        reducer = TSNE(n_components=n_components,
                      metric="precomputed",
                      init="random",  # Use random init for precomputed
                      random_state=random_state,
                      perplexity=min(30, len(dist_matrix) - 1),
                      max_iter=1000)
        coords = reducer.fit_transform(dist_matrix)
        metric_value = float(reducer.kl_divergence_)
        metric_name = "kl_divergence"
    
    elif method == "umap":
        if not UMAP_AVAILABLE:
            raise ImportError("UMAP is not available. Install with: pip install umap-learn")
        reducer = umap.UMAP(n_components=n_components,
                           metric="precomputed",
                           random_state=random_state,
                           n_neighbors=min(15, len(dist_matrix) - 1))
        coords = reducer.fit_transform(dist_matrix)
        metric_value = None  # UMAP has no single metric
        metric_name = "none"
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Compute overall centroid (domain center)
    global_centroid = np.mean(coords, axis=0)
    
    # Compute centroid of each group
    centroids = {}
    for level, indices in group_indices_dict.items():
        group_coords = coords[indices]
        centroid = np.mean(group_coords, axis=0)
        
        # Distance from group centroid to domain center
        distance_to_global = float(np.linalg.norm(centroid - global_centroid))
        
        # Spread within the group
        spread = float(np.mean(np.linalg.norm(group_coords - centroid, axis=1)))
        
        centroids[level] = {
            "coordinates": centroid.tolist(),
            "spread": spread,
            "distance_to_global_centroid": distance_to_global
        }
    
    # Compute distances between centroids
    centroid_distances = {}
    for i, level1 in enumerate(LEVELS):
        for level2 in LEVELS[i+1:]:
            c1 = np.array(centroids[level1]["coordinates"])
            c2 = np.array(centroids[level2]["coordinates"])
            dist = float(np.linalg.norm(c1 - c2))
            centroid_distances[f"{level1}_vs_{level2}"] = dist
    
    return {
        "method": method,
        "global_centroid": global_centroid.tolist(),
        "centroids": centroids,
        "centroid_distances": centroid_distances,
        "coords": coords,
        "metric_name": metric_name,
        "metric_value": metric_value
    }


def visualize_projection_with_centroids(metric: str, 
                                         projection_results: dict, 
                                         group_indices_dict: dict,
                                         output_suffix: str = ""):
    """Visualize projection results with centroids (common for MDS/t-SNE/UMAP)
    
    Parameters
    ----------
    metric : str
        Distance metric name
    projection_results : dict
        Projection results (output of compute_projection_centroids)
    group_indices_dict : dict
        Group index dictionary
    output_suffix : str
        Output filename suffix
    """
    coords = projection_results["coords"]
    centroids = projection_results["centroids"]
    global_centroid = np.array(projection_results["global_centroid"])
    method = projection_results["method"]
    
    fig, ax = plt.subplots(figsize=(14, 11))
    
    # Plot each group
    colors = DOMAIN_LEVEL_COLORS
    markers = {"out_domain": "^", "mid_domain": "s", "in_domain": "v"}
    
    for level, indices in group_indices_dict.items():
        group_coords = coords[indices]
        ax.scatter(group_coords[:, 0], group_coords[:, 1],
                  c=colors[level], marker=markers[level],
                  s=100, alpha=0.6, label=f"{level.capitalize()} group",
                  edgecolors='black', linewidth=0.5)
    
    # Plot domain center (overall centroid)
    ax.scatter(global_centroid[0], global_centroid[1],
              c='black', marker='X', s=1000, 
              edgecolors='white', linewidth=3,
              label='Global centroid (Domain center)', zorder=15)
    
    # Plot centroid of each group
    for level, centroid_data in centroids.items():
        c = np.array(centroid_data["coordinates"])
        ax.scatter(c[0], c[1], 
                  c=colors[level], marker='*',
                  s=800, edgecolors='black', linewidth=2,
                  label=f"{level.capitalize()} centroid", zorder=10)
        
        # Draw line from domain center to group centroid
        ax.plot([global_centroid[0], c[0]], [global_centroid[1], c[1]],
               c=colors[level], linestyle='-', alpha=0.5, linewidth=2.5)
        
        # Display distance as label
        dist_to_center = centroid_data["distance_to_global_centroid"]
        mid_x = (global_centroid[0] + c[0]) / 2
        mid_y = (global_centroid[1] + c[1]) / 2
        ax.text(mid_x, mid_y, f"{dist_to_center:.2f}",
               fontsize=10, ha='center', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.4', 
                        facecolor=colors[level], alpha=0.3, edgecolor='black'))
    
    # Axis labels and title
    method_upper = method.upper()
    ax.set_xlabel(f"{method_upper} Component 1", fontsize=14)
    ax.set_ylabel(f"{method_upper} Component 2", fontsize=14)
    
    title = f"{method_upper} Projection with Centroids - {metric.upper()}"
    if projection_results["metric_value"] is not None:
        title += f"\n{projection_results['metric_name']}: {projection_results['metric_value']:.4f}"
    ax.set_title(title, fontsize=16)
    
    ax.legend(loc='best', fontsize=9, framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_name = f"{metric}_{method}_centroids{output_suffix}.png"
    output_path = OUTPUT_DIR / output_name
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def visualize_centroids(metric: str, mds_results: dict, group_indices_dict: dict):
    """Visualize MDS projection with centroids (kept for backward compatibility)"""
    # Use the new function
    projection_results = {
        "coords": mds_results["mds_coords"],
        "centroids": mds_results["centroids"],
        "global_centroid": np.mean(mds_results["mds_coords"], axis=0),
        "method": "mds",
        "metric_name": "stress",
        "metric_value": mds_results["stress"]
    }
    # If global_centroid is not in centroids, compute additionally
    for level in projection_results["centroids"]:
        if "distance_to_global_centroid" not in projection_results["centroids"][level]:
            c = np.array(projection_results["centroids"][level]["coordinates"])
            gc = projection_results["global_centroid"]
            projection_results["centroids"][level]["distance_to_global_centroid"] = float(np.linalg.norm(c - gc))
    
    visualize_projection_with_centroids(metric, projection_results, group_indices_dict)
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / f"{metric}_mds_centroids.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def visualize_distance_heatmap(metric: str, results: dict):
    """Create heatmap of intra/inter distances"""
    # Create distance matrix
    distance_matrix = np.zeros((3, 3))
    labels = ["out_domain", "mid_domain", "in_domain"]
    
    # Set intra distances on diagonal
    for i, level in enumerate(LEVELS):
        distance_matrix[i, i] = results["intragroup"][level]["mean"]
    
    # Set inter distances on off-diagonal
    for pair, stats in results["intergroup"].items():
        level1, level2 = pair.split("_vs_")
        i = LEVELS.index(level1)
        j = LEVELS.index(level2)
        distance_matrix[i, j] = stats["mean"]
        distance_matrix[j, i] = stats["mean"]  # Symmetric matrix
    
    # Draw heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(distance_matrix, 
                annot=True, fmt='.4f', 
                xticklabels=labels, yticklabels=labels,
                cmap='YlOrRd', square=True, linewidths=1,
                cbar_kws={'label': 'Distance'},
                ax=ax)
    
    ax.set_title(f"Intra/Inter-group Distance Matrix - {metric.upper()}", 
                fontsize=16, pad=20)
    ax.set_xlabel("Group", fontsize=14)
    ax.set_ylabel("Group", fontsize=14)
    
    # Highlight diagonal (intra) elements
    for i in range(3):
        rect = plt.Rectangle((i, i), 1, 1, fill=False, 
                            edgecolor='blue', linewidth=3)
        ax.add_patch(rect)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / f"{metric}_distance_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def visualize_intra_inter_comparison(all_results: dict):
    """Intra/Inter comparison across all distance metrics"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, metric in enumerate(METRICS):
        ax = axes[idx]
        results = all_results[metric]
        
        # Prepare data
        categories = []
        means = []
        stds = []
        types = []  # Intra or Inter
        
        # Intra distances
        for level in LEVELS:
            categories.append(f"{level.capitalize()}\n(Intra)")
            means.append(results["intragroup"][level]["mean"])
            stds.append(results["intragroup"][level]["std"])
            types.append("Intra")
        
        # Inter distances
        for pair in ["out_domain_vs_mid_domain", "out_domain_vs_in_domain", "mid_domain_vs_in_domain"]:
            level1, level2 = pair.split("_vs_")
            categories.append(f"{level1.capitalize()}\nvs\n{level2.capitalize()}")
            means.append(results["intergroup"][pair]["mean"])
            stds.append(results["intergroup"][pair]["std"])
            types.append("Inter")
        
        # Bar chart
        colors = ['lightblue' if t == 'Intra' else 'lightcoral' for t in types]
        x = np.arange(len(categories))
        bars = ax.bar(x, means, yerr=stds, capsize=5, 
                     color=colors, edgecolor='black', linewidth=1.5)
        
        # Distinguish Intra/Inter
        for i, (bar, t) in enumerate(zip(bars, types)):
            if t == "Intra":
                bar.set_hatch('//')
        
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_ylabel("Distance", fontsize=12)
        ax.set_title(f"{metric.upper()}", fontsize=14)
        ax.grid(axis='y', alpha=0.3)
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightblue', edgecolor='black', hatch='//', label='Intra-group'),
            Patch(facecolor='lightcoral', edgecolor='black', label='Inter-group')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
    
    plt.suptitle("Intra-group vs Inter-group Distance Comparison", 
                fontsize=16, y=1.02)
    plt.tight_layout()
    output_path = OUTPUT_DIR / "intra_inter_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def compute_intra_inter_ratios(results: dict) -> dict:
    """Compute intra/inter ratio"""
    ratios = {}
    
    for level in LEVELS:
        intra_mean = results["intragroup"][level]["mean"]
        
        # Mean inter distance with other groups
        inter_means = []
        for pair, stats in results["intergroup"].items():
            if level in pair:
                inter_means.append(stats["mean"])
        
        avg_inter = np.mean(inter_means)
        ratio = intra_mean / avg_inter if avg_inter > 0 else np.nan
        
        ratios[level] = {
            "intra_mean": intra_mean,
            "avg_inter_mean": avg_inter,
            "intra_inter_ratio": ratio
        }
    
    return ratios


def main():
    print("=" * 80)
    print("Computing inter-group distances and group centroids")
    print("=" * 80)
    print()
    
    all_results = {}
    
    for metric in METRICS:
        print(f"\n{'='*60}")
        print(f"Processing: {metric.upper()}")
        print(f"{'='*60}")
        
        # Load data
        print("  Loading data...")
        dist_matrix = load_distance_matrix(metric)
        all_subjects = load_all_subjects(metric)
        
        # Get group indices
        group_indices = {}
        group_subjects = {}
        for level in LEVELS:
            subjects = load_group_subjects(metric, level)
            group_subjects[level] = subjects
            group_indices[level] = get_group_indices(all_subjects, subjects)
            print(f"    {level.capitalize()}: {len(group_indices[level])} subjects")
        
        results = {
            "metric": metric,
            "intragroup": {},
            "intergroup": {},
            "mds_analysis": {},
            "tsne_analysis": {},
            "umap_analysis": {},
            "intra_inter_ratios": {}
        }
        
        # 1. Intra-group distances
        print("\n  Computing intra-group distances...")
        for level in LEVELS:
            intra = compute_intragroup_distances(dist_matrix, group_indices[level])
            results["intragroup"][level] = intra
            print(f"    {level.capitalize()}: mean={intra['mean']:.4f}, std={intra['std']:.4f}")
        
        # 2. Inter-group distances
        print("\n  Computing inter-group distances...")
        for i, level1 in enumerate(LEVELS):
            for level2 in LEVELS[i+1:]:
                inter = compute_intergroup_distances(
                    dist_matrix, 
                    group_indices[level1], 
                    group_indices[level2]
                )
                pair_key = f"{level1}_vs_{level2}"
                results["intergroup"][pair_key] = inter
                print(f"    {level1.capitalize()} vs {level2.capitalize()}: "
                      f"mean={inter['mean']:.4f}, std={inter['std']:.4f}")
        
        # 3. MDS centroid analysis
        print("\n  Computing MDS centroids...")
        mds_results = compute_projection_centroids(dist_matrix, group_indices, method="mds")
        results["mds_analysis"] = mds_results
        print(f"    MDS stress: {mds_results['metric_value']:.4f}")
        for level, centroid_data in mds_results["centroids"].items():
            dist_to_center = centroid_data["distance_to_global_centroid"]
            print(f"    {level.capitalize()}: spread={centroid_data['spread']:.4f}, "
                  f"dist_to_center={dist_to_center:.4f}")
        
        # 4. t-SNE centroid analysis
        print("\n  Computing t-SNE centroids...")
        tsne_results = compute_projection_centroids(dist_matrix, group_indices, method="tsne")
        results["tsne_analysis"] = tsne_results
        print(f"    t-SNE KL divergence: {tsne_results['metric_value']:.4f}")
        for level, centroid_data in tsne_results["centroids"].items():
            dist_to_center = centroid_data["distance_to_global_centroid"]
            print(f"    {level.capitalize()}: spread={centroid_data['spread']:.4f}, "
                  f"dist_to_center={dist_to_center:.4f}")
        
        # 5. UMAP centroid analysis
        if UMAP_AVAILABLE:
            print("\n  Computing UMAP centroids...")
            umap_results = compute_projection_centroids(dist_matrix, group_indices, method="umap")
            results["umap_analysis"] = umap_results
            print(f"    UMAP embedding computed")
            for level, centroid_data in umap_results["centroids"].items():
                dist_to_center = centroid_data["distance_to_global_centroid"]
                print(f"    {level.capitalize()}: spread={centroid_data['spread']:.4f}, "
                      f"dist_to_center={dist_to_center:.4f}")
        else:
            print("\n  Skipping UMAP (not available)")
        
        # 6. Intra/Inter ratio
        print("\n  Computing intra/inter ratios...")
        ratios = compute_intra_inter_ratios(results)
        results["intra_inter_ratios"] = ratios
        for level, ratio_data in ratios.items():
            print(f"    {level.capitalize()}: ratio={ratio_data['intra_inter_ratio']:.4f}")
        
        # Save results (excluding coords arrays)
        save_results = {k: v for k, v in results.items() 
                       if k not in ["mds_analysis", "tsne_analysis", "umap_analysis"]}
        
        # Save projection results without coordinates
        for analysis_key, analysis_data in [
            ("mds_analysis", mds_results),
            ("tsne_analysis", tsne_results if "tsne_results" in locals() else None),
            ("umap_analysis", umap_results if UMAP_AVAILABLE and "umap_results" in locals() else None)
        ]:
            if analysis_data is not None:
                save_results[analysis_key] = {
                    "method": analysis_data["method"],
                    "global_centroid": analysis_data["global_centroid"],
                    "centroids": analysis_data["centroids"],
                    "centroid_distances": analysis_data["centroid_distances"],
                    "metric_name": analysis_data["metric_name"],
                    "metric_value": analysis_data["metric_value"]
                }
        
        output_json = OUTPUT_DIR / f"{metric}_intergroup_analysis.json"
        with open(output_json, "w") as f:
            json.dump(save_results, f, indent=2)
        print(f"\n  Saved: {output_json}")
        
        # Visualization
        print("\n  Generating visualizations...")
        visualize_projection_with_centroids(metric, mds_results, group_indices)
        visualize_projection_with_centroids(metric, tsne_results, group_indices)
        if UMAP_AVAILABLE:
            visualize_projection_with_centroids(metric, umap_results, group_indices)
        visualize_distance_heatmap(metric, results)
        
        all_results[metric] = results
    
    # Overall comparison visualization
    print(f"\n{'='*60}")
    print("Generating comparison visualizations...")
    print(f"{'='*60}")
    visualize_intra_inter_comparison(all_results)
    
    # Create summary table
    print("\n" + "="*80)
    print("Summary: Intra-group vs Inter-group Distances")
    print("="*80)
    
    summary_data = []
    for metric in METRICS:
        results = all_results[metric]
        for level in LEVELS:
            row = {
                "metric": metric,
                "group": level,
                "intra_mean": results["intragroup"][level]["mean"],
                "intra_std": results["intragroup"][level]["std"],
            }
            # Add mean inter distance
            inter_means = []
            for pair, stats in results["intergroup"].items():
                if level in pair:
                    inter_means.append(stats["mean"])
            row["avg_inter_mean"] = np.mean(inter_means)
            row["intra_inter_ratio"] = results["intra_inter_ratios"][level]["intra_inter_ratio"]
            summary_data.append(row)
    
    df_summary = pd.DataFrame(summary_data)
    df_summary = df_summary.round(4)
    
    print("\n" + df_summary.to_string(index=False))
    
    csv_path = OUTPUT_DIR / "intra_inter_summary.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"\nSaved summary: {csv_path}")
    
    print("\n" + "="*80)
    print("✓ Analysis complete!")
    print(f"✓ Results saved to: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
