#!/usr/bin/env python3
"""
Distance computation and visualization from domain center using MDS/t-SNE/UMAP

Computations:
1. Mean distance from each group's domain center to each subject in that group
2. Distance from each group's domain center to the Middle group's domain center
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import MDS, TSNE

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
OUTPUT_DIR = DISTANCE_DIR / "group-wise" / "intergroup_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_projection_with_domain_distances(dist_matrix: np.ndarray, 
                                              group_indices_dict: dict,
                                              method: str = "umap",
                                              n_components: int = 2,
                                              random_state: int = 42) -> dict:
    """Compute distances from domain center via dimensionality reduction
    
    Parameters
    ----------
    method : str
        "mds", "tsne", or "umap"
    
    Returns
    -------
    dict
        - coords: Projected coordinates
        - group_centroids: Centroid coordinates and metrics for each group
        - intra_domain_distances: Mean distance from subjects to group centroid within each group
        - inter_domain_distances: Distance from each group centroid to the Middle centroid
    """
    method = method.lower()
    
    if method == "mds":
        print("  Running MDS embedding...")
        reducer = MDS(
            n_components=n_components,
            dissimilarity="precomputed",
            random_state=random_state,
            n_init=4,
            max_iter=300
        )
        coords = reducer.fit_transform(dist_matrix)
        print("  ✓ MDS embedding completed")
        
    elif method == "tsne":
        print("  Running t-SNE embedding...")
        reducer = TSNE(
            n_components=n_components,
            metric="precomputed",
            init="random",
            random_state=random_state,
            perplexity=min(30, len(dist_matrix) - 1),
            n_iter=1000
        )
        coords = reducer.fit_transform(dist_matrix)
        print("  ✓ t-SNE embedding completed")
        
    elif method == "umap":
        if not UMAP_AVAILABLE:
            raise ImportError("UMAP not available. Install with: pip install umap-learn")
        print("  Running UMAP embedding...")
        reducer = umap.UMAP(
            n_components=n_components,
            metric="precomputed",
            random_state=random_state,
            n_neighbors=min(15, len(dist_matrix) - 1),
            verbose=False
        )
        coords = reducer.fit_transform(dist_matrix)
        print("  ✓ UMAP embedding completed")
        
    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'mds', 'tsne', or 'umap'")
    
    # Compute centroid (domain center) of each group
    group_centroids = {}
    for level, indices in group_indices_dict.items():
        group_coords = coords[indices]
        centroid = np.mean(group_coords, axis=0)
        group_centroids[level] = {
            "coordinates": centroid,
            "indices": indices,
            "coords": group_coords
        }
    
    # 1. Intra-group distance: mean distance from group domain center to subjects
    intra_domain_distances = {}
    for level, data in group_centroids.items():
        centroid = data["coordinates"]
        group_coords = data["coords"]
        
        # Distance from each subject to group centroid
        distances = np.linalg.norm(group_coords - centroid, axis=1)
        
        intra_domain_distances[level] = {
            "mean": float(np.mean(distances)),
            "std": float(np.std(distances)),
            "min": float(np.min(distances)),
            "max": float(np.max(distances)),
            "distances": distances.tolist()
        }
    
    # 2. Inter-group distance: distance from each group domain center to mid_domain center
    middle_centroid = group_centroids["mid_domain"]["coordinates"]
    
    inter_domain_distances = {}
    for level in LEVELS:
        level_centroid = group_centroids[level]["coordinates"]
        distance = float(np.linalg.norm(level_centroid - middle_centroid))
        inter_domain_distances[level] = distance
    
    return {
        "coords": coords,
        "group_centroids": {
            level: {
                "coordinates": data["coordinates"].tolist(),
                "n_subjects": len(data["indices"])
            }
            for level, data in group_centroids.items()
        },
        "intra_domain_distances": intra_domain_distances,
        "inter_domain_distances": inter_domain_distances
    }


def visualize_projection_with_distances(metric: str,
                                         method: str,
                                         projection_results: dict, 
                                         group_indices_dict: dict):
    """Visualize dimensionality reduction projection and distance metrics"""
    coords = projection_results["coords"]
    group_centroids = projection_results["group_centroids"]
    intra_distances = projection_results["intra_domain_distances"]
    inter_distances = projection_results["inter_domain_distances"]
    
    method_upper = method.upper()
    
    fig = plt.figure(figsize=(18, 12))
    
    # Layout: projection plot on top, distance metrics on bottom
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], hspace=0.3, wspace=0.3)
    
    # === Top: Projection plot ===
    ax_proj = fig.add_subplot(gs[0, :])
    
    colors = DOMAIN_LEVEL_COLORS
    markers = {"out_domain": "^", "mid_domain": "s", "in_domain": "v"}
    
    # Plot subjects of each group
    for level, indices in group_indices_dict.items():
        group_coords = coords[indices]
        ax_proj.scatter(
            group_coords[:, 0], group_coords[:, 1],
            c=colors[level], marker=markers[level],
            s=100, alpha=0.6, 
            label=f"{level.capitalize()} subjects (n={len(indices)})",
            edgecolors='black', linewidth=0.5
        )
    
    # Plot domain center of each group
    for level, centroid_data in group_centroids.items():
        c = np.array(centroid_data["coordinates"])
        ax_proj.scatter(
            c[0], c[1], 
            c=colors[level], marker='*',
            s=1200, edgecolors='black', linewidth=3,
            label=f"{level.capitalize()} domain center", 
            zorder=10
        )
        
        # Label with group name
        ax_proj.text(
            c[0], c[1] - 0.5, level.upper(),
            fontsize=12, ha='center', fontweight='bold',
            color=colors[level]
        )
    
    # Draw lines from mid_domain centroid to other centroids
    middle_centroid = np.array(group_centroids["mid_domain"]["coordinates"])
    for level in ["out_domain", "in_domain"]:
        level_centroid = np.array(group_centroids[level]["coordinates"])
        ax_proj.plot(
            [middle_centroid[0], level_centroid[0]],
            [middle_centroid[1], level_centroid[1]],
            'k--', alpha=0.4, linewidth=2
        )
        
        # Display distance as label
        mid_x = (middle_centroid[0] + level_centroid[0]) / 2
        mid_y = (middle_centroid[1] + level_centroid[1]) / 2
        dist = inter_distances[level]
        ax_proj.text(
            mid_x, mid_y, f"{dist:.2f}",
            fontsize=11, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', 
                     facecolor='yellow', alpha=0.7, edgecolor='black')
        )
    
    ax_proj.set_xlabel(f"{method_upper} Component 1", fontsize=14)
    ax_proj.set_ylabel(f"{method_upper} Component 2", fontsize=14)
    ax_proj.set_title(
        f"{method_upper} Projection with Domain Centers - {metric.upper()}\n"
        f"Distance from each domain center to subjects and Middle domain center",
        fontsize=16, fontweight='bold'
    )
    ax_proj.legend(loc='best', fontsize=10, framealpha=0.9, ncol=2)
    ax_proj.grid(True, alpha=0.3)
    
    # === Bottom left: Intra-group distance (mean distance from domain center to subjects) ===
    ax_intra = fig.add_subplot(gs[1, 0])
    
    levels_list = list(LEVELS)
    intra_means = [intra_distances[level]["mean"] for level in levels_list]
    intra_stds = [intra_distances[level]["std"] for level in levels_list]
    
    bars = ax_intra.bar(
        range(len(levels_list)), intra_means, yerr=intra_stds,
        color=[colors[level] for level in levels_list],
        edgecolor='black', linewidth=2, capsize=10, alpha=0.7
    )
    
    # Display values above bars
    for i, (level, mean, std) in enumerate(zip(levels_list, intra_means, intra_stds)):
        ax_intra.text(
            i, mean + std + 0.05 * max(intra_means), 
            f"{mean:.3f}\n±{std:.3f}",
            ha='center', va='bottom', fontsize=11, fontweight='bold'
        )
    
    ax_intra.set_xticks(range(len(levels_list)))
    ax_intra.set_xticklabels([l.capitalize() for l in levels_list], fontsize=12)
    ax_intra.set_ylabel("Average Distance", fontsize=13)
    ax_intra.set_title(
        "Intra-Domain Distance\n"
        "(Domain center → Subjects in same group)",
        fontsize=13, fontweight='bold'
    )
    ax_intra.grid(axis='y', alpha=0.3)
    
    # === Bottom right: Inter-group distance (distance from each domain center to Middle center) ===
    ax_inter = fig.add_subplot(gs[1, 1])
    
    inter_values = [inter_distances[level] for level in levels_list]
    
    bars = ax_inter.bar(
        range(len(levels_list)), inter_values,
        color=[colors[level] for level in levels_list],
        edgecolor='black', linewidth=2, alpha=0.7
    )
    
    # Display values above bars
    for i, (level, value) in enumerate(zip(levels_list, inter_values)):
        ax_inter.text(
            i, value + 0.05 * max(inter_values), 
            f"{value:.3f}",
            ha='center', va='bottom', fontsize=11, fontweight='bold'
        )
    
    ax_inter.set_xticks(range(len(levels_list)))
    ax_inter.set_xticklabels([l.capitalize() for l in levels_list], fontsize=12)
    ax_inter.set_ylabel("Distance to Middle Center", fontsize=13)
    ax_inter.set_title(
        "Inter-Domain Distance\n"
        "(Domain center → Middle domain center)",
        fontsize=13, fontweight='bold'
    )
    ax_inter.grid(axis='y', alpha=0.3)
    
    # Middle itself is 0, so mark it specially
    ax_inter.text(
        1, inter_values[1] + 0.02, "Self",
        ha='center', va='bottom', fontsize=10, 
        style='italic', color='gray'
    )
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / f"{metric}_{method}_domain_distances.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def create_summary_table(all_results: dict, method: str):
    """Create a summary table of all distance metrics"""
    
    # Prepare data for DataFrame
    data = []
    
    for metric in METRICS:
        results = all_results[metric]
        intra = results["intra_domain_distances"]
        inter = results["inter_domain_distances"]
        
        for level in LEVELS:
            row = {
                "Metric": metric.replace("_mean", "").upper(),
                "Group": level.capitalize(),
                "Intra_Mean": intra[level]["mean"],
                "Intra_Std": intra[level]["std"],
                "Inter_to_Middle": inter[level]
            }
            data.append(row)
    
    df = pd.DataFrame(data)
    df = df.round(4)
    
    # Save as CSV
    csv_path = OUTPUT_DIR / f"{method}_domain_distances_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved summary CSV: {csv_path}")
    
    return df


def visualize_comparison_across_metrics(all_results: dict, method: str):
    """Visualization comparing three distance metrics"""
    
    method_upper = method.upper()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Domain Distance Comparison Across Metrics ({method_upper})", 
                 fontsize=18, fontweight='bold')
    
    colors = DOMAIN_LEVEL_COLORS
    
    for col_idx, metric in enumerate(METRICS):
        results = all_results[metric]
        intra = results["intra_domain_distances"]
        inter = results["inter_domain_distances"]
        
        metric_name = metric.replace("_mean", "").upper()
        
        # Top: Intra-domain distances
        ax_intra = axes[0, col_idx]
        levels_list = list(LEVELS)
        intra_means = [intra[level]["mean"] for level in levels_list]
        intra_stds = [intra[level]["std"] for level in levels_list]
        
        ax_intra.bar(
            range(len(levels_list)), intra_means, yerr=intra_stds,
            color=[colors[level] for level in levels_list],
            edgecolor='black', linewidth=1.5, capsize=8, alpha=0.7
        )
        
        for i, (mean, std) in enumerate(zip(intra_means, intra_stds)):
            ax_intra.text(
                i, mean + std * 1.1, f"{mean:.3f}",
                ha='center', va='bottom', fontsize=10, fontweight='bold'
            )
        
        ax_intra.set_xticks(range(len(levels_list)))
        ax_intra.set_xticklabels([l.capitalize() for l in levels_list])
        ax_intra.set_title(f"{metric_name}\nIntra-Domain", fontweight='bold')
        ax_intra.set_ylabel("Distance" if col_idx == 0 else "")
        ax_intra.grid(axis='y', alpha=0.3)
        
        # Bottom: Inter-domain distances (to Middle)
        ax_inter = axes[1, col_idx]
        inter_values = [inter[level] for level in levels_list]
        
        ax_inter.bar(
            range(len(levels_list)), inter_values,
            color=[colors[level] for level in levels_list],
            edgecolor='black', linewidth=1.5, alpha=0.7
        )
        
        for i, value in enumerate(inter_values):
            ax_inter.text(
                i, value * 1.05, f"{value:.3f}",
                ha='center', va='bottom', fontsize=10, fontweight='bold'
            )
        
        ax_inter.set_xticks(range(len(levels_list)))
        ax_inter.set_xticklabels([l.capitalize() for l in levels_list])
        ax_inter.set_title(f"Inter-Domain (to Middle)", fontweight='bold')
        ax_inter.set_ylabel("Distance" if col_idx == 0 else "")
        ax_inter.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / f"{method}_domain_distances_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved comparison: {output_path}")


def main():
    print("=" * 80)
    print("MDS / t-SNE / UMAP domain distance computation")
    print("=" * 80)
    print()
    print("Computations:")
    print("  1. Intra-domain: mean distance from group domain center to each subject")
    print("  2. Inter-domain: distance from each group domain center to Middle domain center")
    print()
    
    # Check available methods
    available_methods = ["mds", "tsne"]
    if UMAP_AVAILABLE:
        available_methods.append("umap")
    else:
        print("Note: UMAP not available. Install with: pip install umap-learn")
    
    print(f"Available methods: {', '.join(available_methods).upper()}")
    print()
    
    all_results = {}
    
    for method in available_methods:
        method_results = {}
        
        print(f"\n{'='*80}")
        print(f"METHOD: {method.upper()}")
        print(f"{'='*80}")
        
        for metric in METRICS:
            print(f"\n{'─'*60}")
            print(f"Processing: {metric.upper()} with {method.upper()}")
            print(f"{'─'*60}")
            
            # Load data
            print("  Loading data...")
            dist_matrix = load_distance_matrix(metric)
            all_subjects = load_all_subjects(metric)
            
            # Get group indices
            group_indices = {}
            for level in LEVELS:
                subjects = load_group_subjects(metric, level)
                group_indices[level] = get_group_indices(all_subjects, subjects)
                print(f"    {level.capitalize()}: {len(group_indices[level])} subjects")
            
            print(f"  Matrix size: {dist_matrix.shape}")
            
            # Dimensionality reduction computation
            projection_results = compute_projection_with_domain_distances(
                dist_matrix, group_indices, method=method
            )
            
            # Display results
            print("\n  Results:")
            print("  ───────────────────────────────────────")
            print("  Intra-domain distances (mean ± std):")
            for level in LEVELS:
                mean = projection_results["intra_domain_distances"][level]["mean"]
                std = projection_results["intra_domain_distances"][level]["std"]
                print(f"    {level.capitalize():8s}: {mean:.4f} ± {std:.4f}")
            
            print("\n  Inter-domain distances (to Middle center):")
            for level in LEVELS:
                dist = projection_results["inter_domain_distances"][level]
                print(f"    {level.capitalize():8s}: {dist:.4f}")
            
            # Visualization
            print("\n  Generating visualization...")
            visualize_projection_with_distances(metric, method, projection_results, group_indices)
            
            # Save results
            save_results = {
                "method": method,
                "metric": metric,
                "group_centroids": projection_results["group_centroids"],
                "intra_domain_distances": projection_results["intra_domain_distances"],
                "inter_domain_distances": projection_results["inter_domain_distances"]
            }
            
            output_json = OUTPUT_DIR / f"{metric}_{method}_domain_distances.json"
            with open(output_json, "w") as f:
                json.dump(save_results, f, indent=2)
            print(f"  Saved JSON: {output_json}")
            
            method_results[metric] = projection_results
        
        all_results[method] = method_results
    
    # Create summary table (all methods)
    print(f"\n{'='*80}")
    print("Creating summary tables...")
    print(f"{'='*80}")
    
    for method in available_methods:
        print(f"\n{method.upper()} Summary:")
        df = create_summary_table(all_results[method], method)
        print("\n" + df.to_string(index=False))
    
    # Comparison visualization (per method)
    print(f"\n{'='*80}")
    print("Creating comparison visualizations...")
    print(f"{'='*80}")
    
    for method in available_methods:
        print(f"\n  Creating comparison for {method.upper()}...")
        visualize_comparison_across_metrics(all_results[method], method)
    
    print("\n" + "="*80)
    print("✓ Domain distance computation complete for all methods")
    print("="*80)
    print()
    print("Generated files:")
    for method in available_methods:
        print(f"\n  {method.upper()}:")
        print(f"    - {{metric}}_{method}_domain_distances.png  : Detailed visualization per metric")
        print(f"    - {{metric}}_{method}_domain_distances.json : Numerical data")
        print(f"    - {method}_domain_distances_summary.csv   : Summary of all metrics")
        print(f"    - {method}_domain_distances_comparison.png: Comparison of 3 metrics")
    print()
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
