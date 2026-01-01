#!/usr/bin/env python3
"""
Generate projection plots for new ranking methods.

This script generates MDS, t-SNE, and UMAP projection plots for the
3 new ranking methods: knn, median_distance, and isolation_forest.

Output:
    results/domain_analysis/distance/subject-wise/{metric}/png/clustering_ranked/{method}/
        - {metric}_mds_{method}_ranked.png
        - {metric}_tsne_{method}_ranked.png
        - {metric}_umap_{method}_ranked.png
        - {metric}_dendrogram_ranked.png
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.clustering_projection_ranked import run_ranked_clustering_analysis


def main():
    """Generate projection plots for new ranking methods."""
    
    # New ranking methods to generate plots for
    new_methods = ["knn", "median_distance", "isolation_forest"]
    
    # Distance metrics
    distance_metrics = ["mmd", "wasserstein", "dtw"]
    
    # Base directories
    base_input = PROJECT_ROOT / "results" / "domain_analysis" / "distance" / "subject-wise"
    
    print("=" * 70)
    print("Generating Projection Plots for New Ranking Methods")
    print("=" * 70)
    print(f"Methods: {new_methods}")
    print(f"Distance Metrics: {distance_metrics}")
    print("=" * 70)
    
    for metric in distance_metrics:
        print(f"\n{'='*60}")
        print(f"Processing: {metric.upper()}")
        print(f"{'='*60}")
        
        # Input/output directories
        metric_dir = base_input / metric
        matrix_path = metric_dir / f"{metric}_matrix.npy"
        subjects_path = metric_dir / f"{metric}_subjects.json"
        png_dir = metric_dir / "png" / "clustering_ranked"
        groups_dir = metric_dir / "groups" / "clustering_ranked"
        
        # Check if input files exist
        if not matrix_path.exists():
            print(f"[WARNING] Matrix file not found: {matrix_path}")
            continue
        if not subjects_path.exists():
            print(f"[WARNING] Subjects file not found: {subjects_path}")
            continue
        
        # Run analysis for new methods only
        try:
            # Use metric-based API (it internally constructs paths)
            results = run_ranked_clustering_analysis(
                metric=metric,
                ranking_methods=new_methods
            )
            
            print(f"\n[SUCCESS] Generated plots for {metric.upper()}")
            for method in new_methods:
                method_dir = png_dir / method
                if method_dir.exists():
                    files = list(method_dir.glob("*.png"))
                    print(f"  - {method}: {len(files)} plots")
                    
        except Exception as e:
            print(f"[ERROR] Failed to process {metric}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("Projection plot generation completed!")
    print("=" * 70)
    
    # Summary of generated files
    print("\nGenerated files:")
    for metric in distance_metrics:
        png_dir = base_input / metric / "png" / "clustering_ranked"
        for method in new_methods:
            method_dir = png_dir / method
            if method_dir.exists():
                files = list(method_dir.glob("*.png"))
                print(f"  {metric}/{method}: {len(files)} plots")
                for f in sorted(files):
                    print(f"    - {f.name}")


if __name__ == "__main__":
    main()
