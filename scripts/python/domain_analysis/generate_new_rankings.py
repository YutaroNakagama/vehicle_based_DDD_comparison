#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_new_rankings.py
========================
Generate ranking files for new ranking methods (knn, median_distance, isolation_forest).

This script generates the out_domain/mid_domain/in_domain group files for each new ranking method,
which can then be used in the training pipeline.
"""

import logging
from pathlib import Path

import numpy as np

from src import config as cfg
from src.utils.io.data_io import load_numpy, load_json
from src.analysis.clustering_projection_ranked import (
    _rank_by_knn,
    _rank_by_median_distance,
    _rank_by_isolation_forest,
    GROUP_SIZE,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# New ranking methods to generate
NEW_RANKING_METHODS = ["knn", "median_distance", "isolation_forest"]


def generate_rankings_for_metric(metric: str, output_subdir: str = "clustering_ranked") -> dict:
    """Generate ranking files for new methods for a specific distance metric.
    
    Parameters
    ----------
    metric : str
        Distance metric name (mmd, wasserstein, dtw).
    output_subdir : str
        Subdirectory name for output files.
    
    Returns
    -------
    dict
        Summary of generated files.
    """
    # Load distance matrix and subjects
    base_dir = Path(cfg.RESULTS_DOMAIN_ANALYSIS_PATH) / "distance" / "subject-wise" / metric
    matrix_path = base_dir / f"{metric}_matrix.npy"
    subjects_path = base_dir / f"{metric}_subjects.json"
    
    if not matrix_path.exists() or not subjects_path.exists():
        logger.warning(f"Files not found for {metric}, skipping")
        return {}
    
    matrix = load_numpy(matrix_path)
    subjects = load_json(subjects_path)
    
    logger.info(f"Loaded {metric.upper()} matrix: {matrix.shape}, {len(subjects)} subjects")
    
    # Output directory
    groups_dir = base_dir / "groups" / output_subdir
    groups_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for method in NEW_RANKING_METHODS:
        logger.info(f"  Generating {method} rankings...")
        
        # Compute ranking
        if method == "knn":
            rank_labels, scores = _rank_by_knn(matrix, group_size=GROUP_SIZE, k=5)
        elif method == "median_distance":
            rank_labels, scores = _rank_by_median_distance(matrix, group_size=GROUP_SIZE)
        elif method == "isolation_forest":
            rank_labels, scores = _rank_by_isolation_forest(matrix, group_size=GROUP_SIZE)
        else:
            logger.warning(f"Unknown method: {method}")
            continue
        
        # Extract groups
        groups = {"out_domain": [], "mid_domain": [], "in_domain": []}
        for i, subj in enumerate(subjects):
            rank = rank_labels[i]
            if rank in groups:
                groups[rank].append(subj)
        
        # Save to files
        for rank_name, members in groups.items():
            if members:
                out_path = groups_dir / f"{metric}_{method}_{rank_name.lower()}.txt"
                out_path.write_text("\n".join(members) + "\n")
                logger.info(f"    Saved: {out_path.name} ({len(members)} subjects)")
        
        results[method] = {rank: len(members) for rank, members in groups.items()}
    
    return results


def main():
    """Generate ranking files for all metrics and new methods."""
    logger.info("=" * 60)
    logger.info("Generating new ranking files")
    logger.info("=" * 60)
    logger.info(f"Methods: {NEW_RANKING_METHODS}")
    logger.info(f"Group size: {GROUP_SIZE}")
    logger.info("")
    
    metrics = cfg.DISTANCE_METRICS  # mmd, wasserstein, dtw
    
    all_results = {}
    for metric in metrics:
        logger.info(f"\nProcessing {metric.upper()}...")
        results = generate_rankings_for_metric(metric)
        all_results[metric] = results
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    
    for metric, results in all_results.items():
        logger.info(f"\n{metric.upper()}:")
        for method, counts in results.items():
            logger.info(f"  {method}: High={counts.get('out_domain', 0)}, Middle={counts.get('mid_domain', 0)}, Low={counts.get('in_domain', 0)}")
    
    logger.info("\n✅ Done! New ranking files generated.")
    logger.info("Next steps:")
    logger.info("  1. Run training with new rankings")
    logger.info("  2. Run evaluation")
    logger.info("  3. Collect metrics with collect_evaluation_metrics_ranked.py")
    
    return 0


if __name__ == "__main__":
    main()
