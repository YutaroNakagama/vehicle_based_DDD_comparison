#!/usr/bin/env python3
"""
Test script to verify that resolve_source_group_subjects now correctly
extracts ranking method from tag and uses matching group file.
"""

import sys
from pathlib import Path
sys.path.insert(0, "/home/s2240011/git/ddd/vehicle_based_DDD_comparison")

from src.utils.io.target_resolution import (
    resolve_source_group_subjects,
    SOURCE_ONLY_TRAIN_GROUP,
)
from src import config as cfg


def read_subjects_directly(ranking_method: str, metric: str, level: str):
    """Read subjects directly from the group file."""
    ranks_dir = Path(cfg.RESULTS_DOMAIN_ANALYSIS_PATH) / "distance" / "subject-wise" / "ranks" / "ranks29" / ranking_method
    source_file = ranks_dir / f"{metric}_{level}.txt"
    
    if not source_file.exists():
        return None, str(source_file)
    
    with open(source_file) as f:
        subjects = [s.strip() for s in f if s.strip()]
    return subjects, str(source_file)


def test_source_group_resolution():
    """Test that source group resolution correctly extracts ranking method from tag."""
    
    test_cases = [
        # (tag, expected_ranking_method, expected_metric, expected_level)
        ("rank_knn_mmd_in_domain", "knn", "mmd", "in_domain"),
        ("full_mean_distance_dtw_in_domain", "mean_distance", "dtw", "in_domain"),
        ("rank_lof_wasserstein_in_domain", "lof", "wasserstein", "in_domain"),
        ("rank_median_distance_mmd_in_domain", "median_distance", "mmd", "in_domain"),
        ("rank_centroid_umap_dtw_in_domain", "centroid_umap", "dtw", "in_domain"),
        ("rank_isolation_forest_wasserstein_in_domain", "isolation_forest", "wasserstein", "in_domain"),
    ]
    
    print("=" * 80)
    print("Testing resolve_source_group_subjects with ranking method extraction")
    print(f"SOURCE_ONLY_TRAIN_GROUP = {SOURCE_ONLY_TRAIN_GROUP}")
    print("=" * 80)
    
    all_passed = True
    
    for tag, expected_method, expected_metric, expected_level in test_cases:
        print(f"\n--- Tag: {tag} ---")
        print(f"Expected ranking method: {expected_method}")
        print(f"Expected distance metric: {expected_metric}")
        
        try:
            # Get subjects from resolve_source_group_subjects (uses tag)
            resolved_subjects = resolve_source_group_subjects(tag)
            print(f"Resolved subjects count: {len(resolved_subjects)}")
            
            # Read expected subjects directly from file (using expected ranking method)
            expected_subjects, expected_file = read_subjects_directly(
                expected_method, expected_metric, SOURCE_ONLY_TRAIN_GROUP
            )
            
            if expected_subjects is None:
                print(f"✗ SKIP: Expected file not found: {expected_file}")
                continue
            
            print(f"Expected file: {expected_file}")
            print(f"Expected subjects count: {len(expected_subjects)}")
            
            # Check if resolved matches expected
            if set(resolved_subjects) == set(expected_subjects):
                print("✓ PASS: Resolved subjects match expected file!")
            else:
                print("✗ FAIL: Resolved subjects do NOT match expected file!")
                print(f"  Extra in resolved: {set(resolved_subjects) - set(expected_subjects)}")
                print(f"  Missing in resolved: {set(expected_subjects) - set(resolved_subjects)}")
                all_passed = False
                
        except Exception as e:
            print(f"✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    # Also verify that different ranking methods produce different results
    print("\n" + "=" * 80)
    print("Verifying that different ranking methods produce different groups")
    print("=" * 80)
    
    knn_subjects, _ = read_subjects_directly("knn", "mmd", "in_domain")
    mean_subjects, _ = read_subjects_directly("mean_distance", "mmd", "in_domain")
    
    if knn_subjects and mean_subjects:
        diff = set(knn_subjects) ^ set(mean_subjects)  # Symmetric difference
        print(f"knn_mmd_in_domain has {len(knn_subjects)} subjects")
        print(f"mean_distance_mmd_in_domain has {len(mean_subjects)} subjects")
        print(f"Differences between them: {len(diff)} subjects")
        if diff:
            print(f"Different subjects: {list(diff)[:10]}...")
            print("✓ As expected, different ranking methods produce different groups")
        else:
            print("Note: These two groups happen to be identical")
    
    print("\n" + "=" * 80)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 80)
    
    return all_passed


if __name__ == "__main__":
    success = test_source_group_resolution()
    sys.exit(0 if success else 1)
