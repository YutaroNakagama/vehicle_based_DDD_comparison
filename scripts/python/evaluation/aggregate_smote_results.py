#!/usr/bin/env python3
"""
Aggregate SMOTE Comparison Experiment Results

Collects training results from multiple jobs and creates a unified DataFrame
for further analysis and visualization.

Usage:
    python scripts/python/evaluation/aggregate_smote_results.py \
        [--output results/analysis/imbalance/smote_comparison/aggregated_results.csv]

Output directories:
    - Pooled mode (imbalance-only): results/analysis/imbalance/smote_comparison/
    - Domain mode (with ranking):   results/analysis/domain/imbalance/smote_comparison/
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Setup path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RESULTS_OUTPUTS_TRAINING_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def parse_tag(tag: str) -> Dict[str, str]:
    """Extract experiment metadata from tag.
    
    Expected tag formats:
    - sw_smote_pooled_seed42
    - smote_rank_knn_out_domain_seed42
    - smote_brf_lof_in_domain_seed42
    - imbal_v2_smote_seed42 (pooled simple smote)
    - imbal_v2_smote_subjectwise_seed42 (pooled subject-wise smote)
    - rank_knn_mmd_out_domain_sw_smote (ranking subject-wise smote)
    - rank_lof_mmd_in_domain_smote (ranking simple smote)
    """
    metadata = {
        "smote_method": "unknown",
        "mode": "unknown",
        "ranking": "none",
        "domain_level": "none",
        "seed": "unknown",
    }
    
    if not tag:
        return metadata
    
    tag_lower = tag.lower()
    
    # Extract seed
    seed_match = re.search(r"seed(\d+)", tag)
    if seed_match:
        metadata["seed"] = seed_match.group(1)
    
    # Determine SMOTE method (order matters - check specific patterns first)
    if "sw_smote" in tag_lower or "subjectwise" in tag_lower or "_sw_smote" in tag_lower:
        metadata["smote_method"] = "subject_wise_smote"
    elif "smote_brf" in tag_lower or "brf" in tag_lower:
        metadata["smote_method"] = "smote_balanced_rf"
    elif "smote" in tag_lower:
        # Default to simple smote if smote is mentioned but not subject-wise or brf
        metadata["smote_method"] = "simple_smote"
    
    # Determine ranking method (check before mode to avoid false positives)
    if "knn" in tag_lower or "rank_knn" in tag_lower:
        metadata["ranking"] = "knn"
    elif "lof" in tag_lower or "rank_lof" in tag_lower:
        metadata["ranking"] = "lof"
    
    # Determine domain level
    if "out_domain" in tag_lower:
        metadata["domain_level"] = "out_domain"
    elif "in_domain" in tag_lower:
        metadata["domain_level"] = "in_domain"
    
    # Determine mode (use result's mode field primarily, but infer from tag if needed)
    if "pooled" in tag_lower or "imbal_v2" in tag_lower:
        metadata["mode"] = "pooled"
    elif "source" in tag_lower:
        metadata["mode"] = "source_only"
    elif "target" in tag_lower:
        metadata["mode"] = "target_only"
    elif metadata["ranking"] != "none":
        # If ranking is specified, it's likely source/target mode
        metadata["mode"] = "ranking_based"
    
    return metadata


def load_single_result(json_path: Path) -> Optional[Dict]:
    """Load a single training result JSON file."""
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        logging.warning(f"Failed to load {json_path}: {e}")
        return None


def extract_metrics(result: Dict) -> Dict[str, float]:
    """Extract key metrics from a training result."""
    metrics = {}
    
    # Common metric keys
    metric_keys = [
        "train_accuracy", "train_f1", "train_precision", "train_recall", "train_auc",
        "val_accuracy", "val_f1", "val_precision", "val_recall", "val_auc",
        "test_accuracy", "test_f1", "test_precision", "test_recall", "test_auc",
        "test_specificity", "test_sensitivity", "test_balanced_accuracy",
    ]
    
    for key in metric_keys:
        if key in result:
            metrics[key] = result[key]
    
    # Handle nested metrics (e.g., {"train": {"f1": 0.8}})
    for split in ["train", "val", "test"]:
        if split in result and isinstance(result[split], dict):
            for metric_name, value in result[split].items():
                if isinstance(value, (int, float)):
                    metrics[f"{split}_{metric_name}"] = value
    
    return metrics


def collect_all_results(
    base_dir: str = None,
    model_names: List[str] = None,
    job_ids: List[str] = None,
) -> pd.DataFrame:
    """Collect all training results from the outputs directory.
    
    Parameters
    ----------
    base_dir : str, optional
        Base directory for training results. Defaults to RESULTS_OUTPUTS_TRAINING_PATH.
    model_names : list of str, optional
        Filter by model names. Defaults to ["RF", "BalancedRF"].
    job_ids : list of str, optional
        Filter by specific job IDs. If None, collect all.
    
    Returns
    -------
    pd.DataFrame
        Aggregated results with experiment metadata.
    """
    if base_dir is None:
        base_dir = RESULTS_OUTPUTS_TRAINING_PATH
    
    if model_names is None:
        model_names = ["RF", "BalancedRF"]
    
    all_records = []
    
    for model_name in model_names:
        model_dir = Path(base_dir) / model_name
        if not model_dir.exists():
            logging.warning(f"Model directory not found: {model_dir}")
            continue
        
        # Iterate over job directories
        for job_dir in model_dir.iterdir():
            if not job_dir.is_dir():
                continue
            
            job_id = job_dir.name
            
            # Filter by job_ids if specified
            if job_ids and job_id not in job_ids:
                continue
            
            # Find all array subdirectories
            for array_dir in job_dir.iterdir():
                if not array_dir.is_dir():
                    continue
                
                # Find JSON result files
                for json_file in array_dir.glob("train_results_*.json"):
                    result = load_single_result(json_file)
                    if result is None:
                        continue
                    
                    # Extract metadata
                    tag = result.get("tag", "")
                    mode = result.get("mode", "unknown")
                    tag_meta = parse_tag(tag)
                    
                    # Extract metrics
                    metrics = extract_metrics(result)
                    
                    # Build record
                    record = {
                        "job_id": job_id,
                        "model_name": model_name,
                        "tag": tag,
                        "mode": mode,
                        "smote_method": tag_meta["smote_method"],
                        "ranking": tag_meta["ranking"],
                        "domain_level": tag_meta["domain_level"],
                        "seed": tag_meta["seed"],
                        "json_path": str(json_file),
                        **metrics,
                    }
                    all_records.append(record)
    
    df = pd.DataFrame(all_records)
    logging.info(f"Collected {len(df)} result records from {len(model_names)} model(s)")
    return df


def filter_smote_experiments(df: pd.DataFrame) -> pd.DataFrame:
    """Filter DataFrame to keep only SMOTE comparison experiments."""
    # Filter by tags that contain smote-related keywords
    # Include: sw_smote, smote_rank, smote_brf, imbal_v2_smote, rank_*_smote
    smote_keywords = [
        "sw_smote", "smote_rank", "smote_brf",  # original patterns
        "imbal_v2_smote",  # pooled experiments
        "_smote_seed", "_smote_subjectwise",  # pooled variations
        "rank_knn", "rank_lof",  # ranking-based experiments
    ]
    mask = df["tag"].apply(
        lambda x: any(kw in str(x).lower() for kw in smote_keywords)
    )
    filtered = df[mask].copy()
    logging.info(f"Filtered to {len(filtered)} SMOTE experiment records")
    return filtered


def main():
    parser = argparse.ArgumentParser(description="Aggregate SMOTE comparison results")
    parser.add_argument(
        "--output", "-o",
        default="results/analysis/imbalance/smote_comparison/aggregated_results.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--models", "-m",
        nargs="+",
        default=["RF", "BalancedRF"],
        help="Model names to include",
    )
    parser.add_argument(
        "--jobs", "-j",
        nargs="+",
        default=None,
        help="Specific job IDs to include (optional)",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Include all experiments, not just SMOTE-related",
    )
    args = parser.parse_args()
    
    # Collect results
    df = collect_all_results(model_names=args.models, job_ids=args.jobs)
    
    if df.empty:
        logging.error("No results found. Check that jobs have completed.")
        sys.exit(1)
    
    # Filter to SMOTE experiments unless --all is specified
    if not args.all:
        df = filter_smote_experiments(df)
    
    if df.empty:
        logging.error("No SMOTE experiment results found after filtering.")
        sys.exit(1)
    
    # Save results
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info(f"Aggregated results saved to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SMOTE Comparison Results Summary")
    print("=" * 60)
    print(f"Total experiments: {len(df)}")
    print(f"\nBy SMOTE method:")
    print(df["smote_method"].value_counts().to_string())
    print(f"\nBy mode:")
    print(df["mode"].value_counts().to_string())
    print(f"\nBy ranking:")
    print(df["ranking"].value_counts().to_string())
    
    # Show key metrics summary
    if "test_f1" in df.columns:
        print(f"\nTest F1 by SMOTE method:")
        print(df.groupby("smote_method")["test_f1"].agg(["mean", "std", "count"]).to_string())


if __name__ == "__main__":
    main()
