#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
domain_analysis.py
==================

Unified tool for domain analysis tasks.

This script provides:
1. collect        - Collect domain metrics (pooled/ranked modes)
2. collect-ranked - Collect ranking comparison results
3. compare        - Compare source vs target analysis
4. table          - Generate comparison tables
5. generate-ranks - Generate ranking group files for new methods

Consolidates functionality from:
- collect_domain_metrics.py (already unified)
- collect_ranking_comparison.py
- compare_source_target_in_domain.py
- generate_comparison_table.py
- generate_new_rankings.py

Usage:
    python domain_analysis.py collect --source pooled
    python domain_analysis.py collect-ranked --metric dtw
    python domain_analysis.py compare --level in_domain
    python domain_analysis.py table --model_dir model/common
    python domain_analysis.py generate-ranks --metric dtw
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src import config as cfg

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# Ranking Comparison Collection
# ============================================================
def find_evaluation_files(eval_dir: Path, pattern: str = "rank_*") -> list:
    """Find all evaluation result files matching the pattern."""
    files = []
    for job_dir in eval_dir.iterdir():
        if job_dir.is_dir():
            for f in job_dir.glob(f"eval_results_RF_*{pattern}*.json"):
                files.append(f)
    return files


def parse_eval_filename(filepath: Path) -> dict:
    """Parse evaluation filename to extract experiment metadata."""
    name = filepath.stem
    parts = name.split("_")
    
    info = {
        "file": str(filepath),
        "model": "RF",
        "mode": None,
        "ranking_method": None,
        "distance": None,
        "level": None,
    }
    
    # Find mode
    if "pooled" in parts:
        info["mode"] = "pooled"
    elif "source_only" in parts:
        info["mode"] = "source_only"
    elif "target_only" in parts:
        info["mode"] = "target_only"
    
    # Find ranking method
    try:
        rank_idx = parts.index("rank")
        if rank_idx + 1 < len(parts):
            method_parts = []
            for i in range(rank_idx + 1, len(parts)):
                if parts[i] in ["dtw", "mmd", "wasserstein"]:
                    info["distance"] = parts[i]
                    break
                method_parts.append(parts[i])
            info["ranking_method"] = "_".join(method_parts)
    except ValueError:
        pass
    
    # Find domain level
    if "out_domain" in name:
        info["level"] = "out_domain"
    elif "mid_domain" in name:
        info["level"] = "mid_domain"
    elif "in_domain" in name:
        info["level"] = "in_domain"
    
    return info


def collect_ranking_results(eval_dir: Path) -> pd.DataFrame:
    """Collect all ranking comparison results."""
    files = find_evaluation_files(eval_dir)
    records = []
    
    for f in files:
        info = parse_eval_filename(f)
        try:
            with open(f) as fp:
                data = json.load(fp)
            info.update({
                "recall": data.get("recall_thr", data.get("recall")),
                "precision": data.get("prec_thr", data.get("precision")),
                "f1": data.get("f1_thr", data.get("f1")),
                "f2": data.get("f2_thr", data.get("f2")),
                "auc_pr": data.get("auc_pr"),
                "auc_roc": data.get("auc_roc"),
            })
            records.append(info)
        except Exception as e:
            logger.warning(f"Error reading {f}: {e}")
    
    return pd.DataFrame(records)


# ============================================================
# Source vs Target Comparison
# ============================================================
def extract_mode_and_details(filepath: str) -> dict:
    """Extract mode, ranking method, distance metric from filename."""
    basename = os.path.basename(filepath)
    
    result = {
        "filepath": filepath,
        "filename": basename,
        "mode": None,
        "ranking_method": None,
        "distance_metric": None,
        "level": None,
    }
    
    if "source_only" in basename:
        result["mode"] = "source_only"
    elif "target_only" in basename:
        result["mode"] = "target_only"
    elif "pooled" in basename:
        result["mode"] = "pooled"
    
    for level in ["in_domain", "mid_domain", "out_domain"]:
        if level in basename:
            result["level"] = level
            break
    
    for metric in ["mmd", "dtw", "wasserstein"]:
        if metric in basename:
            result["distance_metric"] = metric
            break
    
    for method in ["knn", "lof", "mean_distance", "median_distance", "centroid_umap", "isolation_forest"]:
        if method in basename:
            result["ranking_method"] = method
            break
    
    return result


def compare_source_target(eval_dir: Path, level: str = "in_domain") -> pd.DataFrame:
    """Compare source_only vs target_only for a specific level."""
    from glob import glob
    
    pattern = str(eval_dir / "*" / f"eval_results_*{level}*.json")
    files = glob(pattern)
    
    records = []
    for f in files:
        info = extract_mode_and_details(f)
        if info["level"] != level:
            continue
        
        try:
            with open(f) as fp:
                data = json.load(fp)
            info.update({
                "f2": data.get("f2_thr", data.get("f2", 0)),
                "recall": data.get("recall_thr", data.get("recall", 0)),
                "precision": data.get("prec_thr", data.get("precision", 0)),
            })
            records.append(info)
        except Exception as e:
            logger.warning(f"Error: {e}")
    
    df = pd.DataFrame(records)
    
    if df.empty:
        logger.warning(f"No data found for level={level}")
        return df
    
    # Create comparison
    comparison = df.pivot_table(
        index=["ranking_method", "distance_metric"],
        columns="mode",
        values=["f2", "recall", "precision"],
        aggfunc="mean"
    )
    
    return comparison


# ============================================================
# Generate Rankings
# ============================================================
def generate_rankings_for_metric(metric: str) -> dict:
    """Generate ranking files for new methods."""
    try:
        from src.utils.io.data_io import load_numpy, load_json
        from src.analysis.clustering_projection_ranked import (
            _rank_by_knn,
            _rank_by_median_distance,
            _rank_by_isolation_forest,
            GROUP_SIZE,
        )
    except ImportError as e:
        logger.error(f"Could not import required modules: {e}")
        return {}
    
    NEW_RANKING_METHODS = ["knn", "median_distance", "isolation_forest"]
    
    base_dir = Path(cfg.RESULTS_DOMAIN_ANALYSIS_PATH) / "distance" / "subject-wise" / metric
    matrix_path = base_dir / f"{metric}_matrix.npy"
    subjects_path = base_dir / f"{metric}_subjects.json"
    
    if not matrix_path.exists() or not subjects_path.exists():
        logger.warning(f"Files not found for {metric}")
        return {}
    
    matrix = load_numpy(matrix_path)
    subjects = load_json(subjects_path)
    
    logger.info(f"Loaded {metric.upper()} matrix: {matrix.shape}")
    
    groups_dir = base_dir / "groups" / "clustering_ranked"
    groups_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for method in NEW_RANKING_METHODS:
        logger.info(f"  Generating {method} rankings...")
        
        if method == "knn":
            rank_labels, scores = _rank_by_knn(matrix, group_size=GROUP_SIZE, k=5)
        elif method == "median_distance":
            rank_labels, scores = _rank_by_median_distance(matrix, group_size=GROUP_SIZE)
        elif method == "isolation_forest":
            rank_labels, scores = _rank_by_isolation_forest(matrix, group_size=GROUP_SIZE)
        else:
            continue
        
        groups = {"out_domain": [], "mid_domain": [], "in_domain": []}
        for i, subj in enumerate(subjects):
            rank = rank_labels[i]
            if rank in groups:
                groups[rank].append(subj)
        
        for rank_name, members in groups.items():
            if members:
                out_path = groups_dir / f"{metric}_{method}_{rank_name.lower()}.txt"
                out_path.write_text("\n".join(members) + "\n")
                logger.info(f"    Saved: {out_path.name} ({len(members)})")
        
        results[method] = {rank: len(members) for rank, members in groups.items()}
    
    return results


# ============================================================
# CLI Commands
# ============================================================
def cmd_collect(args) -> int:
    """Delegate to existing collect_domain_metrics.py."""
    script_path = Path(__file__).parent / "collect_domain_metrics.py"
    cmd = f"python {script_path} --source {args.source}"
    if args.ranked:
        cmd += " --ranked"
    logger.info(f"Running: {cmd}")
    return os.system(cmd)


def cmd_collect_ranked(args) -> int:
    """Collect ranking comparison results."""
    logger.info("=" * 60)
    logger.info("Collecting ranking comparison results")
    logger.info("=" * 60)
    
    eval_dir = PROJECT_ROOT / "results" / "evaluation" / "RF"
    df = collect_ranking_results(eval_dir)
    
    if df.empty:
        logger.error("No results found")
        return 1
    
    logger.info(f"Collected {len(df)} results")
    
    # Save
    out_dir = Path(cfg.RESULTS_DOMAIN_ANALYSIS_PATH) / "ranking_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_file = out_dir / "ranking_comparison_results.csv"
    df.to_csv(out_file, index=False)
    logger.info(f"Saved: {out_file}")
    
    # Summary
    print("\nSummary by method:")
    print(df.groupby(["ranking_method", "level"])["f2"].mean().unstack().to_string())
    
    return 0


def cmd_compare(args) -> int:
    """Compare source_only vs target_only."""
    logger.info("=" * 60)
    logger.info(f"Comparing source_only vs target_only for: {args.level}")
    logger.info("=" * 60)
    
    eval_dir = PROJECT_ROOT / "results" / "evaluation" / "RF"
    comparison = compare_source_target(eval_dir, level=args.level)
    
    if comparison.empty:
        return 1
    
    print("\nComparison Table:")
    print(comparison.to_string())
    
    # Save
    out_dir = Path(cfg.RESULTS_DOMAIN_ANALYSIS_PATH) / "source_target_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_file = out_dir / f"comparison_{args.level}.csv"
    comparison.to_csv(out_file)
    logger.info(f"Saved: {out_file}")
    
    return 0


def cmd_table(args) -> int:
    """Generate comparison table."""
    try:
        from src.analysis.metrics_tables import summarize_metrics, make_comparison_table
    except ImportError:
        logger.error("Could not import metrics_tables module")
        return 1
    
    model_dir = Path(args.model_dir)
    summary_csv = model_dir / "summary_only10_vs_finetune.csv"
    out_csv = model_dir / "table_only10_vs_finetune_wide.csv"
    
    if not summary_csv.exists():
        summarize_metrics(model_dir=model_dir, model_tag=args.model_tag, 
                          split=args.split, out_csv=summary_csv)
    
    wide = make_comparison_table(summary_df_or_path=summary_csv, out_csv=out_csv)
    print(f"Saved: {out_csv}")
    print(wide)
    
    return 0


def cmd_generate_ranks(args) -> int:
    """Generate ranking group files."""
    logger.info("=" * 60)
    logger.info("Generating new ranking files")
    logger.info("=" * 60)
    
    metrics = [args.metric] if args.metric else ["mmd", "wasserstein", "dtw"]
    
    for metric in metrics:
        logger.info(f"\nProcessing {metric.upper()}...")
        results = generate_rankings_for_metric(metric)
        
        if results:
            for method, counts in results.items():
                logger.info(f"  {method}: out={counts.get('out_domain', 0)}, "
                            f"mid={counts.get('mid_domain', 0)}, "
                            f"in={counts.get('in_domain', 0)}")
    
    logger.info("\n✅ Done!")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Unified domain analysis tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python domain_analysis.py collect --source pooled
    python domain_analysis.py collect-ranked
    python domain_analysis.py compare --level in_domain
    python domain_analysis.py table --model_dir model/common
    python domain_analysis.py generate-ranks --metric dtw
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # collect
    p_collect = subparsers.add_parser("collect", help="Collect domain metrics")
    p_collect.add_argument("--source", default="directory", choices=["directory", "pbs_logs"])
    p_collect.add_argument("--ranked", action="store_true")
    p_collect.set_defaults(func=cmd_collect)
    
    # collect-ranked
    p_ranked = subparsers.add_parser("collect-ranked", help="Collect ranking comparison results")
    p_ranked.set_defaults(func=cmd_collect_ranked)
    
    # compare
    p_compare = subparsers.add_parser("compare", help="Compare source vs target modes")
    p_compare.add_argument("--level", default="in_domain",
                           choices=["in_domain", "mid_domain", "out_domain"])
    p_compare.set_defaults(func=cmd_compare)
    
    # table
    p_table = subparsers.add_parser("table", help="Generate comparison tables")
    p_table.add_argument("--model_dir", default="model/common")
    p_table.add_argument("--model_tag", default="RF")
    p_table.add_argument("--split", default="test")
    p_table.set_defaults(func=cmd_table)
    
    # generate-ranks
    p_ranks = subparsers.add_parser("generate-ranks", help="Generate ranking group files")
    p_ranks.add_argument("--metric", choices=["mmd", "dtw", "wasserstein"],
                         help="Specific metric (default: all)")
    p_ranks.set_defaults(func=cmd_generate_ranks)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
