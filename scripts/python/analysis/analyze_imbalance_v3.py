#!/usr/bin/env python3
"""
analyze_imbalance_v3.py
=======================
Unified script for analyzing imbalance_v3 domain analysis results.

This script consolidates:
- analyze_imbalance_v3_results.py (directory-based metadata extraction)
- analyze_imbalance_v3_with_logs.py (PBS log-based metadata extraction)

Usage:
    python analyze_imbalance_v3.py --source directory   # Extract metadata from directory structure
    python analyze_imbalance_v3.py --source pbs_logs    # Extract metadata from PBS logs
    python analyze_imbalance_v3.py                       # Default: directory-based
"""

import argparse
import json
import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
PROJECT_ROOT = Path("/home/s2240011/git/ddd/vehicle_based_DDD_comparison")
EVAL_BASE = PROJECT_ROOT / "results/domain_analysis/imbalance_v3/evaluation"
EVAL_RF_BASE = PROJECT_ROOT / "results/evaluation"
PBS_LOG_DIR = PROJECT_ROOT / "scripts/hpc/logs"
OUTPUT_DIR = PROJECT_ROOT / "results/domain_analysis/imbalance_v3/analysis"

CONDITIONS = [
    "smote_0.1", "smote_0.5", "smote_1.0",
    "smote_tomek_0.1", "smote_tomek_0.5", "smote_tomek_1.0",
    "smote_balanced_rf_0.1", "smote_balanced_rf_0.5", "smote_balanced_rf_1.0",
    "undersample_rus_0.1", "undersample_rus_0.5", "undersample_rus_1.0",
    "baseline"
]

RANKINGS = ["knn", "lof", "median_distance", "pooled"]
METRICS = ["mmd", "wasserstein", "dtw"]
LEVELS = ["out_domain", "mid_domain", "in_domain"]
MODES = ["source_only", "target_only"]

# Training job mapping (used for directory-based extraction)
JOB_MAPPING = {
    "smote_0.1": "14596401",
    "smote_0.5": "14596394",
    "smote_1.0": "14596394",
    "smote_tomek_0.1": "14596404",
    "smote_tomek_0.5": "14596405",
    "smote_tomek_1.0": "14598151",
    "smote_balanced_rf_0.1": "14598153",
    "smote_balanced_rf_0.5": "14598154",
    "smote_balanced_rf_1.0": "14598155",
    "undersample_rus_0.1": "14598156",
    "undersample_rus_0.5": "14598157",
    "undersample_rus_1.0": "14598158",
    "baseline": "14598159"
}

# Evaluation job IDs (used for PBS log-based extraction)
EVAL_JOBS = ["14600379", "14600382"]


# ============================================================
# Common Functions
# ============================================================
def load_pooled_results():
    """Load pooled evaluation results."""
    results = []
    
    for condition in CONDITIONS:
        pooled_dir = EVAL_BASE / condition / "pooled"
        if not pooled_dir.exists():
            continue
            
        for json_file in pooled_dir.glob("eval_results_*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                
                mode = 'source_only' if 'source_only' in json_file.name else 'target_only'
                
                results.append({
                    "condition": condition,
                    "ranking": "pooled",
                    "metric": "pooled",
                    "level": "pooled",
                    "mode": mode,
                    "accuracy": data.get("accuracy", 0),
                    "precision": data.get("precision", 0),
                    "recall": data.get("recall", 0),
                    "f1": data.get("f1", 0),
                    "auprc": data.get("auprc", data.get("average_precision", 0)),
                    "auroc": data.get("auroc", data.get("roc_auc", 0)),
                    "source": "pooled"
                })
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
    
    return results


# ============================================================
# Directory-based Extraction
# ============================================================
def load_ranking_results_from_directory():
    """Load ranking evaluation results by scanning directory structure."""
    results = []
    
    for condition in CONDITIONS:
        job_id = JOB_MAPPING.get(condition)
        if not job_id:
            continue
        
        model_type = "BalancedRF" if "balanced_rf" in condition else "RF"
        eval_model_dir = EVAL_RF_BASE / model_type / job_id
        
        if not eval_model_dir.exists():
            print(f"Warning: {eval_model_dir} not found")
            continue
        
        for subjob_dir in eval_model_dir.iterdir():
            if not subjob_dir.is_dir():
                continue
            
            for json_file in subjob_dir.glob("eval_results_*.json"):
                try:
                    with open(json_file) as f:
                        data = json.load(f)
                    
                    filename = json_file.stem
                    mode = "source_only" if "source_only" in filename else "target_only"
                    
                    model_files = list(subjob_dir.glob(f"{model_type}_*.pkl"))
                    if not model_files:
                        continue
                    
                    model_name = model_files[0].stem
                    
                    ranking = None
                    for r in ["knn", "lof", "median_distance"]:
                        if f"_{r}_" in model_name:
                            ranking = r
                            break
                    
                    if not ranking:
                        continue
                    
                    metric = None
                    for m in METRICS:
                        if f"_{m}_" in model_name:
                            metric = m
                            break
                    
                    level = None
                    for l in LEVELS:
                        if f"_{l}_" in model_name:
                            level = l
                            break
                    
                    if not metric or not level:
                        continue
                    
                    results.append({
                        "condition": condition,
                        "ranking": ranking,
                        "metric": metric,
                        "level": level,
                        "mode": mode,
                        "accuracy": data.get("accuracy", 0),
                        "precision": data.get("precision", 0),
                        "recall": data.get("recall", 0),
                        "f1": data.get("f1", 0),
                        "auprc": data.get("auprc", data.get("average_precision", 0)),
                        "auroc": data.get("auroc", data.get("roc_auc", 0)),
                        "source": "ranking"
                    })
                except Exception as e:
                    print(f"Error loading {json_file}: {e}")
    
    return results


# ============================================================
# PBS Log-based Extraction
# ============================================================
def parse_pbs_logs():
    """Parse PBS logs to extract condition -> training_job mapping."""
    mapping = {}
    eval_dir_mapping = {}
    
    for eval_job in EVAL_JOBS:
        for log_file in PBS_LOG_DIR.glob(f"{eval_job}*.OU"):
            match = re.search(r'\[(\d+)\]', str(log_file))
            if not match:
                continue
            array_idx = int(match.group(1))
            
            condition = None
            ranking = None
            train_jobid = None
            
            try:
                with open(log_file, 'r') as f:
                    for line in f:
                        if '[INFO] Condition:' in line:
                            condition = line.split(':')[-1].strip()
                        elif '[INFO] Ranking:' in line:
                            ranking = line.split(':')[-1].strip()
                        elif '[INFO] Found Training Job:' in line:
                            train_jobid = line.split(':')[-1].strip()
                        
                        if condition and ranking and train_jobid:
                            break
            except Exception as e:
                print(f"[WARN] Error reading {log_file}: {e}")
                continue
            
            if condition and ranking and train_jobid:
                key = f"{eval_job}[{array_idx}]"
                mapping[key] = {
                    'condition': condition,
                    'ranking': ranking,
                    'train_jobid': train_jobid
                }
                eval_dir_key = f"{train_jobid}[{array_idx}]"
                eval_dir_mapping[eval_dir_key] = {
                    'condition': condition,
                    'ranking': ranking,
                    'train_jobid': train_jobid,
                    'eval_array_idx': array_idx
                }
    
    return mapping, eval_dir_mapping


def load_ranking_results_from_logs(eval_dir_mapping):
    """Load evaluation results using PBS log-derived mapping."""
    results = []
    
    for model_type in ['RF', 'BalancedRF']:
        eval_dir = EVAL_RF_BASE / model_type
        if not eval_dir.exists():
            continue
        
        for job_dir in eval_dir.iterdir():
            if not job_dir.is_dir():
                continue
            
            job_id = job_dir.name
            
            for sub_dir in job_dir.iterdir():
                if not sub_dir.is_dir():
                    continue
                
                sub_dir_name = sub_dir.name
                
                if sub_dir_name in eval_dir_mapping:
                    info = eval_dir_mapping[sub_dir_name]
                    condition = info['condition']
                    ranking = info['ranking']
                else:
                    condition = 'unknown'
                    ranking = 'unknown'
                
                for json_file in sub_dir.glob('eval_results*.json'):
                    try:
                        with open(json_file, 'r') as f:
                            data = json.load(f)
                        
                        if 'source_only' in json_file.name:
                            mode = 'source_only'
                        elif 'target_only' in json_file.name:
                            mode = 'target_only'
                        else:
                            mode = 'unknown'
                        
                        match = re.search(r'\[(\d+)\]', sub_dir_name)
                        sub_idx = int(match.group(1)) if match else 0
                        
                        results.append({
                            'model_type': model_type,
                            'condition': condition,
                            'ranking': ranking,
                            'mode': mode,
                            'train_jobid': job_id,
                            'sub_idx': sub_idx,
                            'accuracy': data.get('accuracy'),
                            'precision': data.get('precision'),
                            'recall': data.get('recall'),
                            'f1': data.get('f1'),
                            'source': 'ranking'
                        })
                    except Exception as e:
                        print(f"[WARN] Error loading {json_file}: {e}")
    
    return results


# ============================================================
# Summary & Visualization
# ============================================================
def create_summary_tables(df):
    """Create summary tables."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Overall summary by condition
    condition_summary = df.groupby("condition").agg({
        "recall": ["mean", "std"],
        "precision": ["mean", "std"],
        "f1": ["mean", "std"],
    }).round(4)
    condition_summary.columns = ['_'.join(col).strip() for col in condition_summary.columns]
    condition_summary = condition_summary.sort_values("recall_mean", ascending=False)
    
    print("\n" + "="*80)
    print("SUMMARY BY CONDITION (sorted by Recall)")
    print("="*80)
    print(condition_summary.to_string())
    condition_summary.to_csv(OUTPUT_DIR / "summary_by_condition.csv")
    
    # 2. Summary by condition and ranking
    if "ranking" in df.columns:
        ranking_summary = df.groupby(["condition", "ranking"]).agg({
            "recall": "mean",
            "precision": "mean",
            "f1": "mean",
        }).round(4)
        
        print("\n" + "="*80)
        print("SUMMARY BY CONDITION AND RANKING")
        print("="*80)
        print(ranking_summary.to_string())
        ranking_summary.to_csv(OUTPUT_DIR / "summary_by_condition_ranking.csv")
    
    # 3. Summary by condition, level (domain)
    if "level" in df.columns:
        level_df = df[df["level"] != "pooled"]
        if len(level_df) > 0:
            level_summary = level_df.groupby(["condition", "level"]).agg({
                "recall": "mean",
                "f1": "mean",
            }).round(4)
            
            level_pivot = level_summary.unstack(level="level")
            print("\n" + "="*80)
            print("SUMMARY BY CONDITION AND DOMAIN LEVEL")
            print("="*80)
            print(level_pivot.to_string())
            level_pivot.to_csv(OUTPUT_DIR / "summary_by_condition_level.csv")
    
    # 4. Best configuration
    print("\n" + "="*80)
    print("TOP 10 CONFIGURATIONS BY RECALL")
    print("="*80)
    cols = [c for c in ["condition", "ranking", "metric", "level", "mode", "recall", "precision", "f1"] 
            if c in df.columns]
    top_configs = df.nlargest(10, "recall")[cols]
    print(top_configs.to_string())
    
    return condition_summary


def create_visualizations(df):
    """Create visualizations."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Recall by condition (boxplot)
    fig, ax = plt.subplots(figsize=(14, 6))
    order = df.groupby("condition")["recall"].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x="condition", y="recall", order=order, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_title("Recall Distribution by Imbalance Handling Method")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Recall")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "recall_by_condition_boxplot.png", dpi=150)
    plt.close()
    
    # 2. Heatmap: Condition vs Domain Level
    if "level" in df.columns:
        level_data = df[df["level"] != "pooled"].copy()
        if len(level_data) > 0:
            heatmap_data = level_data.pivot_table(
                values="recall",
                index="condition",
                columns="level",
                aggfunc="mean"
            )
            
            col_order = ["out_domain", "mid_domain", "in_domain"]
            heatmap_data = heatmap_data[[c for c in col_order if c in heatmap_data.columns]]
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax)
            ax.set_title("Mean Recall: Condition vs Domain Level")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "recall_heatmap_condition_level.png", dpi=150)
            plt.close()
    
    # 3. Comparison: source_only vs target_only
    if "mode" in df.columns:
        mode_data = df[df["mode"].isin(["source_only", "target_only"])].copy()
        if len(mode_data) > 0:
            mode_summary = mode_data.groupby(["condition", "mode"])["recall"].mean().unstack()
            
            fig, ax = plt.subplots(figsize=(12, 6))
            mode_summary.plot(kind="bar", ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_title("Mean Recall: Source-only vs Target-only")
            ax.set_ylabel("Recall")
            ax.legend(title="Mode")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "recall_source_vs_target.png", dpi=150)
            plt.close()
    
    # 4. Ranking comparison
    if "ranking" in df.columns:
        ranking_data = df[df["ranking"] != "pooled"].copy()
        if len(ranking_data) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=ranking_data, x="ranking", y="recall", ax=ax)
            ax.set_title("Recall Distribution by Ranking Method")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "recall_by_ranking.png", dpi=150)
            plt.close()
    
    print(f"\nVisualizations saved to: {OUTPUT_DIR}")


def print_pbs_summary(mapping, eval_dir_mapping):
    """Print PBS log parsing summary (only for pbs_logs mode)."""
    print(f"  Found {len(mapping)} evaluation jobs with mappings")
    print(f"  Created {len(eval_dir_mapping)} eval_dir mappings")
    
    # Create train_jobid -> condition mapping
    train_to_cond = defaultdict(list)
    for key, info in mapping.items():
        train_to_cond[info['train_jobid']].append({
            'condition': info['condition'],
            'ranking': info['ranking'],
            'eval_key': key
        })
    
    print(f"  Found {len(train_to_cond)} training jobs")
    print("\n  Training Job -> Condition Mapping:")
    for job_id, info_list in sorted(train_to_cond.items()):
        conditions = set(i['condition'] for i in info_list)
        rankings = set(i['ranking'] for i in info_list)
        print(f"    {job_id}: {list(conditions)} | Rankings: {list(rankings)}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Analyze imbalance_v3 domain analysis results")
    parser.add_argument(
        "--source", 
        choices=["directory", "pbs_logs"],
        default="directory",
        help="Source for metadata extraction (default: directory)"
    )
    parser.add_argument(
        "--skip-viz",
        action="store_true",
        help="Skip visualization generation"
    )
    args = parser.parse_args()
    
    print("="*80)
    print(f"IMBALANCE V3 DOMAIN ANALYSIS (source={args.source})")
    print("="*80)
    
    # Load pooled results (common to both modes)
    print("\nLoading pooled results...")
    pooled_results = load_pooled_results()
    print(f"  Loaded {len(pooled_results)} pooled results")
    
    # Load ranking results based on source
    if args.source == "pbs_logs":
        print("\n[Step 1] Parsing PBS logs...")
        mapping, eval_dir_mapping = parse_pbs_logs()
        print_pbs_summary(mapping, eval_dir_mapping)
        
        print("\n[Step 2] Loading ranking results from PBS mapping...")
        ranking_results = load_ranking_results_from_logs(eval_dir_mapping)
        
        # Count known vs unknown rankings
        known = sum(1 for r in ranking_results if r.get('ranking') != 'unknown')
        unknown = sum(1 for r in ranking_results if r.get('ranking') == 'unknown')
        print(f"  Loaded {len(ranking_results)} ranking results (known: {known}, unknown: {unknown})")
        
        # Save mapping files
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_DIR / "job_mapping.json", 'w') as f:
            json.dump(mapping, f, indent=2)
        with open(OUTPUT_DIR / "eval_dir_mapping.json", 'w') as f:
            json.dump(eval_dir_mapping, f, indent=2)
    else:
        print("\nLoading ranking results from directory structure...")
        ranking_results = load_ranking_results_from_directory()
        print(f"  Loaded {len(ranking_results)} ranking results")
    
    # Combine results
    all_results = pooled_results + ranking_results
    print(f"\nTotal: {len(all_results)} evaluation results")
    
    if len(all_results) == 0:
        print("ERROR: No results found!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Save raw data
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / "all_results.csv", index=False)
    print(f"\nRaw data saved to: {OUTPUT_DIR / 'all_results.csv'}")
    
    # Create summary tables
    create_summary_tables(df)
    
    # Create visualizations
    if not args.skip_viz:
        create_visualizations(df)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
