#!/usr/bin/env python3
"""
Analyze imbalance_v3 domain analysis results.

Aggregates evaluation results from:
- Pooled experiments (already integrated)
- Ranking experiments (knn, lof, median_distance)

Output: Summary tables and visualizations
"""

import json
import os
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

# Training job mapping
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
                
                results.append({
                    "condition": condition,
                    "ranking": "pooled",
                    "metric": "pooled",
                    "level": "pooled",
                    "mode": "pooled",
                    "accuracy": data.get("accuracy", 0),
                    "precision": data.get("precision", 0),
                    "recall": data.get("recall", 0),
                    "f1": data.get("f1", 0),
                    "auprc": data.get("auprc", data.get("average_precision", 0)),
                    "auroc": data.get("auroc", data.get("roc_auc", 0)),
                })
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
    
    return results


def load_ranking_results():
    """Load ranking evaluation results from results/evaluation/RF/."""
    results = []
    
    for condition in CONDITIONS:
        job_id = JOB_MAPPING.get(condition)
        if not job_id:
            continue
        
        # Determine model type
        model_type = "BalancedRF" if "balanced_rf" in condition else "RF"
        eval_model_dir = EVAL_RF_BASE / model_type / job_id
        
        if not eval_model_dir.exists():
            print(f"Warning: {eval_model_dir} not found")
            continue
        
        # Search for eval results in subjob directories
        for subjob_dir in eval_model_dir.iterdir():
            if not subjob_dir.is_dir():
                continue
            
            for json_file in subjob_dir.glob("eval_results_*.json"):
                try:
                    with open(json_file) as f:
                        data = json.load(f)
                    
                    # Parse filename to extract experiment info
                    filename = json_file.stem
                    
                    # Extract mode (source_only or target_only)
                    mode = "source_only" if "source_only" in filename else "target_only"
                    
                    # Extract ranking, metric, level from tag in model path
                    # Look for patterns in parent model directory
                    model_files = list(subjob_dir.glob(f"{model_type}_*.pkl"))
                    if not model_files:
                        continue
                    
                    model_name = model_files[0].stem
                    
                    # Parse ranking method
                    ranking = None
                    for r in ["knn", "lof", "median_distance"]:
                        if f"_{r}_" in model_name:
                            ranking = r
                            break
                    
                    if not ranking:
                        continue  # Skip pooled (already loaded)
                    
                    # Parse metric
                    metric = None
                    for m in METRICS:
                        if f"_{m}_" in model_name:
                            metric = m
                            break
                    
                    # Parse level
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
                    })
                except Exception as e:
                    print(f"Error loading {json_file}: {e}")
    
    return results


def create_summary_tables(df):
    """Create summary tables."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Overall summary by condition
    condition_summary = df.groupby("condition").agg({
        "recall": ["mean", "std"],
        "precision": ["mean", "std"],
        "f1": ["mean", "std"],
        "auprc": ["mean", "std"],
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
            "auprc": "mean",
        }).round(4)
        
        print("\n" + "="*80)
        print("SUMMARY BY CONDITION AND RANKING")
        print("="*80)
        print(ranking_summary.to_string())
        ranking_summary.to_csv(OUTPUT_DIR / "summary_by_condition_ranking.csv")
    
    # 3. Summary by condition, level (domain)
    if "level" in df.columns:
        level_summary = df[df["level"] != "pooled"].groupby(["condition", "level"]).agg({
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
    top_configs = df.nlargest(10, "recall")[["condition", "ranking", "metric", "level", "mode", "recall", "precision", "f1"]]
    print(top_configs.to_string())
    
    return condition_summary


def create_visualizations(df):
    """Create visualizations."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Recall by condition (boxplot)
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Order conditions by median recall
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
            
            # Reorder columns
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


def main():
    print("="*80)
    print("IMBALANCE V3 DOMAIN ANALYSIS - RESULTS AGGREGATION")
    print("="*80)
    
    # Load results
    print("\nLoading pooled results...")
    pooled_results = load_pooled_results()
    print(f"  Loaded {len(pooled_results)} pooled results")
    
    print("\nLoading ranking results...")
    ranking_results = load_ranking_results()
    print(f"  Loaded {len(ranking_results)} ranking results")
    
    # Combine
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
    create_visualizations(df)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
