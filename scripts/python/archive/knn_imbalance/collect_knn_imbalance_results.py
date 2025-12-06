#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_knn_imbalance_results.py
================================

Collect and visualize results from KNN ranking + imbalance handling experiments.

Usage:
    python scripts/python/domain_analysis/collect_knn_imbalance_results.py
    python scripts/python/domain_analysis/collect_knn_imbalance_results.py --plot
"""

import argparse
import json
import glob
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

# Optional plotting imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


def find_evaluation_results(model: str = "RF") -> List[Dict]:
    """Find all evaluation results for KNN imbalance experiments."""
    results = []
    
    eval_base = Path(f"results/evaluation/{model}")
    if not eval_base.exists():
        print(f"[WARN] Evaluation directory not found: {eval_base}")
        return results
    
    # Pattern for KNN imbalance tags
    # rank_{distance}_knn_{level}_{method}
    pattern = re.compile(
        r"rank_(?P<distance>mmd|wasserstein|dtw)_"
        r"knn_"
        r"(?P<level>out_domain|mid_domain|in_domain)_"
        r"(?P<method>baseline|undersample_rus|undersample_tomek|smote_rus|smote_tomek|smote_enn)"
    )
    
    for jobid_dir in eval_base.iterdir():
        if not jobid_dir.is_dir():
            continue
        
        # Find evaluation JSON files
        for json_file in jobid_dir.rglob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                tag = data.get('tag', '')
                match = pattern.search(tag)
                
                if match:
                    record = {
                        'jobid': jobid_dir.name,
                        'tag': tag,
                        'distance': match.group('distance'),
                        'level': match.group('level'),
                        'method': match.group('method'),
                        'mode': data.get('mode', 'unknown'),
                        'auc': data.get('auc', 0),
                        'auc_pr': data.get('auc_pr', data.get('auprc', 0)),
                        'accuracy': data.get('accuracy', 0),
                        'precision': data.get('precision', 0),
                        'recall': data.get('recall', 0),
                        'f1': data.get('f1', 0),
                        'f2': data.get('f2', 0),
                        'threshold': data.get('thr', 0.5),
                    }
                    results.append(record)
                    
            except (json.JSONDecodeError, KeyError) as e:
                continue
    
    return results


def create_summary_table(results: List[Dict]) -> pd.DataFrame:
    """Create summary DataFrame from results."""
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    
    # Order levels and methods
    level_order = ['out_domain', 'mid_domain', 'in_domain']
    method_order = ['baseline', 'undersample_rus', 'undersample_tomek', 'smote_rus', 'smote_tomek']
    
    df['level'] = pd.Categorical(df['level'], categories=level_order, ordered=True)
    df['method'] = pd.Categorical(df['method'], categories=method_order, ordered=True)
    
    return df.sort_values(['mode', 'level', 'method'])


def plot_comparison(df: pd.DataFrame, output_dir: Path):
    """Generate comparison plots."""
    if not HAS_PLOTTING:
        print("[WARN] matplotlib/seaborn not available, skipping plots")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Method labels for better readability
    method_labels = {
        'baseline': 'Baseline',
        'undersample_rus': 'RUS',
        'undersample_tomek': 'Tomek',
        'smote_rus': 'SMOTE+RUS',
        'smote_tomek': 'SMOTE+Tomek'
    }
    
    level_labels = {
        'out_domain': 'Out-Domain',
        'mid_domain': 'Mid-Domain',
        'in_domain': 'In-Domain'
    }
    
    # Metrics to plot
    metrics = ['auc_pr', 'f2', 'recall', 'f1']
    metric_titles = {
        'auc_pr': 'AUPRC',
        'f2': 'F2 Score',
        'recall': 'Recall',
        'f1': 'F1 Score'
    }
    
    # Color palette
    colors = sns.color_palette("husl", n_colors=5)
    
    for mode in df['mode'].unique():
        df_mode = df[df['mode'] == mode]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'KNN Ranking + Imbalance Comparison ({mode})', fontsize=14, fontweight='bold')
        
        for ax, metric in zip(axes.flat, metrics):
            # Pivot for grouped bar chart
            pivot = df_mode.pivot_table(
                index='level', 
                columns='method', 
                values=metric, 
                aggfunc='mean'
            )
            
            # Reorder columns
            method_order = ['baseline', 'undersample_rus', 'undersample_tomek', 'smote_rus', 'smote_tomek']
            pivot = pivot[[m for m in method_order if m in pivot.columns]]
            
            # Rename for display
            pivot.index = [level_labels.get(l, l) for l in pivot.index]
            pivot.columns = [method_labels.get(m, m) for m in pivot.columns]
            
            # Plot
            pivot.plot(kind='bar', ax=ax, color=colors[:len(pivot.columns)], width=0.8)
            ax.set_title(metric_titles.get(metric, metric))
            ax.set_xlabel('')
            ax.set_ylabel(metric_titles.get(metric, metric))
            ax.legend(title='Method', bbox_to_anchor=(1.02, 1), loc='upper left')
            ax.tick_params(axis='x', rotation=0)
            
            # Add value labels
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f', fontsize=8, rotation=90, padding=3)
        
        plt.tight_layout()
        
        output_file = output_dir / f"knn_imbalance_comparison_{mode}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[SAVED] {output_file}")
    
    # === Heatmap: Method × Level for best metric (AUPRC) ===
    for mode in df['mode'].unique():
        df_mode = df[df['mode'] == mode]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        pivot = df_mode.pivot_table(
            index='method',
            columns='level',
            values='auc_pr',
            aggfunc='mean'
        )
        
        # Reorder
        level_order = ['out_domain', 'mid_domain', 'in_domain']
        method_order = ['baseline', 'undersample_rus', 'undersample_tomek', 'smote_rus', 'smote_tomek']
        pivot = pivot.reindex(index=[m for m in method_order if m in pivot.index],
                              columns=[l for l in level_order if l in pivot.columns])
        
        # Rename
        pivot.index = [method_labels.get(m, m) for m in pivot.index]
        pivot.columns = [level_labels.get(l, l) for l in pivot.columns]
        
        sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax,
                    cbar_kws={'label': 'AUPRC'})
        ax.set_title(f'AUPRC Heatmap: Method × Level ({mode})')
        ax.set_xlabel('Domain Level')
        ax.set_ylabel('Imbalance Method')
        
        plt.tight_layout()
        output_file = output_dir / f"knn_imbalance_heatmap_{mode}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[SAVED] {output_file}")


def print_summary(df: pd.DataFrame):
    """Print summary statistics."""
    if df.empty:
        print("[WARN] No results found")
        return
    
    print("\n" + "=" * 100)
    print("KNN Ranking + Imbalance Comparison Results")
    print("=" * 100)
    
    # Best results per mode
    for mode in df['mode'].unique():
        df_mode = df[df['mode'] == mode]
        
        print(f"\n### Mode: {mode} ###")
        print("-" * 100)
        
        # Summary table
        summary = df_mode.groupby(['level', 'method']).agg({
            'auc_pr': 'mean',
            'f2': 'mean',
            'recall': 'mean',
            'f1': 'mean'
        }).round(4)
        
        print(summary.to_string())
        
        # Best per level
        print(f"\n--- Best AUPRC per level ({mode}) ---")
        for level in ['out_domain', 'mid_domain', 'in_domain']:
            df_level = df_mode[df_mode['level'] == level]
            if not df_level.empty:
                best_idx = df_level['auc_pr'].idxmax()
                best = df_level.loc[best_idx]
                print(f"  {level}: {best['method']} (AUPRC={best['auc_pr']:.4f}, F2={best['f2']:.4f})")
    
    # Overall best
    print("\n" + "=" * 100)
    print("Overall Best Configuration")
    print("=" * 100)
    
    best_idx = df['auc_pr'].idxmax()
    best = df.loc[best_idx]
    print(f"  Mode:   {best['mode']}")
    print(f"  Level:  {best['level']}")
    print(f"  Method: {best['method']}")
    print(f"  AUPRC:  {best['auc_pr']:.4f}")
    print(f"  F2:     {best['f2']:.4f}")
    print(f"  Recall: {best['recall']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Collect KNN imbalance comparison results")
    parser.add_argument("--model", default="RF", help="Model name")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--output", default="results/domain_analysis/summary/knn_imbalance",
                        help="Output directory for plots/CSV")
    args = parser.parse_args()
    
    print("Collecting evaluation results...")
    results = find_evaluation_results(args.model)
    
    if not results:
        print("[ERROR] No results found. Make sure training and evaluation have completed.")
        return 1
    
    print(f"Found {len(results)} evaluation records")
    
    df = create_summary_table(results)
    
    # Print summary
    print_summary(df)
    
    # Save CSV
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_file = output_dir / "knn_imbalance_results.csv"
    df.to_csv(csv_file, index=False)
    print(f"\n[SAVED] {csv_file}")
    
    # Generate plots
    if args.plot:
        plot_comparison(df, output_dir / "png")
    
    return 0


if __name__ == "__main__":
    exit(main())
