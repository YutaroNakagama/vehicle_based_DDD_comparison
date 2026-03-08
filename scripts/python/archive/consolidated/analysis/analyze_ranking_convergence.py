#!/usr/bin/env python3
"""
Detailed convergence analysis script for ranking method comparison experiments
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Font settings
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']

def load_data(csv_path: str) -> pd.DataFrame:
    """Load and parse CSV data"""
    df = pd.read_csv(csv_path)
    
    # Extract info from tags
    def parse_tag(tag):
        parts = tag.replace('rank_cmp_', '').split('_')
        
        # Extract mode
        if 'source_only' in tag:
            mode = 'source_only'
        elif 'target_only' in tag:
            mode = 'target_only'
        else:
            mode = 'unknown'
        
        # Remove s42
        tag_clean = tag.replace('_s42', '').replace('_source_only', '').replace('_target_only', '')
        
        # Extract method and metric
        if 'centroid_umap' in tag:
            method = 'centroid_umap'
        elif 'isolation_forest' in tag:
            method = 'isolation_forest'
        elif 'mean_distance' in tag:
            method = 'mean_distance'
        elif 'median_distance' in tag:
            method = 'median_distance'
        elif 'knn' in tag:
            method = 'knn'
        elif 'lof' in tag:
            method = 'lof'
        else:
            method = 'unknown'
        
        if 'mmd' in tag:
            metric = 'mmd'
        elif 'dtw' in tag:
            metric = 'dtw'
        elif 'wasserstein' in tag:
            metric = 'wasserstein'
        else:
            metric = 'unknown'
        
        return pd.Series({'method': method, 'metric': metric, 'mode': mode})
    
    parsed = df['tag'].apply(parse_tag)
    df = pd.concat([df, parsed], axis=1)
    
    return df


def plot_metric_comparison(raw_df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: str):
    """F2 comparison by distance metric"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['mmd', 'dtw', 'wasserstein']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        metric_df = raw_df[raw_df['tag'].str.contains(metric)]
        
        for tag in metric_df['tag'].unique():
            tag_df = metric_df[metric_df['tag'] == tag]
            label = tag.replace('rank_cmp_', '').replace(f'_{metric}', '').replace('_out_domain_s42', '')
            ax.plot(tag_df['trial'], tag_df['best_f2'], label=label, alpha=0.8)
        
        ax.set_xlabel('Trial')
        ax.set_ylabel('Best F2 Score')
        ax.set_title(f'{metric.upper()}')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='lower right')
    
    plt.suptitle('F2 Convergence by Distance Metric', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'convergence_by_metric.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/convergence_by_metric.png")


def plot_mode_comparison(raw_df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: str):
    """source_only vs target_only comparison"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    methods = ['knn', 'lof', 'centroid_umap', 'mean_distance', 'median_distance', 'isolation_forest']
    
    for idx, method in enumerate(methods):
        ax = axes[idx // 3, idx % 3]
        method_df = raw_df[raw_df['tag'].str.contains(method)]
        
        # source_only
        source_df = method_df[method_df['tag'].str.contains('source_only')]
        for tag in source_df['tag'].unique():
            tag_df = source_df[source_df['tag'] == tag]
            metric = 'mmd' if 'mmd' in tag else ('dtw' if 'dtw' in tag else 'wass')
            ax.plot(tag_df['trial'], tag_df['best_f2'], 
                   label=f'{metric}_src', linestyle='-', alpha=0.8)
        
        # target_only
        target_df = method_df[method_df['tag'].str.contains('target_only')]
        for tag in target_df['tag'].unique():
            tag_df = target_df[target_df['tag'] == tag]
            metric = 'mmd' if 'mmd' in tag else ('dtw' if 'dtw' in tag else 'wass')
            ax.plot(tag_df['trial'], tag_df['best_f2'], 
                   label=f'{metric}_tgt', linestyle='--', alpha=0.8)
        
        ax.set_xlabel('Trial')
        ax.set_ylabel('Best F2 Score')
        ax.set_title(f'{method}')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, loc='lower right')
    
    plt.suptitle('F2 Convergence: source_only (solid) vs target_only (dashed)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'convergence_source_vs_target.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/convergence_source_vs_target.png")


def plot_summary_heatmap(summary_df: pd.DataFrame, output_dir: str):
    """Method x Metric heatmap"""
    
    # Pivot data
    summary = summary_df.copy()
    
    # Extract method, metric, and mode info
    def extract_info(tag):
        if 'centroid_umap' in tag:
            method = 'centroid_umap'
        elif 'isolation_forest' in tag:
            method = 'isolation_forest'
        elif 'mean_distance' in tag:
            method = 'mean_distance'
        elif 'median_distance' in tag:
            method = 'median_distance'
        elif 'knn' in tag:
            method = 'knn'
        elif 'lof' in tag:
            method = 'lof'
        else:
            method = 'other'
        
        if 'mmd' in tag:
            metric = 'mmd'
        elif 'dtw' in tag:
            metric = 'dtw'
        elif 'wasserstein' in tag:
            metric = 'wasserstein'
        else:
            metric = 'other'
        
        mode = 'source' if 'source_only' in tag else 'target'
        
        return method, metric, mode
    
    summary[['method', 'metric', 'mode']] = summary['tag'].apply(
        lambda x: pd.Series(extract_info(x)))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, mode in enumerate(['source', 'target']):
        ax = axes[idx]
        mode_df = summary[summary['mode'] == mode]
        
        pivot = mode_df.pivot_table(values='best_f2', index='method', columns='metric', aggfunc='first')
        
        im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0.16, vmax=0.20)
        
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        
        # Display values in cells
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.iloc[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f'{val:.4f}', ha='center', va='center', fontsize=9)
        
        ax.set_title(f'{mode}_only')
        ax.set_xlabel('Distance Metric')
        ax.set_ylabel('Ranking Method')
    
    plt.suptitle('Best F2 Scores: Method × Metric × Mode', fontsize=14)
    plt.colorbar(im, ax=axes, label='F2 Score', shrink=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'f2_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/f2_heatmap.png")


def main():
    base_dir = "/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
    output_dir = os.path.join(base_dir, "results/ranking_convergence")
    
    raw_csv = os.path.join(output_dir, "convergence_raw.csv")
    summary_csv = os.path.join(output_dir, "convergence_summary.csv")
    
    if not os.path.exists(raw_csv):
        print(f"Error: {raw_csv} not found")
        sys.exit(1)
    
    print("Loading data...")
    raw_df = pd.read_csv(raw_csv)
    summary_df = pd.read_csv(summary_csv)
    
    print(f"Raw data: {len(raw_df)} records")
    print(f"Summary: {len(summary_df)} experiments")
    
    print("\nGenerating detailed plots...")
    plot_metric_comparison(raw_df, summary_df, output_dir)
    plot_mode_comparison(raw_df, summary_df, output_dir)
    plot_summary_heatmap(summary_df, output_dir)
    
    print("\n=== Summary Statistics ===")
    
    # Overall statistics
    print(f"\nOverall Best F2: {summary_df['best_f2'].max():.4f}")
    print(f"Overall Mean Best F2: {summary_df['best_f2'].mean():.4f}")
    print(f"Overall Std Best F2: {summary_df['best_f2'].std():.4f}")
    
    # By metric
    print("\n=== By Distance Metric ===")
    for metric in ['mmd', 'dtw', 'wasserstein']:
        metric_df = summary_df[summary_df['tag'].str.contains(metric)]
        print(f"{metric}: mean={metric_df['best_f2'].mean():.4f}, max={metric_df['best_f2'].max():.4f}")
    
    # By mode
    print("\n=== By Mode ===")
    for mode in ['source_only', 'target_only']:
        mode_df = summary_df[summary_df['tag'].str.contains(mode)]
        print(f"{mode}: mean={mode_df['best_f2'].mean():.4f}, max={mode_df['best_f2'].max():.4f}")
    
    print(f"\nAll plots saved to: {output_dir}")


if __name__ == '__main__':
    main()
