#!/usr/bin/env python3
"""
visualize_ranking_comparison.py
===============================
Unified script for ranking method comparison visualization.

This script consolidates:
- visualize_ranking_final.py (heatmaps & bar charts)
- visualize_ranking_by_mode.py (scatter plots by mode)

Usage:
    python visualize_ranking_comparison.py heatmap    # Heatmaps
    python visualize_ranking_comparison.py bar        # Bar charts
    python visualize_ranking_comparison.py scatter    # Scatter plots
    python visualize_ranking_comparison.py all        # All visualizations
"""

import argparse
import json
import re
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Setup
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (14, 10)

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[4]
RESULTS_DIR = PROJECT_ROOT / "results/evaluation/RF"
OUTPUT_DIR = PROJECT_ROOT / "results/domain_analysis/ranking_comparison"

# Method and distance name mappings
METHOD_NAMES = {
    'centroid_umap': 'Centroid+UMAP',
    'lof': 'LOF',
    'knn': 'KNN',
    'median_distance': 'Median Distance',
    'isolation_forest': 'Isolation Forest',
    'mean_distance': 'Mean Distance'
}

DISTANCE_NAMES = {'dtw': 'DTW', 'mmd': 'MMD', 'wasserstein': 'Wasserstein'}

METHOD_COLORS = {
    'Centroid+UMAP': '#e41a1c', 'LOF': '#377eb8', 'KNN': '#4daf4a',
    'Median Distance': '#984ea3', 'Isolation Forest': '#ff7f00', 'Mean Distance': '#a65628'
}

DISTANCE_MARKERS = {'DTW': 'o', 'MMD': 's', 'Wasserstein': '^'}


# ============================================================
# Data Loading
# ============================================================
def load_ranking_results():
    """Load ranking comparison results from evaluation JSONs."""
    ranking_results = []
    
    for json_file in RESULTS_DIR.glob("**/eval_results_RF_*_rank_cmp_*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            filename = json_file.name
            match = re.match(r'eval_results_RF_(\w+_only)_rank_cmp_(.+)_s(\d+)\.json', filename)
            if not match:
                continue
            
            mode = match.group(1)
            rest = match.group(2)
            seed = match.group(3)
            
            method, distance, level = None, None, None
            for dist in ['dtw', 'mmd', 'wasserstein']:
                if f'_{dist}_' in rest:
                    parts = rest.split(f'_{dist}_')
                    method = parts[0]
                    level = parts[1]
                    distance = dist
                    break
            
            if not all([method, distance, level]):
                continue
            
            precision = data.get('precision', 0)
            recall = data.get('recall', 0)
            auprc = data.get('auc_pr', 0)
            beta = 2.0
            f2 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall) if (precision + recall) > 0 else 0
            
            ranking_results.append({
                'method': method,
                'distance': distance,
                'level': level,
                'mode': mode,
                'f1': data.get('f1', 0),
                'f2': f2,
                'recall': recall,
                'precision': precision,
                'auprc': auprc,
                'seed': seed,
            })
        except Exception:
            pass
    
    df = pd.DataFrame(ranking_results)
    df['method_display'] = df['method'].map(METHOD_NAMES)
    df['distance_display'] = df['distance'].map(DISTANCE_NAMES)
    
    print(f"Loaded {len(df)} results")
    print(f"Modes: {df['mode'].unique().tolist()}")
    print(f"Methods: {df['method_display'].unique().tolist()}")
    print(f"Distances: {df['distance_display'].unique().tolist()}")
    
    return df


# ============================================================
# Visualization Functions
# ============================================================
def create_heatmaps(df):
    """Create heatmap visualizations for F2, Recall, AUPRC, Precision."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # F2 Heatmap
    pivot_f2 = df.groupby(['method_display', 'distance_display'])['f2'].mean().unstack()
    pivot_f2 = pivot_f2.reindex(columns=['DTW', 'MMD', 'Wasserstein'])
    
    sns.heatmap(pivot_f2, annot=True, fmt='.4f', cmap='YlOrRd', ax=axes[0, 0],
                cbar_kws={'label': 'F2 Score'}, vmin=0.12, vmax=0.20)
    axes[0, 0].set_title('F2 Score: Method × Distance Metric\n(Higher is Better)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Distance Metric')
    axes[0, 0].set_ylabel('Ranking Method')
    
    # Recall heatmap
    pivot_recall = df.groupby(['method_display', 'distance_display'])['recall'].mean().unstack()
    pivot_recall = pivot_recall.reindex(columns=['DTW', 'MMD', 'Wasserstein'])
    
    sns.heatmap(pivot_recall, annot=True, fmt='.4f', cmap='YlGnBu', ax=axes[0, 1],
                cbar_kws={'label': 'Recall'}, vmin=0.40, vmax=0.52)
    axes[0, 1].set_title('Recall: Method × Distance Metric\n(Higher is Better)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Distance Metric')
    axes[0, 1].set_ylabel('Ranking Method')
    
    # AUPRC heatmap
    pivot_auprc = df.groupby(['method_display', 'distance_display'])['auprc'].mean().unstack()
    pivot_auprc = pivot_auprc.reindex(columns=['DTW', 'MMD', 'Wasserstein'])
    
    sns.heatmap(pivot_auprc, annot=True, fmt='.4f', cmap='Purples', ax=axes[1, 0],
                cbar_kws={'label': 'AUPRC'})
    axes[1, 0].set_title('AUPRC: Method × Distance Metric\n(Higher is Better)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Distance Metric')
    axes[1, 0].set_ylabel('Ranking Method')
    
    # Precision heatmap
    pivot_precision = df.groupby(['method_display', 'distance_display'])['precision'].mean().unstack()
    pivot_precision = pivot_precision.reindex(columns=['DTW', 'MMD', 'Wasserstein'])
    
    sns.heatmap(pivot_precision, annot=True, fmt='.4f', cmap='Greens', ax=axes[1, 1],
                cbar_kws={'label': 'Precision'})
    axes[1, 1].set_title('Precision: Method × Distance Metric\n(Higher is Better)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Distance Metric')
    axes[1, 1].set_ylabel('Ranking Method')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'ranking_heatmaps.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'ranking_heatmaps.pdf', bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'ranking_heatmaps.png'}")
    plt.close()


def create_bar_charts(df):
    """Create bar chart comparisons."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Method comparison
    method_avg = df.groupby('method_display').agg({'f2': 'mean', 'recall': 'mean'}).sort_values('f2', ascending=True)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(method_avg)))
    
    method_avg['f2'].plot(kind='barh', ax=axes[0], color=colors, edgecolor='black')
    axes[0].set_xlabel('F2 Score')
    axes[0].set_title('F2 Score by Ranking Method\n(Averaged over Distance Metrics)', fontsize=14, fontweight='bold')
    axes[0].axvline(x=method_avg['f2'].max(), color='red', linestyle='--', alpha=0.7, 
                    label=f'Best: {method_avg["f2"].max():.4f}')
    axes[0].legend()
    
    for i, (idx, row) in enumerate(method_avg.iterrows()):
        axes[0].text(row['f2'] + 0.002, i, f'{row["f2"]:.4f}', va='center', fontsize=11)
    
    # Distance comparison
    dist_avg = df.groupby('distance_display').agg({'f2': 'mean', 'recall': 'mean'}).sort_values('f2', ascending=True)
    colors_dist = plt.cm.Blues(np.linspace(0.4, 0.8, len(dist_avg)))
    
    dist_avg['f2'].plot(kind='barh', ax=axes[1], color=colors_dist, edgecolor='black')
    axes[1].set_xlabel('F2 Score')
    axes[1].set_title('F2 Score by Distance Metric\n(Averaged over Ranking Methods)', fontsize=14, fontweight='bold')
    axes[1].axvline(x=dist_avg['f2'].max(), color='red', linestyle='--', alpha=0.7,
                    label=f'Best: {dist_avg["f2"].max():.4f}')
    axes[1].legend()
    
    for i, (idx, row) in enumerate(dist_avg.iterrows()):
        axes[1].text(row['f2'] + 0.002, i, f'{row["f2"]:.4f}', va='center', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'ranking_bar_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'ranking_bar_comparison.pdf', bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'ranking_bar_comparison.png'}")
    plt.close()


def create_scatter_plots(df):
    """Create scatter plot analysis."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    grouped = df.groupby(['method', 'distance']).agg({
        'f2': 'mean', 'recall': 'mean', 'precision': 'mean', 'f1': 'mean', 'auprc': 'mean'
    }).reset_index()
    grouped['method_display'] = grouped['method'].map(METHOD_NAMES)
    grouped['distance_display'] = grouped['distance'].map(DISTANCE_NAMES)
    
    for _, row in grouped.iterrows():
        ax.scatter(row['recall'], row['f2'],
                   s=row['precision'] * 5000,
                   c=METHOD_COLORS.get(row['method_display'], 'gray'),
                   marker=DISTANCE_MARKERS.get(row['distance_display'], 'o'),
                   alpha=0.7, edgecolors='black', linewidths=1.5)
    
    # Legend for methods
    for method, color in METHOD_COLORS.items():
        ax.scatter([], [], c=color, s=100, label=method, edgecolors='black')
    ax.legend(title='Ranking Method', loc='upper left', fontsize=10)
    
    # Second legend for distance
    ax2 = ax.twinx()
    ax2.set_yticks([])
    for d, m in DISTANCE_MARKERS.items():
        ax2.scatter([], [], marker=m, s=100, c='gray', label=d, edgecolors='black')
    ax2.legend(title='Distance Metric', loc='upper right', fontsize=10)
    
    ax.set_xlabel('Recall (↑ Higher is Better)', fontsize=12)
    ax.set_ylabel('F2 Score (↑ Higher is Better)', fontsize=12)
    ax.set_title('Ranking Method Comparison: F2 vs Recall\n(Bubble size = Precision)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Highlight best
    best_row = grouped.loc[grouped['f2'].idxmax()]
    ax.annotate(f"Best: {best_row['method_display']}\n+ {best_row['distance_display']}",
                xy=(best_row['recall'], best_row['f2']),
                xytext=(best_row['recall']+0.02, best_row['f2']+0.01),
                fontsize=11, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'ranking_scatter_analysis.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'ranking_scatter_analysis.pdf', bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'ranking_scatter_analysis.png'}")
    plt.close()
    
    return grouped


def print_summary(df, grouped):
    """Print summary statistics."""
    print("\n" + "="*70)
    print("RANKING METHOD COMPARISON - SUMMARY")
    print("="*70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Total Experiments: {len(df)}")
    
    # Best combinations
    print("\n【TOP 5 COMBINATIONS (by F2 Score)】")
    print("-"*80)
    best_combos = grouped.sort_values('f2', ascending=False).head(5)
    print(f"{'Rank':<6} {'Method':<20} {'Distance':<12} {'F2':>10} {'Recall':>10} {'AUPRC':>10}")
    print("-"*80)
    for i, (_, row) in enumerate(best_combos.iterrows(), 1):
        print(f"{i:<6} {row['method_display']:<20} {row['distance_display']:<12} {row['f2']:>10.4f} {row['recall']:>10.4f} {row['auprc']:>10.4f}")
    
    # Method ranking
    print("\n【METHOD RANKING】")
    method_rank = df.groupby('method_display').agg({'f2': 'mean', 'recall': 'mean', 'auprc': 'mean'}).sort_values('f2', ascending=False)
    for i, (method, row) in enumerate(method_rank.iterrows(), 1):
        star = "★" if i == 1 else ""
        print(f"  {i}. {method}: F2={row['f2']:.4f}, Recall={row['recall']:.4f} {star}")
    
    # Distance ranking
    print("\n【DISTANCE RANKING】")
    dist_rank = df.groupby('distance_display').agg({'f2': 'mean', 'recall': 'mean'}).sort_values('f2', ascending=False)
    for i, (dist, row) in enumerate(dist_rank.iterrows(), 1):
        star = "★" if i == 1 else ""
        print(f"  {i}. {dist}: F2={row['f2']:.4f}, Recall={row['recall']:.4f} {star}")
    
    # Save summary
    summary_path = OUTPUT_DIR / 'ranking_comparison_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f"Ranking Method Comparison Summary\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"="*50 + "\n\n")
        f.write(f"Best Method: {method_rank.index[0]}\n")
        f.write(f"Best Distance: {dist_rank.index[0]}\n")
        f.write(f"Best Combination: {best_combos.iloc[0]['method_display']} + {best_combos.iloc[0]['distance_display']}\n")
        f.write(f"F2 Score: {best_combos.iloc[0]['f2']:.4f}\n")
    print(f"\nSummary saved: {summary_path}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Unified ranking method comparison visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "mode",
        choices=["heatmap", "bar", "scatter", "all"],
        help="Visualization mode: heatmap, bar, scatter, or all"
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"RANKING COMPARISON VISUALIZATION (mode={args.mode})")
    print("=" * 70)
    
    df = load_ranking_results()
    
    if len(df) == 0:
        print("[ERROR] No ranking results found!")
        return
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    grouped = None
    
    if args.mode == "heatmap" or args.mode == "all":
        print("\n--- Creating Heatmaps ---")
        create_heatmaps(df)
    
    if args.mode == "bar" or args.mode == "all":
        print("\n--- Creating Bar Charts ---")
        create_bar_charts(df)
    
    if args.mode == "scatter" or args.mode == "all":
        print("\n--- Creating Scatter Plots ---")
        grouped = create_scatter_plots(df)
    
    if args.mode == "all":
        if grouped is None:
            grouped = df.groupby(['method', 'distance']).agg({
                'f2': 'mean', 'recall': 'mean', 'precision': 'mean', 'auprc': 'mean'
            }).reset_index()
            grouped['method_display'] = grouped['method'].map(METHOD_NAMES)
            grouped['distance_display'] = grouped['distance'].map(DISTANCE_NAMES)
        print_summary(df, grouped)
    
    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
