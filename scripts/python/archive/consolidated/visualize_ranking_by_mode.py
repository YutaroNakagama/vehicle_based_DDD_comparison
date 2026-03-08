#!/usr/bin/env python3
"""
Ranking Method Comparison - Visualization by Mode (source_only / target_only)
Using F2 score as the primary evaluation metric
Visualize latest Ranking V2 experiment results
"""

import json
import re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Setup
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (14, 10)
output_dir = Path("results/domain_analysis/ranking_comparison")
output_dir.mkdir(parents=True, exist_ok=True)

# Load results
results_dir = Path("results/evaluation/RF")
ranking_results = []

print("="*70)
print("Loading Ranking V2 Evaluation Results...")
print("="*70)

for json_file in results_dir.glob("**/eval_results_RF_*_rank_cmp_*.json"):
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        filename = json_file.name
        # Pattern: eval_results_RF_{mode}_rank_cmp_{method}_{distance}_{level}_s{seed}.json
        match = re.match(r'eval_results_RF_(\w+_only)_rank_cmp_(.+)_s(\d+)\.json', filename)
        if match:
            mode = match.group(1)
            rest = match.group(2)
            seed = match.group(3)
            
            # Parse method, distance, and level
            method = None
            distance = None
            level = None
            
            for dist in ['dtw', 'mmd', 'wasserstein']:
                if f'_{dist}_' in rest:
                    parts = rest.split(f'_{dist}_')
                    method = parts[0]
                    level = parts[1]
                    distance = dist
                    break
            
            if not method or not distance or not level:
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
                'file': str(json_file)
            })
    except Exception as e:
        pass

df = pd.DataFrame(ranking_results)
print(f"Loaded {len(df)} results")
print(f"Modes: {df['mode'].unique()}")
print(f"Methods: {df['method'].unique()}")
print(f"Distances: {df['distance'].unique()}")
print(f"Levels: {df['level'].unique()}")

# Method name mapping for display
method_names = {
    'centroid_umap': 'Centroid+UMAP',
    'lof': 'LOF',
    'knn': 'KNN',
    'median_distance': 'Median Distance',
    'isolation_forest': 'Isolation Forest',
    'mean_distance': 'Mean Distance'
}
df['method_display'] = df['method'].map(method_names)

# Distance name mapping
distance_names = {'dtw': 'DTW', 'mmd': 'MMD', 'wasserstein': 'Wasserstein'}
df['distance_display'] = df['distance'].map(distance_names)

# Mode name mapping
mode_names = {'source_only': 'Source Only', 'target_only': 'Target Only'}
df['mode_display'] = df['mode'].map(mode_names)

# Color palette for methods
method_colors = {
    'Centroid+UMAP': '#e41a1c', 
    'LOF': '#377eb8', 
    'KNN': '#4daf4a', 
    'Median Distance': '#984ea3', 
    'Isolation Forest': '#ff7f00', 
    'Mean Distance': '#a65628'
}
distance_markers = {'DTW': 'o', 'MMD': 's', 'Wasserstein': '^'}


def create_scatter_plot(df_mode, mode_name, output_filename):
    """Create scatter plot for a specific mode"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Group by method+distance
    grouped = df_mode.groupby(['method', 'distance']).agg({
        'f2': 'mean', 'recall': 'mean', 'precision': 'mean', 'auprc': 'mean'
    }).reset_index()
    grouped['method_display'] = grouped['method'].map(method_names)
    grouped['distance_display'] = grouped['distance'].map(distance_names)
    
    # Plot each point with uniform size
    for _, row in grouped.iterrows():
        ax.scatter(row['recall'], row['f2'], 
                   s=200,  # Uniform size
                   c=method_colors.get(row['method_display'], 'gray'),
                   marker=distance_markers.get(row['distance_display'], 'o'),
                   alpha=0.7, edgecolors='black', linewidths=1.5)
        # Add AUPRC label next to each point
        ax.annotate(f"AUPRC={row['auprc']:.3f}", 
                    xy=(row['recall'], row['f2']),
                    xytext=(5, -5), textcoords='offset points',
                    fontsize=8, alpha=0.8)
    
    # Legend for methods
    for method, color in method_colors.items():
        if method in grouped['method_display'].values:
            ax.scatter([], [], c=color, s=100, label=method, edgecolors='black')
    ax.legend(title='Ranking Method', loc='upper left', fontsize=10)
    
    # Second legend for distance
    ax2 = ax.twinx()
    ax2.set_yticks([])
    for d, m in distance_markers.items():
        if d in grouped['distance_display'].values:
            ax2.scatter([], [], marker=m, s=100, c='gray', label=d, edgecolors='black')
    ax2.legend(title='Distance Metric', loc='upper right', fontsize=10)
    
    ax.set_xlabel('Recall (↑ Higher is Better)', fontsize=12)
    ax.set_ylabel('F2 Score (↑ Higher is Better)', fontsize=12)
    ax.set_title(f'Ranking Method Comparison: {mode_name}\nF2 vs Recall (AUPRC shown next to each point)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Highlight best
    if len(grouped) > 0:
        best_row = grouped.loc[grouped['f2'].idxmax()]
        ax.annotate(f"Best: {best_row['method_display']}\n+ {best_row['distance_display']}\nF2={best_row['f2']:.4f}", 
                    xy=(best_row['recall'], best_row['f2']),
                    xytext=(best_row['recall']+0.03, best_row['f2']+0.015),
                    fontsize=11, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.tight_layout()
    plt.savefig(output_dir / output_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / output_filename}")
    
    return grouped


# Create separate plots for each mode
print("\n" + "="*70)
print("Creating Scatter Plots by Mode (out_domain only)")
print("="*70)

# Filter to out_domain only
df_out = df[df['level'] == 'out_domain']
print(f"Filtered to out_domain: {len(df_out)} results (from {len(df)} total)")

# Source Only
df_source = df_out[df_out['mode'] == 'source_only']
if len(df_source) > 0:
    grouped_source = create_scatter_plot(df_source, 'Source Only (out_domain)', 'ranking_scatter_source_only.png')
    print(f"  Source Only: {len(df_source)} results")
else:
    print("  No Source Only results found")

# Target Only
df_target = df_out[df_out['mode'] == 'target_only']
if len(df_target) > 0:
    grouped_target = create_scatter_plot(df_target, 'Target Only (out_domain)', 'ranking_scatter_target_only.png')
    print(f"  Target Only: {len(df_target)} results")
else:
    print("  No Target Only results found")


# ============================================================
# Print Summary Statistics
# ============================================================
print("\n" + "="*70)
print("RANKING V2 EXPERIMENT RESULTS - BY MODE (out_domain only)")
print("="*70)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"Total Results: {len(df_out)} (out_domain)")

for mode in ['source_only', 'target_only']:
    df_mode = df_out[df_out['mode'] == mode]
    if len(df_mode) == 0:
        continue
    
    print(f"\n{'='*70}")
    print(f"MODE: {mode_names.get(mode, mode)}")
    print(f"{'='*70}")
    print(f"Results: {len(df_mode)}")
    
    # Group by method+distance
    grouped = df_mode.groupby(['method_display', 'distance_display']).agg({
        'f2': 'mean', 'recall': 'mean', 'precision': 'mean', 'auprc': 'mean'
    }).reset_index()
    
    # Top 5 by F2
    print(f"\n【TOP 5 COMBINATIONS (by F2 Score)】")
    print("-"*80)
    print(f"{'Rank':<6} {'Method':<20} {'Distance':<12} {'F2':>10} {'Recall':>10} {'AUPRC':>10}")
    print("-"*80)
    top5 = grouped.sort_values('f2', ascending=False).head(5)
    for i, (_, row) in enumerate(top5.iterrows(), 1):
        print(f"{i:<6} {row['method_display']:<20} {row['distance_display']:<12} {row['f2']:>10.4f} {row['recall']:>10.4f} {row['auprc']:>10.4f}")
    
    # Method ranking
    print(f"\n【METHOD RANKING】")
    print("-"*60)
    method_rank = df_mode.groupby('method_display').agg({'f2': 'mean', 'recall': 'mean'}).sort_values('f2', ascending=False)
    print(f"{'Rank':<6} {'Method':<20} {'F2':>10} {'Recall':>10}")
    print("-"*60)
    for i, (method, row) in enumerate(method_rank.iterrows(), 1):
        star = "★" if i == 1 else ""
        print(f"{i:<6} {method:<20} {row['f2']:>10.4f} {row['recall']:>10.4f} {star}")
    
    # Distance ranking
    print(f"\n【DISTANCE METRIC RANKING】")
    print("-"*60)
    dist_rank = df_mode.groupby('distance_display').agg({'f2': 'mean', 'recall': 'mean'}).sort_values('f2', ascending=False)
    print(f"{'Rank':<6} {'Distance':<15} {'F2':>10} {'Recall':>10}")
    print("-"*60)
    for i, (dist, row) in enumerate(dist_rank.iterrows(), 1):
        star = "★" if i == 1 else ""
        print(f"{i:<6} {dist:<15} {row['f2']:>10.4f} {row['recall']:>10.4f} {star}")

print("\n" + "="*70)
print("✅ Visualization Complete!")
print("="*70)
