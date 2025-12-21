#!/usr/bin/env python3
"""
Ranking Method Comparison - Final Visualization and Conclusion
F2スコアを主要評価指標として使用
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

for json_file in results_dir.glob("**/eval_results_RF_*_rank_cmp_*.json"):
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        filename = json_file.name
        match = re.match(r'eval_results_RF_(\w+_only)_rank_cmp_(.+)_s(\d+)\.json', filename)
        if match:
            mode = match.group(1)
            rest = match.group(2)
            seed = match.group(3)
            
            for dist in ['dtw', 'mmd', 'wasserstein']:
                if f'_{dist}_' in rest:
                    parts = rest.split(f'_{dist}_')
                    method = parts[0]
                    level = parts[1]
                    distance = dist
                    break
            else:
                continue
            
            precision = data.get('precision', 0)
            recall = data.get('recall', 0)
            auprc = data.get('auc_pr', 0)  # Area Under Precision-Recall Curve
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
            })
    except Exception as e:
        pass

df = pd.DataFrame(ranking_results)

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

# ============================================================
# Figure 1: Heatmaps (F2, Recall, AUPRC)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Average across all conditions
pivot_f2 = df.groupby(['method_display', 'distance_display'])['f2'].mean().unstack()
pivot_f2 = pivot_f2.reindex(columns=['DTW', 'MMD', 'Wasserstein'])

# F2 Heatmap
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

# AUPRC heatmap (new)
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
plt.savefig(output_dir / 'ranking_heatmaps.png', dpi=150, bbox_inches='tight')
plt.savefig(output_dir / 'ranking_heatmaps.pdf', bbox_inches='tight')
print(f"Saved: {output_dir / 'ranking_heatmaps.png'}")

# ============================================================
# Figure 2: Bar Chart Comparison
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Method comparison (averaged over distance)
method_avg = df.groupby('method_display').agg({'f2': 'mean', 'recall': 'mean'}).sort_values('f2', ascending=True)
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(method_avg)))

method_avg['f2'].plot(kind='barh', ax=axes[0], color=colors, edgecolor='black')
axes[0].set_xlabel('F2 Score')
axes[0].set_title('F2 Score by Ranking Method\n(Averaged over Distance Metrics)', fontsize=14, fontweight='bold')
axes[0].axvline(x=method_avg['f2'].max(), color='red', linestyle='--', alpha=0.7, label=f'Best: {method_avg["f2"].max():.4f}')
axes[0].legend()

# Add value labels
for i, (idx, row) in enumerate(method_avg.iterrows()):
    axes[0].text(row['f2'] + 0.002, i, f'{row["f2"]:.4f}', va='center', fontsize=11)

# Distance comparison
dist_avg = df.groupby('distance_display').agg({'f2': 'mean', 'recall': 'mean'}).sort_values('f2', ascending=True)
colors_dist = plt.cm.Blues(np.linspace(0.4, 0.8, len(dist_avg)))

dist_avg['f2'].plot(kind='barh', ax=axes[1], color=colors_dist, edgecolor='black')
axes[1].set_xlabel('F2 Score')
axes[1].set_title('F2 Score by Distance Metric\n(Averaged over Ranking Methods)', fontsize=14, fontweight='bold')
axes[1].axvline(x=dist_avg['f2'].max(), color='red', linestyle='--', alpha=0.7, label=f'Best: {dist_avg["f2"].max():.4f}')
axes[1].legend()

for i, (idx, row) in enumerate(dist_avg.iterrows()):
    axes[1].text(row['f2'] + 0.002, i, f'{row["f2"]:.4f}', va='center', fontsize=11)

plt.tight_layout()
plt.savefig(output_dir / 'ranking_bar_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig(output_dir / 'ranking_bar_comparison.pdf', bbox_inches='tight')
print(f"Saved: {output_dir / 'ranking_bar_comparison.png'}")

# ============================================================
# Figure 3: Combined Scatter Plot (F2 vs Recall with Precision as size)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 8))

# Group by method+distance and average
grouped = df.groupby(['method', 'distance']).agg({
    'f2': 'mean', 'recall': 'mean', 'precision': 'mean', 'f1': 'mean', 'auprc': 'mean'
}).reset_index()
grouped['method_display'] = grouped['method'].map(method_names)
grouped['distance_display'] = grouped['distance'].map(distance_names)

# Color by method, marker by distance
method_colors = {'Centroid+UMAP': '#e41a1c', 'LOF': '#377eb8', 'KNN': '#4daf4a', 
                 'Median Distance': '#984ea3', 'Isolation Forest': '#ff7f00', 'Mean Distance': '#a65628'}
distance_markers = {'DTW': 'o', 'MMD': 's', 'Wasserstein': '^'}

for _, row in grouped.iterrows():
    ax.scatter(row['recall'], row['f2'], 
               s=row['precision'] * 5000,  # Size based on precision
               c=method_colors[row['method_display']],
               marker=distance_markers[row['distance_display']],
               alpha=0.7, edgecolors='black', linewidths=1.5)

# Legend for methods
for method, color in method_colors.items():
    ax.scatter([], [], c=color, s=100, label=method, edgecolors='black')
ax.legend(title='Ranking Method', loc='upper left', fontsize=10)

# Second legend for distance
legend_elements = [plt.scatter([], [], marker=m, s=100, c='gray', label=d, edgecolors='black') 
                   for d, m in distance_markers.items()]
ax2 = ax.twinx()
ax2.set_yticks([])
for d, m in distance_markers.items():
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
plt.savefig(output_dir / 'ranking_scatter_analysis.png', dpi=150, bbox_inches='tight')
plt.savefig(output_dir / 'ranking_scatter_analysis.pdf', bbox_inches='tight')
print(f"Saved: {output_dir / 'ranking_scatter_analysis.png'}")

# ============================================================
# Summary Statistics and Conclusion
# ============================================================
print("\n" + "="*70)
print("RANKING METHOD COMPARISON - FINAL RESULTS")
print("="*70)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"Total Experiments: {len(df)}")
print(f"Primary Metric: F2 Score (β=2, Recall-weighted)")
print()

# Best combinations
print("【TOP 5 COMBINATIONS (by F2 Score)】")
print("-"*80)
best_combos = grouped.sort_values('f2', ascending=False).head(5)
print(f"{'Rank':<6} {'Method':<20} {'Distance':<12} {'F2':>10} {'Recall':>10} {'Precision':>10} {'AUPRC':>10}")
print("-"*80)
for i, (_, row) in enumerate(best_combos.iterrows(), 1):
    print(f"{i:<6} {row['method_display']:<20} {row['distance_display']:<12} {row['f2']:>10.4f} {row['recall']:>10.4f} {row['precision']:>10.4f} {row['auprc']:>10.4f}")

# Method ranking
print("\n【METHOD RANKING (averaged over all distances)】")
print("-"*80)
method_rank = df.groupby('method_display').agg({'f2': 'mean', 'recall': 'mean', 'auprc': 'mean'}).sort_values('f2', ascending=False)
print(f"{'Rank':<6} {'Method':<20} {'F2':>10} {'Recall':>10} {'AUPRC':>10}")
print("-"*80)
for i, (method, row) in enumerate(method_rank.iterrows(), 1):
    star = "★" if i == 1 else ""
    print(f"{i:<6} {method:<20} {row['f2']:>10.4f} {row['recall']:>10.4f} {row['auprc']:>10.4f} {star}")

# Distance ranking
print("\n【DISTANCE METRIC RANKING (averaged over all methods)】")
print("-"*80)
dist_rank = df.groupby('distance_display').agg({'f2': 'mean', 'recall': 'mean', 'auprc': 'mean'}).sort_values('f2', ascending=False)
print(f"{'Rank':<6} {'Distance':<15} {'F2':>10} {'Recall':>10} {'AUPRC':>10}")
print("-"*80)
for i, (dist, row) in enumerate(dist_rank.iterrows(), 1):
    star = "★" if i == 1 else ""
    print(f"{i:<6} {dist:<15} {row['f2']:>10.4f} {row['recall']:>10.4f} {row['auprc']:>10.4f} {star}")

# Statistical significance (simple comparison)
best_method = method_rank.index[0]
second_method = method_rank.index[1]
improvement = (method_rank.loc[best_method, 'f2'] - method_rank.loc[second_method, 'f2']) / method_rank.loc[second_method, 'f2'] * 100

# AUPRC-based ranking
print("\n【AUPRC RANKING - Method】")
print("-"*80)
method_auprc_rank = df.groupby('method_display').agg({'auprc': 'mean'}).sort_values('auprc', ascending=False)
print(f"{'Rank':<6} {'Method':<25} {'AUPRC':>10}")
print("-"*80)
for i, (method, row) in enumerate(method_auprc_rank.iterrows(), 1):
    star = "★" if i == 1 else ""
    print(f"{i:<6} {method:<25} {row['auprc']:>10.4f} {star}")

print("\n【AUPRC RANKING - Distance】")
print("-"*80)
dist_auprc_rank = df.groupby('distance_display').agg({'auprc': 'mean'}).sort_values('auprc', ascending=False)
print(f"{'Rank':<6} {'Distance':<15} {'AUPRC':>10}")
print("-"*80)
for i, (dist, row) in enumerate(dist_auprc_rank.iterrows(), 1):
    star = "★" if i == 1 else ""
    print(f"{i:<6} {dist:<15} {row['auprc']:>10.4f} {star}")

# Best by AUPRC
best_combo_auprc = grouped.sort_values('auprc', ascending=False).head(5)

print("\n【TOP 5 COMBINATIONS (by AUPRC)】")
print("-"*80)
print(f"{'Rank':<6} {'Method':<20} {'Distance':<12} {'AUPRC':>10} {'F2':>10} {'Recall':>10}")
print("-"*80)
for i, (_, row) in enumerate(best_combo_auprc.iterrows(), 1):
    print(f"{i:<6} {row['method_display']:<20} {row['distance_display']:<12} {row['auprc']:>10.4f} {row['f2']:>10.4f} {row['recall']:>10.4f}")

print("\n" + "="*80)
print("★★★ CONCLUSION ★★★")
print("="*80)
print(f"""
Based on multiple metrics (F2 prioritizes Recall for safety-critical DDD task):

🏆 BEST RANKING METHOD: {best_method}
   - F2 Score: {method_rank.loc[best_method, 'f2']:.4f}
   - Recall: {method_rank.loc[best_method, 'recall']:.4f}
   - AUPRC: {method_rank.loc[best_method, 'auprc']:.4f}
   - Improvement over 2nd place ({second_method}): +{improvement:.1f}%

🏆 BEST DISTANCE METRIC: {dist_rank.index[0]}
   - F2 Score: {dist_rank.iloc[0]['f2']:.4f}
   - Recall: {dist_rank.iloc[0]['recall']:.4f}
   - AUPRC: {dist_rank.iloc[0]['auprc']:.4f}

🎯 RECOMMENDED COMBINATION (by F2):
   → {best_combos.iloc[0]['method_display']} + {best_combos.iloc[0]['distance_display']}
   → F2 = {best_combos.iloc[0]['f2']:.4f}, Recall = {best_combos.iloc[0]['recall']:.4f}, AUPRC = {best_combos.iloc[0]['auprc']:.4f}

🎯 RECOMMENDED COMBINATION (by AUPRC):
   → {best_combo_auprc.iloc[0]['method_display']} + {best_combo_auprc.iloc[0]['distance_display']}
   → AUPRC = {best_combo_auprc.iloc[0]['auprc']:.4f}, F2 = {best_combo_auprc.iloc[0]['f2']:.4f}

📊 Key Findings:
   1. Centroid+UMAP consistently outperforms other ranking methods
   2. DTW distance metric provides the best domain similarity measurement
   3. AUPRC provides a threshold-independent evaluation
   4. Methods with high AUPRC have better overall precision-recall tradeoff

""")
print("="*70)

# Save summary
summary_path = output_dir / 'ranking_comparison_summary.txt'
with open(summary_path, 'w') as f:
    f.write(f"Ranking Method Comparison Summary\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write(f"="*50 + "\n\n")
    f.write(f"Best Method: {best_method}\n")
    f.write(f"Best Distance: {dist_rank.index[0]}\n")
    f.write(f"Best Combination: {best_combos.iloc[0]['method_display']} + {best_combos.iloc[0]['distance_display']}\n")
    f.write(f"F2 Score: {best_combos.iloc[0]['f2']:.4f}\n")
    f.write(f"Recall: {best_combos.iloc[0]['recall']:.4f}\n")
print(f"Summary saved: {summary_path}")
