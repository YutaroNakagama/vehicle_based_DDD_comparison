#!/usr/bin/env python3
"""
Plot confusion matrices for all evaluation results.
"""
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

def load_confusion_matrix(json_path):
    """Load confusion matrix from evaluation JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract confusion matrix from test results
    test_results = data.get('test', {})
    
    # Confusion matrix should be stored as [[TN, FP], [FN, TP]]
    if 'confusion_matrix' in test_results:
        cm = np.array(test_results['confusion_matrix'])
    else:
        # Try to compute from other metrics if available
        tn = test_results.get('tn', 0)
        fp = test_results.get('fp', 0)
        fn = test_results.get('fn', 0)
        tp = test_results.get('tp', 0)
        cm = np.array([[tn, fp], [fn, tp]])
    
    return cm, test_results

def parse_filename(filename):
    """Parse evaluation result filename to extract metadata."""
    # eval_results_RF_{mode}_rank_{distance}_mean_mean_{level}.json
    parts = filename.replace('eval_results_', '').replace('.json', '').split('_')
    
    mode = None
    distance = None
    level = None
    
    # Find mode
    if 'pooled' in filename:
        mode = 'pooled'
    elif 'source_only' in filename:
        mode = 'source_only'
    elif 'target_only' in filename:
        mode = 'target_only'
    
    # Find distance
    if 'dtw' in filename:
        distance = 'dtw'
    elif 'mmd' in filename:
        distance = 'mmd'
    elif 'wasserstein' in filename:
        distance = 'wasserstein'
    
    # Find level
    if 'high' in filename:
        level = 'high'
    elif 'middle' in filename:
        level = 'middle'
    elif 'low' in filename:
        level = 'low'
    
    return mode, distance, level

def plot_confusion_matrix(cm, title, ax, metrics=None):
    """Plot a single confusion matrix."""
    # Normalize to percentages
    cm_norm = cm.astype('float') / cm.sum() * 100
    
    # Create annotations with both counts and percentages
    annot = np.array([[f'{cm[i,j]}\n({cm_norm[i,j]:.1f}%)' 
                       for j in range(2)] for i in range(2)])
    
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', 
                square=True, ax=ax, cbar=False,
                xticklabels=['Non-Drowsy', 'Drowsy'],
                yticklabels=['Non-Drowsy', 'Drowsy'])
    
    ax.set_xlabel('Predicted', fontsize=9)
    ax.set_ylabel('True', fontsize=9)
    
    # Add metrics to title if provided
    if metrics:
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)
        f1 = metrics.get('f1', 0)
        title_with_metrics = f"{title}\nP={precision:.3f} R={recall:.3f} F1={f1:.3f}"
    else:
        title_with_metrics = title
    
    ax.set_title(title_with_metrics, fontsize=10, pad=10)

def main():
    # Find all evaluation JSON files
    eval_dir = Path('results/evaluation/RF/14357179')
    
    if not eval_dir.exists():
        print(f"Error: {eval_dir} does not exist")
        return
    
    # Collect all evaluation files
    eval_files = list(eval_dir.rglob('eval_results_*.json'))
    print(f"Found {len(eval_files)} evaluation files")
    
    # Organize by distance and level
    data_dict = defaultdict(lambda: defaultdict(dict))
    
    for eval_file in eval_files:
        mode, distance, level = parse_filename(eval_file.name)
        
        if mode and distance and level:
            cm, metrics = load_confusion_matrix(eval_file)
            data_dict[distance][level][mode] = (cm, metrics)
            print(f"Loaded: {distance}/{level}/{mode}")
    
    # Create output directory
    output_dir = Path('results/domain_analysis/summary/png/confusion_matrices')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot for each distance metric
    distances = ['dtw', 'mmd', 'wasserstein']
    levels = ['high', 'middle', 'low']
    modes = ['pooled', 'source_only', 'target_only']
    
    for distance in distances:
        if distance not in data_dict:
            print(f"Warning: No data for {distance}")
            continue
        
        # Create a 3x3 grid (levels x modes)
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        fig.suptitle(f'Confusion Matrices - {distance.upper()}', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        for i, level in enumerate(levels):
            for j, mode in enumerate(modes):
                ax = axes[i, j]
                
                if level in data_dict[distance] and mode in data_dict[distance][level]:
                    cm, metrics = data_dict[distance][level][mode]
                    title = f"{level.upper()} - {mode.replace('_', ' ').title()}"
                    plot_confusion_matrix(cm, title, ax, metrics)
                else:
                    ax.text(0.5, 0.5, 'No Data', 
                           ha='center', va='center', fontsize=12)
                    ax.set_title(f"{level.upper()} - {mode.replace('_', ' ').title()}")
                    ax.axis('off')
        
        plt.tight_layout()
        output_file = output_dir / f'confusion_matrices_{distance}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()
    
    # Create a combined overview (all distances for pooled only)
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('Confusion Matrices - All Distances (Pooled Mode)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    for i, distance in enumerate(distances):
        for j, level in enumerate(levels):
            ax = axes[i, j]
            mode = 'pooled'
            
            if (distance in data_dict and 
                level in data_dict[distance] and 
                mode in data_dict[distance][level]):
                cm, metrics = data_dict[distance][level][mode]
                title = f"{distance.upper()} - {level.upper()}"
                plot_confusion_matrix(cm, title, ax, metrics)
            else:
                ax.text(0.5, 0.5, 'No Data', 
                       ha='center', va='center', fontsize=12)
                ax.set_title(f"{distance.upper()} - {level.upper()}")
                ax.axis('off')
    
    plt.tight_layout()
    output_file = output_dir / 'confusion_matrices_pooled_overview.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()
    
    print(f"\n✓ All confusion matrices saved to {output_dir}")

if __name__ == '__main__':
    main()
