#!/usr/bin/env python3
"""Visualize 18-case domain analysis results (source_only vs target_only)."""

import sys
import os
import logging
import pandas as pd
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src import config as cfg
from src.utils.visualization.visualization import plot_grouped_bar_chart_raw

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def main():
    csv_path = "./results/domain_analysis/summary/csv/summary_18cases.csv"
    output_path = "./results/domain_analysis/summary/png/summary_18cases_bar.png"
    
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows from {csv_path}")
    
    # Filter to source_only and target_only
    df = df[df['mode'].isin(['source_only', 'target_only'])].copy()
    logger.info(f"Filtered to source_only/target_only: {len(df)} rows")
    
    # Calculate mean positive rate for baseline
    mean_pos = df['pos_rate'].mean()
    logger.info(f"Using mean pos_rate: {mean_pos:.4f}")
    
    # Metrics to plot
    metrics = ['auc', 'auc_pr', 'f1', 'accuracy', 'precision', 'recall']
    modes = ['source_only', 'target_only']
    
    logger.info(f"Processing modes: {modes}")
    
    # Create visualization with 3 rows (one per distance metric)
    fig = plot_grouped_bar_chart_raw(
        data=df,
        metrics=metrics,
        modes=modes,
        distance_col='distance',
        level_col='level',
        mode_col='mode',
        figsize=(6 * len(metrics), 3 * 3),  # 3 rows for DTW, MMD, Wasserstein
        baseline_rates={'auc_pr': mean_pos},
        title_map={
            'auc': 'AUROC',
            'auc_pr': 'AUPRC',
            'f1': 'F1 Score',
            'accuracy': 'Accuracy',
            'precision': 'Precision',
            'recall': 'Recall',
        }
    )
    
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved bar plot → {output_path}")
    logger.info("Bar plot successfully generated.")

if __name__ == "__main__":
    main()
