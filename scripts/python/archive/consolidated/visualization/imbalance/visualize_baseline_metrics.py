#!/usr/bin/env python3
"""Visualize baseline evaluation metrics.

This script reads evaluation results from the latest baseline job
and generates bar charts for key metrics (AUC, F1, Recall, etc.).
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src import config as cfg

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def load_latest_baseline_results(model: str = "RF", mode: str = "pooled") -> dict:
    """Load evaluation results from the latest baseline job.
    
    Parameters
    ----------
    model : str
        Model name (e.g., "RF", "Lstm")
    mode : str
        Evaluation mode (e.g., "pooled")
    
    Returns
    -------
    dict
        Evaluation metrics dictionary
    """
    eval_dir = os.path.join(cfg.RESULTS_EVALUATION_PATH, model)
    
    # Read latest job ID
    latest_file = os.path.join(eval_dir, "latest_job.txt")
    if not os.path.exists(latest_file):
        raise FileNotFoundError(f"No latest_job.txt found in {eval_dir}")
    
    with open(latest_file, 'r') as f:
        job_id = f.read().strip()
    
    logger.info(f"Loading results from job: {job_id}")
    
    # Find result JSON file
    job_dir = os.path.join(eval_dir, job_id)
    result_files = list(Path(job_dir).rglob("eval_result*.json"))
    
    if not result_files:
        raise FileNotFoundError(f"No eval_result*.json found in {job_dir}")
    
    # Load the first result file (there should be only one for baseline)
    result_file = result_files[0]
    logger.info(f"Loading: {result_file}")
    
    with open(result_file, 'r') as f:
        results = json.load(f)
    
    return results


def plot_baseline_metrics(results: dict, output_path: str):
    """Generate bar chart of baseline metrics.
    
    Parameters
    ----------
    results : dict
        Evaluation metrics dictionary
    output_path : str
        Path to save the plot
    """
    # Extract metrics
    metrics = {
        'AUC': results.get('auc', 0),
        'F1': results.get('f1', 0),
        'F2': results.get('f2', 0),
        'Recall': results.get('recall', 0),
        'Precision': results.get('precision', 0),
        'Specificity': results.get('specificity', 0),
        'Accuracy': results.get('accuracy', 0),
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = list(metrics.keys())
    values = list(metrics.values())
    
    bars = ax.bar(names, values, color='steelblue', alpha=0.8)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    ax.set_ylim([0, 1.0])
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Baseline Evaluation Metrics (Mode: {results.get("mode", "pooled")})', 
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot to: {output_path}")
    plt.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Visualize baseline evaluation metrics')
    parser.add_argument('--model', type=str, default='RF', help='Model name')
    parser.add_argument('--mode', type=str, default='pooled', help='Evaluation mode')
    parser.add_argument('--output', type=str, default=None, 
                        help='Output path (default: results/evaluation/<model>/baseline_metrics.png)')
    args = parser.parse_args()
    
    # Load results
    try:
        results = load_latest_baseline_results(model=args.model, mode=args.mode)
    except FileNotFoundError as e:
        logger.error(f"Failed to load results: {e}")
        return 1
    
    # Determine output path
    if args.output is None:
        output_path = os.path.join(cfg.RESULTS_EVALUATION_PATH, args.model, 
                                    "baseline_metrics.png")
    else:
        output_path = args.output
    
    # Generate plot
    plot_baseline_metrics(results, output_path)
    
    logger.info("Baseline visualization complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
