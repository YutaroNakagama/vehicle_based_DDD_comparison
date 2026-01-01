#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_full_comparison.py
============================
Create comprehensive visualization comparing:
1. Imbalance handling methods (11 methods)
2. Ranking methods for domain analysis (6 methods)

Uses the same style as summary_metrics_bar_with_pooled_baseline.png

Output:
- imbalance_methods_comparison.png: Bar chart for imbalance methods
- ranking_methods_comparison.png: Bar chart for ranking methods
- full_comparison_dashboard.png: Combined dashboard
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Setup matplotlib before importing pyplot
from src.utils.visualization.setup import setup_matplotlib_headless
setup_matplotlib_headless()

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src import config as cfg
from src.utils.visualization.color_palettes import (
    IMBALANCE_METHOD_COLORS as IMBALANCE_COLORS,
    RANKING_METHOD_COLORS as RANKING_COLORS,
    TRAINING_MODE_COLORS as MODE_COLORS,
    DOMAIN_LEVEL_COLORS as LEVEL_COLORS,
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# =============================================================================
# Imbalance Methods Data (from experiments)
# =============================================================================

# Results from imbalance comparison experiments (Job 14545817-14545820)
IMBALANCE_RESULTS = {
    # Method: {F2, Recall, Precision, F1}
    "Baseline": {"f2": 0.063, "recall": 0.099, "precision": 0.030, "f1": 0.046},
    "SMOTE": {"f2": 0.156, "recall": 0.450, "precision": 0.036, "f1": 0.067},
    "SMOTE+Tomek": {"f2": 0.174, "recall": 0.508, "precision": 0.039, "f1": 0.073},
    "SMOTE+ENN": {"f2": 0.161, "recall": 0.462, "precision": 0.038, "f1": 0.070},
    "SMOTE+RUS": {"f2": 0.098, "recall": 0.253, "precision": 0.029, "f1": 0.052},
    "BalancedRF": {"f2": 0.154, "recall": 0.443, "precision": 0.036, "f1": 0.066},
    "EasyEnsemble": {"f2": 0.148, "recall": 0.424, "precision": 0.035, "f1": 0.064},
    "Undersample-ENN": {"f2": 0.087, "recall": 0.225, "precision": 0.025, "f1": 0.045},
    "Undersample-RUS": {"f2": 0.073, "recall": 0.161, "precision": 0.030, "f1": 0.050},
    "Undersample-Tomek": {"f2": 0.062, "recall": 0.096, "precision": 0.030, "f1": 0.046},
    "Jitter+Scale": {"f2": 0.077, "recall": 0.087, "precision": 0.070, "f1": 0.077},
}


# =============================================================================
# Ranking Methods Data (from Job 14552850 evaluation)
# =============================================================================

def load_ranking_results_from_logs(log_dir: str = None) -> Dict:
    """Load ranking method results from evaluation logs."""
    import re
    
    if log_dir is None:
        log_dir = "/home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/log"
    
    results = {}
    ranking_methods = ["mean_distance", "centroid_umap", "lof", "knn", "median_distance", "isolation_forest"]
    
    for i, ranking in enumerate(ranking_methods, 1):
        log_file = os.path.join(log_dir, f"14572424[{i}].spcc-adm1.OU")
        
        if not os.path.exists(log_file):
            logger.warning(f"Log file not found: {log_file}")
            continue
        
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Parse results
        lines = content.split('\n')
        current_mode = None
        current_level = None
        current_metric = None
        
        method_results = {
            "by_mode": {"pooled": [], "source_only": [], "target_only": []},
            "by_level": {"out_domain": [], "mid_domain": [], "in_domain": []},
            "all": []
        }
        
        for line in lines:
            # Extract mode and tag
            if '[EVAL] model=RF | mode=' in line:
                match = re.search(r'mode=(\w+) \| tag=full_\w+_(\w+)_(\w+)', line)
                if match:
                    current_mode = match.group(1)
                    current_metric = match.group(2)
                    current_level = match.group(3)
            
            # Extract F1 score
            if '[EVAL DONE]' in line and current_mode:
                f1_match = re.search(r'F1=(\d+\.\d+)', line)
                if f1_match:
                    f1 = float(f1_match.group(1))
                    method_results["all"].append(f1)
                    if current_mode in method_results["by_mode"]:
                        method_results["by_mode"][current_mode].append(f1)
                    if current_level in method_results["by_level"]:
                        method_results["by_level"][current_level].append(f1)
        
        results[ranking] = method_results
    
    return results


def aggregate_ranking_results(results: Dict) -> pd.DataFrame:
    """Aggregate ranking results into a DataFrame."""
    data = []
    
    for ranking, method_results in results.items():
        # Average F1 by mode
        for mode, f1_list in method_results["by_mode"].items():
            if f1_list:
                data.append({
                    "ranking_method": ranking,
                    "mode": mode,
                    "f1": np.mean(f1_list),
                    "f1_std": np.std(f1_list),
                    "n": len(f1_list)
                })
    
    return pd.DataFrame(data)


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_imbalance_comparison(output_path: str = None) -> plt.Figure:
    """Create bar chart comparing imbalance handling methods.
    
    Style matches summary_metrics_bar_with_pooled_baseline.png
    """
    methods = list(IMBALANCE_RESULTS.keys())
    metrics = ["f2", "recall", "precision", "f1"]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        values = [IMBALANCE_RESULTS[m][metric] for m in methods]
        colors = [IMBALANCE_COLORS.get(m, "#7f7f7f") for m in methods]
        
        x = np.arange(len(methods))
        bars = ax.bar(x, values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Highlight best method
        best_idx = np.argmax(values)
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)
        
        ax.set_xlabel("Imbalance Method", fontsize=11)
        ax.set_ylabel(metric.upper(), fontsize=11)
        ax.set_title(f"{metric.upper()} Score by Imbalance Method", fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Set y-axis limits
        ax.set_ylim(0, max(values) * 1.15)
    
    fig.suptitle("Imbalance Handling Methods Comparison\n(RF Model, Pooled Training)", 
                fontsize=14, fontweight='bold')
    fig.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches='tight')
        logger.info(f"Saved: {output_path}")
    
    return fig


def plot_ranking_comparison(output_path: str = None) -> plt.Figure:
    """Create bar chart comparing ranking methods.
    
    Style matches summary_metrics_bar_with_pooled_baseline.png
    """
    # Load results from logs
    results = load_ranking_results_from_logs()
    
    if not results:
        logger.error("No ranking results found")
        return None
    
    ranking_methods = list(RANKING_COLORS.keys())
    modes = ["pooled", "source_only", "target_only"]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, mode in enumerate(modes):
        ax = axes[idx]
        
        values = []
        colors = []
        labels = []
        
        for ranking in ranking_methods:
            if ranking in results:
                mode_values = results[ranking]["by_mode"].get(mode, [])
                if mode_values:
                    values.append(np.mean(mode_values))
                else:
                    values.append(0)
            else:
                values.append(0)
            colors.append(RANKING_COLORS.get(ranking, "#7f7f7f"))
            labels.append(ranking.replace("_", "\n"))
        
        x = np.arange(len(ranking_methods))
        bars = ax.bar(x, values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Highlight best method
        if values:
            best_idx = np.argmax(values)
            bars[best_idx].set_edgecolor('gold')
            bars[best_idx].set_linewidth(3)
        
        ax.set_xlabel("Ranking Method", fontsize=11)
        ax.set_ylabel("F1 Score", fontsize=11)
        ax.set_title(f"{mode.replace('_', ' ').title()}", fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        
        if values:
            ax.set_ylim(0, max(values) * 1.2)
    
    fig.suptitle("Ranking Methods Comparison (F1 Score)\n(RF Model, Domain Analysis)", 
                fontsize=14, fontweight='bold')
    fig.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches='tight')
        logger.info(f"Saved: {output_path}")
    
    return fig


def plot_ranking_by_level(output_path: str = None) -> plt.Figure:
    """Create bar chart comparing ranking methods by domain level."""
    results = load_ranking_results_from_logs()
    
    if not results:
        logger.error("No ranking results found")
        return None
    
    ranking_methods = list(RANKING_COLORS.keys())
    levels = ["out_domain", "mid_domain", "in_domain"]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(ranking_methods))
    width = 0.25
    
    for i, level in enumerate(levels):
        values = []
        for ranking in ranking_methods:
            if ranking in results:
                level_values = results[ranking]["by_level"].get(level, [])
                if level_values:
                    values.append(np.mean(level_values))
                else:
                    values.append(0)
            else:
                values.append(0)
        
        offset = (i - 1) * width
        color = LEVEL_COLORS.get(level, f"C{i}")
        bars = ax.bar(x + offset, values, width, label=level.replace("_", " ").title(), 
                     color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=90)
    
    ax.set_xlabel("Ranking Method", fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_title("Ranking Methods by Domain Level (F1 Score)", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([r.replace("_", "\n") for r in ranking_methods], fontsize=10)
    ax.legend(title="Domain Level", loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    fig.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches='tight')
        logger.info(f"Saved: {output_path}")
    
    return fig


def plot_full_dashboard(output_path: str = None) -> plt.Figure:
    """Create comprehensive dashboard combining all comparisons."""
    fig = plt.figure(figsize=(18, 12))
    
    # Create grid layout
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Top row: Imbalance methods (4 metrics in 2x2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Plot imbalance metrics
    methods = list(IMBALANCE_RESULTS.keys())
    
    for ax, metric, title in zip(
        [ax1, ax2, ax3],
        ["f2", "recall", "f1"],
        ["F2 Score", "Recall", "F1 Score"]
    ):
        values = [IMBALANCE_RESULTS[m][metric] for m in methods]
        colors = [IMBALANCE_COLORS.get(m, "#7f7f7f") for m in methods]
        
        x = np.arange(len(methods))
        bars = ax.bar(x, values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        best_idx = np.argmax(values)
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(2)
        
        ax.set_title(f"Imbalance: {title}", fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=7)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, max(values) * 1.15)
    
    # Bottom row: Ranking methods
    results = load_ranking_results_from_logs()
    ranking_methods = list(RANKING_COLORS.keys())
    
    if results:
        # By mode
        ax4 = fig.add_subplot(gs[1, 0])
        modes = ["source_only", "target_only", "pooled"]
        width = 0.25
        x = np.arange(len(ranking_methods))
        
        for i, mode in enumerate(modes):
            values = []
            for ranking in ranking_methods:
                if ranking in results:
                    mode_values = results[ranking]["by_mode"].get(mode, [])
                    values.append(np.mean(mode_values) if mode_values else 0)
                else:
                    values.append(0)
            
            offset = (i - 1) * width
            color = MODE_COLORS.get(mode, f"C{i}")
            ax4.bar(x + offset, values, width, label=mode.replace("_", " "), 
                   color=color, alpha=0.8)
        
        ax4.set_title("Ranking: F1 by Mode", fontsize=11, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels([r.replace("_", "\n") for r in ranking_methods], fontsize=8)
        ax4.legend(fontsize=8)
        ax4.grid(axis='y', alpha=0.3)
        
        # By level
        ax5 = fig.add_subplot(gs[1, 1])
        levels = ["out_domain", "mid_domain", "in_domain"]
        
        for i, level in enumerate(levels):
            values = []
            for ranking in ranking_methods:
                if ranking in results:
                    level_values = results[ranking]["by_level"].get(level, [])
                    values.append(np.mean(level_values) if level_values else 0)
                else:
                    values.append(0)
            
            offset = (i - 1) * width
            color = LEVEL_COLORS.get(level, f"C{i}")
            ax5.bar(x + offset, values, width, label=level.replace("_", " "), 
                   color=color, alpha=0.8)
        
        ax5.set_title("Ranking: F1 by Level", fontsize=11, fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels([r.replace("_", "\n") for r in ranking_methods], fontsize=8)
        ax5.legend(fontsize=8)
        ax5.grid(axis='y', alpha=0.3)
        
        # Overall ranking comparison
        ax6 = fig.add_subplot(gs[1, 2])
        avg_f1 = []
        for ranking in ranking_methods:
            if ranking in results:
                all_f1 = results[ranking]["all"]
                avg_f1.append(np.mean(all_f1) if all_f1 else 0)
            else:
                avg_f1.append(0)
        
        colors = [RANKING_COLORS.get(r, "#7f7f7f") for r in ranking_methods]
        bars = ax6.bar(x, avg_f1, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        if avg_f1:
            best_idx = np.argmax(avg_f1)
            bars[best_idx].set_edgecolor('gold')
            bars[best_idx].set_linewidth(2)
        
        ax6.set_title("Ranking: Overall F1", fontsize=11, fontweight='bold')
        ax6.set_xticks(x)
        ax6.set_xticklabels([r.replace("_", "\n") for r in ranking_methods], fontsize=8)
        ax6.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, avg_f1):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    fig.suptitle("Domain Analysis Comparison Dashboard\n(Imbalance Methods & Ranking Methods)", 
                fontsize=16, fontweight='bold')
    
    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches='tight')
        logger.info(f"Saved: {output_path}")
    
    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    """Generate all comparison visualizations."""
    out_dir = Path(cfg.RESULTS_DOMAIN_ANALYSIS_PATH) / "summary" / "png" / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Generating Comparison Visualizations")
    logger.info("=" * 60)
    
    # 1. Imbalance methods comparison
    logger.info("\n[1/4] Imbalance Methods Comparison")
    fig1 = plot_imbalance_comparison(out_dir / "imbalance_methods_comparison.png")
    plt.close(fig1)
    
    # 2. Ranking methods comparison
    logger.info("\n[2/4] Ranking Methods Comparison")
    fig2 = plot_ranking_comparison(out_dir / "ranking_methods_comparison.png")
    if fig2:
        plt.close(fig2)
    
    # 3. Ranking by level
    logger.info("\n[3/4] Ranking Methods by Level")
    fig3 = plot_ranking_by_level(out_dir / "ranking_methods_by_level.png")
    if fig3:
        plt.close(fig3)
    
    # 4. Full dashboard
    logger.info("\n[4/4] Full Comparison Dashboard")
    fig4 = plot_full_dashboard(out_dir / "full_comparison_dashboard.png")
    if fig4:
        plt.close(fig4)
    
    logger.info("\n" + "=" * 60)
    logger.info("All visualizations generated!")
    logger.info(f"Output directory: {out_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
