#!/usr/bin/env python3
"""
Visualize SMOTE Comparison Results

Generates publication-ready figures comparing SMOTE methods:
1. Bar charts comparing methods across metrics
2. Heatmaps for ranking × SMOTE method combinations
3. Box plots showing performance distribution

Usage:
    python scripts/python/visualization/plot_smote_comparison.py \
        --input results/analysis/imbalance/smote_comparison/aggregated_results.csv \
        --output-imbalance results/analysis/imbalance/smote_comparison/figures/ \
        --output-domain results/analysis/domain/imbalance/smote_comparison/figures/
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

# Setup path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

# Plotting imports
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# Set style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# Color palettes
SMOTE_COLORS = {
    "subject_wise_smote": "#2ecc71",  # Green
    "simple_smote": "#3498db",         # Blue
    "smote_balanced_rf": "#e74c3c",    # Red
}

SMOTE_LABELS = {
    "subject_wise_smote": "Subject-wise SMOTE",
    "simple_smote": "Simple SMOTE",
    "smote_balanced_rf": "SMOTE + BalancedRF",
}

RANKING_COLORS = {
    "knn": "#9b59b6",  # Purple
    "lof": "#f39c12",  # Orange
    "none": "#95a5a6", # Gray
}

KEY_METRICS = ["test_f1", "test_accuracy", "test_precision", "test_recall", "test_auc"]
METRIC_LABELS = {
    "test_f1": "F1 Score",
    "test_accuracy": "Accuracy",
    "test_precision": "Precision",
    "test_recall": "Recall",
    "test_auc": "AUC-ROC",
    "test_balanced_accuracy": "Balanced Accuracy",
}


def load_results(input_path: str) -> pd.DataFrame:
    """Load aggregated results CSV."""
    df = pd.read_csv(input_path)
    logging.info(f"Loaded {len(df)} records from {input_path}")
    return df


def plot_pooled_comparison_bars(df: pd.DataFrame, output_dir: Path, metrics: List[str] = None):
    """Bar chart comparing SMOTE methods in pooled mode."""
    pooled = df[df["mode"] == "pooled"].copy()
    
    if pooled.empty:
        logging.warning("No pooled mode data for bar chart")
        return
    
    if metrics is None:
        metrics = [m for m in KEY_METRICS if m in pooled.columns]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    methods = pooled["smote_method"].unique()
    x = np.arange(len(metrics))
    width = 0.25
    
    for i, method in enumerate(methods):
        subset = pooled[pooled["smote_method"] == method]
        means = [subset[m].mean() for m in metrics]
        stds = [subset[m].std() for m in metrics]
        
        color = SMOTE_COLORS.get(method, "#7f8c8d")
        label = SMOTE_LABELS.get(method, method)
        
        bars = ax.bar(
            x + i * width - width,
            means,
            width,
            yerr=stds,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            capsize=3,
        )
    
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title("SMOTE Methods Comparison (Pooled Mode)")
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS.get(m, m) for m in metrics], rotation=15, ha="right")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    
    plt.tight_layout()
    fig.savefig(output_dir / "pooled_comparison_bars.png")
    fig.savefig(output_dir / "pooled_comparison_bars.pdf")
    plt.close(fig)
    logging.info(f"Saved: {output_dir / 'pooled_comparison_bars.png'}")


def plot_ranking_heatmap(df: pd.DataFrame, output_dir: Path, metric: str = "test_f1"):
    """Heatmap showing ranking × SMOTE method × domain level performance."""
    ranking_df = df[df["ranking"].isin(["knn", "lof"])].copy()
    
    if ranking_df.empty:
        logging.warning("No ranking-based data for heatmap")
        return
    
    if metric not in ranking_df.columns:
        logging.warning(f"Metric {metric} not found in data")
        return
    
    # Create pivot for heatmap
    pivot = ranking_df.pivot_table(
        index=["ranking", "domain_level"],
        columns=["smote_method", "mode"],
        values=metric,
        aggfunc="mean",
    )
    
    if pivot.empty:
        logging.warning("Empty pivot table for heatmap")
        return
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Create heatmap
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        center=0.5,
        vmin=0.3,
        vmax=0.9,
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": METRIC_LABELS.get(metric, metric)},
    )
    
    ax.set_title(f"Ranking × SMOTE Method Performance ({METRIC_LABELS.get(metric, metric)})")
    ax.set_xlabel("SMOTE Method × Mode")
    ax.set_ylabel("Ranking × Domain Level")
    
    plt.tight_layout()
    fig.savefig(output_dir / f"ranking_heatmap_{metric}.png")
    fig.savefig(output_dir / f"ranking_heatmap_{metric}.pdf")
    plt.close(fig)
    logging.info(f"Saved: {output_dir / f'ranking_heatmap_{metric}.png'}")


def plot_boxplots(df: pd.DataFrame, output_dir: Path, metric: str = "test_f1"):
    """Box plots showing score distribution by SMOTE method."""
    if metric not in df.columns:
        logging.warning(f"Metric {metric} not found in data")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: By SMOTE method (all modes)
    ax1 = axes[0]
    order = list(SMOTE_LABELS.keys())
    available_order = [m for m in order if m in df["smote_method"].unique()]
    
    palette = [SMOTE_COLORS.get(m, "#7f8c8d") for m in available_order]
    
    sns.boxplot(
        data=df,
        x="smote_method",
        y=metric,
        order=available_order,
        palette=palette,
        ax=ax1,
    )
    sns.stripplot(
        data=df,
        x="smote_method",
        y=metric,
        order=available_order,
        color="black",
        alpha=0.4,
        size=4,
        ax=ax1,
    )
    
    ax1.set_xticklabels([SMOTE_LABELS.get(m, m) for m in available_order], rotation=15, ha="right")
    ax1.set_xlabel("SMOTE Method")
    ax1.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax1.set_title(f"Performance Distribution by SMOTE Method")
    ax1.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    
    # Plot 2: By mode
    ax2 = axes[1]
    mode_order = ["pooled", "source_only", "target_only"]
    available_modes = [m for m in mode_order if m in df["mode"].unique()]
    
    sns.boxplot(
        data=df,
        x="mode",
        y=metric,
        hue="smote_method",
        order=available_modes,
        hue_order=available_order,
        palette=SMOTE_COLORS,
        ax=ax2,
    )
    
    ax2.set_xlabel("Mode")
    ax2.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax2.set_title(f"Performance by Mode and SMOTE Method")
    ax2.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax2.legend(
        title="SMOTE Method",
        labels=[SMOTE_LABELS.get(m, m) for m in available_order],
        loc="upper right",
    )
    
    plt.tight_layout()
    fig.savefig(output_dir / f"boxplots_{metric}.png")
    fig.savefig(output_dir / f"boxplots_{metric}.pdf")
    plt.close(fig)
    logging.info(f"Saved: {output_dir / f'boxplots_{metric}.png'}")


def plot_ranking_comparison(df: pd.DataFrame, output_dir: Path, metric: str = "test_f1"):
    """Compare KNN vs LOF rankings for each SMOTE method."""
    ranking_df = df[df["ranking"].isin(["knn", "lof"])].copy()
    
    if ranking_df.empty:
        logging.warning("No ranking data for comparison plot")
        return
    
    if metric not in ranking_df.columns:
        logging.warning(f"Metric {metric} not found in data")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot for each domain level
    for ax, domain in zip(axes, ["out_domain", "in_domain"]):
        subset = ranking_df[ranking_df["domain_level"] == domain]
        
        if subset.empty:
            ax.set_title(f"No data for {domain}")
            continue
        
        methods = subset["smote_method"].unique()
        x = np.arange(len(methods))
        width = 0.35
        
        for i, ranking in enumerate(["knn", "lof"]):
            data = subset[subset["ranking"] == ranking]
            means = [data[data["smote_method"] == m][metric].mean() for m in methods]
            stds = [data[data["smote_method"] == m][metric].std() for m in methods]
            
            color = RANKING_COLORS.get(ranking, "#7f8c8d")
            offset = -width/2 if i == 0 else width/2
            
            ax.bar(
                x + offset,
                means,
                width,
                yerr=stds,
                label=ranking.upper(),
                color=color,
                edgecolor="black",
                linewidth=0.5,
                capsize=3,
            )
        
        ax.set_xlabel("SMOTE Method")
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.set_title(f"Ranking Comparison ({domain.replace('_', ' ').title()})")
        ax.set_xticks(x)
        ax.set_xticklabels([SMOTE_LABELS.get(m, m) for m in methods], rotation=15, ha="right")
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    fig.savefig(output_dir / f"ranking_comparison_{metric}.png")
    fig.savefig(output_dir / f"ranking_comparison_{metric}.pdf")
    plt.close(fig)
    logging.info(f"Saved: {output_dir / f'ranking_comparison_{metric}.png'}")


def plot_multi_metric_radar(df: pd.DataFrame, output_dir: Path):
    """Radar chart comparing SMOTE methods across multiple metrics."""
    pooled = df[df["mode"] == "pooled"].copy()
    
    if pooled.empty:
        logging.warning("No pooled data for radar chart")
        return
    
    metrics = [m for m in KEY_METRICS if m in pooled.columns]
    if len(metrics) < 3:
        logging.warning("Need at least 3 metrics for radar chart")
        return
    
    methods = pooled["smote_method"].unique()
    
    # Prepare data
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    for method in methods:
        subset = pooled[pooled["smote_method"] == method]
        values = [subset[m].mean() for m in metrics]
        values += values[:1]  # Close the polygon
        
        color = SMOTE_COLORS.get(method, "#7f8c8d")
        label = SMOTE_LABELS.get(method, method)
        
        ax.plot(angles, values, "o-", linewidth=2, color=color, label=label)
        ax.fill(angles, values, alpha=0.25, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([METRIC_LABELS.get(m, m) for m in metrics])
    ax.set_ylim(0, 1)
    ax.set_title("Multi-Metric Comparison (Pooled Mode)", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    fig.savefig(output_dir / "radar_chart.png")
    fig.savefig(output_dir / "radar_chart.pdf")
    plt.close(fig)
    logging.info(f"Saved: {output_dir / 'radar_chart.png'}")


def main():
    parser = argparse.ArgumentParser(description="Visualize SMOTE comparison results")
    parser.add_argument(
        "--input", "-i",
        default="results/analysis/imbalance/smote_comparison/aggregated_results.csv",
        help="Input CSV path",
    )
    parser.add_argument(
        "--output-imbalance",
        default="results/analysis/imbalance/smote_comparison/figures/",
        help="Output directory for imbalance-only (pooled) figures",
    )
    parser.add_argument(
        "--output-domain",
        default="results/analysis/domain/imbalance/smote_comparison/figures/",
        help="Output directory for domain (ranking-based) figures",
    )
    parser.add_argument(
        "--metric", "-m",
        default="test_f1",
        help="Primary metric for detailed plots",
    )
    args = parser.parse_args()
    
    # Load data
    input_path = PROJECT_ROOT / args.input
    if not input_path.exists():
        logging.error(f"Input file not found: {input_path}")
        logging.info("Run aggregate_smote_results.py first.")
        sys.exit(1)
    
    df = load_results(input_path)
    
    # Create output directories
    output_imbalance = PROJECT_ROOT / args.output_imbalance
    output_domain = PROJECT_ROOT / args.output_domain
    output_imbalance.mkdir(parents=True, exist_ok=True)
    output_domain.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("Generating SMOTE Comparison Visualizations")
    print("=" * 60)
    
    # Generate plots for imbalance-only (pooled mode)
    print("\n[Imbalance-only figures]")
    plot_pooled_comparison_bars(df, output_imbalance)
    plot_multi_metric_radar(df, output_imbalance)
    
    # Generate plots for domain analysis (ranking-based)
    print("\n[Domain analysis figures]")
    plot_ranking_heatmap(df, output_domain, metric=args.metric)
    plot_ranking_comparison(df, output_domain, metric=args.metric)
    
    # Boxplots go to both (show all data)
    print("\n[Combined figures]")
    plot_boxplots(df, output_imbalance, metric=args.metric)
    plot_boxplots(df, output_domain, metric=args.metric)
    
    print("\n" + "=" * 60)
    print(f"Imbalance figures saved to: {output_imbalance}")
    print(f"Domain figures saved to:    {output_domain}")
    print("=" * 60)


if __name__ == "__main__":
    main()
