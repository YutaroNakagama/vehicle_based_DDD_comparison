"""Performance Results Visualization for Imbalance Methods.

This module provides visualization functions to compare performance metrics
across different imbalanced data handling methods.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import logging
logger = logging.getLogger(__name__)


def create_performance_results_data() -> pd.DataFrame:
    """Create DataFrame with performance results from experiments.
    
    Data extracted from HPC job logs (imbalance_comparison_v2).
    
    Returns
    -------
    pd.DataFrame
        DataFrame with performance metrics for each method
    """
    # Results from job logs (imbalance_comparison_v2, Dec 2024)
    # Metrics extracted from eval_results JSON files
    # Updated with all 11 methods (Dec 3, 2025)
    data = [
        # method, jobid, test_recall, test_precision, test_f1, test_accuracy, auprc, auroc, f2_thr
        ("Baseline", "14489461", 0.4481, 0.0389, 0.0716, 0.5452, 0.0389, 0.5027, 0.0543),
        ("SMOTE", "14489462", 0.3278, 0.0441, 0.0777, 0.6956, 0.0465, 0.5374, 0.1433),
        ("SMOTE+Tomek", "14489463", 0.5083, 0.0433, 0.0799, 0.5418, 0.0425, 0.5379, 0.1739),
        ("SMOTE+ENN", "14489464", 0.2490, 0.0484, 0.0810, 0.7791, 0.0456, 0.5415, 0.1358),
        ("SMOTE+RUS", "14489465", 0.2905, 0.0465, 0.0801, 0.7391, 0.0476, 0.5389, 0.1419),
        ("BalancedRF", "14489466", 0.9834, 0.0393, 0.0756, 0.0592, 0.0415, 0.5205, 0.1703),
        ("EasyEnsemble", "14489467", 0.5187, 0.0405, 0.0751, 0.5004, 0.0397, 0.5065, 0.1687),
        # New methods (Dec 3, 2025)
        ("Undersample-ENN", "14545817", 0.4212, 0.0376, 0.0691, 0.5488, 0.0402, 0.5088, 0.1386),
        ("Undersample-RUS", "14545818", 0.0000, 0.0000, 0.0000, 0.9609, 0.0416, 0.5127, 0.1057),
        ("Undersample-Tomek", "14545819", 0.4066, 0.0350, 0.0645, 0.5320, 0.0414, 0.5225, 0.1175),
        ("Jitter+Scale", "14545820", 0.0021, 0.0769, 0.0041, 0.9604, 0.0406, 0.5135, 0.0000),
    ]
    
    df = pd.DataFrame(data, columns=[
        "method", "jobid", 
        "test_recall", "test_precision", "test_f1", "test_accuracy",
        "auprc", "auroc", "f2_thr"
    ])
    
    # Calculate F2 score (beta=2, recall-weighted)
    # F2 = (1 + 2^2) * (precision * recall) / (2^2 * precision + recall)
    df["test_f2"] = (5 * df["test_precision"] * df["test_recall"]) / (4 * df["test_precision"] + df["test_recall"])
    df["test_f2"] = df["test_f2"].fillna(0)
    
    return df


def plot_recall_comparison(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 6),
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Create bar chart comparing Recall across methods.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with performance data
    figsize : tuple
        Figure size
    output_path : Path, optional
        If provided, save figure to this path
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    df_sorted = df.sort_values("test_recall", ascending=True)
    
    # Color by recall level
    colors = []
    for recall in df_sorted["test_recall"]:
        if recall >= 0.4:
            colors.append("#2ecc71")  # Green - good
        elif recall >= 0.2:
            colors.append("#f39c12")  # Orange - moderate
        else:
            colors.append("#e74c3c")  # Red - poor
    
    bars = ax.barh(df_sorted["method"], df_sorted["test_recall"] * 100, color=colors, edgecolor="black")
    
    # Add value labels
    for bar, recall in zip(bars, df_sorted["test_recall"]):
        label = f"{recall*100:.1f}%"
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                label, va="center", fontsize=10)
    
    ax.set_xlabel("Test Recall (%)", fontsize=12)
    ax.set_title("Recall Comparison: Drowsy Detection Rate", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 110)
    ax.axvline(50, color="gray", linestyle="--", alpha=0.5, label="50% target")
    ax.legend(loc="lower right")
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {output_path}")
    
    return fig


def plot_precision_recall_scatter(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 8),
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Create scatter plot of Precision vs Recall.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with performance data
    figsize : tuple
        Figure size
    output_path : Path, optional
        If provided, save figure to this path
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(df)))
    
    for i, (_, row) in enumerate(df.iterrows()):
        # Default threshold point
        ax.scatter(row["test_recall"] * 100, row["test_precision"] * 100,
                   c=[colors[i]], s=150, marker="o", edgecolors="black", linewidths=1.5,
                   label=row["method"], zorder=3)
        
        # Add method label
        offset_x = 2 if row["test_recall"] < 0.3 else -15
        ax.annotate(row["method"], 
                    (row["test_recall"] * 100, row["test_precision"] * 100),
                    xytext=(offset_x, 5), textcoords="offset points", fontsize=8)
    
    ax.set_xlabel("Recall (%)", fontsize=12)
    ax.set_ylabel("Precision (%)", fontsize=12)
    ax.set_title("Precision vs Recall Trade-off (Test Set)", fontsize=14, fontweight="bold")
    ax.set_xlim(-5, 105)
    ax.set_ylim(-0.5, 6)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {output_path}")
    
    return fig


def plot_metrics_radar(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 10),
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Create radar chart comparing multiple metrics.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with performance data
    figsize : tuple
        Figure size
    output_path : Path, optional
        If provided, save figure to this path
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    # Select methods with non-zero recall for radar chart
    df_valid = df[df["test_recall"] > 0].copy()
    
    if len(df_valid) == 0:
        logger.warning("No valid methods for radar chart")
        return None
    
    # Normalize metrics to 0-1 scale for radar chart
    metrics = ["test_recall", "test_precision", "test_f1", "auprc"]
    metric_labels = ["Recall", "Precision", "F1", "AUPRC"]
    
    # Number of metrics
    num_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(df_valid)))
    
    for i, (_, row) in enumerate(df_valid.iterrows()):
        values = [row[m] for m in metrics]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, "o-", linewidth=2, label=row["method"], color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_title("Performance Metrics Comparison (Radar)", fontsize=14, fontweight="bold", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {output_path}")
    
    return fig


def plot_performance_summary_dashboard(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (16, 12),
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Create comprehensive performance dashboard.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with performance data
    figsize : tuple
        Figure size
    output_path : Path, optional
        If provided, save figure to this path
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
    
    # Panel 1: Recall comparison (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    df_sorted = df.sort_values("test_recall", ascending=True)
    colors = ["#2ecc71" if r >= 0.4 else "#f39c12" if r >= 0.1 else "#e74c3c" 
              for r in df_sorted["test_recall"]]
    bars = ax1.barh(df_sorted["method"], df_sorted["test_recall"] * 100, color=colors, edgecolor="black")
    ax1.set_xlabel("Recall (%)")
    ax1.set_title("Recall (Drowsy Detection Rate)", fontweight="bold")
    ax1.set_xlim(0, 110)
    for bar, val in zip(bars, df_sorted["test_recall"]):
        ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f"{val*100:.1f}%", va="center", fontsize=9)
    
    # Panel 2: Precision comparison (top-middle)
    ax2 = fig.add_subplot(gs[0, 1])
    df_sorted = df.sort_values("test_precision", ascending=True)
    colors = ["#2ecc71" if p >= 0.05 else "#f39c12" if p >= 0.03 else "#e74c3c"
              for p in df_sorted["test_precision"]]
    bars = ax2.barh(df_sorted["method"], df_sorted["test_precision"] * 100, color=colors, edgecolor="black")
    ax2.set_xlabel("Precision (%)")
    ax2.set_title("Precision (Positive Predictive Value)", fontweight="bold")
    ax2.set_xlim(0, 7)
    for bar, val in zip(bars, df_sorted["test_precision"]):
        ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, f"{val*100:.2f}%", va="center", fontsize=9)
    
    # Panel 3: AUPRC comparison (top-right)
    ax3 = fig.add_subplot(gs[0, 2])
    df_sorted = df.sort_values("auprc", ascending=True)
    random_baseline = 0.039  # positive rate
    colors = ["#2ecc71" if a >= 0.045 else "#f39c12" if a >= random_baseline else "#e74c3c"
              for a in df_sorted["auprc"]]
    bars = ax3.barh(df_sorted["method"], df_sorted["auprc"], color=colors, edgecolor="black")
    ax3.axvline(random_baseline, color="gray", linestyle="--", linewidth=2, label=f"Random ({random_baseline:.1%})")
    ax3.set_xlabel("AUPRC")
    ax3.set_title("AUPRC (Area Under PR Curve)", fontweight="bold")
    ax3.set_xlim(0, 0.06)
    ax3.legend(loc="lower right", fontsize=8)
    for bar, val in zip(bars, df_sorted["auprc"]):
        ax3.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, f"{val:.4f}", va="center", fontsize=9)
    
    # Panel 4: F2 Score comparison (bottom-left)
    ax4 = fig.add_subplot(gs[1, 0])
    df_sorted = df.sort_values("test_f2", ascending=True)
    colors = ["#2ecc71" if f >= 0.15 else "#f39c12" if f >= 0.05 else "#e74c3c"
              for f in df_sorted["test_f2"]]
    bars = ax4.barh(df_sorted["method"], df_sorted["test_f2"], color=colors, edgecolor="black")
    ax4.set_xlabel("F2 Score")
    ax4.set_title("F2 Score (Recall-weighted)", fontweight="bold")
    ax4.set_xlim(0, 0.25)
    for bar, val in zip(bars, df_sorted["test_f2"]):
        ax4.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, f"{val:.3f}", va="center", fontsize=9)
    
    # Panel 5: Summary table (bottom-middle and bottom-right, spanning 2 columns)
    ax5 = fig.add_subplot(gs[1, 1:])
    ax5.axis("off")
    
    # Sort by F2 for table
    df_table = df.sort_values("test_f2", ascending=False)
    
    table_data = []
    for _, row in df_table.iterrows():
        table_data.append([
            row["method"],
            f"{row['test_recall']*100:.1f}%",
            f"{row['test_precision']*100:.2f}%",
            f"{row['test_f1']:.3f}",
            f"{row['test_f2']:.3f}",
            f"{row['auprc']:.4f}",
            f"{row['auroc']:.4f}",
        ])
    
    table = ax5.table(
        cellText=table_data,
        colLabels=["Method", "Recall", "Precision", "F1", "F2", "AUPRC", "AUROC"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.4)
    
    # Color header
    for j in range(7):
        table[(0, j)].set_facecolor("#34495e")
        table[(0, j)].set_text_props(color="white", fontweight="bold")
    
    # Highlight best F2
    table[(1, 4)].set_facecolor("#d5f5e3")
    
    ax5.set_title("Performance Summary (sorted by F2)", fontweight="bold", y=0.95)
    
    fig.suptitle("Imbalance Methods - Performance Comparison (Test Set)", 
                 fontsize=16, fontweight="bold", y=0.98)
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {output_path}")
    
    return fig


def plot_method_ranking(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 8),
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Create ranking visualization across multiple metrics.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with performance data
    figsize : tuple
        Figure size
    output_path : Path, optional
        If provided, save figure to this path
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    metrics = ["test_recall", "test_precision", "test_f2", "auprc"]
    metric_labels = ["Recall", "Precision", "F2", "AUPRC"]
    
    # Calculate ranks for each metric (higher is better)
    ranks = pd.DataFrame()
    ranks["method"] = df["method"]
    for m in metrics:
        ranks[m + "_rank"] = df[m].rank(ascending=False)
    
    # Average rank
    rank_cols = [m + "_rank" for m in metrics]
    ranks["avg_rank"] = ranks[rank_cols].mean(axis=1)
    ranks = ranks.sort_values("avg_rank")
    
    x = np.arange(len(ranks))
    width = 0.2
    
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]
    
    for i, (m, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, ranks[m + "_rank"], width, label=label, color=color, edgecolor="black")
    
    ax.set_xticks(x)
    ax.set_xticklabels(ranks["method"], rotation=45, ha="right")
    ax.set_ylabel("Rank (1 = Best)")
    ax.set_title("Method Rankings by Metric", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left")
    ax.set_ylim(0, len(df) + 1)
    ax.invert_yaxis()  # Best rank (1) at top
    
    # Add average rank annotation
    for i, (_, row) in enumerate(ranks.iterrows()):
        ax.text(i, 0.3, f"Avg: {row['avg_rank']:.1f}", ha="center", fontsize=8, fontweight="bold")
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {output_path}")
    
    return fig


def generate_performance_visualizations(output_dir: str = "results/imbalance_analysis") -> Dict[str, str]:
    """Generate all performance visualizations.
    
    Parameters
    ----------
    output_dir : str
        Output directory for figures
        
    Returns
    -------
    dict
        Dictionary mapping figure names to file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create data
    df = create_performance_results_data()
    
    saved_files = {}
    
    # Generate visualizations
    fig1 = plot_recall_comparison(df, output_path=output_path / "recall_comparison.png")
    plt.close(fig1)
    saved_files["recall_comparison"] = str(output_path / "recall_comparison.png")
    
    fig2 = plot_precision_recall_scatter(df, output_path=output_path / "precision_recall_scatter.png")
    plt.close(fig2)
    saved_files["precision_recall_scatter"] = str(output_path / "precision_recall_scatter.png")
    
    fig3 = plot_metrics_radar(df, output_path=output_path / "metrics_radar.png")
    if fig3:
        plt.close(fig3)
        saved_files["metrics_radar"] = str(output_path / "metrics_radar.png")
    
    fig4 = plot_performance_summary_dashboard(df, output_path=output_path / "performance_dashboard.png")
    plt.close(fig4)
    saved_files["performance_dashboard"] = str(output_path / "performance_dashboard.png")
    
    fig5 = plot_method_ranking(df, output_path=output_path / "method_ranking.png")
    plt.close(fig5)
    saved_files["method_ranking"] = str(output_path / "method_ranking.png")
    
    # Save CSV
    csv_path = output_path / "performance_results.csv"
    df.to_csv(csv_path, index=False)
    saved_files["csv"] = str(csv_path)
    
    print(f"\n{'='*60}")
    print("PERFORMANCE VISUALIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"\nOutput directory: {output_path}")
    print(f"\nGenerated files:")
    for name, path in saved_files.items():
        print(f"  - {name}: {path}")
    
    return saved_files


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_performance_visualizations()
