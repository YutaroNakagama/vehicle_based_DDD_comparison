"""Performance Results Visualization for Imbalance Methods.

This module provides visualization functions to compare performance metrics
across different imbalanced data handling methods.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import logging
logger = logging.getLogger(__name__)


# Full V2 model configurations: 11 methods × 3 ratios (0.1, 0.5, 1.0)
# Updated with bug-fixed job IDs (14608xxx series) for SMOTE+Tomek, SMOTE+ENN, SMOTE+RUS
DEFAULT_V2_MODELS = [
    # Baseline (RF without sampling) × 3 ratios
    ("RF", "14592990", "Baseline (0.1)", "imbal_v2_baseline_ratio0_1"),
    ("RF", "14593013", "Baseline (0.5)", "imbal_v2_baseline_ratio0_5"),
    ("RF", "14593038", "Baseline (1.0)", "imbal_v2_baseline_ratio1_0"),
    # SMOTE × 3 ratios
    ("RF", "14593005", "SMOTE (0.1)", "imbal_v2_smote_ratio0_1"),
    ("RF", "14593030", "SMOTE (0.5)", "imbal_v2_smote_ratio0_5"),
    ("RF", "14593052", "SMOTE (1.0)", "imbal_v2_smote_ratio1_0"),
    # SMOTE+Tomek × 3 ratios (FIXED: 14608427-14608429)
    ("RF", "14608427", "SMOTE+Tomek (0.1)", "imbal_v2_smote_tomek_ratio0_1_fixed"),
    ("RF", "14608428", "SMOTE+Tomek (0.5)", "imbal_v2_smote_tomek_ratio0_5_fixed"),
    ("RF", "14608429", "SMOTE+Tomek (1.0)", "imbal_v2_smote_tomek_ratio1_0_fixed"),
    # SMOTE+ENN × 3 ratios (FIXED: 14608430-14608432)
    ("RF", "14608430", "SMOTE+ENN (0.1)", "imbal_v2_smote_enn_ratio0_1_fixed"),
    ("RF", "14608431", "SMOTE+ENN (0.5)", "imbal_v2_smote_enn_ratio0_5_fixed"),
    ("RF", "14608432", "SMOTE+ENN (1.0)", "imbal_v2_smote_enn_ratio1_0_fixed"),
    # SMOTE+RUS × 3 ratios (FIXED: 14608305-14608306, 0.1 has error)
    ("RF", "14608305", "SMOTE+RUS (0.5)", "imbal_v2_smote_rus_ratio0_5_fix"),
    ("RF", "14608306", "SMOTE+RUS (1.0)", "imbal_v2_smote_rus_ratio1_0_fix"),
    # SMOTE+BalancedRF × 3 ratios
    ("BalancedRF", "14592992", "SMOTE+BalancedRF (0.1)", "imbal_v2_smote_balanced_rf_ratio0_1"),
    ("BalancedRF", "14593015", "SMOTE+BalancedRF (0.5)", "imbal_v2_smote_balanced_rf_ratio0_5"),
    ("BalancedRF", "14593040", "SMOTE+BalancedRF (1.0)", "imbal_v2_smote_balanced_rf_ratio1_0"),
    # BalancedRF × 3 ratios
    ("BalancedRF", "14593000", "BalancedRF (0.1)", "imbal_v2_balanced_rf_ratio0_1"),
    ("BalancedRF", "14593023", "BalancedRF (0.5)", "imbal_v2_balanced_rf_ratio0_5"),
    ("BalancedRF", "14593048", "BalancedRF (1.0)", "imbal_v2_balanced_rf_ratio1_0"),
    # EasyEnsemble × 3 ratios
    ("EasyEnsemble", "14592996", "EasyEnsemble (0.1)", "imbal_v2_easy_ensemble_ratio0_1"),
    ("EasyEnsemble", "14593019", "EasyEnsemble (0.5)", "imbal_v2_easy_ensemble_ratio0_5"),
    ("EasyEnsemble", "14593044", "EasyEnsemble (1.0)", "imbal_v2_easy_ensemble_ratio1_0"),
    # Undersample-ENN × 3 ratios
    ("RF", "14593003", "Undersample-ENN (0.1)", "imbal_v2_undersample_enn_ratio0_1"),
    ("RF", "14593027", "Undersample-ENN (0.5)", "imbal_v2_undersample_enn_ratio0_5"),
    ("RF", "14593050", "Undersample-ENN (1.0)", "imbal_v2_undersample_enn_ratio1_0"),
    # Undersample-RUS × 3 ratios
    ("RF", "14592994", "Undersample-RUS (0.1)", "imbal_v2_undersample_rus_ratio0_1"),
    ("RF", "14593017", "Undersample-RUS (0.5)", "imbal_v2_undersample_rus_ratio0_5"),
    ("RF", "14593042", "Undersample-RUS (1.0)", "imbal_v2_undersample_rus_ratio1_0"),
    # Undersample-Tomek × 3 ratios
    ("RF", "14592982", "Undersample-Tomek (0.1)", "imbal_v2_undersample_tomek_ratio0_1"),
    ("RF", "14593007", "Undersample-Tomek (0.5)", "imbal_v2_undersample_tomek_ratio0_5"),
    ("RF", "14593032", "Undersample-Tomek (1.0)", "imbal_v2_undersample_tomek_ratio1_0"),
]


def load_evaluation_results_from_json(
    model_type: str, 
    jobid: str, 
    tag: str,
    base_path: str = "results/evaluation"
) -> Optional[Dict]:
    """Load evaluation results from JSON file.
    
    Parameters
    ----------
    model_type : str
        Model type (RF, BalancedRF, EasyEnsemble)
    jobid : str
        Job ID
    tag : str
        Model tag (not used in current path structure)
    base_path : str
        Base path for evaluation results
        
    Returns
    -------
    dict or None
        Evaluation results or None if not found
    """
    base = Path(base_path)
    
    # Try different path patterns
    # Pattern 1: {base}/{model_type}/{jobid}/{jobid}[1]/*.json
    eval_dir = base / model_type / jobid / f"{jobid}[1]"
    
    if eval_dir.exists():
        json_files = list(eval_dir.glob("eval_results_*.json"))
        if json_files:
            eval_file = json_files[0]
            try:
                with open(eval_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading {eval_file}: {e}")
                return None
    
    # Pattern 2: {base}/{model_type}/pooled_{tag}_{jobid}/evaluation_results.json
    eval_dir = base / model_type / f"pooled_{tag}_{jobid}"
    eval_file = eval_dir / "evaluation_results.json"
    
    if eval_file.exists():
        try:
            with open(eval_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading {eval_file}: {e}")
            return None
    
    logger.warning(f"Evaluation file not found for {model_type}/{jobid}")
    return None


def create_performance_results_data(models: Optional[List[Tuple]] = None) -> pd.DataFrame:
    """Create DataFrame with performance results from evaluation JSON files.
    
    Automatically loads results from evaluation JSON files for all 33 cases
    (11 methods × 3 ratios).
    
    Parameters
    ----------
    models : list of tuples, optional
        List of (model_type, jobid, display_name, tag) tuples.
        If None, uses DEFAULT_V2_MODELS.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with performance metrics for each method
    """
    if models is None:
        models = DEFAULT_V2_MODELS
    
    data = []
    for model_type, jobid, display_name, tag in models:
        results = load_evaluation_results_from_json(model_type, jobid, tag)
        
        if results is None:
            logger.warning(f"Skipping {display_name} ({jobid}): no results found")
            continue
        
        # Extract metrics from JSON - metrics are at top level
        # F2 is stored as f2_thr (threshold-optimized)
        row = {
            "method": display_name,
            "jobid": jobid,
            "model_type": model_type,
            "tag": tag,
            "test_recall": results.get("recall", 0.0),
            "test_precision": results.get("precision", 0.0),
            "test_f1": results.get("f1", 0.0),
            "test_f2": results.get("f2_thr", results.get("f2", 0.0)),
            "test_accuracy": results.get("accuracy", 0.0),
            "auprc": results.get("auc_pr", results.get("auprc", 0.0)),
            "auroc": results.get("roc_auc", results.get("auc_roc", 0.0)),
            # Threshold-based metrics
            "test_recall_thr": results.get("recall_thr", results.get("recall", 0.0)),
            "test_precision_thr": results.get("prec_thr", results.get("precision", 0.0)),
            "test_f1_thr": results.get("f1_thr", results.get("f1", 0.0)),
        }
        data.append(row)
    
    if not data:
        logger.error("No data loaded! Check evaluation results paths.")
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=[
            "method", "jobid", "model_type", "tag",
            "test_recall", "test_precision", "test_f1", "test_f2", "test_accuracy",
            "auprc", "auroc", "test_recall_thr", "test_precision_thr", "test_f1_thr"
        ])
    
    df = pd.DataFrame(data)
    logger.info(f"Loaded {len(df)} evaluation results")
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
    ax.set_xlim(0, 60)
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
    ax.set_title("Precision vs Recall Trade-off (Test Set, thr=0.5)", fontsize=14, fontweight="bold")
    ax.set_xlim(-5, 60)
    ax.set_ylim(-0.5, 10)
    ax.grid(True, alpha=0.3)
    
    # Add annotation for baseline (recall=0)
    ax.annotate("Baseline & RUS:\nRecall=0%\n(No detection)", 
                xy=(0, 0), xytext=(5, 5),
                fontsize=9, style="italic", color="red",
                arrowprops=dict(arrowstyle="->", color="red", alpha=0.5))
    
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
    ax.set_ylim(0, 0.6)
    ax.set_title("Performance Metrics Comparison (Radar)", fontsize=14, fontweight="bold", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {output_path}")
    
    return fig


def plot_performance_summary_dashboard(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (20, 12),
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
    
    # Color scheme: Baseline = red, others = blue
    def get_colors(df_sorted):
        return ["#e74c3c" if m.startswith("Baseline") else "#3498db" for m in df_sorted["method"]]
    
    # Panel 1: Recall comparison (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    df_sorted = df.sort_values("test_recall", ascending=True)
    colors = get_colors(df_sorted)
    bars = ax1.barh(df_sorted["method"], df_sorted["test_recall"] * 100, color=colors, edgecolor="black")
    ax1.set_xlabel("Recall (%)")
    ax1.set_title("Recall (Drowsy Detection Rate)", fontweight="bold")
    ax1.set_xlim(0, 100)
    for bar, val in zip(bars, df_sorted["test_recall"]):
        ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f"{val*100:.0f}%", va="center", fontsize=9)
    
    # Panel 2: Precision comparison (top-middle)
    ax2 = fig.add_subplot(gs[0, 1])
    df_sorted = df.sort_values("test_precision", ascending=True)
    colors = get_colors(df_sorted)
    bars = ax2.barh(df_sorted["method"], df_sorted["test_precision"] * 100, color=colors, edgecolor="black")
    ax2.set_xlabel("Precision (%)")
    ax2.set_title("Precision (False Alarm Rate)", fontweight="bold")
    ax2.set_xlim(0, 10)
    for bar, val in zip(bars, df_sorted["test_precision"]):
        ax2.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2, f"{val*100:.1f}%", va="center", fontsize=9)
    
    # Panel 3: AUPRC comparison (top-right) - key metric for imbalanced anomaly detection
    ax3 = fig.add_subplot(gs[0, 2])
    df_sorted = df.sort_values("auprc", ascending=True)
    colors = get_colors(df_sorted)
    # Random classifier baseline = positive rate (~0.039)
    random_baseline = 0.039
    bars = ax3.barh(df_sorted["method"], df_sorted["auprc"], color=colors, edgecolor="black")
    ax3.axvline(random_baseline, color="gray", linestyle="--", linewidth=2, label=f"Random ({random_baseline:.1%})")
    ax3.set_xlabel("AUPRC")
    ax3.set_title("AUPRC (Primary Metric)", fontweight="bold")
    ax3.set_xlim(0, 0.10)
    ax3.legend(loc="lower right", fontsize=8)
    for bar, val in zip(bars, df_sorted["auprc"]):
        ax3.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2, f"{val:.3f}", va="center", fontsize=9)
    
    # Panel 4: F2 Score comparison (bottom-left)
    ax4 = fig.add_subplot(gs[1, 0])
    df_sorted = df.sort_values("test_f2", ascending=True)
    colors = get_colors(df_sorted)
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
            f"{row['test_recall']*100:.0f}%",
            f"{row['test_precision']*100:.1f}%",
            f"{row['test_f1']:.3f}",
            f"{row['test_f2']:.3f}",
            f"{row['auprc']:.3f}",
        ])
    
    table = ax5.table(
        cellText=table_data,
        colLabels=["Method", "Recall", "Precision", "F1", "F2", "AUPRC"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.4)
    
    # Color header
    for j in range(6):
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


def plot_single_metric(
    df: pd.DataFrame,
    metric: str,
    title: str,
    xlabel: str,
    figsize: Tuple[int, int] = (12, 10),
    xlim: Optional[Tuple[float, float]] = None,
    percentage: bool = False,
    reference_line: Optional[float] = None,
    reference_label: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Create a single metric bar chart with large, readable format.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with performance data
    metric : str
        Column name of the metric to plot
    title : str
        Chart title
    xlabel : str
        X-axis label
    figsize : tuple
        Figure size (default: 12x10 for readability)
    xlim : tuple, optional
        X-axis limits
    percentage : bool
        If True, multiply values by 100 and show as percentage
    reference_line : float, optional
        If provided, draw a vertical reference line
    reference_label : str, optional
        Label for reference line
    output_path : Path, optional
        If provided, save figure to this path
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort by metric value
    df_sorted = df.sort_values(metric, ascending=True)
    
    # Color scheme: Baseline = red, others = blue
    colors = ["#e74c3c" if m.startswith("Baseline") else "#3498db" for m in df_sorted["method"]]
    
    # Get values
    values = df_sorted[metric]
    if percentage:
        values = values * 100
    
    # Create horizontal bar chart
    bars = ax.barh(df_sorted["method"], values, color=colors, edgecolor="black", height=0.7)
    
    # Add reference line if specified
    if reference_line is not None:
        ref_val = reference_line * 100 if percentage else reference_line
        label = reference_label or f"Reference ({ref_val:.1f})"
        ax.axvline(ref_val, color="gray", linestyle="--", linewidth=2, label=label)
        ax.legend(loc="lower right", fontsize=12)
    
    # Add value labels
    for bar, val in zip(bars, values):
        if percentage:
            label = f"{val:.1f}%"
        else:
            label = f"{val:.3f}"
        ax.text(bar.get_width() + (xlim[1] - xlim[0]) * 0.02 if xlim else bar.get_width() * 0.02, 
                bar.get_y() + bar.get_height()/2, 
                label, va="center", fontsize=11, fontweight="bold")
    
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    ax.tick_params(axis='y', labelsize=11)
    ax.tick_params(axis='x', labelsize=11)
    
    if xlim:
        ax.set_xlim(xlim)
    
    # Add grid for readability
    ax.xaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {output_path}")
    
    return fig


def plot_recall_standalone(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 10),
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Create standalone Recall chart."""
    return plot_single_metric(
        df=df,
        metric="test_recall",
        title="Recall (Drowsy Detection Rate)",
        xlabel="Recall (%)",
        figsize=figsize,
        xlim=(0, 100),
        percentage=True,
        output_path=output_path,
    )


def plot_precision_standalone(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 10),
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Create standalone Precision chart."""
    return plot_single_metric(
        df=df,
        metric="test_precision",
        title="Precision (Positive Predictive Value)",
        xlabel="Precision (%)",
        figsize=figsize,
        xlim=(0, 15),
        percentage=True,
        output_path=output_path,
    )


def plot_f2_standalone(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 10),
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Create standalone F2 Score chart."""
    return plot_single_metric(
        df=df,
        metric="test_f2",
        title="F2 Score (Recall-weighted Harmonic Mean)",
        xlabel="F2 Score",
        figsize=figsize,
        xlim=(0, 0.30),
        percentage=False,
        output_path=output_path,
    )


def plot_auprc_standalone(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 10),
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Create standalone AUPRC chart."""
    # Random classifier baseline = positive rate (~0.039)
    random_baseline = 0.039
    return plot_single_metric(
        df=df,
        metric="auprc",
        title="AUPRC (Area Under Precision-Recall Curve)",
        xlabel="AUPRC",
        figsize=figsize,
        xlim=(0, 0.12),
        percentage=False,
        reference_line=random_baseline,
        reference_label=f"Random Baseline ({random_baseline:.1%})",
        output_path=output_path,
    )


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


def generate_performance_visualizations(output_dir: str = "results/imbalance/analysis") -> Dict[str, str]:
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
    
    # Individual metric plots (larger, more readable)
    fig_recall = plot_recall_standalone(df, output_path=output_path / "metric_recall.png")
    plt.close(fig_recall)
    saved_files["metric_recall"] = str(output_path / "metric_recall.png")
    
    fig_precision = plot_precision_standalone(df, output_path=output_path / "metric_precision.png")
    plt.close(fig_precision)
    saved_files["metric_precision"] = str(output_path / "metric_precision.png")
    
    fig_f2 = plot_f2_standalone(df, output_path=output_path / "metric_f2.png")
    plt.close(fig_f2)
    saved_files["metric_f2"] = str(output_path / "metric_f2.png")
    
    fig_auprc = plot_auprc_standalone(df, output_path=output_path / "metric_auprc.png")
    plt.close(fig_auprc)
    saved_files["metric_auprc"] = str(output_path / "metric_auprc.png")
    
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
