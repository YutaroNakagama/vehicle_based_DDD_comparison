"""Imbalance Comparison Analysis Module.

This module provides functions for analyzing and visualizing results from
imbalanced data experiments, including:
- Loading evaluation results from multiple methods
- Computing summary statistics
- Generating comparison visualizations (bar charts, heatmaps, radar plots)

The module focuses on metrics suitable for imbalanced data:
- AUPRC (PR-AUC): Threshold-independent, most reliable for imbalanced data
- F2 Score: Recall-weighted F-measure (β=2)
- AUROC: For reference, but less informative for severe imbalance
"""

import json
import glob
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.utils.io.data_io import load_csv, save_csv, load_json, save_json
from src.utils.visualization.visualization import save_figure, save_current_figure

logger = logging.getLogger(__name__)


# === Data Loading ===

def load_imbalance_results(
    results_base: str,
    models_info: List[Tuple[str, str, str, str]],
) -> pd.DataFrame:
    """Load evaluation results from imbalance comparison experiments.
    
    Parameters
    ----------
    results_base : str
        Base path for results (e.g., 'results/outputs/evaluation').
    models_info : list of tuple
        List of (model_type, jobid, display_name, tag) tuples.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: name, model, tag, jobid, and all metric columns.
    
    Examples
    --------
    >>> models = [
    ...     ("RF", "14468417", "Baseline RF", "imbal_v2_baseline"),
    ...     ("RF", "14468421", "SMOTE+RUS", "imbal_v2_smote_rus"),
    ... ]
    >>> df = load_imbalance_results("results/outputs/evaluation", models)
    """
    results = []
    
    for model_type, jobid, display_name, tag in models_info:
        pattern = f"{results_base}/{model_type}/{jobid}/**/*.json"
        files = glob.glob(pattern, recursive=True)
        
        if not files:
            logger.warning(f"No results found for {display_name} (jobid={jobid})")
            continue
        
        # Get the first/latest result file
        for f in files:
            try:
                with open(f, 'r') as file:
                    data = json.load(file)
                
                result = {
                    'name': display_name,
                    'model': model_type,
                    'tag': tag,
                    'jobid': jobid,
                    'file': f,
                    # Core metrics
                    'auprc': data.get('auc_pr', data.get('auprc', np.nan)),
                    'auroc': data.get('auc', data.get('auroc', np.nan)),
                    'accuracy': data.get('accuracy', np.nan),
                    'precision': data.get('precision', np.nan),
                    'recall': data.get('recall', np.nan),
                    'f1': data.get('f1', np.nan),
                    'f2': data.get('f2', np.nan),
                    'specificity': data.get('specificity', np.nan),
                    # Threshold-optimized metrics
                    'thr': data.get('thr', 0.5),
                    'recall_thr': data.get('recall_thr', np.nan),
                    'precision_thr': data.get('prec_thr', np.nan),
                    'f1_thr': data.get('f1_thr', np.nan),
                    'f2_thr': data.get('f2_thr', np.nan),
                    'specificity_thr': data.get('specificity_thr', np.nan),
                }
                results.append(result)
                break  # Only take first file per model
                
            except Exception as e:
                logger.warning(f"Failed to load {f}: {e}")
                continue
    
    return pd.DataFrame(results)


def load_ensemble_results(
    ensemble_dir: str,
) -> pd.DataFrame:
    """Load ensemble evaluation results.
    
    Parameters
    ----------
    ensemble_dir : str
        Directory containing ensemble result JSON files.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with ensemble results.
    """
    results = []
    
    files = sorted(glob.glob(f"{ensemble_dir}/*.json"), key=os.path.getmtime)
    
    for f in files:
        try:
            with open(f, 'r') as file:
                data = json.load(file)
            
            models = data.get('models', [])
            n_models = len(models)
            strategy = data.get('strategy', 'unknown')
            metrics = data.get('metrics', {})
            
            # Create ensemble name
            if n_models == 6:
                ensemble_name = "All 6 Methods"
            elif n_models == 2:
                if 'smote_rus' in str(models):
                    ensemble_name = "SMOTE+RUS + EasyEnsemble"
                else:
                    ensemble_name = "Baseline + EasyEnsemble"
            else:
                ensemble_name = f"{n_models} Models"
            
            result = {
                'name': ensemble_name,
                'strategy': strategy,
                'threshold': data.get('threshold', 0.5),
                'n_models': n_models,
                'models': models,
                'auprc': metrics.get('auprc', np.nan),
                'auroc': metrics.get('auroc', np.nan),
                'accuracy': metrics.get('accuracy', np.nan),
                'precision': metrics.get('precision', np.nan),
                'recall': metrics.get('recall', np.nan),
                'f1': metrics.get('f1', np.nan),
                'f2': metrics.get('f2', np.nan),
                'specificity': metrics.get('specificity', np.nan),
                'file': f,
            }
            results.append(result)
            
        except Exception as e:
            logger.warning(f"Failed to load ensemble result {f}: {e}")
            continue
    
    return pd.DataFrame(results)


# === Visualization Functions ===

def plot_auprc_comparison(
    df: pd.DataFrame,
    baseline_auprc: float = 0.039,
    figsize: Tuple[int, int] = (12, 6),
    title: str = "AUPRC Comparison (Imbalanced Data)",
) -> matplotlib.figure.Figure:
    """Create bar chart comparing AUPRC across methods.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'name' and 'auprc' columns.
    baseline_auprc : float, default=0.039
        Random baseline AUPRC (typically equals positive class rate).
    figsize : tuple, default=(12, 6)
        Figure size.
    title : str
        Plot title.
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure object.
    """
    # Sort by AUPRC descending
    df_sorted = df.sort_values('auprc', ascending=True)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Colors based on relative performance
    colors = []
    for auprc in df_sorted['auprc']:
        if auprc >= baseline_auprc * 1.2:
            colors.append('#2ecc71')  # Green - good
        elif auprc >= baseline_auprc * 1.1:
            colors.append('#f39c12')  # Orange - moderate
        else:
            colors.append('#e74c3c')  # Red - poor
    
    # Horizontal bar chart
    bars = ax.barh(df_sorted['name'], df_sorted['auprc'], color=colors, edgecolor='black')
    
    # Add baseline line
    ax.axvline(baseline_auprc, color='gray', linestyle='--', linewidth=2, label=f'Random Baseline ({baseline_auprc:.3f})')
    
    # Add value labels
    for bar, auprc in zip(bars, df_sorted['auprc']):
        improvement = (auprc / baseline_auprc - 1) * 100
        label = f'{auprc:.4f} ({improvement:+.0f}%)'
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                label, va='center', fontsize=9)
    
    ax.set_xlabel('AUPRC (PR-AUC)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_xlim(0, max(df_sorted['auprc']) * 1.3)
    
    plt.tight_layout()
    return fig


def plot_f2_comparison(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 6),
    title: str = "F2 Score Comparison (Recall-weighted)",
) -> matplotlib.figure.Figure:
    """Create bar chart comparing F2 scores (default and threshold-optimized).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'name', 'f2', and 'f2_thr' columns.
    figsize : tuple, default=(12, 6)
        Figure size.
    title : str
        Plot title.
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure object.
    """
    df_sorted = df.sort_values('f2_thr', ascending=True)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y = np.arange(len(df_sorted))
    height = 0.35
    
    # Default threshold (0.5) bars
    bars1 = ax.barh(y - height/2, df_sorted['f2'], height, 
                    label='F2 (thr=0.5)', color='#3498db', edgecolor='black')
    
    # Optimized threshold bars
    bars2 = ax.barh(y + height/2, df_sorted['f2_thr'], height,
                    label='F2 (optimized thr)', color='#2ecc71', edgecolor='black')
    
    ax.set_yticks(y)
    ax.set_yticklabels(df_sorted['name'])
    ax.set_xlabel('F2 Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    
    # Add value labels
    for bar1, bar2, f2, f2_thr in zip(bars1, bars2, df_sorted['f2'], df_sorted['f2_thr']):
        if not np.isnan(f2):
            ax.text(bar1.get_width() + 0.002, bar1.get_y() + bar1.get_height()/2,
                    f'{f2:.3f}', va='center', fontsize=8)
        if not np.isnan(f2_thr):
            ax.text(bar2.get_width() + 0.002, bar2.get_y() + bar2.get_height()/2,
                    f'{f2_thr:.3f}', va='center', fontsize=8)
    
    plt.tight_layout()
    return fig


def plot_recall_precision_tradeoff(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 8),
    title: str = "Recall vs Precision Trade-off",
) -> matplotlib.figure.Figure:
    """Create scatter plot showing recall-precision trade-off.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'name', 'recall', 'precision', 'recall_thr', 'precision_thr' columns.
    figsize : tuple, default=(10, 8)
        Figure size.
    title : str
        Plot title.
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color map for methods
    colors = plt.cm.tab10(np.linspace(0, 1, len(df)))
    
    for i, (_, row) in enumerate(df.iterrows()):
        # Default threshold point
        ax.scatter(row['recall'], row['precision'], 
                   c=[colors[i]], s=100, marker='o', label=f"{row['name']} (thr=0.5)")
        
        # Optimized threshold point
        if not np.isnan(row['recall_thr']) and not np.isnan(row['precision_thr']):
            ax.scatter(row['recall_thr'], row['precision_thr'],
                       c=[colors[i]], s=150, marker='s', alpha=0.7)
            
            # Arrow from default to optimized
            ax.annotate('', xy=(row['recall_thr'], row['precision_thr']),
                        xytext=(row['recall'], row['precision']),
                        arrowprops=dict(arrowstyle='->', color=colors[i], lw=1.5))
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, max(df['precision'].max(), df['precision_thr'].max()) * 1.5)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_metrics_heatmap(
    df: pd.DataFrame,
    metrics: List[str] = ['auprc', 'auroc', 'f2', 'f2_thr', 'recall', 'precision'],
    figsize: Tuple[int, int] = (10, 8),
    title: str = "Metrics Comparison Heatmap",
    cmap: str = "YlGnBu",
) -> matplotlib.figure.Figure:
    """Create heatmap of metrics across methods.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'name' column and metric columns.
    metrics : list of str
        List of metric column names to include.
    figsize : tuple, default=(10, 8)
        Figure size.
    title : str
        Plot title.
    cmap : str, default="YlGnBu"
        Colormap name.
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure object.
    """
    # Prepare data matrix
    data_matrix = df.set_index('name')[metrics].astype(float)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(data_matrix.values, cmap=cmap, aspect='auto')
    
    # Set ticks
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(data_matrix)))
    ax.set_xticklabels([m.upper() for m in metrics], rotation=45, ha='right')
    ax.set_yticklabels(data_matrix.index)
    
    # Add text annotations
    for i in range(len(data_matrix)):
        for j in range(len(metrics)):
            val = data_matrix.iloc[i, j]
            if not np.isnan(val):
                color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center', color=color, fontsize=9)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Value', fontsize=10)
    
    plt.tight_layout()
    return fig


def plot_summary_dashboard(
    df: pd.DataFrame,
    ensemble_df: Optional[pd.DataFrame] = None,
    baseline_auprc: float = 0.039,
    figsize: Tuple[int, int] = (16, 12),
    title: str = "Imbalanced Data Classification - Results Dashboard",
) -> matplotlib.figure.Figure:
    """Create comprehensive summary dashboard with multiple panels.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with single model results.
    ensemble_df : pd.DataFrame, optional
        DataFrame with ensemble results.
    baseline_auprc : float, default=0.039
        Random baseline AUPRC.
    figsize : tuple, default=(16, 12)
        Figure size.
    title : str
        Main title.
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure object with 4 panels.
    """
    fig = plt.figure(figsize=figsize)
    
    # Create 2x2 grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Panel 1: AUPRC Bar Chart
    ax1 = fig.add_subplot(gs[0, 0])
    df_sorted = df.sort_values('auprc', ascending=True)
    colors = ['#2ecc71' if auprc >= baseline_auprc * 1.1 else '#e74c3c' 
              for auprc in df_sorted['auprc']]
    bars = ax1.barh(df_sorted['name'], df_sorted['auprc'], color=colors, edgecolor='black')
    ax1.axvline(baseline_auprc, color='gray', linestyle='--', linewidth=2)
    ax1.set_xlabel('AUPRC')
    ax1.set_title('AUPRC Comparison', fontweight='bold')
    ax1.text(baseline_auprc, len(df_sorted)-0.5, f'Random\n({baseline_auprc:.3f})', 
             fontsize=8, va='center', ha='left')
    
    # Panel 2: F2 Score Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    y = np.arange(len(df_sorted))
    height = 0.35
    bars1 = ax2.barh(y - height/2, df_sorted['f2'].fillna(0), height, 
                     label='Default (thr=0.5)', color='#3498db')
    bars2 = ax2.barh(y + height/2, df_sorted['f2_thr'].fillna(0), height,
                     label='Optimized', color='#2ecc71')
    ax2.set_yticks(y)
    ax2.set_yticklabels(df_sorted['name'])
    ax2.set_xlabel('F2 Score')
    ax2.set_title('F2 Score (Recall-weighted)', fontweight='bold')
    ax2.legend(loc='lower right', fontsize=8)
    
    # Panel 3: Recall vs Specificity
    ax3 = fig.add_subplot(gs[1, 0])
    colors_scatter = plt.cm.tab10(np.linspace(0, 1, len(df)))
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        ax3.scatter(row['recall'], 1 - row.get('specificity', 0), 
                    c=[colors_scatter[i]], s=100, label=row['name'])
        if not np.isnan(row.get('recall_thr', np.nan)):
            ax3.scatter(row['recall_thr'], 1 - row.get('specificity_thr', 0),
                        c=[colors_scatter[i]], s=150, marker='s', alpha=0.7)
    ax3.set_xlabel('Recall (Sensitivity)')
    ax3.set_ylabel('False Positive Rate (1 - Specificity)')
    ax3.set_title('Recall vs FPR Trade-off', fontweight='bold')
    ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Summary Table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    # Create summary table
    summary_data = []
    for _, row in df.sort_values('auprc', ascending=False).iterrows():
        improvement = (row['auprc'] / baseline_auprc - 1) * 100
        summary_data.append([
            row['name'],
            f"{row['auprc']:.4f}",
            f"{improvement:+.1f}%",
            f"{row['f2_thr']:.3f}" if not np.isnan(row['f2_thr']) else "N/A",
            f"{row['recall_thr']:.1%}" if not np.isnan(row['recall_thr']) else "N/A",
        ])
    
    table = ax4.table(
        cellText=summary_data,
        colLabels=['Method', 'AUPRC', 'vs Random', 'F2 (opt)', 'Recall (opt)'],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax4.set_title('Summary Statistics', fontweight='bold', y=0.95)
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    return fig


def generate_imbalance_report(
    df: pd.DataFrame,
    ensemble_df: Optional[pd.DataFrame] = None,
    output_dir: str = "results/analysis/imbalance",
    baseline_auprc: float = 0.039,
) -> Dict[str, str]:
    """Generate complete analysis report with all visualizations.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with single model results.
    ensemble_df : pd.DataFrame, optional
        DataFrame with ensemble results.
    output_dir : str
        Output directory for figures and CSV.
    baseline_auprc : float, default=0.039
        Random baseline AUPRC.
    
    Returns
    -------
    dict
        Dictionary mapping figure names to file paths.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    
    # 1. AUPRC Comparison
    fig = plot_auprc_comparison(df, baseline_auprc=baseline_auprc)
    path = output_path / "auprc_comparison.png"
    save_figure(fig, path)
    plt.close(fig)
    saved_files['auprc_comparison'] = str(path)
    logger.info(f"Saved: {path}")
    
    # 2. F2 Comparison
    fig = plot_f2_comparison(df)
    path = output_path / "f2_comparison.png"
    save_figure(fig, path)
    plt.close(fig)
    saved_files['f2_comparison'] = str(path)
    logger.info(f"Saved: {path}")
    
    # 3. Recall-Precision Trade-off
    fig = plot_recall_precision_tradeoff(df)
    path = output_path / "recall_precision_tradeoff.png"
    save_figure(fig, path)
    plt.close(fig)
    saved_files['recall_precision_tradeoff'] = str(path)
    logger.info(f"Saved: {path}")
    
    # 4. Metrics Heatmap
    fig = plot_metrics_heatmap(df)
    path = output_path / "metrics_heatmap.png"
    save_figure(fig, path)
    plt.close(fig)
    saved_files['metrics_heatmap'] = str(path)
    logger.info(f"Saved: {path}")
    
    # 5. Summary Dashboard
    fig = plot_summary_dashboard(df, ensemble_df, baseline_auprc=baseline_auprc)
    path = output_path / "summary_dashboard.png"
    save_figure(fig, path, dpi=150)
    plt.close(fig)
    saved_files['summary_dashboard'] = str(path)
    logger.info(f"Saved: {path}")
    
    # 6. Save summary CSV
    csv_path = output_path / "summary_metrics.csv"
    df.to_csv(csv_path, index=False)
    saved_files['summary_csv'] = str(csv_path)
    logger.info(f"Saved: {csv_path}")
    
    # 7. Save ensemble CSV if provided
    if ensemble_df is not None and not ensemble_df.empty:
        csv_path = output_path / "ensemble_metrics.csv"
        ensemble_df.to_csv(csv_path, index=False)
        saved_files['ensemble_csv'] = str(csv_path)
        logger.info(f"Saved: {csv_path}")
    
    return saved_files
