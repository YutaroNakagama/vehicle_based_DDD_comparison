"""Visualization Utilities for Driver Drowsiness Detection Data and Results.

This module provides functions for visualizing various aspects of the EEG-based
feature distributions and model evaluation results within the driver drowsiness
detection project. It includes tools for plotting histograms with custom bin
coloring and KDE overlays, as well as generating ROC curves for model comparison.

Key functionalities include:
- Outlier removal for robust visualization.
- Custom colored histograms to highlight specific data ranges.
- Overlaying Kernel Density Estimation (KDE) curves for distribution shape analysis.
- Plotting ROC curves from model evaluation metrics for performance comparison.
- Common figure saving utilities with consistent DPI and layout settings.
"""

import numpy as np
import logging
import scipy.stats
import matplotlib
import os
import json
from pathlib import Path
from typing import Union, Optional
from collections import defaultdict
from datetime import datetime
import re

matplotlib.use('Agg')  # Non-interactive backend for HPC environments
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# === Common Figure Saving Utilities ===

def save_figure(
    fig,
    path: Union[str, Path],
    dpi: int = 200,
    bbox_inches: str = "tight",
    **kwargs
) -> None:
    """Save a matplotlib figure with consistent settings.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    path : str or Path
        Output path for the figure.
    dpi : int, default=200
        Resolution in dots per inch.
    bbox_inches : str, default="tight"
        Bounding box setting for the saved figure.
    **kwargs
        Additional arguments passed to fig.savefig.
    
    Returns
    -------
    None
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.debug(f"Saving figure to {path}")
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches, **kwargs)


def save_current_figure(
    path: Union[str, Path],
    dpi: int = 200,
    bbox_inches: str = "tight",
    close: bool = True,
    **kwargs
) -> None:
    """Save the current matplotlib figure and optionally close it.
    
    Parameters
    ----------
    path : str or Path
        Output path for the figure.
    dpi : int, default=200
        Resolution in dots per inch.
    bbox_inches : str, default="tight"
        Bounding box setting for the saved figure.
    close : bool, default=True
        If True, close the figure after saving.
    **kwargs
        Additional arguments passed to plt.savefig.
    
    Returns
    -------
    None
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.debug(f"Saving current figure to {path}")
    plt.savefig(path, dpi=dpi, bbox_inches=bbox_inches, **kwargs)
    
    if close:
        plt.close()


def remove_outliers(data: list, threshold: float = 3) -> list:
    """
    Remove outliers based on standard deviation threshold.

    Parameters
    ----------
    data : list or numpy.ndarray
        Input numerical dataset.
    threshold : float, default=3
        Number of standard deviations from the mean to define outliers.

    Returns
    -------
    list
        Data with outliers removed.
    """
    mean, std = np.mean(data), np.std(data)
    return [x for x in data if abs(x - mean) <= threshold * std]


def colorize_histogram(patches):
    """
    Apply custom colors to histogram bins.

    Parameters
    ----------
    patches : list of matplotlib.patches.Rectangle
        Histogram patches from ``plt.hist``.
    """
    for i, patch in enumerate(patches):
        if i < 6:
            patch.set_facecolor('green')
        elif i >= 7:
            patch.set_facecolor('yellow')
        else:
            patch.set_facecolor('gray')


def plot_custom_colored_distribution(data, output_path: str = None, threshold: float = None):
    """
    Plot histogram with custom colored bins and KDE overlay.

    Parameters
    ----------
    data : list or numpy.ndarray
        Numerical data to visualize.
    output_path : str, optional
        File path to save the figure. If ``None``, shows interactively.
    threshold : float, optional
        If set, outliers beyond this threshold (std dev) are removed.

    Returns
    -------
    None
    """
    # Optional outlier removal
    if threshold is not None:
        data = remove_outliers(data, threshold)

    # Histogram bins
    bins = np.linspace(min(data), max(data), 10)

    plt.figure(figsize=(12, 8))

    counts, _, patches = plt.hist(
        data, bins=bins, alpha=0.6, edgecolor='black', label="Histogram"
    )

    colorize_histogram(patches)

    # KDE
    density = scipy.stats.gaussian_kde(data)
    x_vals = np.linspace(min(data), max(data), 200)
    plt.plot(x_vals, density(x_vals) * len(data) * np.diff(bins)[0],
             color='blue', linewidth=2, label="KDE")

    # Plot settings
    plt.title("Custom Colored Distribution of Data", fontsize=16)
    plt.xlabel("Theta/Alpha Ratio", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)

    if output_path:
        plt.savefig(output_path)
        logging.info(f"Plot saved to {output_path}")
    else:
        plt.show()


def extract_model_and_time(filename: str) -> tuple[str | None, datetime | None]:
    """
    Extract model name and timestamp from metrics JSON filename.

    Parameters
    ----------
    filename : str
        Filename in format ``metrics_<model>_<tag>_YYYYMMDD_HHMMSS.json``.

    Returns
    -------
    tuple of (str or None, datetime or None)
        Model name and timestamp if parsed successfully, otherwise ``(None, None)``.
    """
    match = re.match(r"metrics_(.+?)_.*?_(\d{8}_\d{6})\.json", filename)
    if match:
        model, timestamp = match.groups()
        dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
        return model, dt
    return None, None


def find_latest_metric_files(directory: str) -> dict:
    """
    Find the latest metrics JSON file per model in a directory.

    Parameters
    ----------
    directory : str
        Path to directory containing metrics JSON files.

    Returns
    -------
    dict
        Mapping from model name (str) to latest JSON filename (str).
    """
    model_files = defaultdict(list)

    for fname in os.listdir(directory):
        if fname.startswith("metrics_") and fname.endswith(".json"):
            model, dt = extract_model_and_time(fname)
            if model and dt:
                model_files[model].append((dt, fname))

    latest_files = {}
    for model, filelist in model_files.items():
        latest_files[model] = max(filelist)[1]

    return latest_files


def plot_roc_curves_from_latest_json(results_dir: str, title: str = "ROC Curve Comparison"):
    """
    Plot ROC curves for multiple models from their latest JSON files.

    Parameters
    ----------
    results_dir : str
        Directory containing ``metrics_*.json`` files.
    title : str, default="ROC Curve Comparison"
        Plot title.

    Returns
    -------
    None
    """
    latest_files = find_latest_metric_files(results_dir)

    plt.figure(figsize=(10, 8))

    for model, fname in sorted(latest_files.items()):
        full_path = os.path.join(results_dir, fname)
        with open(full_path, "r") as f:
            result = json.load(f)

        roc = result.get("roc_curve")
        if not roc:
            logging.warning(f"No ROC data in {fname}, skipping.")
            continue

        fpr = roc["fpr"]
        tpr = roc["tpr"]
        auc_value = roc.get("auc", None)

        label = f"{model} (AUC={auc_value:.2f})" if auc_value else model
        plt.plot(fpr, tpr, label=label)

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.title(title, fontsize=16)
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# === Domain Analysis Visualization ===

def plot_grouped_bar_chart(
    data: "pd.DataFrame",
    metrics: list,
    modes: list,
    distance_col: str = "distance",
    level_col: str = "level",
    figsize: tuple = None,
    baseline_rates: Optional[dict] = None,
    title_map: Optional[dict] = None,
) -> matplotlib.figure.Figure:
    """Create multi-panel bar chart for domain analysis metrics.
    
    Generates a grid of subplots showing metrics across different distances and levels
    (e.g., high/middle/low) for source-only vs target-only comparisons.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns for model, distance, level, and metric values.
    metrics : list of str
        List of metric names to plot (e.g., ['auc', 'auc_pr', 'f1']).
    modes : list of str
        List of mode names (e.g., ['source_only', 'target_only']).
    distance_col : str, default="distance"
        Column name containing distance type (e.g., 'mmd', 'wasserstein', 'dtw').
    level_col : str, default="level"
        Column name containing level type (e.g., 'high', 'middle', 'low').
    figsize : tuple, optional
        Figure size (width, height). If None, auto-calculated based on layout.
    baseline_rates : dict, optional
        Mapping from metric name to baseline value for reference lines.
    title_map : dict, optional
        Mapping from metric name to display title.
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the bar chart grid.
    
    Examples
    --------
    >>> fig = plot_grouped_bar_chart(
    ...     df, 
    ...     metrics=['auc', 'f1'],
    ...     modes=['source_only', 'target_only'],
    ...     baseline_rates={'auc_pr': 0.033}
    ... )
    >>> fig.savefig('metrics_comparison.png')
    """
    import pandas as pd
    
    distances = sorted(data[distance_col].unique())
    ordered_levels = ["high", "middle", "low"]
    
    if figsize is None:
        figsize = (5 * len(metrics), 3 * len(distances))
    
    if title_map is None:
        title_map = {
            "auc": "AUROC",
            "auc_pr": "AUPRC",
            "accuracy": "Accuracy",
            "f1": "F1",
            "f2": "F2",
            "precision": "Precision (pos)",
            "recall": "Recall (pos)",
        }
    
    fig, axes = plt.subplots(
        len(distances), len(metrics), 
        figsize=figsize, 
        squeeze=False
    )
    
    colors = ["#66cc99", "#6699cc", "#ff9966"]
    mode_labels = {"pooled": "Pooled", "source_only": "Source-only", "target_only": "Target-only"}
    
    for i, dist in enumerate(distances):
        sub = data[data[distance_col] == dist]
        
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            
            if sub.empty:
                ax.axis("off")
                continue
            
            # Filter to present levels
            levels_present = [lvl for lvl in ordered_levels if lvl in sub[level_col].unique()]
            x = np.arange(len(levels_present))
            width = 0.25
            
            # Collect values for each mode
            mode_values = {}
            for mode in modes:
                vals = []
                for lvl in levels_present:
                    sub_lvl = sub[sub[level_col] == lvl]
                    col = f"{metric}_{mode}"
                    vals.append(sub_lvl[col].mean() if col in sub_lvl.columns else np.nan)
                mode_values[mode] = vals
            
            # Plot bars
            bars = []
            for idx, mode in enumerate(modes):
                offset = (idx - len(modes)/2 + 0.5) * width
                bar = ax.bar(
                    x + offset, 
                    mode_values[mode], 
                    width, 
                    label=mode_labels.get(mode, mode),
                    color=colors[idx % len(colors)]
                )
                bars.append(bar)
            
            ax.set_xticks(x)
            ax.set_xticklabels([lvl.capitalize() for lvl in levels_present])
            
            # Baseline line for specific metrics
            if baseline_rates and metric in baseline_rates:
                baseline = baseline_rates[metric]
                ax.axhline(baseline, color='gray', linestyle='--', linewidth=1)
                ax.text(
                    len(levels_present)-0.5, baseline + 0.01,
                    f"Baseline ({baseline:.3f})", 
                    fontsize=8, color='gray'
                )
                
                # Dynamic y-axis for metrics with baseline
                all_vals = [v for vals in mode_values.values() for v in vals if not np.isnan(v)]
                if all_vals:
                    ymin, ymax = min(all_vals), max(all_vals)
                    margin = (ymax - ymin) * 0.3 if ymax > ymin else 0.02
                    ax.set_ylim(max(0, ymin - margin), min(1.0, ymax + margin))
            else:
                ax.set_ylim(0, 1.0)
            
            # Title on top row
            if i == 0:
                ax.set_title(title_map.get(metric, metric.upper()), fontsize=11)
            
            # Distance label on left column
            if j == 0:
                pretty_dist = {"dtw": "DTW", "mmd": "MMD", "wasserstein": "Wasserstein"}
                dist_label = pretty_dist.get(str(dist).lower(), str(dist))
                ax.text(
                    0.02, 0.95, dist_label, 
                    transform=ax.transAxes, 
                    ha="left", va="top",
                    fontsize=12, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6)
                )
            
            # Legend on top-right subplot
            if i == 0 and j == len(metrics) - 1:
                ax.legend(
                    handles=bars,
                    labels=[mode_labels.get(m, m) for m in modes],
                    loc="upper right",
                    fontsize=8,
                    frameon=False,
                )
    
    plt.tight_layout()
    return fig


def plot_grouped_bar_chart_raw(
    data: "pd.DataFrame",
    metrics: list,
    modes: list,
    distance_col: str = "distance",
    level_col: str = "level",
    mode_col: str = "mode",
    figsize: tuple = None,
    baseline_rates: Optional[dict] = None,
    title_map: Optional[dict] = None,
) -> matplotlib.figure.Figure:
    """Create multi-panel bar chart from raw (unpivoted) data.
    
    Parameters
    ----------
    data : pd.DataFrame
        Raw DataFrame with 'mode' column and metric columns.
    metrics : list of str
        List of metric names to plot.
    modes : list of str
        List of mode names (e.g., ['source_only', 'target_only']).
    distance_col : str, default="distance"
        Column name containing distance type.
    level_col : str, default="level"
        Column name containing level type.
    mode_col : str, default="mode"
        Column name containing mode (source_only/target_only).
    figsize : tuple, optional
        Figure size (width, height).
    baseline_rates : dict, optional
        Mapping from metric name to baseline value.
    title_map : dict, optional
        Mapping from metric name to display title.
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the bar chart grid.
    """
    import pandas as pd
    
    distances = sorted(data[distance_col].unique())
    ordered_levels = ["high", "middle", "low"]
    
    if figsize is None:
        figsize = (5 * len(metrics), 3 * len(distances))
    
    if title_map is None:
        title_map = {
            "auc": "AUROC",
            "auc_pr": "AUPRC",
            "accuracy": "Accuracy",
            "f1": "F1",
            "f2": "F2",
            "precision": "Precision (pos)",
            "recall": "Recall (pos)",
        }
    
    fig, axes = plt.subplots(
        len(distances), len(metrics), 
        figsize=figsize, 
        squeeze=False
    )
    
    colors = ["#66cc99", "#6699cc", "#ff9966"]
    mode_labels = {"pooled": "Pooled", "source_only": "Source-only", "target_only": "Target-only"}
    
    for i, dist in enumerate(distances):
        sub = data[data[distance_col] == dist]
        
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            
            if sub.empty:
                ax.axis("off")
                continue
            
            # Filter to present levels
            levels_present = [lvl for lvl in ordered_levels if lvl in sub[level_col].unique()]
            x = np.arange(len(levels_present))
            width = 0.25
            
            # Collect values for each mode from raw data
            mode_values = {}
            for mode in modes:
                vals = []
                for lvl in levels_present:
                    # Filter by level and mode
                    mode_data = sub[(sub[level_col] == lvl) & (sub[mode_col] == mode)]
                    vals.append(mode_data[metric].mean() if not mode_data.empty else np.nan)
                mode_values[mode] = vals
            
            # Plot bars
            bars = []
            for idx, mode in enumerate(modes):
                offset = (idx - len(modes)/2 + 0.5) * width
                bar = ax.bar(
                    x + offset, 
                    mode_values[mode], 
                    width, 
                    label=mode_labels.get(mode, mode),
                    color=colors[idx % len(colors)]
                )
                bars.append(bar)
            
            ax.set_xticks(x)
            ax.set_xticklabels([lvl.capitalize() for lvl in levels_present])
            
            # Baseline line for specific metrics
            if baseline_rates and metric in baseline_rates:
                baseline = baseline_rates[metric]
                ax.axhline(baseline, color='gray', linestyle='--', linewidth=1)
                ax.text(
                    len(levels_present)-0.5, baseline + 0.01,
                    f"Baseline ({baseline:.3f})", 
                    fontsize=8, color='gray'
                )
                
                # Dynamic y-axis for metrics with baseline
                all_vals = [v for vals in mode_values.values() for v in vals if not np.isnan(v)]
                if all_vals:
                    ymin, ymax = min(all_vals), max(all_vals)
                    margin = (ymax - ymin) * 0.3 if ymax > ymin else 0.02
                    ax.set_ylim(max(0, ymin - margin), min(1.0, ymax + margin))
            else:
                ax.set_ylim(0, 1.0)
            
            # Title on top row
            if i == 0:
                ax.set_title(title_map.get(metric, metric.upper()), fontsize=11)
            
            # Distance label on left column
            if j == 0:
                pretty_dist = {"dtw": "DTW", "mmd": "MMD", "wasserstein": "Wasserstein"}
                dist_label = pretty_dist.get(str(dist).lower(), str(dist))
                ax.text(
                    0.02, 0.95, dist_label, 
                    transform=ax.transAxes, 
                    ha="left", va="top",
                    fontsize=12, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6)
                )
            
            # Legend on top-right subplot
            if i == 0 and j == len(metrics) - 1:
                ax.legend(
                    handles=bars,
                    labels=[mode_labels.get(m, m) for m in modes],
                    loc="upper right",
                    fontsize=8,
                    frameon=False,
                )
    
    plt.tight_layout()
    return fig


def plot_metric_difference_heatmap(
    data: "pd.DataFrame",
    metrics: list,
    comparisons: list,
    distance_col: str = "distance",
    level_col: str = "level",
    figsize: tuple = None,
    cmap: str = "coolwarm",
    vmin: float = -1,
    vmax: float = 1,
) -> matplotlib.figure.Figure:
    """Create heatmap showing metric differences between modes.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with metric columns in format '{metric}_{mode}'.
    metrics : list of str
        List of metrics to compare.
    comparisons : list of tuple
        List of (mode_a, mode_b, label) tuples for comparison.
    distance_col : str, default="distance"
        Column name for distance type.
    level_col : str, default="level"
        Column name for level type.
    figsize : tuple, optional
        Figure size. If None, auto-calculated.
    cmap : str, default="coolwarm"
        Colormap name.
    vmin, vmax : float, default=-1, 1
        Color scale limits.
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing heatmap grid.
    
    Examples
    --------
    >>> fig = plot_metric_difference_heatmap(
    ...     df,
    ...     metrics=['auc', 'f1'],
    ...     comparisons=[('source_only', 'target_only', 'Source - Target')]
    ... )
    """
    import pandas as pd
    
    if figsize is None:
        figsize = (4 * len(comparisons), 4 * len(metrics))
    
    fig, axes = plt.subplots(
        len(metrics), len(comparisons),
        figsize=figsize,
        squeeze=False
    )
    
    for i, metric in enumerate(metrics):
        for j, (mode_a, mode_b, title) in enumerate(comparisons):
            ax = axes[i, j]
            
            diffs, labels = [], []
            for _, row in data.iterrows():
                val_a = row.get(f"{metric}_{mode_a}", np.nan)
                val_b = row.get(f"{metric}_{mode_b}", np.nan)
                
                if pd.notna(val_a) and pd.notna(val_b):
                    diffs.append(val_a - val_b)
                else:
                    diffs.append(np.nan)
                
                lbl_dist = str(row.get(distance_col, "unknown"))
                lbl_lvl = str(row.get(level_col, "unknown"))
                labels.append(f"{lbl_dist}/{lbl_lvl}")
            
            # Reshape to column matrix
            mat = np.array(diffs).reshape(-1, 1)
            
            im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=9)
            ax.set_xticks([])
            ax.set_title(f"{metric.upper()} ({title})", fontsize=10)
            
            # Annotate values
            for k, v in enumerate(diffs):
                if not np.isnan(v):
                    color = 'white' if abs(v) > 0.5 else 'black'
                    ax.text(
                        0, k, f"{v:.2f}",
                        ha="center", va="center",
                        fontsize=9, color=color
                    )
    
    plt.tight_layout()
    return fig
