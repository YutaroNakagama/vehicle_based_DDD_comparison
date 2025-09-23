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
"""

import numpy as np
import logging
import scipy.stats
import matplotlib
import os
import json
from collections import defaultdict
from datetime import datetime
import re

matplotlib.use('TkAgg')  # For compatibility with interactive environments
import matplotlib.pyplot as plt


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
