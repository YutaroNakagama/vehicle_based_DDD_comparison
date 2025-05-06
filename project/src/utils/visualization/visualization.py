"""Visualization utilities for EEG-based feature distributions.

This module provides tools to:
- Remove outliers from numerical data
- Plot histograms with custom bin coloring
- Overlay KDE curve on histograms
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
    """Remove outliers based on standard deviation.

    Args:
        data (list or np.ndarray): Input numerical data.
        threshold (float): How many standard deviations to use for filtering.

    Returns:
        list: Data with outliers removed.
    """
    mean, std = np.mean(data), np.std(data)
    return [x for x in data if abs(x - mean) <= threshold * std]


def colorize_histogram(patches):
    """Apply color coding to histogram bins.

    Green: lower 6 bins (label: "negative"),
    Gray: center bins (label: "neutral"),
    Yellow: upper 3 bins (label: "positive")

    Args:
        patches (list): List of matplotlib Patch objects (histogram bars).
    """
    for i, patch in enumerate(patches):
        if i < 6:
            patch.set_facecolor('green')
        elif i >= 7:
            patch.set_facecolor('yellow')
        else:
            patch.set_facecolor('gray')


def plot_custom_colored_distribution(data, output_path: str = None, threshold: float = None):
    """Plot a histogram of data with KDE overlay and colored bins.

    Args:
        data (list or np.ndarray): Numerical data to visualize.
        output_path (str, optional): If specified, saves plot to path. Otherwise, displays interactively.
        threshold (float, optional): If set, removes outliers beyond N standard deviations.

    Returns:
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


def extract_model_and_time(filename):
    match = re.match(r"metrics_(.+?)_.*?_(\d{8}_\d{6})\.json", filename)
    if match:
        model, timestamp = match.groups()
        dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
        return model, dt
    return None, None


def find_latest_metric_files(directory):
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
    Plot ROC curves from the latest JSON metrics per model in the given directory.

    Args:
        results_dir (str): Directory where metrics_*.json files are stored.
        title (str): Title of the ROC plot.

    Returns:
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
