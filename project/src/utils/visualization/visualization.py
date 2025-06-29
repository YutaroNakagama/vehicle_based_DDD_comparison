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
    """Removes outliers from a numerical dataset based on a standard deviation threshold.

    Outliers are defined as data points that fall outside a specified number of
    standard deviations from the mean. This function helps in cleaning data
    for more representative visualizations.

    Args:
        data (list | np.ndarray): The input numerical data (list or NumPy array).
        threshold (float): The number of standard deviations from the mean to define
                           the outlier boundaries. Defaults to 3.

    Returns:
        list: A new list containing the data with outliers removed.
    """
    mean, std = np.mean(data), np.std(data)
    return [x for x in data if abs(x - mean) <= threshold * std]


def colorize_histogram(patches):
    """Applies custom color coding to histogram bins based on their position.

    This function is designed to visually segment a histogram into three conceptual
    regions: 'negative' (green, lower bins), 'neutral' (gray, center bins), and
    'positive' (yellow, upper bins). This helps in quickly interpreting the distribution
    of data points across different ranges.

    Args:
        patches (list): A list of `matplotlib.patches.Rectangle` objects, typically
                        obtained from the `plt.hist()` function, representing the bars of the histogram.
    """
    for i, patch in enumerate(patches):
        if i < 6:
            patch.set_facecolor('green')
        elif i >= 7:
            patch.set_facecolor('yellow')
        else:
            patch.set_facecolor('gray')


def plot_custom_colored_distribution(data, output_path: str = None, threshold: float = None):
    """Plots a histogram of numerical data with custom colored bins and an overlaid KDE curve.

    This function provides a visually informative representation of data distribution.
    It can optionally remove outliers and applies a predefined color scheme to histogram
    bins (green for lower, gray for middle, yellow for upper) to highlight different ranges.
    A Kernel Density Estimate (KDE) curve is overlaid to show the smoothed probability density.

    Args:
        data (list | np.ndarray): The numerical data to be visualized.
        output_path (str, optional): If specified, the plot will be saved to this file path.
                                     Otherwise, the plot will be displayed interactively.
                                     Defaults to None.
        threshold (float, optional): If set, outliers beyond this number of standard deviations
                                     from the mean will be removed from the data before plotting.
                                     Defaults to None.

    Returns:
        None: The function either saves the plot to a file or displays it, and does not return any value.
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
    """Extracts the model name and timestamp from a metrics JSON filename.

    This function parses a standardized filename format (e.g., 'metrics_model_tag_YYYYMMDD_HHMMSS.json')
    to extract the model identifier and the timestamp of the evaluation. This is useful
    for organizing and retrieving specific evaluation results.

    Args:
        filename (str): The name of the metrics JSON file.

    Returns:
        tuple[str | None, datetime | None]: A tuple containing:
            - str | None: The extracted model name, or None if not found.
            - datetime | None: The parsed datetime object from the timestamp, or None if not found.
    """
    match = re.match(r"metrics_(.+?)_.*?_(\d{8}_\d{6})\.json", filename)
    if match:
        model, timestamp = match.groups()
        dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
        return model, dt
    return None, None


def find_latest_metric_files(directory: str) -> dict:
    """Finds the latest metrics JSON file for each model within a specified directory.

    This function scans a directory for files matching the 'metrics_*.json' pattern,
    extracts the model name and timestamp from each, and identifies the most recent
    file for each unique model. This is useful for ensuring that ROC curves or other
    visualizations are generated from the most up-to-date evaluation results.

    Args:
        directory (str): The path to the directory containing the metrics JSON files.

    Returns:
        dict: A dictionary where keys are model names (str) and values are the filenames
              (str) of the latest metrics JSON file for that model.
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
    """Plots ROC curves for multiple models from their latest metrics JSON files.

    This function automatically identifies the most recent evaluation results for each model
    within a specified directory, loads their ROC data (False Positive Rate, True Positive Rate,
    and AUC), and plots them on a single graph for easy comparison. A diagonal line representing
    random chance is also included.

    Args:
        results_dir (str): The directory where the metrics JSON files (e.g., 'metrics_*.json') are stored.
        title (str): The title of the ROC curve plot. Defaults to "ROC Curve Comparison".

    Returns:
        None: The function displays the plot and does not return any value.
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
