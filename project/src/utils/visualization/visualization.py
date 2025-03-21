import numpy as np
import logging
import scipy.stats
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def remove_outliers(data, threshold=3):
    """Remove outliers from data based on standard deviation threshold."""
    mean, std = np.mean(data), np.std(data)
    return [x for x in data if abs(x - mean) <= threshold * std]

def colorize_histogram(patches):
    """Apply custom colors to histogram bins."""
    for i, patch in enumerate(patches):
        if i < 6:
            patch.set_facecolor('green')
        elif i >= 7:
            patch.set_facecolor('yellow')
        else:
            patch.set_facecolor('gray')

def plot_custom_colored_distribution(data, output_path=None, threshold=None):
    """
    Plot a data distribution with custom-colored histogram bins, optionally removing outliers.

    Parameters:
        data (np.ndarray or list): Data to plot.
        output_path (str, optional): Path to save the plot (displays if None).
        threshold (float, optional): Standard deviation threshold for outlier removal (no removal if None).
    """
    # Remove outliers (optional)
    if threshold is not None:
        data = remove_outliers(data, threshold)

    # Create histogram bins
    bins = np.linspace(min(data), max(data), 10)

    plt.figure(figsize=(12, 8))

    counts, _, patches = plt.hist(
        data, bins=bins, alpha=0.6, edgecolor='black', label="Histogram"
    )

    # Apply colors to histogram bins
    colorize_histogram(patches)
    
    # KDE plot (implemented using matplotlib only)
    density = scipy.stats.gaussian_kde(data)
    x_vals = np.linspace(min(data), max(data), 200)
    plt.plot(x_vals, density(x_vals) * len(data) * np.diff(bins)[0], color='blue', linewidth=2, label="KDE")
    
    # Labels and legend
    plt.title("Custom Colored Distribution of Data", fontsize=16)
    plt.xlabel("Theta/Alpha Ratio", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    # Save or display plot
    if output_path:
        plt.savefig(output_path)
        logging.info(f"Plot saved to {output_path}")
    else:
        plt.show()
    
