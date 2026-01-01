#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Centralized color palettes for consistent visualization across the project.

This module provides standard color definitions to ensure visual consistency
in all plots and charts throughout the codebase.

Usage:
    from src.utils.visualization.color_palettes import (
        RANKING_METHOD_COLORS,
        DOMAIN_LEVEL_COLORS,
        TRAINING_MODE_COLORS,
        IMBALANCE_METHOD_COLORS,
    )
"""

# =============================================================================
# Ranking Methods (for domain analysis)
# =============================================================================
RANKING_METHOD_COLORS = {
    "mean_distance": "#1f77b4",      # blue
    "centroid_umap": "#ff7f0e",      # orange
    "lof": "#2ca02c",                # green
    "knn": "#d62728",                # red
    "median_distance": "#9467bd",    # purple
    "isolation_forest": "#8c564b",   # brown
    # Display name variants
    "Centroid+UMAP": "#e41a1c",      # red
    "LOF": "#377eb8",                # blue
    "KNN": "#4daf4a",                # green
    "Mean Distance": "#1f77b4",      # blue
    "Median Distance": "#9467bd",    # purple
}

# =============================================================================
# Domain Levels (for subject partitioning)
# =============================================================================
DOMAIN_LEVEL_COLORS = {
    "out_domain": "#e41a1c",    # red (outliers)
    "mid_domain": "#999999",    # gray
    "in_domain": "#377eb8",     # blue (central)
    # Short variants
    "out": "#e41a1c",
    "mid": "#999999",
    "in": "#377eb8",
}

# =============================================================================
# Training Modes
# =============================================================================
TRAINING_MODE_COLORS = {
    "pooled": "#2ca02c",        # green
    "source_only": "#1f77b4",   # blue
    "target_only": "#ff7f0e",   # orange
    # Display name variants
    "Pooled": "#2ca02c",
    "Source Only": "#1f77b4",
    "Target Only": "#ff7f0e",
}

# =============================================================================
# Imbalance Handling Methods
# =============================================================================
IMBALANCE_METHOD_COLORS = {
    # Core methods
    "baseline": "#7f7f7f",           # gray
    "smote": "#1f77b4",              # blue
    "smote_tomek": "#2ca02c",        # green (best performer)
    "smote_enn": "#ff7f0e",          # orange
    "smote_rus": "#9467bd",          # purple
    "smote_balanced_rf": "#17becf",  # cyan
    "balanced_rf": "#8c564b",        # brown
    "easy_ensemble": "#e377c2",      # pink
    "undersample_enn": "#17becf",    # cyan
    "undersample_rus": "#bcbd22",    # olive
    "undersample_tomek": "#d62728",  # red
    "jitter_scale": "#ff9896",       # light red
    # Display name variants
    "Baseline": "#7f7f7f",
    "SMOTE": "#1f77b4",
    "SMOTE+Tomek": "#2ca02c",
    "SMOTE+ENN": "#ff7f0e",
    "SMOTE+RUS": "#9467bd",
    "BalancedRF": "#8c564b",
    "EasyEnsemble": "#e377c2",
    "Undersample-ENN": "#17becf",
    "Undersample-RUS": "#bcbd22",
    "Undersample-Tomek": "#d62728",
    "Jitter+Scale": "#ff9896",
}

# =============================================================================
# Model Types
# =============================================================================
MODEL_TYPE_COLORS = {
    "RF": "#1f77b4",            # blue
    "BalancedRF": "#2ca02c",    # green
    "EasyEnsemble": "#ff7f0e",  # orange
    "SVM": "#d62728",           # red
    "LSTM": "#9467bd",          # purple
}

# =============================================================================
# Metric Types
# =============================================================================
METRIC_COLORS = {
    "f2": "#2ca02c",            # green
    "recall": "#1f77b4",        # blue
    "precision": "#ff7f0e",     # orange
    "auprc": "#d62728",         # red
    "specificity": "#9467bd",   # purple
    "auroc": "#8c564b",         # brown
}


def get_color(color_dict: dict, key: str, default: str = "#7f7f7f") -> str:
    """Get color from dictionary with fallback to default.
    
    Parameters
    ----------
    color_dict : dict
        Color dictionary to search
    key : str
        Key to look up
    default : str
        Default color if key not found (default: gray)
        
    Returns
    -------
    str
        Hex color code
    """
    return color_dict.get(key, default)
