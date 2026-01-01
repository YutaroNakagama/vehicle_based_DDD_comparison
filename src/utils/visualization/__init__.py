"""Visualization utilities for plotting and data visualization."""

from .visualization import (
    plot_roc_curves_from_latest_json,
    plot_custom_colored_distribution,
    save_figure,
    save_current_figure,
)
from .radar import make_radar
from .setup import setup_matplotlib_headless, setup_publication_style
from .color_palettes import (
    RANKING_METHOD_COLORS,
    DOMAIN_LEVEL_COLORS,
    TRAINING_MODE_COLORS,
    IMBALANCE_METHOD_COLORS,
    MODEL_TYPE_COLORS,
    METRIC_COLORS,
    get_color,
)

__all__ = [
    # Core visualization
    "plot_roc_curves_from_latest_json",
    "plot_custom_colored_distribution",
    "save_figure",
    "save_current_figure",
    "make_radar",
    # Setup utilities
    "setup_matplotlib_headless",
    "setup_publication_style",
    # Color palettes
    "RANKING_METHOD_COLORS",
    "DOMAIN_LEVEL_COLORS",
    "TRAINING_MODE_COLORS",
    "IMBALANCE_METHOD_COLORS",
    "MODEL_TYPE_COLORS",
    "METRIC_COLORS",
    "get_color",
]
