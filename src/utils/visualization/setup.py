#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Matplotlib setup utilities for headless/HPC environments.

This module provides common matplotlib configuration functions
to ensure consistent behavior in non-interactive (HPC) environments.

Usage:
    from src.utils.visualization.setup import setup_matplotlib_headless
    setup_matplotlib_headless()  # Call before importing pyplot
    import matplotlib.pyplot as plt
"""

import logging


def setup_matplotlib_headless(backend: str = "Agg", suppress_font_warnings: bool = True):
    """Configure matplotlib for headless (HPC/server) environments.
    
    This function should be called BEFORE importing matplotlib.pyplot
    to ensure proper backend configuration.
    
    Parameters
    ----------
    backend : str
        Matplotlib backend to use (default: 'Agg' for non-interactive)
    suppress_font_warnings : bool
        Whether to suppress font manager warnings (default: True)
        
    Example
    -------
    >>> from src.utils.visualization.setup import setup_matplotlib_headless
    >>> setup_matplotlib_headless()
    >>> import matplotlib.pyplot as plt
    >>> # Now matplotlib is configured for headless operation
    """
    import matplotlib as mpl
    
    # Set non-interactive backend
    mpl.use(backend)
    
    # Suppress matplotlib logs
    mpl.set_loglevel("warning")
    
    # Suppress font manager spam
    if suppress_font_warnings:
        logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


def setup_publication_style():
    """Configure matplotlib for publication-quality figures.
    
    Sets up consistent styling for academic publications.
    """
    import matplotlib.pyplot as plt
    
    plt.rcParams.update({
        # Figure
        "figure.figsize": (8, 6),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        
        # Font
        "font.size": 12,
        "font.family": "sans-serif",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        
        # Lines
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
        
        # Grid
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        
        # Legend
        "legend.framealpha": 0.9,
        "legend.edgecolor": "gray",
    })
