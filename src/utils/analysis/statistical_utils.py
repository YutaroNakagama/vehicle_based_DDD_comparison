#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistical utility functions for analysis scripts.

This module provides common statistical functions used across
imbalance analysis and domain analysis scripts.

Functions:
    cohens_d: Calculate Cohen's d effect size
    interpret_cohens_d: Interpret effect size magnitude
    wilcoxon_test: Perform Wilcoxon signed-rank test
    paired_ttest: Perform paired t-test
    bootstrap_ci: Compute bootstrap confidence interval
    format_p_value: Format p-value for display
"""

from typing import Tuple

import numpy as np
from scipy import stats


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size for paired samples.
    
    Cohen's d = (mean1 - mean2) / pooled_std
    
    Interpretation:
        |d| < 0.2: negligible
        |d| ≈ 0.2: small
        |d| ≈ 0.5: medium
        |d| ≈ 0.8: large
        |d| > 1.0: very large
    
    Parameters
    ----------
    group1 : np.ndarray
        First group of samples
    group2 : np.ndarray
        Second group of samples
        
    Returns
    -------
    float
        Cohen's d effect size
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size magnitude.
    
    Parameters
    ----------
    d : float
        Cohen's d value
        
    Returns
    -------
    str
        Interpretation string
    """
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    elif abs_d < 1.0:
        return "large"
    else:
        return "very large"


def wilcoxon_test(
    group1: np.ndarray,
    group2: np.ndarray,
    alternative: str = "two-sided",
) -> Tuple[float, float]:
    """Perform Wilcoxon signed-rank test (paired).
    
    Parameters
    ----------
    group1 : np.ndarray
        First group of samples
    group2 : np.ndarray
        Second group of samples
    alternative : str
        Alternative hypothesis: 'two-sided', 'greater', or 'less'
        
    Returns
    -------
    statistic : float
        Test statistic
    p_value : float
        P-value
    """
    # Handle identical arrays
    if np.allclose(group1, group2):
        return np.nan, 1.0
    
    try:
        stat, p = stats.wilcoxon(group1, group2, alternative=alternative)
        return float(stat), float(p)
    except Exception:
        return np.nan, np.nan


def paired_ttest(
    group1: np.ndarray,
    group2: np.ndarray,
) -> Tuple[float, float]:
    """Perform paired t-test.
    
    Parameters
    ----------
    group1 : np.ndarray
        First group of samples
    group2 : np.ndarray
        Second group of samples
        
    Returns
    -------
    statistic : float
        Test statistic
    p_value : float
        P-value
    """
    try:
        stat, p = stats.ttest_rel(group1, group2)
        return float(stat), float(p)
    except Exception:
        return np.nan, np.nan


def bootstrap_ci(
    data: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for the mean.
    
    Parameters
    ----------
    data : np.ndarray
        Data samples
    confidence : float
        Confidence level (default: 0.95)
    n_bootstrap : int
        Number of bootstrap iterations (default: 10000)
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    lower : float
        Lower bound of CI
    upper : float
        Upper bound of CI
    """
    if len(data) == 0:
        return np.nan, np.nan
    
    rng = np.random.default_rng(seed)
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    
    return lower, upper


def format_p_value(p: float, stars: bool = True) -> str:
    """Format p-value for reporting with optional significance stars.
    
    Parameters
    ----------
    p : float
        P-value to format
    stars : bool
        Whether to include significance stars (default: True)
        
    Returns
    -------
    str
        Formatted p-value string
        
    Notes
    -----
    Significance levels:
        *** : p < 0.001
        **  : p < 0.01
        *   : p < 0.05
    """
    if np.isnan(p):
        return "N/A"
    
    if stars:
        if p < 0.001:
            return "< 0.001***"
        elif p < 0.01:
            return f"{p:.4f}**"
        elif p < 0.05:
            return f"{p:.4f}*"
    
    return f"{p:.4f}"
