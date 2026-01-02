#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sampling Analysis Module
========================

Core functionality for sampling distribution analysis:
- Log-based extraction of actual sampling distributions
- Theoretical calculation of expected distributions
- Comparison utilities

This module contains the business logic extracted from CLI wrappers.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Default original training data distribution
DEFAULT_TRAIN_ALERT = 35522
DEFAULT_TRAIN_DROWSY = 1445


def extract_sampling_distribution(
    log_dir: Path,
    job_prefix: str = "",
    train_alert: int = DEFAULT_TRAIN_ALERT,
    train_drowsy: int = DEFAULT_TRAIN_DROWSY
) -> pd.DataFrame:
    """Extract sampling distribution from job logs.
    
    Parameters
    ----------
    log_dir : Path
        Directory containing job log files (*.OU)
    job_prefix : str
        Job ID prefix to filter log files
    train_alert : int
        Original number of alert samples
    train_drowsy : int
        Original number of drowsy samples
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: jobid, tag, method, ratio, seed,
        orig_neg, orig_pos, after_neg, after_pos, after_total, after_ratio
    """
    results = []
    
    pattern = f"{job_prefix}*.OU" if job_prefix else "*.OU"
    
    for logfile in sorted(log_dir.glob(pattern)):
        jobid = logfile.stem.split(".")[0]
        try:
            content = logfile.read_text()
        except Exception:
            continue
        
        # Extract tag
        tag_match = re.search(r"tag=(\S+)", content)
        tag = tag_match.group(1) if tag_match else "unknown"
        
        # Extract original distribution
        orig_match = re.search(r"train n=(\d+) pos=(\d+)", content)
        if not orig_match:
            continue
        
        orig_total = int(orig_match.group(1))
        orig_pos = int(orig_match.group(2))
        orig_neg = orig_total - orig_pos
        
        # Extract after-sampling distribution
        dist_match = re.search(
            r"Class distribution after oversampling: \[(\d+)\s+(\d+)\]",
            content
        )
        if dist_match:
            after_neg = int(dist_match.group(1))
            after_pos = int(dist_match.group(2))
        else:
            after_neg = orig_neg
            after_pos = orig_pos
        
        # Parse method and ratio from tag
        method = tag.replace("imbal_v2_", "").split("_seed")[0]
        method = re.sub(r"_ratio\d+_\d+", "", method)
        
        ratio_match = re.search(r"ratio(\d+)_(\d+)", tag)
        ratio = float(f"{ratio_match.group(1)}.{ratio_match.group(2)}") if ratio_match else None
        
        seed_match = re.search(r"seed(\d+)", tag)
        seed = int(seed_match.group(1)) if seed_match else 42
        
        results.append({
            "jobid": jobid,
            "tag": tag,
            "method": method,
            "ratio": ratio,
            "seed": seed,
            "orig_neg": orig_neg,
            "orig_pos": orig_pos,
            "after_neg": after_neg,
            "after_pos": after_pos,
            "after_total": after_neg + after_pos,
            "after_ratio": after_pos / after_neg if after_neg > 0 else 0,
        })
    
    return pd.DataFrame(results)


def calculate_sampling_distribution(
    method: str,
    ratio: float,
    train_alert: int = DEFAULT_TRAIN_ALERT,
    train_drowsy: int = DEFAULT_TRAIN_DROWSY
) -> dict:
    """Calculate expected training data distribution after sampling.
    
    Parameters
    ----------
    method : str
        Sampling method name (e.g., "smote", "smote_tomek", "undersample_rus")
    ratio : float
        Target ratio of minority to majority class
    train_alert : int
        Original number of alert (majority) samples
    train_drowsy : int
        Original number of drowsy (minority) samples
        
    Returns
    -------
    dict
        Expected distribution with keys: method, ratio, alert, drowsy, total, drowsy_pct
    """
    alert = train_alert
    drowsy = train_drowsy
    
    if method == "baseline":
        pass
    elif method.startswith("smote") or method.startswith("adasyn"):
        # SMOTE-based oversampling
        new_drowsy = int(alert * ratio)
        drowsy = max(drowsy, new_drowsy)
        
        # Handle hybrid methods
        if "tomek" in method:
            # Tomek links remove borderline samples
            reduction = 0.01
            alert = int(alert * (1 - reduction))
            drowsy = int(drowsy * (1 - reduction * 0.5))
        elif "enn" in method:
            # ENN removes noisy samples
            reduction = 0.02
            alert = int(alert * (1 - reduction))
            drowsy = int(drowsy * (1 - reduction * 0.3))
        elif "rus" in method:
            # RUS undersamples majority class
            new_alert = int(drowsy / ratio) if ratio > 0 else alert
            alert = min(alert, new_alert)
            
    elif method.startswith("undersample"):
        # Pure undersampling
        new_alert = int(drowsy / ratio) if ratio > 0 else alert
        alert = min(alert, new_alert)
    
    total = alert + drowsy
    return {
        "method": method,
        "ratio": ratio,
        "alert": alert,
        "drowsy": drowsy,
        "total": total,
        "drowsy_pct": drowsy / total * 100 if total > 0 else 0,
    }


def calculate_batch_distributions(
    methods: List[str],
    ratio: float,
    train_alert: int = DEFAULT_TRAIN_ALERT,
    train_drowsy: int = DEFAULT_TRAIN_DROWSY
) -> pd.DataFrame:
    """Calculate distributions for multiple methods.
    
    Parameters
    ----------
    methods : List[str]
        List of sampling method names
    ratio : float
        Target ratio
    train_alert : int
        Original alert samples
    train_drowsy : int
        Original drowsy samples
        
    Returns
    -------
    pd.DataFrame
        DataFrame with distribution for each method
    """
    data = [
        calculate_sampling_distribution(m, ratio, train_alert, train_drowsy)
        for m in methods
    ]
    return pd.DataFrame(data)


def compare_actual_vs_theoretical(
    df_actual: pd.DataFrame,
    ratio: Optional[float] = None,
    train_alert: int = DEFAULT_TRAIN_ALERT,
    train_drowsy: int = DEFAULT_TRAIN_DROWSY
) -> pd.DataFrame:
    """Compare actual sampling results vs theoretical expectations.
    
    Parameters
    ----------
    df_actual : pd.DataFrame
        Actual results from extract_sampling_distribution()
    ratio : float, optional
        Target ratio. If None, inferred from actual data.
    train_alert : int
        Original alert samples
    train_drowsy : int
        Original drowsy samples
        
    Returns
    -------
    pd.DataFrame
        Comparison DataFrame with actual vs theoretical values
    """
    if ratio is None:
        ratio_series = df_actual["ratio"].dropna()
        ratio = ratio_series.mode().iloc[0] if not ratio_series.empty else 0.5
    
    methods = df_actual["method"].unique().tolist()
    
    comparison_data = []
    for method in methods:
        actual = df_actual[df_actual["method"] == method]
        if actual.empty:
            continue
        
        theoretical = calculate_sampling_distribution(
            method, ratio, train_alert, train_drowsy
        )
        
        actual_ratio = actual["after_ratio"].mean()
        theo_ratio = (
            theoretical["drowsy"] / theoretical["alert"]
            if theoretical["alert"] > 0 else 0
        )
        
        comparison_data.append({
            "method": method,
            "actual_ratio": actual_ratio,
            "theoretical_ratio": theo_ratio,
            "difference": actual_ratio - theo_ratio,
            "actual_alert_mean": actual["after_neg"].mean(),
            "actual_drowsy_mean": actual["after_pos"].mean(),
            "theoretical_alert": theoretical["alert"],
            "theoretical_drowsy": theoretical["drowsy"],
            "n_samples": len(actual),
        })
    
    return pd.DataFrame(comparison_data)
