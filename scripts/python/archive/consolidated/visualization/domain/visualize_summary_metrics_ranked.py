#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_summary_metrics_ranked.py
===================================
Create visualization plots comparing multiple ranking methods.

This script generates:
1. Bar chart comparing metrics across ranking methods
2. Heatmap showing performance by method × level × mode

Input:
- summary_ranked_test.csv

Output:
- summary_metrics_ranked_bar.png: Grouped bar chart
- summary_metrics_ranked_heatmap.png: Heatmap comparison
"""

import logging
from pathlib import Path

# Setup matplotlib before importing pyplot
from src.utils.visualization.setup import setup_matplotlib_headless
setup_matplotlib_headless()

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from src import config as cfg
from src.utils.io.data_io import load_csv
from src.utils.visualization.visualization import save_figure, plot_grouped_bar_chart_raw
from src.utils.visualization.color_palettes import (
    RANKING_METHOD_COLORS as METHOD_COLORS,
    DOMAIN_LEVEL_COLORS as LEVEL_COLORS,
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# Paths
IN_CSV = Path(cfg.RESULTS_DOMAIN_ANALYSIS_PATH) / "summary" / "csv" / "summary_ranked_test.csv"
POOLED_CSV = Path(cfg.RESULTS_DOMAIN_ANALYSIS_PATH) / "summary" / "csv" / "summary_40cases_test.csv"
OUT_DIR_BASE = Path(cfg.RESULTS_DOMAIN_ANALYSIS_PATH) / "summary" / "png"
OUT_DIR_BASE.mkdir(parents=True, exist_ok=True)

# Comparison folder (for cross-method comparisons)
OUT_DIR_COMPARISON = OUT_DIR_BASE / "comparison"
OUT_DIR_COMPARISON.mkdir(parents=True, exist_ok=True)

# Metrics to plot
METRICS = ["accuracy", "recall", "precision", "f1", "f2", "auc", "auc_pr"]
METRICS_THR = ["recall_thr", "precision_thr", "f1_thr", "f2_thr"]


def plot_method_comparison_bar(df: pd.DataFrame, metric: str, mode: str = "source_only") -> plt.Figure:
    """Create grouped bar chart comparing ranking methods.
    
    Parameters
    ----------
    df : pd.DataFrame
        Summary data with ranking_method, level, and metric columns.
    metric : str
        Metric to plot.
    mode : str
        Training mode to filter by.
    
    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    # Filter by mode
    plot_df = df[df["mode"] == mode].copy() if "mode" in df.columns else df.copy()
    
    if len(plot_df) == 0:
        logger.warning(f"No data for mode={mode}")
        return None
    
    # Get unique methods and levels
    methods = plot_df["ranking_method"].unique() if "ranking_method" in plot_df.columns else ["unknown"]
    levels = ["out_domain", "mid_domain", "in_domain"]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(levels))
    width = 0.25
    n_methods = len(methods)
    
    for i, method in enumerate(methods):
        method_df = plot_df[plot_df["ranking_method"] == method] if "ranking_method" in plot_df.columns else plot_df
        
        values = []
        for level in levels:
            level_df = method_df[method_df["level"] == level] if "level" in method_df.columns else method_df
            if len(level_df) > 0 and metric in level_df.columns:
                values.append(level_df[metric].mean())
            else:
                values.append(0)
        
        offset = (i - (n_methods - 1) / 2) * width
        color = METHOD_COLORS.get(method, f"C{i}")
        ax.bar(x + offset, values, width, label=method, color=color, alpha=0.8)
    
    ax.set_xlabel("Level", fontsize=12)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_title(f"{metric.upper()} by Ranking Method ({mode})", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(["High\n(Outliers)", "mid_domain", "Low\n(Central)"])
    ax.legend(title="Ranking Method")
    ax.grid(axis="y", alpha=0.3)
    
    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)
    
    fig.tight_layout()
    return fig


def plot_multi_metric_comparison(df: pd.DataFrame, mode: str = "source_only") -> plt.Figure:
    """Create multi-panel bar chart for all metrics.
    
    Parameters
    ----------
    df : pd.DataFrame
        Summary data.
    mode : str
        Training mode to filter by.
    
    Returns
    -------
    plt.Figure
        Matplotlib figure with subplots.
    """
    plot_df = df[df["mode"] == mode].copy() if "mode" in df.columns else df.copy()
    
    if len(plot_df) == 0:
        logger.warning(f"No data for mode={mode}")
        return None
    
    metrics = [m for m in METRICS if m in plot_df.columns]
    n_metrics = len(metrics)
    
    if n_metrics == 0:
        logger.warning("No metrics found in data")
        return None
    
    # Create subplot grid
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_metrics > 1 else [axes]
    
    methods = plot_df["ranking_method"].unique() if "ranking_method" in plot_df.columns else ["unknown"]
    levels = ["out_domain", "mid_domain", "in_domain"]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        x = np.arange(len(levels))
        width = 0.25
        n_methods = len(methods)
        
        for i, method in enumerate(methods):
            method_df = plot_df[plot_df["ranking_method"] == method] if "ranking_method" in plot_df.columns else plot_df
            
            values = []
            for level in levels:
                level_df = method_df[method_df["level"] == level] if "level" in method_df.columns else method_df
                if len(level_df) > 0 and metric in level_df.columns:
                    val = level_df[metric].mean()
                    values.append(val if pd.notna(val) else 0)
                else:
                    values.append(0)
            
            offset = (i - (n_methods - 1) / 2) * width
            color = METHOD_COLORS.get(method, f"C{i}")
            ax.bar(x + offset, values, width, label=method, color=color, alpha=0.8)
        
        ax.set_title(metric.upper(), fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(["out_domain", "mid_domain", "in_domain"], fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 1.0)
        
        if idx == 0:
            ax.legend(title="Method", fontsize=8, title_fontsize=9)
    
    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(f"Metrics Comparison by Ranking Method (mode={mode})", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_heatmap_comparison(df: pd.DataFrame, metric: str = "f1") -> plt.Figure:
    """Create heatmap showing metric values across methods and levels.
    
    Parameters
    ----------
    df : pd.DataFrame
        Summary data.
    metric : str
        Metric to visualize.
    
    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    # Pivot table: rows=method, columns=level, values=metric
    if "ranking_method" not in df.columns or "level" not in df.columns:
        logger.warning("Missing ranking_method or level columns")
        return None
    
    # Filter to source_only mode for main comparison
    plot_df = df[df["mode"] == "source_only"].copy() if "mode" in df.columns else df.copy()
    
    if metric not in plot_df.columns:
        logger.warning(f"Metric {metric} not found")
        return None
    
    # Create pivot table
    pivot = plot_df.pivot_table(
        index="ranking_method",
        columns="level",
        values=metric,
        aggfunc="mean"
    )
    
    # Reorder columns
    col_order = [c for c in ["in_domain", "mid_domain", "out_domain"] if c in pivot.columns]
    pivot = pivot[col_order]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        ax=ax,
        vmin=0,
        vmax=max(0.5, pivot.max().max()),
        cbar_kws={"label": metric.upper()}
    )
    
    ax.set_title(f"{metric.upper()} by Ranking Method and Level (source_only)", fontsize=12)
    ax.set_xlabel("Level (Low=Central, High=Outlier)")
    ax.set_ylabel("Ranking Method")
    
    fig.tight_layout()
    return fig


def plot_single_method_bar(df: pd.DataFrame, metric: str, method_name: str) -> plt.Figure:
    """Create bar chart for a single ranking method.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data for a single ranking method.
    metric : str
        Metric to plot.
    method_name : str
        Name of the ranking method.
    
    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    if metric not in df.columns:
        return None
    
    levels = ["out_domain", "mid_domain", "in_domain"]
    modes = df["mode"].unique().tolist() if "mode" in df.columns else ["unknown"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(levels))
    width = 0.25
    n_modes = len(modes)
    
    mode_colors = {
        "source_only": "#1f77b4",
        "target_only": "#ff7f0e",
        "pooled": "#2ca02c",
    }
    
    for i, mode in enumerate(modes):
        mode_df = df[df["mode"] == mode] if "mode" in df.columns else df
        
        values = []
        for level in levels:
            level_df = mode_df[mode_df["level"] == level] if "level" in mode_df.columns else mode_df
            if len(level_df) > 0 and metric in level_df.columns:
                val = level_df[metric].mean()
                values.append(val if pd.notna(val) else 0)
            else:
                values.append(0)
        
        offset = (i - (n_modes - 1) / 2) * width
        color = mode_colors.get(mode, f"C{i}")
        ax.bar(x + offset, values, width, label=mode, color=color, alpha=0.8)
    
    ax.set_xlabel("Level", fontsize=12)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_title(f"{method_name}: {metric.upper()} by Level", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(["High\n(Outliers)", "mid_domain", "Low\n(Central)"])
    ax.legend(title="Training Mode")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)
    
    fig.tight_layout()
    return fig


def plot_single_method_multi_metric(df: pd.DataFrame, method_name: str) -> plt.Figure:
    """Create multi-panel bar chart for a single ranking method (like summary_metrics_bar.png).
    
    Parameters
    ----------
    df : pd.DataFrame
        Data for a single ranking method.
    method_name : str
        Name of the ranking method.
    
    Returns
    -------
    plt.Figure
        Matplotlib figure with subplots.
    """
    metrics = [m for m in METRICS if m in df.columns]
    n_metrics = len(metrics)
    
    if n_metrics == 0:
        return None
    
    # Create subplot grid
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_metrics > 1 else [axes]
    
    levels = ["out_domain", "mid_domain", "in_domain"]
    modes = df["mode"].unique().tolist() if "mode" in df.columns else ["source_only"]
    
    mode_colors = {
        "source_only": "#1f77b4",
        "target_only": "#ff7f0e",
        "pooled": "#2ca02c",
    }
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        x = np.arange(len(levels))
        width = 0.25
        n_modes = len(modes)
        
        for i, mode in enumerate(modes):
            mode_df = df[df["mode"] == mode] if "mode" in df.columns else df
            
            values = []
            for level in levels:
                level_df = mode_df[mode_df["level"] == level] if "level" in mode_df.columns else mode_df
                if len(level_df) > 0 and metric in level_df.columns:
                    val = level_df[metric].mean()
                    values.append(val if pd.notna(val) else 0)
                else:
                    values.append(0)
            
            offset = (i - (n_modes - 1) / 2) * width
            color = mode_colors.get(mode, f"C{i}")
            ax.bar(x + offset, values, width, label=mode, color=color, alpha=0.8)
        
        ax.set_title(metric.upper(), fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(["out_domain", "mid_domain", "in_domain"], fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 1.0)
        
        if idx == 0:
            ax.legend(title="Mode", fontsize=8, title_fontsize=9)
    
    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(f"{method_name}: Summary Metrics by Level", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def main():
    print("=" * 60)
    print("Visualizing ranked domain analysis results")
    print("=" * 60)
    
    # Load data
    if not IN_CSV.exists():
        print(f"[ERROR] Input file not found: {IN_CSV}")
        print("Please run collect_evaluation_metrics_ranked.py first.")
        return
    
    df = load_csv(str(IN_CSV))
    print(f"[INFO] Loaded {len(df)} records from {IN_CSV}")
    
    ranking_methods = []
    if "ranking_method" in df.columns:
        ranking_methods = df['ranking_method'].unique().tolist()
        print(f"[INFO] Ranking methods: {ranking_methods}")
    if "mode" in df.columns:
        print(f"[INFO] Modes: {df['mode'].unique().tolist()}")
    if "level" in df.columns:
        print(f"[INFO] Levels: {df['level'].unique().tolist()}")
    
    # Normalize distance column to handle cases like "centroid_umap_dtw" -> "dtw"
    if "distance" in df.columns:
        print("[INFO] Normalizing distance column...")
        df["distance"] = df["distance"].apply(lambda x: x.split("_")[-1] if isinstance(x, str) else x)
        print(f"[INFO] Distances after normalization: {df['distance'].unique().tolist()}")

    # Load pooled data
    df_pooled = pd.DataFrame()
    if POOLED_CSV.exists():
        df_all_40 = load_csv(str(POOLED_CSV))
        if "mode" in df_all_40.columns:
            df_pooled = df_all_40[df_all_40["mode"] == "pooled"].copy()
            # Normalize pooled distance (e.g., "lof_dtw" -> "dtw", matching ranked data normalization)
            if "distance" in df_pooled.columns:
                 df_pooled["distance"] = df_pooled["distance"].apply(lambda x: x.split("_")[-1] if isinstance(x, str) else x)
            print(f"[INFO] Loaded {len(df_pooled)} pooled records from {POOLED_CSV}")
    else:
        print(f"[WARN] Pooled data file not found: {POOLED_CSV}")

    # === Cross-method comparison plots (saved in comparison folder) ===
    print("\n" + "-" * 40)
    print("[INFO] Creating cross-method comparison plots...")
    print("-" * 40)
    
    # Plot 1: Multi-metric comparison (source_only)
    print("\n[INFO] Creating multi-metric bar chart...")
    fig = plot_multi_metric_comparison(df, mode="source_only")
    if fig:
        out_path = OUT_DIR_COMPARISON / "summary_metrics_ranked_bar.png"
        save_figure(fig, str(out_path), dpi=200)
        print(f"  Saved: {out_path}")
    
    # Plot 2: Heatmap for F1
    print("\n[INFO] Creating F1 heatmap...")
    fig = plot_heatmap_comparison(df, metric="f1")
    if fig:
        out_path = OUT_DIR_COMPARISON / "summary_metrics_ranked_heatmap_f1.png"
        save_figure(fig, str(out_path), dpi=200)
        print(f"  Saved: {out_path}")
    
    # Plot 3: Heatmap for Recall
    print("\n[INFO] Creating Recall heatmap...")
    fig = plot_heatmap_comparison(df, metric="recall")
    if fig:
        out_path = OUT_DIR_COMPARISON / "summary_metrics_ranked_heatmap_recall.png"
        save_figure(fig, str(out_path), dpi=200)
        print(f"  Saved: {out_path}")
    
    # Plot 4: Individual metric comparison
    for metric in ["f1", "recall", "f2_thr"]:
        if metric in df.columns:
            print(f"\n[INFO] Creating {metric} comparison...")
            fig = plot_method_comparison_bar(df, metric=metric, mode="source_only")
            if fig:
                out_path = OUT_DIR_COMPARISON / f"summary_{metric}_ranked_bar.png"
                save_figure(fig, str(out_path), dpi=200)
                print(f"  Saved: {out_path}")
    
    # === Per-method plots (saved in method-specific folders) ===
    print("\n" + "-" * 40)
    print("[INFO] Creating per-method plots...")
    print("-" * 40)
    
    for method in ranking_methods:
        print(f"\n[INFO] Creating plots for: {method}")
        
        # Create method-specific folder
        method_dir = OUT_DIR_BASE / method
        method_dir.mkdir(parents=True, exist_ok=True)
        
        # Filter data for this method
        method_df = df[df["ranking_method"] == method].copy()
        
        if len(method_df) == 0:
            print(f"  [WARN] No data for {method}, skipping...")
            continue
        
        # Add pooled data to method_df for visualization
        if not df_pooled.empty:
            # Ensure columns match
            common_cols = method_df.columns.intersection(df_pooled.columns)
            
            # Create a copy of pooled data with the current ranking method assigned
            # This is needed so it doesn't get filtered out if we were filtering by method
            # (though plot_grouped_bar_chart_raw doesn't filter by method column)
            pooled_to_add = df_pooled[common_cols].copy()
            pooled_to_add["ranking_method"] = method
            
            # Concatenate
            method_df_with_pooled = pd.concat([method_df, pooled_to_add], ignore_index=True)
        else:
            method_df_with_pooled = method_df
            
        print(f"  Records: {len(method_df)} (Total with pooled: {len(method_df_with_pooled)})")
        
        # Plot summary_metrics_bar.png using the SAME function as original
        fig = plot_grouped_bar_chart_raw(
            data=method_df_with_pooled,
            metrics=METRICS,
            modes=["pooled", "source_only", "target_only"],
            distance_col="distance",
            level_col="level",
            baseline_rates={"auc_pr": method_df["pos_rate"].mean() if "pos_rate" in method_df.columns else 0.033}
        )
        if fig:
            out_path = method_dir / "summary_metrics_bar.png"
            save_figure(fig, str(out_path), dpi=200)
            print(f"    Saved: {out_path}")
            plt.close(fig)
        
        # Plot individual metrics for this method
        for metric in ["f1", "recall", "f2_thr", "auc", "auc_pr"]:
            if metric in method_df.columns:
                fig = plot_single_method_bar(method_df, metric=metric, method_name=method)
                if fig:
                    out_path = method_dir / f"{method}_{metric}_bar.png"
                    save_figure(fig, str(out_path), dpi=200)
                    print(f"    Saved: {out_path}")
                    plt.close(fig)
        
        # Summary stats
        if "f1" in method_df.columns:
            mean_f1 = method_df["f1"].mean()
            print(f"  Mean F1: {mean_f1:.4f}")
    
    print("\n" + "=" * 60)
    print("[DONE] Visualization complete")
    print(f"  Comparison plots: {OUT_DIR_COMPARISON}")
    print(f"  Per-method plots: {OUT_DIR_BASE}/<method>/")
    print("=" * 60)


if __name__ == "__main__":
    main()
