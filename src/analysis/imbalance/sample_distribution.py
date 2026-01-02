"""Sample Distribution Visualization for Imbalance Methods.

This module provides visualization functions to compare sample counts
across different imbalanced data handling methods.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import logging
logger = logging.getLogger(__name__)


def create_sample_distribution_data() -> pd.DataFrame:
    """Create DataFrame with sample distribution data from experiments.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: method, jobid, alert, drowsy, total, ratio
    """
    data = [
        # method, jobid, alert, drowsy
        ("Baseline (class_weight)", "14468417", 35522, 1445),
        ("SMOTE+Tomek", "14468418", 35380, 35380),
        ("SMOTE+ENN", "14468419", 25259, 34244),
        ("SMOTE+RUS", "14468421", 22201, 17761),
        ("EasyEnsemble", "14468501", 35522, 1445),
        ("Undersample-RUS", "14471478", 4378, 1445),
        ("Undersample-Tomek", "14471479", 35400, 1445),
        ("Jittering+Scaling", "14471460", 35522, 11722),  # augmented minority: 1445 -> 11722
    ]
    
    df = pd.DataFrame(data, columns=["method", "jobid", "alert", "drowsy"])
    df["total"] = df["alert"] + df["drowsy"]
    df["ratio"] = df["alert"] / df["drowsy"]
    df["drowsy_pct"] = df["drowsy"] / df["total"] * 100
    df["alert_change_pct"] = (df["alert"] / 35522 - 1) * 100
    df["drowsy_change_pct"] = (df["drowsy"] / 1445 - 1) * 100
    
    return df


def create_split_distribution_data() -> pd.DataFrame:
    """Create DataFrame with train/val/test split distribution data.
    
    Data splits are consistent across all methods (before oversampling).
    Oversampling is applied only to training data.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with split information
    """
    # Original split (same for all methods, before oversampling)
    original_splits = {
        "split": ["Train (Original)", "Validation", "Test", "Total"],
        "total": [36967, 12322, 12323, 61612],
        "alert": [35522, 11841, 11841, 59204],
        "drowsy": [1445, 481, 482, 2408],
        "drowsy_pct": [3.9, 3.9, 3.9, 3.9],
    }
    
    return pd.DataFrame(original_splits)


def create_train_after_sampling_data() -> pd.DataFrame:
    """Create DataFrame with training data after each sampling method.
    
    Note: Validation and Test sets remain unchanged.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with training data after sampling for each method
    """
    data = [
        # method, train_alert, train_drowsy, val_alert, val_drowsy, test_alert, test_drowsy
        ("Baseline", 35522, 1445, 11841, 481, 11841, 482),
        ("SMOTE+Tomek", 35380, 35380, 11841, 481, 11841, 482),
        ("SMOTE+ENN", 25259, 34244, 11841, 481, 11841, 482),
        ("SMOTE+RUS", 22201, 17761, 11841, 481, 11841, 482),
        ("EasyEnsemble", 35522, 1445, 11841, 481, 11841, 482),
        ("Undersample-RUS", 4378, 1445, 11841, 481, 11841, 482),
        ("Undersample-Tomek", 35400, 1445, 11841, 481, 11841, 482),
        ("Jittering+Scaling", 35522, 11722, 11841, 481, 11841, 482),  # augmented minority: 1445 -> 11722
    ]
    
    df = pd.DataFrame(data, columns=[
        "method", "train_alert", "train_drowsy", 
        "val_alert", "val_drowsy", "test_alert", "test_drowsy"
    ])
    
    df["train_total"] = df["train_alert"] + df["train_drowsy"]
    df["val_total"] = df["val_alert"] + df["val_drowsy"]
    df["test_total"] = df["test_alert"] + df["test_drowsy"]
    df["train_drowsy_pct"] = df["train_drowsy"] / df["train_total"] * 100
    df["val_drowsy_pct"] = df["val_drowsy"] / df["val_total"] * 100
    df["test_drowsy_pct"] = df["test_drowsy"] / df["test_total"] * 100
    
    return df


def plot_sample_counts_bar(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 8),
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Create stacked bar chart of sample counts per method.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from create_sample_distribution_data()
    figsize : tuple
        Figure size
    output_path : Path, optional
        If provided, save figure to this path
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(df))
    width = 0.6
    
    # Stacked bars
    bars_alert = ax.bar(x, df["alert"], width, label="Alert (Majority)", 
                        color="#3498db", edgecolor="black")
    bars_drowsy = ax.bar(x, df["drowsy"], width, bottom=df["alert"], 
                         label="Drowsy (Minority)", color="#e74c3c", edgecolor="black")
    
    # Original baseline line
    original_total = 35522 + 1445
    ax.axhline(original_total, color="gray", linestyle="--", linewidth=2, 
               label=f"Original Total ({original_total:,})")
    
    # Labels
    ax.set_xlabel("Sampling Method", fontsize=12)
    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.set_title("Sample Distribution by Imbalance Handling Method", 
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(df["method"], rotation=45, ha="right", fontsize=10)
    ax.legend(loc="upper right", fontsize=10)
    
    # Add total count labels on top of bars
    for i, (alert, drowsy, total) in enumerate(zip(df["alert"], df["drowsy"], df["total"])):
        ax.text(i, total + 1000, f"{total:,}", ha="center", va="bottom", fontsize=9)
        # Add ratio annotation
        ratio = alert / drowsy
        ax.text(i, total / 2, f"1:{ratio:.1f}", ha="center", va="center", 
                fontsize=8, color="white", fontweight="bold")
    
    ax.set_ylim(0, max(df["total"]) * 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {output_path}")
    
    return fig


def plot_class_ratio_comparison(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 6),
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Create bar chart comparing minority class ratio across methods.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from create_sample_distribution_data()
    figsize : tuple
        Figure size
    output_path : Path, optional
        If provided, save figure to this path
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort by drowsy percentage
    df_sorted = df.sort_values("drowsy_pct", ascending=True)
    
    # Color by balance quality
    colors = []
    for pct in df_sorted["drowsy_pct"]:
        if pct >= 40:  # Near balanced
            colors.append("#2ecc71")
        elif pct >= 20:  # Moderate
            colors.append("#f39c12")
        elif pct >= 10:  # Slight improvement
            colors.append("#e67e22")
        else:  # Still highly imbalanced
            colors.append("#e74c3c")
    
    bars = ax.barh(df_sorted["method"], df_sorted["drowsy_pct"], color=colors, 
                   edgecolor="black")
    
    # Original baseline
    original_pct = 1445 / (35522 + 1445) * 100
    ax.axvline(original_pct, color="gray", linestyle="--", linewidth=2,
               label=f"Original ({original_pct:.1f}%)")
    
    # Ideal balance line
    ax.axvline(50, color="green", linestyle=":", linewidth=2, alpha=0.7,
               label="Perfect Balance (50%)")
    
    # Value labels
    for bar, pct in zip(bars, df_sorted["drowsy_pct"]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{pct:.1f}%", va="center", fontsize=10)
    
    ax.set_xlabel("Minority Class (Drowsy) Percentage", fontsize=12)
    ax.set_title("Class Balance Comparison Across Methods", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim(0, 60)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {output_path}")
    
    return fig


def plot_sample_change_waterfall(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 8),
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Create waterfall-style chart showing sample count changes.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from create_sample_distribution_data()
    figsize : tuple
        Figure size
    output_path : Path, optional
        If provided, save figure to this path
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Exclude baseline for change visualization
    df_methods = df[df["method"] != "Baseline (class_weight)"].copy()
    
    # Left: Alert class changes
    colors_alert = ["#e74c3c" if x < 0 else "#2ecc71" for x in df_methods["alert_change_pct"]]
    ax1.barh(df_methods["method"], df_methods["alert_change_pct"], color=colors_alert, 
             edgecolor="black")
    ax1.axvline(0, color="black", linewidth=1)
    ax1.set_xlabel("Change from Original (%)", fontsize=11)
    ax1.set_title("Alert (Majority) Class Change", fontsize=12, fontweight="bold")
    
    for i, (method, pct) in enumerate(zip(df_methods["method"], df_methods["alert_change_pct"])):
        ax1.text(pct + (2 if pct >= 0 else -2), i, f"{pct:+.0f}%", 
                 va="center", ha="left" if pct >= 0 else "right", fontsize=9)
    
    # Right: Drowsy class changes
    colors_drowsy = ["#2ecc71" for _ in df_methods["drowsy_change_pct"]]  # All increases are good
    ax2.barh(df_methods["method"], df_methods["drowsy_change_pct"], color=colors_drowsy,
             edgecolor="black")
    ax2.axvline(0, color="black", linewidth=1)
    ax2.set_xlabel("Change from Original (%)", fontsize=11)
    ax2.set_title("Drowsy (Minority) Class Change", fontsize=12, fontweight="bold")
    
    for i, (method, pct) in enumerate(zip(df_methods["method"], df_methods["drowsy_change_pct"])):
        if pct > 100:
            label = f"+{pct:.0f}%"
        else:
            label = f"{pct:+.0f}%"
        ax2.text(pct + 50, i, label, va="center", ha="left", fontsize=9)
    
    fig.suptitle("Sample Count Changes by Method (vs Original)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {output_path}")
    
    return fig


def plot_method_summary_dashboard(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (16, 10),
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Create comprehensive dashboard with all visualizations.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from create_sample_distribution_data()
    figsize : tuple
        Figure size
    output_path : Path, optional
        If provided, save figure to this path
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)
    
    # Panel 1: Stacked bar (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(df))
    width = 0.6
    ax1.bar(x, df["alert"], width, label="Alert", color="#3498db", edgecolor="black")
    ax1.bar(x, df["drowsy"], width, bottom=df["alert"], label="Drowsy", 
            color="#e74c3c", edgecolor="black")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["method"], rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Samples")
    ax1.set_title("Sample Counts by Method", fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x/1000)}K"))
    
    # Panel 2: Minority percentage (top-right)
    ax2 = fig.add_subplot(gs[0, 1])
    df_sorted = df.sort_values("drowsy_pct")
    colors = ["#2ecc71" if p >= 30 else "#f39c12" if p >= 10 else "#e74c3c" 
              for p in df_sorted["drowsy_pct"]]
    ax2.barh(df_sorted["method"], df_sorted["drowsy_pct"], color=colors, edgecolor="black")
    ax2.axvline(3.9, color="gray", linestyle="--", label="Original (3.9%)")
    ax2.axvline(50, color="green", linestyle=":", alpha=0.5, label="Balanced (50%)")
    ax2.set_xlabel("Minority %")
    ax2.set_title("Minority Class Percentage", fontweight="bold")
    ax2.legend(loc="lower right", fontsize=8)
    ax2.set_xlim(0, 55)
    
    # Panel 3: Total sample comparison (bottom-left)
    ax3 = fig.add_subplot(gs[1, 0])
    df_by_total = df.sort_values("total")
    colors = ["#3498db" if t >= 30000 else "#f39c12" if t >= 10000 else "#e74c3c"
              for t in df_by_total["total"]]
    bars = ax3.barh(df_by_total["method"], df_by_total["total"], color=colors, edgecolor="black")
    ax3.axvline(36967, color="gray", linestyle="--", label="Original (36,967)")
    ax3.set_xlabel("Total Samples")
    ax3.set_title("Total Training Samples", fontweight="bold")
    ax3.legend(loc="lower right", fontsize=8)
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x/1000)}K"))
    for bar, total in zip(bars, df_by_total["total"]):
        ax3.text(bar.get_width() + 500, bar.get_y() + bar.get_height()/2,
                 f"{total:,}", va="center", fontsize=8)
    
    # Panel 4: Summary table (bottom-right)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    
    table_data = []
    for _, row in df.iterrows():
        table_data.append([
            row["method"],
            f"{row['alert']:,}",
            f"{row['drowsy']:,}",
            f"{row['total']:,}",
            f"1:{row['ratio']:.1f}",
            f"{row['drowsy_pct']:.1f}%"
        ])
    
    table = ax4.table(
        cellText=table_data,
        colLabels=["Method", "Alert", "Drowsy", "Total", "Ratio", "Min %"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.4)
    
    # Color header
    for j in range(6):
        table[(0, j)].set_facecolor("#34495e")
        table[(0, j)].set_text_props(color="white", fontweight="bold")
    
    ax4.set_title("Sample Distribution Summary", fontweight="bold", y=0.95)
    
    fig.suptitle("Imbalance Handling Methods - Sample Distribution Analysis", 
                 fontsize=16, fontweight="bold", y=0.98)
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {output_path}")
    
    return fig


def generate_all_visualizations(output_dir: str = "results/analysis/imbalance") -> Dict[str, str]:
    """Generate all sample distribution visualizations.
    
    Parameters
    ----------
    output_dir : str
        Output directory for figures
        
    Returns
    -------
    dict
        Dictionary mapping figure names to file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create data
    df = create_sample_distribution_data()
    
    saved_files = {}
    
    # Generate each visualization
    fig1 = plot_sample_counts_bar(df, output_path=output_path / "sample_counts_bar.png")
    plt.close(fig1)
    saved_files["sample_counts_bar"] = str(output_path / "sample_counts_bar.png")
    
    fig2 = plot_class_ratio_comparison(df, output_path=output_path / "class_ratio_comparison.png")
    plt.close(fig2)
    saved_files["class_ratio_comparison"] = str(output_path / "class_ratio_comparison.png")
    
    fig3 = plot_sample_change_waterfall(df, output_path=output_path / "sample_change_waterfall.png")
    plt.close(fig3)
    saved_files["sample_change_waterfall"] = str(output_path / "sample_change_waterfall.png")
    
    fig4 = plot_method_summary_dashboard(df, output_path=output_path / "sample_distribution_dashboard.png")
    plt.close(fig4)
    saved_files["sample_distribution_dashboard"] = str(output_path / "sample_distribution_dashboard.png")
    
    # Generate train/val/test split visualization
    fig5 = plot_train_val_test_split(output_path=output_path / "train_val_test_split.png")
    plt.close(fig5)
    saved_files["train_val_test_split"] = str(output_path / "train_val_test_split.png")
    
    # Save CSVs
    csv_path = output_path / "sample_distribution.csv"
    df.to_csv(csv_path, index=False)
    saved_files["csv"] = str(csv_path)
    
    # Save train/val/test split data
    df_split = create_train_after_sampling_data()
    split_csv_path = output_path / "train_val_test_distribution.csv"
    df_split.to_csv(split_csv_path, index=False)
    saved_files["split_csv"] = str(split_csv_path)
    
    print(f"\n{'='*60}")
    print("SAMPLE DISTRIBUTION VISUALIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"\nOutput directory: {output_path}")
    print(f"\nGenerated files:")
    for name, path in saved_files.items():
        print(f"  - {name}: {path}")
    
    return saved_files


def plot_train_val_test_split(
    figsize: Tuple[int, int] = (16, 10),
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Create visualization of train/val/test split across methods.
    
    Parameters
    ----------
    figsize : tuple
        Figure size
    output_path : Path, optional
        If provided, save figure to this path
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    df = create_train_after_sampling_data()
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
    
    # Panel 1: Original split (pie chart style info)
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Grouped bar for original split
    splits = ["Train\n(Original)", "Validation", "Test"]
    alert_counts = [35522, 11841, 11841]
    drowsy_counts = [1445, 481, 482]
    
    x = np.arange(len(splits))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, alert_counts, width, label="Alert", color="#3498db", edgecolor="black")
    bars2 = ax1.bar(x + width/2, drowsy_counts, width, label="Drowsy", color="#e74c3c", edgecolor="black")
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(splits)
    ax1.set_ylabel("Samples")
    ax1.set_title("Original Data Split (Before Sampling)", fontweight="bold")
    ax1.legend()
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x/1000)}K" if x >= 1000 else str(int(x))))
    
    # Add count labels
    for bar, count in zip(bars1, alert_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200, 
                 f"{count:,}", ha="center", va="bottom", fontsize=8)
    for bar, count in zip(bars2, drowsy_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                 f"{count:,}", ha="center", va="bottom", fontsize=8)
    
    # Panel 2: Training data after sampling (horizontal grouped bar)
    ax2 = fig.add_subplot(gs[0, 1])
    
    y = np.arange(len(df))
    height = 0.35
    
    bars1 = ax2.barh(y - height/2, df["train_alert"], height, label="Alert", color="#3498db", edgecolor="black")
    bars2 = ax2.barh(y + height/2, df["train_drowsy"], height, label="Drowsy", color="#e74c3c", edgecolor="black")
    
    ax2.set_yticks(y)
    ax2.set_yticklabels(df["method"], fontsize=9)
    ax2.set_xlabel("Samples")
    ax2.set_title("Training Data After Sampling", fontweight="bold")
    ax2.legend(loc="lower right")
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x/1000)}K"))
    
    # Panel 3: Minority percentage by split for each method
    ax3 = fig.add_subplot(gs[1, 0])
    
    x = np.arange(len(df))
    width = 0.25
    
    bars1 = ax3.bar(x - width, df["train_drowsy_pct"], width, label="Train", color="#2ecc71", edgecolor="black")
    bars2 = ax3.bar(x, df["val_drowsy_pct"], width, label="Val", color="#f39c12", edgecolor="black")
    bars3 = ax3.bar(x + width, df["test_drowsy_pct"], width, label="Test", color="#9b59b6", edgecolor="black")
    
    ax3.set_xticks(x)
    ax3.set_xticklabels(df["method"], rotation=45, ha="right", fontsize=8)
    ax3.set_ylabel("Minority Class %")
    ax3.set_title("Minority % by Split (Val/Test Unchanged)", fontweight="bold")
    ax3.legend(loc="upper right")
    ax3.axhline(3.9, color="gray", linestyle="--", alpha=0.7, label="Original (3.9%)")
    ax3.set_ylim(0, 60)
    
    # Panel 4: Summary table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    
    table_data = []
    for _, row in df.iterrows():
        table_data.append([
            row["method"],
            f"{row['train_total']:,}",
            f"{row['train_drowsy_pct']:.1f}%",
            f"{row['val_total']:,}",
            f"{row['val_drowsy_pct']:.1f}%",
            f"{row['test_total']:,}",
            f"{row['test_drowsy_pct']:.1f}%",
        ])
    
    table = ax4.table(
        cellText=table_data,
        colLabels=["Method", "Train N", "Train %", "Val N", "Val %", "Test N", "Test %"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.1, 1.3)
    
    # Color header
    for j in range(7):
        table[(0, j)].set_facecolor("#34495e")
        table[(0, j)].set_text_props(color="white", fontweight="bold")
    
    # Highlight that Val/Test are unchanged
    for i in range(1, len(df) + 1):
        for j in [3, 4, 5, 6]:  # Val and Test columns
            table[(i, j)].set_facecolor("#ecf0f1")
    
    ax4.set_title("Train/Val/Test Distribution Summary", fontweight="bold", y=0.95)
    
    fig.suptitle("Data Split Analysis: Train/Validation/Test", fontsize=16, fontweight="bold", y=0.98)
    
    # Add note
    fig.text(0.5, 0.01, "Note: Sampling is applied ONLY to training data. Validation and Test sets remain unchanged.", 
             ha="center", fontsize=10, style="italic", color="gray")
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {output_path}")
    
    return fig


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_all_visualizations()
