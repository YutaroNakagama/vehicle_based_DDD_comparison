#!/usr/bin/env python3
"""Training Data After Sampling Visualization.

This script creates visualization of training data distribution
after applying various sampling methods from the latest experiments.
Based on src/analysis/imbalance/sample_distribution.py style.
"""

import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_sampling_distribution(log_dir: Path, job_prefix: str = "14618") -> pd.DataFrame:
    """Extract sampling distribution from job logs.
    
    Parameters
    ----------
    log_dir : Path
        Directory containing job logs
    job_prefix : str
        Job ID prefix to filter
    
    Returns
    -------
    pd.DataFrame
        DataFrame with sampling distribution data
    """
    results = []
    
    for logfile in sorted(log_dir.glob(f"{job_prefix}*.OU")):
        jobid = logfile.stem.split(".")[0]
        content = logfile.read_text()
        
        # Extract tag
        tag_match = re.search(r'tag=(\S+)', content)
        tag = tag_match.group(1) if tag_match else "unknown"
        
        # Extract original training data distribution
        orig_match = re.search(r'train n=(\d+) pos=(\d+)', content)
        if orig_match:
            orig_total = int(orig_match.group(1))
            orig_pos = int(orig_match.group(2))
            orig_neg = orig_total - orig_pos
        else:
            continue
        
        # Extract distribution after sampling
        dist_match = re.search(r'Class distribution after oversampling: \[(\d+)\s+(\d+)\]', content)
        if dist_match:
            after_neg = int(dist_match.group(1))  # alert (0)
            after_pos = int(dist_match.group(2))  # drowsy (1)
        else:
            # No sampling (baseline etc.)
            after_neg = orig_neg
            after_pos = orig_pos
        
        # Extract method name
        method = tag.replace("imbal_v2_", "").split("_seed")[0]
        method = re.sub(r'_ratio\d+_\d+', '', method)
        
        # Extract ratio
        ratio_match = re.search(r'ratio(\d+)_(\d+)', tag)
        ratio = float(f"{ratio_match.group(1)}.{ratio_match.group(2)}") if ratio_match else None
        
        # Extract seed
        seed_match = re.search(r'seed(\d+)', tag)
        seed = int(seed_match.group(1)) if seed_match else 42
        
        results.append({
            "jobid": jobid,
            "tag": tag,
            "method": method,
            "ratio": ratio,
            "seed": seed,
            "orig_alert": orig_neg,
            "orig_drowsy": orig_pos,
            "after_alert": after_neg,
            "after_drowsy": after_pos,
        })
    
    return pd.DataFrame(results)


def create_display_name(method: str, ratio: float) -> str:
    """Create display name for method with ratio."""
    method_names = {
        "baseline": "Baseline",
        "balanced_rf": "BalancedRF",
        "easy_ensemble": "EasyEnsemble",
        "smote": "SMOTE",
        "smote_tomek": "SMOTE+Tomek",
        "smote_enn": "SMOTE+ENN",
        "smote_rus": "SMOTE+RUS",
        "smote_balanced_rf": "SMOTE+BalancedRF",
        "undersample_rus": "RUS",
        "undersample_tomek": "Tomek Links",
        "undersample_enn": "ENN",
    }
    
    name = method_names.get(method, method)
    if ratio is not None:
        name = f"{name} (r={ratio})"
    return name


def plot_training_data_after_sampling(df: pd.DataFrame, output_path: Path) -> None:
    """Create 'Training Data After Sampling' visualization.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with sampling distribution data
    output_path : Path
        Output path for the figure
    """
    # Use seed=42 and get unique method+ratio combinations
    df_plot = df[df["seed"] == 42].copy()
    df_plot = df_plot.drop_duplicates(subset=["method", "ratio"])
    
    # Create display names
    df_plot["display_name"] = df_plot.apply(
        lambda x: create_display_name(x["method"], x["ratio"]), axis=1
    )
    
    # Sort by total samples after sampling
    df_plot["after_total"] = df_plot["after_alert"] + df_plot["after_drowsy"]
    df_plot = df_plot.sort_values("after_total", ascending=True)
    
    # Create figure with 2 panels
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.3, wspace=0.3)
    
    # Panel 1: Original data split (before sampling)
    ax1 = fig.add_subplot(gs[0, 0])
    
    splits = ["Train\n(Original)", "Validation", "Test"]
    alert_counts = [35522, 11841, 11841]
    drowsy_counts = [1445, 481, 482]
    
    x = np.arange(len(splits))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, alert_counts, width, label="Alert (Negative)", 
                    color="#3498db", edgecolor="black")
    bars2 = ax1.bar(x + width/2, drowsy_counts, width, label="Drowsy (Positive)", 
                    color="#e74c3c", edgecolor="black")
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(splits)
    ax1.set_ylabel("Samples")
    ax1.set_title("Original Data Split (Before Sampling)", fontweight="bold", fontsize=12)
    ax1.legend()
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x/1000)}K" if x >= 1000 else str(int(x))))
    
    # Add count labels
    for bar, count in zip(bars1, alert_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500, 
                 f"{count:,}", ha="center", va="bottom", fontsize=9)
    for bar, count in zip(bars2, drowsy_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                 f"{count:,}", ha="center", va="bottom", fontsize=9)
    
    # Panel 2: Training data after sampling (horizontal grouped bar)
    ax2 = fig.add_subplot(gs[0, 1])
    
    y = np.arange(len(df_plot))
    height = 0.35
    
    bars1 = ax2.barh(y - height/2, df_plot["after_alert"], height, 
                     label="Alert (Negative)", color="#3498db", edgecolor="black", alpha=0.8)
    bars2 = ax2.barh(y + height/2, df_plot["after_drowsy"], height, 
                     label="Drowsy (Positive)", color="#e74c3c", edgecolor="black", alpha=0.8)
    
    ax2.set_yticks(y)
    ax2.set_yticklabels(df_plot["display_name"], fontsize=9)
    ax2.set_xlabel("Samples")
    ax2.set_title("Training Data After Sampling", fontweight="bold", fontsize=12)
    ax2.legend(loc="lower right")
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x/1000)}K"))
    
    # Add original line
    ax2.axvline(35522, color="blue", linestyle="--", alpha=0.5, linewidth=1)
    ax2.axvline(1445, color="red", linestyle="--", alpha=0.5, linewidth=1)
    
    # Panel 3: Class ratio (positive/total) comparison
    ax3 = fig.add_subplot(gs[1, 0])
    
    df_plot["drowsy_pct"] = df_plot["after_drowsy"] / df_plot["after_total"] * 100
    df_plot_sorted = df_plot.sort_values("drowsy_pct", ascending=True)
    
    colors = plt.cm.RdYlGn(df_plot_sorted["drowsy_pct"] / 100)
    
    bars = ax3.barh(range(len(df_plot_sorted)), df_plot_sorted["drowsy_pct"], 
                    color=colors, edgecolor="black", alpha=0.8)
    
    ax3.set_yticks(range(len(df_plot_sorted)))
    ax3.set_yticklabels(df_plot_sorted["display_name"], fontsize=9)
    ax3.set_xlabel("Positive Class Percentage (%)")
    ax3.set_title("Positive Class Ratio After Sampling", fontweight="bold", fontsize=12)
    ax3.axvline(3.9, color="gray", linestyle="--", linewidth=2, label="Original (3.9%)")
    ax3.axvline(50, color="green", linestyle="--", linewidth=1, alpha=0.7, label="Balanced (50%)")
    ax3.legend(loc="lower right")
    ax3.set_xlim(0, 60)
    
    # Add percentage labels
    for i, (idx, row) in enumerate(df_plot_sorted.iterrows()):
        ax3.text(row["drowsy_pct"] + 1, i, f"{row['drowsy_pct']:.1f}%", 
                 va="center", fontsize=8)
    
    # Panel 4: Summary table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    
    table_data = []
    df_table = df_plot.sort_values("after_total", ascending=False)
    for _, row in df_table.iterrows():
        alert_change = (row["after_alert"] / row["orig_alert"] - 1) * 100
        drowsy_change = (row["after_drowsy"] / row["orig_drowsy"] - 1) * 100
        table_data.append([
            row["display_name"],
            f"{row['after_alert']:,}",
            f"{alert_change:+.0f}%",
            f"{row['after_drowsy']:,}",
            f"{drowsy_change:+.0f}%",
            f"{row['drowsy_pct']:.1f}%",
        ])
    
    table = ax4.table(
        cellText=table_data,
        colLabels=["Method", "Alert", "Δ%", "Drowsy", "Δ%", "Pos%"],
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
    
    ax4.set_title("Sampling Summary (Δ from Original)", fontweight="bold", fontsize=12, y=0.95)
    
    fig.suptitle("Multi-seed Experiment: Training Data Distribution Analysis\n"
                 "(Original: Alert=35,522, Drowsy=1,445, Positive=3.9%)", 
                 fontsize=14, fontweight="bold", y=0.98)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {output_path}")
    plt.close(fig)


def plot_ratio_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """Create ratio comparison visualization for variable-ratio methods.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with sampling distribution data
    output_path : Path
        Output path for the figure
    """
    # Filter for ratio-variable methods (seed=42)
    df_ratio = df[(df["seed"] == 42) & (df["ratio"].notna())].copy()
    df_ratio = df_ratio.drop_duplicates(subset=["method", "ratio"])
    
    # Get unique methods with ratio
    methods = df_ratio["method"].unique()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ratios = [0.1, 0.5, 1.0]
    x = np.arange(len(ratios))
    width = 0.12
    
    # Panel 1: Alert (Negative) count by ratio
    ax1 = axes[0, 0]
    for i, method in enumerate(sorted(methods)):
        data = df_ratio[df_ratio["method"] == method]
        values = [data[data["ratio"] == r]["after_alert"].values[0] if len(data[data["ratio"] == r]) > 0 else 0 for r in ratios]
        ax1.bar(x + i * width, values, width, label=method.replace("_", " ").title(), alpha=0.8)
    
    ax1.set_xticks(x + width * (len(methods) - 1) / 2)
    ax1.set_xticklabels(["0.1", "0.5", "1.0"])
    ax1.set_xlabel("Target Ratio (minority/majority)")
    ax1.set_ylabel("Alert (Negative) Samples")
    ax1.set_title("Negative Class Count by Ratio", fontweight="bold")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax1.axhline(35522, color="blue", linestyle="--", alpha=0.5, label="Original")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x/1000)}K"))
    
    # Panel 2: Drowsy (Positive) count by ratio
    ax2 = axes[0, 1]
    for i, method in enumerate(sorted(methods)):
        data = df_ratio[df_ratio["method"] == method]
        values = [data[data["ratio"] == r]["after_drowsy"].values[0] if len(data[data["ratio"] == r]) > 0 else 0 for r in ratios]
        ax2.bar(x + i * width, values, width, label=method.replace("_", " ").title(), alpha=0.8)
    
    ax2.set_xticks(x + width * (len(methods) - 1) / 2)
    ax2.set_xticklabels(["0.1", "0.5", "1.0"])
    ax2.set_xlabel("Target Ratio (minority/majority)")
    ax2.set_ylabel("Drowsy (Positive) Samples")
    ax2.set_title("Positive Class Count by Ratio", fontweight="bold")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax2.axhline(1445, color="red", linestyle="--", alpha=0.5, label="Original")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x/1000)}K"))
    
    # Panel 3: Total samples by ratio
    ax3 = axes[1, 0]
    for i, method in enumerate(sorted(methods)):
        data = df_ratio[df_ratio["method"] == method]
        values = [data[data["ratio"] == r]["after_alert"].values[0] + data[data["ratio"] == r]["after_drowsy"].values[0] 
                  if len(data[data["ratio"] == r]) > 0 else 0 for r in ratios]
        ax3.bar(x + i * width, values, width, label=method.replace("_", " ").title(), alpha=0.8)
    
    ax3.set_xticks(x + width * (len(methods) - 1) / 2)
    ax3.set_xticklabels(["0.1", "0.5", "1.0"])
    ax3.set_xlabel("Target Ratio (minority/majority)")
    ax3.set_ylabel("Total Samples")
    ax3.set_title("Total Training Samples by Ratio", fontweight="bold")
    ax3.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax3.axhline(36967, color="gray", linestyle="--", alpha=0.5, label="Original")
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x/1000)}K"))
    
    # Panel 4: Positive percentage by ratio
    ax4 = axes[1, 1]
    for i, method in enumerate(sorted(methods)):
        data = df_ratio[df_ratio["method"] == method]
        values = []
        for r in ratios:
            subset = data[data["ratio"] == r]
            if len(subset) > 0:
                total = subset["after_alert"].values[0] + subset["after_drowsy"].values[0]
                pct = subset["after_drowsy"].values[0] / total * 100 if total > 0 else 0
                values.append(pct)
            else:
                values.append(0)
        ax4.bar(x + i * width, values, width, label=method.replace("_", " ").title(), alpha=0.8)
    
    ax4.set_xticks(x + width * (len(methods) - 1) / 2)
    ax4.set_xticklabels(["0.1", "0.5", "1.0"])
    ax4.set_xlabel("Target Ratio (minority/majority)")
    ax4.set_ylabel("Positive Class %")
    ax4.set_title("Positive Class Percentage by Ratio", fontweight="bold")
    ax4.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax4.axhline(3.9, color="gray", linestyle="--", alpha=0.7, label="Original (3.9%)")
    ax4.axhline(50, color="green", linestyle="--", alpha=0.5, label="Balanced (50%)")
    
    fig.suptitle("Training Data by Sampling Ratio\n(Comparing ratio=0.1, 0.5, 1.0)", 
                 fontsize=14, fontweight="bold", y=0.98)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {output_path}")
    plt.close(fig)


def main():
    """Main function."""
    project_root = Path("/home/s2240011/git/ddd/vehicle_based_DDD_comparison")
    log_dir = project_root / "scripts/hpc/log"
    output_dir = project_root / "results/imbalance_analysis/multiseed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Extracting sampling distribution from logs...")
    df = extract_sampling_distribution(log_dir, job_prefix="14618")
    logger.info(f"Extracted {len(df)} records")
    
    # Generate visualizations
    logger.info("Generating 'Training Data After Sampling' plot...")
    plot_training_data_after_sampling(df, output_dir / "training_data_after_sampling.png")
    
    logger.info("Generating ratio comparison plot...")
    plot_ratio_comparison(df, output_dir / "sampling_ratio_comparison.png")
    
    # Save raw data
    df.to_csv(output_dir / "sampling_distribution.csv", index=False)
    logger.info(f"Saved raw data to {output_dir / 'sampling_distribution.csv'}")
    
    print(f"\n✅ Visualizations saved to: {output_dir}")
    print("  - training_data_after_sampling.png")
    print("  - sampling_ratio_comparison.png")


if __name__ == "__main__":
    main()
