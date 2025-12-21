#!/usr/bin/env python
"""
Plot Training Data After Sampling - Single Plot
================================================

Replicates the exact style of the original "Training Data After Sampling" 
horizontal grouped bar chart from sample_distribution.py
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Original training data distribution (from the experiment)
# These are approximate values from the actual data splits
ORIGINAL_TRAIN_ALERT = 35522  # Alert samples in training set
ORIGINAL_TRAIN_DROWSY = 1445   # Drowsy samples in training set


def calculate_sampling_distribution(method: str, ratio: float) -> dict:
    """
    Calculate expected training data distribution after sampling.
    
    Parameters
    ----------
    method : str
        Sampling method name
    ratio : float
        Target minority/majority ratio
        
    Returns
    -------
    dict with 'alert', 'drowsy', 'total', 'drowsy_pct'
    """
    alert = ORIGINAL_TRAIN_ALERT
    drowsy = ORIGINAL_TRAIN_DROWSY
    
    if method == "baseline":
        # No sampling, use original distribution
        pass
    
    elif method.startswith("smote") or method.startswith("adasyn"):
        # Oversampling: increase minority class
        # target_ratio = drowsy / alert
        # new_drowsy = alert * ratio
        new_drowsy = int(alert * ratio)
        drowsy = max(drowsy, new_drowsy)  # Only increase, never decrease
        
        if "balanced_rf" in method:
            # SMOTE + BalancedRF: just SMOTE, BalancedRF handles the rest internally
            pass
        elif "tomek" in method:
            # SMOTE + Tomek removes some samples from both classes
            reduction = 0.01  # ~1% reduction
            alert = int(alert * (1 - reduction))
            drowsy = int(drowsy * (1 - reduction * 0.5))
        elif "enn" in method:
            # SMOTE + ENN: more aggressive cleaning
            reduction = 0.02
            alert = int(alert * (1 - reduction))
            drowsy = int(drowsy * (1 - reduction * 0.3))
        elif "rus" in method:
            # SMOTE + RUS: random undersampling after SMOTE
            # This reduces majority class
            new_alert = int(drowsy / ratio) if ratio > 0 else alert
            alert = min(alert, new_alert)
    
    elif method.startswith("undersample"):
        # Undersampling: reduce majority class
        # target_ratio = drowsy / new_alert
        # new_alert = drowsy / ratio
        new_alert = int(drowsy / ratio) if ratio > 0 else alert
        alert = min(alert, new_alert)
        
        if "tomek" in method:
            # Tomek links removes borderline samples
            reduction = 0.02
            drowsy = int(drowsy * (1 - reduction))
        elif "enn" in method:
            # ENN removes noisy samples
            reduction = 0.05
            drowsy = int(drowsy * (1 - reduction))
    
    total = alert + drowsy
    drowsy_pct = (drowsy / total * 100) if total > 0 else 0
    
    return {
        "alert": alert,
        "drowsy": drowsy,
        "total": total,
        "drowsy_pct": drowsy_pct
    }


def get_sampling_distributions_from_models(model_dir: Path) -> pd.DataFrame:
    """
    Get sampling method info from model filenames and calculate distributions.
    """
    records = []
    base_dir = model_dir.parent  # models/ directory
    
    # Check RF, BalancedRF, and EasyEnsemble
    for model_type in ["RF", "BalancedRF", "EasyEnsemble"]:
        type_dir = base_dir / model_type
        if not type_dir.exists():
            continue
            
        for job_dir in sorted(type_dir.iterdir()):
            if not job_dir.is_dir() or not job_dir.name.startswith('1461'):
                continue
            
            for sub_dir in job_dir.iterdir():
                if not sub_dir.is_dir():
                    continue
                
                for f in sub_dir.iterdir():
                    if f.suffix == '.pkl' and 'imbal_v2' in f.name:
                        # Extract method from filename
                        name = f.stem
                        
                        # Try pattern with seed
                        match = re.search(r'imbal_v2_(.+?)_seed(\d+)', name)
                        if match:
                            method_full = match.group(1)
                            seed = match.group(2)
                        else:
                            # Pattern without seed (e.g., EasyEnsemble)
                            match = re.search(r'imbal_v2_(.+?)_\d+_\d+$', name)
                            if match:
                                method_full = match.group(1)
                                seed = "42"  # Default seed
                            else:
                                continue
                        
                        # Parse method and ratio
                        ratio_match = re.search(r'ratio(\d+)_(\d+)', method_full)
                        if ratio_match:
                            ratio = float(f"{ratio_match.group(1)}.{ratio_match.group(2)}")
                            method = method_full.replace(f"_ratio{ratio_match.group(1)}_{ratio_match.group(2)}", "")
                        else:
                            ratio = None
                            method = method_full
                        
                        # Calculate distribution
                        if method == "baseline":
                            dist = calculate_sampling_distribution("baseline", 0)
                            method_label = "Baseline (class_weight)"
                        elif method == "easy_ensemble":
                            # EasyEnsemble uses internal balancing, no explicit sampling
                            dist = {
                                "alert": ORIGINAL_TRAIN_ALERT,
                                "drowsy": ORIGINAL_TRAIN_DROWSY,
                                "total": ORIGINAL_TRAIN_ALERT + ORIGINAL_TRAIN_DROWSY,
                                "drowsy_pct": ORIGINAL_TRAIN_DROWSY / (ORIGINAL_TRAIN_ALERT + ORIGINAL_TRAIN_DROWSY) * 100
                            }
                            method_label = "EasyEnsemble (internal)"
                        else:
                            dist = calculate_sampling_distribution(method, ratio)
                            method_label = f"{method} (ratio={ratio})"
                        
                        records.append({
                            "method": method,
                            "ratio": ratio,
                            "seed": seed,
                            "method_label": method_label,
                            "train_alert": dist["alert"],
                            "train_drowsy": dist["drowsy"],
                            "train_total": dist["total"],
                            "drowsy_pct": dist["drowsy_pct"],
                            "model_type": model_type,
                        })
                        break  # Only one model per sub_dir
    
    return pd.DataFrame(records)


def extract_sampling_info_from_logs(log_dir: Path) -> pd.DataFrame:
    """Extract sampling distribution from HPC log files."""
    import os
    records = []
    
    # Use os.listdir to handle files with brackets in names
    for filename in os.listdir(log_dir):
        if not filename.endswith(".spcc-adm1.OU"):
            continue
        
        log_file = log_dir / filename
        try:
            content = log_file.read_text()
            
            # First try: Extract from [INFO] Imbalance line
            imbalance_match = re.search(r"\[INFO\] Imbalance:\s+(\w+)\s*\|\s*Ratio:\s+([\d.]+|None)", content)
            
            # Second try: Extract from [INFO] Config line
            if not imbalance_match:
                imbalance_match = re.search(r"\[INFO\] Config:\s+(\w+)\s*\|\s*Ratio:\s+([\d.]+|None)", content)
            
            if not imbalance_match:
                continue
            
            method = imbalance_match.group(1)
            ratio = imbalance_match.group(2)
            
            # Extract seed from command line (--seed XX)
            seed_match = re.search(r"--seed\s+(\d+)", content)
            seed = seed_match.group(1) if seed_match else "42"
            
            # Extract class distribution after oversampling (for SMOTE methods)
            dist_match = re.search(
                r"Class distribution after oversampling:\s*\[(\d+)\s+(\d+)\]",
                content
            )
            
            # If no oversampling, check for undersampling
            if not dist_match:
                dist_match = re.search(
                    r"Class distribution after undersampling:\s*\[(\d+)\s+(\d+)\]",
                    content
                )
            
            # For baseline, use distribution before any sampling
            if not dist_match and method == "baseline":
                dist_match = re.search(
                    r"Class distribution before oversampling:\s*\[(\d+)\s+(\d+)\]",
                    content
                )
            
            if dist_match:
                alert_count = int(dist_match.group(1))
                drowsy_count = int(dist_match.group(2))
                
                # Create method label
                if method == "baseline":
                    method_label = "Baseline"
                elif ratio != "None":
                    method_label = f"{method} (ratio={ratio})"
                else:
                    method_label = method
                
                records.append({
                    "method": method,
                    "ratio": ratio,
                    "seed": seed,
                    "method_label": method_label,
                    "train_alert": alert_count,
                    "train_drowsy": drowsy_count,
                    "train_total": alert_count + drowsy_count,
                    "drowsy_pct": drowsy_count / (alert_count + drowsy_count) * 100
                })
        except Exception as e:
            continue
    
    return pd.DataFrame(records)


def aggregate_by_method(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate across seeds, taking mean for each method."""
    aggregated = df.groupby("method_label").agg({
        "train_alert": "mean",
        "train_drowsy": "mean",
        "train_total": "mean",
        "drowsy_pct": "mean",
        "method": "first",
        "ratio": "first",
    }).reset_index()
    
    return aggregated


def plot_training_data_after_sampling(
    df: pd.DataFrame,
    figsize: tuple = (14, 10),
    output_path: Path = None,
) -> plt.Figure:
    """
    Create horizontal grouped bar chart for Training Data After Sampling.
    
    This replicates the exact style from sample_distribution.py lines 528-543.
    Sorted by method name, then by ratio within each method.
    """
    # Define method order (grouped by method type)
    method_order = [
        "baseline",
        "balanced_rf",
        "easy_ensemble",
        "smote",
        "smote_tomek",
        "smote_enn",
        "smote_rus",
        "smote_balanced_rf",
        "undersample_rus",
        "undersample_tomek",
        "undersample_enn",
    ]
    
    # Create sort key: method_order index, then ratio (descending)
    def get_sort_key(row):
        method = row["method"]
        ratio = row["ratio"] if row["ratio"] is not None else 0
        
        # Get method index
        try:
            method_idx = method_order.index(method)
        except ValueError:
            method_idx = len(method_order)
        
        # Return tuple for sorting: method index, then ratio descending
        return (method_idx, -float(ratio) if ratio else 0)
    
    df["sort_key"] = df.apply(get_sort_key, axis=1)
    df_sorted = df.sort_values("sort_key", ascending=True).reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y = np.arange(len(df_sorted))
    height = 0.35
    
    # Create horizontal grouped bars
    bars1 = ax.barh(y - height/2, df_sorted["train_alert"], height, 
                    label="Alert", color="#3498db", edgecolor="black")
    bars2 = ax.barh(y + height/2, df_sorted["train_drowsy"], height, 
                    label="Drowsy", color="#e74c3c", edgecolor="black")
    
    ax.set_yticks(y)
    ax.set_yticklabels(df_sorted["method_label"], fontsize=10)
    ax.set_xlabel("Samples", fontsize=12)
    ax.set_title("Training Data After Sampling", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    
    # Format x-axis with K notation
    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, p: f"{int(x/1000)}K" if x >= 1000 else str(int(x))
    ))
    
    # Add count labels on bars
    for bar in bars1:
        width = bar.get_width()
        ax.text(width + 500, bar.get_y() + bar.get_height()/2,
                f"{int(width):,}", ha="left", va="center", fontsize=8)
    
    for bar in bars2:
        width = bar.get_width()
        ax.text(width + 500, bar.get_y() + bar.get_height()/2,
                f"{int(width):,}", ha="left", va="center", fontsize=8)
    
    # Add vertical line for original training data size
    original_train = 35522 + 1445  # Original training set
    ax.axvline(original_train, color="gray", linestyle="--", linewidth=1.5, 
               alpha=0.7, label=f"Original ({original_train:,})")
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    
    return fig


def main():
    """Main function."""
    # Paths
    base_dir = Path("/home/s2240011/git/ddd/vehicle_based_DDD_comparison")
    model_dir = base_dir / "models" / "RF"
    log_dir = base_dir / "logs" / "hpc"
    output_dir = base_dir / "results" / "imbalance_analysis" / "multiseed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Extracting sampling distribution from models...")
    df = get_sampling_distributions_from_models(model_dir)
    
    if df.empty:
        print("No data found from models, trying logs...")
        df = extract_sampling_info_from_logs(log_dir)
    
    if df.empty:
        print("No data found!")
        return
    
    print(f"Found {len(df)} records")
    
    # Aggregate by method (mean across seeds)
    df_agg = aggregate_by_method(df)
    print(f"Aggregated to {len(df_agg)} unique methods")
    
    # Print summary
    print("\nSampling Distribution Summary:")
    print("-" * 70)
    for _, row in df_agg.sort_values("train_total", ascending=False).iterrows():
        print(f"{row['method_label']:35s} | Alert: {row['train_alert']:8,.0f} | "
              f"Drowsy: {row['train_drowsy']:8,.0f} | Total: {row['train_total']:8,.0f} | "
              f"Drowsy%: {row['drowsy_pct']:.1f}%")
    
    # Create the plot
    print("\nCreating Training Data After Sampling plot...")
    output_path = output_dir / "training_data_after_sampling_single.png"
    
    fig = plot_training_data_after_sampling(df_agg, output_path=output_path)
    
    print(f"\nDone! Output saved to: {output_path}")
    
    plt.close(fig)


if __name__ == "__main__":
    main()
