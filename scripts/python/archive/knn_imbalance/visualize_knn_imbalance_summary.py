#!/usr/bin/env python3
"""
Visualize KNN + Imbalance handling results in the SAME format as summary_metrics_bar.png.

Uses the same plot_grouped_bar_chart_raw function from src/utils/visualization/visualization.py
to ensure identical formatting.

Output:
- results/domain_analysis/summary/png/knn/summary_metrics_bar_{imbalance_method}.png
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import matplotlib as mpl
mpl.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from src import config as cfg
from src.utils.io.data_io import load_csv
from src.utils.visualization.visualization import save_figure, plot_grouped_bar_chart_raw

# Paths
IN_CSV = Path(cfg.RESULTS_DOMAIN_ANALYSIS_PATH) / "summary" / "csv" / "summary_ranked_test.csv"
POOLED_CSV = Path(cfg.RESULTS_DOMAIN_ANALYSIS_PATH) / "summary" / "csv" / "summary_40cases_test.csv"
KNN_IMBALANCE_DIR = PROJECT_ROOT / "results" / "domain_analysis" / "knn_imbalance"
OUT_DIR = Path(cfg.RESULTS_DOMAIN_ANALYSIS_PATH) / "summary" / "png" / "knn"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Metrics to plot (same as original)
METRICS = ["accuracy", "recall", "precision", "f1", "f2", "auc", "auc_pr"]

# Imbalance methods
IMBALANCE_METHODS = ["baseline", "undersample_rus", "undersample_tomek", "smote_rus", "smote_tomek"]
IMBALANCE_LABELS = {
    "baseline": "Baseline",
    "undersample_rus": "RUS",
    "undersample_tomek": "Tomek",
    "smote_rus": "SMOTE+RUS",
    "smote_tomek": "SMOTE+Tomek",
}


def load_knn_imbalance_results():
    """Load all KNN imbalance results from CSV files."""
    all_results = []
    for csv_file in KNN_IMBALANCE_DIR.glob("summary_*.csv"):
        df = pd.read_csv(csv_file)
        # Parse filename: summary_{jobid}_{mode}_{distance}.csv
        filename = csv_file.stem
        parts = filename.replace("summary_", "").split("_")
        if len(parts) >= 4:
            df["job_id"] = parts[0]
            df["mode"] = f"{parts[1]}_{parts[2]}"
            df["distance"] = parts[3]
        all_results.append(df)
    
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()


def convert_to_ranked_format(imbalance_df: pd.DataFrame) -> pd.DataFrame:
    """Convert imbalance results to the format expected by plot_grouped_bar_chart_raw.
    
    Required columns: mode, level, distance, + metric columns
    """
    # Rename columns to match expected format
    rename_map = {
        "acc_thr": "accuracy",
        "recall_thr": "recall",
        "prec_thr": "precision",
        "f1_thr": "f1",
        "f2_thr": "f2",
        "auc": "auc",
        "auc_pr": "auc_pr",
    }
    
    df = imbalance_df.copy()
    
    # Rename metric columns if they exist
    for old_name, new_name in rename_map.items():
        if old_name in df.columns and new_name not in df.columns:
            df[new_name] = df[old_name]
    
    # Ensure required columns exist
    required_cols = ["mode", "level", "distance"]
    for col in required_cols:
        if col not in df.columns:
            print(f"[WARN] Missing column: {col}")
    
    return df


def main():
    print("=" * 60)
    print("Creating summary_metrics_bar_{imbalance_method}.png")
    print("Using SAME format as summary_metrics_bar.png")
    print("=" * 60)
    
    # Load KNN imbalance results
    print("\n[INFO] Loading KNN imbalance results...")
    imbalance_df = load_knn_imbalance_results()
    
    if imbalance_df.empty:
        print("[ERROR] No KNN imbalance results found!")
        print(f"  Expected location: {KNN_IMBALANCE_DIR}/summary_*.csv")
        return
    
    print(f"[INFO] Loaded {len(imbalance_df)} records")
    print(f"  Distances: {imbalance_df['distance'].unique().tolist() if 'distance' in imbalance_df.columns else 'N/A'}")
    print(f"  Modes: {imbalance_df['mode'].unique().tolist() if 'mode' in imbalance_df.columns else 'N/A'}")
    print(f"  Methods: {imbalance_df['method'].unique().tolist() if 'method' in imbalance_df.columns else 'N/A'}")
    
    # Load pooled data for baseline comparison
    print("\n[INFO] Loading pooled baseline data...")
    df_pooled = pd.DataFrame()
    if POOLED_CSV.exists():
        df_all_40 = load_csv(str(POOLED_CSV))
        if "mode" in df_all_40.columns:
            df_pooled = df_all_40[df_all_40["mode"] == "pooled"].copy()
            if "distance" in df_pooled.columns:
                df_pooled["distance"] = df_pooled["distance"].apply(
                    lambda x: x.split("_")[-1] if isinstance(x, str) else x
                )
            print(f"[INFO] Loaded {len(df_pooled)} pooled records")
    else:
        print(f"[WARN] Pooled data not found: {POOLED_CSV}")
    
    # Create plot for each imbalance method
    print("\n" + "-" * 40)
    print("[INFO] Creating plots for each imbalance method...")
    print("-" * 40)
    
    for imb_method in IMBALANCE_METHODS:
        print(f"\n[INFO] Processing: {imb_method}")
        
        # Filter data for this imbalance method
        method_df = imbalance_df[imbalance_df["method"] == imb_method].copy()
        
        if method_df.empty:
            print(f"  [WARN] No data for {imb_method}, skipping...")
            continue
        
        # Convert to expected format
        method_df = convert_to_ranked_format(method_df)
        
        # Check which metrics are available
        available_metrics = [m for m in METRICS if m in method_df.columns]
        print(f"  Available metrics: {available_metrics}")
        
        if not available_metrics:
            print(f"  [WARN] No metrics found for {imb_method}, skipping...")
            continue
        
        # Add pooled data if available
        if not df_pooled.empty:
            common_cols = method_df.columns.intersection(df_pooled.columns)
            pooled_to_add = df_pooled[common_cols].copy()
            method_df_with_pooled = pd.concat([method_df, pooled_to_add], ignore_index=True)
        else:
            method_df_with_pooled = method_df
        
        print(f"  Records: {len(method_df)} (Total with pooled: {len(method_df_with_pooled)})")
        
        # Calculate baseline rate for auc_pr
        pos_rate = method_df["pos_rate"].mean() if "pos_rate" in method_df.columns else 0.033
        
        # Create plot using the SAME function as original
        fig = plot_grouped_bar_chart_raw(
            data=method_df_with_pooled,
            metrics=available_metrics,
            modes=["pooled", "source_only", "target_only"],
            distance_col="distance",
            level_col="level",
            baseline_rates={"auc_pr": pos_rate}
        )
        
        if fig:
            out_path = OUT_DIR / f"summary_metrics_bar_{imb_method}.png"
            save_figure(fig, str(out_path), dpi=200)
            print(f"  Saved: {out_path}")
            plt.close(fig)
        else:
            print(f"  [WARN] Failed to create plot for {imb_method}")
    
    print("\n" + "=" * 60)
    print("[DONE] All plots created!")
    print(f"  Output directory: {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
