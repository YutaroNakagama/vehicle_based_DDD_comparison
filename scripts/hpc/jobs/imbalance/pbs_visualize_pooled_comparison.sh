#!/bin/bash
#PBS -N imbal_v2_viz_pooled
#PBS -l select=1:ncpus=2:mem=4gb
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -q SINGLE
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/imbalance/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/imbalance/
#PBS -M yutaro.nakagama@bosch.com
#PBS -m abe

# ============================================================
# Imbalance Comparison V2: Pooled Mode Comparison Visualization
# Generate comparison figures for all pooled experiments
# ============================================================
set -euo pipefail

export PATH=~/conda/bin:$PATH
source ~/conda/etc/profile.d/conda.sh
conda activate python310

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export JOBLIB_MULTIPROCESSING=0

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
export PYTHONPATH=$PROJECT_ROOT:${PYTHONPATH:-}
export PBS_JOBID="${PBS_JOBID:-manual_$(date +%Y%m%d_%H%M%S)}"

cd "$PROJECT_ROOT"

echo "============================================================"
echo "[IMBALANCE V2] Pooled Mode Comparison Visualization"
echo "============================================================"
echo "PBS_JOBID: $PBS_JOBID"
echo "Date: $(date)"
echo "============================================================"

# Create output directory
OUTPUT_DIR="results/analysis/imbalance/pooled_comparison_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "[STEP 1] Collecting all pooled experiment results..."

# Python script to collect and visualize pooled results
python3 << 'PYTHON_SCRIPT'
import json
import glob
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Configuration
RESULTS_DIR = Path("results/outputs/evaluation")
OUTPUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/analysis/imbalance/pooled_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Output directory: {OUTPUT_DIR}")

# Collect all pooled evaluation results
records = []

# Search patterns for different model types
search_patterns = [
    ("RF", "results/outputs/evaluation/RF/*/*/eval_results_RF_pooled*.json"),
    ("BalancedRF", "results/outputs/evaluation/BalancedRF/*/*/eval_results_BalancedRF_pooled*.json"),
    ("EasyEnsemble", "results/outputs/evaluation/EasyEnsemble/*/*/eval_results_EasyEnsemble_pooled*.json"),
]

for model_type, pattern in search_patterns:
    for json_file in glob.glob(pattern):
        try:
            with open(json_file) as f:
                data = json.load(f)
            
            tag = data.get("tag", "unknown")
            
            # Extract method name from tag
            method = tag.replace("imbal_v2_", "").replace("imbalance_pooled_", "")
            
            record = {
                "model": model_type,
                "method": method,
                "tag": tag,
                "file": json_file,
                "recall": data.get("recall", 0),
                "precision": data.get("precision", 0),
                "f1": data.get("f1", 0),
                "f2": data.get("f2", 0),
                "auc_pr": data.get("auc_pr", 0),
                "auc_roc": data.get("auc", data.get("auc_roc", 0)),
                "accuracy": data.get("accuracy", 0),
            }
            records.append(record)
            print(f"  Found: {model_type} - {method} (Recall: {record['recall']:.4f})")
        except Exception as e:
            print(f"  [WARN] Failed to load {json_file}: {e}")

if not records:
    print("[ERROR] No pooled results found!")
    sys.exit(1)

df = pd.DataFrame(records)

# Remove duplicates, keep latest
df = df.drop_duplicates(subset=["method"], keep="last")
df = df.sort_values("recall", ascending=False)

print(f"\nTotal methods found: {len(df)}")
print(df[["method", "recall", "precision", "f1", "auc_pr"]].to_string())

# Save CSV
csv_path = OUTPUT_DIR / "pooled_comparison.csv"
df.to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path}")

# ============================================================
# Plot 1: Recall Comparison Bar Chart
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))

colors = []
for recall in df["recall"]:
    if recall >= 0.7:
        colors.append("#2ecc71")  # Green - good
    elif recall >= 0.5:
        colors.append("#f39c12")  # Orange - moderate
    else:
        colors.append("#e74c3c")  # Red - poor

bars = ax.barh(df["method"], df["recall"] * 100, color=colors, edgecolor="black")

for bar, recall in zip(bars, df["recall"]):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
            f"{recall*100:.1f}%", va="center", fontsize=10)

ax.set_xlabel("Recall (%)", fontsize=12)
ax.set_ylabel("Method", fontsize=12)
ax.set_title("Imbalance Methods Comparison - Recall (Pooled Mode)", fontsize=14)
ax.set_xlim(0, 110)
ax.axvline(x=70, color="green", linestyle="--", alpha=0.7, label="Target: 70%")
ax.legend()
ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "recall_comparison.png", dpi=150)
print(f"Saved: {OUTPUT_DIR / 'recall_comparison.png'}")
plt.close()

# ============================================================
# Plot 2: Multi-Metric Comparison
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics = [
    ("recall", "Recall", axes[0, 0]),
    ("precision", "Precision", axes[0, 1]),
    ("auc_pr", "AUPRC", axes[1, 0]),
    ("f1", "F1 Score", axes[1, 1]),
]

for metric, title, ax in metrics:
    df_sorted = df.sort_values(metric, ascending=True)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(df_sorted)))
    
    bars = ax.barh(df_sorted["method"], df_sorted[metric] * 100, color=colors)
    ax.set_xlabel(f"{title} (%)")
    ax.set_title(f"{title} Comparison")
    ax.grid(axis="x", alpha=0.3)
    
    for bar, val in zip(bars, df_sorted[metric]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{val*100:.1f}%", va="center", fontsize=9)

plt.suptitle("Imbalance Methods - Multi-Metric Comparison (Pooled Mode)", fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "multi_metric_comparison.png", dpi=150)
print(f"Saved: {OUTPUT_DIR / 'multi_metric_comparison.png'}")
plt.close()

# ============================================================
# Plot 3: Recall vs AUPRC Scatter
# ============================================================
fig, ax = plt.subplots(figsize=(10, 8))

scatter = ax.scatter(df["recall"] * 100, df["auc_pr"] * 100, 
                     s=200, c=range(len(df)), cmap="tab10", edgecolors="black", alpha=0.8)

for i, row in df.iterrows():
    ax.annotate(row["method"], (row["recall"] * 100 + 1, row["auc_pr"] * 100 + 0.1),
                fontsize=9)

ax.set_xlabel("Recall (%)", fontsize=12)
ax.set_ylabel("AUPRC (%)", fontsize=12)
ax.set_title("Recall vs AUPRC - Imbalance Methods (Pooled Mode)", fontsize=14)
ax.grid(True, alpha=0.3)
ax.axvline(x=70, color="green", linestyle="--", alpha=0.5, label="Recall Target: 70%")
ax.legend()

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "recall_vs_auprc.png", dpi=150)
print(f"Saved: {OUTPUT_DIR / 'recall_vs_auprc.png'}")
plt.close()

# ============================================================
# Summary Report
# ============================================================
report = f"""
================================================================================
IMBALANCE METHODS COMPARISON - POOLED MODE
================================================================================

Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

RANKING BY RECALL:
{df[['method', 'recall', 'precision', 'f1', 'auc_pr']].to_string(index=False)}

--------------------------------------------------------------------------------
TOP RECOMMENDATIONS:
--------------------------------------------------------------------------------
1. Best Recall:    {df.iloc[0]['method']} (Recall: {df.iloc[0]['recall']*100:.1f}%)
2. Best AUPRC:     {df.loc[df['auc_pr'].idxmax(), 'method']} (AUPRC: {df['auc_pr'].max()*100:.2f}%)
3. Best F1:        {df.loc[df['f1'].idxmax(), 'method']} (F1: {df['f1'].max()*100:.2f}%)

--------------------------------------------------------------------------------
OUTPUT FILES:
--------------------------------------------------------------------------------
- {OUTPUT_DIR / 'pooled_comparison.csv'}
- {OUTPUT_DIR / 'recall_comparison.png'}
- {OUTPUT_DIR / 'multi_metric_comparison.png'}
- {OUTPUT_DIR / 'recall_vs_auprc.png'}
================================================================================
"""

print(report)

with open(OUTPUT_DIR / "summary_report.txt", "w") as f:
    f.write(report)
print(f"Saved: {OUTPUT_DIR / 'summary_report.txt'}")

print("\n[DONE] Visualization complete!")
PYTHON_SCRIPT

echo ""
echo "============================================================"
echo "Visualization Complete!"
echo "============================================================"
