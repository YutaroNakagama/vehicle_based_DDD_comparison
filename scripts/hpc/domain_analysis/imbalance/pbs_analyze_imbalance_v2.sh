#!/bin/bash
#PBS -N RF_imbal_v2_analysis
#PBS -l select=1:ncpus=4:mem=8gb
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -q DEFAULT
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/log/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/log/
#PBS -M yutaro.nakagama@bosch.com
#PBS -m abe

# ==============================================================================
# pbs_analyze_imbalance_v2.sh
# ==============================================================================
# Purpose: Analyze and visualize results from imbalance V2 experiments
#          - Collect all evaluation results
#          - Generate comparison tables and plots
#          - Run statistical tests
#
# Run after: pbs_eval_imbalance_v2_batch.sh
# ==============================================================================
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

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="results/imbalance_analysis/v2_${TIMESTAMP}"
export OUTPUT_DIR  # Export for Python scripts
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "[ANALYSIS] Imbalance V2 Results Analysis"
echo "============================================================"
echo "PBS_JOBID: $PBS_JOBID"
echo "Date: $(date)"
echo "Output: $OUTPUT_DIR"
echo "============================================================"

# ==============================================================================
# Step 1: Collect all evaluation results
# ==============================================================================
echo ""
echo "[STEP 1] Collecting evaluation results..."

python3 << 'COLLECT_EOF'
import json
import glob
from pathlib import Path
import pandas as pd
import re

PROJECT_ROOT = Path("/home/s2240011/git/ddd/vehicle_based_DDD_comparison")

# Read OUTPUT_DIR from environment or use placeholder
import os
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "results/imbalance_analysis/temp"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Pattern to match ALL json results in the training job directories
# We'll filter by reading the 'tag' field inside each file
patterns = [
    "results/evaluation/RF/*/14578369*/eval_results_*.json",
    "results/evaluation/BalancedRF/*/14578369*/eval_results_*.json",
]

records = []
for pattern in patterns:
    for f in glob.glob(str(PROJECT_ROOT / pattern)):
        try:
            with open(f) as fp:
                d = json.load(fp)
            
            # Get tag from JSON content, not filename
            tag = d.get("tag", "")
            if not tag or not tag.startswith("imbalv2"):
                continue
            
            # Parse tag: imbalv2_{ranking}_{metric}_{level}_{imbalance}
            # or: imbalv2_pooled_{imbalance}
            
            if "pooled" in tag:
                match = re.search(r"imbalv2_pooled_(\w+)", tag)
                if match:
                    ranking = "pooled"
                    metric = "pooled"
                    level = "pooled"
                    imbalance = match.group(1)
                else:
                    continue
            else:
                match = re.search(r"imbalv2_(\w+)_(mmd|dtw|wasserstein)_(out_domain|mid_domain|in_domain)_(\w+)", tag)
                if match:
                    ranking = match.group(1)
                    metric = match.group(2)
                    level = match.group(3)
                    imbalance = match.group(4)
                else:
                    continue
            
            mode = d.get("mode", "unknown")
            
            records.append({
                "ranking": ranking,
                "metric": metric,
                "level": level,
                "imbalance": imbalance,
                "mode": mode,
                "accuracy": d.get("accuracy", 0),
                "precision": d.get("precision", 0),
                "recall": d.get("recall", 0),
                "f1": d.get("f1", 0),
                "f2": d.get("f2", d.get("f2_thr", 0)),
                "auc": d.get("auc", 0),
                "specificity": d.get("specificity", 0),
                "file": f,
                "tag": tag,
            })
        except Exception as e:
            print(f"[WARN] Failed to parse {f}: {e}")

if records:
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_DIR / "all_results.csv", index=False)
    print(f"Collected {len(df)} results -> {OUTPUT_DIR / 'all_results.csv'}")
else:
    print("[WARN] No results found!")
COLLECT_EOF

# ==============================================================================
# Step 2: Generate summary tables
# ==============================================================================
echo ""
echo "[STEP 2] Generating summary tables..."

python3 << SUMMARY_EOF
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("$OUTPUT_DIR")
csv_path = OUTPUT_DIR / "all_results.csv"

if not csv_path.exists():
    print("[SKIP] No results CSV found")
    exit(0)

df = pd.read_csv(csv_path)

# Summary by imbalance method
print("\n=== Summary by Imbalance Method ===")
summary_imbalance = df.groupby("imbalance").agg({
    "recall": ["mean", "std"],
    "precision": ["mean", "std"],
    "f1": ["mean", "std"],
    "f2": ["mean", "std"],
    "auc": ["mean", "std"],
}).round(4)
print(summary_imbalance)
summary_imbalance.to_csv(OUTPUT_DIR / "summary_by_imbalance.csv")

# Summary by ranking method
print("\n=== Summary by Ranking Method ===")
summary_ranking = df.groupby("ranking").agg({
    "recall": ["mean", "std"],
    "precision": ["mean", "std"],
    "f1": ["mean", "std"],
    "f2": ["mean", "std"],
    "auc": ["mean", "std"],
}).round(4)
print(summary_ranking)
summary_ranking.to_csv(OUTPUT_DIR / "summary_by_ranking.csv")

# Summary by domain level
print("\n=== Summary by Domain Level ===")
summary_level = df.groupby("level").agg({
    "recall": ["mean", "std"],
    "precision": ["mean", "std"],
    "f1": ["mean", "std"],
    "f2": ["mean", "std"],
    "auc": ["mean", "std"],
}).round(4)
print(summary_level)
summary_level.to_csv(OUTPUT_DIR / "summary_by_level.csv")

# Cross-tabulation: Imbalance × Ranking (F2 score)
print("\n=== F2 Score: Imbalance × Ranking ===")
pivot_f2 = df.pivot_table(
    values="f2", 
    index="imbalance", 
    columns="ranking", 
    aggfunc="mean"
).round(4)
print(pivot_f2)
pivot_f2.to_csv(OUTPUT_DIR / "pivot_f2_imbalance_ranking.csv")

# Cross-tabulation: Imbalance × Level (Recall)
print("\n=== Recall: Imbalance × Level ===")
pivot_recall = df.pivot_table(
    values="recall", 
    index="imbalance", 
    columns="level", 
    aggfunc="mean"
).round(4)
print(pivot_recall)
pivot_recall.to_csv(OUTPUT_DIR / "pivot_recall_imbalance_level.csv")

print(f"\nSaved summary tables to {OUTPUT_DIR}")
SUMMARY_EOF

# ==============================================================================
# Step 3: Generate visualizations
# ==============================================================================
echo ""
echo "[STEP 3] Generating visualizations..."

python3 << VIZ_EOF
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

OUTPUT_DIR = Path("$OUTPUT_DIR")
csv_path = OUTPUT_DIR / "all_results.csv"

if not csv_path.exists():
    print("[SKIP] No results CSV found")
    exit(0)

df = pd.read_csv(csv_path)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# 1. Bar plot: F2 by imbalance method
fig, ax = plt.subplots(figsize=(10, 6))
order = ["baseline", "smote", "smote_enn", "balanced_rf"]
existing_order = [o for o in order if o in df["imbalance"].unique()]
sns.barplot(data=df, x="imbalance", y="f2", order=existing_order, ax=ax, errorbar="sd")
ax.set_xlabel("Imbalance Method")
ax.set_ylabel("F2 Score")
ax.set_title("F2 Score by Imbalance Handling Method")
ax.set_xticklabels(["Baseline", "SMOTE", "SMOTE+ENN", "BalancedRF"][:len(existing_order)])
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "f2_by_imbalance.png", dpi=150)
plt.close()

# 2. Bar plot: Recall by imbalance method
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=df, x="imbalance", y="recall", order=existing_order, ax=ax, errorbar="sd")
ax.set_xlabel("Imbalance Method")
ax.set_ylabel("Recall")
ax.set_title("Recall by Imbalance Handling Method")
ax.set_xticklabels(["Baseline", "SMOTE", "SMOTE+ENN", "BalancedRF"][:len(existing_order)])
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "recall_by_imbalance.png", dpi=150)
plt.close()

# 3. Heatmap: F2 by Ranking × Imbalance
fig, ax = plt.subplots(figsize=(10, 6))
pivot = df.pivot_table(values="f2", index="imbalance", columns="ranking", aggfunc="mean")
# Reorder
if len(pivot) > 0:
    row_order = [r for r in ["baseline", "smote", "smote_enn", "balanced_rf"] if r in pivot.index]
    col_order = [c for c in ["knn", "lof", "median_distance", "pooled"] if c in pivot.columns]
    pivot = pivot.reindex(index=row_order, columns=col_order)
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax)
    ax.set_title("F2 Score: Imbalance Method × Ranking Method")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "heatmap_f2.png", dpi=150)
plt.close()

# 4. Heatmap: Recall by Level × Imbalance
fig, ax = plt.subplots(figsize=(10, 6))
pivot = df.pivot_table(values="recall", index="imbalance", columns="level", aggfunc="mean")
if len(pivot) > 0:
    row_order = [r for r in ["baseline", "smote", "smote_enn", "balanced_rf"] if r in pivot.index]
    col_order = [c for c in ["out_domain", "mid_domain", "in_domain", "pooled"] if c in pivot.columns]
    pivot = pivot.reindex(index=row_order, columns=col_order)
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax)
    ax.set_title("Recall: Imbalance Method × Domain Level")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "heatmap_recall_level.png", dpi=150)
plt.close()

# 5. Multi-metric comparison
metrics = ["recall", "precision", "f1", "f2", "auc"]
fig, axes = plt.subplots(1, len(metrics), figsize=(20, 5))
for i, metric in enumerate(metrics):
    sns.barplot(data=df, x="imbalance", y=metric, order=existing_order, ax=axes[i], errorbar="sd")
    axes[i].set_title(metric.upper())
    axes[i].set_xlabel("")
    axes[i].set_xticklabels(["Base", "SMOTE", "S+ENN", "BRF"][:len(existing_order)], rotation=45)
plt.suptitle("Performance Metrics by Imbalance Method", fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "multi_metric_comparison.png", dpi=150)
plt.close()

print(f"Saved visualizations to {OUTPUT_DIR}")
VIZ_EOF

# ==============================================================================
# Step 4: Statistical tests
# ==============================================================================
echo ""
echo "[STEP 4] Running statistical tests..."

python3 << STATS_EOF
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

OUTPUT_DIR = Path("$OUTPUT_DIR")
csv_path = OUTPUT_DIR / "all_results.csv"

if not csv_path.exists():
    print("[SKIP] No results CSV found")
    exit(0)

df = pd.read_csv(csv_path)

# Wilcoxon signed-rank test: Compare each method to baseline
baseline_mask = df["imbalance"] == "baseline"
if not baseline_mask.any():
    print("[WARN] No baseline results found")
    exit(0)

methods = ["smote", "smote_enn", "balanced_rf"]
metrics = ["recall", "precision", "f1", "f2", "auc"]

results = []
for method in methods:
    method_mask = df["imbalance"] == method
    if not method_mask.any():
        continue
    
    for metric in metrics:
        # Get paired samples (matching ranking, metric, level, mode)
        baseline_df = df[baseline_mask].set_index(["ranking", "metric", "level", "mode"])
        method_df = df[method_mask].set_index(["ranking", "metric", "level", "mode"])
        
        # Find common indices
        common_idx = baseline_df.index.intersection(method_df.index)
        if len(common_idx) < 5:
            continue
        
        baseline_vals = baseline_df.loc[common_idx, metric].values
        method_vals = method_df.loc[common_idx, metric].values
        
        # Wilcoxon test
        try:
            stat, pval = stats.wilcoxon(method_vals, baseline_vals, alternative="greater")
        except:
            stat, pval = np.nan, np.nan
        
        # Effect size (Cohen's d)
        diff = method_vals - baseline_vals
        d = np.mean(diff) / (np.std(diff) + 1e-8)
        
        results.append({
            "method": method,
            "metric": metric,
            "baseline_mean": baseline_vals.mean(),
            "method_mean": method_vals.mean(),
            "diff_mean": diff.mean(),
            "wilcoxon_stat": stat,
            "p_value": pval,
            "cohens_d": d,
            "n_pairs": len(common_idx),
            "significant": pval < 0.05 if not np.isnan(pval) else False,
        })

if results:
    stats_df = pd.DataFrame(results)
    stats_df.to_csv(OUTPUT_DIR / "statistical_tests.csv", index=False)
    
    print("\n=== Statistical Tests (vs Baseline) ===")
    print(stats_df.to_string())
    
    # Summary
    print("\n=== Significant Improvements (p < 0.05) ===")
    sig = stats_df[stats_df["significant"]]
    if len(sig) > 0:
        for _, row in sig.iterrows():
            print(f"  {row['method']} > baseline on {row['metric']}: "
                  f"Δ={row['diff_mean']:.4f}, p={row['p_value']:.4f}, d={row['cohens_d']:.3f}")
    else:
        print("  No significant improvements found")
else:
    print("[WARN] Could not compute statistical tests")
STATS_EOF

# ==============================================================================
# Final summary
# ==============================================================================
echo ""
echo "============================================================"
echo "[DONE] Analysis completed"
echo "============================================================"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Files generated:"
ls -la "$OUTPUT_DIR"
echo "============================================================"
