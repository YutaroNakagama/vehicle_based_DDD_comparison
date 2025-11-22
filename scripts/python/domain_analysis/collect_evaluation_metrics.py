# misc/aggregate_summary_40cases.py
import os
import pandas as pd
from pathlib import Path

from src import config as cfg
from src.utils.io.data_io import load_json_glob, save_csv
from src.utils.metrics_helper import extract_metrics_from_eval_json

EVAL_DIR = os.path.join(cfg.RESULTS_EVALUATION_PATH, "RF")
OUT_DIR  = os.path.join(cfg.RESULTS_DOMAIN_ANALYSIS_PATH, "summary", "csv")
os.makedirs(OUT_DIR, exist_ok=True)

# --- collect all eval_results_*.json recursively ---
latest_job_file = os.path.join(EVAL_DIR, cfg.LATEST_JOB_FILENAME)
if os.path.exists(latest_job_file):
    with open(latest_job_file, "r") as f:
        latest_jobid = f.read().strip()
    search_pattern = os.path.join(EVAL_DIR, latest_jobid, "*", "eval_results_*.json")
    print(f"[INFO] Using latest_jobid={latest_jobid} (pattern={search_pattern})")
else:
    search_pattern = os.path.join(EVAL_DIR, "*", "*", "eval_results_*.json")
    print(f"[WARN] latest_job.txt not found — scanning all jobs.")

# Load all JSON files using common utility
json_files = load_json_glob(search_pattern, skip_errors=True)

if not json_files:
    raise FileNotFoundError(f"No eval_results_*.json found under {EVAL_DIR}")

# Extract metrics from each JSON file
records = []
for path, data in json_files:
    filename = path.name
    metrics = extract_metrics_from_eval_json(data, filename=filename)
    records.append(metrics)

all_metrics = pd.DataFrame(records)
print(f"[INFO] Loaded {len(all_metrics)} evaluation JSONs.")

# Save all metrics
all_path = os.path.join(OUT_DIR, "summary_40cases_all_splits.csv")
save_csv(all_metrics, all_path)

test_df = all_metrics[all_metrics["split"]=="test"].copy()
if "level" in test_df.columns:
    cat = pd.CategoricalDtype(categories=["high", "middle", "low"], ordered=True)
    test_df["level"] = test_df["level"].astype(cat)

test_path = os.path.join(OUT_DIR, "summary_40cases_test.csv")
save_csv(test_df, test_path)

# --- ensure distance/level exist before pivoting ---
for col in ["distance", "level"]:
    if col not in test_df.columns:
        test_df[col] = "unknown"

def pivot_metric(df, metric):
    """Simplified pivot for JSON-based results (no rank_key hierarchy)."""
    pv = df.pivot_table(
        index=["model", "distance", "level"],
        columns="mode",
        values=metric,
        aggfunc="mean"
    ).reset_index()

    # --- Ensure distance/level columns are preserved even if pivot drops them ---
    for col in ["distance", "level"]:
        if col not in pv.columns:
            pv[col] = "unknown"

    pv = pv.rename(columns={
        c: f"{metric}_{c}" for c in pv.columns if c not in ["model", "distance", "level"]
    })
    return pv

# --- simplified summary ---
pivot_auc  = pivot_metric(test_df, "auc")
pivot_aucpr = pivot_metric(test_df, "auc_pr")
pivot_prec = pivot_metric(test_df, "precision")
pivot_rec  = pivot_metric(test_df, "recall")
pivot_acc  = pivot_metric(test_df, "accuracy")
pivot_f1   = pivot_metric(test_df, "f1")
pivot_f2   = pivot_metric(test_df, "f2")

cmp = (
    pivot_auc
    .merge(pivot_aucpr, on=["model", "distance", "level"], how="outer")
    .merge(pivot_prec, on=["model", "distance", "level"], how="outer")
    .merge(pivot_rec,  on=["model", "distance", "level"], how="outer")
    .merge(pivot_f1,   on=["model", "distance", "level"], how="outer")
    .merge(pivot_f2,   on=["model", "distance", "level"], how="outer")
    .merge(pivot_acc,  on=["model", "distance", "level"], how="outer")
)

def add_delta(df, metric):
    a = f"{metric}_source_only"
    c = f"{metric}_target_only"
    if a in df.columns and c in df.columns:
        df[f"delta_{metric}_source_minus_target"] = df[a] - df[c]
 
for met in ["auc", "auc_pr", "precision", "recall", "accuracy", "f1", "f2"]:
    add_delta(cmp, met)

cmp_path = os.path.join(OUT_DIR, "summary_40cases_test_mode_compare_with_levels.csv")
save_csv(cmp, cmp_path)
print("[INFO] Added distance/level columns for domain-level visualization.")

print("saved:", all_path)
print("saved:", test_path)
print("saved:", cmp_path)
