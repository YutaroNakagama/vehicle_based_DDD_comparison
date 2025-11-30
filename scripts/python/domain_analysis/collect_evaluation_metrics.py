# misc/aggregate_summary_40cases.py
import os
import pandas as pd
from pathlib import Path

from src import config as cfg
from src.utils.io.data_io import load_json_glob, save_csv
from src.utils.evaluation.metrics import extract_metrics_from_eval_json

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

# Save only test metrics
test_df = all_metrics[all_metrics["split"]=="test"].copy()
if "level" in test_df.columns:
    cat = pd.CategoricalDtype(categories=["out_domain", "mid_domain", "in_domain"], ordered=True)
    test_df["level"] = test_df["level"].astype(cat)

test_path = os.path.join(OUT_DIR, "summary_40cases_test.csv")
save_csv(test_df, test_path)

print(f"[INFO] Saved summary table: {test_path}")
