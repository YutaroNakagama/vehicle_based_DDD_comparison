# misc/aggregate_summary_40cases.py
import os, re, glob
import json
import pandas as pd
from typing import Optional, Dict

from src import config as cfg

EVAL_DIR = os.path.join(cfg.RESULTS_EVALUATION_PATH, "RF")
OUT_DIR  = os.path.join(cfg.RESULTS_DOMAIN_ANALYSIS_PATH, "summary", "csv")
os.makedirs(OUT_DIR, exist_ok=True)

# --- collect all eval_results_*.json recursively ---
# (extended: handle nested metrics and backward compatibility)
records = []
latest_job_file = os.path.join(EVAL_DIR, cfg.LATEST_JOB_FILENAME)
if os.path.exists(latest_job_file):
    with open(latest_job_file, "r") as f:
        latest_jobid = f.read().strip()
    search_pattern = os.path.join(EVAL_DIR, latest_jobid, "*", "eval_results_*.json")
    print(f"[INFO] Using latest_jobid={latest_jobid} (pattern={search_pattern})")
else:
    search_pattern = os.path.join(EVAL_DIR, "*", "*", "eval_results_*.json")
    print(f"[WARN] latest_job.txt not found — scanning all jobs.")

def _get_pos_block(cr: Dict) -> Optional[Dict]:
    """Return the positive-class block from classification_report.
    This handles historical key variations such as "1", "1.0", "True", "pos", "positive".
    """
    if not isinstance(cr, dict):
        return None
    for k in ("1", "1.0", "True", "pos", "positive"):
        blk = cr.get(k)
        if isinstance(blk, dict):
            return blk
    return None

def _get_metric_from_pos(cr: Dict, field: str) -> Optional[float]:
    """Get a metric field (e.g., 'precision', 'recall', 'f1-score') from the positive-class block.
    Returns None if not found.
    """
    blk = _get_pos_block(cr)
    if blk is None:
        return None
    val = blk.get(field, None)
    try:
        return float(val) if val is not None else None
    except Exception:
        return None



for path in glob.glob(search_pattern):
    try:
        with open(path, "r") as f:
            data = json.load(f)
            # --- robust distance/level extraction ---
            fname = os.path.basename(path)
            dist = data.get("distance")
            level = data.get("level")
            if not dist or not level:
                m = re.search(r"rank_([^_]+(?:_[^_]+)*)_([A-Za-z]+)", fname)
                if m:
                    dist = dist or m.group(1)
                    level = level or m.group(2)
            dist = dist or "unknown"
            level = level or "unknown"

            # --- estimate positive rate (baseline) if available ---
            pos_rate = None
            if "classification_report" in data:
                cr = data["classification_report"]
                # detect binary keys ('0', '1') or float keys ('0.0', '1.0')
                keys = list(cr.keys())
                # detect likely negative/positive label keys
                key_pairs = [
                    ("0", "1"),
                    ("0.0", "1.0"),
                    ("False", "True"),
                    ("neg", "pos"),
                    ("negative", "positive"),
                ]
                for k0, k1 in key_pairs:
                    if k0 in cr and k1 in cr and "support" in cr[k0] and "support" in cr[k1]:
                        n0 = cr[k0]["support"]
                        n1 = cr[k1]["support"]
                        pos_rate = n1 / (n0 + n1)
                        break
            else:
                pos_rate = data.get("pos_rate") or data.get("positive_rate")

            cr = data.get("classification_report", {}) or {}
            # backward-compatible columns (robust metric extraction)
            row = {
                "file": fname,
                "model": data.get("model", "RF"),
                "mode": data.get("mode", "source_only"),
                "distance": dist,
                "level": level,
                "pos_rate": pos_rate,
                # --- safely extract metrics whether top-level or inside classification_report ---
                "auc": (
                    data.get("auc")
                    or data.get("roc_auc")
                    or data.get("metrics", {}).get("auc")
                ),
                # --- Add AUPRC (Average Precision) ---
                "auc_pr": (
                    data.get("auc_pr")
                    or data.get("metrics", {}).get("auc_pr")
                    or data.get("pr_curve", {}).get("auc_pr")
                ),
                # --- Positive-class F1 (binary, pos_label=1). Fall back only if truly missing.
                "f1": (
                    data.get("f1_pos")
                    or _get_metric_from_pos(cr, "f1-score")
                    or cr.get("macro avg", {}).get("f1-score")
                ),
                "mse": data.get("mse") or data.get("metrics", {}).get("mse"),
                "accuracy": (
                    data.get("accuracy")
                    or cr.get("accuracy")
                ),
                # --- Positive-class precision/recall; keep explicit and avoid accidental accuracy/weighted swaps.
                "precision": (
                    data.get("precision_pos")
                    or _get_metric_from_pos(cr, "precision")
                    or cr.get("macro avg", {}).get("precision")
                ),
                "recall": (
                    data.get("recall_pos")
                    or _get_metric_from_pos(cr, "recall")
                    or cr.get("macro avg", {}).get("recall")
                ),
                "specificity": data.get("specificity"),
                # --- Thresholded (positive-class) metrics. Prefer explicit *_thr_pos; keep legacy keys as fallback.
                "precision_thr": (
                    data.get("precision_thr_pos")
                    or data.get("prec_thr")
                ),
                "recall_thr": (
                    data.get("recall_thr_pos")
                    or data.get("recall_thr")
                ),
                "f1_thr": (
                    data.get("f1_thr_pos")
                    or data.get("f1_thr")
                ),
                "f2_thr": (
                    data.get("f2_thr_pos")
                    or data.get("f2_thr")
                ),
                "specificity_thr": data.get("specificity_thr"),
                "split": "test",  # for compatibility with old structure
            }

            # --- New: derive non-threshold F2 from precision/recall when available ---
            # F2 = 5 * P * R / (4*P + R)
            try:
                P = row.get("precision", None)
                R = row.get("recall", None)
                if P is not None and R is not None and (4*P + R) not in (0, None):
                    row["f2"] = 5 * P * R / (4 * P + R)
                else:
                    # fallback if JSON already provided a non-threshold F2
                    row["f2"] = data.get("f2")
            except Exception:
                row["f2"] = data.get("f2")

            records.append(row)
    except Exception as e:
        print(f"[WARN] skip {path}: {e}")

if not records:
    raise FileNotFoundError(f"No eval_results_*.json found under {EVAL_DIR}")

all_metrics = pd.DataFrame(records)
print(f"[INFO] Loaded {len(all_metrics)} evaluation JSONs.")

all_path = os.path.join(OUT_DIR, "summary_40cases_all_splits.csv")
all_metrics.to_csv(all_path, index=False)

test_df = all_metrics[all_metrics["split"]=="test"].copy()
if "level" in test_df.columns:
    cat = pd.CategoricalDtype(categories=["high", "middle", "low"], ordered=True)
    test_df["level"] = test_df["level"].astype(cat)

test_path = os.path.join(OUT_DIR, "summary_40cases_test.csv")
test_df.to_csv(test_path, index=False)

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
cmp.to_csv(cmp_path, index=False)
print("[INFO] Added distance/level columns for domain-level visualization.")

print("saved:", all_path)
print("saved:", test_path)
print("saved:", cmp_path)

