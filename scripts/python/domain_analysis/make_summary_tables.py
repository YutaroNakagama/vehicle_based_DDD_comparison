# misc/aggregate_summary_40cases.py
import os, re, glob
import pandas as pd

import json

EVAL_DIR = "results/evaluation/RF"
OUT_DIR  = "results/domain_analysis/summary/csv"
os.makedirs(OUT_DIR, exist_ok=True)

# --- collect all eval_results_*.json recursively ---
# (extended: handle nested metrics and backward compatibility)
records = []
for path in glob.glob(os.path.join(EVAL_DIR, "*", "*", "eval_results_*.json")):
    try:
        with open(path, "r") as f:
            data = json.load(f)
            # --- extract distance & level from filename ---
            fname = os.path.basename(path)
            # accept both lowercase and uppercase for robustness
            m = re.search(r"rank_([A-Za-z]+)_mean_([A-Za-z]+)", fname)
            dist, level = (m.groups() if m else ("unknown", "unknown"))

            # backward-compatible columns (robust metric extraction)
            row = {
                "file": fname,
                "model": data.get("model", "RF"),
                "mode": data.get("mode", "source_only"),
                "distance": dist,
                "level": level,
                # --- safely extract metrics whether top-level or inside classification_report ---
                "auc": (
                    data.get("auc")
                    or data.get("roc_auc")
                    or data.get("metrics", {}).get("auc")
                ),
                "f1": (
                    data.get("f1")
                    or data.get("classification_report", {}).get("weighted avg", {}).get("f1-score")
                ),
                "mse": data.get("mse") or data.get("metrics", {}).get("mse"),
                "accuracy": (
                    data.get("accuracy")
                    or data.get("classification_report", {}).get("accuracy")
                ),
                "precision": (
                    data.get("precision")
                    or data.get("classification_report", {}).get("weighted avg", {}).get("precision")
                ),
                "recall": (
                    data.get("recall")
                    or data.get("classification_report", {}).get("weighted avg", {}).get("recall")
                ),
                "split": "test",  # for compatibility with old structure
            }
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
pivot_prec = pivot_metric(test_df, "precision")
pivot_rec  = pivot_metric(test_df, "recall")
pivot_acc  = pivot_metric(test_df, "accuracy")
pivot_f1   = pivot_metric(test_df, "f1")

cmp = (
    pivot_auc
    .merge(pivot_prec, on=["model", "distance", "level"], how="outer")
    .merge(pivot_rec,  on=["model", "distance", "level"], how="outer")
    .merge(pivot_acc,  on=["model", "distance", "level"], how="outer")
    .merge(pivot_f1,   on=["model", "distance", "level"], how="outer")
)

def add_delta(df, metric):
    a = f"{metric}_only_general"
    b = f"{metric}_finetune"
    c = f"{metric}_only_target"
    if a in df.columns and b in df.columns:
        df[f"delta_{metric}_finetune_vs_only_general"] = df[b] - df[a]
    if c in df.columns and b in df.columns:
        df[f"delta_{metric}_finetune_vs_only_target"] = df[b] - df[c]

for met in ["auc","precision","recall","accuracy","f1"]:
    add_delta(cmp, met)


cmp_path = os.path.join(OUT_DIR, "summary_40cases_test_mode_compare_with_levels.csv")
cmp.to_csv(cmp_path, index=False)
print("[INFO] Added distance/level columns for domain-level visualization.")

print("saved:", all_path)
print("saved:", test_path)
print("saved:", cmp_path)

