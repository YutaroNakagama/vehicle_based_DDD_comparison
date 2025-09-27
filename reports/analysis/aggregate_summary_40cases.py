# misc/aggregate_summary_40cases.py
import os, re, glob
import pandas as pd

MODEL_DIR = "results/evaluation/common"
OUT_DIR   = "results/analysis"
os.makedirs(OUT_DIR, exist_ok=True)

RANK_KEYS = [
    "dtw_mean_high", "dtw_mean_middle", "dtw_mean_low",
    "mmd_mean_high", "mmd_mean_middle", "mmd_mean_low",
    "wasserstein_mean_high", "wasserstein_mean_middle", "wasserstein_mean_low"
]

def infer_rank_key(text: str):
    for k in RANK_KEYS:
        if k in text:
            return k
    return None

def split_rank_key(k: str):
    if not k: return (None, None, None)
    parts = k.split("_")
    if len(parts) != 3: return (None, None, None)
    return parts[0], parts[1], parts[2]

def infer_mode_from_fname(fname: str):
    if fname.endswith("_evalonly_on_targets.csv"):
        return "only_general"
    elif fname.endswith("_common_evalonly.csv"):
        return "only_target"
    elif fname.endswith("_finetune.csv"):
        return "finetune"
    elif fname.endswith("_only_target.csv"):
        return "only_target"
    return None

rows = []
for path in glob.glob(os.path.join(MODEL_DIR, "metrics_*.csv")):
    base = os.path.basename(path)
    try:
        df = pd.read_csv(path)
    except Exception:
        continue
    if "split" not in df.columns:
        df["split"] = "test"

    mode = infer_mode_from_fname(base)
    if mode is None:
        if "rank_" in base:
            mode = "only_target"
        else:
            continue
    rank_key = infer_rank_key(base)
    dist, stat, level = split_rank_key(rank_key)

    m = re.match(r"metrics_([^_]+)_", base)
    model = m.group(1) if m else ""

    for _, r in df.iterrows():
        rows.append({
            "file": base,
            "model": model,
            "distance": dist,
            "stat": stat,
            "level": level,
            "rank_key": rank_key,
            "mode": mode,
            "split": r.get("split"),
            "auc": r.get("auc"),
            #"ap": r.get("ap"),
            "precision": r.get("weighted_avg_precision"),
            "recall": r.get("weighted_avg_recall"),
            "accuracy": r.get("accuracy"),
            "f1": r.get("f1"),
        })

all_metrics = pd.DataFrame(rows)

all_path = os.path.join(OUT_DIR, "summary_40cases_all_splits.csv")
all_metrics.to_csv(all_path, index=False)

test_df = all_metrics[all_metrics["split"]=="test"].copy()
test_path = os.path.join(OUT_DIR, "summary_40cases_test.csv")
test_df.to_csv(test_path, index=False)

# --- Helper: pivot per metric and enforce naming {metric}_{mode} ---
def pivot_metric(df, metric):
    pv = df.pivot_table(
        index=["distance","stat","level","rank_key"],
        columns="mode", values=metric, aggfunc="mean"
    ).reset_index()
    pv = pv.rename(columns={
        c: f"{metric}_{c}" for c in pv.columns if c not in ["distance","stat","level","rank_key"]
    })
    return pv

pivot_auc  = pivot_metric(test_df, "auc")
pivot_prec = pivot_metric(test_df, "precision")
pivot_rec  = pivot_metric(test_df, "recall")
pivot_acc  = pivot_metric(test_df, "accuracy")
pivot_f1   = pivot_metric(test_df, "f1")

cmp = pivot_auc \
    .merge(pivot_prec, on=["distance","stat","level","rank_key"]) \
    .merge(pivot_rec,  on=["distance","stat","level","rank_key"]) \
    .merge(pivot_acc,  on=["distance","stat","level","rank_key"]) \
    .merge(pivot_f1,   on=["distance","stat","level","rank_key"])

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


cmp_path = os.path.join(OUT_DIR, "summary_40cases_test_mode_compare.csv")
cmp.to_csv(cmp_path, index=False)

print("saved:", all_path)
print("saved:", test_path)
print("saved:", cmp_path)

