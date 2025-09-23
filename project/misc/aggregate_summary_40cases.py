# misc/aggregate_summary_40cases.py
import os, re, glob
import pandas as pd

MODEL_DIR = "model/common"             # metrics_*.csv の保存先
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

#def infer_mode_from_fname(fname: str):
#    if fname.endswith("_evalonly_on_targets.csv"):
#        return "only_general"
#    elif fname.endswith("_finetune.csv"):
#        return "finetune"
#    elif fname.endswith("_only_target.csv"):
#        return "only_target"
#    return None

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
        continue

    mode = infer_mode_from_fname(base)
    if mode is None:
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
            "ap": r.get("ap"),
            "accuracy": r.get("accuracy"),
            "f1": r.get("f1"),
        })

all_metrics = pd.DataFrame(rows)

# ① 全split
all_path = os.path.join(OUT_DIR, "summary_40cases_all_splits.csv")
all_metrics.to_csv(all_path, index=False)

# ② testのみ
test_df = all_metrics[all_metrics["split"]=="test"].copy()
test_path = os.path.join(OUT_DIR, "summary_40cases_test.csv")
test_df.to_csv(test_path, index=False)

# ③ モード比較（test）
pivot_auc = test_df.pivot_table(
    index=["distance","stat","level","rank_key"],
    columns="mode", values="auc", aggfunc="mean"
).reset_index()
pivot_ap = test_df.pivot_table(
    index=["distance","stat","level","rank_key"],
    columns="mode", values="ap", aggfunc="mean"
).reset_index()
pivot_acc = test_df.pivot_table(
    index=["distance","stat","level","rank_key"],
    columns="mode", values="accuracy", aggfunc="mean"
).reset_index()
pivot_f1 = test_df.pivot_table(
    index=["distance","stat","level","rank_key"],
    columns="mode", values="f1", aggfunc="mean"
).reset_index()

cmp = pivot_auc.merge(pivot_ap, on=["distance","stat","level","rank_key"], suffixes=("_auc","_ap"))
cmp = cmp.merge(pivot_acc, on=["distance","stat","level","rank_key"])
cmp = cmp.merge(pivot_f1, on=["distance","stat","level","rank_key"], suffixes=("_acc","_f1"))

#cmp.columns = [c if isinstance(c, str) else "_".join([x for x in c if x]) for c in cmp.columns]
# 列名を "{mode}_{metric}" 形式に変換する
new_cols = []
for c in cmp.columns:
    if isinstance(c, str):
        new_cols.append(c)
    else:
        # MultiIndex の場合 (例: ('auc', 'only_general'))
        if len(c) == 2 and c[0] and c[1]:
            new_cols.append(f"{c[1]}_{c[0]}")
        else:
            new_cols.append("_".join([x for x in c if x]))
cmp.columns = new_cols

def add_delta(df, metric):
    a = f"{metric}_only_general"
    b = f"{metric}_finetune"
    c = f"{metric}_only_target"
    if a in df.columns and b in df.columns:
        df[f"delta_{metric}_finetune_vs_only_general"] = df[b] - df[a]
    if c in df.columns and b in df.columns:
        df[f"delta_{metric}_finetune_vs_only_target"] = df[b] - df[c]

for met in ["auc","ap","accuracy","f1"]:
    add_delta(cmp, met)

cmp_path = os.path.join(OUT_DIR, "summary_40cases_test_mode_compare.csv")
cmp.to_csv(cmp_path, index=False)

print("saved:", all_path)
print("saved:", test_path)
print("saved:", cmp_path)

