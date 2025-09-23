# misc/aggregate_compare_36.py
import re
import os
import glob
import pandas as pd

IN_DIR = "./model/common"
OUT_DIR = "./results/analysis"
os.makedirs(OUT_DIR, exist_ok=True)

rows = []
# 1) fine-tune（= target only）
for fp in glob.glob(os.path.join(IN_DIR, "metrics_RF_rank_*.csv")):
    base = os.path.basename(fp)
    if base.endswith("_evalonly_on_targets.csv"):
        continue
    m = re.match(r"metrics_RF_rank_(.+)\.csv$", base)
    if not m: 
        continue
    key = m.group(1)  # e.g., "dtw_mean_high"
    dist, stat, level = key.split("_")  # dtw/mmd/wasserstein, mean/std, high/low/middle
    mode = "target_only"

    df = pd.read_csv(fp)
    for _, r in df.iterrows():
        rows.append({
            "rank_key": key,
            "distance": dist,
            "stat": stat,
            "level": level,
            "mode": mode,
            "split": r["split"],
            "accuracy": r.get("accuracy"),
            "precision": r.get("precision"),
            "recall": r.get("recall"),
            "f1": r.get("f1"),
            "auc": r.get("auc"),
            "ap": r.get("ap"),
            "source_file": base,
        })

# 2) eval-only（= general only）
for fp in glob.glob(os.path.join(IN_DIR, "metrics_RF_rank_*_evalonly_on_targets.csv")):
    base = os.path.basename(fp)
    m = re.match(r"metrics_RF_rank_(.+)_evalonly_on_targets\.csv$", base)
    if not m:
        continue
    key = m.group(1)  # e.g., "dtw_mean_high"
    dist, stat, level = key.split("_")
    mode = "general_only"

    df = pd.read_csv(fp)
    for _, r in df.iterrows():
        rows.append({
            "rank_key": key,
            "distance": dist,
            "stat": stat,
            "level": level,
            "mode": mode,
            "split": r["split"],
            "accuracy": r.get("accuracy"),
            "precision": r.get("precision"),
            "recall": r.get("recall"),
            "f1": r.get("f1"),
            "auc": r.get("auc"),
            "ap": r.get("ap"),
            "source_file": base,
        })

agg = pd.DataFrame(rows)

# ===== 主要比較テーブル（Test 指標） =====
test = agg[agg["split"]=="test"].copy()

# ① 36条件×2モードの素表
test_sorted = test.sort_values(["distance","stat","level","mode"])
test_sorted.to_csv(os.path.join(OUT_DIR, "compare36_test_full.csv"), index=False)

# ② モード横持ち（AUC/AP）
pivot_auc = test.pivot_table(index=["distance","stat","level","rank_key"],
                             columns="mode", values="auc", aggfunc="mean")
pivot_ap  = test.pivot_table(index=["distance","stat","level","rank_key"],
                             columns="mode", values="ap", aggfunc="mean")

pivot_auc.to_csv(os.path.join(OUT_DIR, "compare36_test_auc_wide.csv"))
pivot_ap.to_csv(os.path.join(OUT_DIR, "compare36_test_ap_wide.csv"))

# ③ 差分列（general_only - target_only）
def with_diff(piv: pd.DataFrame, metric: str):
    out = piv.copy()
    if {"general_only","target_only"}.issubset(out.columns):
        out[metric+"_diff(general-target)"] = out["general_only"] - out["target_only"]
    return out.sort_values(metric+"_diff(general-target)", ascending=False)

with_diff(pivot_auc, "auc").to_csv(os.path.join(OUT_DIR, "compare36_test_auc_diff.csv"))
with_diff(pivot_ap,  "ap" ).to_csv(os.path.join(OUT_DIR, "compare36_test_ap_diff.csv"))

# ④ 距離/統計/レベルごとの平均（各モードの平均性能）
group_mean = test.groupby(["distance","stat","level","mode"])[["auc","ap","f1","accuracy"]].mean().reset_index()
group_mean.to_csv(os.path.join(OUT_DIR, "compare36_group_means.csv"), index=False)

print("Saved:")
for f in ["compare36_test_full.csv",
          "compare36_test_auc_wide.csv",
          "compare36_test_ap_wide.csv",
          "compare36_test_auc_diff.csv",
          "compare36_test_ap_diff.csv",
          "compare36_group_means.csv"]:
    print(" -", os.path.join(OUT_DIR, f))

