# misc/aggregate_compare_40.py
import re
import os
import glob
import pandas as pd

IN_DIR = "./model/common"   # rank40用の保存先
OUT_DIR = "./results/analysis"
os.makedirs(OUT_DIR, exist_ok=True)

rows = []

# === 全モード (finetune, only_general, only_target) を収集 ===
for fp in glob.glob(os.path.join(IN_DIR, "metrics_RF_rank_*.csv")):
    base = os.path.basename(fp)

    # モード判定
    if "_evalonly_on_targets" in base:
        mode = "only_general"
        key = re.match(r"metrics_RF_rank_(.+)_evalonly_on_targets\.csv$", base).group(1)
    elif "_finetune.csv" in base:
        mode = "finetune"
        key = re.match(r"metrics_RF_rank_(.+)_finetune\.csv$", base).group(1)
    else:
        mode = "only_target"
        key = re.match(r"metrics_RF_rank_(.+)\.csv$", base).group(1)

    if not key:
        continue

    dist, stat, level = key.split("_")  # e.g., dtw_mean_high

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

# ===== テスト指標だけに絞って保存 =====
test = agg[agg["split"] == "test"].copy()

# ① 生の表
test_sorted = test.sort_values(["distance","stat","level","mode"])
test_sorted.to_csv(os.path.join(OUT_DIR, "compare40_test_full.csv"), index=False)

# ② ピボット形式（AUC / AP）
pivot_auc = test.pivot_table(index=["distance","stat","level","rank_key"],
                             columns="mode", values="auc", aggfunc="mean")
pivot_ap = test.pivot_table(index=["distance","stat","level","rank_key"],
                             columns="mode", values="ap", aggfunc="mean")

pivot_auc.to_csv(os.path.join(OUT_DIR, "compare40_test_auc_wide.csv"))
pivot_ap.to_csv(os.path.join(OUT_DIR, "compare40_test_ap_wide.csv"))

# ③ 平均性能
group_mean = test.groupby(["distance","stat","level","mode"])[
    ["auc","ap","f1","accuracy"]
].mean().reset_index()
group_mean.to_csv(os.path.join(OUT_DIR, "compare40_group_means.csv"), index=False)

# ④ 27cases形式の summary を生成（プロット用）
summary_out = os.path.join(OUT_DIR, "summary_40cases_test_mode_compare.csv")
pivot = test.pivot_table(index=["distance","stat","level","rank_key"],
                         columns="mode",
                         values=["auc","ap","accuracy","f1"],
                         aggfunc="mean").reset_index()

# MultiIndex を平坦化
pivot.columns = ["_".join([c for c in col if c]) for col in pivot.columns.to_flat_index()]
pivot.to_csv(summary_out, index=False)

print("Saved:")
for f in ["compare40_test_full.csv",
          "compare40_test_auc_wide.csv",
          "compare40_test_ap_wide.csv",
          "compare40_group_means.csv",
          "summary_40cases_test_mode_compare.csv"]:
    print(" -", os.path.join(OUT_DIR, f))

