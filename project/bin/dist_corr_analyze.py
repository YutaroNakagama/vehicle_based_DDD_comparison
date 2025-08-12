import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 各結果ディレクトリ（適宜パス調整）
result_dirs = {
    "MMD": "model/common/dist_corr_mmd/correlations_dUG_vs_deltas.csv",
    "Wasserstein": "model/common/dist_corr_wasserstein/correlations_dUG_vs_deltas.csv",
    "DTW": "model/common/dist_corr_dtw/correlations_dUG_vs_deltas.csv",
}

# CSV読み込み＆整形
dfs = []
for name, path in result_dirs.items():
    df = pd.read_csv(path)
    df["distance_type"] = name
    dfs.append(df)

pearson_df = pd.concat(dfs, ignore_index=True)

# ターミナル出力
print("=== Pearson and Spearman Correlations for dUG vs Δmetrics ===")
print(pearson_df)

# 保存
pearson_df.to_csv("correlation_summary_all.csv", index=False)

# ヒートマップ作成（Pearsonのみ例示）
pivot_df = pearson_df.pivot(index="metric", columns="distance_type", values="pearson_r")

plt.figure(figsize=(8, 5))
sns.heatmap(pivot_df, annot=True, cmap="coolwarm", center=0, fmt=".2f")
plt.title("Pearson correlation (dUG vs Δmetrics)")
plt.tight_layout()
plt.savefig("correlation_heatmap_all.png", dpi=300)
plt.close()

