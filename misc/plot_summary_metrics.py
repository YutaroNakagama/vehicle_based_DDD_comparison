import os
import pandas as pd
import matplotlib.pyplot as plt

IN = "results/analysis/summary_metrics_36.csv"
OUT = "results/analysis/summary_metrics_mean.png"

df = pd.read_csv(IN)

# ① mean だけに絞る
df = df[df["stat"] == "mean"].copy()

# ② 指標リスト
metrics = ["auc", "ap", "accuracy", "f1"]

# ③ distance ごとに図を分ける
distances = df["distance"].unique()
fig, axes = plt.subplots(len(distances), len(metrics), figsize=(16, 4*len(distances)), sharey=False)

if len(distances) == 1:
    axes = [axes]  # サブプロットが1段だけなら二重リスト化

for i, dist in enumerate(distances):
    sub = df[df["distance"] == dist]

    for j, m in enumerate(metrics):
        ax = axes[i][j] if len(distances) > 1 else axes[j]
        width = 0.35
        levels = ["high", "middle", "low"]

        x = range(len(levels))
        general_vals = [sub.loc[sub["level"]==lv, f"{m}_general"].values[0] for lv in levels]
        target_vals  = [sub.loc[sub["level"]==lv, f"{m}_target"].values[0]  for lv in levels]

        ax.bar([p - width/2 for p in x], general_vals, width=width, label="General")
        ax.bar([p + width/2 for p in x], target_vals,  width=width, label="Target")

        ax.set_xticks(x)
        ax.set_xticklabels(levels)
        ax.set_title(f"{dist} - {m.upper()}")
        ax.set_ylim(0, 1.05)

        if j == 0:
            ax.set_ylabel("Score")

        if i == 0 and j == len(metrics)-1:
            ax.legend()

plt.tight_layout()
plt.savefig(OUT, dpi=200)
print(f"Saved: {OUT}")

