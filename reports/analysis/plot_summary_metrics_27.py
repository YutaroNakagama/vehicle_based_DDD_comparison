# misc/plot_summary_metrics_27.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

IN   = "results/analysis/summary_27cases_test_mode_compare.csv"
OUT1 = "results/analysis/summary_metrics_27_mean_tri_bar.png"
OUT2 = "results/analysis/diff_heatmap_auc_ap_27_tri.png"

df = pd.read_csv(IN)

# mean のみを対象
df = df[df["stat"] == "mean"].copy()

# 描画順の指定
dist_order  = ["dtw", "mmd", "wasserstein"]
level_order = ["high", "middle", "low"]

# CSV の列名サフィックスに対応
metrics = [("auc", "AUC"), ("ap", "AP"), ("acc", "Accuracy"), ("f1", "F1")]

# 3モードを横並びで比較
# 表示順と凡例名
modes = [("only_general", "General"), ("only_target", "Target"), ("finetune", "Finetune")]

# ========== 図1: 3モード比較の棒グラフ ==========
n_rows = len(dist_order)
n_cols = len(metrics)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4*n_rows), sharey=False)
if n_rows == 1:
    axes = [axes]

for i, dist in enumerate(dist_order):
    sub_d = df[df["distance"] == dist].copy()
    if sub_d.empty:
        continue
    sub_d["level"] = pd.Categorical(sub_d["level"], level_order, ordered=True)
    sub_d = sub_d.sort_values("level")

    for j, (m_key, m_label) in enumerate(metrics):
        ax = axes[i][j] if n_rows > 1 else axes[j]
        width = 0.23
        x = np.arange(len(level_order))

        vals = []
        for mode_key, _ in modes:
            vals.append([
                sub_d.loc[sub_d["level"] == lv, f"{mode_key}_{m_key}"].values[0]
                if not sub_d.loc[sub_d["level"] == lv, f"{mode_key}_{m_key}"].empty
                else np.nan
                for lv in level_order
            ])

        # 3 本バー（General / Target / Finetune）
        ax.bar(x - width,     vals[0], width=width, label=modes[0][1])
        ax.bar(x,             vals[1], width=width, label=modes[1][1])
        ax.bar(x + width,     vals[2], width=width, label=modes[2][1])

        ax.set_xticks(x)
        ax.set_xticklabels(level_order)
        ax.set_title(f"{dist} - {m_label}")
        # 精度系は 0〜1 を想定
        ax.set_ylim(0, 1.05)
        if j == 0:
            ax.set_ylabel("Score")
        if i == 0 and j == n_cols - 1:
            ax.legend(loc="upper right")

plt.tight_layout()
os.makedirs(os.path.dirname(OUT1), exist_ok=True)
plt.savefig(OUT1, dpi=200)
print(f"Saved: {OUT1}")

# ========== 図2: 差分ヒートマップ（AUC/AP） ==========
# 3種類の差分:
#   1) General - Target
#   2) Finetune - Target
#   3) General - Finetune
comparisons = [
    ("only_general", "only_target",   "General - Target"),
    ("finetune",     "only_target",   "Finetune - Target"),
    ("only_general", "finetune",      "General - Finetune"),
]

def make_diff_df(metric_key: str):
    recs = []
    for dist in dist_order:
        for lv in level_order:
            row = df[(df["distance"] == dist) & (df["level"] == lv)]
            if len(row) != 1:
                continue
            r = row.iloc[0]
            recs.append({
                "distance": dist,
                "level": lv,
                "only_general": r[f"only_general_{metric_key}"],
                "only_target":  r[f"only_target_{metric_key}"],
                "finetune":     r[f"finetune_{metric_key}"],
            })
    return pd.DataFrame(recs)

def draw_heatmaps(metric_key: str, axs_row, title_suffix: str):
    base = make_diff_df(metric_key)
    for ax, (a, b, comp_title) in zip(axs_row, comparisons):
        diff = base.copy()
        diff["diff"] = diff[a] - diff[b]
        mat = diff.pivot(index="distance", columns="level", values="diff") \
                 .reindex(index=dist_order, columns=level_order)

        im = ax.imshow(mat.values, aspect="auto")
        ax.set_xticks(range(len(level_order))); ax.set_xticklabels(level_order)
        ax.set_yticks(range(len(dist_order)));  ax.set_yticklabels(dist_order)
        ax.set_title(f"{metric_key.upper()} ({comp_title})")
        # 値をセル内に表示
        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                val = mat.values[r, c]
                if not np.isnan(val):
                    ax.text(c, r, f"{val:.3f}", ha="center", va="center", fontsize=9)

fig2, axes2 = plt.subplots(2, 3, figsize=(16, 8))
draw_heatmaps("auc", axes2[0, :], "AUC")
draw_heatmaps("ap",  axes2[1, :], "AP")
plt.tight_layout()
plt.savefig(OUT2, dpi=200)
print(f"Saved: {OUT2}")

