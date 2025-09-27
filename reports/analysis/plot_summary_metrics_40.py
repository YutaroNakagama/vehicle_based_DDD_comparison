# misc/plot_summary_metrics_40.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

IN   = "results/analysis/summary_40cases_test_mode_compare.csv"
OUT1 = "results/analysis/summary_metrics_40_mean_tri_bar.png"
OUT2 = "results/analysis/diff_heatmap_all5_40.png"

df = pd.read_csv(IN)

print("=== DEBUG: Loaded CSV columns ===")
print(df.columns.tolist())
print("=== First rows ===")
print(df.head())

df = df[df["stat"] == "mean"].copy()

dist_order  = ["dtw", "mmd", "wasserstein"]
level_order = ["high", "middle", "low"]

metrics = [
    ("auc", "AUC"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("accuracy", "Accuracy"),
    ("f1", "F1")
]

modes = [("only_general", "General"), ("only_target", "Target"), ("finetune", "Finetune")]

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
            col_name = f"{m_key}_{mode_key}"
            if col_name not in sub_d.columns:
                vals.append([np.nan] * len(level_order))
                continue
            vals.append([
                sub_d.loc[sub_d["level"] == lv, col_name].values[0]
                if not sub_d.loc[sub_d["level"] == lv, col_name].empty
                else np.nan
                for lv in level_order
            ])

        offsets = [-width, 0, width]
        for k, (mode_key, mode_label) in enumerate(modes):
            if all(np.isnan(vals[k])):  
                continue
            ax.bar(x + offsets[k], vals[k], width=width, label=mode_label)

        ax.set_xticks(x)
        ax.set_xticklabels(level_order)
        ax.set_title(f"{dist} - {m_label}")
        ax.set_ylim(0, 1.05)
        if j == 0:
            ax.set_ylabel("Score")
        if i == 0 and j == n_cols - 1:
            ax.legend(loc="upper right")

plt.tight_layout()
os.makedirs(os.path.dirname(OUT1), exist_ok=True)
plt.savefig(OUT1, dpi=200)
print(f"Saved: {OUT1}")

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
            rec = {
                "distance": dist,
                "level": lv,
                "only_general": r.get(f"{metric_key}_only_general", np.nan),
                "only_target":  r.get(f"{metric_key}_only_target", np.nan),
            }
            if f"{metric_key}_finetune" in r.index:
                rec["finetune"] = r[f"{metric_key}_finetune"]
            else:
                rec["finetune"] = np.nan
            recs.append(rec)
    return pd.DataFrame(recs)

def draw_heatmaps(metric_key: str, axs_row):
    base = make_diff_df(metric_key)
    for ax, (a, b, comp_title) in zip(axs_row, comparisons):
        if a not in base.columns or b not in base.columns:
            ax.axis("off")
            ax.set_title(f"{metric_key.upper()} ({comp_title}) [SKIPPED]")
            continue

        diff = base.copy()
        diff["diff"] = diff[a] - diff[b]
        mat = diff.pivot(index="distance", columns="level", values="diff") \
                 .reindex(index=dist_order, columns=level_order)

        im = ax.imshow(mat.values, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(level_order))); ax.set_xticklabels(level_order)
        ax.set_yticks(range(len(dist_order)));  ax.set_yticklabels(dist_order)
        ax.set_title(f"{metric_key.upper()} ({comp_title})")
        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                val = mat.values[r, c]
                if not np.isnan(val):
                    ax.text(c, r, f"{val:.3f}", ha="center", va="center", fontsize=9)

fig2, axes2 = plt.subplots(5, 3, figsize=(18, 20))
draw_heatmaps("auc",       axes2[0, :])
draw_heatmaps("precision", axes2[1, :])
draw_heatmaps("recall",    axes2[2, :])
draw_heatmaps("accuracy",  axes2[3, :])
draw_heatmaps("f1",        axes2[4, :])

plt.tight_layout()
plt.savefig(OUT2, dpi=200)
print(f"Saved: {OUT2}")

