# save_radar_pngs.py
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === 入力: wide形式CSV のパス ===
CSV = "/home/s2240011/git/ddd/vehicle_based_DDD_comparison/project/model/common/summary_6groups_only10_vs_finetune_wide.csv"
OUTDIR = os.path.join(os.path.dirname(CSV), "radar_png")
os.makedirs(OUTDIR, exist_ok=True)

# 対象指標（列名の接頭辞）
METRICS = ["accuracy", "f1", "auc", "precision", "recall"]

df = pd.read_csv(CSV)

def radar(ax, labels, vals1, vals2, title, legend=("finetune","only10")):
    N = len(labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    v1 = vals1 + vals1[:1]
    v2 = vals2 + vals2[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_rlabel_position(0)
    ax.set_ylim(0, 1.0)
    ax.grid(True, linewidth=0.5, alpha=0.6)

    ax.plot(angles, v1, linewidth=2)
    ax.fill(angles, v1, alpha=0.10)
    ax.plot(angles, v2, linewidth=2)
    ax.fill(angles, v2, alpha=0.10)

    ax.set_title(title, fontsize=12, pad=12)
    ax.legend(legend, loc="upper right", bbox_to_anchor=(1.25, 1.10), fontsize=9)

# 各グループのPNGを作成
png_paths = []
for _, row in df.iterrows():
    group = row["group"]
    vals_finetune = [row[f"{m}_finetune"] for m in METRICS]
    vals_only10   = [row[f"{m}_only10"]   for m in METRICS]

    fig = plt.figure(figsize=(5, 5))
    ax = plt.subplot(111, polar=True)
    radar(ax, METRICS, vals_finetune, vals_only10, f"{group}")
    out_png = os.path.join(OUTDIR, f"radar_{group}.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)
    png_paths.append(out_png)

# 6枚を1枚にまとめた一覧PNG（任意）
cols = 3
rows = math.ceil(len(png_paths) / cols)
fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*5))
axes = np.array(axes).reshape(rows, cols)

for i, (ax, png) in enumerate(zip(axes.ravel(), png_paths)):
    img = plt.imread(png)
    ax.imshow(img)
    ax.set_axis_off()
    ax.set_title(os.path.basename(png), fontsize=9)

# 余白サブプロットがある場合は非表示
for j in range(len(png_paths), rows*cols):
    axes.ravel()[j].set_visible(False)

plt.tight_layout()
overview_png = os.path.join(OUTDIR, "radar_overview.png")
plt.savefig(overview_png, dpi=200)
plt.close(fig)

print("Saved PNGs to:", OUTDIR)

