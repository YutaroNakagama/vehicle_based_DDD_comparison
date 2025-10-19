# misc/plot_summary_metrics_40.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

IN   = "results/domain_analysis/summary/csv/summary_40cases_test_mode_compare_with_levels.csv"
OUT1 = "results/domain_analysis/summary/png/summary_metrics_bar.png"
OUT2 = "results/domain_analysis/summary/png/summary_diff_heatmap.png"

# --- fallback (for backward compatibility) ---
if not os.path.exists(IN):
    alt_path = IN.replace("_with_levels", "")
    if os.path.exists(alt_path):
        print(f"[WARN] Fallback to: {alt_path}")
        IN = alt_path

df = pd.read_csv(IN)

print("=== DEBUG: Loaded CSV columns ===")
print(df.columns.tolist())
print("=== First rows ===")
print(df.head())

# --- backward compatibility ---
if "stat" in df.columns:
    df = df[df["stat"] == "mean"].copy()
else:
    print("[WARN] 'stat' column not found — skipping filter (JSON-based summary detected).")

# --- detect available modes dynamically ---
modes = []
for mode in ["source_only", "target_only", "joint_train"]:
    if any(col.endswith(mode) for col in df.columns):
        modes.append(mode)
print(f"[INFO] Detected modes: {modes}")

# --- detect available distance/level combinations ---
distances = sorted(df["distance"].unique()) if "distance" in df.columns else ["unknown"]
levels    = sorted(df["level"].unique()) if "level" in df.columns else ["unknown"]
print(f"[INFO] Detected 9 combinations: {[(d,l) for d in distances for l in levels]}")

# --- extract metric base names ---
metrics = sorted(set(c.split("_")[0] for c in df.columns if any(c.endswith(m) for m in modes)))
models  = df["model"].unique().tolist() if "model" in df.columns else ["RF"]

# === BAR PLOT (Model × Metric × Mode) ===
n_rows = len(distances)
n_cols = len(levels)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows), squeeze=False)

for i, dist in enumerate(distances):
    for j, lvl in enumerate(levels):
        ax = axes[i, j]
        sub = df[(df["distance"]==dist)&(df["level"]==lvl)]
        if sub.empty: 
            ax.axis("off")
            continue

        model = sub["model"].iloc[0] if "model" in sub.columns else "RF"
        # --- plot grouped bars for each mode (source_only vs target_only) ---
        labels = [m.upper() for m in metrics]
        x = np.arange(len(metrics))
        width = 0.35

        vals_source, vals_target = [], []
        for metric in metrics:
            col_s = f"{metric}_source_only"
            col_t = f"{metric}_target_only"
            vals_source.append(sub[col_s].mean() if col_s in sub.columns else np.nan)
            vals_target.append(sub[col_t].mean() if col_t in sub.columns else np.nan)

        ax.bar(x - width/2, vals_source, width, label="Source-only", color="#6699cc")
        ax.bar(x + width/2, vals_target, width, label="Target-only", color="#ff9966")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.set_title(f"{dist.upper()} - {lvl} ({model})", fontsize=10)

plt.tight_layout()
os.makedirs(os.path.dirname(OUT1), exist_ok=True)
plt.savefig(OUT1, dpi=200)
print(f"[SAVED] {OUT1}")

# === HEATMAP (Metric difference between modes) ===
comparisons = [
    ("finetune", "only_target", "Finetune - Target"),
    ("finetune", "only_general", "Finetune - General"),
    ("only_general", "only_target", "General - Target"),
]

fig2, axes2 = plt.subplots(len(metrics), len(comparisons),
                           figsize=(4*len(comparisons), 4*len(metrics)))

for i, metric in enumerate(metrics):
    for j, (a, b, title) in enumerate(comparisons):
        diff = []
        for _, r in df.iterrows():
            va, vb = r.get(f"{metric}_{a}", np.nan), r.get(f"{metric}_{b}", np.nan)
            diff.append(va - vb if pd.notna(va) and pd.notna(vb) else np.nan)
        mat = np.array(diff).reshape(-1, 1)
        im = axes2[i, j].imshow(mat, cmap="coolwarm", vmin=-1, vmax=1)
        axes2[i, j].set_yticks(range(len(models)))
        axes2[i, j].set_yticklabels(models)
        axes2[i, j].set_xticks([])
        axes2[i, j].set_title(f"{metric.upper()} ({title})")
        for k, v in enumerate(diff):
            if not np.isnan(v):
                axes2[i, j].text(0, k, f"{v:.2f}", ha="center", va="center", fontsize=9)

plt.tight_layout()
plt.savefig(OUT2, dpi=200)
print(f"[SAVED] {OUT2}")
