# misc/plot_summary_metrics_40.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib as mpl
import logging

mpl.set_loglevel("warning")  # Suppress overall Matplotlib logs
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # Stop findfont spam

# Initialize module logger (used below)
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


IN   = "results/domain_analysis/summary/csv/summary_40cases_test_mode_compare_with_levels.csv"
OUT1 = "results/domain_analysis/summary/png/summary_metrics_bar.png"
OUT2 = "results/domain_analysis/summary/png/summary_diff_heatmap.png"

# --- fallback (for backward compatibility) ---
if not os.path.exists(IN):
    alt_path = IN.replace("_with_levels", "")
    if os.path.exists(alt_path):
        logger.warning(f"Fallback to: {alt_path}")
        IN = alt_path

df = pd.read_csv(IN)

# --- Determine baseline positive rate dynamically (for AUPRC baseline line only) ---
base_csv = IN.replace("_mode_compare_with_levels.csv", ".csv")
if os.path.exists(base_csv):
    df_base = pd.read_csv(base_csv)
    if "pos_rate" in df_base.columns and df_base["pos_rate"].notna().any():
        BASELINE_POS_RATE = df_base["pos_rate"].mean()
        logger.info(f"Using mean pos_rate from {os.path.basename(base_csv)}: {BASELINE_POS_RATE:.4f}")
    else:
        BASELINE_POS_RATE = 0.033
        logger.warning(f"pos_rate missing in {base_csv}, using default baseline={BASELINE_POS_RATE:.4f}")
else:
    BASELINE_POS_RATE = 0.033
    logger.warning(f"{base_csv} not found, using default baseline={BASELINE_POS_RATE:.4f}")

logger.debug("=== DEBUG: Loaded CSV columns ===")
logger.debug(df.columns.tolist())
logger.debug("=== First rows ===")
logger.debug(df.head())

# --- backward compatibility ---
if "stat" in df.columns:
    df = df[df["stat"] == "mean"].copy()
else:
    logger.info("JSON-based summary detected (no 'stat' column).")

# --- detect available modes dynamically (compact logging) ---
modes = [m for m in ["source_only", "target_only", "joint_train"] if any(col.endswith(m) for col in df.columns)]
logger.info(f"Detected modes: {modes}")

# --- detect available distance/level combinations ---
distances = sorted(df["distance"].unique()) if "distance" in df.columns else ["unknown"]
levels    = sorted(df["level"].unique()) if "level" in df.columns else ["unknown"]
logger.info(f"Detected combinations: {[(d,l) for d in distances for l in levels]}")

# --- metrics to plot (requested order) ---
# AUPRC, AUC, Recall, F2, Precision, F1, Accuracy
metrics = ["auc_pr", "auc", "recall", "f2", "precision", "f1", "accuracy"]
models  = df["model"].unique().tolist() if "model" in df.columns else ["RF"]

# === BAR PLOT (Model × Metric × Mode) ==
fig, axes = plt.subplots(len(distances), len(metrics), figsize=(5*len(metrics), 3*len(distances)), squeeze=False)

for i, dist in enumerate(distances):
    for j, metric in enumerate(metrics):
        ax = axes[i, j]
        sub = df[df["distance"] == dist]
        if sub.empty:
            ax.axis("off")
            continue

        model = sub["model"].iloc[0] if "model" in sub.columns else "RF"
        # --- Enforce High → Middle → Low order ---
        ordered_levels = ["high", "middle", "low"]
        levels_present = [lvl for lvl in ordered_levels if lvl in sub["level"].unique()]

        x = np.arange(len(levels_present))
        width = 0.35

        vals_source, vals_target = [], []
        for lvl in levels_present:
            sub_lvl = sub[sub["level"] == lvl]
            col_s = f"{metric}_source_only"
            col_t = f"{metric}_target_only"

            vals_source.append(sub_lvl[col_s].mean() if col_s in sub_lvl.columns else np.nan)
            vals_target.append(sub_lvl[col_t].mean() if col_t in sub_lvl.columns else np.nan)

        bars_source = ax.bar(x - width/2, vals_source, width, label="Source-only", color="#6699cc")
        bars_target = ax.bar(x + width/2, vals_target, width, label="Target-only", color="#ff9966")

        ax.set_xticks(x)
        ax.set_xticklabels([lvl.capitalize() for lvl in levels_present])

        # --- Dynamic y-axis scaling and baseline line only for AUPRC ---
        if "auc_pr" in metric:
            valid = [v for v in (vals_source + vals_target) if not np.isnan(v)]
            if valid:
                ymin = min(valid)
                ymax = max(valid)
                margin = (ymax - ymin) * 0.3 if ymax > ymin else 0.02
                ax.set_ylim(max(0, ymin - margin), min(1.0, ymax + margin))
            else:
                ax.set_ylim(0, 1.0)

            # Baseline line (pos_rate)
            ax.axhline(BASELINE_POS_RATE, color='gray', linestyle='--', linewidth=1)
            ax.text(len(levels_present)-0.5, BASELINE_POS_RATE + 0.01,
                    f"Baseline ({BASELINE_POS_RATE:.3f})", fontsize=8, color='gray')
        else:
            ax.set_ylim(0, 1.0)

        if i == 0:
            title_map = {
                "auc": "AUROC",
                "auc_pr": "AUPRC",
                "accuracy": "Accuracy",
                "f1": "F1",
                "f2": "F2",
                "precision": "Precision (pos)",
                "recall": "Recall (pos)",
            }
            ax.set_title(title_map.get(metric, metric.upper()), fontsize=11)

        # Add a distance label only on the left-most subplot in each row
        if j == 0:
            pretty_dist = {"dtw": "DTW", "mmd": "MMD", "wasserstein": "Wasserstein"}
            dist_label = pretty_dist.get(str(dist).lower(), str(dist))
            ax.text(0.02, 0.95, dist_label, transform=ax.transAxes, ha="left", va="top",
                    fontsize=12, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6))

        # --- Add legend only once in top-right subplot ---
        if i == 0 and j == len(metrics) - 1:
            ax.legend(
                handles=[bars_source, bars_target],
                labels=["Source-only", "Target-only"],
                loc="upper right",
                fontsize=8,
                frameon=False,
            )

plt.tight_layout()
os.makedirs(os.path.dirname(OUT1), exist_ok=True)
plt.savefig(OUT1, dpi=200)
plt.close(fig)
logger.info(f"Saved bar plot → {OUT1}")

# === HEATMAP (Metric difference between modes) ===
comparisons = [
    ("source_only", "target_only", "Source - Target"),
]

fig2, axes2 = plt.subplots(len(metrics), len(comparisons),
                           figsize=(4*len(comparisons), 4*len(metrics)),
                           squeeze=False)

for i, metric in enumerate(metrics):
    for j, (a, b, title) in enumerate(comparisons):
        diffs, labels = [], []
        for _, r in df.iterrows():
            va, vb = r.get(f"{metric}_{a}", np.nan), r.get(f"{metric}_{b}", np.nan)
            diffs.append(va - vb if pd.notna(va) and pd.notna(vb) else np.nan)
            # label example: "mmd / high" など
            lbl_dist = str(r.get("distance", "unknown"))
            lbl_lvl  = str(r.get("level", "unknown"))
            labels.append(f"{lbl_dist}/{lbl_lvl}")
        mat = np.array(diffs).reshape(-1, 1)
        im = axes2[i, j].imshow(mat, cmap="coolwarm", vmin=-1, vmax=1)
        axes2[i, j].set_yticks(range(len(labels)))
        axes2[i, j].set_yticklabels(labels)
        axes2[i, j].set_xticks([])
        axes2[i, j].set_title(f"{metric.upper()} ({title})")
        for k, v in enumerate(diffs):
            if not np.isnan(v):
                axes2[i, j].text(0, k, f"{v:.2f}", ha="center", va="center", fontsize=9)

plt.tight_layout()
plt.savefig(OUT2, dpi=200)
plt.close(fig2)
logger.info(f"Saved heatmap → {OUT2}")

if not os.path.exists(OUT1) or not os.path.exists(OUT2):
    logger.warning("Some output plots were not generated as expected.")
else:
    logger.info("All output plots successfully generated.")

