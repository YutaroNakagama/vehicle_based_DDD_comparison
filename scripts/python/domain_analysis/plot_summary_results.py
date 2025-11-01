# misc/plot_summary_metrics_40.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

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

# --- Determine baseline positive rate dynamically ---
if "pos_rate" in df.columns and df["pos_rate"].notna().any():
    BASELINE_POS_RATE = df["pos_rate"].mean()
    logger.info(f"Using mean positive rate from JSONs (in summary): {BASELINE_POS_RATE:.4f}")
else:
    # try fallback: read from summary_40cases_test.csv
    base_csv = IN.replace("_mode_compare_with_levels.csv", ".csv")
    if os.path.exists(base_csv):
        df_base = pd.read_csv(base_csv)
        if "pos_rate" in df_base.columns and df_base["pos_rate"].notna().any():
            BASELINE_POS_RATE = df_base["pos_rate"].mean()
            logger.info(f"Retrieved baseline pos_rate from {os.path.basename(base_csv)}: {BASELINE_POS_RATE:.4f}")
        else:
            BASELINE_POS_RATE = 0.033
            logger.warning(f"pos_rate missing even in {base_csv}, using default baseline={BASELINE_POS_RATE:.4f}")
    else:
        BASELINE_POS_RATE = 0.033
        logger.warning(f"pos_rate not found, using default baseline={BASELINE_POS_RATE:.4f}")

# --- Normalize AUPRC by baseline ---
for mode in ["source_only", "target_only"]:
    col = f"auc_pr_{mode}"
    if col in df.columns:
        df[f"auc_pr_norm_{mode}"] = (df[col] - BASELINE_POS_RATE) / (1 - BASELINE_POS_RATE)
        logger.info(f"Normalized AUPRC for {mode} (0–1 scale).")

logger.debug("=== DEBUG: Loaded CSV columns ===")
logger.debug(df.columns.tolist())
logger.debug("=== First rows ===")
logger.debug(df.head())

# --- backward compatibility ---
if "stat" in df.columns:
    df = df[df["stat"] == "mean"].copy()
else:
    logger.info("JSON-based summary detected (no 'stat' column).")

# --- detect available modes dynamically ---
modes = []
for mode in ["source_only", "target_only", "joint_train"]:
    if any(col.endswith(mode) for col in df.columns):
        modes.append(mode)
logger.info(f"Detected modes: {modes}")

# --- detect available distance/level combinations ---
distances = sorted(df["distance"].unique()) if "distance" in df.columns else ["unknown"]
levels    = sorted(df["level"].unique()) if "level" in df.columns else ["unknown"]
logger.info(f"Detected combinations: {[(d,l) for d in distances for l in levels]}")

metrics = []
for c in df.columns:
    # capture e.g. "auc_pr", "auc_pr_norm", "f1", etc.
    if any(c.endswith(m) for m in ["source_only", "target_only", "joint_train"]):
        base = c.rsplit("_", 2)[0]  # keep full prefix before last two parts
        metrics.append(base)
metrics = sorted(set(metrics))

if "auc_pr_norm_source_only" in df.columns:
    logger.info("Using normalized AUPRC columns for visualization.")

# --- unify normalized and raw AUPRC for visualization ---
metrics_unified = []
for m in metrics:
    if m.startswith("auc_pr_norm"):
        if "auc_pr" not in metrics_unified:
            metrics_unified.append("auc_pr")
    else:
        metrics_unified.append(m)
metrics = metrics_unified
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
            # --- Prefer normalized AUPRC if available ---
            if "auc_pr" in metric and "auc_pr_norm_source_only" in df.columns:
                col_s = "auc_pr_norm_source_only"
                col_t = "auc_pr_norm_target_only"
            else:
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
            ymin = min(vals_source + vals_target)
            ymax = max(vals_source + vals_target)
            margin = (ymax - ymin) * 0.3 if ymax > ymin else 0.02
            ax.set_ylim(max(0, ymin - margin), min(1.0, ymax + margin))

            # Baseline line (pos_rate)
            ax.axhline(BASELINE_POS_RATE, color='gray', linestyle='--', linewidth=1)
            ax.text(len(levels_present)-0.5, BASELINE_POS_RATE + 0.01,
                    "Baseline", fontsize=8, color='gray')
        else:
            ax.set_ylim(0, 1.0)

        if i == 0:
            title = metric.upper()
            if metric == "auc_pr":
                title += " (normalized)"
            ax.set_title(title, fontsize=11)
        if j == 0:
            ax.set_ylabel(dist.upper(), fontsize=10)

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
plt.close(fig2)
logger.info(f"Saved heatmap → {OUT2}")

if not os.path.exists(OUT1) or not os.path.exists(OUT2):
    logger.warning("Some output plots were not generated as expected.")
else:
    logger.info("All output plots successfully generated.")

