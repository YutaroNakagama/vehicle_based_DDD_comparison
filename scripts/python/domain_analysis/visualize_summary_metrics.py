# misc/plot_summary_metrics_40.py
import os
import logging

import matplotlib as mpl
mpl.set_loglevel("warning")  # Suppress overall Matplotlib logs
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # Stop findfont spam

from src import config as cfg
from src.utils.io.data_io import load_csv
from src.utils.visualization.visualization import (
    plot_grouped_bar_chart,
    plot_metric_difference_heatmap,
    save_figure
)

# Initialize module logger (used below)
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


IN   = os.path.join(cfg.RESULTS_DOMAIN_ANALYSIS_PATH, "summary", "csv", "summary_40cases_test_mode_compare_with_levels.csv")
OUT1 = os.path.join(cfg.RESULTS_DOMAIN_ANALYSIS_PATH, "summary", "png", "summary_metrics_bar.png")
OUT2 = os.path.join(cfg.RESULTS_DOMAIN_ANALYSIS_PATH, "summary", "png", "summary_diff_heatmap.png")

# --- fallback (for backward compatibility) ---
if not os.path.exists(IN):
    alt_path = IN.replace("_with_levels", "")
    if os.path.exists(alt_path):
        logger.warning(f"Fallback to: {alt_path}")
        IN = alt_path

df = load_csv(IN)

# --- Determine baseline positive rate dynamically (for AUPRC baseline line only) ---
base_csv = IN.replace("_mode_compare_with_levels.csv", ".csv")
if os.path.exists(base_csv):
    df_base = load_csv(base_csv)
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

# === BAR PLOT (Model × Metric × Mode) ==
fig = plot_grouped_bar_chart(
    data=df,
    metrics=metrics,
    modes=modes,
    distance_col="distance",
    level_col="level",
    baseline_rates={"auc_pr": BASELINE_POS_RATE}
)

save_figure(fig, OUT1, dpi=200)
logger.info(f"Saved bar plot → {OUT1}")

# === HEATMAP (Metric difference between modes) ===
comparisons = [
    ("source_only", "target_only", "Source - Target"),
]

fig2 = plot_metric_difference_heatmap(
    data=df,
    metrics=metrics,
    comparisons=comparisons,
    distance_col="distance",
    level_col="level"
)

save_figure(fig2, OUT2, dpi=200)
logger.info(f"Saved heatmap → {OUT2}")

if not os.path.exists(OUT1) or not os.path.exists(OUT2):
    logger.warning("Some output plots were not generated as expected.")
else:
    logger.info("All output plots successfully generated.")
