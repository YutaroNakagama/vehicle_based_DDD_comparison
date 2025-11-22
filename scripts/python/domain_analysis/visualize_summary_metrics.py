# misc/plot_summary_metrics_40.py
import os
import logging

import matplotlib as mpl
mpl.set_loglevel("warning")  # Suppress overall Matplotlib logs
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # Stop findfont spam

from src import config as cfg
from src.utils.io.data_io import load_csv
from src.utils.visualization.visualization import (
    plot_grouped_bar_chart_raw,
    save_figure
)

# Initialize module logger (used below)
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


IN   = os.path.join(cfg.RESULTS_DOMAIN_ANALYSIS_PATH, "summary", "csv", "summary_40cases_test.csv")
OUT1 = os.path.join(cfg.RESULTS_DOMAIN_ANALYSIS_PATH, "summary", "png", "summary_metrics_bar.png")

df = load_csv(IN)

# --- Determine baseline positive rate dynamically (for AUPRC baseline line only) ---
if "pos_rate" in df.columns and df["pos_rate"].notna().any():
    BASELINE_POS_RATE = df["pos_rate"].mean()
    logger.info(f"Using mean pos_rate: {BASELINE_POS_RATE:.4f}")
else:
    BASELINE_POS_RATE = 0.033
    logger.warning(f"pos_rate missing, using default baseline={BASELINE_POS_RATE:.4f}")

logger.debug("=== DEBUG: Loaded CSV columns ===")
logger.debug(df.columns.tolist())
logger.debug("=== First rows ===")
logger.debug(df.head())

# --- detect available modes dynamically ---
modes = ["pooled", "source_only", "target_only"]
logger.info(f"Processing modes: {modes}")

# --- detect available distance/level combinations ---
distances = sorted(df["distance"].unique()) if "distance" in df.columns else ["unknown"]
levels    = sorted(df["level"].unique()) if "level" in df.columns else ["unknown"]
logger.info(f"Detected combinations: {[(d,l) for d in distances for l in levels]}")

# --- metrics to plot (requested order) ---
# AUPRC, AUC, Recall, F2, Precision, F1, Accuracy
metrics = ["auc_pr", "auc", "recall", "f2", "precision", "f1", "accuracy"]

# === BAR PLOT (Model × Metric × Mode) ==
fig = plot_grouped_bar_chart_raw(
    data=df,
    metrics=metrics,
    modes=modes,
    distance_col="distance",
    level_col="level",
    baseline_rates={"auc_pr": BASELINE_POS_RATE}
)

save_figure(fig, OUT1, dpi=200)
logger.info(f"Saved bar plot → {OUT1}")

if not os.path.exists(OUT1):
    logger.warning("Bar plot was not generated as expected.")
else:
    logger.info("Bar plot successfully generated.")
