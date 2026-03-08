#!/usr/bin/env python3
"""Generate summary ROC curves per condition, aggregated across seeds.

For each condition (baseline, smote_plain, undersample_rus, sw_smote) and
ratio (r01, r05 — baseline has no ratio), produces a figure with 6 subplots
(3 modes × 2 levels) showing:
  - Individual seed ROC curves (thin, semi-transparent)
  - Mean ROC curve (bold)
  - ±1 std band (shaded)
  - Mean AUC in the legend

Output paths:
  results/analysis/exp2_domain_shift/figures/png/split2/{cond_dir}/{prefix}_{ratio}_summary_roc.png
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Reuse filename parser from the collector
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_split2_rf_metrics import parse_eval_filename

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[4]
EVAL_DIR = PROJECT_ROOT / "results" / "outputs" / "evaluation" / "RF"
PNG_BASE = (
    PROJECT_ROOT
    / "results"
    / "analysis"
    / "exp2_domain_shift"
    / "figures"
    / "png"
    / "split2"
)

# Condition directory & prefix mapping
CONDITION_MAP = {
    "baseline_domain": ("baseline", "baseline"),
    "smote_plain": ("smote_plain", "smote"),
    "undersample_rus": ("undersample_rus", "rus"),
    "sw_smote": ("sw_smote", "swsmote"),
}

MODES = ["source_only", "target_only", "mixed"]
DISTANCES = ["mmd", "dtw", "wasserstein"]
LEVELS = ["in_domain", "out_domain"]

# Common FPR grid for interpolation
COMMON_FPR = np.linspace(0.0, 1.0, 200)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------
def collect_roc_data() -> dict:
    """Scan eval JSONs and extract ROC curves.

    Returns a nested dict:
        {condition: {ratio: {(mode, distance, level): [(fpr, tpr, auc, seed), ...]}}}
    """
    data: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for json_path in sorted(EVAL_DIR.rglob("eval_results_RF_*knn*.json")):
        meta = parse_eval_filename(json_path.name)
        if meta is None:
            continue

        with open(json_path) as f:
            d = json.load(f)

        roc = d.get("roc_curve")
        if roc is None:
            continue

        fpr = np.array(roc["fpr"])
        tpr = np.array(roc["tpr"])
        auc_val = float(roc.get("auc", np.nan))
        seed = int(meta["seed"])
        condition = meta["condition"]
        ratio = meta.get("ratio", "") or ""
        mode = meta["mode"]
        distance = meta["distance"]
        level = meta["domain"]

        key = (mode, distance, level)
        data[condition][ratio][key].append((fpr, tpr, auc_val, seed))

    return data


# ---------------------------------------------------------------------------
# De-duplicate: keep latest (highest job_id) per seed per cell
# ---------------------------------------------------------------------------
def _dedup_by_seed(entries: list) -> list:
    """Keep latest entry per seed (entries added in job_id order)."""
    seen: dict = {}
    for fpr, tpr, auc_val, seed in entries:
        seen[seed] = (fpr, tpr, auc_val, seed)
    return list(seen.values())


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _ratio_tag(ratio: str) -> str:
    """'0.1' → 'r01', '0.5' → 'r05', '' → ''."""
    if not ratio:
        return ""
    return "r" + ratio.replace(".", "")


MODE_LABELS = {
    "source_only": "Cross-domain",
    "target_only": "Within-domain",
    "mixed": "Mixed",
}


def plot_summary_roc(
    condition: str,
    ratio: str,
    cell_data: dict,
    out_path: Path,
) -> None:
    """Create a 2×3 subplot figure (levels × modes) with summary ROC."""
    fig, axes = plt.subplots(
        len(LEVELS),
        len(MODES),
        figsize=(16, 9),
        constrained_layout=True,
    )

    ratio_label = _ratio_tag(ratio)
    cond_label = CONDITION_MAP.get(condition, (condition, condition))[1]
    title_ratio = f" (ratio={ratio})" if ratio else ""
    fig.suptitle(
        f"Summary ROC — {cond_label}{title_ratio}",
        fontsize=15,
        fontweight="bold",
    )

    for row_idx, level in enumerate(LEVELS):
        for col_idx, mode in enumerate(MODES):
            ax = axes[row_idx, col_idx]
            mode_lbl = MODE_LABELS.get(mode, mode)
            ax.set_title(f"{mode_lbl} / {level}", fontsize=11)
            ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4, label="Chance")

            for dist_idx, distance in enumerate(DISTANCES):
                key = (mode, distance, level)
                entries = cell_data.get(key, [])
                entries = _dedup_by_seed(entries)

                if not entries:
                    continue

                # Interpolate all curves to COMMON_FPR
                interp_tprs = []
                aucs = []
                for fpr, tpr, auc_val, seed in entries:
                    interp_tpr = np.interp(COMMON_FPR, fpr, tpr)
                    interp_tpr[0] = 0.0
                    interp_tprs.append(interp_tpr)
                    aucs.append(auc_val)

                interp_tprs = np.array(interp_tprs)
                mean_tpr = interp_tprs.mean(axis=0)
                std_tpr = interp_tprs.std(axis=0)
                mean_auc = np.mean(aucs)
                n_seeds = len(entries)

                colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
                color = colors[dist_idx % len(colors)]

                # Individual seed curves (thin)
                for i, (fpr_i, tpr_i, _, _) in enumerate(entries):
                    ax.plot(fpr_i, tpr_i, color=color, alpha=0.12, lw=0.5)

                # Mean curve
                ax.plot(
                    COMMON_FPR,
                    mean_tpr,
                    color=color,
                    lw=2.0,
                    label=f"{distance} (AUC={mean_auc:.3f}, n={n_seeds})",
                )

                # ±1 std band
                ax.fill_between(
                    COMMON_FPR,
                    np.clip(mean_tpr - std_tpr, 0, 1),
                    np.clip(mean_tpr + std_tpr, 0, 1),
                    color=color,
                    alpha=0.15,
                )

            ax.set_xlim([-0.02, 1.02])
            ax.set_ylim([-0.02, 1.02])
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            ax.legend(fontsize=8, loc="lower right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    logger.info("Collecting ROC data from eval JSONs...")
    data = collect_roc_data()

    total_files = 0
    for condition, cond_label in [
        ("baseline_domain", "baseline"),
        ("smote_plain", "smote"),
        ("undersample_rus", "rus"),
        ("sw_smote", "swsmote"),
    ]:
        cond_dir, prefix = CONDITION_MAP[condition]
        ratios = sorted(data.get(condition, {}).keys())
        if not ratios:
            logger.warning(f"No data for {condition}")
            continue

        logger.info(f"{condition}: ratios={ratios}")

        for ratio in ratios:
            cell_data = data[condition][ratio]
            ratio_tag = _ratio_tag(ratio)

            if ratio_tag:
                fname = f"{prefix}_{ratio_tag}_summary_roc.png"
            else:
                fname = f"{prefix}_summary_roc.png"

            out_path = PNG_BASE / cond_dir / fname
            plot_summary_roc(condition, ratio, cell_data, out_path)
            total_files += 1

    logger.info(f"Done. Generated {total_files} summary ROC plots.")


if __name__ == "__main__":
    main()
