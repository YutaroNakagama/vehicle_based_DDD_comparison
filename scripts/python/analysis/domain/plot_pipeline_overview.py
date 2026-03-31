#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_pipeline_overview.py
=========================
Generate a system overview / pipeline diagram (Fig. 1) for the journal paper.
Matches IEEE T-IV formatting (double-column width, 8pt serif, SVG output).

Usage:
    python scripts/python/analysis/domain/plot_pipeline_overview.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = (
    PROJECT_ROOT / "results" / "analysis" / "exp2_domain_shift"
    / "figures" / "svg" / "split2" / "journal_v2"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# IEEE T-IV journal style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

# Colour palette (Okabe-Ito-aligned pastels, colorblind-safe)
COLORS = {
    "data":     "#CCE5FF",  # light blue
    "feature":  "#FFF2CC",  # light yellow
    "domain":   "#D5F5E3",  # light green
    "factorial": "#E8DAEF", # light purple
    "classify": "#FADBD8",  # light pink
    "analyse":  "#FCF3CF",  # light gold
}
BORDER = {
    "data":     "#0072B2",  # Okabe-Ito blue
    "feature":  "#E69F00",  # Okabe-Ito orange
    "domain":   "#009E73",  # Okabe-Ito bluish green
    "factorial": "#CC79A7", # Okabe-Ito reddish purple
    "classify": "#D55E00",  # Okabe-Ito vermilion
    "analyse":  "#F0E442",  # Okabe-Ito yellow
}


def draw_pipeline():
    fig, ax = plt.subplots(figsize=(7.16, 2.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.4)
    ax.axis("off")

    # ----- box specifications: (x, y, w, h, key, title, details) -----
    boxes = [
        (0.05, 0.6, 1.35, 1.5, "data",
         "Data Acquisition",
         ["Driving simulator", r"$f_s = 60$ Hz", "87 subjects",
          r"5 signals: $\delta, \dot{\delta},$",
          r"$a_y, a_x, e_{\mathrm{lane}}$"]),
        (1.70, 0.6, 1.35, 1.5, "feature",
         "Feature Engineering",
         ["3 s window, 50% overlap",
          r"135 features $\rightarrow$ top 10",
          "Spectral, PE,",
          "smooth/std/PE, CWT"]),
        (3.35, 0.6, 1.35, 1.5, "domain",
         "Domain Grouping",
         ["MMD / DTW / Wass.",
          r"KNN score ($K{=}5$)",
          "Median split:",
          "44 in / 43 out"]),
        (5.00, 0.6, 1.55, 1.5, "factorial",
         "Factorial Design",
         [r"$R$ (7) $\times$ $M$ (3)",
          r"$\times$ $D$ (3) $\times$ $G$ (2)",
          r"= 126 cells $\times$ 12 seeds",
          "= 1,512 observations"]),
        (6.85, 0.6, 1.35, 1.5, "classify",
         "Classification",
         ["RF + Optuna (100 TPE)",
          "3-fold CV, F2 optim.",
          "Threshold tuning",
          "Platt calibration"]),
        (8.50, 0.6, 1.40, 1.5, "analyse",
         "Sensitivity Analysis",
         [r"OFAT spaghetti $\rightarrow$",
          r"Sobol\u2013Hoeffding",
          "decomposition",
          r"$S_i$, $S_{Ti}$, bootstrap CI"]),
    ]

    for x, y, w, h, key, title, details in boxes:
        # Box
        fancy = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.05",
            facecolor=COLORS[key],
            edgecolor=BORDER[key],
            linewidth=1.2,
        )
        ax.add_patch(fancy)
        # Title
        ax.text(
            x + w / 2, y + h - 0.18, title,
            ha="center", va="top",
            fontsize=7.5, fontweight="bold",
            color=BORDER[key],
        )
        # Details
        for i, line in enumerate(details):
            ax.text(
                x + w / 2, y + h - 0.43 - i * 0.22, line,
                ha="center", va="top",
                fontsize=6.5, color="#333333",
            )

    # ----- Arrows between boxes -----
    arrow_style = "Simple,tail_width=2.5,head_width=7,head_length=5"
    arrow_kw = dict(
        arrowstyle=arrow_style,
        color="#555555",
        connectionstyle="arc3,rad=0",
    )
    arrow_xs = [
        (1.40, 1.70),
        (3.05, 3.35),
        (4.70, 5.00),
        (6.55, 6.85),
        (8.20, 8.50),
    ]
    for x_start, x_end in arrow_xs:
        arrow = FancyArrowPatch(
            (x_start, 1.35), (x_end, 1.35),
            **arrow_kw,
        )
        ax.add_patch(arrow)

    fig.savefig(
        OUT_DIR / "fig1_pipeline_overview.pdf",
        format="pdf", bbox_inches="tight", pad_inches=0.02,
    )
    print(f"Saved: {OUT_DIR / 'fig1_pipeline_overview.pdf'}")
    plt.close(fig)


if __name__ == "__main__":
    draw_pipeline()
