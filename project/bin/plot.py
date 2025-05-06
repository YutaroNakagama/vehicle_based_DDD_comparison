"""Plot ROC curves for the latest evaluation results across models.

Usage:
    $ python plot_roc.py --results_dir results/common
"""

import sys
import os
import argparse
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.visualization.visualization import plot_roc_curves_from_latest_json

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="Plot ROC curves from latest evaluation metrics.")
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing metrics_*.json result files."
    )
    parser.add_argument(
        "--title",
        type=str,
        default="ROC Curve Comparison",
        help="Title for the ROC plot"
    )

    args = parser.parse_args()

    logging.info(f"Plotting ROC curves from: {args.results_dir}")
    plot_roc_curves_from_latest_json(args.results_dir, title=args.title)


if __name__ == "__main__":
    main()

