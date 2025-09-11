"""Plot ROC curves for the latest evaluation results across models.

This script loads the most recent evaluation result files (metrics_*.json)
from a given results directory and plots ROC curves for comparison.

Examples
--------
Plot ROC curves from results in ``results/common``:

    $ python plot.py --results_dir results/common
"""

import sys
import os
import argparse
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.visualization.visualization import plot_roc_curves_from_latest_json

logging.basicConfig(level=logging.INFO)


def main():
    """Parse CLI arguments and plot ROC curves from the latest evaluation results.

    Parameters
    ----------
    None

    Other Parameters
    ----------------
    --results_dir : str
        Required. Path to the directory containing ``metrics_*.json`` files.
    --title : str, default="ROC Curve Comparison"
        Title for the generated ROC curve plot.

    Returns
    -------
    None

    Raises
    ------
    SystemExit
        If required arguments are missing or invalid.
    """
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

