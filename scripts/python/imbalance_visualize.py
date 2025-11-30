#!/usr/bin/env python
"""CLI script for visualizing imbalance comparison results.

This script generates visualizations for imbalanced data experiments,
comparing various sampling methods and ensemble approaches using
appropriate metrics (AUPRC, F2, Recall).

Usage
-----
Basic usage (use default V2 model configurations):
    $ python scripts/python/imbalance_visualize.py

Custom output directory:
    $ python scripts/python/imbalance_visualize.py --output-dir results/my_analysis

Include ensemble results:
    $ python scripts/python/imbalance_visualize.py --ensemble-dir results/evaluation/ensemble

Custom model specifications:
    $ python scripts/python/imbalance_visualize.py \
        --models "RF:14468417:Baseline RF:imbal_v2_baseline" \
        --models "RF:14468421:SMOTE+RUS:imbal_v2_smote_rus"

Examples
--------
# Generate all visualizations with default V2 configurations
python scripts/python/imbalance_visualize.py

# Only generate specific plots
python scripts/python/imbalance_visualize.py --plots auprc f2 dashboard
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple, Optional

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.analysis.imbalance_analysis import (
    load_imbalance_results,
    load_ensemble_results,
    plot_auprc_comparison,
    plot_f2_comparison,
    plot_recall_precision_tradeoff,
    plot_metrics_heatmap,
    plot_summary_dashboard,
    generate_imbalance_report,
)
from src.utils.visualization.visualization import save_figure

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Default V2 model configurations
DEFAULT_V2_MODELS = [
    ("RF", "14468417", "Baseline RF", "imbal_v2_baseline"),
    ("RF", "14468418", "SMOTE+Tomek", "imbal_v2_smote_tomek"),
    ("RF", "14468419", "SMOTE+ENN", "imbal_v2_smote_enn"),
    ("RF", "14468420", "BalancedRF", "imbal_v2_balanced_rf"),
    ("RF", "14468421", "SMOTE+RUS", "imbal_v2_smote_rus"),
    ("RF", "14468501", "EasyEnsemble", "imbal_v2_easyensemble"),
]


def parse_model_spec(spec: str) -> Tuple[str, str, str, str]:
    """Parse model specification string.
    
    Format: "model_type:jobid:display_name:tag"
    Example: "RF:14468417:Baseline RF:imbal_v2_baseline"
    """
    parts = spec.split(':')
    if len(parts) != 4:
        raise ValueError(f"Invalid model spec: {spec}. Expected format: model_type:jobid:display_name:tag")
    return tuple(parts)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate visualizations for imbalance comparison experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results/evaluation',
        help='Base directory for evaluation results (default: results/evaluation)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/imbalance_analysis',
        help='Output directory for generated figures (default: results/imbalance_analysis)'
    )
    
    parser.add_argument(
        '--ensemble-dir',
        type=str,
        default=None,
        help='Directory containing ensemble evaluation results (optional)'
    )
    
    parser.add_argument(
        '--models',
        type=str,
        action='append',
        default=None,
        help='Model specification in format "model_type:jobid:display_name:tag". '
             'Can be specified multiple times. If not provided, uses default V2 models.'
    )
    
    parser.add_argument(
        '--baseline-auprc',
        type=float,
        default=0.039,
        help='Random baseline AUPRC (typically equals positive class rate). Default: 0.039'
    )
    
    parser.add_argument(
        '--plots',
        type=str,
        nargs='+',
        choices=['auprc', 'f2', 'tradeoff', 'heatmap', 'dashboard', 'all'],
        default=['all'],
        help='Which plots to generate (default: all)'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='DPI for saved figures (default: 150)'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['png', 'pdf', 'svg'],
        default='png',
        help='Output format for figures (default: png)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse model specifications
    if args.models:
        models_info = [parse_model_spec(spec) for spec in args.models]
    else:
        models_info = DEFAULT_V2_MODELS
    
    logger.info(f"Loading results for {len(models_info)} models...")
    
    # Load single model results
    df = load_imbalance_results(args.results_dir, models_info)
    
    if df.empty:
        logger.error("No results loaded. Please check the results directory and model specifications.")
        sys.exit(1)
    
    logger.info(f"Loaded results for {len(df)} models:")
    for _, row in df.iterrows():
        logger.info(f"  - {row['name']}: AUPRC={row['auprc']:.4f}, F2={row['f2']:.4f}")
    
    # Load ensemble results if provided
    ensemble_df = None
    if args.ensemble_dir:
        ensemble_df = load_ensemble_results(args.ensemble_dir)
        if not ensemble_df.empty:
            logger.info(f"Loaded {len(ensemble_df)} ensemble results")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which plots to generate
    plots = set(args.plots)
    if 'all' in plots:
        plots = {'auprc', 'f2', 'tradeoff', 'heatmap', 'dashboard'}
    
    logger.info(f"Generating plots: {', '.join(plots)}")
    saved_files = {}
    
    # Generate plots
    if 'auprc' in plots:
        logger.info("Generating AUPRC comparison plot...")
        fig = plot_auprc_comparison(df, baseline_auprc=args.baseline_auprc)
        path = output_dir / f"auprc_comparison.{args.format}"
        save_figure(fig, path, dpi=args.dpi)
        plt.close(fig)
        saved_files['auprc'] = str(path)
        logger.info(f"  Saved: {path}")
    
    if 'f2' in plots:
        logger.info("Generating F2 comparison plot...")
        fig = plot_f2_comparison(df)
        path = output_dir / f"f2_comparison.{args.format}"
        save_figure(fig, path, dpi=args.dpi)
        plt.close(fig)
        saved_files['f2'] = str(path)
        logger.info(f"  Saved: {path}")
    
    if 'tradeoff' in plots:
        logger.info("Generating Recall-Precision trade-off plot...")
        fig = plot_recall_precision_tradeoff(df)
        path = output_dir / f"recall_precision_tradeoff.{args.format}"
        save_figure(fig, path, dpi=args.dpi)
        plt.close(fig)
        saved_files['tradeoff'] = str(path)
        logger.info(f"  Saved: {path}")
    
    if 'heatmap' in plots:
        logger.info("Generating metrics heatmap...")
        fig = plot_metrics_heatmap(df)
        path = output_dir / f"metrics_heatmap.{args.format}"
        save_figure(fig, path, dpi=args.dpi)
        plt.close(fig)
        saved_files['heatmap'] = str(path)
        logger.info(f"  Saved: {path}")
    
    if 'dashboard' in plots:
        logger.info("Generating summary dashboard...")
        fig = plot_summary_dashboard(df, ensemble_df, baseline_auprc=args.baseline_auprc)
        path = output_dir / f"summary_dashboard.{args.format}"
        save_figure(fig, path, dpi=args.dpi)
        plt.close(fig)
        saved_files['dashboard'] = str(path)
        logger.info(f"  Saved: {path}")
    
    # Save summary CSV
    csv_path = output_dir / "summary_metrics.csv"
    df.to_csv(csv_path, index=False)
    saved_files['csv'] = str(csv_path)
    logger.info(f"Saved summary CSV: {csv_path}")
    
    # Save ensemble CSV if available
    if ensemble_df is not None and not ensemble_df.empty:
        csv_path = output_dir / "ensemble_metrics.csv"
        ensemble_df.to_csv(csv_path, index=False)
        saved_files['ensemble_csv'] = str(csv_path)
        logger.info(f"Saved ensemble CSV: {csv_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nGenerated files:")
    for name, path in saved_files.items():
        print(f"  - {name}: {path}")
    
    # Print brief results summary
    print("\n" + "-"*60)
    print("RESULTS SUMMARY (sorted by AUPRC)")
    print("-"*60)
    df_sorted = df.sort_values('auprc', ascending=False)
    for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
        improvement = (row['auprc'] / args.baseline_auprc - 1) * 100
        f2_str = f"{row['f2_thr']:.3f}" if not row['f2_thr'] != row['f2_thr'] else f"{row['f2']:.3f}"
        print(f"  {i}. {row['name']:<15} | AUPRC: {row['auprc']:.4f} ({improvement:+.1f}%) | F2: {f2_str}")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    main()
