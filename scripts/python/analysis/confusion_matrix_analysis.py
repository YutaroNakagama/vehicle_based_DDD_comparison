#!/usr/bin/env python3
"""
confusion_matrix_analysis.py - CLI Wrapper
===========================================

Thin CLI wrapper for confusion matrix analysis and visualization.
All business logic is delegated to src/analysis/confusion_matrix.py

Usage:
    python confusion_matrix_analysis.py plot       # Generate PNG heatmaps
    python confusion_matrix_analysis.py table      # Generate console output + CSV
    python confusion_matrix_analysis.py aggregate  # Multi-seed aggregation analysis
    python confusion_matrix_analysis.py all        # Run all modes
"""

import argparse
from pathlib import Path

from src.analysis.confusion_matrix import (
    # Data loading
    collect_eval_data,
    # Visualization
    generate_distance_plot,
    generate_overview_plot,
    # Tables
    generate_summary_table,
    print_detailed_tables,
    # Multi-seed aggregation
    extract_multiseed_results,
    aggregate_multiseed_results,
    generate_rates_visualization,
    create_aggregate_summary,
    # Constants
    DISTANCES,
)

# Configuration paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EVAL_DIR = PROJECT_ROOT / "results/outputs/evaluation/RF/14357179"
OUTPUT_DIR_PNG = PROJECT_ROOT / "results/analysis/domain/summary/png/confusion_matrices"
OUTPUT_DIR_CSV = PROJECT_ROOT / "results/analysis/domain/summary/csv"
OUTPUT_DIR_MULTISEED = PROJECT_ROOT / "results/analysis/imbalance/multiseed"


def run_plot_mode(eval_dir: Path) -> None:
    """Generate PNG heatmap visualizations."""
    print("\n=== Mode: plot (PNG heatmaps) ===")
    
    data_dict = collect_eval_data(eval_dir)
    OUTPUT_DIR_PNG.mkdir(parents=True, exist_ok=True)
    
    # Plot for each distance metric
    for distance in DISTANCES:
        output_file = OUTPUT_DIR_PNG / f'confusion_matrices_{distance}.png'
        generate_distance_plot(data_dict, distance, output_file)
    
    # Combined overview
    output_file = OUTPUT_DIR_PNG / 'confusion_matrices_pooled_overview.png'
    generate_overview_plot(data_dict, output_file, mode='pooled')
    
    print(f"\n✓ All confusion matrices saved to {OUTPUT_DIR_PNG}")


def run_table_mode(eval_dir: Path) -> None:
    """Generate console output and CSV."""
    print("\n=== Mode: table (console + CSV) ===")
    
    data_dict = collect_eval_data(eval_dir)
    OUTPUT_DIR_CSV.mkdir(parents=True, exist_ok=True)
    
    # Print detailed tables
    print_detailed_tables(data_dict)
    
    # Summary table
    print("\n" + "=" * 120)
    print("SUMMARY: All Cases")
    print("=" * 120)
    
    summary_df = generate_summary_table(data_dict)
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
        
        output_file = OUTPUT_DIR_CSV / 'confusion_matrices_all_cases.csv'
        summary_df.to_csv(output_file, index=False)
        print(f"\n✓ Saved to: {output_file}")


def run_aggregate_mode(eval_dir: Path) -> None:
    """Run multi-seed aggregation analysis."""
    print("\n=== Mode: aggregate (multi-seed analysis) ===")
    
    OUTPUT_DIR_MULTISEED.mkdir(parents=True, exist_ok=True)
    
    print("Extracting evaluation results...")
    df = extract_multiseed_results(eval_dir)
    
    if len(df) == 0:
        print("No multi-seed results found.")
        return
    
    print(f"Found {len(df)} evaluation records")
    print(f"Seeds: {sorted(df['seed'].unique())}")
    print(f"Methods: {df['method_label'].nunique()} unique")
    
    # Aggregate results
    agg_df = aggregate_multiseed_results(df)
    
    # Create and save summary
    summary_df = create_aggregate_summary(agg_df)
    csv_path = OUTPUT_DIR_MULTISEED / "confusion_matrix_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")
    
    # Print summary
    print("\n" + "=" * 120)
    print("Confusion Matrix Summary (Mean ± Std across seeds)")
    print("=" * 120)
    print(summary_df.to_string(index=False))
    
    # Create rates visualization
    print("\nCreating confusion matrix rates visualization...")
    output_path = OUTPUT_DIR_MULTISEED / "confusion_matrix_rates.png"
    generate_rates_visualization(agg_df, output_path)
    
    print(f"\n✓ All outputs saved to {OUTPUT_DIR_MULTISEED}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified confusion matrix analysis and visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python confusion_matrix_analysis.py plot
    python confusion_matrix_analysis.py table --eval-dir results/evaluation/RF/14357179
    python confusion_matrix_analysis.py aggregate
    python confusion_matrix_analysis.py all
        """
    )
    parser.add_argument(
        "mode",
        choices=["plot", "table", "aggregate", "all"],
        help="Analysis mode: plot (PNG), table (CSV), aggregate (multi-seed), or all"
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=None,
        help=f"Evaluation directory (default: {DEFAULT_EVAL_DIR})"
    )
    args = parser.parse_args()
    
    # Determine eval directory
    if args.eval_dir:
        eval_dir = args.eval_dir
    elif args.mode == "aggregate":
        eval_dir = PROJECT_ROOT / "results/evaluation"
    else:
        eval_dir = DEFAULT_EVAL_DIR
    
    print("=" * 80)
    print(f"CONFUSION MATRIX ANALYSIS (mode={args.mode})")
    print("=" * 80)
    print(f"Evaluation directory: {eval_dir}")
    
    if args.mode == "plot":
        run_plot_mode(eval_dir)
    elif args.mode == "table":
        run_table_mode(eval_dir)
    elif args.mode == "aggregate":
        run_aggregate_mode(eval_dir)
    elif args.mode == "all":
        run_plot_mode(eval_dir)
        run_table_mode(eval_dir)
        run_aggregate_mode(PROJECT_ROOT / "results/evaluation")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
