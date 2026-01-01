#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_imbalance.py
====================

Unified tool for imbalance experiment analysis.

This script provides:
1. v3          - Imbalance v3 analysis (replaces analyze_imbalance_v3)
2. collect     - Collect imbalance metrics (replaces collect_imbalance_v3_metrics)
3. multiseed   - Multi-seed analysis (replaces multiseed_analysis)
4. optuna      - Optuna convergence analysis (replaces optuna_analysis)
5. statistical - Statistical tests (replaces statistical_tests_unified)
6. scores      - Subject scores computation (replaces compute_subject_scores)

Consolidates functionality from:
- analyze_imbalance_v3.py
- collect_imbalance_v3_metrics.py
- multiseed_analysis.py
- optuna_analysis.py
- statistical_tests_unified.py
- compute_subject_scores.py

Usage:
    python analyze_imbalance.py v3 --source directory
    python analyze_imbalance.py collect --output results/...
    python analyze_imbalance.py multiseed aggregate
    python analyze_imbalance.py optuna visualize
    python analyze_imbalance.py statistical per-subject
    python analyze_imbalance.py scores --input results/...
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# Delegated Commands (thin wrappers to existing scripts)
# ============================================================
def cmd_v3(args) -> int:
    """Run imbalance v3 analysis."""
    from analysis.analyze_imbalance_v3 import main as v3_main
    sys.argv = ["analyze_imbalance_v3.py", "--source", args.source]
    if args.ranked:
        sys.argv.append("--ranked")
    return v3_main()


def cmd_collect(args) -> int:
    """Collect imbalance v3 metrics."""
    from analysis.imbalance.collect_imbalance_v3_metrics import main as collect_main
    return collect_main()


def cmd_multiseed(args) -> int:
    """Run multi-seed analysis."""
    from analysis.imbalance.multiseed_analysis import main as multiseed_main
    # Reconstruct argv
    sys.argv = ["multiseed_analysis.py", args.subcommand]
    if hasattr(args, "job_prefixes") and args.job_prefixes:
        sys.argv.extend(["--job-prefixes", args.job_prefixes])
    if hasattr(args, "baseline") and args.baseline:
        sys.argv.extend(["--baseline", args.baseline])
    return multiseed_main()


def cmd_optuna(args) -> int:
    """Run Optuna analysis."""
    from analysis.imbalance.optuna_analysis import main as optuna_main
    sys.argv = ["optuna_analysis.py", args.subcommand]
    if hasattr(args, "method") and args.method:
        sys.argv.extend(["--method", args.method])
    if hasattr(args, "log_dir") and args.log_dir:
        sys.argv.extend(["--log-dir", args.log_dir])
    return optuna_main()


def cmd_statistical(args) -> int:
    """Run statistical tests."""
    from analysis.imbalance.statistical_tests_unified import main as stat_main
    sys.argv = ["statistical_tests_unified.py", args.subcommand]
    return stat_main()


def cmd_scores(args) -> int:
    """Compute subject scores."""
    from analysis.imbalance.compute_subject_scores import main as scores_main
    return scores_main()


def main():
    parser = argparse.ArgumentParser(
        description="Unified imbalance analysis tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python analyze_imbalance.py v3 --source directory
    python analyze_imbalance.py multiseed aggregate
    python analyze_imbalance.py optuna visualize
    python analyze_imbalance.py statistical per-subject
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # v3
    p_v3 = subparsers.add_parser("v3", help="Imbalance v3 analysis")
    p_v3.add_argument("--source", default="directory", choices=["directory", "pbs_logs"])
    p_v3.add_argument("--ranked", action="store_true")
    p_v3.set_defaults(func=cmd_v3)
    
    # collect
    p_collect = subparsers.add_parser("collect", help="Collect imbalance metrics")
    p_collect.set_defaults(func=cmd_collect)
    
    # multiseed
    p_multi = subparsers.add_parser("multiseed", help="Multi-seed analysis")
    p_multi.add_argument("subcommand", choices=["aggregate", "visualize", "report", "all"])
    p_multi.add_argument("--job-prefixes", dest="job_prefixes", help="Job ID prefixes")
    p_multi.add_argument("--baseline", default="baseline")
    p_multi.set_defaults(func=cmd_multiseed)
    
    # optuna
    p_optuna = subparsers.add_parser("optuna", help="Optuna convergence analysis")
    p_optuna.add_argument("subcommand", choices=["simulate", "collect", "visualize", "visualize-logs"])
    p_optuna.add_argument("--method", help="Imbalance method")
    p_optuna.add_argument("--log-dir", dest="log_dir", help="Log directory")
    p_optuna.set_defaults(func=cmd_optuna)
    
    # statistical
    p_stat = subparsers.add_parser("statistical", help="Statistical tests")
    p_stat.add_argument("subcommand", choices=["per-subject", "aggregate", "all"])
    p_stat.set_defaults(func=cmd_statistical)
    
    # scores
    p_scores = subparsers.add_parser("scores", help="Compute subject scores")
    p_scores.set_defaults(func=cmd_scores)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
