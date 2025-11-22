#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified CLI for analysis tasks in the DDD pipeline.

Provides subcommands for distance computation, correlation analyses,
summary report generation, metrics aggregation, ranking export, and
pretrain group reporting.

Subcommands (summary):
  comp-dist            Compute MMD / Wasserstein / DTW distance matrices
  corr                 Correlate group distances with delta metrics
  summarize            Summarize only10 vs finetune results (radar optional)
  summarize-metrics    Aggregate metrics_* CSVs to wide summary
  make-table           Build wide comparison table (only10 vs finetune)
  report-pretrain-groups  Intra/Inter/NN distance report for pretrain groups
  corr-collect         Collect correlation CSVs into heatmap summary
  rank-export          Export top/bottom-k subject rankings

Use `python analyze.py <subcommand> -h` for subcommand-specific options.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to import path (align with train/preprocess/evaluate style)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Limit BLAS threads (HPC-friendly)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# ---- import backend functions from src/analysis ----
from src import config as cfg
from src.analysis.distance_computation import run_comp_dist
from src.analysis.distance_correlation import run_distance_vs_delta
from src.analysis.group_comparison import run_summarize_only10_vs_finetune
from src.analysis.metrics_tables import summarize_metrics, make_comparison_table
from src.analysis.group_distance_report import run_report_pretrain_groups
from src.analysis.subject_ranking import run_rank_export
from src.analysis.distance_correlation import run_corr_all

# ---------------------- subcommand handlers ----------------------
def cmd_comp_dist(args) -> int:
    """Compute distance matrices (MMD, Wasserstein, DTW) and summaries.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments parsed from CLI.

    Returns
    -------
    int
        Return code (0 = success, non-zero = error).
    """
    logging.info(
        "[RUN] comp-dist | subject_list=%s | data_root=%s | groups_file=%s",
        args.subject_list, args.data_root, args.groups_file
    )
    metrics_to_run = [args.metric] if args.metric != "all" else cfg.DISTANCE_METRICS
    for metric in metrics_to_run:
        print(f"[INFO] Computing {metric.upper()} distance...")
        rc = run_comp_dist(
            subject_list_path=args.subject_list,
            data_root=args.data_root,
            groups_file=args.groups_file,
            metric=metric,
            n_jobs=args.n_jobs,
        )
        if rc != 0:
            print(f"[WARN] run_comp_dist returned non-zero exit code for {metric.upper()}")
    return 0


def cmd_corr(args) -> int:
    """Correlate group distances with Δ metrics.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments parsed from CLI.

    Returns
    -------
    int
        Return code (0 = success, non-zero = error).
    """
    logging.info(
        "[RUN] corr | summary=%s | distance=%s | outdir=%s",
        args.summary_csv, args.distance, args.outdir
    )
    rc = run_distance_vs_delta(
        summary_csv=args.summary_csv,
        distance_path=args.distance,
        groups_dir=args.groups_dir,
        group_names_file=args.group_names_file,
        outdir=args.outdir,
        subjects_json=args.subjects_json,
        subject_list=args.subject_list,
    )
    logging.info("[DONE] corr rc=%s", rc)
    return rc


def cmd_summarize(args) -> int:
    """Summarize only10 vs finetune results.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments parsed from CLI.

    Returns
    -------
    int
        Always 0 (success).
    """
    PRJ = Path(__file__).resolve().parents[1]
    logging.info(
        "[RUN] summarize | names=%s | model_dir=%s | out_prefix=%s | make_radar=%s",
        args.names_file, args.model_dir, args.out_prefix, args.make_radar
    )
    run_summarize_only10_vs_finetune(
        names_file=Path(args.names_file) if args.names_file else (PRJ / "misc" / "pretrain_groups" / "group_names.txt"),
        model_dir=Path(args.model_dir),
        out_prefix=args.out_prefix,
        model=args.model,
        split=args.split,
        make_radar=args.make_radar,
        only10_pattern=args.only10_pattern,
        finetune_pattern=args.finetune_pattern,
    )
    logging.info("[DONE] summarize")
    return 0


def cmd_summarize_metrics(args) -> int:
    """Generate a wide-format comparison table (only10 vs finetune).

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments parsed from CLI.

    Returns
    -------
    int
        Return code (0 = success, non-zero = error).
    """
    out_csv = Path(args.out_csv) if args.out_csv else (Path(args.model_dir) / "summary_only10_vs_finetune.csv")
    logging.info(
        "[RUN] summarize-metrics | model_dir=%s | model_tag=%s | split=%s | out=%s",
        args.model_dir, args.model_tag, args.split, out_csv
    )
    df = summarize_metrics(
        model_dir=Path(args.model_dir),
        model_tag=args.model_tag,
        split=args.split,
        out_csv=out_csv,
    )
    logging.info("Saved: %s", out_csv)
    # echo small table
    try:
        import pandas as _pd  # noqa
        print(df)
    except Exception:
        pass
    return 0


def cmd_make_table(args) -> int:
    """Generate a wide-format comparison table (only10 vs finetune).

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments parsed from CLI.

    Returns
    -------
    int
        Return code (0 = success, non-zero = error).
    """
    model_dir = Path(args.model_dir)
    summary_csv = Path(args.summary_csv) if args.summary_csv else (model_dir / "summary_only10_vs_finetune.csv")
    out_csv = Path(args.out_csv) if args.out_csv else (model_dir / "table_only10_vs_finetune_wide.csv")

    logging.info("[RUN] make-table | model_dir=%s | out=%s", model_dir, out_csv)

    if not summary_csv.exists():
        logging.info("summary not found; generating: %s", summary_csv)
        summarize_metrics(model_dir=model_dir, model_tag=args.model_tag, split=args.split, out_csv=summary_csv)

    wide = make_comparison_table(summary_df_or_path=summary_csv, out_csv=out_csv)
    logging.info("Saved CSV: %s", out_csv)
    try:
        print(wide)
    except Exception:
        pass
    return 0


def cmd_report_pretrain_groups(args) -> int:
    """Report intra/inter/NN statistics for pretrain groups.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments parsed from CLI.

    Returns
    -------
    int
        Return code (0 = success, non-zero = error).
    """
    logging.info("[RUN] report-pretrain-groups | group_dir=%s", args.group_dir)
    run_report_pretrain_groups(
        group_dir=Path(args.group_dir),
        out_summary_json=Path(args.out_summary_json),
        out_summary_csv=Path(args.out_summary_csv),
        metrics_root=Path(args.metrics_root),
    )
    logging.info("[DONE] wrote %s and %s", args.out_summary_json, args.out_summary_csv)
    return 0


def cmd_corr_collect(args) -> int:
    """Collect correlation CSVs and draw a Pearson heatmap.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments parsed from CLI.

    Returns
    -------
    int
        Return code (0 = success, non-zero = error).
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    logging.info("[RUN] corr-collect")
    items = []
    if args.mmd and Path(args.mmd).exists():
        df = pd.read_csv(args.mmd); df["distance_type"] = "MMD"; items.append(df)
    if args.wass and Path(args.wass).exists():
        df = pd.read_csv(args.wass); df["distance_type"] = "Wasserstein"; items.append(df)
    if args.dtw and Path(args.dtw).exists():
        df = pd.read_csv(args.dtw); df["distance_type"] = "DTW"; items.append(df)
    if not items:
        logging.error("No input CSVs found.")
        return 1

    pearson_df = pd.concat(items, ignore_index=True)
    pearson_df.to_csv(args.out_csv, index=False)
    logging.info("Saved: %s", args.out_csv)

    # Heatmap on pearson_r
    pivot_df = pearson_df.pivot(index="metric", columns="distance_type", values="pearson_r")
    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot_df, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Pearson correlation (d(U,G) vs Δmetrics)")
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=300)
    plt.close()
    logging.info("Saved: %s", args.out_png)
    return 0

def cmd_rank_export(args) -> int:
    """Export top/bottom-k subject rankings from distance matrices.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments parsed from CLI.

    Returns
    -------
    int
        Return code (0 = success, non-zero = error).
    """
    logging.info(
        "[RUN] rank-export | outdir=%s | k=%s", args.outdir, args.k
    )
    rc = run_rank_export(
        outdir=Path(args.outdir),
        k=int(args.k),
        metrics_root=Path(cfg.RESULTS_DOMAIN_GENERALIZATION_PATH),
    )
    logging.info("[DONE] rank-export rc=%s", rc)
    return rc

# ---------------------- CLI setup ----------------------
def build_parser() -> argparse.ArgumentParser:
    PRJ = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Unified analysis CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    # comp-dist
    s = sub.add_parser("comp-dist", help="Compute distance matrices and summaries.")
    s.add_argument("--subject_list", default=str(PRJ / "config" / "subjects" / "subject_list.txt"))
    s.add_argument("--data_root", default=cfg.PROCESS_CSV_COMMON_PATH)
    s.add_argument("--groups_file", default=str(PRJ / "misc" / "target_groups.txt"))
    s.add_argument(
        "--metric",
        choices=["mmd", "wasserstein", "dtw", "all"],
        default="all",
        help="Which metric(s) to compute. Default: all"
    )
    s.add_argument(
        "--n_jobs",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4, recommended ≤8 on HPC)"
    )
    s.set_defaults(func=cmd_comp_dist)

    # corr
    s = sub.add_parser("corr", help="Correlate group distances with delta metrics.")
    s.add_argument("--summary_csv", required=True)
    s.add_argument("--distance", required=True, help=".csv or .npy")
    s.add_argument("--subjects_json", default=None, help="Required when distance is .npy")
    s.add_argument("--groups_dir", required=True)
    s.add_argument("--group_names_file", required=True)
    s.add_argument("--outdir", default="model/common/dist_corr")
    s.add_argument("--subject_list", default=None)
    s.set_defaults(func=cmd_corr)

    # corr-all
    s = sub.add_parser("corr-all", help="Run correlation analysis for all metrics (MMD/Wasserstein/DTW).")
    s.add_argument("--summary_csv", required=True,
                   help="Path to wide-format summary CSV (e.g., model/common/summary_6groups_only10_vs_finetune_wide.csv).")
    s.add_argument("--groups_dir", required=True,
                   help="Directory containing group definition text files.")
    s.add_argument("--group_names_file", required=True,
                   help="Path to file listing group names (e.g., misc/pretrain_groups/group_names.txt).")
    s.add_argument("--metrics_root", default=cfg.RESULTS_DOMAIN_GENERALIZATION_PATH,
                   help="Root directory containing mmd/, wasserstein/, dtw/ subfolders.")
    s.add_argument("--out_root", default=os.path.join(cfg.RESULTS_DOMAIN_GENERALIZATION_PATH, "corr_all"),
                   help="Output directory for combined correlation results.")
    s.set_defaults(func=lambda args: run_corr_all(
        summary_csv=Path(args.summary_csv),
        groups_dir=Path(args.groups_dir),
        group_names_file=Path(args.group_names_file),
        metrics_root=Path(args.metrics_root),
        out_root=Path(args.out_root),
    ))

    # summarize
    s = sub.add_parser("summarize", help="Summarize only10 vs finetune (optionally with radar).")
    s.add_argument("--names_file", default=str(PRJ / "misc" / "pretrain_groups" / "group_names.txt"))
    s.add_argument("--model_dir",  default=str(PRJ / "model" / "common"))
    s.add_argument("--out_prefix", default="summary_6groups_only10_vs_finetune")
    s.add_argument("--model", default="RF")
    s.add_argument("--split", default="test")
    s.add_argument("--make_radar", action="store_true")
    s.add_argument("--only10_pattern",   default="metrics_{model}_only10_{group}.csv")
    s.add_argument("--finetune_pattern", default="metrics_{model}_finetune_{group}_finetune.csv")
    s.set_defaults(func=cmd_summarize)

    # summarize-metrics
    s = sub.add_parser("summarize-metrics", help="Summarize metrics_* CSVs under model_dir.")
    s.add_argument("--model_dir", default=str(PRJ / "model" / "common"))
    s.add_argument("--model_tag", default="RF")
    s.add_argument("--split", default="test")
    s.add_argument("--out_csv", default=None)
    s.set_defaults(func=cmd_summarize_metrics)

    # make-table
    s = sub.add_parser("make-table", help="Make wide comparison table (only10 vs finetune).")
    s.add_argument("--model_dir", default=str(PRJ / "model" / "common"))
    s.add_argument("--model_tag", default="RF")
    s.add_argument("--split", default="test")
    s.add_argument("--summary_csv", default=None)
    s.add_argument("--out_csv", default=None)
    s.set_defaults(func=cmd_make_table)

    # report-pretrain-groups
    s = sub.add_parser("report-pretrain-groups", help="Report Intra/Inter/NN for 10 vs 78 groups.")
    s.add_argument("--group_dir", default=str(PRJ / "misc" / "pretrain_groups"))
    s.add_argument("--out_summary_json", default=str(PRJ / "misc" / "pretrain_groups" / "summary_report_ext.json"))
    s.add_argument("--out_summary_csv",  default=str(PRJ / "misc" / "pretrain_groups" / "summary_report_ext.csv"))
    s.add_argument("--metrics_root", default=str(PRJ / cfg.RESULTS_DOMAIN_GENERALIZATION_PATH),
                   help="Root directory containing metric results (mmd/wasserstein/dtw).")
    s.set_defaults(func=cmd_report_pretrain_groups)

    # corr-collect (optional)
    s = sub.add_parser("corr-collect", help="Collect correlation CSVs and draw heatmap.")
    s.add_argument("--mmd", default=None, help=".../correlations_dUG_vs_deltas.csv for MMD")
    s.add_argument("--wass", default=None, help=".../correlations_dUG_vs_deltas.csv for Wasserstein")
    s.add_argument("--dtw", default=None, help=".../correlations_dUG_vs_deltas.csv for DTW")
    s.add_argument("--out_csv", default="correlation_summary_all.csv")
    s.add_argument("--out_png", default="correlation_heatmap_all.png")
    s.set_defaults(func=cmd_corr_collect)

    # rank-export (neutral, research-friendly naming)
    s = sub.add_parser("rank-export", help="Export top/bottom-k subject lists from mean/std rankings (MMD/Wasserstein/DTW).")
    s.add_argument("--outdir", default="results/ranks")
    s.add_argument("--k", type=int, default=10)
    s.add_argument("--metrics_root", default=cfg.RESULTS_DOMAIN_GENERALIZATION_PATH,
                   help="Root directory containing metric results (mmd/wasserstein/dtw).")
    s.set_defaults(func=cmd_rank_export)

    return p


def main() -> int:
    """Entry point: parse subcommand and dispatch.

    Returns
    -------
    int
        Process exit code (0 = success, non-zero = failure).
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

