"""Correlation analysis between group-level distances and evaluation deltas.

This module provides functions to compute correlations between group distances
(e.g., mean inter-group distance d(U,G), within-group dispersion disp(G)) and
changes in evaluation metrics (Δ accuracy, Δ F1, Δ AUC, etc.).

It supports both CSV and NPY distance matrices and generates correlation tables
and scatter plots as output.

Functions
---------
_read_group_members(groups_dir, group_names_file)
    Read group definitions from text files.
_mean_cross_distance(D, A, B)
    Compute mean pairwise distance between two sets of subjects.
_mean_within_distance(D, A)
    Compute mean pairwise distance within a group of subjects.
_load_distance_matrix(distance_path, subjects_json)
    Load a square distance matrix from CSV or NPY format.
run_distance_vs_delta(summary_csv, distance_path, groups_dir, group_names_file, ...)
    Compute correlations between group distances and evaluation deltas.
"""

from __future__ import annotations
import os
import logging
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

from src import config as cfg
from src.utils.io.data_io import (
    load_csv, save_csv, load_json, save_json,
    load_numpy, load_distance_data
)
from src.utils.visualization import save_current_figure

logger = logging.getLogger(__name__)


def _read_group_members(groups_dir: str | Path, group_names_file: str | Path) -> dict[str, list[str]]:
    """Read group member lists from text files.

    Parameters
    ----------
    groups_dir : str or Path
        Directory containing text files with subject IDs for each group.
    group_names_file : str or Path
        File listing group names, one per line.

    Returns
    -------
    dict of {str: list of str}
        Mapping from group name to list of subject IDs.
    """
    names = [ln.strip() for ln in Path(group_names_file).read_text(encoding="utf-8").splitlines() if ln.strip()]
    groups: dict[str, list[str]] = {}
    for name in names:
        p = Path(groups_dir) / f"{name}.txt"
        members = [x for x in p.read_text(encoding="utf-8").split() if x]
        groups[name] = members
    return groups


def _mean_cross_distance(D: pd.DataFrame, A: list[str], B: list[str]) -> float:
    """Compute mean pairwise distance between two sets of subjects.

    Parameters
    ----------
    D : pandas.DataFrame
        Square distance matrix (subjects × subjects).
    A : list of str
        Subject IDs for set A.
    B : list of str
        Subject IDs for set B.

    Returns
    -------
    float
        Mean distance between subjects in A and B, or NaN if invalid.
    """
    A2 = [a for a in A if a in D.index]
    B2 = [b for b in B if b in D.columns]
    if not A2 or not B2:
        return float("nan")
    return float(np.nanmean(D.loc[A2, B2].to_numpy()))


def _mean_within_distance(D: pd.DataFrame, A: list[str]) -> float:
    """Compute mean pairwise distance within a set of subjects.

    Parameters
    ----------
    D : pandas.DataFrame
        Square distance matrix (subjects × subjects).
    A : list of str
        Subject IDs for the group.

    Returns
    -------
    float
        Mean within-group distance, or NaN if insufficient subjects.
    """
    A2 = [a for a in A if a in D.index]
    if len(A2) < 2:
        return float("nan")
    vals: list[float] = []
    for i, j in combinations(A2, 2):
        v1 = D.at[i, j] if (i in D.index and j in D.columns) else np.nan
        v2 = D.at[j, i] if (j in D.index and i in D.columns) else np.nan
        if not (isinstance(v1, float) and np.isnan(v1)):
            vals.append(float(v1))
        if not (isinstance(v2, float) and np.isnan(v2)):
            vals.append(float(v2))
    return float(np.mean(vals)) if vals else float("nan")


def _load_distance_matrix(
    distance_path: str | Path,
    subjects_json: str | Path | None = None,
) -> pd.DataFrame:
    """Load a square distance matrix as a DataFrame.

    Supports:
    - CSV: ``*.csv`` file with header and index.
    - NPY: ``*.npy`` file with subjects specified in ``subjects_json``.

    Parameters
    ----------
    distance_path : str or Path
        Path to the distance matrix file (.csv or .npy).
    subjects_json : str or Path, optional
        JSON file containing subject IDs. Required if ``distance_path`` is ``.npy``.

    Returns
    -------
    pandas.DataFrame
        Square distance matrix with subject IDs as both index and columns.

    Raises
    ------
    ValueError
        If file type is unsupported or the number of subjects does not match
        the matrix shape.
    """
    distance_path = Path(distance_path)
    if distance_path.suffix.lower() == ".csv":
        D = load_csv(distance_path, index_col=0)
        D.index = D.index.str.strip()
        D.columns = D.columns.str.strip()
        return D
    elif distance_path.suffix.lower() == ".npy":
        if subjects_json is None:
            raise ValueError("subjects_json is required when using a .npy distance matrix.")
        arr = load_numpy(distance_path)
        subjects = load_json(subjects_json)
        if len(subjects) != arr.shape[0]:
            raise ValueError(f"subjects length ({len(subjects)}) and matrix shape {arr.shape} mismatch.")
        D = pd.DataFrame(arr, index=subjects, columns=subjects)
        return D
    else:
        raise ValueError(f"Unsupported distance file: {distance_path}")


def run_distance_vs_delta(
    summary_csv: str | Path,
    distance_path: str | Path,
    groups_dir: str | Path,
    group_names_file: str | Path,
    outdir: str | Path = "model/common/dist_corr",
    subjects_json: str | Path | None = None,
    subject_list: str | Path | None = None,
) -> int:
    """Correlate group-level distances with evaluation deltas.

    This function computes:
    - d(U,G): mean distance between group G and the complement U
    - disp(G): within-group dispersion of G

    These are then correlated with Δ metrics (accuracy, F1, AUC, precision, recall)
    from the summary CSV.

    Parameters
    ----------
    summary_csv : str or Path
        Path to summary CSV containing columns ``group`` and Δ metrics.
    distance_path : str or Path
        Path to distance matrix (.csv or .npy).
    groups_dir : str or Path
        Directory containing group membership files.
    group_names_file : str or Path
        File listing group names, one per line.
    outdir : str or Path, default="model/common/dist_corr"
        Output directory for correlation results.
    subjects_json : str or Path, optional
        Required if ``distance_path`` is ``.npy``. JSON file with subject IDs.
    subject_list : str or Path, optional
        Path to text file with the global subject list. If omitted, inferred
        from the distance matrix.

    Returns
    -------
    int
        Return code (0 indicates success).

    Notes
    -----
    This function generates the following output files:
    - ``distance_vs_delta_merged.csv``: merged table of distances and deltas
    - ``correlations_dUG_vs_deltas.csv``: correlations between d(U,G) and Δ metrics
    - ``correlations_dispG_vs_deltas.csv``: correlations between disp(G) and Δ metrics
    - scatter plots (PNG) for each Δ metric
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    # Load distance matrix
    D = _load_distance_matrix(distance_path, subjects_json=subjects_json)

    # All subjects
    if subject_list and Path(subject_list).exists():
        all_subjects = [ln.strip() for ln in Path(subject_list).read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        all_subjects = sorted(set(D.index.tolist()) | set(D.columns.tolist()))

    # Groups and summary
    groups = _read_group_members(groups_dir, group_names_file)

    df = load_csv(summary_csv)
    df = df.rename(columns={c: c.lower() for c in df.columns})
    needed_delta = ["accuracy_delta", "f1_delta", "auc_delta", "precision_delta", "recall_delta"]
    for c in needed_delta:
        if c not in df.columns:
            raise ValueError(f"{summary_csv} is missing column: {c}")
    if "group" not in df.columns:
        raise ValueError("summary CSV must contain 'group' column.")

    df["group_norm"] = df["group"].str.strip().str.lower()

    # Distance features per group
    rows: list[dict] = []
    for name, members in groups.items():
        G = members
        U = [s for s in all_subjects if s not in G]
        d_u_g = _mean_cross_distance(D, U, G)
        disp_g = _mean_within_distance(D, G)
        rows.append({"group": name, "d_UG": d_u_g, "disp_G": disp_g})
    dist_df = pd.DataFrame(rows)
    dist_df["group_norm"] = dist_df["group"].str.strip().str.lower()

    # Merge
    merged = dist_df.merge(df.drop(columns=["group"], errors="ignore"), on="group_norm", how="left")
    merged_out = merged.drop(columns=["group_norm"]).copy()
    merged_csv = out / "distance_vs_delta_merged.csv"
    save_csv(merged_out, merged_csv)

    # Correlations: d(U,G)
    metrics = ["accuracy_delta", "f1_delta", "auc_delta", "precision_delta", "recall_delta"]
    corr_rows = []
    for m in metrics:
        x = merged["d_UG"].to_numpy()
        y = merged[m].to_numpy()
        mask = ~(np.isnan(x) | np.isnan(y))
        xv, yv = x[mask], y[mask]
        if len(xv) >= 3:
            p_r, p_p = pearsonr(xv, yv)
            s_r, s_p = spearmanr(xv, yv)
        else:
            p_r = p_p = s_r = s_p = np.nan
        corr_rows.append({"metric": m, "pearson_r": p_r, "pearson_p": p_p, "spearman_rho": s_r, "spearman_p": s_p})
    corr_df = pd.DataFrame(corr_rows)
    save_csv(corr_df, out / "correlations_dUG_vs_deltas.csv")

    # Correlations: disp(G)
    corr_rows2 = []
    for m in metrics:
        x = merged["disp_G"].to_numpy()
        y = merged[m].to_numpy()
        mask = ~(np.isnan(x) | np.isnan(y))
        xv, yv = x[mask], y[mask]
        if len(xv) >= 3:
            p_r, p_p = pearsonr(xv, yv)
            s_r, s_p = spearmanr(xv, yv)
        else:
            p_r = p_p = s_r = s_p = np.nan
        corr_rows2.append({"metric": m, "pearson_r": p_r, "pearson_p": p_p, "spearman_rho": s_r, "spearman_p": s_p})
    save_csv(pd.DataFrame(corr_rows2), out / "correlations_dispG_vs_deltas.csv")

    # Plots
    def _annotate(ax, x, y, labels):
        for xi, yi, lb in zip(x, y, labels):
            ax.text(float(xi), float(yi), str(lb), fontsize=9, ha="left", va="bottom")

    # d(U,G) vs Δ
    for m, label in [
        ("accuracy_delta", "Δ Accuracy (finetune − only10)"),
        ("f1_delta",       "Δ F1"),
        ("auc_delta",      "Δ AUC"),
        ("precision_delta","Δ Precision"),
        ("recall_delta",   "Δ Recall"),
    ]:
        x = merged["d_UG"].to_numpy()
        y = merged[m].to_numpy()
        labs = merged["group"].to_numpy()
        mask = ~(np.isnan(x) | np.isnan(y))
        xv, yv, lv = x[mask], y[mask], labs[mask]
        plt.figure(figsize=(6, 4))
        plt.scatter(xv, yv)
        if len(xv) >= 2:
            a, b = np.polyfit(xv, yv, 1)
            xs = np.linspace(min(xv), max(xv), 100)
            ys = a * xs + b
            plt.plot(xs, ys)
        _annotate(plt.gca(), xv, yv, lv)
        plt.xlabel("Mean distance d(U, G)")
        plt.ylabel(label)
        plt.title(f"d(U,G) vs {label}")
        plt.tight_layout()
        save_current_figure(out / f"scatter_dUG_vs_{m}.png", dpi=200, close=True)

    # disp(G) vs Δ
    for m, label in [
        ("accuracy_delta", "Δ Accuracy"),
        ("f1_delta",       "Δ F1"),
        ("auc_delta",      "Δ AUC"),
        ("precision_delta","Δ Precision"),
        ("recall_delta",   "Δ Recall"),
    ]:
        x = merged["disp_G"].to_numpy()
        y = merged[m].to_numpy()
        labs = merged["group"].to_numpy()
        mask = ~(np.isnan(x) | np.isnan(y))
        xv, yv, lv = x[mask], y[mask], labs[mask]
        plt.figure(figsize=(6, 4))
        plt.scatter(xv, yv)
        if len(xv) >= 2:
            a, b = np.polyfit(xv, yv, 1)
            xs = np.linspace(min(xv), max(xv), 100)
            ys = a * xs + b
            plt.plot(xs, ys)
        _annotate(plt.gca(), xv, yv, lv)
        plt.xlabel("Within-group dispersion disp(G)")
        plt.ylabel(label)
        plt.title(f"disp(G) vs {label}")
        plt.tight_layout()
        save_current_figure(out / f"scatter_dispG_vs_{m}.png", dpi=200, close=True)

    return 0


# ============================================================
# Unified runner for all metrics (MMD / Wasserstein / DTW)
# ============================================================

def run_corr_all(
    summary_csv: Path,
    groups_dir: Path,
    group_names_file: Path,
    metrics_root: Path = None,
    out_root: Path = None,
) -> None:
    """
    Run correlation analysis for all available metrics (MMD, Wasserstein, DTW)
    and aggregate results into a unified summary.

    Parameters
    ----------
    summary_csv : Path
        Path to wide-format summary file (only10 vs finetune results).
    groups_dir : Path
        Directory containing group definition text files.
    group_names_file : Path
        File listing group names (used for iteration).
    metrics_root : Path, default=None
        Root directory containing mmd/, wasserstein/, and dtw/ folders.
        If None, uses cfg.RESULTS_DOMAIN_GENERALIZATION_PATH.
    out_root : Path, default=None
        Output directory where correlation summaries and heatmaps will be stored.
        If None, uses cfg.RESULTS_DOMAIN_GENERALIZATION_PATH/corr_all.

    Returns
    -------
    None
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    if metrics_root is None:
        metrics_root = Path(cfg.RESULTS_DOMAIN_GENERALIZATION_PATH)
    if out_root is None:
        out_root = Path(cfg.RESULTS_DOMAIN_GENERALIZATION_PATH) / "corr_all"

    out_root.mkdir(parents=True, exist_ok=True)
    metrics = ["mmd", "wasserstein", "dtw"]
    all_corrs = []

    for metric in metrics:
        mat_path = metrics_root / metric / f"{metric}_matrix.npy"
        subj_path = metrics_root / metric / f"{metric}_subjects.json"
        outdir = out_root / f"dist_corr_{metric}"

        if not mat_path.exists() or not subj_path.exists():
            print(f"[WARN] Skipping {metric.upper()} (missing files)")
            continue

        print(f"[INFO] Running correlation for {metric.upper()}")
        run_distance_vs_delta(
            summary_csv=summary_csv,
            distance_path=mat_path,
            groups_dir=groups_dir,
            group_names_file=group_names_file,
            outdir=outdir,
            subjects_json=subj_path,
        )

        corr_file = corr_dir / "correlations_dUG_vs_deltas.csv"
        if corr_file.exists():
            df = load_csv(corr_file)
            df["metric_type"] = metric
            all_corrs.append(df)

    # Merge all correlations into one table
    if not all_corr:
        logging.warning("No correlation files found.")
        return

    merged = pd.concat(all_corrs, ignore_index=True)
    merged_csv = out_root / "correlation_summary_all.csv"
    save_csv(merged, merged_csv)

    # Draw heatmap (Pearson r)
    pivot = merged.pivot(index="metric", columns="metric_type", values="pearson_r")
    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Pearson correlation (d(U,G) vs Δmetrics)")
    plt.tight_layout()
    out_png = out_root / "correlation_heatmap_all.png"
    save_current_figure(out_png, dpi=300, close=True)

    print(f"[DONE] Merged correlations written to {merged_csv}")
    print(f"[DONE] Heatmap saved to {out_png}")


def collect_correlation_csvs(
    mmd_csv: Path | None = None,
    wasserstein_csv: Path | None = None,
    dtw_csv: Path | None = None,
    out_csv: Path = None,
    out_png: Path = None,
) -> int:
    """Collect individual correlation CSVs and create unified summary with heatmap.

    This function merges correlation results from multiple distance metrics
    (MMD, Wasserstein, DTW) into a single CSV and generates a Pearson correlation
    heatmap visualization.

    Parameters
    ----------
    mmd_csv : Path, optional
        Path to MMD correlation CSV file.
    wasserstein_csv : Path, optional
        Path to Wasserstein correlation CSV file.
    dtw_csv : Path, optional
        Path to DTW correlation CSV file.
    out_csv : Path, optional
        Output path for merged correlation CSV.
        Defaults to current directory / "correlation_summary_collected.csv".
    out_png : Path, optional
        Output path for heatmap PNG.
        Defaults to current directory / "correlation_heatmap_collected.png".

    Returns
    -------
    int
        Return code (0 = success, 1 = no input files found).

    Examples
    --------
    >>> collect_correlation_csvs(
    ...     mmd_csv=Path("corr/mmd/correlations_dUG_vs_deltas.csv"),
    ...     wasserstein_csv=Path("corr/wasserstein/correlations_dUG_vs_deltas.csv"),
    ...     out_csv=Path("results/correlation_summary.csv"),
    ...     out_png=Path("results/correlation_heatmap.png"),
    ... )
    0
    """
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    if out_csv is None:
        out_csv = Path("correlation_summary_collected.csv")
    if out_png is None:
        out_png = Path("correlation_heatmap_collected.png")

    items = []
    if mmd_csv and mmd_csv.exists():
        df = load_csv(mmd_csv)
        df["distance_type"] = "MMD"
        items.append(df)
    if wasserstein_csv and wasserstein_csv.exists():
        df = load_csv(wasserstein_csv)
        df["distance_type"] = "Wasserstein"
        items.append(df)
    if dtw_csv and dtw_csv.exists():
        df = load_csv(dtw_csv)
        df["distance_type"] = "DTW"
        items.append(df)

    if not items:
        logging.error("No input CSVs found.")
        return 1

    pearson_df = pd.concat(items, ignore_index=True)
    save_csv(pearson_df, out_csv)
    logging.info("Saved merged correlations: %s", out_csv)

    # Generate heatmap
    pivot_df = pearson_df.pivot(index="metric", columns="distance_type", values="pearson_r")
    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot_df, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Pearson correlation (d(U,G) vs Δmetrics)")
    plt.tight_layout()
    save_current_figure(out_png, dpi=300, close=True)
    logging.info("Saved heatmap: %s", out_png)

    return 0
