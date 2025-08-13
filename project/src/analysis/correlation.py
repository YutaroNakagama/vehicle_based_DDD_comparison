# src/analysis/correlation.py
from __future__ import annotations
import os
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


def _read_group_members(groups_dir: str | Path, group_names_file: str | Path) -> dict[str, list[str]]:
    names = [ln.strip() for ln in Path(group_names_file).read_text(encoding="utf-8").splitlines() if ln.strip()]
    groups: dict[str, list[str]] = {}
    for name in names:
        p = Path(groups_dir) / f"{name}.txt"
        members = [x for x in p.read_text(encoding="utf-8").split() if x]
        groups[name] = members
    return groups


def _mean_cross_distance(D: pd.DataFrame, A: list[str], B: list[str]) -> float:
    A2 = [a for a in A if a in D.index]
    B2 = [b for b in B if b in D.columns]
    if not A2 or not B2:
        return float("nan")
    return float(np.nanmean(D.loc[A2, B2].to_numpy()))


def _mean_within_distance(D: pd.DataFrame, A: list[str]) -> float:
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
    """Load a square distance matrix as a DataFrame with index/columns = subject IDs.

    Supports:
      - CSV: distance_path=*.csv (header+index assumed)
      - NPY: distance_path=*.npy with subjects_json pointing to a JSON list of IDs
    """
    distance_path = Path(distance_path)
    if distance_path.suffix.lower() == ".csv":
        D = pd.read_csv(distance_path, index_col=0)
        D.index = D.index.str.strip()
        D.columns = D.columns.str.strip()
        return D
    elif distance_path.suffix.lower() == ".npy":
        if subjects_json is None:
            raise ValueError("subjects_json is required when using a .npy distance matrix.")
        arr = np.load(distance_path)
        with open(subjects_json, "r", encoding="utf-8") as f:
            subjects = [s.strip() for s in pd.read_json(f, typ="series").tolist()] if distance_path.name.endswith("_bad.json") else None
        # safer JSON load:
        import json
        with open(subjects_json, "r", encoding="utf-8") as f:
            subjects = json.load(f)
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
    """Correlate group-level distances (d(U,G), disp(G)) with delta metrics from summary CSV.

    - summary_csv: wide format with columns: group, accuracy_delta, f1_delta, auc_delta, precision_delta, recall_delta
    - distance_path: CSV (square matrix with header+index) or NPY (use subjects_json)
    - groups_dir & group_names_file: group members and ordered group names
    - outdir: output directory
    - subject_list: optional full list of subjects; otherwise inferred from distance matrix
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

    df = pd.read_csv(summary_csv)
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
    merged_out.to_csv(merged_csv, index=False)

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
    corr_df.to_csv(out / "correlations_dUG_vs_deltas.csv", index=False)

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
    pd.DataFrame(corr_rows2).to_csv(out / "correlations_dispG_vs_deltas.csv", index=False)

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
        plt.savefig(out / f"scatter_dUG_vs_{m}.png", dpi=200)
        plt.close()

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
        plt.savefig(out / f"scatter_dispG_vs_{m}.png", dpi=200)
        plt.close()

    # Save merged table again for convenience
    merged_out.to_csv(out / "distance_vs_delta_merged.csv", index=False)
    return 0

