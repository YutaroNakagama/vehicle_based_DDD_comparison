from __future__ import annotations
import os, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.manifold import MDS
from scipy.stats import wasserstein_distance
from tslearn.metrics import dtw
from tqdm import tqdm
from joblib import Parallel, delayed

CACHE_VERSION = "v1"  # bump this when feature layout/filters change

# Limit BLAS threads
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

def _cache_key(subjects, data_root: Path) -> str:
    """Generate a unique cache key for feature extraction.

    Parameters
    ----------
    subjects : list of str
        List of subject identifiers.
    data_root : Path
        Path to the root directory of processed data.

    Returns
    -------
    str
        MD5 hash string used as cache key.
    """
    s = CACHE_VERSION + "|" + "|".join(subjects) + "@" + str(data_root.resolve())
    return hashlib.md5(s.encode()).hexdigest()

def _extract_features_with_cache(subjects, data_root: Path, cache_dir: Path = Path("results/.cache")):
    """Extract features with caching.

    Parameters
    ----------
    subjects : list of str
        List of subject identifiers.
    data_root : Path
        Root directory where processed CSVs are stored.
    cache_dir : Path, default="results/.cache"
        Directory where cached features are saved.

    Returns
    -------
    dict of {str: ndarray}
        Mapping from subject ID to feature matrix.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(subjects, data_root)
    npz = cache_dir / f"features_{key}.npz"
    if npz.exists():
        z = np.load(npz, allow_pickle=True)
        return {k: z[k] for k in z.files}
    feats = _extract_features(subjects, data_root)
    np.savez_compressed(npz, **feats)
    return feats


# ===== Helpers =====
def _median_gamma(X: np.ndarray, Y: np.ndarray, max_samples: int = 1000) -> float:
    """Estimate RBF kernel gamma using the median heuristic.

    Parameters
    ----------
    X : ndarray of shape (n_samples_X, n_features)
        Feature matrix for dataset X.
    Y : ndarray of shape (n_samples_Y, n_features)
        Feature matrix for dataset Y.
    max_samples : int, default=1000
        Maximum number of samples to use for distance estimation.

    Returns
    -------
    float
        Estimated gamma value.
    """
    A = X if len(X) <= max_samples else X[np.random.choice(len(X), max_samples, replace=False)]
    B = Y if len(Y) <= max_samples else Y[np.random.choice(len(Y), max_samples, replace=False)]
    D = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=2)
    med = float(np.median(D))
    if not np.isfinite(med) or med <= 0:
        return 1.0 / X.shape[1]
    return 1.0 / (2.0 * med)

def compute_mmd(X: np.ndarray, Y: np.ndarray, gamma: float | None = None) -> float:
    """Compute the Maximum Mean Discrepancy (MMD) between two datasets.

    Parameters
    ----------
    X : ndarray of shape (n_samples_X, n_features)
        Feature matrix for dataset X.
    Y : ndarray of shape (n_samples_Y, n_features)
        Feature matrix for dataset Y.
    gamma : float, optional
        RBF kernel width parameter. If ``None``, it is estimated using the median heuristic.

    Returns
    -------
    float
        Estimated MMD^2 value.
    """
    if gamma is None:
        gamma = _median_gamma(X, Y)
    Kxx = rbf_kernel(X, X, gamma=gamma)
    Kyy = rbf_kernel(Y, Y, gamma=gamma)
    Kxy = rbf_kernel(X, Y, gamma=gamma)
    m, n = len(X), len(Y)
    return float(Kxx.sum() / (m * m) + Kyy.sum() / (n * n) - 2.0 * Kxy.sum() / (m * n))

def _load_features_one(subject_str: str, data_root: Path) -> np.ndarray | None:
    """Load features for a single subject.

    Parameters
    ----------
    subject_str : str
        Identifier of the subject (format: "<id>_<version>").
    data_root : Path
        Path to directory containing processed CSVs.

    Returns
    -------
    ndarray or None
        Feature matrix for the subject, or ``None`` if missing.
    """
    subject_id, version = subject_str.split("_")
    path = data_root / f"processed_{subject_id}_{version}.csv"
    if not path.exists():
        print(f"[warn] Missing file: {path}")
        return None
    df = pd.read_csv(path)
    feature_cols = [
        c for c in df.columns
        if c not in ["Timestamp", "KSS", "subject"]
        and not c.startswith("Channel_")
        and not c.startswith("KSS_")
        and not c.startswith("theta_alpha_over_beta")
    ]
    return df[feature_cols].dropna().to_numpy()

def _extract_features(subjects: list[str], data_root: Path) -> dict[str, np.ndarray]:
    """Extract features for a list of subjects.

    Parameters
    ----------
    subjects : list of str
        List of subject identifiers.
    data_root : Path
        Root directory where processed CSVs are stored.

    Returns
    -------
    dict of {str: ndarray}
        Mapping from subject ID to feature matrix.
    """
    out: dict[str, np.ndarray] = {}
    for s in subjects:
        x = _load_features_one(s, data_root)
        if x is not None:
            out[s] = x
    return out

def _load_subject_list(path: Path) -> list[str]:
    """Load a list of subject IDs from a text file.

    Parameters
    ----------
    path : Path
        Path to subject list file.

    Returns
    -------
    list of str
        List of subject identifiers.
    """
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]

def _load_groups(path: Path) -> dict[str, list[str]]:
    """Load subject groups from a text file.

    Parameters
    ----------
    path : Path
        Path to file where each line defines a group (subjects separated by spaces).

    Returns
    -------
    dict of {str: list of str}
        Mapping from group name (G1, G2, ...) to subject IDs.
    """
    rows = [line.strip().split() for line in path.read_text().splitlines() if line.strip()]
    return {f"G{idx+1}": group for idx, group in enumerate(rows)}

def _plot_heatmap(matrix: np.ndarray, row_labels: list[str], col_labels: list[str],
                  title: str, save_path: Path, annot: bool = True, fmt: str = ".2f") -> None:
    """Plot and save a heatmap of a matrix.

    Parameters
    ----------
    matrix : ndarray of shape (n_rows, n_cols)
        The 2D matrix to visualize.
    row_labels : list of str
        Labels for the rows.
    col_labels : list of str
        Labels for the columns.
    title : str
        Title of the heatmap.
    save_path : Path
        Path to save the heatmap image.
    annot : bool, default=True
        Whether to annotate the heatmap cells with values.
    fmt : str, default=".2f"
        Format string for annotations.

    Returns
    -------
    None
        Saves the heatmap to the specified file path.
    """
    plt.figure(figsize=(max(8, len(row_labels)//2), max(6, len(col_labels)//2)))
    sns.heatmap(matrix, xticklabels=col_labels, yticklabels=row_labels,
                cmap="viridis", annot=annot, fmt=fmt)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()



# === Unified Plotting Utilities ===

def _plot_save_wrapper(fig, save_path: Path, title: str = ""):
    """Helper to safely save and close Matplotlib figures."""
    try:
        if title:
            fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(save_path, dpi=300)
        print(f"[PLOT] Saved: {save_path}")
    except Exception as e:
        print(f"[WARN] Failed to save plot {save_path}: {e}")
    finally:
        plt.close(fig)


def _plot_heatmap_auto(matrix, labels, metric: str, kind: str, outdir: Path):
    """Unified heatmap generator with consistent naming.

    Parameters
    ----------
    matrix : np.ndarray
        2D distance matrix.
    labels : list[str]
        Axis labels (subjects or groups).
    metric : str
        Metric name (e.g., "mmd", "wasserstein", "dtw").
    kind : str
        Type of matrix ("subject", "group", "centroid", etc.).
    outdir : Path
        Output directory to save the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix, cmap="viridis", xticklabels=labels, yticklabels=labels, annot=False, ax=ax)
    ax.set_title(f"{metric.upper()} {kind.capitalize()} Heatmap")
    fig.tight_layout()
    _plot_save_wrapper(fig, outdir / f"{metric}_{kind}_heatmap.png")


def _plot_bar_auto(names, means, stds, metric: str, kind: str, outdir: Path):
    """Unified bar plot for mean/std metrics.

    Parameters
    ----------
    names : list[str]
        Labels along the x-axis.
    means : np.ndarray
        Mean values.
    stds : np.ndarray
        Standard deviations.
    metric : str
        Metric type ("mmd", "wasserstein", "dtw", etc.).
    kind : str
        Context of the plot ("subjects", "groups", "intra", etc.).
    outdir : Path
        Save directory.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(names)), means, yerr=stds, capsize=3, color="steelblue")
    ax.set_title(f"{metric.upper()} {kind.capitalize()} Mean ± Std")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90)
    ax.set_ylabel("Distance")
    fig.tight_layout()
    _plot_save_wrapper(fig, outdir / f"{metric}_{kind}_bar.png")


def _plot_intra_inter_auto(stats: dict[str, dict[str, float]], metric: str, outdir: Path):
    """Unified intra/inter comparison plot generator."""
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(stats))
    intra_means = [stats[g]["intra_mean"] for g in stats]
    inter_means = [stats[g]["inter_mean"] for g in stats]
    ax.bar(x - 0.2, intra_means, 0.4, label="Intra")
    ax.bar(x + 0.2, inter_means, 0.4, label="Inter")
    ax.set_xticks(x)
    ax.set_xticklabels(list(stats.keys()), rotation=45)
    ax.set_ylabel("Distance")
    ax.legend()
    _plot_save_wrapper(fig, outdir / f"{metric}_intra_inter_comparison.png",
                       f"Intra vs Inter Group Distances ({metric.upper()})")

# === Unified Group Projection ===

def _plot_projection_auto(matrix: np.ndarray, subjects: list[str], groups: dict[str, list[str]],
                          metric: str, outdir: Path) -> None:
    """Unified projection plot generator using MDS (2D).

    Parameters
    ----------
    matrix : np.ndarray
        Pairwise distance matrix.
    subjects : list[str]
        Subject identifiers corresponding to the matrix.
    groups : dict[str, list[str]]
        Mapping from group names to subject IDs.
    metric : str
        Metric name (e.g., 'mmd', 'wasserstein', 'dtw').
    outdir : Path
        Output directory for the saved figure.
    """
    coords = MDS(n_components=2, dissimilarity="precomputed", random_state=42)\
        .fit_transform(np.nan_to_num(matrix))
    s2xy = {s: coords[i] for i, s in enumerate(subjects)}

    fig, ax = plt.subplots(figsize=(10, 8))
    for gname, members in groups.items():
        pts = [(s2xy[s], s) for s in members if s in s2xy]
        if not pts: continue
        arr = np.array([p[0] for p in pts])
        ax.scatter(arr[:, 0], arr[:, 1], label=gname)
        for (xy, sid) in pts:
            ax.text(xy[0], xy[1], sid, fontsize=6)
    ax.set_title(f"{metric.upper()} Group Projection (MDS)")
    ax.legend()
    _plot_save_wrapper(fig, outdir / f"{metric}_group_projection.png")

def _plot_bar(labels: list[str], means: np.ndarray, stds: np.ndarray,
              title: str, ylabel: str, save_path: Path) -> None:
    """Plot and save a bar chart with error bars.

    Parameters
    ----------
    labels : list of str
        Labels for the x-axis (e.g., subject or group names).
    means : ndarray of shape (n_labels,)
        Mean values to plot as bars.
    stds : ndarray of shape (n_labels,)
        Standard deviation values for error bars.
    title : str
        Title of the bar chart.
    ylabel : str
        Label for the y-axis.
    save_path : Path
        Path to save the bar chart image.

    Returns
    -------
    None
        Saves the bar chart to the specified file path.
    """
    x = np.arange(len(labels))
    plt.figure(figsize=(max(12, len(labels)//2), 6))
    plt.bar(x, means, yerr=stds, capsize=5)
    plt.title(title); plt.ylabel(ylabel)
    plt.xticks(x, labels, rotation=90)
    plt.tight_layout(); plt.savefig(save_path); plt.close()

def _compute_mmd_matrix(features: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """Compute the pairwise MMD distance matrix between subjects.

    Parameters
    ----------
    features : dict of {str: ndarray}
        Mapping from subject ID to feature matrix.

    Returns
    -------
    matrix : ndarray of shape (n_subjects, n_subjects)
        Symmetric MMD distance matrix.
    subjects : list of str
        Subject identifiers corresponding to the rows/columns.
    """
    valid_subjects = list(features.keys())
    n = len(valid_subjects)
    M = np.zeros((n, n), dtype=np.float64)
    for i in tqdm(range(n), desc="MMD matrix (symmetric)"):
        Xi = features[valid_subjects[i]]
        for j in range(i, n):
            if i == j:
                val = 0.0
            else:
                Xj = features[valid_subjects[j]]
                if Xi.shape[1] != Xj.shape[1]:
                    raise ValueError(f"Feature dim mismatch: {valid_subjects[i]} vs {valid_subjects[j]}")
                val = compute_mmd(Xi, Xj)
            M[i, j] = M[j, i] = val
    np.fill_diagonal(M, 0.0)
    return M, valid_subjects

def _compute_group_dist_matrix(matrix: np.ndarray, subjects: list[str], groups: dict[str, list[str]]) -> tuple[np.ndarray, list[str]]:
    """Compute average distances between groups.

    Parameters
    ----------
    matrix : ndarray of shape (n_subjects, n_subjects)
        Pairwise subject distance matrix.
    subjects : list of str
        Subject identifiers corresponding to the matrix.
    groups : dict of {str: list of str}
        Mapping from group names to subject IDs.

    Returns
    -------
    group_matrix : ndarray of shape (n_groups, n_groups)
        Symmetric matrix of mean distances between groups.
    group_names : list of str
        Names of the groups.
    """
    s2i = {s: i for i, s in enumerate(subjects)}
    gnames = list(groups.keys()); n = len(gnames)
    G = np.zeros((n, n))
    gidx = {g: [s2i[s] for s in members if s in s2i] for g, members in groups.items()}
    for i in range(n):
        for j in range(n):
            A, B = gidx[gnames[i]], gidx[gnames[j]]
            if not A or not B:
                G[i, j] = np.nan; continue
            if i == j:
                vals = [matrix[p, q] for p in A for q in A if p < q]
            else:
                vals = [matrix[p, q] for p in A for q in B]
            G[i, j] = float(np.mean(vals)) if vals else np.nan
    return G, gnames

def _compute_group_centroids_from_distance_matrix(matrix: np.ndarray, subjects: list[str], groups: dict[str, list[str]]) -> dict[str, np.ndarray]:
    """Compute 2D centroids of groups from a distance matrix using MDS.

    Parameters
    ----------
    matrix : ndarray of shape (n_subjects, n_subjects)
        Pairwise subject distance matrix.
    subjects : list of str
        Subject identifiers.
    groups : dict of {str: list of str}
        Mapping from group names to subject IDs.

    Returns
    -------
    centroids : dict of {str: ndarray of shape (2,)}
        Mapping of group names to their centroid coordinates in 2D space.
    """
    coords = MDS(n_components=2, dissimilarity="precomputed", random_state=42)\
        .fit_transform(np.nan_to_num(matrix))
    s2xy = {s: coords[i] for i, s in enumerate(subjects)}
    centroids: dict[str, np.ndarray] = {}
    for gname, members in groups.items():
        pts = [s2xy[s] for s in members if s in s2xy]
        if pts:
            centroids[gname] = np.mean(np.stack(pts), axis=0)
    return centroids

def _compute_group_centroid_distances(centroids: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """Compute pairwise distances between group centroids.

    Parameters
    ----------
    centroids : dict of {str: ndarray of shape (2,)}
        Mapping of group names to centroid coordinates.

    Returns
    -------
    distances : ndarray of shape (n_groups, n_groups)
        Symmetric matrix of centroid-to-centroid distances.
    group_names : list of str
        Names of the groups.
    """
    gnames = list(centroids.keys()); n = len(gnames)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = float(np.linalg.norm(centroids[gnames[i]] - centroids[gnames[j]]))
    return D, gnames

def _compute_intra_group_variability(matrix: np.ndarray, subjects: list[str], groups: dict[str, list[str]]) -> dict[str, dict[str, float]]:
    """Compute within-group variability for each group.

    Parameters
    ----------
    matrix : ndarray of shape (n_subjects, n_subjects)
        Pairwise subject distance matrix.
    subjects : list of str
        Subject identifiers.
    groups : dict of {str: list of str}
        Mapping from group names to subject IDs.

    Returns
    -------
    variability : dict of {str: dict[str, float]}
        Mapping from group name to statistics:
        - 'mean': mean within-group distance
        - 'std': standard deviation of within-group distances
    """
    s2i = {s: i for i, s in enumerate(subjects)}
    out: dict[str, dict[str, float]] = {}
    for gname, members in groups.items():
        idx = [s2i[s] for s in members if s in s2i]
        if len(idx) < 2:
            out[gname] = {"mean": np.nan, "std": np.nan}; continue
        dists = [matrix[i, j] for i in idx for j in idx if i < j]
        arr = np.array(dists, dtype=float)
        out[gname] = {"mean": float(np.nanmean(arr)) if arr.size else np.nan,
                      "std":  float(np.nanstd(arr))  if arr.size else np.nan}
    return out

def _compute_intra_inter_stats(matrix: np.ndarray, subjects: list[str], groups: dict[str, list[str]]) -> dict[str, dict[str, float]]:
    """Compute intra-group and inter-group statistics for each group.

    Parameters
    ----------
    matrix : ndarray of shape (n_subjects, n_subjects)
        Pairwise subject distance matrix.
    subjects : list of str
        Subject identifiers.
    groups : dict of {str: list of str}
        Mapping from group names to subject IDs.

    Returns
    -------
    stats : dict of {str: dict[str, float]}
        Mapping from group name to statistics:
        - 'intra_mean': mean intra-group distance
        - 'intra_std': standard deviation of intra-group distances
        - 'inter_mean': mean distance to subjects outside the group
        - 'inter_std': standard deviation of inter-group distances
    """
    s2i = {s: i for i, s in enumerate(subjects)}
    stats: dict[str, dict[str, float]] = {}
    for gname, gmembers in groups.items():
        idx_in  = [s2i[s] for s in gmembers if s in s2i]
        idx_out = [i for i in range(len(subjects)) if i not in idx_in]
        intra_vals = np.array([matrix[i, j] for i in idx_in for j in idx_in if i < j], dtype=float)
        inter_vals = np.array([matrix[i, j] for i in idx_in for j in idx_out], dtype=float)
        stats[gname] = {
            "intra_mean": float(np.nanmean(intra_vals)) if intra_vals.size else np.nan,
            "intra_std":  float(np.nanstd(intra_vals))  if intra_vals.size else np.nan,
            "inter_mean": float(np.nanmean(inter_vals)) if inter_vals.size else np.nan,
            "inter_std":  float(np.nanstd(inter_vals))  if inter_vals.size else np.nan,
        }
    return stats

def _plot_intra_inter(stats: dict[str, dict[str, float]], dist_name: str, save_path: Path) -> None:
    """Plot a comparison of intra- vs inter-group distances.

    Parameters
    ----------
    stats : dict of {str: dict[str, float]}
        Mapping from group names to intra/inter distance statistics.
    dist_name : str
        Name of the distance type (e.g., 'MMD', 'Wasserstein', 'DTW').
    save_path : Path
        Path to save the plot.

    Returns
    -------
    None
        Saves the intra/inter comparison plot to the specified file path.
    """
    gnames = list(stats.keys())
    intra_means = np.array([stats[g]["intra_mean"] for g in gnames])
    intra_stds  = np.array([stats[g]["intra_std"]  for g in gnames])
    inter_means = np.array([stats[g]["inter_mean"] for g in gnames])
    inter_stds  = np.array([stats[g]["inter_std"]  for g in gnames])
    x = np.arange(len(gnames)); width = 0.35
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, intra_means, width, yerr=intra_stds, label="Intra", capsize=5)
    plt.bar(x + width/2, inter_means, width, yerr=inter_stds, label="Inter", capsize=5)
    plt.xticks(x, gnames, rotation=45); plt.ylabel("Distance")
    plt.title(f"Intra vs Inter Group Distance ({dist_name})")
    plt.legend(); plt.tight_layout(); plt.savefig(save_path); plt.close()



def _pairwise_symmetric_matrix(subjects, compute_func, desc: str, n_jobs: int = -1) -> np.ndarray:
    """Compute symmetric pairwise matrix for any distance function, using parallel processing.

    Parameters
    ----------
    subjects : list[str]
        Subject identifiers.
    compute_func : callable
        Function taking (subject_i, subject_j) → distance (float).
    desc : str
        Description for progress display.
    n_jobs : int, default=-1
        Number of parallel workers (-1 = use all cores).
    """
    n = len(subjects)
    pairs = [(i, j) for i in range(n) for j in range(i, n)]

    def safe_compute(i, j):
        if i == j:
            return (i, j, 0.0)
        try:
            val = compute_func(subjects[i], subjects[j])
        except Exception as e:
            print(f"[WARN] {desc}: {subjects[i]} vs {subjects[j]} -> {e}")
            val = np.nan
        return (i, j, val)

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(safe_compute)(i, j) for i, j in tqdm(pairs, desc=f"{desc} (parallel)")
    )

    M = np.zeros((n, n), dtype=np.float64)
    for i, j, val in results:
        M[i, j] = M[j, i] = val

    np.fill_diagonal(M, 0.0)
    return M


def _compute_distance_matrix(features: dict[str, np.ndarray], metric: str) -> tuple[np.ndarray, list[str]]:
    """Compute pairwise distance matrix for MMD, Wasserstein, or DTW."""
    # To extend: Add new metrics (e.g., cosine, energy, correlation)
    # by adding another elif metric == "<new_metric>": block.
    subjects = list(features.keys())
    metric = metric.lower()

    if metric == "mmd":
        def func(si, sj):
            Xi, Xj = features[si], features[sj]
            if Xi.shape[1] != Xj.shape[1]:
                raise ValueError(f"Feature dim mismatch: {si} vs {sj}")
            return compute_mmd(Xi, Xj)

    elif metric == "wasserstein":
        MAX_DIST = 1e6
        def func(si, sj):
            Xi, Xj = features[si], features[sj]
            common_cols = min(Xi.shape[1], Xj.shape[1])
            dists = []
            for k in range(common_cols):
                d = wasserstein_distance(Xi[:, k], Xj[:, k])
                if np.isfinite(d):
                    dists.append(min(float(d), MAX_DIST))
            return float(np.mean(dists)) if dists else np.nan

    elif metric == "dtw":
        mean_series = {s: features[s].mean(axis=1) for s in subjects}
        MAX_DIST = 1e6
        def func(si, sj):
            d = dtw(mean_series[si], mean_series[sj])
            return float(d) if np.isfinite(d) and d < MAX_DIST else np.nan

    else:
        raise ValueError(f"Unsupported metric: {metric}")

    M = _pairwise_symmetric_matrix(subjects, func, desc=metric.upper())
    return M, subjects


# === Unified Group Analysis Helpers ===

def _compute_group_analysis(matrix: np.ndarray, subjects: list[str], groups: dict[str, list[str]]) -> dict:
    """Compute multiple group-level statistics in a unified structure.

    Parameters
    ----------
    matrix : np.ndarray
        Pairwise distance matrix among all subjects.
    subjects : list of str
        Subject identifiers corresponding to matrix rows/columns.
    groups : dict
        Group mapping {group_name: [subject_ids]}.

    Returns
    -------
    dict
        {
            "group_matrix": <np.ndarray>,
            "group_names": <list[str]>,
            "centroids": <dict[str, np.ndarray]>,
            "centroid_matrix": <np.ndarray>,
            "centroid_names": <list[str]>,
            "intra": <dict[str, dict[str, float]]>,
            "intra_inter_stats": <dict>
        }
    """
    out = {}

    # === Group Distance Matrix ===
    group_matrix, group_names = _compute_group_dist_matrix(matrix, subjects, groups)
    out["group_matrix"] = group_matrix
    out["group_names"] = group_names

    # === Group Centroid Analysis ===
    centroids = _compute_group_centroids_from_distance_matrix(matrix, subjects, groups)
    if centroids:
        centroid_matrix, centroid_names = _compute_group_centroid_distances(centroids)
        out["centroids"] = centroids
        out["centroid_matrix"] = centroid_matrix
        out["centroid_names"] = centroid_names

    # === Intra-group Variability ===
    intra = _compute_intra_group_variability(matrix, subjects, groups)
    out["intra"] = intra

    # === Intra/Inter Statistics ===
    intra_inter_stats = _compute_intra_inter_stats(matrix, subjects, groups)
    out["intra_inter_stats"] = intra_inter_stats

    return out


def _save_group_analysis_results(base_dir: Path, metric: str, analysis: dict):
    """Save and visualize all group-level results in consistent directory layout."""
    group_dir = base_dir / "distances" / metric
    group_dir.mkdir(parents=True, exist_ok=True)

    # (1) Group matrix heatmap
    np.save(group_dir / "group_matrix.npy", analysis["group_matrix"])
    (group_dir / "group_names.json").write_text(json.dumps(analysis["group_names"]))
    _plot_heatmap(analysis["group_matrix"], analysis["group_names"], analysis["group_names"],
                  f"{metric.upper()} Distance Between Groups",
                  group_dir / "group_heatmap.png")

    # (2) Centroid distances
    if "centroid_matrix" in analysis:
        _plot_heatmap(analysis["centroid_matrix"], analysis["centroid_names"], analysis["centroid_names"],
                      f"{metric.upper()}: Group Centroid Distance",
                      group_dir / "group_centroid_distance_heatmap.png")

    # (3) Intra-group variability
    intra_dir = group_dir / "intra"; intra_dir.mkdir(exist_ok=True)
    intra = analysis["intra"]
    (intra_dir / "intra_group_variability.json").write_text(json.dumps(intra, indent=2))
    pd.DataFrame(intra).T.to_csv(intra_dir / "intra_group_variability.csv")
    _plot_bar(list(intra.keys()),
              np.array([intra[g]["mean"] for g in intra]),
              np.array([intra[g]["std"] for g in intra]),
              f"Intra-group Variability - {metric.upper()}", "Mean Distance",
              intra_dir / "intra_variability.png")

    # (4) Intra/Inter Comparison
    intra_inter_dir = group_dir / "intra_inter"; intra_inter_dir.mkdir(exist_ok=True)
    _plot_intra_inter(analysis["intra_inter_stats"], metric.upper(),
                      intra_inter_dir / "intra_inter_comparison.png")



# ===== Orchestrator =====
def run_comp_dist(
    subject_list_path: str = "dataset/mdapbe/subject_list.txt",
    data_root: str = "data/processed/common",
    groups_file: str = "config/target_groups.txt",
    metric: str = "all",
) -> int:
    """Run the full computation pipeline for subject/group distances.

    This orchestrator function performs the following steps:

    1. Load subject list and extract features.
    2. Normalize features across subjects.
    3. Compute subject-level distance matrices:
    
       - MMD (Maximum Mean Discrepancy)
       - Wasserstein distance
       - Dynamic Time Warping (DTW)

    4. Save results (NPY, JSON, PNG).
    5. Compute group-level summaries:
    
       - Group distance matrices
       - Group centroid distances
       - Intra/inter-group variability

    6. Generate plots (heatmaps, bar charts, projections).

    Parameters
    ----------
    subject_list_path : str, default="../../dataset/mdapbe/subject_list.txt"
        Path to subject list file.
    data_root : str, default="data/processed/common"
        Directory containing processed feature CSVs.
    groups_file : str, default="../misc/target_groups.txt"
        Path to group definitions file.

    Returns
    -------
    int
        Return code (0 = success).
    """
    # reproducibility for median-heuristic subsampling
    np.random.seed(42)

    # === Load subjects and extract + normalize features ===
    subjects = _load_subject_list(Path(subject_list_path))
    features = _extract_features_with_cache(subjects, Path(data_root))

    # === Global z-score normalization across all subjects ===
    if features:
        all_X = np.vstack(list(features.values()))
        mu = all_X.mean(axis=0, keepdims=True)
        sigma = all_X.std(axis=0, keepdims=True) + 1e-12
        features = {k: ((v - mu) / sigma).astype(np.float32, copy=False)
                    for k, v in features.items()}

    # === Unified computation & output ===
    if metric == "all":
        metrics = ["mmd", "wasserstein", "dtw"]
    else:
        metrics = [metric]
    base_dir = Path("results/domain_analysis/distance")
    groups = _load_groups(Path(groups_file))

    for metric in metrics:
        print(f"[INFO] Computing {metric.upper()} distance matrix... ({len(features)} subjects)")
        # --- ensure output directories exist ---
        metric_dir = base_dir / metric
        metric_dir.mkdir(parents=True, exist_ok=True)
        (base_dir / "distances").mkdir(parents=True, exist_ok=True)

        # --- compute and save subject-level distance matrix ---
        matrix, subjects_valid = _compute_distance_matrix(features, metric)
        np.save(metric_dir / f"{metric}_matrix.npy", matrix)
        (metric_dir / f"{metric}_subjects.json").write_text(json.dumps(subjects_valid))
        _plot_heatmap_auto(matrix, subjects_valid, metric, "subject", metric_dir)

        # ensure saved npy path exists before writing
        Path(metric_dir).mkdir(parents=True, exist_ok=True)

        # === Subject-level summary ===
        n = matrix.shape[0]
        mask = ~np.eye(n, dtype=bool)
        masked = np.where(mask, matrix, np.nan)
        means = np.nanmean(masked, axis=1)
        stds  = np.nanstd(masked, axis=1)

        means_for_sort = np.nan_to_num(means, nan=-np.inf)
        sorted_idx = np.argsort(-means_for_sort)
        subj_sorted = [subjects_valid[i] for i in sorted_idx]

        np.save(metric_dir / f"{metric}_mean.npy", means)
        np.save(metric_dir / f"{metric}_std.npy", stds)
        np.save(metric_dir / f"{metric}_mean_sorted.npy", means[sorted_idx])
        np.save(metric_dir / f"{metric}_std_sorted.npy", stds[sorted_idx])
        (metric_dir / f"{metric}_subjects_sorted.json").write_text(json.dumps(subj_sorted))
        _plot_bar_auto(subj_sorted, means[sorted_idx], stds[sorted_idx],
                       metric, "subjects", metric_dir)

        # === Group-level analyses ===
        group_analysis = _compute_group_analysis(matrix, subjects_valid, groups)
        _save_group_analysis_results(base_dir, metric, group_analysis)

        # === Unified projection visualization ===
        _plot_projection_auto(matrix, subjects_valid, groups,
                              metric, metric_dir)
 

    print("[DONE] All distance computations and analyses complete.")
    return 0
