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

CACHE_VERSION = "v1"  # bump this when feature layout/filters change

def _cache_key(subjects, data_root: Path) -> str:
#    s = "|".join(subjects) + "@" + str(data_root.resolve())
    s = CACHE_VERSION + "|" + "|".join(subjects) + "@" + str(data_root.resolve())
    return hashlib.md5(s.encode()).hexdigest()

def _extract_features_with_cache(subjects, data_root: Path, cache_dir: Path = Path("results/.cache")):
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(subjects, data_root)
    npz = cache_dir / f"features_{key}.npz"
    if npz.exists():
        z = np.load(npz, allow_pickle=True)
        return {k: z[k] for k in z.files}
    feats = _extract_features(subjects, data_root)
    np.savez_compressed(npz, **feats)
    return feats

# Limit BLAS threads
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ===== Helpers =====
def _median_gamma(X: np.ndarray, Y: np.ndarray, max_samples: int = 1000) -> float:
    """Estimate RBF gamma via the median heuristic on squared L2 distances."""
    A = X if len(X) <= max_samples else X[np.random.choice(len(X), max_samples, replace=False)]
    B = Y if len(Y) <= max_samples else Y[np.random.choice(len(Y), max_samples, replace=False)]
    D = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=2)
    med = float(np.median(D))
    if not np.isfinite(med) or med <= 0:
        return 1.0 / X.shape[1]
    return 1.0 / (2.0 * med)

def compute_mmd(X: np.ndarray, Y: np.ndarray, gamma: float | None = None) -> float:
    """Biased MMD^2 estimate with RBF kernel (keeps previous behavior)."""
    if gamma is None:
        gamma = _median_gamma(X, Y)
    Kxx = rbf_kernel(X, X, gamma=gamma)
    Kyy = rbf_kernel(Y, Y, gamma=gamma)
    Kxy = rbf_kernel(X, Y, gamma=gamma)
    m, n = len(X), len(Y)
    return float(Kxx.sum() / (m * m) + Kyy.sum() / (n * n) - 2.0 * Kxy.sum() / (m * n))

def _load_features_one(subject_str: str, data_root: Path) -> np.ndarray | None:
    """Load a subject's feature matrix from processed CSV."""
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
    out: dict[str, np.ndarray] = {}
    for s in subjects:
        x = _load_features_one(s, data_root)
        if x is not None:
            out[s] = x
    return out

def _load_subject_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]

def _load_groups(path: Path) -> dict[str, list[str]]:
    rows = [line.strip().split() for line in path.read_text().splitlines() if line.strip()]
    return {f"G{idx+1}": group for idx, group in enumerate(rows)}

def _plot_heatmap(matrix: np.ndarray, row_labels: list[str], col_labels: list[str],
                  title: str, save_path: Path, annot: bool = True, fmt: str = ".2f") -> None:
    plt.figure(figsize=(max(8, len(row_labels)//2), max(6, len(col_labels)//2)))
    sns.heatmap(matrix, xticklabels=col_labels, yticklabels=row_labels,
                cmap="viridis", annot=annot, fmt=fmt)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def _plot_bar(labels: list[str], means: np.ndarray, stds: np.ndarray,
              title: str, ylabel: str, save_path: Path) -> None:
    x = np.arange(len(labels))
    plt.figure(figsize=(max(12, len(labels)//2), 6))
    plt.bar(x, means, yerr=stds, capsize=5)
    plt.title(title); plt.ylabel(ylabel)
    plt.xticks(x, labels, rotation=90)
    plt.tight_layout(); plt.savefig(save_path); plt.close()

def _plot_group_projection(matrix: np.ndarray, subjects: list[str], groups: dict[str, list[str]],
                           method_name: str, save_path: Path) -> None:
    coords = MDS(n_components=2, dissimilarity="precomputed", random_state=42)\
        .fit_transform(np.nan_to_num(matrix))
    s2xy = {s: coords[i] for i, s in enumerate(subjects)}
    plt.figure(figsize=(10, 8))
    for gname, members in groups.items():
        pts = [(s2xy[s], s) for s in members if s in s2xy]
        if not pts: continue
        arr = np.array([p[0] for p in pts])
        plt.scatter(arr[:, 0], arr[:, 1], label=gname)
        for (xy, sid) in pts:
            plt.text(xy[0], xy[1], sid, fontsize=6)
    plt.title(f"{method_name}: Subject Group Distribution")
    plt.legend(); plt.tight_layout(); plt.savefig(save_path); plt.close()

def _compute_mmd_matrix(features: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
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
    gnames = list(centroids.keys()); n = len(gnames)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = float(np.linalg.norm(centroids[gnames[i]] - centroids[gnames[j]]))
    return D, gnames

def _compute_intra_group_variability(matrix: np.ndarray, subjects: list[str], groups: dict[str, list[str]]) -> dict[str, dict[str, float]]:
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

# ===== Orchestrator =====
def run_comp_dist(
    subject_list_path: str = "../../dataset/mdapbe/subject_list.txt",
    data_root: str = "data/processed/common",
    out_mmd_dir: str = "results/mmd",
    out_dist_dir: str = "results/distances",
    groups_file: str = "../misc/target_groups.txt",
) -> int:
    """Re-implementation of legacy bin/comp_dist.py with parametrized I/O."""
    # reproducibility for median-heuristic subsampling
    np.random.seed(42)
    subjects = _load_subject_list(Path(subject_list_path))
    #features = _extract_features(subjects, Path(data_root))
    features = _extract_features_with_cache(subjects, Path(data_root))

    # Global z-score across all subjects
    if features:
        all_X = np.vstack(list(features.values()))
        mu = all_X.mean(axis=0, keepdims=True)
        sigma = all_X.std(axis=0, keepdims=True) + 1e-12
#        features = {k: (v - mu) / sigma for k, v in features.items()}
        features = {k: ((v - mu) / sigma).astype(np.float32, copy=False) for k, v in features.items()}

    # MMD
    out_mmd = Path(out_mmd_dir); out_mmd.mkdir(parents=True, exist_ok=True)
    mmd_matrix, valid_subjects = _compute_mmd_matrix(features)
    np.save(out_mmd / "mmd_matrix.npy", mmd_matrix)
    (out_mmd / "mmd_subjects.json").write_text(json.dumps(valid_subjects))
    _plot_heatmap(mmd_matrix, valid_subjects, valid_subjects,
                  "MMD Distance Matrix", out_mmd / "mmd_matrix.png",
                  annot=False, fmt=".2f")

    # Wasserstein / DTW
    out_dist = Path(out_dist_dir); out_dist.mkdir(parents=True, exist_ok=True)
    n = len(valid_subjects)
    wass_matrix = np.zeros((n, n), dtype=np.float64)
    dtw_matrix  = np.zeros((n, n), dtype=np.float64)
    mean_series = [features[s].mean(axis=1) for s in valid_subjects]
    MAX_DIST = 1e6

    for i in tqdm(range(n), desc="Wass/DTW (symmetric)"):
        Xi = features[valid_subjects[i]]
        for j in range(i, n):
            if i == j:
                wass_val = 0.0; dtw_val = 0.0
            else:
                Xj = features[valid_subjects[j]]
                common_cols = min(Xi.shape[1], Xj.shape[1])
                w_dists = []
                for k in range(common_cols):
                    d = wasserstein_distance(Xi[:, k], Xj[:, k])
                    if np.isfinite(d):
                        w_dists.append(min(float(d), MAX_DIST))
                wass_val = float(np.mean(w_dists)) if w_dists else np.nan
                try:
                    d = dtw(mean_series[i], mean_series[j])
                    dtw_val = float(d) if np.isfinite(d) and d < MAX_DIST else np.nan
                except Exception as e:
                    print(f"[DTW warn] {valid_subjects[i]} vs {valid_subjects[j]} -> {e}")
                    dtw_val = np.nan
            wass_matrix[i, j] = wass_matrix[j, i] = wass_val
            dtw_matrix[i, j]  = dtw_matrix[j, i]  = dtw_val

    np.fill_diagonal(wass_matrix, 0.0); np.fill_diagonal(dtw_matrix, 0.0)
    np.save(out_dist / "wasserstein_matrix.npy", wass_matrix)
    np.save(out_dist / "dtw_matrix.npy", dtw_matrix)
    (out_dist / "subjects.json").write_text(json.dumps(valid_subjects))
    _plot_heatmap(wass_matrix, valid_subjects, valid_subjects,
                  "Wasserstein Distance Matrix", out_dist / "wasserstein_matrix.png",
                  annot=False, fmt=".2f")
    _plot_heatmap(dtw_matrix, valid_subjects, valid_subjects,
                  "DTW Distance Matrix", out_dist / "dtw_matrix.png",
                  annot=False, fmt=".2f")

    # Group summaries
    groups = _load_groups(Path(groups_file))
    distance_types = {
        "mmd":         (out_mmd / "mmd_matrix.npy",          out_mmd / "mmd_subjects.json"),
        "wasserstein": (out_dist / "wasserstein_matrix.npy", out_dist / "subjects.json"),
        "dtw":         (out_dist / "dtw_matrix.npy",         out_dist / "subjects.json"),
    }

    for dist_name, (matrix_path, subject_path) in distance_types.items():
        matrix = np.load(matrix_path)
        subjs = json.loads(Path(subject_path).read_text())

        n = matrix.shape[0]
        mask = ~np.eye(n, dtype=bool)
        masked = np.where(mask, matrix, np.nan)
        means = np.nanmean(masked, axis=1); stds = np.nanstd(masked, axis=1)
        means_for_sort = np.nan_to_num(means, nan=-np.inf)
        sorted_idx = np.argsort(-means_for_sort)
        subj_sorted = [subjs[i] for i in sorted_idx]

        save_dir = Path(f"results/{dist_name.lower()}"); save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / f"{dist_name.lower()}_mean.npy", means)
        np.save(save_dir / f"{dist_name.lower()}_std.npy", stds)
        np.save(save_dir / f"{dist_name.lower()}_mean_sorted.npy", means[sorted_idx])
        np.save(save_dir / f"{dist_name.lower()}_std_sorted.npy", stds[sorted_idx])
        (save_dir / f"{dist_name.lower()}_subjects_sorted.json").write_text(json.dumps(subj_sorted))
        _plot_bar(subj_sorted, means[sorted_idx], stds[sorted_idx],
                  f"Mean and StdDev of {dist_name} per Subject (Sorted)",
                  dist_name, save_dir / f"{dist_name.lower()}_mean_std_sorted.png")

        group_matrix, group_names = _compute_group_dist_matrix(matrix, subjs, groups)
        group_dir = Path(f"results/group_distances/{dist_name.lower()}"); group_dir.mkdir(parents=True, exist_ok=True)
        np.save(group_dir / "group_matrix.npy", group_matrix)
        (group_dir / "group_names.json").write_text(json.dumps(group_names))
        _plot_heatmap(group_matrix, group_names, group_names,
                      f"{dist_name} Distance Between Groups",
                      group_dir / "group_heatmap.png")

        _plot_group_projection(matrix, subjs, groups, dist_name,
                               save_dir / "group_overlay_projection.png")

        centroids = _compute_group_centroids_from_distance_matrix(matrix, subjs, groups)
        if centroids:
            centroid_matrix, centroid_names = _compute_group_centroid_distances(centroids)
            _plot_heatmap(centroid_matrix, centroid_names, centroid_names,
                          f"{dist_name}: Group Centroid Distance",
                          save_dir / "group_centroid_distance_heatmap.png")

        intra = _compute_intra_group_variability(matrix, subjs, groups)
        intra_dir = group_dir / "intra"; intra_dir.mkdir(parents=True, exist_ok=True)
        (intra_dir / "intra_group_variability.json").write_text(json.dumps(intra, indent=2))
        pd.DataFrame(intra).T.to_csv(intra_dir / "intra_group_variability.csv")
        _plot_bar(list(intra.keys()),
                  np.array([intra[g]["mean"] for g in intra]),
                  np.array([intra[g]["std"] for g in intra]),
                  f"Intra-group Variability - {dist_name}", "Mean Distance",
                  intra_dir / "intra_variability.png")

        stats = _compute_intra_inter_stats(matrix, subjs, groups)
        intra_inter_dir = group_dir / "intra_inter"; intra_inter_dir.mkdir(parents=True, exist_ok=True)
        _plot_intra_inter(stats, dist_name, intra_inter_dir / "intra_inter_comparison.png")

    return 0

