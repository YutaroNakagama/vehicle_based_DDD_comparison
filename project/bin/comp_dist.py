import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from scipy.stats import wasserstein_distance
from tslearn.metrics import dtw
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.manifold import MDS
from tqdm import tqdm

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

def _median_gamma(X, Y, max_samples=1000):
    """Estimate RBF gamma via median heuristic on squared L2 distances."""
    A = X if len(X) <= max_samples else X[np.random.choice(len(X), max_samples, replace=False)]
    B = Y if len(Y) <= max_samples else Y[np.random.choice(len(Y), max_samples, replace=False)]
    D = np.sum((A[:, None, :] - B[None, :, :])**2, axis=2)  # squared L2
    med = np.median(D)
    if not np.isfinite(med) or med <= 0:
        return 1.0 / X.shape[1]
    return 1.0 / (2.0 * med)

def compute_mmd(X, Y, gamma=None):
    """Compute MMD (Maximum Mean Discrepancy) between two feature matrices."""
    if gamma is None:
        gamma = _median_gamma(X, Y)  # <- changed from 1.0 / X.shape[1]
    Kxx = rbf_kernel(X, X, gamma=gamma)
    Kyy = rbf_kernel(Y, Y, gamma=gamma)
    Kxy = rbf_kernel(X, Y, gamma=gamma)
    m, n = len(X), len(Y)
    return Kxx.sum() / (m * m) + Kyy.sum() / (n * n) - 2 * Kxy.sum() / (m * n)

def load_features(subject_str):
    subject_id, version = subject_str.split("_")
    path = f"data/processed/common/processed_{subject_id}_{version}.csv"
    if not os.path.exists(path):
        print(f"Missing file: {path}")
        return None
    df = pd.read_csv(path)
    feature_cols = [
        col for col in df.columns
        if col not in ["Timestamp", "KSS", "subject"]
        and not col.startswith("Channel_")
        and not col.startswith("KSS_")
        and not col.startswith("theta_alpha_over_beta")
    ]
    return df[feature_cols].dropna().values

def extract_features(subjects):
    out = {}
    for s in subjects:
        x = load_features(s)
        if x is not None:
            out[s] = x
    return out

def load_subject_list(path):
    with open(path) as f:
        return [line.strip() for line in f.readlines()]

def load_json(path):
    with open(path) as f:
        return json.load(f)

def load_groups(path):
    with open(path) as f:
        subjects = [line.strip().split() for line in f.readlines()]
    return {f"G{idx+1}": group for idx, group in enumerate(subjects)}

def compute_mmd_matrix(features):
    valid_subjects = list(features.keys())
    n = len(valid_subjects)
    mmd_matrix = np.zeros((n, n), dtype=np.float64)
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
            mmd_matrix[i, j] = val
            mmd_matrix[j, i] = val
    np.fill_diagonal(mmd_matrix, 0.0)
    return mmd_matrix, valid_subjects

def plot_heatmap(matrix, row_labels, col_labels, title, save_path, annot=True, fmt=".2f"):
    plt.figure(figsize=(max(8, len(row_labels)//2), max(6, len(col_labels)//2)))
    sns.heatmap(matrix, xticklabels=col_labels, yticklabels=row_labels, cmap="viridis", annot=annot, fmt=fmt)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_bar(labels, means, stds, title, ylabel, save_path):
    plt.figure(figsize=(max(12, len(labels)//2), 6))
    plt.bar(labels, means, yerr=stds, capsize=5)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_group_projection(matrix, subjects, groups, method_name, save_path):
    coords = MDS(n_components=2, dissimilarity="precomputed", random_state=42).fit_transform(np.nan_to_num(matrix))
    subj_to_coords = {s: coords[i] for i, s in enumerate(subjects)}

    plt.figure(figsize=(10, 8))
    for gname, members in groups.items():
        pairs = [(subj_to_coords[s], s) for s in members if s in subj_to_coords]
        if not pairs:
            continue
        arr = np.array([p[0] for p in pairs])
        plt.scatter(arr[:, 0], arr[:, 1], label=gname)
        for (xy, sid) in pairs:
            plt.text(xy[0], xy[1], sid, fontsize=6)

    plt.title(f"{method_name}: Subject Group Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def compute_group_dist_matrix(matrix, subjects, groups):
    subj_to_idx = {s: i for i, s in enumerate(subjects)}
    group_names = list(groups.keys())
    n = len(group_names)
    group_matrix = np.zeros((n, n))

    # Precompute index lists per group
    group_idxs = {
        g: [subj_to_idx[s] for s in members if s in subj_to_idx]
        for g, members in groups.items()
    }

    for i in range(n):
        for j in range(n):
            A = group_idxs[group_names[i]]
            B = group_idxs[group_names[j]]
            if not A or not B:
                group_matrix[i, j] = np.nan
                continue

            if i == j:
                # within-group: off-diagonal only (p < q)
                vals = [matrix[p, q] for p in A for q in A if p < q]
            else:
                # between-groups: all cross pairs
                vals = [matrix[p, q] for p in A for q in B]

            group_matrix[i, j] = np.mean(vals) if len(vals) > 0 else np.nan

    return group_matrix, group_names

def compute_group_centroids_from_distance_matrix(matrix, subjects, groups):
    coords = MDS(n_components=2, dissimilarity='precomputed', random_state=42).fit_transform(np.nan_to_num(matrix))
    subj_to_coords = {s: coords[i] for i, s in enumerate(subjects)}
    centroids = {gname: np.mean([subj_to_coords[s] for s in members if s in subj_to_coords], axis=0)
                 for gname, members in groups.items() if [subj_to_coords[s] for s in members if s in subj_to_coords]}
    return centroids

def compute_group_centroid_distances(centroids):
    group_names = list(centroids.keys())
    n = len(group_names)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = np.linalg.norm(centroids[group_names[i]] - centroids[group_names[j]])
    return dist_matrix, group_names

def compute_intra_group_variability(matrix, subjects, groups):
    subj_to_idx = {s: i for i, s in enumerate(subjects)}
    results = {}
    for group_name, group_members in groups.items():
        indices = [subj_to_idx[s] for s in group_members if s in subj_to_idx]
        if len(indices) < 2:
            results[group_name] = {"mean": np.nan, "std": np.nan}
            continue
        dists = [matrix[i, j] for i in indices for j in indices if i < j]
        arr = np.array(dists, dtype=float)
        results[group_name] = {
            "mean": float(np.nanmean(arr)) if arr.size else np.nan,
            "std":  float(np.nanstd(arr))  if arr.size else np.nan,
        }
    return results

def compute_intra_inter_stats(matrix, subjects, groups):
    subj_to_idx = {s: i for i, s in enumerate(subjects)}
    stats = {}
    for gname, gmembers in groups.items():
        idx_in = [subj_to_idx[s] for s in gmembers if s in subj_to_idx]
        idx_out = [i for i in range(len(subjects)) if i not in idx_in]
        #intra_vals = np.array([matrix[i, j] for i in idx_in for j in idx_in if i != j], dtype=float)
        intra_vals = np.array([matrix[i, j] for i in idx_in for j in idx_in if i < j], dtype=float)
        inter_vals = np.array([matrix[i, j] for i in idx_in for j in idx_out], dtype=float)
        stats[gname] = {
            "intra_mean": float(np.nanmean(intra_vals)) if intra_vals.size else np.nan,
            "intra_std":  float(np.nanstd(intra_vals))  if intra_vals.size else np.nan,
            "inter_mean": float(np.nanmean(inter_vals)) if inter_vals.size else np.nan,
            "inter_std":  float(np.nanstd(inter_vals))  if inter_vals.size else np.nan,
        }
    return stats

def plot_intra_inter(stats, dist_name, save_path):
    group_names = list(stats.keys())
    intra_means = [stats[g]["intra_mean"] for g in group_names]
    intra_stds = [stats[g]["intra_std"] for g in group_names]
    inter_means = [stats[g]["inter_mean"] for g in group_names]
    inter_stds = [stats[g]["inter_std"] for g in group_names]
    x = np.arange(len(group_names))
    width = 0.35
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, intra_means, width, yerr=intra_stds, label="Intra", capsize=5)
    plt.bar(x + width/2, inter_means, width, yerr=inter_stds, label="Inter", capsize=5)
    plt.xticks(x, group_names, rotation=45)
    plt.ylabel("Distance")
    plt.title(f"Intra vs Inter Group Distance ({dist_name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    np.random.seed(42)
    subject_list_path = "../../dataset/mdapbe/subject_list.txt"
    out_dir = "results/mmd"
    # 1. Subject List
    subjects = load_subject_list(subject_list_path)
    # 2. Feature Extraction
    features = extract_features(subjects)
    # --- Optional but recommended: z-score normalization across all subjects ---
    # This makes Wasserstein distances less sensitive to raw feature scales.
    if len(features) > 0:
        all_X = np.vstack(list(features.values()))
        mu = all_X.mean(axis=0, keepdims=True)
        sigma = all_X.std(axis=0, keepdims=True) + 1e-12
        features = {k: (v - mu) / sigma for k, v in features.items()}
    # 3. MMD Matrix
    mmd_matrix, valid_subjects = compute_mmd_matrix(features)
    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/mmd_matrix.npy", mmd_matrix)
    print(f"Saved MMD matrix to {out_dir}/mmd_matrix.npy")
    with open(f"{out_dir}/mmd_subjects.json", "w") as f:
        json.dump(valid_subjects, f)
    print(f"Saved subject list to {out_dir}/mmd_subjects.json")

    # Note: annot=False to avoid heavy rendering for large N
    plot_heatmap(
        mmd_matrix, valid_subjects, valid_subjects,
        "MMD Distance Matrix", f"{out_dir}/mmd_matrix.png",
        annot=False, fmt=".2f"
    )
    print(f"Saved MMD heatmap to {out_dir}/mmd_matrix.png")

    n = len(valid_subjects)

    print("Computing Wasserstein and DTW distance matrices (symmetric)...")
    wass_matrix = np.zeros((n, n), dtype=np.float64)
    dtw_matrix  = np.zeros((n, n), dtype=np.float64)
    mean_series = [features[subj].mean(axis=1) for subj in valid_subjects]
    MAX_DIST = 1e6
    
    for i in tqdm(range(n), desc="Wass/DTW (symmetric)"):
        Xi = features[valid_subjects[i]]  # fetch once per i
        for j in range(i, n):  # upper triangle only
            if i == j:
                wass_val = 0.0
                dtw_val  = 0.0
            else:
                Xj = features[valid_subjects[j]]
    
                # --- Wasserstein: average over common feature columns ---
                common_cols = min(Xi.shape[1], Xj.shape[1])
                w_dists = []
                for k in range(common_cols):
                    d = wasserstein_distance(Xi[:, k], Xj[:, k])
                    if np.isfinite(d):
                        w_dists.append(min(d, MAX_DIST))
                # use NaN if nothing finite; later handle with nan_to_num
                wass_val = float(np.mean(w_dists)) if len(w_dists) > 0 else np.nan
    
                # --- DTW on 1D series (mean over features) ---
                try:
                    d = dtw(mean_series[i], mean_series[j])
                    dtw_val = float(d) if np.isfinite(d) and d < MAX_DIST else np.nan
                except Exception as e:
                    # Log the pair to help debugging
                    print(f"[DTW warn] {valid_subjects[i]} vs {valid_subjects[j]} -> {e}")
                    dtw_val = np.nan
    
            # mirror
            wass_matrix[i, j] = wass_val
            wass_matrix[j, i] = wass_val
            dtw_matrix[i, j]  = dtw_val
            dtw_matrix[j, i]  = dtw_val
    
    # keep diagonals strictly 0
    np.fill_diagonal(wass_matrix, 0.0)
    np.fill_diagonal(dtw_matrix, 0.0)

    os.makedirs("results/distances", exist_ok=True)
    np.save("results/distances/wasserstein_matrix.npy", wass_matrix)
    print("Saved Wasserstein distance matrix to results/distances/wasserstein_matrix.npy")
    np.save("results/distances/dtw_matrix.npy", dtw_matrix)
    print("Saved DTW distance matrix to results/distances/dtw_matrix.npy")
    with open("results/distances/subjects.json", "w") as f:
        json.dump(valid_subjects, f)
    print("Saved subjects list to results/distances/subjects.json")

    # --- NEW: save Wasserstein/DTW heatmap images ---
    plot_heatmap(
        wass_matrix, valid_subjects, valid_subjects,
        "Wasserstein Distance Matrix", "results/distances/wasserstein_matrix.png",
        annot=False, fmt=".2f"
    )
    print("Saved Wasserstein heatmap to results/distances/wasserstein_matrix.png")
    plot_heatmap(
        dtw_matrix, valid_subjects, valid_subjects,
        "DTW Distance Matrix", "results/distances/dtw_matrix.png",
        annot=False, fmt=".2f"
    )
    print("Saved DTW heatmap to results/distances/dtw_matrix.png")

    distance_types = {
        "mmd": ("results/mmd/mmd_matrix.npy", "results/mmd/mmd_subjects.json"),
        "wasserstein": ("results/distances/wasserstein_matrix.npy", "results/distances/subjects.json"),
        "dtw": ("results/distances/dtw_matrix.npy", "results/distances/subjects.json")
    }
    groups = load_groups("../misc/target_groups.txt")
    for dist_name, (matrix_path, subject_path) in distance_types.items():

        matrix = np.load(matrix_path)  # ← NaNのまま保持
        subjects = load_json(subject_path)
        
        # Exclude diagonal & NaN at the same time
        n = matrix.shape[0]
        mask = ~np.eye(n, dtype=bool)  # off-diagonal True
        masked = np.where(mask, matrix, np.nan)
        means = np.nanmean(masked, axis=1)
        stds  = np.nanstd(masked, axis=1)
        means_for_sort = np.nan_to_num(means, nan=-np.inf)  # NaNは最小扱い→降順で末尾へ
        sorted_idx = np.argsort(-means_for_sort)
        
        subj_sorted = [subjects[i] for i in sorted_idx]
        save_dir = f"results/{dist_name.lower()}"
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, f"{dist_name.lower()}_mean.npy"), means)
        print(f"Saved mean distances ({dist_name}) to {os.path.join(save_dir, f'{dist_name.lower()}_mean.npy')}")
        np.save(os.path.join(save_dir, f"{dist_name.lower()}_std.npy"), stds)
        print(f"Saved standard deviations ({dist_name}) to {os.path.join(save_dir, f'{dist_name.lower()}_std.npy')}")
        np.save(os.path.join(save_dir, f"{dist_name.lower()}_mean_sorted.npy"), means[sorted_idx])
        print(f"Saved sorted mean distances ({dist_name}) to {os.path.join(save_dir, f'{dist_name.lower()}_mean_sorted.npy')}")
        np.save(os.path.join(save_dir, f"{dist_name.lower()}_std_sorted.npy"), stds[sorted_idx])
        print(f"Saved sorted standard deviations ({dist_name}) to {os.path.join(save_dir, f'{dist_name.lower()}_std_sorted.npy')}")
        with open(os.path.join(save_dir, f"{dist_name.lower()}_subjects_sorted.json"), "w") as f:
            json.dump(subj_sorted, f)
        print(f"Saved sorted subject list ({dist_name}) to {os.path.join(save_dir, f'{dist_name.lower()}_subjects_sorted.json')}")
        plot_bar(subj_sorted, means[sorted_idx], stds[sorted_idx], f"Mean and StdDev of {dist_name} per Subject (Sorted)", dist_name, os.path.join(save_dir, f"{dist_name.lower()}_mean_std_sorted.png"))
        print(f"Saved sorted bar plot ({dist_name}) to {os.path.join(save_dir, f'{dist_name.lower()}_mean_std_sorted.png')}")

        group_matrix, group_names = compute_group_dist_matrix(matrix, subjects, groups)
        group_dir = f"results/group_distances/{dist_name.lower()}"
        os.makedirs(group_dir, exist_ok=True)
        np.save(os.path.join(group_dir, "group_matrix.npy"), group_matrix)
        print(f"Saved group distance matrix ({dist_name}) to {os.path.join(group_dir, 'group_matrix.npy')}")
        with open(os.path.join(group_dir, "group_names.json"), "w") as f:
            json.dump(group_names, f)
        print(f"Saved group names to {os.path.join(group_dir, 'group_names.json')}")
        plot_heatmap(group_matrix, group_names, group_names, 
                     f"{dist_name} Distance Between Groups", 
                     os.path.join(group_dir, "group_heatmap.png"))
        print(f"Saved group heatmap ({dist_name}) to {os.path.join(group_dir, 'group_heatmap.png')}")
        plot_group_projection(matrix, subjects, groups, dist_name, 
                              os.path.join(save_dir, "group_overlay_projection.png"))
        print(f"Saved group overlay projection ({dist_name}) to {os.path.join(save_dir, 'group_overlay_projection.png')}")

        centroids = compute_group_centroids_from_distance_matrix(matrix, subjects, groups)
        if centroids:
            centroid_matrix, centroid_names = compute_group_centroid_distances(centroids)
            centroid_heatmap_path = os.path.join(save_dir, "group_centroid_distance_heatmap.png")
            plot_heatmap(centroid_matrix, centroid_names, centroid_names, 
                         f"{dist_name}: Group Centroid Distance", 
                         centroid_heatmap_path)
            print(f"Saved centroid distance heatmap ({dist_name}) to {centroid_heatmap_path}")

        intra_results = compute_intra_group_variability(matrix, subjects, groups)
        intra_dir = f"results/group_distances/{dist_name.lower()}/intra"
        os.makedirs(intra_dir, exist_ok=True)
        intra_json_path = os.path.join(intra_dir, "intra_group_variability.json")
        with open(intra_json_path, "w") as f:
            json.dump(intra_results, f, indent=2)
        print(f"Saved intra-group variability JSON ({dist_name}) to {intra_json_path}")
        intra_csv_path = os.path.join(intra_dir, "intra_group_variability.csv")
        pd.DataFrame(intra_results).T.to_csv(intra_csv_path)
        print(f"Saved intra-group variability CSV ({dist_name}) to {intra_csv_path}")
        intra_plot_path = os.path.join(intra_dir, "intra_variability.png")
        plot_bar(list(intra_results.keys()), 
                 [intra_results[g]["mean"] for g in intra_results], 
                 [intra_results[g]["std"] for g in intra_results], 
                 f"Intra-group Variability - {dist_name}", "Mean Distance", 
                 intra_plot_path)
        print(f"Saved intra-group variability plot ({dist_name}) to {intra_plot_path}")

        stats = compute_intra_inter_stats(matrix, subjects, groups)
        intra_inter_dir = f"results/group_distances/{dist_name.lower()}/intra_inter"
        os.makedirs(intra_inter_dir, exist_ok=True)
        intra_inter_plot_path = os.path.join(intra_inter_dir, "intra_inter_comparison.png")
        plot_intra_inter(stats, dist_name, intra_inter_plot_path)
        print(f"Saved intra-inter comparison plot ({dist_name}) to {intra_inter_plot_path}")

if __name__ == "__main__":
    main()

